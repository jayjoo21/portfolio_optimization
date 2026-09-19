from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H5B. Non-Corrected Document Selector Schema Diagnostic
#            + Robust Balance Extraction
#
# 목적
# ------------------------------------------------------------
# H5A에서 document.zip parsing은 성공했지만
# document_assets/liabilities/equity가 전부 None이었던 원인을 확인한다.
#
# 핵심:
# - API 재호출 없음
# - 이미 다운로드된 same-receipt document.zip만 사용
# - C1 -> C4 -> C5 결과의 실제 schema/value를 출력
# - "selected"로 명시된 row만 보수적으로 balance metric으로 인정
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_balance_document_c4_candidates_debug.csv
# dart_noncorrected_balance_document_c5_selected_debug.csv
# dart_noncorrected_balance_document_robust_crosscheck.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h5b_dart_document_selector_schema_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "noncorrected_balance_qa"
)

RECONCILED = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
)

ORIGINAL = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide.parquet"
)

OUT_C4 = (
    INTERIM
    / "dart_noncorrected_balance_document_c4_candidates_debug.csv"
)

OUT_C5 = (
    INTERIM
    / "dart_noncorrected_balance_document_c5_selected_debug.csv"
)

OUT_SUMMARY = (
    INTERIM
    / "dart_noncorrected_balance_document_robust_crosscheck.csv"
)

ABS_TOL = 2_000_000
REL_TOL = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"module load 실패: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("\u3000", "")
        .lower()
    )


def parse_num(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {
        "",
        "-",
        "nan",
        "None",
        "<NA>",
    }:
        return np.nan

    text = text.replace(
        ",",
        "",
    )

    if (
        text.startswith("(")
        and text.endswith(")")
    ):
        text = (
            "-"
            + text[1:-1]
        )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def balance_qa(
    assets,
    liabilities,
    equity,
):
    if any(
        pd.isna(x)
        for x in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return (
            "missing",
            np.nan,
            np.nan,
        )

    gap = (
        float(assets)
        - float(liabilities)
        - float(equity)
    )

    rel = (
        abs(gap)
        / abs(float(assets))
        if float(assets) != 0
        else np.nan
    )

    if gap == 0:
        qa = "exact_pass"

    elif (
        abs(gap)
        <= ABS_TOL
        or (
            pd.notna(rel)
            and rel <= REL_TOL
        )
    ):
        qa = "rounding_pass"

    else:
        qa = "fail"

    return (
        qa,
        gap,
        rel,
    )


def find_zip_for_receipt(
    receipt: str,
) -> Path | None:

    matches = list(
        RAW_ROOT.glob(
            f"**/*{receipt}*_document.zip"
        )
    )

    if not matches:
        return None

    return sorted(
        matches
    )[0]


def candidate_column(
    columns,
    candidates,
):
    normalized = {
        normalize_text(c):
        c
        for c in columns
    }

    for name in candidates:
        n = normalize_text(
            name
        )

        if n in normalized:
            return normalized[
                n
            ]

    return None


def infer_metric_from_row(
    row: pd.Series,
) -> str | None:

    text_parts = []

    for col in row.index:
        col_norm = normalize_text(
            col
        )

        if any(
            token in col_norm
            for token in [
                "family",
                "metric",
                "account",
                "name",
                "label",
            ]
        ):
            text_parts.append(
                normalize_text(
                    row.get(
                        col
                    )
                )
            )

    text = " | ".join(
        text_parts
    )

    # Order matters: equity total before generic equity.
    if (
        "assets" in text
        or "자산총계" in text
    ):
        if (
            "currentassets" not in text
            and "noncurrentassets" not in text
            and "유동자산" not in text
            and "비유동자산" not in text
        ):
            return "assets"

    if (
        "liabilities" in text
        or "부채총계" in text
    ):
        if (
            "currentliabilities" not in text
            and "noncurrentliabilities" not in text
            and "유동부채" not in text
            and "비유동부채" not in text
        ):
            return "liabilities"

    if any(
        token in text
        for token in [
            "equity_total",
            "totalequity",
            "equitytotal",
            "자본총계",
        ]
    ):
        return "equity_total"

    # Exact family label may simply be "equity".
    family_values = [
        normalize_text(
            row.get(col)
        )
        for col in row.index
        if "family" in normalize_text(
            col
        )
        or "metric" in normalize_text(
            col
        )
    ]

    if any(
        v == "equity"
        for v in family_values
    ):
        return "equity_total"

    return None


def row_is_selected(
    row: pd.Series,
) -> bool:

    status_cols = [
        c
        for c in row.index
        if any(
            token in normalize_text(
                c
            )
            for token in [
                "status",
                "selected",
                "choice",
            ]
        )
    ]

    # C5 output itself may contain only selected rows and have no status.
    if not status_cols:
        return True

    values = [
        normalize_text(
            row.get(
                c
            )
        )
        for c in status_cols
    ]

    positive = {
        "selected",
        "selected_consensus",
        "selectedconsensus",
        "true",
        "1",
        "pass",
    }

    if any(
        v in positive
        for v in values
    ):
        return True

    # If there is a selection_status column and it is not selected,
    # reject conservatively.
    explicit_status_cols = [
        c
        for c in status_cols
        if "status" in normalize_text(
            c
        )
    ]

    if explicit_status_cols:
        return False

    return True


def get_selected_value(
    row: pd.Series,
):

    value_priority = [
        "selected_value_krw",
        "selected_value",
        "value_krw",
        "candidate_value_krw",
        "amount_krw",
        "value",
        "amount",
    ]

    col = candidate_column(
        row.index,
        value_priority,
    )

    if col is not None:
        value = parse_num(
            row.get(
                col
            )
        )

        if pd.notna(
            value
        ):
            return (
                value,
                col,
            )

    # Last-resort scan of numeric-looking value columns,
    # but never use rank/score/count/ord columns.
    for col in row.index:
        c = normalize_text(
            col
        )

        if not any(
            token in c
            for token in [
                "value",
                "amount",
                "금액",
            ]
        ):
            continue

        if any(
            token in c
            for token in [
                "score",
                "rank",
                "count",
                "ord",
                "gap",
            ]
        ):
            continue

        value = parse_num(
            row.get(
                col
            )
        )

        if pd.notna(
            value
        ):
            return (
                value,
                col,
            )

    return (
        np.nan,
        None,
    )


def main():

    source_path = (
        RECONCILED
        if RECONCILED.exists()
        else ORIGINAL
    )

    if not source_path.exists():
        raise FileNotFoundError(
            source_path
        )

    wide = pd.read_parquet(
        source_path
    )

    wide[
        "rcept_no"
    ] = (
        wide[
            "rcept_no"
        ]
        .astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    targets = wide.loc[
        wide[
            "balance_qa_class"
        ].eq(
            "fail"
        )
    ].copy()

    c1 = load_module(
        "dart_h5b_c1",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    c4 = load_module(
        "dart_h5b_c4",
        SCRIPTS_DIR
        / "05a5c4_dart_conservative_document_selector.py",
    )

    c5 = load_module(
        "dart_h5b_c5",
        SCRIPTS_DIR
        / "05a5c5_dart_strict_consolidated_document_selector.py",
    )

    c1.RAW_ROOT = RAW_ROOT

    print(
        "\n"
        + "=" * 100
    )
    print(
        "05A5-H5B DOCUMENT SELECTOR SCHEMA DIAGNOSTIC"
    )
    print(
        "=" * 100
    )

    print(
        f"\nTargets: {len(targets):,}"
    )

    all_c4 = []
    all_c5 = []
    summary_rows = []

    printed_schema = False

    for idx, (_, target) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        receipt = str(
            target[
                "rcept_no"
            ]
        )

        stock = str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        )

        zip_path = find_zip_for_receipt(
            receipt
        )

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{stock} | "
            f"{target['canonical_period_key']} | "
            f"{receipt}"
        )

        if zip_path is None:
            print(
                "  ZIP not found"
            )

            summary_rows.append(
                {
                    "stock_code":
                    stock,
                    "corp_name":
                    target.get(
                        "corp_name"
                    ),
                    "canonical_period_key":
                    target[
                        "canonical_period_key"
                    ],
                    "rcept_no":
                    receipt,
                    "status":
                    "zip_not_found",
                }
            )

            continue

        raw = pd.DataFrame(
            c1.extract_document_candidates(
                zip_path
            )
        )

        c4df = pd.DataFrame(
            c4.build_account_candidates(
                raw
            )
        )

        c5df = pd.DataFrame(
            c5.select_values(
                c4df
            )
        )

        for df, label in [
            (
                c4df,
                "c4",
            ),
            (
                c5df,
                "c5",
            ),
        ]:
            if not df.empty:
                df = df.copy()
                df[
                    "_target_stock_code"
                ] = stock
                df[
                    "_target_rcept_no"
                ] = receipt
                df[
                    "_target_period_key"
                ] = target[
                    "canonical_period_key"
                ]

                if label == "c4":
                    all_c4.append(
                        df
                    )
                else:
                    all_c5.append(
                        df
                    )

        if not printed_schema:
            print(
                "\n[C4 columns]"
            )
            print(
                list(
                    c4df.columns
                )
            )

            print(
                "\n[C5 columns]"
            )
            print(
                list(
                    c5df.columns
                )
            )

            printed_schema = True

        # Print useful unique values from likely schema columns.
        print(
            f"  raw={len(raw):,} "
            f"| c4={len(c4df):,} "
            f"| c5={len(c5df):,}"
        )

        for name, df in [
            (
                "C4",
                c4df,
            ),
            (
                "C5",
                c5df,
            ),
        ]:
            interesting_cols = [
                c
                for c in df.columns
                if any(
                    token
                    in normalize_text(
                        c
                    )
                    for token in [
                        "family",
                        "metric",
                        "status",
                        "account",
                        "selection",
                    ]
                )
            ]

            if interesting_cols:
                print(
                    f"\n  [{name} key columns]"
                )

                preview_cols = (
                    interesting_cols[
                        :12
                    ]
                )

                print(
                    df[
                        preview_cols
                    ]
                    .head(
                        20
                    )
                    .to_string(
                        index=False
                    )
                )

        extracted = {}

        if not c5df.empty:

            for row_idx, row in (
                c5df.iterrows()
            ):

                if not row_is_selected(
                    row
                ):
                    continue

                metric = infer_metric_from_row(
                    row
                )

                if metric is None:
                    continue

                value, value_col = (
                    get_selected_value(
                        row
                    )
                )

                if pd.isna(
                    value
                ):
                    continue

                if metric not in extracted:
                    extracted[
                        metric
                    ] = {
                        "value":
                        float(
                            value
                        ),
                        "value_column":
                        value_col,
                        "row_index":
                        row_idx,
                    }

        assets = (
            extracted.get(
                "assets",
                {},
            ).get(
                "value",
                np.nan,
            )
        )

        liabilities = (
            extracted.get(
                "liabilities",
                {},
            ).get(
                "value",
                np.nan,
            )
        )

        equity = (
            extracted.get(
                "equity_total",
                {},
            ).get(
                "value",
                np.nan,
            )
        )

        qa, gap, rel = balance_qa(
            assets,
            liabilities,
            equity,
        )

        print(
            "\n  [Robust extracted]"
        )
        print(
            f"  assets      = {assets}"
        )
        print(
            f"  liabilities = {liabilities}"
        )
        print(
            f"  equity      = {equity}"
        )
        print(
            f"  QA          = {qa}"
        )

        summary_rows.append(
            {
                "stock_code":
                stock,
                "corp_name":
                target.get(
                    "corp_name"
                ),
                "canonical_period_key":
                target[
                    "canonical_period_key"
                ],
                "rcept_no":
                receipt,

                "api_assets":
                target.get(
                    "assets"
                ),
                "api_liabilities":
                target.get(
                    "liabilities"
                ),
                "api_equity":
                target.get(
                    "equity_total"
                ),
                "api_gap":
                target.get(
                    "balance_gap"
                ),
                "api_rel_gap":
                target.get(
                    "balance_relative_gap"
                ),

                "document_assets":
                assets,
                "document_liabilities":
                liabilities,
                "document_equity":
                equity,
                "document_gap":
                gap,
                "document_rel_gap":
                rel,
                "document_qa":
                qa,

                "assets_value_column":
                extracted.get(
                    "assets",
                    {},
                ).get(
                    "value_column"
                ),
                "liabilities_value_column":
                extracted.get(
                    "liabilities",
                    {},
                ).get(
                    "value_column"
                ),
                "equity_value_column":
                extracted.get(
                    "equity_total",
                    {},
                ).get(
                    "value_column"
                ),
            }
        )

    c4_all = (
        pd.concat(
            all_c4,
            ignore_index=True,
        )
        if all_c4
        else pd.DataFrame()
    )

    c5_all = (
        pd.concat(
            all_c5,
            ignore_index=True,
        )
        if all_c5
        else pd.DataFrame()
    )

    summary = pd.DataFrame(
        summary_rows
    )

    c4_all.to_csv(
        OUT_C4,
        index=False,
        encoding="utf-8-sig",
    )

    c5_all.to_csv(
        OUT_C5,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "H5B SUMMARY"
    )
    print(
        "=" * 100
    )

    if (
        "document_qa"
        in summary.columns
    ):
        print(
            "\n[Document QA]"
        )
        print(
            summary[
                "document_qa"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Summary]"
    )

    show = [
        c
        for c in [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "api_gap",
            "api_rel_gap",
            "document_assets",
            "document_liabilities",
            "document_equity",
            "document_gap",
            "document_rel_gap",
            "document_qa",
        ]
        if c in summary.columns
    ]

    print(
        summary[
            show
        ].to_string(
            index=False
        )
    )

    print(
        f"\nC4 debug : "
        f"{OUT_C4}"
    )

    print(
        f"C5 debug : "
        f"{OUT_C5}"
    )

    print(
        f"Summary  : "
        f"{OUT_SUMMARY}"
    )

    print(
        "\n다음 판단:"
        "\n- document_qa가 PASS면 H5A의 schema bridge 문제였음"
        "\n- 여전히 missing이면 C5 key columns/debug CSV에서 실제 family/status/value schema 확인"
        "\n- document_qa가 FAIL이면 same-receipt document도 balance 불일치"
    )


if __name__ == "__main__":
    main()
