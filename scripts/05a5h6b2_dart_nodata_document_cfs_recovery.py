from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B2. Non-Corrected No-Data Document Parser + CFS Recovery QA
#
# 목적
# ------------------------------------------------------------
# H6B1에서 확보한 890개 same-receipt document ZIP을 대상으로
# 기존 corrected pipeline의
#
#   C1 candidate extractor
#   -> C4 current-period/document selector
#   -> C5 strict consolidated selector
#
# 를 재사용해 core-6 recovery coverage를 측정한다.
#
# IMPORTANT
# ------------------------------------------------------------
# - API 호출 없음
# - XBRL 014였던 receipt의 same-receipt document만 사용
# - C5 selection_status가
#     selected
#     selected_consensus
#   인 값만 인정
# - ambiguous/no_reliable 값은 자동 채택하지 않음
# - 이 단계는 "strict CFS recovery QA" 단계
# - OFS fallback은 다음 단계에서 별도 판단
#
# 실행
# ------------------------------------------------------------
# test 20:
#   python scripts\05a5h6b2_dart_nodata_document_cfs_recovery.py
#
# full 890:
#   python scripts\05a5h6b2_dart_nodata_document_cfs_recovery.py --full
#
# 처음부터 재파싱:
#   python scripts\05a5h6b2_dart_nodata_document_cfs_recovery.py --full --reset
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

SOURCE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.parquet"
)

TARGET_SCOPE = (
    INTERIM
    / "dart_noncorrected_full_api_no_data_scope.csv"
)

OUT_SELECTED_PARQUET = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_selected.parquet"
)

OUT_SELECTED_CSV = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_selected.csv"
)

OUT_RECEIPT_PARQUET = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_receipt_coverage.parquet"
)

OUT_RECEIPT_CSV = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_receipt_coverage.csv"
)

OUT_STATUS_CSV = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_selection_status.csv"
)

ACCEPTED_STATUSES = {
    "selected",
    "selected_consensus",
}

FAMILY_TO_METRIC = {
    "assets":
    "assets",

    "liabilities":
    "liabilities",

    "equity":
    "equity_total",

    "revenue":
    "revenue_cumulative",

    "operating_income":
    "operating_income_cumulative",

    "net_income":
    "net_income_total_cumulative",
}

CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]

ABS_BALANCE_TOL = 2_000_000
REL_BALANCE_TOL = 1e-6


def load_module(
    name: str,
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"module load 실패: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def receipt_string(
    series: pd.Series,
) -> pd.Series:

    return (
        series.astype(
            "string"
        )
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
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
        float(
            assets
        )
        - float(
            liabilities
        )
        - float(
            equity
        )
    )

    rel = (
        abs(
            gap
        )
        / abs(
            float(
                assets
            )
        )
        if float(
            assets
        ) != 0
        else np.nan
    )

    if gap == 0:
        qa = "exact_pass"

    elif (
        abs(
            gap
        )
        <= ABS_BALANCE_TOL
        or (
            pd.notna(
                rel
            )
            and rel
            <= REL_BALANCE_TOL
        )
    ):
        qa = (
            "rounding_pass"
        )

    else:
        qa = "fail"

    return (
        qa,
        gap,
        rel,
    )


def load_existing_receipts():

    if not OUT_RECEIPT_PARQUET.exists():
        return set()

    existing = pd.read_parquet(
        OUT_RECEIPT_PARQUET
    )

    if existing.empty:
        return set()

    existing[
        "rcept_no"
    ] = receipt_string(
        existing[
            "rcept_no"
        ]
    )

    return set(
        existing[
            "rcept_no"
        ].astype(str)
    )


def append_save(
    selected_records: list[
        dict[
            str,
            Any,
        ]
    ],
    receipt_records: list[
        dict[
            str,
            Any,
        ]
    ],
    status_records: list[
        dict[
            str,
            Any,
        ]
    ],
):

    # --------------------------------------------------------
    # Selected
    # --------------------------------------------------------

    if selected_records:

        new_selected = pd.DataFrame(
            selected_records
        )

        if OUT_SELECTED_PARQUET.exists():
            old_selected = (
                pd.read_parquet(
                    OUT_SELECTED_PARQUET
                )
            )

            selected = pd.concat(
                [
                    old_selected,
                    new_selected,
                ],
                ignore_index=True,
            )

        else:
            selected = (
                new_selected
            )

        selected[
            "rcept_no"
        ] = receipt_string(
            selected[
                "rcept_no"
            ]
        )

        selected = (
            selected.sort_values(
                [
                    "rcept_no",
                    "metric",
                ]
            )
            .drop_duplicates(
                subset=[
                    "rcept_no",
                    "metric",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        selected.to_parquet(
            OUT_SELECTED_PARQUET,
            index=False,
        )

        selected.to_csv(
            OUT_SELECTED_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Receipt coverage
    # --------------------------------------------------------

    if receipt_records:

        new_receipts = pd.DataFrame(
            receipt_records
        )

        if OUT_RECEIPT_PARQUET.exists():

            old_receipts = (
                pd.read_parquet(
                    OUT_RECEIPT_PARQUET
                )
            )

            receipts = pd.concat(
                [
                    old_receipts,
                    new_receipts,
                ],
                ignore_index=True,
            )

        else:
            receipts = (
                new_receipts
            )

        receipts[
            "rcept_no"
        ] = receipt_string(
            receipts[
                "rcept_no"
            ]
        )

        receipts = (
            receipts.sort_values(
                "rcept_no"
            )
            .drop_duplicates(
                subset=[
                    "rcept_no",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        receipts.to_parquet(
            OUT_RECEIPT_PARQUET,
            index=False,
        )

        receipts.to_csv(
            OUT_RECEIPT_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Raw C5 status
    # --------------------------------------------------------

    if status_records:

        new_status = pd.DataFrame(
            status_records
        )

        if OUT_STATUS_CSV.exists():

            old_status = pd.read_csv(
                OUT_STATUS_CSV,
                dtype={
                    "rcept_no":
                    str,
                },
                low_memory=False,
            )

            status = pd.concat(
                [
                    old_status,
                    new_status,
                ],
                ignore_index=True,
            )

        else:
            status = (
                new_status
            )

        status[
            "rcept_no"
        ] = receipt_string(
            status[
                "rcept_no"
            ]
        )

        status = (
            status.sort_values(
                [
                    "rcept_no",
                    "account_family",
                ]
            )
            .drop_duplicates(
                subset=[
                    "rcept_no",
                    "account_family",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        status.to_csv(
            OUT_STATUS_CSV,
            index=False,
            encoding="utf-8-sig",
        )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
    )

    args = parser.parse_args()

    for path in [
        SOURCE_MANIFEST,
        TARGET_SCOPE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    if args.reset:

        for path in [
            OUT_SELECTED_PARQUET,
            OUT_SELECTED_CSV,
            OUT_RECEIPT_PARQUET,
            OUT_RECEIPT_CSV,
            OUT_STATUS_CSV,
        ]:
            if path.exists():
                path.unlink()

    c1 = load_module(
        "dart_h6b2_c1",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    c4 = load_module(
        "dart_h6b2_c4",
        SCRIPTS_DIR
        / "05a5c4_dart_conservative_document_selector.py",
    )

    c5 = load_module(
        "dart_h6b2_c5",
        SCRIPTS_DIR
        / "05a5c5_dart_strict_consolidated_document_selector.py",
    )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    scope = pd.read_csv(
        TARGET_SCOPE,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
            "corp_code":
            str,
            "reprt_code":
            str,
        },
        low_memory=False,
    )

    manifest[
        "rcept_no"
    ] = receipt_string(
        manifest[
            "rcept_no"
        ]
    )

    scope[
        "rcept_no"
    ] = receipt_string(
        scope[
            "rcept_no"
        ]
    )

    # TARGET_SCOPE may already contain collection/provenance columns
    # inherited from H6A. Merge H6B1 manifest with temporary names so
    # pandas does not silently create *_x / *_y columns.
    manifest_small = (
        manifest[
            [
                "rcept_no",
                "collection_status",
                "selected_source",
                "selected_path",
            ]
        ]
        .rename(
            columns={
                "collection_status":
                "h6b1_collection_status",

                "selected_source":
                "h6b1_selected_source",

                "selected_path":
                "h6b1_selected_path",
            }
        )
    )

    source = (
        scope.merge(
            manifest_small,
            on="rcept_no",
            how="left",
            validate="one_to_one",
        )
    )

    # H6B1 manifest is the source of truth for the actual raw source
    # collected in the previous step. Always use it for H6B2.
    source[
        "collection_status"
    ] = source[
        "h6b1_collection_status"
    ]

    source[
        "selected_source"
    ] = source[
        "h6b1_selected_source"
    ]

    source[
        "selected_path"
    ] = source[
        "h6b1_selected_path"
    ]

    source = source.drop(
        columns=[
            "h6b1_collection_status",
            "h6b1_selected_source",
            "h6b1_selected_path",
        ]
    )

    # Sanity check: H6B1 summary showed all 890 as
    # available_document/document.
    available_document_count = int(
        source[
            "collection_status"
        ].eq(
            "available_document"
        )
        .sum()
    )

    selected_document_count = int(
        source[
            "selected_source"
        ].eq(
            "document"
        )
        .sum()
    )

    if (
        available_document_count
        != len(
            source
        )
        or selected_document_count
        != len(
            source
        )
    ):
        print(
            "WARNING: H6B1 source manifest is not uniformly "
            "available_document/document. "
            f"rows={len(source):,}, "
            f"available_document={available_document_count:,}, "
            f"document={selected_document_count:,}"
        )

    targets = source.loc[
        source[
            "collection_status"
        ].eq(
            "available_document"
        )
        & source[
            "selected_source"
        ].eq(
            "document"
        )
    ].copy()

    if len(
        targets
    ) != 890:

        print(
            "WARNING: expected 890 available_document targets, "
            f"current={len(targets):,}"
        )

    done = (
        set()
        if args.reset
        else load_existing_receipts()
    )

    todo = targets.loc[
        ~targets[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            done
        )
    ].copy()

    if args.full:
        limit = None

    elif (
        args.limit
        is not None
    ):
        limit = max(
            int(
                args.limit
            ),
            0,
        )

    else:
        limit = 20

    if (
        limit is not None
        and limit > 0
    ):
        todo = todo.head(
            limit
        ).copy()

    elif limit == 0:
        todo = todo.head(
            0
        ).copy()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H6B2 NON-CORRECTED NO-DATA DOCUMENT CFS RECOVERY QA"
    )

    print(
        "=" * 100
    )

    print(
        f"\nAll document targets: "
        f"{len(targets):,}"
    )

    print(
        f"Already parsed      : "
        f"{len(done):,}"
    )

    print(
        f"This run            : "
        f"{len(todo):,}"
    )

    selected_buffer = []
    receipt_buffer = []
    status_buffer = []

    for idx, (_, target) in enumerate(
        todo.iterrows(),
        start=1,
    ):

        receipt = str(
            target[
                "rcept_no"
            ]
        )

        stock = str(
            target.get(
                "stock_code",
                "",
            )
        ).zfill(
            6
        )

        period = str(
            target.get(
                "canonical_period_key",
                "",
            )
        )

        selected_path = Path(
            str(
                target[
                    "selected_path"
                ]
            )
        )

        print(
            f"\n[{idx}/{len(todo)}] "
            f"{stock} | "
            f"{period} | "
            f"{receipt}"
        )

        record_base = {
            "stock_code":
            stock,

            "corp_code":
            target.get(
                "corp_code"
            ),

            "corp_name":
            target.get(
                "corp_name"
            ),

            "canonical_period_key":
            period,

            "rcept_no":
            receipt,

            "reprt_code":
            target.get(
                "reprt_code"
            ),

            "rcept_dt":
            target.get(
                "rcept_dt"
            ),
        }

        try:

            if not selected_path.exists():
                raise FileNotFoundError(
                    selected_path
                )

            # C1 derives metadata from path. Point RAW_ROOT to
            # the actual H6B1 raw root parent layout.
            c1.RAW_ROOT = (
                selected_path.parents[
                    2
                ]
            )

            raw = pd.DataFrame(
                c1.extract_document_candidates(
                    selected_path
                )
            )

            if raw.empty:

                receipt_buffer.append(
                    {
                        **record_base,

                        "parse_status":
                        "no_candidates",

                        "strict_cfs_core_count":
                        0,

                        "missing_core_metrics":
                        " | ".join(
                            CORE_METRICS
                        ),

                        "balance_qa":
                        "missing",

                        "balance_gap":
                        np.nan,

                        "balance_relative_gap":
                        np.nan,
                    }
                )

                print(
                    "  no candidates"
                )

                continue

            # Normalize metadata to the H6 target source of truth.
            raw[
                "stock_code"
            ] = stock

            raw[
                "period_key"
            ] = period

            raw[
                "rcept_no"
            ] = receipt

            raw[
                "rcept_dt"
            ] = pd.to_datetime(
                target.get(
                    "rcept_dt"
                ),
                errors="coerce",
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

            if c5df.empty:

                receipt_buffer.append(
                    {
                        **record_base,

                        "parse_status":
                        "c5_empty",

                        "strict_cfs_core_count":
                        0,

                        "missing_core_metrics":
                        " | ".join(
                            CORE_METRICS
                        ),

                        "balance_qa":
                        "missing",

                        "balance_gap":
                        np.nan,

                        "balance_relative_gap":
                        np.nan,
                    }
                )

                print(
                    "  C5 empty"
                )

                continue

            c5df[
                "rcept_no"
            ] = receipt

            c5df[
                "stock_code"
            ] = stock

            c5df[
                "period_key"
            ] = period

            values = {}

            for _, sel in (
                c5df.iterrows()
            ):

                family = str(
                    sel.get(
                        "account_family",
                        "",
                    )
                )

                status = str(
                    sel.get(
                        "selection_status",
                        "",
                    )
                )

                metric = (
                    FAMILY_TO_METRIC.get(
                        family
                    )
                )

                status_buffer.append(
                    {
                        **record_base,

                        "account_family":
                        family,

                        "metric":
                        metric,

                        "selection_status":
                        status,

                        "selected_value_krw":
                        sel.get(
                            "selected_value_krw"
                        ),

                        "table_index":
                        sel.get(
                            "table_index"
                        ),

                        "row_index":
                        sel.get(
                            "row_index"
                        ),

                        "primary_row_label":
                        sel.get(
                            "primary_row_label"
                        ),

                        "selected_column":
                        sel.get(
                            "selected_column"
                        ),

                        "strict_score":
                        sel.get(
                            "strict_score"
                        ),

                        "consolidated_evidence_score":
                        sel.get(
                            "consolidated_evidence_score"
                        ),

                        "consolidated_evidence_reason":
                        sel.get(
                            "consolidated_evidence_reason"
                        ),

                        "heading_context":
                        sel.get(
                            "heading_context"
                        ),
                    }
                )

                if (
                    metric is None
                    or status
                    not in ACCEPTED_STATUSES
                ):
                    continue

                value = pd.to_numeric(
                    sel.get(
                        "selected_value_krw"
                    ),
                    errors="coerce",
                )

                if pd.isna(
                    value
                ):
                    continue

                values[
                    metric
                ] = float(
                    value
                )

                selected_buffer.append(
                    {
                        **record_base,

                        "metric":
                        metric,

                        "metric_value":
                        float(
                            value
                        ),

                        "account_family":
                        family,

                        "selection_status":
                        status,

                        "source":
                        "same_receipt_document_strict_cfs",

                        "table_index":
                        sel.get(
                            "table_index"
                        ),

                        "row_index":
                        sel.get(
                            "row_index"
                        ),

                        "primary_row_label":
                        sel.get(
                            "primary_row_label"
                        ),

                        "selected_column":
                        sel.get(
                            "selected_column"
                        ),

                        "unit_hint":
                        sel.get(
                            "unit_hint"
                        ),

                        "strict_score":
                        sel.get(
                            "strict_score"
                        ),

                        "consolidated_evidence_score":
                        sel.get(
                            "consolidated_evidence_score"
                        ),

                        "consolidated_evidence_reason":
                        sel.get(
                            "consolidated_evidence_reason"
                        ),

                        "heading_context":
                        sel.get(
                            "heading_context"
                        ),

                        "row_text":
                        sel.get(
                            "row_text"
                        ),
                    }
                )

            core_count = sum(
                metric in values
                for metric in CORE_METRICS
            )

            missing = [
                metric
                for metric
                in CORE_METRICS
                if metric
                not in values
            ]

            qa, gap, rel = (
                balance_qa(
                    values.get(
                        "assets"
                    ),
                    values.get(
                        "liabilities"
                    ),
                    values.get(
                        "equity_total"
                    ),
                )
            )

            receipt_buffer.append(
                {
                    **record_base,

                    "parse_status":
                    "parsed",

                    **{
                        metric:
                        values.get(
                            metric,
                            np.nan,
                        )
                        for metric
                        in CORE_METRICS
                    },

                    "strict_cfs_core_count":
                    core_count,

                    "missing_core_metrics":
                    " | ".join(
                        missing
                    ),

                    "balance_qa":
                    qa,

                    "balance_gap":
                    gap,

                    "balance_relative_gap":
                    rel,
                }
            )

            print(
                f"  strict CFS core="
                f"{core_count}/6 | "
                f"balance={qa}"
            )

        except Exception as exc:

            receipt_buffer.append(
                {
                    **record_base,

                    "parse_status":
                    "parse_error",

                    "strict_cfs_core_count":
                    0,

                    "missing_core_metrics":
                    " | ".join(
                        CORE_METRICS
                    ),

                    "balance_qa":
                    "missing",

                    "balance_gap":
                    np.nan,

                    "balance_relative_gap":
                    np.nan,

                    "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            print(
                "  ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        if len(
            receipt_buffer
        ) >= 25:

            append_save(
                selected_buffer,
                receipt_buffer,
                status_buffer,
            )

            selected_buffer = []
            receipt_buffer = []
            status_buffer = []

    if (
        selected_buffer
        or receipt_buffer
        or status_buffer
    ):

        append_save(
            selected_buffer,
            receipt_buffer,
            status_buffer,
        )

    # --------------------------------------------------------
    # Summary over all parsed receipts in target scope
    # --------------------------------------------------------

    if not OUT_RECEIPT_PARQUET.exists():

        print(
            "\nNo receipt coverage output."
        )

        return

    coverage = pd.read_parquet(
        OUT_RECEIPT_PARQUET
    )

    coverage[
        "rcept_no"
    ] = receipt_string(
        coverage[
            "rcept_no"
        ]
    )

    target_receipts = set(
        targets[
            "rcept_no"
        ].astype(str)
    )

    coverage = coverage.loc[
        coverage[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            target_receipts
        )
    ].copy()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "H6B2 STRICT CFS RECOVERY SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nParsed receipts: "
        f"{len(coverage):,} / "
        f"{len(targets):,}"
    )

    print(
        "\n[Parse status]"
    )

    print(
        coverage[
            "parse_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Strict CFS core metric count]"
    )

    print(
        coverage[
            "strict_cfs_core_count"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    print(
        "\n[Metric coverage]"
    )

    for metric in CORE_METRICS:

        count = int(
            coverage[
                metric
            ]
            .notna()
            .sum()
        ) if metric in coverage.columns else 0

        print(
            f"{metric:<38} "
            f"{count:>4} / "
            f"{len(coverage):>4}"
        )

    print(
        "\n[Balance QA]"
    )

    print(
        coverage[
            "balance_qa"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if "missing_core_metrics" in coverage.columns:

        print(
            "\n[Top missing combinations]"
        )

        missing = (
            coverage.loc[
                coverage[
                    "strict_cfs_core_count"
                ].lt(
                    6
                ),
                "missing_core_metrics",
            ]
            .value_counts()
            .head(
                20
            )
        )

        if missing.empty:
            print(
                "None"
            )

        else:
            print(
                missing.to_string()
            )

    hard_fail = coverage.loc[
        coverage[
            "balance_qa"
        ].eq(
            "fail"
        )
    ].copy()

    parse_errors = coverage.loc[
        coverage[
            "parse_status"
        ].eq(
            "parse_error"
        )
    ].copy()

    print(
        f"\nHard balance failures: "
        f"{len(hard_fail):,}"
    )

    print(
        f"Parse errors         : "
        f"{len(parse_errors):,}"
    )

    print(
        f"\nSelected : "
        f"{OUT_SELECTED_PARQUET}"
    )

    print(
        f"Coverage : "
        f"{OUT_RECEIPT_PARQUET}"
    )

    print(
        f"Statuses : "
        f"{OUT_STATUS_CSV}"
    )

    print(
        "\n다음 단계:"
        "\n- test 20에서 parse_error=0인지 확인"
        "\n- 정상이면 --full로 890건 전체 파싱"
        "\n- full 결과에서 strict CFS 6/6 복구분은 QA 후 채택 후보"
        "\n- strict CFS 불완전분은 CFS 부재 vs ambiguous를 나눠 OFS fallback/추가 audit"
    )


if __name__ == "__main__":
    main()
