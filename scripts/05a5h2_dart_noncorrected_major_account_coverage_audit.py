from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H2. Non-Corrected Major-Account Coverage Audit
#
# 목적
# ------------------------------------------------------------
# H1 fnlttMultiAcnt 결과에서 PIT verified receipt만 사용해
# 핵심 재무계정 coverage를 점검하고,
# fnlttSinglAcntAll selective fallback 대상 receipt를 만든다.
#
# IMPORTANT
# ------------------------------------------------------------
# - api_error가 남아 있으면 중단한다.
# - receipt_mismatch는 사용하지 않는다.
# - CFS(연결) 우선, 없으면 OFS(별도) fallback.
# - parent-attributable metrics는 여기서 필수 fallback 조건으로 삼지 않는다.
#   (major-account API에 항상 존재하지 않을 수 있으므로)
# - selective full-API fallback 조건:
#       1) H1 no_data
#       2) available이지만 핵심 6개 중 하나 이상 missing
#
# 핵심 6개
# ------------------------------------------------------------
# assets
# liabilities
# equity_total
# revenue_cumulative
# operating_income_cumulative
# net_income_total_cumulative
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_major_account_coverage.parquet/csv
#   dart_noncorrected_selective_fallback_targets.parquet/csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h2_dart_noncorrected_major_account_coverage_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TARGETS_PATH = INTERIM / "dart_noncorrected_target_periods.parquet"
MANIFEST_PATH = INTERIM / "dart_noncorrected_multi_account_manifest.parquet"
ROWS_PATH = INTERIM / "dart_noncorrected_multi_account_rows.parquet"

COVERAGE_PARQUET = INTERIM / "dart_noncorrected_major_account_coverage.parquet"
COVERAGE_CSV = INTERIM / "dart_noncorrected_major_account_coverage.csv"

FALLBACK_PARQUET = INTERIM / "dart_noncorrected_selective_fallback_targets.parquet"
FALLBACK_CSV = INTERIM / "dart_noncorrected_selective_fallback_targets.csv"


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = (
        text.replace("－", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    return text


def parse_number(value):
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

    # accounting negative notation
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


def classify_metric(
    account_nm: str,
) -> str | None:

    label = normalize_label(
        account_nm
    )

    # Balance sheet: exact labels only.
    if label == "자산총계":
        return "assets"

    if label == "부채총계":
        return "liabilities"

    if label == "자본총계":
        return "equity_total"

    # Revenue:
    revenue_labels = {
        "매출액",
        "수익(매출액)",
        "영업수익",
        "영업수익(매출액)",
    }

    if label in revenue_labels:
        return "revenue_cumulative"

    # Operating income:
    operating_income_labels = {
        "영업이익",
        "영업이익(손실)",
        "영업손익",
    }

    if label in operating_income_labels:
        return "operating_income_cumulative"

    # Net income:
    net_income_labels = {
        "당기순이익",
        "당기순이익(손실)",
        "분기순이익",
        "분기순이익(손실)",
        "반기순이익",
        "반기순이익(손실)",
    }

    if label in net_income_labels:
        return "net_income_total_cumulative"

    return None


def selected_amount(
    row: pd.Series,
    metric: str,
):
    """
    BS는 thstrm_amount.
    IS/CIS flow metrics는 cumulative를 우선:
      thstrm_add_amount -> thstrm_amount fallback.

    OpenDART full-statement docs also define thstrm_add_amount as
    accumulated current-period amount for quarterly/semiannual IS/CIS.
    """

    if metric in {
        "assets",
        "liabilities",
        "equity_total",
    }:
        return parse_number(
            row.get(
                "thstrm_amount"
            )
        )

    add_value = parse_number(
        row.get(
            "thstrm_add_amount"
        )
    )

    if pd.notna(
        add_value
    ):
        return add_value

    return parse_number(
        row.get(
            "thstrm_amount"
        )
    )


def main():

    for path in [
        TARGETS_PATH,
        MANIFEST_PATH,
        ROWS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    targets = pd.read_parquet(
        TARGETS_PATH
    )

    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    rows = pd.read_parquet(
        ROWS_PATH
    )

    for df in [
        targets,
        manifest,
        rows,
    ]:
        if "stock_code" in df.columns:
            df[
                "stock_code"
            ] = (
                df[
                    "stock_code"
                ]
                .astype(str)
                .str.zfill(6)
            )

        if "rcept_no" in df.columns:
            df[
                "rcept_no"
            ] = (
                df[
                    "rcept_no"
                ]
                .astype(str)
            )

        if "expected_rcept_no" in df.columns:
            df[
                "expected_rcept_no"
            ] = (
                df[
                    "expected_rcept_no"
                ]
                .astype(str)
            )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-H2 NON-CORRECTED MAJOR-ACCOUNT COVERAGE AUDIT"
    )
    print(
        "=" * 80
    )

    status_counts = (
        manifest[
            "collection_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n[H1 collection status]"
    )

    print(
        status_counts.to_string()
    )

    api_error_count = int(
        status_counts.get(
            "api_error",
            0,
        )
    )

    if api_error_count > 0:
        raise RuntimeError(
            "\nH1에 api_error가 아직 남아 있습니다.\n"
            f"api_error={api_error_count:,}\n\n"
            "먼저 아래를 실행하세요:\n"
            "python scripts\\05a5h1_dart_noncorrected_batch_collector.py --retry-errors --full\n"
            "\n그 후 H2를 다시 실행하세요."
        )

    available_manifest = (
        manifest.loc[
            manifest[
                "collection_status"
            ].eq(
                "available"
            )
            & manifest[
                "pit_verified"
            ].eq(
                True
            )
        ]
        .copy()
    )

    verified_receipts = set(
        available_manifest[
            "expected_rcept_no"
        ]
        .astype(str)
    )

    verified_rows = rows.loc[
        rows[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            verified_receipts
        )
    ].copy()

    if verified_rows.empty:
        raise RuntimeError(
            "PIT verified API rows가 없습니다."
        )

    verified_rows[
        "metric"
    ] = verified_rows[
        "account_nm"
    ].apply(
        classify_metric
    )

    metric_rows = verified_rows.loc[
        verified_rows[
            "metric"
        ].notna()
    ].copy()

    metric_rows[
        "metric_value"
    ] = metric_rows.apply(
        lambda r:
        selected_amount(
            r,
            r[
                "metric"
            ],
        ),
        axis=1,
    )

    metric_rows = metric_rows.loc[
        metric_rows[
            "metric_value"
        ].notna()
    ].copy()

    # --------------------------------------------------------
    # CFS priority / OFS fallback
    # --------------------------------------------------------

    fs_presence = (
        metric_rows.groupby(
            [
                "rcept_no",
                "fs_div",
            ]
        )[
            "metric"
        ]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    def choose_fs(
        rcept_no: str,
    ) -> str | None:

        if rcept_no not in fs_presence.index:
            return None

        row = fs_presence.loc[
            rcept_no
        ]

        if (
            "CFS" in row.index
            and row.get(
                "CFS",
                0,
            )
            > 0
        ):
            return "CFS"

        if (
            "OFS" in row.index
            and row.get(
                "OFS",
                0,
            )
            > 0
        ):
            return "OFS"

        return None

    chosen_fs_map = {
        receipt:
        choose_fs(
            receipt
        )
        for receipt
        in verified_receipts
    }

    metric_rows[
        "chosen_fs_div"
    ] = metric_rows[
        "rcept_no"
    ].map(
        chosen_fs_map
    )

    selected_rows = metric_rows.loc[
        metric_rows[
            "fs_div"
        ].astype(str)
        .eq(
            metric_rows[
                "chosen_fs_div"
            ].astype(str)
        )
    ].copy()

    # Duplicate same metric can exist; keep first after deterministic sort.
    sort_cols = [
        c
        for c in [
            "rcept_no",
            "metric",
            "ord",
            "account_nm",
        ]
        if c in selected_rows.columns
    ]

    selected_rows = selected_rows.sort_values(
        sort_cols
    )

    dedup = (
        selected_rows.drop_duplicates(
            subset=[
                "rcept_no",
                "metric",
            ],
            keep="first",
        )
    )

    wide_metrics = (
        dedup.pivot(
            index="rcept_no",
            columns="metric",
            values="metric_value",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Coverage table for ALL targets
    # --------------------------------------------------------

    base = targets.copy()

    base = base.merge(
        manifest[
            [
                "expected_rcept_no",
                "collection_status",
                "pit_verified",
                "api_row_count",
                "api_error",
            ]
        ],
        on="expected_rcept_no",
        how="left",
    )

    base = base.rename(
        columns={
            "expected_rcept_no":
            "rcept_no",
        }
    )

    base = base.merge(
        wide_metrics,
        on="rcept_no",
        how="left",
    )

    base[
        "selected_fs_div"
    ] = base[
        "rcept_no"
    ].map(
        chosen_fs_map
    )

    for metric in CORE_METRICS:
        if metric not in base.columns:
            base[
                metric
            ] = np.nan

        base[
            f"has_{metric}"
        ] = base[
            metric
        ].notna()

    has_cols = [
        f"has_{metric}"
        for metric in CORE_METRICS
    ]

    base[
        "core_metric_count"
    ] = base[
        has_cols
    ].sum(
        axis=1
    )

    base[
        "missing_core_metrics"
    ] = base.apply(
        lambda r:
        " | ".join(
            metric
            for metric in CORE_METRICS
            if not bool(
                r[
                    f"has_{metric}"
                ]
            )
        ),
        axis=1,
    )

    base[
        "needs_full_api_fallback"
    ] = (
        base[
            "collection_status"
        ].eq(
            "no_data"
        )
        | (
            base[
                "collection_status"
            ].eq(
                "available"
            )
            & base[
                "core_metric_count"
            ].lt(
                len(
                    CORE_METRICS
                )
            )
        )
    )

    base[
        "fallback_reason"
    ] = np.select(
        [
            base[
                "collection_status"
            ].eq(
                "no_data"
            ),
            (
                base[
                    "collection_status"
                ].eq(
                    "available"
                )
                & base[
                    "core_metric_count"
                ].lt(
                    len(
                        CORE_METRICS
                    )
                )
            ),
        ],
        [
            "multi_account_no_data",
            "major_account_core_metric_missing",
        ],
        default="not_needed",
    )

    base.to_parquet(
        COVERAGE_PARQUET,
        index=False,
    )

    base.to_csv(
        COVERAGE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    fallback = (
        base.loc[
            base[
                "needs_full_api_fallback"
            ]
        ]
        .copy()
    )

    fallback.to_parquet(
        FALLBACK_PARQUET,
        index=False,
    )

    fallback.to_csv(
        FALLBACK_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        "\n[Selected FS division]"
    )

    print(
        base[
            "selected_fs_div"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Core metric coverage]"
    )

    coverage = pd.DataFrame(
        {
            "count":
            [
                int(
                    base[
                        f"has_{metric}"
                    ].sum()
                )
                for metric
                in CORE_METRICS
            ],
        },
        index=CORE_METRICS,
    )

    coverage[
        "pct_all_targets"
    ] = (
        coverage[
            "count"
        ]
        / len(
            base
        )
        * 100
    )

    print(
        coverage.to_string()
    )

    print(
        "\n[Core metric count]"
    )

    print(
        base[
            "core_metric_count"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    print(
        "\n[Fallback reason]"
    )

    print(
        base[
            "fallback_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Top missing combinations]"
    )

    missing_combo = (
        base.loc[
            base[
                "missing_core_metrics"
            ]
            .ne(
                ""
            ),
            "missing_core_metrics",
        ]
        .value_counts()
        .head(
            20
        )
    )

    if missing_combo.empty:
        print(
            "None"
        )
    else:
        print(
            missing_combo.to_string()
        )

    print(
        "\n[Unique major-account labels]"
    )

    labels = (
        verified_rows[
            "account_nm"
        ]
        .astype("string")
        .value_counts(
            dropna=False
        )
        .head(
            80
        )
    )

    print(
        labels.to_string()
    )

    print(
        f"\nCoverage : {COVERAGE_PARQUET}"
    )

    print(
        f"Fallback : {FALLBACK_PARQUET}"
    )

    print(
        "\n다음 단계:"
        "\n05A5-H3 fnlttSinglAcntAll selective fallback collector"
        "\n→ fallback receipt만 full financial statement 조회"
        "\n→ account_id 기반 strict selector"
    )


if __name__ == "__main__":
    main()
