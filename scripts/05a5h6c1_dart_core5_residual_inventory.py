from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6C1. CORE5 RESIDUAL INVENTORY
#
# 목적
# ------------------------------------------------------------
# H6B에서 freeze한 validated non-corrected wide를 기준으로,
# core5 residual 248건을 별도 트랙으로 분리한다.
#
# 여기서는 복구하지 않는다.
#
# 확인:
# - 정확히 248건인지
# - missing metric별 개수
# - stock_code / period_key / report type 분포
# - 특정 종목 반복 여부
# - missing metric별 receipt 목록 저장
#
# 실행:
# python scripts\05a5h6c1_dart_core5_residual_inventory.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_validated.parquet"
)

OUT_CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

OUT_CORE5_CSV = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.csv"
)

OUT_SUMMARY = (
    INTERIM
    / "dart_noncorrected_core5_residual_summary.csv"
)

OUT_STOCK_REPEAT = (
    INTERIM
    / "dart_noncorrected_core5_residual_stock_repeat.csv"
)


EXPECTED_CORE5 = 248


TARGET_ALIASES = {
    "assets": [
        "assets",
        "assets_total",
    ],
    "liabilities": [
        "liabilities",
        "liabilities_total",
    ],
    "equity_total": [
        "equity",
        "equity_total",
    ],
    "revenue_cumulative": [
        "revenue",
        "revenue_cumulative",
    ],
    "operating_income_cumulative": [
        "operating_income",
        "operating_income_cumulative",
    ],
    "net_income_total_cumulative": [
        "net_income",
        "net_income_total",
        "net_income_cumulative",
        "net_income_total_cumulative",
    ],
}


def resolve_metric_mapping(
    frame: pd.DataFrame,
) -> dict[str, str]:

    mapping = {}

    for canonical, candidates in TARGET_ALIASES.items():

        hits = [
            col
            for col in candidates
            if col in frame.columns
        ]

        if len(hits) != 1:
            raise RuntimeError(
                f"Metric mapping failed for {canonical}. "
                f"Found {hits}; tried {candidates}"
            )

        mapping[canonical] = hits[0]

    return mapping


def report_type(
    period_key,
) -> str:

    text = str(
        period_key
    )

    if "사업보고서" in text:
        return "FY"

    if "반기보고서" in text:
        return "H1"

    if "분기보고서" in text:
        return "QUARTER"

    return "UNKNOWN"


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_parquet(
        INPUT
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C1 CORE5 RESIDUAL INVENTORY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    mapping = resolve_metric_mapping(
        df
    )

    print(
        "\n[Metric mapping]"
    )

    for canonical, actual in mapping.items():
        print(
            f"{canonical:30s} -> {actual}"
        )

    metric_cols = list(
        mapping.values()
    )

    core_count = (
        df[
            metric_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    core5 = df.loc[
        core_count.eq(
            5
        )
    ].copy()

    if len(core5) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 rows, "
            f"found {len(core5)}."
        )

    print(
        "\n[Core5 rows]"
    )

    print(
        f"{len(core5):,}"
    )

    # --------------------------------------------------------
    # Missing metric
    # --------------------------------------------------------

    reverse_mapping = {
        actual:
        canonical
        for canonical, actual in mapping.items()
    }

    missing_metric = []

    for _, row in core5.iterrows():

        missing = [
            reverse_mapping[col]
            for col in metric_cols
            if pd.isna(
                row[col]
            )
        ]

        if len(missing) != 1:
            raise RuntimeError(
                "A core5 row does not have exactly one missing metric."
            )

        missing_metric.append(
            missing[0]
        )

    core5[
        "missing_metric"
    ] = missing_metric

    core5[
        "report_type"
    ] = core5[
        "period_key"
    ].map(
        report_type
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n[Missing metric distribution]"
    )

    missing_dist = (
        core5[
            "missing_metric"
        ]
        .value_counts()
    )

    print(
        missing_dist.to_string()
    )

    print(
        "\n[Missing metric x report type]"
    )

    cross = pd.crosstab(
        core5[
            "missing_metric"
        ],
        core5[
            "report_type"
        ],
        dropna=False,
    )

    print(
        cross.to_string()
    )

    # --------------------------------------------------------
    # Stock repetition
    # --------------------------------------------------------

    stock_repeat = (
        core5.groupby(
            [
                "stock_code",
                "missing_metric",
            ],
            dropna=False,
        )
        .agg(
            receipt_count=(
                "rcept_no",
                "size",
            ),

            period_count=(
                "period_key",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "receipt_count",
                "stock_code",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    print(
        "\n[Top repeated stocks]"
    )

    print(
        stock_repeat.head(
            40
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Per-metric stock concentration
    # --------------------------------------------------------

    print(
        "\n[Top stocks by missing metric]"
    )

    for metric in missing_dist.index:

        temp = (
            core5.loc[
                core5[
                    "missing_metric"
                ].eq(
                    metric
                )
            ]
            .groupby(
                "stock_code",
                dropna=False,
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(
                20
            )
        )

        print(
            f"\n{metric}:"
        )

        print(
            temp.to_string()
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    core5.to_parquet(
        OUT_CORE5,
        index=False,
    )

    core5.to_csv(
        OUT_CORE5_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for metric, count in (
        missing_dist.items()
    ):

        subset = core5.loc[
            core5[
                "missing_metric"
            ].eq(
                metric
            )
        ]

        summary_rows.append(
            {
                "missing_metric":
                metric,

                "receipt_count":
                int(
                    count
                ),

                "unique_stocks":
                int(
                    subset[
                        "stock_code"
                    ].nunique()
                ),

                "FY":
                int(
                    subset[
                        "report_type"
                    ].eq(
                        "FY"
                    ).sum()
                ),

                "H1":
                int(
                    subset[
                        "report_type"
                    ].eq(
                        "H1"
                    ).sum()
                ),

                "QUARTER":
                int(
                    subset[
                        "report_type"
                    ].eq(
                        "QUARTER"
                    ).sum()
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    stock_repeat.to_csv(
        OUT_STOCK_REPEAT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Core5 parquet : "
        f"{OUT_CORE5}"
    )

    print(
        f"- Core5 CSV     : "
        f"{OUT_CORE5_CSV}"
    )

    print(
        f"- Summary       : "
        f"{OUT_SUMMARY}"
    )

    print(
        f"- Stock repeats : "
        f"{OUT_STOCK_REPEAT}"
    )

    print(
        "\nNext interpretation:"
        "\n- revenue-heavy이면 financial-sector semantics 트랙 우선"
        "\n- net-income residual이면 total vs attributable selector audit"
        "\n- op/assets 소수건은 exact receipt manual-style semantic audit"
        "\n- 아직 어떤 값도 복구/병합하지 않음"
    )


if __name__ == "__main__":
    main()
