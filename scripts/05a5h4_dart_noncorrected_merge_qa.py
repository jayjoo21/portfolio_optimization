from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H4. Non-Corrected Major + Full-Fallback Merge QA
#
# 목적
# ------------------------------------------------------------
# H2 major-account snapshot과 H3 selective full-statement 결과를 합친다.
#
# 병합 원칙
# ------------------------------------------------------------
# 1) H2 major-account 값이 이미 있으면 그대로 유지한다.
# 2) H2 값이 missing일 때만 H3 full-statement 값으로 채운다.
# 3) H3가 기존 H2 값을 덮어쓰지 않는다.
# 4) 같은 receipt/metric에 H2와 H3 값이 둘 다 존재하면
#    overlap QA로 숫자 일치 여부를 별도 점검한다.
# 5) receipt mismatch/api_error가 있는 H3 값은 사용하지 않는다.
# 6) balance equation은 corrected pipeline과 같은 tolerance로 QA한다.
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_pit_receipt_wide.parquet/csv
#   dart_noncorrected_merge_overlap_qa.csv
#   dart_noncorrected_source_level_fallback_targets.parquet/csv
#
# 다음 단계
# ------------------------------------------------------------
# - unresolved가 거의 없으면 corrected reconciled와 결합
# - unresolved가 의미 있게 남으면 05A5-H5 receipt-specific
#   XBRL/document source-level fallback
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h4_dart_noncorrected_merge_qa.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

H2_COVERAGE = (
    INTERIM
    / "dart_noncorrected_major_account_coverage.parquet"
)

H3_SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

H3_MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

OUT_WIDE_PARQUET = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide.parquet"
)

OUT_WIDE_CSV = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide.csv"
)

OUT_OVERLAP = (
    INTERIM
    / "dart_noncorrected_merge_overlap_qa.csv"
)

OUT_UNRESOLVED_PARQUET = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_targets.parquet"
)

OUT_UNRESOLVED_CSV = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_targets.csv"
)


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]

EXTRA_METRICS = [
    "equity_parent",
    "net_income_parent_cumulative",
]

ALL_METRICS = (
    CORE_METRICS
    + EXTRA_METRICS
)

ABS_BALANCE_TOLERANCE_KRW = 2_000_000
REL_BALANCE_TOLERANCE = 1e-6

OVERLAP_ABS_TOLERANCE_KRW = 1.0
OVERLAP_REL_TOLERANCE = 1e-9


def to_receipt_string(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def safe_relative_diff(
    a: pd.Series,
    b: pd.Series,
) -> pd.Series:
    denom = pd.concat(
        [
            a.abs(),
            b.abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    denom = denom.replace(
        0,
        np.nan,
    )

    result = (
        (a - b).abs()
        / denom
    )

    both_zero = (
        a.eq(0)
        & b.eq(0)
    )

    result.loc[
        both_zero
    ] = 0.0

    return result


def balance_classification(
    row: pd.Series,
) -> str:

    values = [
        row.get(
            "assets"
        ),
        row.get(
            "liabilities"
        ),
        row.get(
            "equity_total"
        ),
    ]

    if any(
        pd.isna(
            x
        )
        for x in values
    ):
        return "missing"

    gap = float(
        row[
            "balance_gap"
        ]
    )

    rel_gap = row[
        "balance_relative_gap"
    ]

    if gap == 0:
        return "exact_pass"

    if (
        abs(
            gap
        )
        <= ABS_BALANCE_TOLERANCE_KRW
        or (
            pd.notna(
                rel_gap
            )
            and rel_gap
            <= REL_BALANCE_TOLERANCE
        )
    ):
        return "rounding_pass"

    return "fail"


def main():

    for path in [
        H2_COVERAGE,
        H3_SELECTED,
        H3_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    h2 = pd.read_parquet(
        H2_COVERAGE
    )

    h3_selected = pd.read_parquet(
        H3_SELECTED
    )

    h3_manifest = pd.read_parquet(
        H3_MANIFEST
    )

    h2[
        "rcept_no"
    ] = to_receipt_string(
        h2[
            "rcept_no"
        ]
    )

    h3_manifest[
        "rcept_no"
    ] = to_receipt_string(
        h3_manifest[
            "rcept_no"
        ]
    )

    h3_selected[
        "_expected_rcept_no"
    ] = to_receipt_string(
        h3_selected[
            "_expected_rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "05A5-H4 NON-CORRECTED MAJOR + FULL-FALLBACK MERGE QA"
    )

    print(
        "=" * 90
    )

    print(
        f"\nH2 target receipts : "
        f"{h2['rcept_no'].nunique():,}"
    )

    print(
        f"H3 manifest receipts: "
        f"{h3_manifest['rcept_no'].nunique():,}"
    )

    # --------------------------------------------------------
    # 0) H3 safety gate
    # --------------------------------------------------------

    unsafe_h3 = h3_manifest.loc[
        h3_manifest[
            "collection_status"
        ].isin(
            [
                "api_error",
                "receipt_mismatch",
            ]
        )
    ].copy()

    if not unsafe_h3.empty:
        raise RuntimeError(
            "H3에 api_error/receipt_mismatch가 남아 있습니다.\n"
            + unsafe_h3[
                [
                    "stock_code",
                    "canonical_period_key",
                    "rcept_no",
                    "collection_status",
                    "api_error",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    verified_h3_receipts = set(
        h3_manifest.loc[
            h3_manifest[
                "collection_status"
            ].eq(
                "available"
            )
            & h3_manifest[
                "pit_verified"
            ].eq(
                True
            ),
            "rcept_no",
        ].astype(str)
    )

    h3_selected = h3_selected.loc[
        h3_selected[
            "_expected_rcept_no"
        ]
        .astype(str)
        .isin(
            verified_h3_receipts
        )
    ].copy()

    # --------------------------------------------------------
    # 1) H3 metric wide
    # --------------------------------------------------------

    if h3_selected.empty:
        h3_wide = pd.DataFrame(
            {
                "rcept_no":
                pd.Series(
                    dtype="string"
                )
            }
        )
    else:
        h3_wide = (
            h3_selected.pivot_table(
                index="_expected_rcept_no",
                columns="metric",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "_expected_rcept_no":
                    "rcept_no",
                }
            )
        )

    for metric in ALL_METRICS:
        if metric not in h3_wide.columns:
            h3_wide[
                metric
            ] = np.nan

    rename_map = {
        metric:
        f"{metric}__h3"
        for metric in ALL_METRICS
    }

    h3_wide = h3_wide.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # 2) Merge H2 + H3
    # --------------------------------------------------------

    out = h2.copy()

    for metric in CORE_METRICS:
        if metric not in out.columns:
            out[
                metric
            ] = np.nan

    out = out.merge(
        h3_wide,
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 3) Overlap QA
    # --------------------------------------------------------

    overlap_records = []

    for metric in CORE_METRICS:

        h2_value = pd.to_numeric(
            out[
                metric
            ],
            errors="coerce",
        )

        h3_value = pd.to_numeric(
            out[
                f"{metric}__h3"
            ],
            errors="coerce",
        )

        both = (
            h2_value.notna()
            & h3_value.notna()
        )

        if not both.any():
            continue

        abs_diff = (
            h2_value.loc[
                both
            ]
            - h3_value.loc[
                both
            ]
        ).abs()

        rel_diff = safe_relative_diff(
            h2_value.loc[
                both
            ],
            h3_value.loc[
                both
            ],
        )

        mismatch = (
            abs_diff
            > OVERLAP_ABS_TOLERANCE_KRW
        ) & (
            rel_diff
            > OVERLAP_REL_TOLERANCE
        )

        temp = out.loc[
            both,
            [
                "stock_code",
                "corp_code",
                "corp_name",
                "canonical_period_key",
                "rcept_no",
                "selected_fs_div",
            ],
        ].copy()

        temp[
            "metric"
        ] = metric

        temp[
            "h2_major_value"
        ] = h2_value.loc[
            both
        ].values

        temp[
            "h3_full_value"
        ] = h3_value.loc[
            both
        ].values

        temp[
            "absolute_difference"
        ] = abs_diff.values

        temp[
            "relative_difference"
        ] = rel_diff.values

        temp[
            "overlap_match"
        ] = (
            ~mismatch
        ).values

        overlap_records.append(
            temp
        )

    if overlap_records:
        overlap = pd.concat(
            overlap_records,
            ignore_index=True,
        )
    else:
        overlap = pd.DataFrame()

    overlap.to_csv(
        OUT_OVERLAP,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4) Fill only H2 missing metrics from H3
    # --------------------------------------------------------

    for metric in CORE_METRICS:

        h2_missing = out[
            metric
        ].isna()

        h3_available = out[
            f"{metric}__h3"
        ].notna()

        fill_mask = (
            h2_missing
            & h3_available
        )

        source_col = (
            f"{metric}_source"
        )

        out[
            source_col
        ] = np.where(
            out[
                metric
            ].notna(),
            "multi_account",
            "missing",
        )

        out.loc[
            fill_mask,
            metric,
        ] = out.loc[
            fill_mask,
            f"{metric}__h3",
        ]

        out.loc[
            fill_mask,
            source_col,
        ] = (
            "full_statement_fallback"
        )

    # H3-only parent metrics
    for metric in EXTRA_METRICS:
        out[
            metric
        ] = out[
            f"{metric}__h3"
        ]

        out[
            f"{metric}_source"
        ] = np.where(
            out[
                metric
            ].notna(),
            "full_statement_fallback",
            "missing",
        )

    # --------------------------------------------------------
    # 5) Pre/post coverage
    # --------------------------------------------------------

    before = {}
    after = {}
    filled = {}

    for metric in CORE_METRICS:

        before[
            metric
        ] = int(
            h2[
                metric
            ].notna()
            .sum()
        )

        after[
            metric
        ] = int(
            out[
                metric
            ].notna()
            .sum()
        )

        filled[
            metric
        ] = (
            after[
                metric
            ]
            - before[
                metric
            ]
        )

    before_complete = int(
        h2[
            CORE_METRICS
        ]
        .notna()
        .all(
            axis=1
        )
        .sum()
    )

    out[
        "core_metric_count_final"
    ] = (
        out[
            CORE_METRICS
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    out[
        "missing_core_metrics_final"
    ] = out.apply(
        lambda r:
        " | ".join(
            metric
            for metric
            in CORE_METRICS
            if pd.isna(
                r[
                    metric
                ]
            )
        ),
        axis=1,
    )

    after_complete = int(
        out[
            "core_metric_count_final"
        ]
        .eq(
            len(
                CORE_METRICS
            )
        )
        .sum()
    )

    # --------------------------------------------------------
    # 6) Balance equation QA
    # --------------------------------------------------------

    out[
        "balance_gap"
    ] = (
        out[
            "assets"
        ]
        - out[
            "liabilities"
        ]
        - out[
            "equity_total"
        ]
    )

    out[
        "balance_relative_gap"
    ] = (
        out[
            "balance_gap"
        ]
        .abs()
        / out[
            "assets"
        ]
        .abs()
        .replace(
            0,
            np.nan,
        )
    )

    out[
        "balance_qa_class"
    ] = out.apply(
        balance_classification,
        axis=1,
    )

    out[
        "balance_equation_pass"
    ] = out[
        "balance_qa_class"
    ].isin(
        [
            "exact_pass",
            "rounding_pass",
        ]
    )

    # --------------------------------------------------------
    # 7) Final provenance / unresolved targets
    # --------------------------------------------------------

    h3_manifest_small = (
        h3_manifest[
            [
                "rcept_no",
                "collection_status",
                "pit_verified",
                "chosen_fs_div",
                "api_status",
                "api_message",
                "fallback_reason",
            ]
        ]
        .rename(
            columns={
                "collection_status":
                "h3_collection_status",
                "pit_verified":
                "h3_pit_verified",
                "chosen_fs_div":
                "h3_chosen_fs_div",
                "api_status":
                "h3_api_status",
                "api_message":
                "h3_api_message",
                "fallback_reason":
                "h3_fallback_reason",
            }
        )
    )

    out = out.merge(
        h3_manifest_small,
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    out[
        "needs_source_level_fallback"
    ] = (
        out[
            "core_metric_count_final"
        ]
        .lt(
            len(
                CORE_METRICS
            )
        )
    )

    def unresolved_reason(
        row: pd.Series,
    ) -> str:

        reasons = []

        if row.get(
            "h3_collection_status"
        ) == "no_data":
            reasons.append(
                "full_api_no_data"
            )

        if (
            row[
                "core_metric_count_final"
            ]
            < len(
                CORE_METRICS
            )
        ):
            reasons.append(
                "core_metric_still_missing"
            )

        if not reasons:
            return ""

        return " | ".join(
            reasons
        )

    out[
        "source_level_fallback_reason"
    ] = out.apply(
        unresolved_reason,
        axis=1,
    )

    unresolved = out.loc[
        out[
            "needs_source_level_fallback"
        ]
    ].copy()

    # --------------------------------------------------------
    # 8) Save
    # --------------------------------------------------------

    # Drop temporary H3-wide helper columns from final wide output.
    helper_cols = [
        f"{metric}__h3"
        for metric in ALL_METRICS
        if f"{metric}__h3"
        in out.columns
    ]

    final_out = out.drop(
        columns=helper_cols,
    )

    final_out.to_parquet(
        OUT_WIDE_PARQUET,
        index=False,
    )

    final_out.to_csv(
        OUT_WIDE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    unresolved.to_parquet(
        OUT_UNRESOLVED_PARQUET,
        index=False,
    )

    unresolved.to_csv(
        OUT_UNRESOLVED_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 9) Print summary
    # --------------------------------------------------------

    print(
        "\n[Core metric coverage: before -> after]"
    )

    coverage_table = pd.DataFrame(
        {
            "before_h3":
            pd.Series(
                before
            ),
            "after_h3":
            pd.Series(
                after
            ),
            "filled_by_h3":
            pd.Series(
                filled
            ),
        }
    )

    coverage_table[
        "after_pct"
    ] = (
        coverage_table[
            "after_h3"
        ]
        / len(
            final_out
        )
        * 100
    )

    print(
        coverage_table.to_string()
    )

    print(
        "\n[Complete core-6 receipts]"
    )

    print(
        f"before H3 : "
        f"{before_complete:,} / {len(final_out):,}"
    )

    print(
        f"after H3  : "
        f"{after_complete:,} / {len(final_out):,}"
    )

    print(
        f"newly completed: "
        f"{after_complete - before_complete:,}"
    )

    print(
        "\n[Final core metric count]"
    )

    print(
        final_out[
            "core_metric_count_final"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    print(
        "\n[Top remaining missing combinations]"
    )

    remaining_combo = (
        final_out.loc[
            final_out[
                "missing_core_metrics_final"
            ]
            .ne(
                ""
            ),
            "missing_core_metrics_final",
        ]
        .value_counts()
        .head(
            20
        )
    )

    if remaining_combo.empty:
        print(
            "None"
        )
    else:
        print(
            remaining_combo.to_string()
        )

    print(
        "\n[Overlap H2 major vs H3 full]"
    )

    if overlap.empty:
        print(
            "No overlapping values"
        )
    else:
        print(
            f"comparisons : "
            f"{len(overlap):,}"
        )

        print(
            f"matches     : "
            f"{int(overlap['overlap_match'].sum()):,}"
        )

        print(
            f"mismatches  : "
            f"{int((~overlap['overlap_match']).sum()):,}"
        )

        mismatch_summary = (
            overlap.loc[
                ~overlap[
                    "overlap_match"
                ]
            ]
            .groupby(
                "metric"
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        if not mismatch_summary.empty:
            print(
                "\n[Mismatch by metric]"
            )

            print(
                mismatch_summary.to_string()
            )

    print(
        "\n[Balance QA]"
    )

    print(
        final_out[
            "balance_qa_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    hard_balance_fail = (
        final_out.loc[
            final_out[
                "balance_qa_class"
            ]
            .eq(
                "fail"
            )
        ]
    )

    print(
        f"\nHard balance failures: "
        f"{len(hard_balance_fail):,}"
    )

    if not hard_balance_fail.empty:
        print(
            hard_balance_fail[
                [
                    "stock_code",
                    "corp_name",
                    "canonical_period_key",
                    "rcept_no",
                    "assets",
                    "liabilities",
                    "equity_total",
                    "balance_gap",
                    "balance_relative_gap",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n[Source-level fallback candidates]"
    )

    print(
        f"{len(unresolved):,}"
    )

    if not unresolved.empty:
        print(
            unresolved[
                "source_level_fallback_reason"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        f"\nFinal wide : "
        f"{OUT_WIDE_PARQUET}"
    )

    print(
        f"Overlap QA : "
        f"{OUT_OVERLAP}"
    )

    print(
        f"Unresolved : "
        f"{OUT_UNRESOLVED_PARQUET}"
    )

    print(
        "\n다음 판단:"
        "\n- overlap mismatch / hard balance failure가 있으면 먼저 audit"
        "\n- unresolved가 의미 있게 남으면 H5 receipt-specific XBRL/document fallback"
        "\n- QA가 깨끗하면 corrected reconciled + non-corrected 통합으로 이동"
    )


if __name__ == "__main__":
    main()
