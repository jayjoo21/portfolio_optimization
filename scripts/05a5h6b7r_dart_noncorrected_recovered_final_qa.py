from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7R. Non-corrected Recovered Wide Final QA
#
# 목적
# ------------------------------------------------------------
# H6B7Q recovered wide를 canonical 승격하기 전에
# 원본 vs recovered 전체 16,869행을 마지막으로 검증한다.
#
# 검증:
# 1) schema / row count / rcept_no key 동일
# 2) core6 6개 metric 이외의 컬럼은 단 1 cell도 변경되지 않았는지
# 3) 변경된 core metric cell이 정확히 142 receipt × 6 = 852개인지
# 4) 변경 receipt set이 H6B7N source-ready + H6B7P YTD PASS set과 정확히 같은지
# 5) REVIEW 35건이 하나도 변경되지 않았는지
# 6) 각 merged receipt가 before core0 → after core6인지
# 7) recovered 값이 dry-run source core6와 일치하는지
# 8) 전체 balance identity hard fail = 0인지
# 9) unchanged receipt의 core count / balance status가 그대로인지
#
# 원본/recovered parquet은 수정하지 않는다.
#
# 실행:
# python scripts\05a5h6b7r_dart_noncorrected_recovered_final_qa.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

ORIGINAL = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_final.parquet"
)

RECOVERED = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7q_ytd_recovered.parquet"
)

DRYRUN = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_receipt_summary.csv"
)

YTD_FINAL = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_final_audit.csv"
)

OUT_DIFF = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7r_cell_diff.csv"
)

OUT_RECEIPT_QA = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7r_receipt_qa.csv"
)

OUT_INCOMPLETE = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7r_remaining_incomplete.csv"
)


EXPECTED_ROWS = 16_869
EXPECTED_MERGED_RECEIPTS = 142
EXPECTED_CHANGED_CORE_CELLS = 142 * 6
EXPECTED_REVIEW_YTD_RECEIPTS = 35


TARGET_ALIASES = {
    "assets": [
        "assets",
        "assets_total",
    ],
    "liabilities": [
        "liabilities",
        "liabilities_total",
    ],
    "equity": [
        "equity",
        "equity_total",
    ],
    "revenue": [
        "revenue",
        "revenue_cumulative",
    ],
    "operating_income": [
        "operating_income",
        "operating_income_cumulative",
    ],
    "net_income": [
        "net_income",
        "net_income_total",
        "net_income_cumulative",
        "net_income_total_cumulative",
    ],
}


# ============================================================
# Helpers
# ============================================================


def normalize_receipt(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def resolve_metric_mapping(
    frame: pd.DataFrame,
) -> dict[str, str]:

    mapping = {}

    for source_col, candidates in TARGET_ALIASES.items():

        hits = [
            col
            for col in candidates
            if col in frame.columns
        ]

        if len(hits) != 1:
            raise RuntimeError(
                f"Metric mapping for {source_col} must resolve to exactly one column. "
                f"Found: {hits}; tried: {candidates}"
            )

        mapping[source_col] = hits[0]

    return mapping


def equal_series(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Exact equality with NaN==NaN.
    We expect no transformation of unchanged cells.
    """
    return (
        left.eq(right)
        | (
            left.isna()
            & right.isna()
        )
    )


def near_equal(
    a,
    b,
) -> bool:

    try:
        a = float(a)
        b = float(b)
    except Exception:
        return False

    if not (
        math.isfinite(a)
        and math.isfinite(b)
    ):
        return False

    diff = abs(a - b)

    denom = max(
        abs(a),
        abs(b),
        1.0,
    )

    return (
        diff <= 1_000_000
        or diff / denom <= 1e-6
    )


def balance_status(
    assets,
    liabilities,
    equity,
) -> tuple[str, float, float]:

    if any(
        pd.isna(x)
        for x in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return (
            "MISSING",
            np.nan,
            np.nan,
        )

    a = float(assets)
    l = float(liabilities)
    e = float(equity)

    gap = a - l - e
    abs_gap = abs(gap)

    denom = max(
        abs(a),
        abs(l) + abs(e),
        1.0,
    )

    rel_gap = abs_gap / denom

    if abs_gap == 0:
        status = "EXACT"

    elif (
        abs_gap <= 2_000_000
        or rel_gap <= 1e-6
    ):
        status = "ROUNDING"

    else:
        status = "HARD_FAIL"

    return (
        status,
        gap,
        rel_gap,
    )


def boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    Robust bool conversion for CSV-loaded flags.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        ORIGINAL,
        RECOVERED,
        DRYRUN,
        YTD_FINAL,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    original = pd.read_parquet(ORIGINAL)
    recovered = pd.read_parquet(RECOVERED)

    dryrun = pd.read_csv(
        DRYRUN,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    ytd = pd.read_csv(
        YTD_FINAL,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "05A5-H6B7R NON-CORRECTED RECOVERED WIDE FINAL QA"
    )
    print(
        "=" * 120
    )

    print(
        f"\nOriginal rows : {len(original):,}"
    )
    print(
        f"Recovered rows: {len(recovered):,}"
    )

    # --------------------------------------------------------
    # 1. Structural checks
    # --------------------------------------------------------

    if len(original) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Original row count expected {EXPECTED_ROWS:,}, got {len(original):,}"
        )

    if len(recovered) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Recovered row count expected {EXPECTED_ROWS:,}, got {len(recovered):,}"
        )

    if list(original.columns) != list(recovered.columns):
        raise RuntimeError(
            "Original/recovered schema or column order differs."
        )

    if "rcept_no" not in original.columns:
        raise RuntimeError(
            "rcept_no missing from wide panel."
        )

    for frame in [
        original,
        recovered,
        dryrun,
        ytd,
    ]:
        frame["rcept_no"] = normalize_receipt(
            frame["rcept_no"]
        )

    if original["rcept_no"].duplicated().any():
        raise RuntimeError(
            "Original contains duplicate rcept_no."
        )

    if recovered["rcept_no"].duplicated().any():
        raise RuntimeError(
            "Recovered contains duplicate rcept_no."
        )

    original_keys = original["rcept_no"].tolist()
    recovered_keys = recovered["rcept_no"].tolist()

    if original_keys != recovered_keys:
        raise RuntimeError(
            "Row order or rcept_no keys changed between original and recovered."
        )

    metric_mapping = resolve_metric_mapping(
        original
    )

    target_cols = list(
        metric_mapping.values()
    )

    print(
        "\n[Metric mapping]"
    )
    for source, target in metric_mapping.items():
        print(
            f"{source:20s} -> {target}"
        )

    # --------------------------------------------------------
    # 2. Build validated and review sets
    # --------------------------------------------------------

    dryrun_ready = dryrun.loc[
        boolean_series(
            dryrun[
                "core6_source_ready"
            ]
        )
    ].copy()

    validated = dryrun_ready.merge(
        ytd[
            [
                "rcept_no",
                "income_ytd_final_status",
                "income_ytd_final_reason",
            ]
        ],
        on="rcept_no",
        how="inner",
        validate="one_to_one",
    )

    validated = validated.loc[
        validated[
            "income_ytd_final_status"
        ].eq(
            "PASS_ALL_INCOME_YTD"
        )
    ].copy()

    validated_set = set(
        validated[
            "rcept_no"
        ]
    )

    review_set = set(
        ytd.loc[
            ytd[
                "income_ytd_final_status"
            ].eq(
                "REVIEW_INCOME_DURATION"
            ),
            "rcept_no",
        ]
    )

    if len(validated_set) != EXPECTED_MERGED_RECEIPTS:
        raise RuntimeError(
            f"Expected {EXPECTED_MERGED_RECEIPTS} validated receipts, "
            f"got {len(validated_set)}."
        )

    if len(review_set) != EXPECTED_REVIEW_YTD_RECEIPTS:
        raise RuntimeError(
            f"Expected {EXPECTED_REVIEW_YTD_RECEIPTS} YTD review receipts, "
            f"got {len(review_set)}."
        )

    if validated_set & review_set:
        raise RuntimeError(
            "Validated and review receipt sets overlap."
        )

    # --------------------------------------------------------
    # 3. Whole-panel non-core diff
    # --------------------------------------------------------

    non_core_cols = [
        col
        for col in original.columns
        if col not in target_cols
    ]

    changed_non_core = []

    for col in non_core_cols:

        equal = equal_series(
            original[col],
            recovered[col],
        )

        if not equal.all():

            bad_indices = np.flatnonzero(
                ~equal.to_numpy()
            )

            for idx in bad_indices[:1000]:

                changed_non_core.append(
                    {
                        "rcept_no":
                        original.iloc[idx][
                            "rcept_no"
                        ],

                        "column":
                        col,

                        "old_value":
                        original.iloc[idx][
                            col
                        ],

                        "new_value":
                        recovered.iloc[idx][
                            col
                        ],
                    }
                )

    print(
        "\n[Non-core changed cells]"
    )
    print(
        len(changed_non_core)
    )

    if changed_non_core:

        pd.DataFrame(
            changed_non_core
        ).to_csv(
            OUT_DIFF,
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Non-core columns changed. QA failed."
        )

    # --------------------------------------------------------
    # 4. Core cell diff
    # --------------------------------------------------------

    diff_records = []

    for source_col, target_col in (
        metric_mapping.items()
    ):

        equal = equal_series(
            original[
                target_col
            ],
            recovered[
                target_col
            ],
        )

        changed_indices = np.flatnonzero(
            ~equal.to_numpy()
        )

        for idx in changed_indices:

            old_value = original.iloc[idx][
                target_col
            ]

            new_value = recovered.iloc[idx][
                target_col
            ]

            diff_records.append(
                {
                    "rcept_no":
                    original.iloc[idx][
                        "rcept_no"
                    ],

                    "stock_code":
                    (
                        original.iloc[idx][
                            "stock_code"
                        ]
                        if "stock_code"
                        in original.columns
                        else ""
                    ),

                    "period_key":
                    (
                        original.iloc[idx][
                            "period_key"
                        ]
                        if "period_key"
                        in original.columns
                        else ""
                    ),

                    "source_metric":
                    source_col,

                    "target_metric":
                    target_col,

                    "old_value":
                    old_value,

                    "new_value":
                    new_value,

                    "old_was_null":
                    pd.isna(
                        old_value
                    ),

                    "new_is_non_null":
                    pd.notna(
                        new_value
                    ),
                }
            )

    diff = pd.DataFrame(
        diff_records
    )

    diff.to_csv(
        OUT_DIFF,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Changed core cells]"
    )
    print(
        f"{len(diff):,}"
    )

    if len(diff) != EXPECTED_CHANGED_CORE_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHANGED_CORE_CELLS} changed core cells, "
            f"got {len(diff)}."
        )

    if not (
        diff[
            "old_was_null"
        ].all()
        and diff[
            "new_is_non_null"
        ].all()
    ):
        raise RuntimeError(
            "A changed core cell was not NULL -> non-NULL."
        )

    changed_receipts = set(
        diff[
            "rcept_no"
        ]
    )

    print(
        "\n[Changed receipts]"
    )
    print(
        len(
            changed_receipts
        )
    )

    if changed_receipts != validated_set:

        missing = sorted(
            validated_set
            - changed_receipts
        )

        extra = sorted(
            changed_receipts
            - validated_set
        )

        raise RuntimeError(
            "Changed receipt set != validated receipt set.\n"
            f"Missing changed receipts: {missing[:20]}\n"
            f"Unexpected changed receipts: {extra[:20]}"
        )

    per_receipt_changed = (
        diff.groupby(
            "rcept_no"
        )
        .size()
    )

    if not per_receipt_changed.eq(
        6
    ).all():

        bad = per_receipt_changed.loc[
            ~per_receipt_changed.eq(
                6
            )
        ]

        raise RuntimeError(
            "Every validated receipt must have exactly six changed core cells.\n"
            + bad.to_string()
        )

    accidental_review = (
        changed_receipts
        & review_set
    )

    print(
        "\n[Changed YTD-review receipts]"
    )
    print(
        len(
            accidental_review
        )
    )

    if accidental_review:
        raise RuntimeError(
            "YTD REVIEW receipt was changed."
        )

    # --------------------------------------------------------
    # 5. Core-count before / after
    # --------------------------------------------------------

    original_core_count = (
        original[
            target_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    recovered_core_count = (
        recovered[
            target_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    original_lookup_core = pd.Series(
        original_core_count.to_numpy(),
        index=original[
            "rcept_no"
        ],
    )

    recovered_lookup_core = pd.Series(
        recovered_core_count.to_numpy(),
        index=recovered[
            "rcept_no"
        ],
    )

    validated_before = (
        original_lookup_core.loc[
            sorted(
                validated_set
            )
        ]
    )

    validated_after = (
        recovered_lookup_core.loc[
            sorted(
                validated_set
            )
        ]
    )

    if not validated_before.eq(
        0
    ).all():
        raise RuntimeError(
            "Not every merged receipt was core0 before merge."
        )

    if not validated_after.eq(
        6
    ).all():
        raise RuntimeError(
            "Not every merged receipt is core6 after merge."
        )

    unchanged_mask = ~original[
        "rcept_no"
    ].isin(
        validated_set
    )

    if not original_core_count.loc[
        unchanged_mask
    ].reset_index(
        drop=True
    ).equals(
        recovered_core_count.loc[
            unchanged_mask
        ].reset_index(
            drop=True
        )
    ):
        raise RuntimeError(
            "Core count changed for an unvalidated receipt."
        )

    print(
        "\n[Core-count distribution BEFORE]"
    )
    print(
        original_core_count
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n[Core-count distribution AFTER]"
    )
    print(
        recovered_core_count
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 6. Compare recovered values to dry-run source
    # --------------------------------------------------------

    recovered_lookup = (
        recovered.set_index(
            "rcept_no",
            drop=False,
        )
    )

    value_mismatches = []

    for row in validated.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        for source_col, target_col in (
            metric_mapping.items()
        ):

            expected = getattr(
                row,
                source_col,
            )

            actual = recovered_lookup.at[
                rcept_no,
                target_col,
            ]

            if not near_equal(
                expected,
                actual,
            ):

                value_mismatches.append(
                    {
                        "rcept_no":
                        rcept_no,

                        "metric":
                        source_col,

                        "expected_dryrun":
                        expected,

                        "actual_recovered":
                        actual,
                    }
                )

    print(
        "\n[Recovered vs dry-run value mismatches]"
    )
    print(
        len(
            value_mismatches
        )
    )

    if value_mismatches:
        raise RuntimeError(
            "Recovered values do not match dry-run source."
        )

    # --------------------------------------------------------
    # 7. Whole-panel balance QA
    # --------------------------------------------------------

    asset_col = metric_mapping[
        "assets"
    ]

    liabilities_col = metric_mapping[
        "liabilities"
    ]

    equity_col = metric_mapping[
        "equity"
    ]

    original_balance = []
    recovered_balance = []

    original_gaps = []
    recovered_gaps = []

    original_rel = []
    recovered_rel = []

    for a, l, e in zip(
        original[
            asset_col
        ],
        original[
            liabilities_col
        ],
        original[
            equity_col
        ],
    ):

        (
            status,
            gap,
            rel_gap,
        ) = balance_status(
            a,
            l,
            e,
        )

        original_balance.append(
            status
        )

        original_gaps.append(
            gap
        )

        original_rel.append(
            rel_gap
        )

    for a, l, e in zip(
        recovered[
            asset_col
        ],
        recovered[
            liabilities_col
        ],
        recovered[
            equity_col
        ],
    ):

        (
            status,
            gap,
            rel_gap,
        ) = balance_status(
            a,
            l,
            e,
        )

        recovered_balance.append(
            status
        )

        recovered_gaps.append(
            gap
        )

        recovered_rel.append(
            rel_gap
        )

    original_balance_s = pd.Series(
        original_balance,
        index=original.index,
    )

    recovered_balance_s = pd.Series(
        recovered_balance,
        index=recovered.index,
    )

    print(
        "\n[Balance QA BEFORE]"
    )
    print(
        original_balance_s
        .value_counts()
        .to_string()
    )

    print(
        "\n[Balance QA AFTER]"
    )
    print(
        recovered_balance_s
        .value_counts()
        .to_string()
    )

    hard_fail_count = int(
        recovered_balance_s.eq(
            "HARD_FAIL"
        ).sum()
    )

    print(
        "\n[Recovered hard balance fail]"
    )
    print(
        hard_fail_count
    )

    if hard_fail_count != 0:
        raise RuntimeError(
            "Recovered wide contains hard balance failures."
        )

    # Unchanged receipt balance status must stay identical.
    if not original_balance_s.loc[
        unchanged_mask
    ].reset_index(
        drop=True
    ).equals(
        recovered_balance_s.loc[
            unchanged_mask
        ].reset_index(
            drop=True
        )
    ):
        raise RuntimeError(
            "Balance status changed for an unvalidated receipt."
        )

    # All validated receipts must move MISSING -> EXACT/ROUNDING.
    validated_indices = original.index[
        original[
            "rcept_no"
        ].isin(
            validated_set
        )
    ]

    if not original_balance_s.loc[
        validated_indices
    ].eq(
        "MISSING"
    ).all():
        raise RuntimeError(
            "A validated receipt had non-missing balance before merge."
        )

    if not recovered_balance_s.loc[
        validated_indices
    ].isin(
        [
            "EXACT",
            "ROUNDING",
        ]
    ).all():
        raise RuntimeError(
            "A validated receipt failed recovered balance QA."
        )

    # --------------------------------------------------------
    # 8. Receipt QA table
    # --------------------------------------------------------

    receipt_qa = recovered[
        [
            col
            for col in [
                "rcept_no",
                "stock_code",
                "period_key",
            ]
            if col in recovered.columns
        ]
    ].copy()

    receipt_qa[
        "core_count_before"
    ] = original_core_count.to_numpy()

    receipt_qa[
        "core_count_after"
    ] = recovered_core_count.to_numpy()

    receipt_qa[
        "balance_status_before"
    ] = original_balance_s.to_numpy()

    receipt_qa[
        "balance_status_after"
    ] = recovered_balance_s.to_numpy()

    receipt_qa[
        "balance_gap_after"
    ] = recovered_gaps

    receipt_qa[
        "balance_rel_gap_after"
    ] = recovered_rel

    receipt_qa[
        "h6b7q_validated_merge"
    ] = receipt_qa[
        "rcept_no"
    ].isin(
        validated_set
    )

    receipt_qa[
        "ytd_review_unmerged"
    ] = receipt_qa[
        "rcept_no"
    ].isin(
        review_set
    )

    receipt_qa.to_csv(
        OUT_RECEIPT_QA,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 9. Remaining incomplete classification
    # --------------------------------------------------------

    incomplete = receipt_qa.loc[
        receipt_qa[
            "core_count_after"
        ].lt(
            6
        )
    ].copy()

    incomplete[
        "remaining_bucket"
    ] = np.select(
        [
            incomplete[
                "ytd_review_unmerged"
            ],

            incomplete[
                "core_count_after"
            ].eq(
                5
            ),

            incomplete[
                "core_count_after"
            ].eq(
                3
            ),

            incomplete[
                "core_count_after"
            ].eq(
                0
            ),
        ],
        [
            "H6B_YTD_REVIEW_UNMERGED",
            "PREEXISTING_CORE5",
            "PREEXISTING_CORE3",
            "REMAINING_CORE0_OTHER",
        ],
        default="OTHER_INCOMPLETE",
    )

    incomplete.to_csv(
        OUT_INCOMPLETE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Remaining incomplete buckets]"
    )
    print(
        incomplete[
            "remaining_bucket"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Final assertions from current pipeline state
    # --------------------------------------------------------

    after_dist = (
        recovered_core_count
        .value_counts()
        .to_dict()
    )

    expected_after = {
        0: 748,
        3: 2,
        5: 248,
        6: 15_871,
    }

    actual_after = {
        int(k):
        int(v)
        for k, v in after_dist.items()
    }

    if actual_after != expected_after:
        raise RuntimeError(
            "Recovered core-count distribution differs from expected current pipeline state.\n"
            f"Expected: {expected_after}\n"
            f"Actual:   {actual_after}"
        )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 120
    )
    print(
        "FINAL QA: PASS"
    )
    print(
        "=" * 120
    )

    print(
        f"\nRows preserved                  : {len(recovered):,}"
    )
    print(
        f"Validated merged receipts       : {len(validated_set):,}"
    )
    print(
        f"Changed core cells              : {len(diff):,}"
    )
    print(
        f"Non-core changed cells          : 0"
    )
    print(
        f"YTD review receipts changed     : 0"
    )
    print(
        f"Recovered vs dry-run mismatches : 0"
    )
    print(
        f"Hard balance failures           : 0"
    )
    print(
        f"Final core6                     : {actual_after.get(6, 0):,}"
    )
    print(
        f"Remaining core5                 : {actual_after.get(5, 0):,}"
    )
    print(
        f"Remaining core3                 : {actual_after.get(3, 0):,}"
    )
    print(
        f"Remaining core0                 : {actual_after.get(0, 0):,}"
    )

    print(
        "\nOutputs:"
    )
    print(
        f"- Cell diff           : {OUT_DIFF}"
    )
    print(
        f"- Receipt QA          : {OUT_RECEIPT_QA}"
    )
    print(
        f"- Remaining incomplete: {OUT_INCOMPLETE}"
    )

    print(
        "\nIf this script prints FINAL QA: PASS, "
        "the H6B7Q recovered parquet is safe to promote as the "
        "validated non-corrected wide candidate."
    )


if __name__ == "__main__":
    main()
