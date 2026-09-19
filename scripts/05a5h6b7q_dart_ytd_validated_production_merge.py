from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7Q. YTD-Validated Production Merge
#
# 목적
# ------------------------------------------------------------
# H6B7P까지 통과한 receipt만 non-corrected wide panel에 병합한다.
#
# 병합 대상:
#   H6B7N core6_source_ready == True
#   AND
#   H6B7P income_ytd_final_status == PASS_ALL_INCOME_YTD
#
# 안전 규칙:
#   - 원본 wide parquet 절대 덮어쓰지 않음
#   - exact rcept_no 기준 merge
#   - 기존 null cell만 채움
#   - 기존 non-null 값과 새 값이 충돌하면 WRITE ABORT
#   - REVIEW / BLOCK receipt는 절대 merge하지 않음
#   - CFS/OFS / account-label / entity-scope / YTD 검증 결과를 그대로 존중
#
# 출력:
#   새 recovered parquet
#   merge audit CSV
#   receipt QA CSV
#
# 실행:
# python scripts\05a5h6b7q_dart_ytd_validated_production_merge.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

WIDE_INPUT = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_final.parquet"
)

DRYRUN_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_receipt_summary.csv"
)

YTD_FINAL = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_final_audit.csv"
)

WIDE_OUTPUT = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7q_ytd_recovered.parquet"
)

MERGE_AUDIT = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7q_merge_audit.csv"
)

RECEIPT_QA = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7q_receipt_qa.csv"
)


EXPECTED_PASS_COUNT = 142


# ============================================================
# Metric aliases
#
# Left side = dry-run source column.
# Right side = acceptable wide-panel target names.
# Exactly one existing target column must be identified.
# ============================================================

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
    wide: pd.DataFrame,
) -> dict[str, str]:

    mapping = {}

    for source_col, candidates in TARGET_ALIASES.items():

        hits = [
            col
            for col in candidates
            if col in wide.columns
        ]

        if len(hits) == 0:
            raise RuntimeError(
                f"Wide metric column not found for {source_col}. "
                f"Tried: {candidates}"
            )

        if len(hits) > 1:
            raise RuntimeError(
                f"Ambiguous wide metric mapping for {source_col}: {hits}. "
                "Resolve explicitly before merge."
            )

        mapping[
            source_col
        ] = hits[
            0
        ]

    return mapping


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

    diff = abs(
        a - b
    )

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
        pd.isna(
            value
        )
        for value in [
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

    a = float(
        assets
    )

    l = float(
        liabilities
    )

    e = float(
        equity
    )

    gap = (
        a
        - l
        - e
    )

    abs_gap = abs(
        gap
    )

    denom = max(
        abs(a),
        abs(l)
        + abs(e),
        1.0,
    )

    rel_gap = (
        abs_gap
        / denom
    )

    if abs_gap == 0:
        return (
            "EXACT",
            gap,
            rel_gap,
        )

    if (
        abs_gap <= 2_000_000
        or rel_gap <= 1e-6
    ):
        return (
            "ROUNDING",
            gap,
            rel_gap,
        )

    return (
        "HARD_FAIL",
        gap,
        rel_gap,
    )


def core_count(
    frame: pd.DataFrame,
    target_cols: list[str],
) -> pd.Series:

    return (
        frame[
            target_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        WIDE_INPUT,
        DRYRUN_RECEIPT,
        YTD_FINAL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    wide = pd.read_parquet(
        WIDE_INPUT
    )

    dryrun = pd.read_csv(
        DRYRUN_RECEIPT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    ytd = pd.read_csv(
        YTD_FINAL,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7Q YTD-VALIDATED PRODUCTION MERGE"
    )

    print(
        "=" * 120
    )

    print(
        f"\nWide rows: "
        f"{len(wide):,}"
    )

    print(
        f"Dry-run receipts: "
        f"{len(dryrun):,}"
    )

    print(
        f"YTD audit receipts: "
        f"{len(ytd):,}"
    )

    # --------------------------------------------------------
    # Basic key validation
    # --------------------------------------------------------

    if "rcept_no" not in wide.columns:
        raise RuntimeError(
            "Wide input has no rcept_no column."
        )

    wide[
        "rcept_no"
    ] = normalize_receipt(
        wide[
            "rcept_no"
        ]
    )

    dryrun[
        "rcept_no"
    ] = normalize_receipt(
        dryrun[
            "rcept_no"
        ]
    )

    ytd[
        "rcept_no"
    ] = normalize_receipt(
        ytd[
            "rcept_no"
        ]
    )

    if wide[
        "rcept_no"
    ].duplicated().any():

        dup = (
            wide.loc[
                wide[
                    "rcept_no"
                ].duplicated(
                    keep=False
                ),
                "rcept_no",
            ]
            .value_counts()
            .head(
                20
            )
        )

        raise RuntimeError(
            "Wide input contains duplicate rcept_no values.\n"
            + dup.to_string()
        )

    if dryrun[
        "rcept_no"
    ].duplicated().any():
        raise RuntimeError(
            "Dry-run receipt summary has duplicate rcept_no."
        )

    if ytd[
        "rcept_no"
    ].duplicated().any():
        raise RuntimeError(
            "YTD final audit has duplicate rcept_no."
        )

    metric_mapping = resolve_metric_mapping(
        wide
    )

    print(
        "\n[Resolved metric mapping]"
    )

    for source, target in metric_mapping.items():
        print(
            f"{source:20s} -> {target}"
        )

    target_cols = [
        metric_mapping[
            source
        ]
        for source in TARGET_ALIASES
    ]

    # --------------------------------------------------------
    # Build validated merge source
    # --------------------------------------------------------

    source_ready = dryrun.loc[
        dryrun[
            "core6_source_ready"
        ].fillna(
            False
        ).astype(
            bool
        )
    ].copy()

    validated = source_ready.merge(
        ytd[
            [
                "rcept_no",
                "income_ytd_final_status",
                "income_ytd_final_reason",
            ]
        ],
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    validated = validated.loc[
        validated[
            "income_ytd_final_status"
        ].eq(
            "PASS_ALL_INCOME_YTD"
        )
    ].copy()

    print(
        "\n[Validated merge receipts]"
    )

    print(
        f"{len(validated):,}"
    )

    if len(
        validated
    ) != EXPECTED_PASS_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_PASS_COUNT} validated receipts, "
            f"found {len(validated)}. "
            "Stop and inspect upstream audit before merge."
        )

    # All six source values must exist.
    source_metric_cols = list(
        TARGET_ALIASES.keys()
    )

    missing_source = (
        validated[
            source_metric_cols
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if missing_source.any():

        bad = validated.loc[
            missing_source,
            [
                "stock_code",
                "period_key",
                "rcept_no",
            ]
            + source_metric_cols
        ]

        raise RuntimeError(
            "Validated receipt contains missing core6 values.\n"
            + bad.head(
                30
            ).to_string(
                index=False
            )
        )

    # Every validated receipt must exist in wide.
    missing_keys = sorted(
        set(
            validated[
                "rcept_no"
            ]
        )
        - set(
            wide[
                "rcept_no"
            ]
        )
    )

    if missing_keys:
        raise RuntimeError(
            f"{len(missing_keys)} validated rcept_no not found in wide. "
            f"Sample: {missing_keys[:20]}"
        )

    # Ensure no non-PASS source sneaks in.
    allowed_receipts = set(
        validated[
            "rcept_no"
        ]
    )

    review_receipts = set(
        ytd.loc[
            ~ytd[
                "income_ytd_final_status"
            ].eq(
                "PASS_ALL_INCOME_YTD"
            ),
            "rcept_no",
        ]
    )

    overlap = (
        allowed_receipts
        & review_receipts
    )

    if overlap:
        raise RuntimeError(
            "PASS merge set overlaps REVIEW/BLOCK set. "
            f"Sample: {sorted(overlap)[:20]}"
        )

    # --------------------------------------------------------
    # Before counts
    # --------------------------------------------------------

    wide[
        "_core_count_before"
    ] = core_count(
        wide,
        target_cols,
    )

    before_distribution = (
        wide[
            "_core_count_before"
        ]
        .value_counts()
        .sort_index()
    )

    before_core6 = int(
        wide[
            "_core_count_before"
        ].eq(
            6
        ).sum()
    )

    # --------------------------------------------------------
    # Cell-level preflight
    # --------------------------------------------------------

    wide_index = (
        wide.set_index(
            "rcept_no",
            drop=False,
        )
    )

    audit_records = []

    conflict_records = []

    for row in validated.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        for source_col, target_col in metric_mapping.items():

            new_value = getattr(
                row,
                source_col
            )

            old_value = wide_index.at[
                rcept_no,
                target_col,
            ]

            if pd.isna(
                old_value
            ):

                action = (
                    "FILL_NULL"
                )

                conflict = False

            else:

                if near_equal(
                    old_value,
                    new_value,
                ):
                    action = (
                        "KEEP_EXISTING_EQUIVALENT"
                    )

                    conflict = False

                else:
                    action = (
                        "CONFLICT_EXISTING_NON_NULL"
                    )

                    conflict = True

                    conflict_records.append(
                        {
                            "rcept_no":
                            rcept_no,

                            "stock_code":
                            getattr(
                                row,
                                "stock_code",
                                "",
                            ),

                            "period_key":
                            getattr(
                                row,
                                "period_key",
                                "",
                            ),

                            "source_metric":
                            source_col,

                            "target_metric":
                            target_col,

                            "old_value":
                            old_value,

                            "new_value":
                            new_value,
                        }
                    )

            audit_records.append(
                {
                    "rcept_no":
                    rcept_no,

                    "stock_code":
                    getattr(
                        row,
                        "stock_code",
                        "",
                    ),

                    "period_key":
                    getattr(
                        row,
                        "period_key",
                        "",
                    ),

                    "source_metric":
                    source_col,

                    "target_metric":
                    target_col,

                    "old_value":
                    old_value,

                    "new_value":
                    new_value,

                    "merge_action":
                    action,

                    "is_conflict":
                    conflict,

                    "income_ytd_final_reason":
                    getattr(
                        row,
                        "income_ytd_final_reason",
                        "",
                    ),
                }
            )

    merge_audit = pd.DataFrame(
        audit_records
    )

    print(
        "\n[Cell-level preflight]"
    )

    print(
        merge_audit[
            "merge_action"
        ]
        .value_counts()
        .to_string()
    )

    if conflict_records:

        conflict_df = pd.DataFrame(
            conflict_records
        )

        merge_audit.to_csv(
            MERGE_AUDIT,
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Existing non-null conflicts detected. "
            "Output parquet NOT written.\n"
            + conflict_df.head(
                30
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Apply fill-null-only merge
    # --------------------------------------------------------

    recovered = wide.copy()

    recovered_index = (
        recovered.set_index(
            "rcept_no",
            drop=False,
        )
    )

    for row in validated.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        for source_col, target_col in metric_mapping.items():

            current = recovered_index.at[
                rcept_no,
                target_col,
            ]

            if pd.isna(
                current
            ):
                recovered_index.at[
                    rcept_no,
                    target_col,
                ] = getattr(
                    row,
                    source_col,
                )

    recovered = (
        recovered_index.reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # After QA
    # --------------------------------------------------------

    recovered[
        "_core_count_after"
    ] = core_count(
        recovered,
        target_cols,
    )

    after_distribution = (
        recovered[
            "_core_count_after"
        ]
        .value_counts()
        .sort_index()
    )

    after_core6 = int(
        recovered[
            "_core_count_after"
        ].eq(
            6
        ).sum()
    )

    # QA only the 142 validated receipts.
    qa_rows = []

    recovered_lookup = (
        recovered.set_index(
            "rcept_no",
            drop=False,
        )
    )

    for row in validated.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        wide_row = recovered_lookup.loc[
            rcept_no
        ]

        assets = wide_row[
            metric_mapping[
                "assets"
            ]
        ]

        liabilities = wide_row[
            metric_mapping[
                "liabilities"
            ]
        ]

        equity = wide_row[
            metric_mapping[
                "equity"
            ]
        ]

        (
            bal_status,
            bal_gap,
            bal_rel_gap,
        ) = balance_status(
            assets,
            liabilities,
            equity,
        )

        core_after = int(
            wide_row[
                target_cols
            ]
            .notna()
            .sum()
        )

        qa_rows.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                getattr(
                    row,
                    "stock_code",
                    "",
                ),

                "period_key":
                getattr(
                    row,
                    "period_key",
                    "",
                ),

                "core_count_after":
                core_after,

                "balance_status":
                bal_status,

                "balance_gap":
                bal_gap,

                "balance_rel_gap":
                bal_rel_gap,

                "income_ytd_final_reason":
                getattr(
                    row,
                    "income_ytd_final_reason",
                    "",
                ),
            }
        )

    receipt_qa = pd.DataFrame(
        qa_rows
    )

    # --------------------------------------------------------
    # Hard assertions
    # --------------------------------------------------------

    if not receipt_qa[
        "core_count_after"
    ].eq(
        6
    ).all():

        bad = receipt_qa.loc[
            ~receipt_qa[
                "core_count_after"
            ].eq(
                6
            )
        ]

        raise RuntimeError(
            "Some validated receipts are not core6 after merge.\n"
            + bad.to_string(
                index=False
            )
        )

    hard_fail = receipt_qa.loc[
        receipt_qa[
            "balance_status"
        ].eq(
            "HARD_FAIL"
        )
    ]

    if not hard_fail.empty:

        raise RuntimeError(
            "Balance hard-fail found after merge. "
            "Output parquet NOT written.\n"
            + hard_fail.head(
                30
            ).to_string(
                index=False
            )
        )

    # Every validated receipt should become core6.
    newly_core6 = (
        after_core6
        - before_core6
    )

    print(
        "\n[Core6 before / after]"
    )

    print(
        f"before: {before_core6:,}"
    )

    print(
        f"after : {after_core6:,}"
    )

    print(
        f"delta : {newly_core6:+,}"
    )

    print(
        "\n[Core-count distribution BEFORE]"
    )

    print(
        before_distribution.to_string()
    )

    print(
        "\n[Core-count distribution AFTER]"
    )

    print(
        after_distribution.to_string()
    )

    print(
        "\n[Validated receipt balance QA]"
    )

    print(
        receipt_qa[
            "balance_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[YTD provenance among merged receipts]"
    )

    print(
        validated[
            "income_ytd_final_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # --------------------------------------------------------
    # Ensure REVIEW receipts were not modified into source values
    # via this merge operation.
    #
    # We only modify allowed_receipts, so this is guaranteed by
    # construction, but print the count explicitly.
    # --------------------------------------------------------

    changed_receipts = set(
        merge_audit.loc[
            merge_audit[
                "merge_action"
            ].eq(
                "FILL_NULL"
            ),
            "rcept_no",
        ]
    )

    accidental_review = (
        changed_receipts
        & review_receipts
    )

    print(
        "\n[Review receipts accidentally changed]"
    )

    print(
        len(
            accidental_review
        )
    )

    if accidental_review:
        raise RuntimeError(
            "REVIEW receipt was accidentally modified."
        )

    # --------------------------------------------------------
    # Write audit + recovered parquet only after all assertions
    # --------------------------------------------------------

    merge_audit.to_csv(
        MERGE_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    receipt_qa.to_csv(
        RECEIPT_QA,
        index=False,
        encoding="utf-8-sig",
    )

    # Drop temporary QA columns from final artifact.
    recovered_out = recovered.drop(
        columns=[
            "_core_count_before",
            "_core_count_after",
        ],
        errors="ignore",
    )

    recovered_out.to_parquet(
        WIDE_OUTPUT,
        index=False,
    )

    print(
        "\n[PASS]"
    )

    print(
        f"Validated receipts merged: "
        f"{len(validated):,}"
    )

    print(
        f"Changed receipts: "
        f"{len(changed_receipts):,}"
    )

    print(
        f"Output rows: "
        f"{len(recovered_out):,}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Recovered wide : "
        f"{WIDE_OUTPUT}"
    )

    print(
        f"- Merge audit    : "
        f"{MERGE_AUDIT}"
    )

    print(
        f"- Receipt QA     : "
        f"{RECEIPT_QA}"
    )

    print(
        "\n중요:"
        "\n- 원본 dart_noncorrected_pit_receipt_wide_final.parquet은 수정하지 않았음"
        "\n- PASS_ALL_INCOME_YTD receipt만 병합"
        "\n- REVIEW 35건은 그대로 unresolved"
        "\n- 기존 non-null 값과 충돌 시 output write 전에 중단"
        "\n- 다음 단계에서 recovered wide 전체 QA 후 canonical final로 승격"
    )


if __name__ == "__main__":
    main()
