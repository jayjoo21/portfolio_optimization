from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7G. Selector V2 Attrition & Ambiguity Diagnostic
#
# 목적
# ------------------------------------------------------------
# H6B7F가 지나치게 보수적으로 10개 core6만 남긴 이유를 계량화한다.
#
# 1) receipt × metric별 후보 탈락 단계 확인
# 2) BASIS_UNRESOLVED의 실제 원인 분해
# 3) MULTIPLE_VALUES를
#       - near-equivalent / rounding duplicate
#       - genuinely conflicting
#    으로 분리
# 4) 과거 H6B2 strict CFS 6/6 receipt가 현재 어디서 탈락했는지 비교
#
# IMPORTANT
# ------------------------------------------------------------
# - ZIP 재파싱 없음
# - final selection 없음
# - 규칙 변경 전 진단만 수행
#
# 실행:
# python scripts\05a5h6b7g_dart_selector_v2_attrition_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CANDIDATES = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_candidate_audit.parquet"
)

METRIC_SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_metric_summary.csv"
)

OLD_H6B2_COVERAGE = (
    INTERIM
    / "dart_noncorrected_nodata_document_cfs_receipt_coverage.parquet"
)

OUT_METRIC_ATTRITION = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_metric_attrition.csv"
)

OUT_BASIS_DIAG = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_basis_attrition.csv"
)

OUT_AMBIGUITY = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_ambiguity_clusters.csv"
)

OUT_OLD101 = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_old101_comparison.csv"
)


CORE = [
    "assets",
    "liabilities",
    "equity",
    "revenue",
    "operating_income",
    "net_income",
]


def receipt_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def bool_col(group: pd.DataFrame, col: str) -> pd.Series:
    if col not in group.columns:
        return pd.Series(False, index=group.index)

    return group[col].fillna(False).astype(bool)


def stage_reason(group: pd.DataFrame) -> str:

    if group.empty:
        return "NO_ROWS"

    numeric = pd.to_numeric(
        group.get("selector_value_krw"),
        errors="coerce",
    ).notna()

    if not numeric.any():
        return "NO_NUMERIC_VALUE"

    g = group.loc[numeric].copy()

    if not bool_col(g, "selector_period_ok").any():
        return "NO_STRONG_PERIOD"

    g = g.loc[
        bool_col(g, "selector_period_ok")
    ]

    if g.empty:
        return "NO_STRONG_PERIOD"

    if (~bool_col(g, "selector_non_main_context")).sum() == 0:
        return "ALL_BLOCKED_NON_MAIN_CONTEXT"

    g = g.loc[
        ~bool_col(g, "selector_non_main_context")
    ]

    if g.empty:
        return "ALL_BLOCKED_NON_MAIN_CONTEXT"

    if not bool_col(g, "selector_main_statement").any():
        return "NO_MAIN_STATEMENT_TITLE"

    g = g.loc[
        bool_col(g, "selector_main_statement")
    ]

    if g.empty:
        return "NO_MAIN_STATEMENT_TITLE"

    if not bool_col(g, "selector_row_eligible").any():
        return "ROW_ELIGIBILITY_OTHER"

    g = g.loc[
        bool_col(g, "selector_row_eligible")
    ]

    if g.empty:
        return "ROW_ELIGIBILITY_OTHER"

    if not bool_col(g, "selector_basis_eligible").any():

        policies = (
            g.get(
                "selector_receipt_basis_policy",
                pd.Series(dtype="object"),
            )
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        policy = (
            policies[0]
            if len(policies) == 1
            else "|".join(policies)
        )

        if policy == "BASIS_UNRESOLVED":
            return "BASIS_UNRESOLVED"

        if policy == "OFS_PRESENT_BUT_NO_FALLBACK_PROOF":
            return "OFS_NO_FALLBACK_PROOF"

        if policy == "CFS_ONLY":
            return "CFS_POLICY_BUT_NO_ELIGIBLE_CFS_ROW"

        return "BASIS_NOT_ELIGIBLE"

    g = g.loc[
        bool_col(g, "selector_basis_eligible")
    ]

    if g.empty:
        return "BASIS_NOT_ELIGIBLE"

    if not bool_col(g, "selector_auto_pool").any():
        return "NOT_IN_AUTO_POOL"

    return "AUTO_POOL_PRESENT"


def near_equal(a: float, b: float) -> bool:
    """
    회계표 단위 차이에 따른 반올림 중복 탐지용 진단 기준.

    - 상대 오차 <= 1e-5
      OR
    - 절대 차이 <= 1,000,000 KRW

    자동채택 기준이 아니라 "near-equivalent" 진단용일 뿐.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return False

    diff = abs(a - b)

    denom = max(
        abs(a),
        abs(b),
        1.0,
    )

    return (
        diff <= 1_000_000
        or diff / denom <= 1e-5
    )


def cluster_values(values: list[float]) -> list[list[float]]:

    values = sorted(
        set(
            float(v)
            for v in values
            if pd.notna(v)
        )
    )

    clusters = []

    for value in values:

        placed = False

        for cluster in clusters:
            if any(
                near_equal(
                    value,
                    member,
                )
                for member in cluster
            ):
                cluster.append(
                    value
                )
                placed = True
                break

        if not placed:
            clusters.append(
                [
                    value
                ]
            )

    return clusters


def main():

    if not CANDIDATES.exists():
        raise FileNotFoundError(
            CANDIDATES
        )

    if not METRIC_SUMMARY.exists():
        raise FileNotFoundError(
            METRIC_SUMMARY
        )

    df = pd.read_parquet(
        CANDIDATES
    )

    df["rcept_no"] = receipt_string(
        df["rcept_no"]
    )

    metric_summary = pd.read_csv(
        METRIC_SUMMARY,
        dtype={
            "rcept_no":
            str,
        },
        low_memory=False,
    )

    metric_summary[
        "rcept_no"
    ] = receipt_string(
        metric_summary[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7G SELECTOR V2 ATTRITION & AMBIGUITY DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCandidate rows: "
        f"{len(df):,}"
    )

    # ========================================================
    # A. Row-level waterfall
    # ========================================================

    core = df.loc[
        df[
            "account_family"
        ].isin(
            CORE
        )
    ].copy()

    numeric_mask = pd.to_numeric(
        core[
            "selector_value_krw"
        ],
        errors="coerce",
    ).notna()

    print(
        "\n[Row attrition waterfall]"
    )

    waterfall = [
        (
            "core rows",
            len(
                core
            ),
        ),
        (
            "numeric value",
            int(
                numeric_mask.sum()
            ),
        ),
        (
            "period_ok",
            int(
                (
                    numeric_mask
                    & bool_col(
                        core,
                        "selector_period_ok",
                    )
                ).sum()
            ),
        ),
        (
            "period_ok + not non-main",
            int(
                (
                    numeric_mask
                    & bool_col(
                        core,
                        "selector_period_ok",
                    )
                    & ~bool_col(
                        core,
                        "selector_non_main_context",
                    )
                ).sum()
            ),
        ),
        (
            "+ main statement title",
            int(
                (
                    numeric_mask
                    & bool_col(
                        core,
                        "selector_period_ok",
                    )
                    & ~bool_col(
                        core,
                        "selector_non_main_context",
                    )
                    & bool_col(
                        core,
                        "selector_main_statement",
                    )
                ).sum()
            ),
        ),
        (
            "row eligible",
            int(
                bool_col(
                    core,
                    "selector_row_eligible",
                ).sum()
            ),
        ),
        (
            "basis eligible",
            int(
                (
                    bool_col(
                        core,
                        "selector_row_eligible",
                    )
                    & bool_col(
                        core,
                        "selector_basis_eligible",
                    )
                ).sum()
            ),
        ),
        (
            "auto pool",
            int(
                bool_col(
                    core,
                    "selector_auto_pool",
                ).sum()
            ),
        ),
    ]

    for label, count in waterfall:
        print(
            f"{label:<32} "
            f"{count:>8,}"
        )

    # ========================================================
    # B. Receipt × metric attrition
    # ========================================================

    receipts = (
        df[
            "rcept_no"
        ]
        .drop_duplicates()
        .tolist()
    )

    grid = pd.MultiIndex.from_product(
        [
            receipts,
            CORE,
        ],
        names=[
            "rcept_no",
            "account_family",
        ],
    ).to_frame(
        index=False
    )

    records = []

    grouped = {
        key:
        group
        for key, group
        in core.groupby(
            [
                "rcept_no",
                "account_family",
            ],
            sort=False,
        )
    }

    for row in grid.itertuples(
        index=False
    ):

        key = (
            str(
                row.rcept_no
            ),
            row.account_family,
        )

        group = grouped.get(
            key,
            core.iloc[
                0:0
            ],
        )

        records.append(
            {
                "rcept_no":
                key[
                    0
                ],

                "account_family":
                key[
                    1
                ],

                "attrition_reason":
                stage_reason(
                    group
                ),

                "row_count":
                len(
                    group
                ),

                "numeric_row_count":
                int(
                    pd.to_numeric(
                        group.get(
                            "selector_value_krw",
                            pd.Series(
                                index=group.index,
                                dtype=float,
                            ),
                        ),
                        errors="coerce",
                    )
                    .notna()
                    .sum()
                ),

                "period_ok_count":
                int(
                    bool_col(
                        group,
                        "selector_period_ok",
                    )
                    .sum()
                ),

                "main_statement_count":
                int(
                    bool_col(
                        group,
                        "selector_main_statement",
                    )
                    .sum()
                ),

                "row_eligible_count":
                int(
                    bool_col(
                        group,
                        "selector_row_eligible",
                    )
                    .sum()
                ),

                "basis_eligible_count":
                int(
                    bool_col(
                        group,
                        "selector_basis_eligible",
                    )
                    .sum()
                ),

                "auto_pool_count":
                int(
                    bool_col(
                        group,
                        "selector_auto_pool",
                    )
                    .sum()
                ),
            }
        )

    attrition = pd.DataFrame(
        records
    )

    attrition.to_csv(
        OUT_METRIC_ATTRITION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Receipt × metric attrition reason]"
    )

    print(
        attrition[
            "attrition_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Attrition by metric]"
    )

    print(
        pd.crosstab(
            attrition[
                "account_family"
            ],
            attrition[
                "attrition_reason"
            ],
        )
        .to_string()
    )

    # ========================================================
    # C. Basis unresolved diagnostic
    # ========================================================

    receipt_basis = (
        df[
            [
                "rcept_no",
                "selector_receipt_basis_policy",
                "selector_receipt_cfs_exists",
                "selector_receipt_no_cfs_affirmed",
                "selector_receipt_ofs_exists",
            ]
        ]
        .drop_duplicates(
            "rcept_no"
        )
        .copy()
    )

    basis_records = []

    for rcept_no, group in (
        df.groupby(
            "rcept_no",
            sort=False,
        )
    ):

        period_core = group.loc[
            group[
                "account_family"
            ].isin(
                CORE
            )
            & bool_col(
                group,
                "selector_period_ok",
            )
            & pd.to_numeric(
                group[
                    "selector_value_krw"
                ],
                errors="coerce",
            )
            .notna()
        ].copy()

        all_basis_counts = (
            group[
                "basis_v2"
            ]
            .fillna(
                "NA"
            )
            .astype(str)
            .value_counts()
            .to_dict()
        )

        table_basis_counts = (
            group[
                "selector_table_basis"
            ]
            .fillna(
                "NA"
            )
            .astype(str)
            .value_counts()
            .to_dict()
        )

        period_basis_counts = (
            period_core[
                "selector_table_basis"
            ]
            .fillna(
                "NA"
            )
            .astype(str)
            .value_counts()
            .to_dict()
        )

        policy_values = (
            group[
                "selector_receipt_basis_policy"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        basis_records.append(
            {
                "rcept_no":
                rcept_no,

                "policy":
                (
                    policy_values[
                        0
                    ]
                    if policy_values
                    else ""
                ),

                "row_basis_CFS":
                all_basis_counts.get(
                    "CFS",
                    0,
                ),

                "row_basis_OFS":
                all_basis_counts.get(
                    "OFS",
                    0,
                ),

                "row_basis_OFS_ONLY_NO_CFS":
                all_basis_counts.get(
                    "OFS_ONLY_NO_CFS",
                    0,
                ),

                "row_basis_UNKNOWN":
                all_basis_counts.get(
                    "UNKNOWN",
                    0,
                ),

                "table_basis_CFS":
                table_basis_counts.get(
                    "CFS",
                    0,
                ),

                "table_basis_OFS":
                table_basis_counts.get(
                    "OFS",
                    0,
                ),

                "table_basis_OFS_ONLY_NO_CFS":
                table_basis_counts.get(
                    "OFS_ONLY_NO_CFS",
                    0,
                ),

                "table_basis_UNKNOWN":
                table_basis_counts.get(
                    "UNKNOWN",
                    0,
                ),

                "period_core_table_CFS":
                period_basis_counts.get(
                    "CFS",
                    0,
                ),

                "period_core_table_OFS":
                (
                    period_basis_counts.get(
                        "OFS",
                        0,
                    )
                    + period_basis_counts.get(
                        "OFS_ONLY_NO_CFS",
                        0,
                    )
                ),

                "period_core_table_UNKNOWN":
                period_basis_counts.get(
                    "UNKNOWN",
                    0,
                ),
            }
        )

    basis_diag = pd.DataFrame(
        basis_records
    )

    basis_diag.to_csv(
        OUT_BASIS_DIAG,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[BASIS_UNRESOLVED receipt diagnostic]"
    )

    unresolved = basis_diag.loc[
        basis_diag[
            "policy"
        ].eq(
            "BASIS_UNRESOLVED"
        )
    ]

    print(
        f"receipts: "
        f"{len(unresolved):,}"
    )

    if not unresolved.empty:
        print(
            "\nStrong period rows in unresolved receipts:"
        )

        print(
            unresolved[
                [
                    "period_core_table_CFS",
                    "period_core_table_OFS",
                    "period_core_table_UNKNOWN",
                ]
            ]
            .value_counts()
            .head(
                20
            )
            .to_string()
        )

        print(
            "\nNo-CFS evidence inside unresolved receipts:"
        )

        print(
            unresolved[
                "row_basis_OFS_ONLY_NO_CFS"
            ]
            .gt(
                0
            )
            .value_counts()
            .to_string()
        )

    # ========================================================
    # D. Multiple-value ambiguity clustering
    # ========================================================

    ambiguity_records = []

    multi = metric_summary.loc[
        metric_summary[
            "metric_candidate_status"
        ].eq(
            "MULTIPLE_VALUES"
        )
    ].copy()

    auto = df.loc[
        bool_col(
            df,
            "selector_auto_pool",
        )
    ].copy()

    for row in multi.itertuples(
        index=False
    ):

        group = auto.loc[
            auto[
                "rcept_no"
            ].eq(
                str(
                    row.rcept_no
                )
            )
            & auto[
                "account_family"
            ].eq(
                row.account_family
            )
        ].copy()

        values = (
            pd.to_numeric(
                group[
                    "selector_value_krw"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )

        clusters = cluster_values(
            values
        )

        near_duplicate = (
            len(
                values
            ) > 1
            and len(
                clusters
            ) == 1
        )

        ambiguity_records.append(
            {
                "rcept_no":
                str(
                    row.rcept_no
                ),

                "account_family":
                row.account_family,

                "raw_unique_value_count":
                len(
                    values
                ),

                "tolerance_cluster_count":
                len(
                    clusters
                ),

                "ambiguity_class":
                (
                    "ROUNDING_NEAR_EQUIVALENT"
                    if near_duplicate
                    else "GENUINE_MULTIPLE_CLUSTERS"
                ),

                "raw_values":
                "|".join(
                    f"{v:.12g}"
                    for v in sorted(
                        values
                    )
                ),

                "clusters":
                " || ".join(
                    ",".join(
                        f"{v:.12g}"
                        for v in cluster
                    )
                    for cluster in clusters
                ),
            }
        )

    ambiguity = pd.DataFrame(
        ambiguity_records
    )

    ambiguity.to_csv(
        OUT_AMBIGUITY,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Multiple-value ambiguity class]"
    )

    if ambiguity.empty:
        print(
            "None"
        )
    else:
        print(
            ambiguity[
                "ambiguity_class"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\n[Near-equivalent sample]"
        )

        sample = ambiguity.loc[
            ambiguity[
                "ambiguity_class"
            ].eq(
                "ROUNDING_NEAR_EQUIVALENT"
            )
        ]

        if sample.empty:
            print(
                "None"
            )
        else:
            print(
                sample.head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    # ========================================================
    # E. Compare with old H6B2 complete6 receipts if available
    # ========================================================

    if OLD_H6B2_COVERAGE.exists():

        old = pd.read_parquet(
            OLD_H6B2_COVERAGE
        )

        if "rcept_no" in old.columns:
            old[
                "rcept_no"
            ] = receipt_string(
                old[
                    "rcept_no"
                ]
            )

            complete_col = None

            for col in [
                "core_count",
                "complete_core_count",
                "non_null_core_count",
                "selected_core_count",
            ]:
                if col in old.columns:
                    complete_col = col
                    break

            if complete_col is not None:
                old101 = old.loc[
                    pd.to_numeric(
                        old[
                            complete_col
                        ],
                        errors="coerce",
                    )
                    .eq(
                        6
                    ),
                    [
                        "rcept_no",
                        complete_col,
                    ],
                ].copy()

                compare = (
                    old101.merge(
                        attrition,
                        on="rcept_no",
                        how="left",
                    )
                )

                compare.to_csv(
                    OUT_OLD101,
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    "\n[Old H6B2 complete6 receipts]"
                )

                print(
                    f"receipts: "
                    f"{old101['rcept_no'].nunique():,}"
                )

                print(
                    "\nAttrition reason among their 6 metrics:"
                )

                print(
                    compare[
                        "attrition_reason"
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .to_string()
                )

            else:
                print(
                    "\n[Old H6B2 comparison]"
                )

                print(
                    "Coverage file exists, but no recognized core-count column."
                )

        else:
            print(
                "\n[Old H6B2 comparison]"
            )

            print(
                "Coverage file exists, but rcept_no column missing."
            )

    else:
        print(
            "\n[Old H6B2 comparison]"
        )

        print(
            "Old H6B2 coverage parquet not found; skipped."
        )

    print(
        "\nOutputs:"
    )

    print(
        f"- Metric attrition : "
        f"{OUT_METRIC_ATTRITION}"
    )

    print(
        f"- Basis diagnostic : "
        f"{OUT_BASIS_DIAG}"
    )

    print(
        f"- Ambiguity audit  : "
        f"{OUT_AMBIGUITY}"
    )

    if OUT_OLD101.exists():
        print(
            f"- Old101 compare   : "
            f"{OUT_OLD101}"
        )

    print(
        "\n해석 원칙:"
        "\n- H6B7F의 10 core6는 최종 coverage가 아님"
        "\n- 어디에서 과도하게 잘렸는지 먼저 확인"
        "\n- near-equivalent는 자동 merge 전 진단만 수행"
        "\n- basis propagation 규칙 수정은 이 결과를 본 뒤 결정"
    )


if __name__ == "__main__":
    main()
