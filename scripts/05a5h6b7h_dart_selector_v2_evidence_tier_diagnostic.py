from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7H. Selector V2 Evidence-Tier Diagnostic
#
# 목적
# ------------------------------------------------------------
# H6B7G에서 확인된 두 병목:
#
#   1) main statement title hard gate
#   2) basis propagation / receipt policy
#
# 를 "완화해서 바로 채택"하지 않고, 여러 shadow policy로
# coverage가 어떻게 변하는지 비교한다.
#
# ZIP 재파싱 없음.
# production 값 변경 없음.
#
# 비교 정책:
#
# STRICT_CURRENT
#   period + non-main guard + explicit main-statement title
#
# TABLE_BASIS_SUPPORTED
#   period + non-main guard +
#   explicit table basis(CFS/OFS)가 있으면
#   main-statement title이 heading에 없어도 후보 유지
#
# C4_STATEMENT_SUPPORTED
#   period + non-main guard +
#   main-statement title OR 기존 C4 statement_score > 0
#
# UPPER_BOUND
#   period + non-main guard만 적용
#   (자동채택용 아님. 최대 회복 가능량 진단)
#
# 각 정책별로:
#   - receipt basis policy
#   - metric coverage
#   - raw unique values
#   - rounding-near-equivalent cluster count
#   - core6 cluster-unique receipt 수
# 를 비교한다.
#
# 실행:
# python scripts\05a5h6b7h_dart_selector_v2_evidence_tier_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_candidate_audit.parquet"
)

OUT_ROW = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_rows.parquet"
)

OUT_METRIC = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_metric_summary.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_receipt_summary.csv"
)

OUT_SCORE_SAMPLE = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_statement_score_samples.csv"
)


CORE = [
    "assets",
    "liabilities",
    "equity",
    "revenue",
    "operating_income",
    "net_income",
]

POLICIES = [
    "STRICT_CURRENT",
    "TABLE_BASIS_SUPPORTED",
    "C4_STATEMENT_SUPPORTED",
    "UPPER_BOUND",
]


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(bool)
    )


def near_equal(a: float, b: float) -> bool:
    """
    진단용 near-equivalence 기준.
    production merge 기준 아님.
    """
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
        or diff / denom <= 1e-5
    )


def cluster_values(values: list[float]) -> list[list[float]]:

    vals = sorted(
        set(
            float(v)
            for v in values
            if pd.notna(v)
        )
    )

    clusters: list[list[float]] = []

    for value in vals:

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


def basis_family(value: str) -> str:

    if value == "CFS":
        return "CFS"

    if value in {
        "OFS",
        "OFS_ONLY_NO_CFS",
    }:
        return "OFS"

    return value


def policy_row_mask(
    df: pd.DataFrame,
    policy: str,
) -> pd.Series:

    base = (
        df[
            "account_family"
        ].isin(
            CORE
        )
        & pd.to_numeric(
            df[
                "selector_value_krw"
            ],
            errors="coerce",
        ).notna()
        & as_bool(
            df[
                "selector_period_ok"
            ]
        )
        & ~as_bool(
            df[
                "selector_non_main_context"
            ]
        )
    )

    if policy == "STRICT_CURRENT":

        return (
            base
            & as_bool(
                df[
                    "selector_main_statement"
                ]
            )
        )

    if policy == "TABLE_BASIS_SUPPORTED":

        explicit_basis = (
            df[
                "selector_table_basis"
            ]
            .astype(str)
            .isin(
                {
                    "CFS",
                    "OFS",
                    "OFS_ONLY_NO_CFS",
                }
            )
        )

        return (
            base
            & (
                as_bool(
                    df[
                        "selector_main_statement"
                    ]
                )
                | explicit_basis
            )
        )

    if policy == "C4_STATEMENT_SUPPORTED":

        if "statement_score" in df.columns:
            statement_score_source = df["statement_score"]
        else:
            statement_score_source = pd.Series(
                0.0,
                index=df.index,
                dtype="float64",
            )

        statement_score = pd.to_numeric(
            statement_score_source,
            errors="coerce",
        ).fillna(
            0
        )

        return (
            base
            & (
                as_bool(
                    df[
                        "selector_main_statement"
                    ]
                )
                | statement_score.gt(
                    0
                )
            )
        )

    if policy == "UPPER_BOUND":
        return base

    raise ValueError(
        policy
    )


def derive_receipt_basis_policy(
    df: pd.DataFrame,
    row_mask: pd.Series,
) -> pd.DataFrame:

    records = []

    work = df.loc[
        row_mask
    ].copy()

    for rcept_no, group in (
        df.groupby(
            "rcept_no",
            sort=False,
        )
    ):

        eligible = work.loc[
            work[
                "rcept_no"
            ].eq(
                rcept_no
            )
        ]

        cfs_exists = (
            eligible[
                "selector_table_basis"
            ]
            .astype(str)
            .eq(
                "CFS"
            )
            .any()
        )

        no_cfs_affirmed = (
            group[
                "basis_v2"
            ]
            .astype(str)
            .eq(
                "OFS_ONLY_NO_CFS"
            )
            .any()
            or group[
                "selector_table_basis"
            ]
            .astype(str)
            .eq(
                "OFS_ONLY_NO_CFS"
            )
            .any()
        )

        ofs_exists = (
            eligible[
                "selector_table_basis"
            ]
            .astype(str)
            .isin(
                {
                    "OFS",
                    "OFS_ONLY_NO_CFS",
                }
            )
            .any()
        )

        if cfs_exists:
            receipt_policy = "CFS_ONLY"

        elif (
            no_cfs_affirmed
            and ofs_exists
        ):
            receipt_policy = (
                "OFS_FALLBACK_EXPLICIT_NO_CFS"
            )

        elif ofs_exists:
            receipt_policy = (
                "OFS_PRESENT_BUT_NO_FALLBACK_PROOF"
            )

        else:
            receipt_policy = (
                "BASIS_UNRESOLVED"
            )

        records.append(
            {
                "rcept_no":
                rcept_no,

                "receipt_basis_policy":
                receipt_policy,

                "receipt_cfs_exists":
                cfs_exists,

                "receipt_no_cfs_affirmed":
                no_cfs_affirmed,

                "receipt_ofs_exists":
                ofs_exists,
            }
        )

    return pd.DataFrame(
        records
    )


def basis_allowed(
    rows: pd.DataFrame,
    policy_map: pd.DataFrame,
) -> pd.Series:

    policy_lookup = (
        policy_map.set_index(
            "rcept_no"
        )[
            "receipt_basis_policy"
        ]
    )

    receipt_policy = (
        rows[
            "rcept_no"
        ]
        .map(
            policy_lookup
        )
    )

    table_basis = (
        rows[
            "selector_table_basis"
        ]
        .astype(str)
    )

    cfs_ok = (
        receipt_policy.eq(
            "CFS_ONLY"
        )
        & table_basis.eq(
            "CFS"
        )
    )

    ofs_ok = (
        receipt_policy.eq(
            "OFS_FALLBACK_EXPLICIT_NO_CFS"
        )
        & table_basis.isin(
            {
                "OFS",
                "OFS_ONLY_NO_CFS",
            }
        )
    )

    return (
        cfs_ok
        | ofs_ok
    )


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
        "05A5-H6B7H SELECTOR V2 EVIDENCE-TIER DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Statement-score diagnostic
    # --------------------------------------------------------

    if "statement_score" in df.columns:
        statement_score_source = df["statement_score"]
    else:
        statement_score_source = pd.Series(
            0.0,
            index=df.index,
            dtype="float64",
        )

    df[
        "statement_score_num"
    ] = pd.to_numeric(
        statement_score_source,
        errors="coerce",
    ).fillna(
        0
    )

    base = (
        df[
            "account_family"
        ].isin(
            CORE
        )
        & pd.to_numeric(
            df[
                "selector_value_krw"
            ],
            errors="coerce",
        ).notna()
        & as_bool(
            df[
                "selector_period_ok"
            ]
        )
        & ~as_bool(
            df[
                "selector_non_main_context"
            ]
        )
    )

    score_diag = (
        df.loc[
            base
        ]
        .groupby(
            [
                "selector_main_statement",
                "selector_table_basis",
            ],
            dropna=False,
        )[
            "statement_score_num"
        ]
        .agg(
            row_count="size",
            min_score="min",
            p25=lambda s:
            float(
                s.quantile(
                    0.25
                )
            ),
            median_score="median",
            p75=lambda s:
            float(
                s.quantile(
                    0.75
                )
            ),
            max_score="max",
            positive_score_count=lambda s:
            int(
                s.gt(
                    0
                ).sum()
            ),
        )
        .reset_index()
    )

    print(
        "\n[Statement score distribution by main-title / table-basis]"
    )

    print(
        score_diag.to_string(
            index=False
        )
    )

    # Save samples specifically from no-main-title rows.
    no_title = df.loc[
        base
        & ~as_bool(
            df[
                "selector_main_statement"
            ]
        )
    ].copy()

    sample_cols = [
        c
        for c in [
            "stock_code",
            "period_key",
            "rcept_no",
            "account_family",
            "table_index",
            "row_index",
            "basis_v2",
            "selector_table_basis",
            "statement_score_num",
            "statement_score_reason",
            "period_final_status",
            "period_v2_selected_column",
            "selector_value_krw",
            "heading_context",
        ]
        if c in no_title.columns
    ]

    no_title_sample = (
        no_title[
            sample_cols
        ]
        .sort_values(
            [
                "statement_score_num",
            ],
            ascending=False,
        )
        .head(
            500
        )
    )

    no_title_sample.to_csv(
        OUT_SCORE_SAMPLE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Policy comparison
    # --------------------------------------------------------

    row_outputs = []
    metric_outputs = []
    receipt_outputs = []

    receipts = (
        df[
            "rcept_no"
        ]
        .drop_duplicates()
        .astype(str)
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

    for policy in POLICIES:

        mask = policy_row_mask(
            df,
            policy,
        )

        receipt_basis = (
            derive_receipt_basis_policy(
                df,
                mask,
            )
        )

        allowed = basis_allowed(
            df,
            receipt_basis,
        )

        final_mask = (
            mask
            & allowed
        )

        policy_rows = df.loc[
            final_mask
        ].copy()

        policy_rows[
            "evidence_policy"
        ] = policy

        row_outputs.append(
            policy_rows
        )

        # --------------------------------------------
        # Receipt × metric clusters
        # --------------------------------------------

        records = []

        grouped = {
            key:
            group
            for key, group
            in policy_rows.groupby(
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
                key
            )

            if group is None:
                values = []
            else:
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

            raw_unique = len(
                set(
                    float(x)
                    for x in values
                )
            )

            cluster_count = len(
                clusters
            )

            if cluster_count == 0:
                status = (
                    "NO_CANDIDATE"
                )

            elif cluster_count == 1:
                status = (
                    "ONE_VALUE_CLUSTER"
                )

            else:
                status = (
                    "MULTIPLE_VALUE_CLUSTERS"
                )

            records.append(
                {
                    "evidence_policy":
                    policy,

                    "rcept_no":
                    key[
                        0
                    ],

                    "account_family":
                    key[
                        1
                    ],

                    "candidate_row_count":
                    0
                    if group is None
                    else len(
                        group
                    ),

                    "raw_unique_value_count":
                    raw_unique,

                    "tolerance_cluster_count":
                    cluster_count,

                    "metric_cluster_status":
                    status,

                    "rounding_only_ambiguity":
                    (
                        raw_unique > 1
                        and cluster_count == 1
                    ),
                }
            )

        metric = pd.DataFrame(
            records
        )

        metric_outputs.append(
            metric
        )

        # --------------------------------------------
        # Receipt-level coverage
        # --------------------------------------------

        receipt_cov = (
            metric.groupby(
                [
                    "evidence_policy",
                    "rcept_no",
                ],
                as_index=False,
            )
            .agg(
                metric_with_candidate=(
                    "tolerance_cluster_count",
                    lambda s:
                    int(
                        s.gt(
                            0
                        ).sum()
                    ),
                ),

                metric_single_cluster=(
                    "tolerance_cluster_count",
                    lambda s:
                    int(
                        s.eq(
                            1
                        ).sum()
                    ),
                ),

                metric_multiple_cluster=(
                    "tolerance_cluster_count",
                    lambda s:
                    int(
                        s.gt(
                            1
                        ).sum()
                    ),
                ),

                rounding_only_metric_count=(
                    "rounding_only_ambiguity",
                    "sum",
                ),
            )
        )

        receipt_cov[
            "core6_single_cluster"
        ] = (
            receipt_cov[
                "metric_single_cluster"
            ].eq(
                6
            )
        )

        receipt_cov = (
            receipt_cov.merge(
                receipt_basis,
                on="rcept_no",
                how="left",
                validate="one_to_one",
            )
        )

        receipt_outputs.append(
            receipt_cov
        )

    all_rows = pd.concat(
        row_outputs,
        ignore_index=True,
    )

    all_metric = pd.concat(
        metric_outputs,
        ignore_index=True,
    )

    all_receipt = pd.concat(
        receipt_outputs,
        ignore_index=True,
    )

    all_rows.to_parquet(
        OUT_ROW,
        index=False,
    )

    all_metric.to_csv(
        OUT_METRIC,
        index=False,
        encoding="utf-8-sig",
    )

    all_receipt.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Policy row counts after basis policy]"
    )

    print(
        all_rows[
            "evidence_policy"
        ]
        .value_counts()
        .reindex(
            POLICIES
        )
        .to_string()
    )

    print(
        "\n[Receipt basis policy by evidence policy]"
    )

    print(
        pd.crosstab(
            all_receipt[
                "evidence_policy"
            ],
            all_receipt[
                "receipt_basis_policy"
            ],
        )
        .reindex(
            POLICIES
        )
        .to_string()
    )

    print(
        "\n[Metric cluster status by policy]"
    )

    print(
        pd.crosstab(
            all_metric[
                "evidence_policy"
            ],
            all_metric[
                "metric_cluster_status"
            ],
        )
        .reindex(
            POLICIES
        )
        .to_string()
    )

    print(
        "\n[Core6 single-cluster receipts by policy]"
    )

    core6 = (
        all_receipt.groupby(
            "evidence_policy"
        )[
            "core6_single_cluster"
        ]
        .sum()
        .reindex(
            POLICIES
        )
    )

    print(
        core6.to_string()
    )

    print(
        "\n[Receipts with all 6 metrics present, regardless ambiguity]"
    )

    all6_present = (
        all_receipt.assign(
            all6=lambda x:
            x[
                "metric_with_candidate"
            ].eq(
                6
            )
        )
        .groupby(
            "evidence_policy"
        )[
            "all6"
        ]
        .sum()
        .reindex(
            POLICIES
        )
    )

    print(
        all6_present.to_string()
    )

    print(
        "\n[Rounding-only ambiguity metric count by policy]"
    )

    rounding_counts = (
        all_receipt.groupby(
            "evidence_policy"
        )[
            "rounding_only_metric_count"
        ]
        .sum()
        .reindex(
            POLICIES
        )
    )

    print(
        rounding_counts.to_string()
    )

    print(
        "\n[No-main-title rows by statement-score sign]"
    )

    no_title_score = (
        no_title.assign(
            score_sign=np.select(
                [
                    no_title[
                        "statement_score_num"
                    ].gt(
                        0
                    ),

                    no_title[
                        "statement_score_num"
                    ].eq(
                        0
                    ),
                ],
                [
                    "POSITIVE",
                    "ZERO",
                ],
                default="NEGATIVE",
            )
        )[
            "score_sign"
        ]
        .value_counts()
    )

    print(
        no_title_score.to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Row policy audit : "
        f"{OUT_ROW}"
    )

    print(
        f"- Metric summary   : "
        f"{OUT_METRIC}"
    )

    print(
        f"- Receipt summary  : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Score samples    : "
        f"{OUT_SCORE_SAMPLE}"
    )

    print(
        "\n해석 원칙:"
        "\n- STRICT_CURRENT는 현재 H6B7F 기준"
        "\n- TABLE_BASIS_SUPPORTED는 explicit table basis를 title 대체증거로 시험"
        "\n- C4_STATEMENT_SUPPORTED는 기존 selector의 statement_score를 보조증거로 시험"
        "\n- UPPER_BOUND는 자동채택 후보가 아니라 회복 가능량 상한"
        "\n- near-equivalent cluster는 아직 production merge가 아님"
    )


if __name__ == "__main__":
    main()
