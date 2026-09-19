from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7I. Statement-Bundle Candidate Audit
#
# 목적
# ------------------------------------------------------------
# H6B7H에서 production 후보로 가장 유망했던
# TABLE_BASIS_SUPPORTED policy만 사용한다.
#
# metric별로 독립적으로 값을 고르지 않고,
#
#   Balance Sheet bundle:
#       assets + liabilities + equity
#
#   Income Statement bundle:
#       revenue + operating_income + net_income
#
# 를 "같은 table_index" 안에서 묶어 본다.
#
# 이렇게 해야:
# - 서로 다른 표의 값을 metric별로 섞는 위험을 줄이고
# - balance equation으로 total equity vs parent equity 등을 구분하고
# - 하나의 본표 안에서 core metric이 얼마나 완성되는지 확인할 수 있다.
#
# IMPORTANT
# ------------------------------------------------------------
# - final production selection 아님
# - ZIP 재파싱 없음
# - CFS/OFS mixing 없음 (H6B7H receipt policy를 그대로 사용)
# - near-equivalent clustering은 진단용
# - 금융업 revenue semantics는 이후 별도 검토 대상
#
# 실행:
# python scripts\05a5h6b7i_dart_statement_bundle_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_rows.parquet"
)

OUT_TABLE = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_table_summary.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_receipt_summary.csv"
)

OUT_BALANCE_COMBOS = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_balance_combos.csv"
)

OUT_AMBIGUOUS = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_ambiguous_samples.csv"
)


POLICY = "TABLE_BASIS_SUPPORTED"

BALANCE_METRICS = [
    "assets",
    "liabilities",
    "equity",
]

INCOME_METRICS = [
    "revenue",
    "operating_income",
    "net_income",
]


# ============================================================
# Numeric equivalence helpers
# ============================================================

def near_equal(a: float, b: float) -> bool:
    """
    진단용 near-equivalence.
    production merge rule 아님.
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
                [value]
            )

    return clusters


def cluster_repr(cluster: list[float]) -> float:
    """
    출력용 representative.
    선택값 확정용이 아님.
    """
    if not cluster:
        return np.nan

    return float(
        np.median(
            np.array(
                cluster,
                dtype=float,
            )
        )
    )


# ============================================================
# Balance QA
# ============================================================

def balance_pass(
    assets: float,
    liabilities: float,
    equity: float,
) -> bool:

    gap = (
        assets
        - liabilities
        - equity
    )

    abs_gap = abs(
        gap
    )

    denom = max(
        abs(assets),
        abs(liabilities)
        + abs(equity),
        1.0,
    )

    rel_gap = (
        abs_gap
        / denom
    )

    return (
        abs_gap <= 2_000_000
        or rel_gap <= 1e-6
    )


def best_balance_raw_combo(
    a_cluster: list[float],
    l_cluster: list[float],
    e_cluster: list[float],
):
    """
    cluster triple 안의 raw values 중 balance gap이 가장 작은 조합을 찾는다.
    """
    best = None

    # 보통 cluster는 1~2개 값이므로 충분히 작다.
    # 혹시 비정상적으로 많으면 앞 20개만 진단.
    for a, l, e in itertools.product(
        a_cluster[:20],
        l_cluster[:20],
        e_cluster[:20],
    ):

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

        record = (
            abs_gap,
            rel_gap,
            a,
            l,
            e,
            gap,
        )

        if (
            best is None
            or record[:2]
            < best[:2]
        ):
            best = record

    return best


# ============================================================
# Table helpers
# ============================================================

def metric_clusters(
    group: pd.DataFrame,
    metric: str,
):

    values = (
        pd.to_numeric(
            group.loc[
                group[
                    "account_family"
                ].eq(
                    metric
                ),
                "selector_value_krw",
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    return cluster_values(
        values
    )


def serialize_clusters(
    clusters: list[list[float]],
) -> str:

    return " || ".join(
        ",".join(
            f"{v:.12g}"
            for v in cluster
        )
        for cluster in clusters
    )


# ============================================================
# Main
# ============================================================

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
        "05A5-H6B7I STATEMENT-BUNDLE CANDIDATE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nAll policy rows: "
        f"{len(df):,}"
    )

    if "evidence_policy" not in df.columns:
        raise RuntimeError(
            "evidence_policy column missing"
        )

    work = df.loc[
        df[
            "evidence_policy"
        ].eq(
            POLICY
        )
    ].copy()

    print(
        f"{POLICY} rows: "
        f"{len(work):,}"
    )

    if work.empty:
        raise RuntimeError(
            f"No rows for policy {POLICY}"
        )

    # --------------------------------------------------------
    # Table-level bundle audit
    # --------------------------------------------------------

    table_records = []
    balance_combo_records = []

    group_cols = [
        "rcept_no",
        "table_index",
    ]

    for (
        rcept_no,
        table_index,
    ), group in work.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):

        table_basis_values = (
            group[
                "selector_table_basis"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        table_basis = (
            table_basis_values[0]
            if len(table_basis_values) == 1
            else "|".join(
                sorted(
                    table_basis_values
                )
            )
        )

        stock_values = (
            group.get(
                "stock_code",
                pd.Series(dtype="object"),
            )
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        stock_code = (
            stock_values[0]
            if stock_values
            else ""
        )

        period_values = (
            group.get(
                "period_key",
                pd.Series(dtype="object"),
            )
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        period_key = (
            period_values[0]
            if period_values
            else ""
        )

        # -----------------------------
        # Balance bundle
        # -----------------------------

        a_clusters = metric_clusters(
            group,
            "assets",
        )

        l_clusters = metric_clusters(
            group,
            "liabilities",
        )

        e_clusters = metric_clusters(
            group,
            "equity",
        )

        balance_metric_count = sum(
            len(x) > 0
            for x in [
                a_clusters,
                l_clusters,
                e_clusters,
            ]
        )

        coherent_cluster_combos = []

        if balance_metric_count == 3:

            for (
                ai,
                a_cluster,
            ), (
                li,
                l_cluster,
            ), (
                ei,
                e_cluster,
            ) in itertools.product(
                enumerate(
                    a_clusters
                ),
                enumerate(
                    l_clusters
                ),
                enumerate(
                    e_clusters
                ),
            ):

                best = (
                    best_balance_raw_combo(
                        a_cluster,
                        l_cluster,
                        e_cluster,
                    )
                )

                if best is None:
                    continue

                (
                    abs_gap,
                    rel_gap,
                    a,
                    l,
                    e,
                    gap,
                ) = best

                if balance_pass(
                    a,
                    l,
                    e,
                ):
                    coherent_cluster_combos.append(
                        {
                            "a_cluster":
                            ai,

                            "l_cluster":
                            li,

                            "e_cluster":
                            ei,

                            "assets":
                            a,

                            "liabilities":
                            l,

                            "equity":
                            e,

                            "gap":
                            gap,

                            "abs_gap":
                            abs_gap,

                            "rel_gap":
                            rel_gap,
                        }
                    )

                    balance_combo_records.append(
                        {
                            "rcept_no":
                            rcept_no,

                            "stock_code":
                            stock_code,

                            "period_key":
                            period_key,

                            "table_index":
                            table_index,

                            "selector_table_basis":
                            table_basis,

                            "a_cluster":
                            ai,

                            "l_cluster":
                            li,

                            "e_cluster":
                            ei,

                            "assets":
                            a,

                            "liabilities":
                            l,

                            "equity":
                            e,

                            "gap":
                            gap,

                            "abs_gap":
                            abs_gap,

                            "rel_gap":
                            rel_gap,
                        }
                    )

        if balance_metric_count < 3:
            balance_status = (
                "BALANCE_INCOMPLETE"
            )

        elif len(
            coherent_cluster_combos
        ) == 0:
            balance_status = (
                "BALANCE_NO_COHERENT_COMBO"
            )

        elif len(
            coherent_cluster_combos
        ) == 1:
            balance_status = (
                "BALANCE_UNIQUE_COHERENT_BUNDLE"
            )

        else:
            balance_status = (
                "BALANCE_MULTIPLE_COHERENT_BUNDLES"
            )

        # -----------------------------
        # Income bundle
        # -----------------------------

        r_clusters = metric_clusters(
            group,
            "revenue",
        )

        op_clusters = metric_clusters(
            group,
            "operating_income",
        )

        n_clusters = metric_clusters(
            group,
            "net_income",
        )

        income_metric_count = sum(
            len(x) > 0
            for x in [
                r_clusters,
                op_clusters,
                n_clusters,
            ]
        )

        income_cluster_counts = [
            len(
                r_clusters
            ),
            len(
                op_clusters
            ),
            len(
                n_clusters
            ),
        ]

        if income_metric_count < 3:
            income_status = (
                "INCOME_INCOMPLETE"
            )

        elif all(
            count == 1
            for count in income_cluster_counts
        ):
            income_status = (
                "INCOME_SINGLE_CLUSTER_BUNDLE"
            )

        else:
            income_status = (
                "INCOME_MULTIPLE_CLUSTER_BUNDLE"
            )

        # -----------------------------
        # Statement evidence summary
        # -----------------------------

        main_statement_count = int(
            group.get(
                "selector_main_statement",
                pd.Series(
                    False,
                    index=group.index,
                ),
            )
            .fillna(
                False
            )
            .astype(
                bool
            )
            .sum()
        )

        basis_v2_values = (
            group.get(
                "basis_v2",
                pd.Series(dtype="object"),
            )
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        table_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                stock_code,

                "period_key":
                period_key,

                "table_index":
                table_index,

                "selector_table_basis":
                table_basis,

                "row_count":
                len(
                    group
                ),

                "main_statement_row_count":
                main_statement_count,

                "basis_v2_CFS_rows":
                basis_v2_values.get(
                    "CFS",
                    0,
                ),

                "basis_v2_OFS_rows":
                basis_v2_values.get(
                    "OFS",
                    0,
                ),

                "basis_v2_OFS_ONLY_NO_CFS_rows":
                basis_v2_values.get(
                    "OFS_ONLY_NO_CFS",
                    0,
                ),

                "basis_v2_UNKNOWN_rows":
                basis_v2_values.get(
                    "UNKNOWN",
                    0,
                ),

                "balance_metric_count":
                balance_metric_count,

                "assets_cluster_count":
                len(
                    a_clusters
                ),

                "liabilities_cluster_count":
                len(
                    l_clusters
                ),

                "equity_cluster_count":
                len(
                    e_clusters
                ),

                "balance_coherent_cluster_combo_count":
                len(
                    coherent_cluster_combos
                ),

                "balance_bundle_status":
                balance_status,

                "assets_clusters":
                serialize_clusters(
                    a_clusters
                ),

                "liabilities_clusters":
                serialize_clusters(
                    l_clusters
                ),

                "equity_clusters":
                serialize_clusters(
                    e_clusters
                ),

                "income_metric_count":
                income_metric_count,

                "revenue_cluster_count":
                len(
                    r_clusters
                ),

                "operating_income_cluster_count":
                len(
                    op_clusters
                ),

                "net_income_cluster_count":
                len(
                    n_clusters
                ),

                "income_bundle_status":
                income_status,

                "revenue_clusters":
                serialize_clusters(
                    r_clusters
                ),

                "operating_income_clusters":
                serialize_clusters(
                    op_clusters
                ),

                "net_income_clusters":
                serialize_clusters(
                    n_clusters
                ),
            }
        )

    table_summary = pd.DataFrame(
        table_records
    )

    balance_combos = pd.DataFrame(
        balance_combo_records
    )

    table_summary.to_csv(
        OUT_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    balance_combos.to_csv(
        OUT_BALANCE_COMBOS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt-level bundle audit
    # --------------------------------------------------------

    receipt_records = []
    ambiguous_records = []

    for rcept_no, group in (
        table_summary.groupby(
            "rcept_no",
            sort=False,
        )
    ):

        balance_unique_tables = group.loc[
            group[
                "balance_bundle_status"
            ].eq(
                "BALANCE_UNIQUE_COHERENT_BUNDLE"
            )
        ].copy()

        balance_multi_tables = group.loc[
            group[
                "balance_bundle_status"
            ].eq(
                "BALANCE_MULTIPLE_COHERENT_BUNDLES"
            )
        ].copy()

        income_single_tables = group.loc[
            group[
                "income_bundle_status"
            ].eq(
                "INCOME_SINGLE_CLUSTER_BUNDLE"
            )
        ].copy()

        income_multi_tables = group.loc[
            group[
                "income_bundle_status"
            ].eq(
                "INCOME_MULTIPLE_CLUSTER_BUNDLE"
            )
        ].copy()

        if len(
            balance_unique_tables
        ) == 0:
            balance_receipt_status = (
                "NO_UNIQUE_BALANCE_TABLE"
            )

        elif len(
            balance_unique_tables
        ) == 1:
            balance_receipt_status = (
                "ONE_UNIQUE_BALANCE_TABLE"
            )

        else:
            balance_receipt_status = (
                "MULTIPLE_UNIQUE_BALANCE_TABLES"
            )

        if len(
            income_single_tables
        ) == 0:
            income_receipt_status = (
                "NO_SINGLE_CLUSTER_INCOME_TABLE"
            )

        elif len(
            income_single_tables
        ) == 1:
            income_receipt_status = (
                "ONE_SINGLE_CLUSTER_INCOME_TABLE"
            )

        else:
            income_receipt_status = (
                "MULTIPLE_SINGLE_CLUSTER_INCOME_TABLES"
            )

        bundle_ready = (
            len(
                balance_unique_tables
            ) == 1
            and len(
                income_single_tables
            ) == 1
        )

        stock_values = (
            group[
                "stock_code"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        period_values = (
            group[
                "period_key"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                stock_values[
                    0
                ]
                if stock_values
                else "",

                "period_key":
                period_values[
                    0
                ]
                if period_values
                else "",

                "table_count":
                len(
                    group
                ),

                "unique_balance_table_count":
                len(
                    balance_unique_tables
                ),

                "multi_coherent_balance_table_count":
                len(
                    balance_multi_tables
                ),

                "single_cluster_income_table_count":
                len(
                    income_single_tables
                ),

                "multi_cluster_income_table_count":
                len(
                    income_multi_tables
                ),

                "balance_receipt_status":
                balance_receipt_status,

                "income_receipt_status":
                income_receipt_status,

                "bundle_ready_unique":
                bundle_ready,

                "candidate_balance_tables":
                "|".join(
                    str(
                        x
                    )
                    for x in balance_unique_tables[
                        "table_index"
                    ].tolist()
                ),

                "candidate_income_tables":
                "|".join(
                    str(
                        x
                    )
                    for x in income_single_tables[
                        "table_index"
                    ].tolist()
                ),
            }
        )

        if (
            len(
                balance_unique_tables
            ) > 1
            or len(
                income_single_tables
            ) > 1
            or len(
                balance_multi_tables
            ) > 0
            or len(
                income_multi_tables
            ) > 0
        ):
            for row in group.loc[
                group[
                    "balance_bundle_status"
                ].isin(
                    {
                        "BALANCE_UNIQUE_COHERENT_BUNDLE",
                        "BALANCE_MULTIPLE_COHERENT_BUNDLES",
                    }
                )
                | group[
                    "income_bundle_status"
                ].isin(
                    {
                        "INCOME_SINGLE_CLUSTER_BUNDLE",
                        "INCOME_MULTIPLE_CLUSTER_BUNDLE",
                    }
                )
            ].itertuples(
                index=False
            ):

                ambiguous_records.append(
                    {
                        "rcept_no":
                        rcept_no,

                        "stock_code":
                        row.stock_code,

                        "period_key":
                        row.period_key,

                        "table_index":
                        row.table_index,

                        "selector_table_basis":
                        row.selector_table_basis,

                        "balance_bundle_status":
                        row.balance_bundle_status,

                        "income_bundle_status":
                        row.income_bundle_status,

                        "assets_cluster_count":
                        row.assets_cluster_count,

                        "liabilities_cluster_count":
                        row.liabilities_cluster_count,

                        "equity_cluster_count":
                        row.equity_cluster_count,

                        "revenue_cluster_count":
                        row.revenue_cluster_count,

                        "operating_income_cluster_count":
                        row.operating_income_cluster_count,

                        "net_income_cluster_count":
                        row.net_income_cluster_count,
                    }
                )

    receipt_summary = pd.DataFrame(
        receipt_records
    )

    ambiguous_summary = pd.DataFrame(
        ambiguous_records
    )

    receipt_summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    ambiguous_summary.to_csv(
        OUT_AMBIGUOUS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Table balance bundle status]"
    )

    print(
        table_summary[
            "balance_bundle_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Table income bundle status]"
    )

    print(
        table_summary[
            "income_bundle_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Receipt balance status]"
    )

    print(
        receipt_summary[
            "balance_receipt_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Receipt income status]"
    )

    print(
        receipt_summary[
            "income_receipt_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Bundle-ready unique receipts]"
    )

    print(
        int(
            receipt_summary[
                "bundle_ready_unique"
            ].sum()
        )
    )

    print(
        "\n[Receipts with >=1 unique coherent balance table]"
    )

    print(
        int(
            receipt_summary[
                "unique_balance_table_count"
            ]
            .gt(
                0
            )
            .sum()
        )
    )

    print(
        "\n[Receipts with >=1 single-cluster income table]"
    )

    print(
        int(
            receipt_summary[
                "single_cluster_income_table_count"
            ]
            .gt(
                0
            )
            .sum()
        )
    )

    # CFS/OFS basis among good bundles
    good_balance = table_summary.loc[
        table_summary[
            "balance_bundle_status"
        ].eq(
            "BALANCE_UNIQUE_COHERENT_BUNDLE"
        )
    ]

    good_income = table_summary.loc[
        table_summary[
            "income_bundle_status"
        ].eq(
            "INCOME_SINGLE_CLUSTER_BUNDLE"
        )
    ]

    print(
        "\n[Good balance bundles by table basis]"
    )

    if good_balance.empty:
        print(
            "None"
        )
    else:
        print(
            good_balance[
                "selector_table_basis"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Good income bundles by table basis]"
    )

    if good_income.empty:
        print(
            "None"
        )
    else:
        print(
            good_income[
                "selector_table_basis"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Multiple candidate table sample]"
    )

    multi_receipts = receipt_summary.loc[
        receipt_summary[
            "unique_balance_table_count"
        ].gt(
            1
        )
        | receipt_summary[
            "single_cluster_income_table_count"
        ].gt(
            1
        )
    ]

    if multi_receipts.empty:
        print(
            "None"
        )
    else:
        print(
            multi_receipts[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "unique_balance_table_count",
                    "single_cluster_income_table_count",
                    "candidate_balance_tables",
                    "candidate_income_tables",
                ]
            ]
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

    print(
        "\nOutputs:"
    )

    print(
        f"- Table summary   : "
        f"{OUT_TABLE}"
    )

    print(
        f"- Receipt summary : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Balance combos  : "
        f"{OUT_BALANCE_COMBOS}"
    )

    print(
        f"- Ambiguous audit : "
        f"{OUT_AMBIGUOUS}"
    )

    print(
        "\n해석 원칙:"
        "\n- metric별 독립선택보다 같은 statement table bundle을 우선"
        "\n- balance는 assets = liabilities + equity를 만족하는 cluster 조합만 인정"
        "\n- income는 revenue/op/net이 같은 table에 모두 있고 각 1 cluster인지 확인"
        "\n- 여러 정상 table이 남으면 아직 임의 선택하지 않음"
        "\n- 이 결과를 보고 table ranking / account-label ranking 규칙을 결정"
    )


if __name__ == "__main__":
    main()
