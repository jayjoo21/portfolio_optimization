from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7J. Duplicate-Equivalent Statement Bundle Audit
#
# 목적
# ------------------------------------------------------------
# H6B7I에서 receipt 안에 여러 정상 table이 남은 경우,
# 그 여러 table이:
#
#   A) 같은 경제적 값을 반복한 duplicate-equivalent table인지
#   B) 실제 숫자가 다른 genuine conflict인지
#
# 분리한다.
#
# 핵심:
# - 아직 "어느 table을 최종 선택"하지 않는다.
# - 같은 경제적 signature면 table index가 달라도 1개 value bundle로 collapse.
# - genuine conflict만 다음 table/account ranking 단계로 넘긴다.
#
# INPUT
# ------------------------------------------------------------
# dart_noncorrected_nodata_statement_bundle_table_summary.csv
# dart_noncorrected_nodata_statement_bundle_balance_combos.csv
# dart_noncorrected_nodata_selector_v2_evidence_tier_rows.parquet
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_nodata_bundle_equivalence_receipt_summary.csv
# dart_noncorrected_nodata_bundle_equivalence_table_signatures.csv
# dart_noncorrected_nodata_bundle_genuine_conflict_rows.csv
#
# 실행:
# python scripts\05a5h6b7j_dart_bundle_equivalence_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TABLE_SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_table_summary.csv"
)

BALANCE_COMBOS = (
    INTERIM
    / "dart_noncorrected_nodata_statement_bundle_balance_combos.csv"
)

POLICY_ROWS = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_rows.parquet"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_bundle_equivalence_receipt_summary.csv"
)

OUT_SIGNATURES = (
    INTERIM
    / "dart_noncorrected_nodata_bundle_equivalence_table_signatures.csv"
)

OUT_CONFLICTS = (
    INTERIM
    / "dart_noncorrected_nodata_bundle_genuine_conflict_rows.csv"
)


# ============================================================
# Helpers
# ============================================================

def near_equal(a: float, b: float) -> bool:
    """
    진단용 economic equivalence 기준.
    아직 production merge 규칙은 아님.

    - 절대 차이 <= 1,000,000 KRW
      OR
    - 상대 차이 <= 1e-5
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


def parse_cluster_string(value) -> list[list[float]]:
    """
    H6B7I의
      "100,101 || 200"
    형태를
      [[100,101],[200]]
    로 복원.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    out = []

    for cluster_text in text.split("||"):

        vals = []

        for piece in cluster_text.split(","):

            piece = piece.strip()

            if not piece:
                continue

            try:
                vals.append(
                    float(piece)
                )
            except Exception:
                pass

        if vals:
            out.append(
                vals
            )

    return out


def representative_of_single_cluster(value) -> float:
    clusters = parse_cluster_string(
        value
    )

    if len(clusters) != 1:
        return np.nan

    return float(
        np.median(
            np.array(
                clusters[0],
                dtype=float,
            )
        )
    )


def signatures_equivalent(
    left: tuple,
    right: tuple,
) -> bool:

    if len(left) != len(right):
        return False

    for a, b in zip(
        left,
        right,
    ):
        if pd.isna(a) or pd.isna(b):
            return False

        if not near_equal(
            float(a),
            float(b),
        ):
            return False

    return True


def cluster_signatures(
    records: list[dict],
    signature_key: str,
) -> list[list[dict]]:

    clusters: list[list[dict]] = []

    for record in records:

        sig = record[
            signature_key
        ]

        placed = False

        for cluster in clusters:

            if signatures_equivalent(
                sig,
                cluster[0][
                    signature_key
                ],
            ):
                cluster.append(
                    record
                )
                placed = True
                break

        if not placed:
            clusters.append(
                [
                    record
                ]
            )

    return clusters


def compact_context(
    series: pd.Series,
    limit: int = 600,
) -> str:

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = [
        x
        for x in values
        if x
    ]

    values = list(
        dict.fromkeys(
            values
        )
    )

    text = " || ".join(
        values[:4]
    )

    return text[
        :limit
    ]


# ============================================================
# Main
# ============================================================

def main():

    for path in [
        TABLE_SUMMARY,
        BALANCE_COMBOS,
        POLICY_ROWS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    table = pd.read_csv(
        TABLE_SUMMARY,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    balance = pd.read_csv(
        BALANCE_COMBOS,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    rows = pd.read_parquet(
        POLICY_ROWS
    )

    rows = rows.loc[
        rows[
            "evidence_policy"
        ].eq(
            "TABLE_BASIS_SUPPORTED"
        )
    ].copy()

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7J DUPLICATE-EQUIVALENT STATEMENT BUNDLE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nTable rows   : "
        f"{len(table):,}"
    )

    print(
        f"Balance combos: "
        f"{len(balance):,}"
    )

    print(
        f"Policy rows  : "
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # 1. Build table context metadata
    # --------------------------------------------------------

    context = (
        rows.groupby(
            [
                "rcept_no",
                "table_index",
            ],
            dropna=False,
        )
        .agg(
            heading_context=(
                "heading_context",
                compact_context,
            ),

            account_families=(
                "account_family",
                lambda s:
                "|".join(
                    sorted(
                        set(
                            s.dropna()
                            .astype(str)
                        )
                    )
                ),
            ),

            primary_labels=(
                "primary_row_label",
                compact_context,
            )
            if "primary_row_label"
            in rows.columns
            else (
                "account_family",
                lambda s:
                "",
            ),

            basis_v2_values=(
                "basis_v2",
                lambda s:
                "|".join(
                    sorted(
                        set(
                            s.dropna()
                            .astype(str)
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    table = table.merge(
        context,
        on=[
            "rcept_no",
            "table_index",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 2. Balance signatures
    # --------------------------------------------------------

    # Only tables H6B7I called unique coherent.
    good_balance = table.loc[
        table[
            "balance_bundle_status"
        ].eq(
            "BALANCE_UNIQUE_COHERENT_BUNDLE"
        )
    ].copy()

    # balance_combos already contains exact coherent raw combo.
    bal_one = (
        balance.sort_values(
            [
                "rcept_no",
                "table_index",
                "abs_gap",
                "rel_gap",
            ]
        )
        .drop_duplicates(
            [
                "rcept_no",
                "table_index",
            ],
            keep="first",
        )
        [
            [
                "rcept_no",
                "table_index",
                "assets",
                "liabilities",
                "equity",
                "gap",
                "abs_gap",
                "rel_gap",
            ]
        ]
    )

    good_balance = good_balance.merge(
        bal_one,
        on=[
            "rcept_no",
            "table_index",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 3. Income signatures
    # --------------------------------------------------------

    good_income = table.loc[
        table[
            "income_bundle_status"
        ].eq(
            "INCOME_SINGLE_CLUSTER_BUNDLE"
        )
    ].copy()

    good_income[
        "revenue_rep"
    ] = good_income[
        "revenue_clusters"
    ].map(
        representative_of_single_cluster
    )

    good_income[
        "operating_income_rep"
    ] = good_income[
        "operating_income_clusters"
    ].map(
        representative_of_single_cluster
    )

    good_income[
        "net_income_rep"
    ] = good_income[
        "net_income_clusters"
    ].map(
        representative_of_single_cluster
    )

    # --------------------------------------------------------
    # 4. Table signature rows
    # --------------------------------------------------------

    signature_records = []

    for row in good_balance.itertuples(
        index=False
    ):

        signature_records.append(
            {
                "bundle_type":
                "BALANCE",

                "rcept_no":
                row.rcept_no,

                "stock_code":
                row.stock_code,

                "period_key":
                row.period_key,

                "table_index":
                row.table_index,

                "selector_table_basis":
                row.selector_table_basis,

                "v1":
                row.assets,

                "v2":
                row.liabilities,

                "v3":
                row.equity,

                "signature_text":
                (
                    f"A={row.assets:.12g}"
                    f"|L={row.liabilities:.12g}"
                    f"|E={row.equity:.12g}"
                ),

                "balance_abs_gap":
                row.abs_gap,

                "balance_rel_gap":
                row.rel_gap,

                "heading_context":
                getattr(
                    row,
                    "heading_context",
                    "",
                ),

                "primary_labels":
                getattr(
                    row,
                    "primary_labels",
                    "",
                ),
            }
        )

    for row in good_income.itertuples(
        index=False
    ):

        signature_records.append(
            {
                "bundle_type":
                "INCOME",

                "rcept_no":
                row.rcept_no,

                "stock_code":
                row.stock_code,

                "period_key":
                row.period_key,

                "table_index":
                row.table_index,

                "selector_table_basis":
                row.selector_table_basis,

                "v1":
                row.revenue_rep,

                "v2":
                row.operating_income_rep,

                "v3":
                row.net_income_rep,

                "signature_text":
                (
                    f"R={row.revenue_rep:.12g}"
                    f"|OP={row.operating_income_rep:.12g}"
                    f"|N={row.net_income_rep:.12g}"
                ),

                "balance_abs_gap":
                np.nan,

                "balance_rel_gap":
                np.nan,

                "heading_context":
                getattr(
                    row,
                    "heading_context",
                    "",
                ),

                "primary_labels":
                getattr(
                    row,
                    "primary_labels",
                    "",
                ),
            }
        )

    signatures = pd.DataFrame(
        signature_records
    )

    # --------------------------------------------------------
    # 5. Receipt-level equivalence clustering
    # --------------------------------------------------------

    receipt_records = []
    conflict_records = []

    all_receipts = sorted(
        set(
            table[
                "rcept_no"
            ].astype(str)
        )
    )

    for rcept_no in all_receipts:

        sub_bal = signatures.loc[
            signatures[
                "rcept_no"
            ].eq(
                rcept_no
            )
            & signatures[
                "bundle_type"
            ].eq(
                "BALANCE"
            )
        ].copy()

        sub_inc = signatures.loc[
            signatures[
                "rcept_no"
            ].eq(
                rcept_no
            )
            & signatures[
                "bundle_type"
            ].eq(
                "INCOME"
            )
        ].copy()

        bal_records = []

        for row in sub_bal.itertuples(
            index=False
        ):

            bal_records.append(
                {
                    "table_index":
                    row.table_index,

                    "signature":
                    (
                        row.v1,
                        row.v2,
                        row.v3,
                    ),

                    "signature_text":
                    row.signature_text,

                    "selector_table_basis":
                    row.selector_table_basis,

                    "heading_context":
                    row.heading_context,

                    "primary_labels":
                    row.primary_labels,
                }
            )

        inc_records = []

        for row in sub_inc.itertuples(
            index=False
        ):

            inc_records.append(
                {
                    "table_index":
                    row.table_index,

                    "signature":
                    (
                        row.v1,
                        row.v2,
                        row.v3,
                    ),

                    "signature_text":
                    row.signature_text,

                    "selector_table_basis":
                    row.selector_table_basis,

                    "heading_context":
                    row.heading_context,

                    "primary_labels":
                    row.primary_labels,
                }
            )

        bal_clusters = cluster_signatures(
            bal_records,
            "signature",
        )

        inc_clusters = cluster_signatures(
            inc_records,
            "signature",
        )

        def status(
            table_count: int,
            cluster_count: int,
            prefix: str,
        ) -> str:

            if table_count == 0:
                return (
                    f"{prefix}_NONE"
                )

            if table_count == 1:
                return (
                    f"{prefix}_ONE_TABLE"
                )

            if cluster_count == 1:
                return (
                    f"{prefix}_MULTIPLE_TABLES_DUPLICATE_EQUIVALENT"
                )

            return (
                f"{prefix}_MULTIPLE_GENUINE_SIGNATURES"
            )

        bal_status = status(
            len(
                bal_records
            ),
            len(
                bal_clusters
            ),
            "BALANCE",
        )

        inc_status = status(
            len(
                inc_records
            ),
            len(
                inc_clusters
            ),
            "INCOME",
        )

        meta = table.loc[
            table[
                "rcept_no"
            ].eq(
                rcept_no
            )
        ].iloc[
            0
        ]

        economic_balance_unique = (
            len(
                bal_clusters
            ) == 1
        )

        economic_income_unique = (
            len(
                inc_clusters
            ) == 1
        )

        economic_bundle_ready = (
            economic_balance_unique
            and economic_income_unique
        )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                meta[
                    "stock_code"
                ],

                "period_key":
                meta[
                    "period_key"
                ],

                "balance_table_count":
                len(
                    bal_records
                ),

                "balance_signature_cluster_count":
                len(
                    bal_clusters
                ),

                "balance_equivalence_status":
                bal_status,

                "income_table_count":
                len(
                    inc_records
                ),

                "income_signature_cluster_count":
                len(
                    inc_clusters
                ),

                "income_equivalence_status":
                inc_status,

                "economic_balance_unique":
                economic_balance_unique,

                "economic_income_unique":
                economic_income_unique,

                "economic_bundle_ready":
                economic_bundle_ready,

                "balance_tables_by_cluster":
                " || ".join(
                    ",".join(
                        str(
                            rec[
                                "table_index"
                            ]
                        )
                        for rec in cluster
                    )
                    for cluster in bal_clusters
                ),

                "income_tables_by_cluster":
                " || ".join(
                    ",".join(
                        str(
                            rec[
                                "table_index"
                            ]
                        )
                        for rec in cluster
                    )
                    for cluster in inc_clusters
                ),
            }
        )

        if len(
            bal_clusters
        ) > 1:

            for cluster_id, cluster in enumerate(
                bal_clusters,
                start=1,
            ):

                for rec in cluster:

                    conflict_records.append(
                        {
                            "bundle_type":
                            "BALANCE",

                            "rcept_no":
                            rcept_no,

                            "stock_code":
                            meta[
                                "stock_code"
                            ],

                            "period_key":
                            meta[
                                "period_key"
                            ],

                            "signature_cluster":
                            cluster_id,

                            "table_index":
                            rec[
                                "table_index"
                            ],

                            "selector_table_basis":
                            rec[
                                "selector_table_basis"
                            ],

                            "signature_text":
                            rec[
                                "signature_text"
                            ],

                            "primary_labels":
                            rec[
                                "primary_labels"
                            ],

                            "heading_context":
                            rec[
                                "heading_context"
                            ],
                        }
                    )

        if len(
            inc_clusters
        ) > 1:

            for cluster_id, cluster in enumerate(
                inc_clusters,
                start=1,
            ):

                for rec in cluster:

                    conflict_records.append(
                        {
                            "bundle_type":
                            "INCOME",

                            "rcept_no":
                            rcept_no,

                            "stock_code":
                            meta[
                                "stock_code"
                            ],

                            "period_key":
                            meta[
                                "period_key"
                            ],

                            "signature_cluster":
                            cluster_id,

                            "table_index":
                            rec[
                                "table_index"
                            ],

                            "selector_table_basis":
                            rec[
                                "selector_table_basis"
                            ],

                            "signature_text":
                            rec[
                                "signature_text"
                            ],

                            "primary_labels":
                            rec[
                                "primary_labels"
                            ],

                            "heading_context":
                            rec[
                                "heading_context"
                            ],
                        }
                    )

    receipt_summary = pd.DataFrame(
        receipt_records
    )

    conflicts = pd.DataFrame(
        conflict_records
    )

    # --------------------------------------------------------
    # 6. Add cluster IDs to signature table
    # --------------------------------------------------------

    signature_cluster_records = []

    for (
        rcept_no,
        bundle_type,
    ), group in signatures.groupby(
        [
            "rcept_no",
            "bundle_type",
        ],
        sort=False,
    ):

        records = []

        for row in group.itertuples(
            index=False
        ):

            records.append(
                {
                    "row":
                    row,

                    "signature":
                    (
                        row.v1,
                        row.v2,
                        row.v3,
                    ),
                }
            )

        clusters = cluster_signatures(
            records,
            "signature",
        )

        for cluster_id, cluster in enumerate(
            clusters,
            start=1,
        ):

            for record in cluster:

                row = record[
                    "row"
                ]

                signature_cluster_records.append(
                    {
                        "bundle_type":
                        bundle_type,

                        "rcept_no":
                        rcept_no,

                        "stock_code":
                        row.stock_code,

                        "period_key":
                        row.period_key,

                        "table_index":
                        row.table_index,

                        "signature_cluster":
                        cluster_id,

                        "signature_cluster_count":
                        len(
                            clusters
                        ),

                        "selector_table_basis":
                        row.selector_table_basis,

                        "v1":
                        row.v1,

                        "v2":
                        row.v2,

                        "v3":
                        row.v3,

                        "signature_text":
                        row.signature_text,

                        "balance_abs_gap":
                        row.balance_abs_gap,

                        "balance_rel_gap":
                        row.balance_rel_gap,

                        "primary_labels":
                        row.primary_labels,

                        "heading_context":
                        row.heading_context,
                    }
                )

    signature_out = pd.DataFrame(
        signature_cluster_records
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    receipt_summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    signature_out.to_csv(
        OUT_SIGNATURES,
        index=False,
        encoding="utf-8-sig",
    )

    conflicts.to_csv(
        OUT_CONFLICTS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Balance equivalence status]"
    )

    print(
        receipt_summary[
            "balance_equivalence_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Income equivalence status]"
    )

    print(
        receipt_summary[
            "income_equivalence_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Economic-unique balance receipts]"
    )

    print(
        int(
            receipt_summary[
                "economic_balance_unique"
            ].sum()
        )
    )

    print(
        "\n[Economic-unique income receipts]"
    )

    print(
        int(
            receipt_summary[
                "economic_income_unique"
            ].sum()
        )
    )

    print(
        "\n[Economic bundle-ready receipts]"
    )

    print(
        int(
            receipt_summary[
                "economic_bundle_ready"
            ].sum()
        )
    )

    print(
        "\n[Multiple-table balance: duplicate-equivalent vs genuine]"
    )

    multi_bal = receipt_summary.loc[
        receipt_summary[
            "balance_table_count"
        ].gt(
            1
        )
    ]

    if multi_bal.empty:
        print(
            "None"
        )
    else:
        print(
            multi_bal[
                "balance_equivalence_status"
            ]
            .value_counts()
            .to_string()
        )

    print(
        "\n[Multiple-table income: duplicate-equivalent vs genuine]"
    )

    multi_inc = receipt_summary.loc[
        receipt_summary[
            "income_table_count"
        ].gt(
            1
        )
    ]

    if multi_inc.empty:
        print(
            "None"
        )
    else:
        print(
            multi_inc[
                "income_equivalence_status"
            ]
            .value_counts()
            .to_string()
        )

    print(
        "\n[Genuine conflict receipt sample]"
    )

    conflict_receipts = receipt_summary.loc[
        receipt_summary[
            "balance_signature_cluster_count"
        ].gt(
            1
        )
        | receipt_summary[
            "income_signature_cluster_count"
        ].gt(
            1
        )
    ]

    if conflict_receipts.empty:
        print(
            "None"
        )
    else:
        print(
            conflict_receipts[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "balance_table_count",
                    "balance_signature_cluster_count",
                    "balance_tables_by_cluster",
                    "income_table_count",
                    "income_signature_cluster_count",
                    "income_tables_by_cluster",
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
        "\n[Conflict table detail sample]"
    )

    if conflicts.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            320,
            "display.max_rows",
            40,
        ):
            print(
                conflicts[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "signature_cluster",
                        "table_index",
                        "selector_table_basis",
                        "signature_text",
                        "primary_labels",
                        "heading_context",
                    ]
                ]
                .head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\nOutputs:"
    )

    print(
        f"- Receipt summary : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Table signatures: "
        f"{OUT_SIGNATURES}"
    )

    print(
        f"- Genuine conflicts: "
        f"{OUT_CONFLICTS}"
    )

    print(
        "\n해석 원칙:"
        "\n- 여러 table이어도 동일 economic signature면 duplicate-equivalent"
        "\n- duplicate-equivalent는 value ambiguity가 아니라 presentation duplication"
        "\n- genuine signature conflict만 다음 ranking 대상으로 넘김"
        "\n- 아직 어느 table index를 production source로 확정하지 않음"
    )


if __name__ == "__main__":
    main()
