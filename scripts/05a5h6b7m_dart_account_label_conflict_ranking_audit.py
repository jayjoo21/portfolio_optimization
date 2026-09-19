from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7M. Account-Label Semantics + Conflict Ranking Audit
#
# 목적
# ------------------------------------------------------------
# H6B7L까지 table-level basis / entity owner를 정리했다.
# 이제 두 가지를 동시에 진단한다.
#
# 1) 각 statement bundle 안의 metric이 정말 "총계" 계정인지 확인
#    - assets        -> 자산총계 / 총자산
#    - liabilities   -> 부채총계 / 총부채
#    - equity        -> 자본총계 / 총자본
#    - revenue       -> 매출액 / 영업수익 등 total top-line
#    - op income     -> 영업이익(손실)
#    - net income    -> 당기순이익(손실) total
#
# 2) H6B7L KEEP_FOR_RANKING genuine conflicts에 대해
#    arbitrary weighted score가 아니라 "lexicographic evidence tuple"로
#    어떤 table이 의미론적으로 우위인지 shadow ranking한다.
#
# Ranking priority:
#   A. refined basis strength
#   B. issuer / group owner evidence
#   C. total-account label quality
#   D. explicit statement-context specificity
#
# IMPORTANT
# ------------------------------------------------------------
# - 아직 production selection 아님
# - tie는 tie로 남김
# - 금융업 revenue semantics는 별도 review가 필요하므로
#   확실한 부분계정(예: 기타영업수익)만 강하게 경고
# - ZIP 재파싱 없음
#
# 실행:
# python scripts\05a5h6b7m_dart_account_label_conflict_ranking_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TABLE_SEMANTICS = (
    INTERIM
    / "dart_noncorrected_nodata_table_refined_semantics.csv"
)

CONFLICT_SEMANTICS = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_refined_semantics.csv"
)

POLICY_ROWS = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_evidence_tier_rows.parquet"
)

OUT_SIGNATURE_LABELS = (
    INTERIM
    / "dart_noncorrected_nodata_signature_account_label_audit.csv"
)

OUT_CONFLICT_RANKING = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_semantic_ranking_audit.csv"
)

OUT_GROUP_SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_semantic_ranking_group_summary.csv"
)


# ============================================================
# General helpers
# ============================================================


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def compact(value) -> str:
    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        clean(value).lower(),
    )


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


def near_equal(a: float, b: float) -> bool:
    """
    H6B7H/J와 같은 진단용 economic-near-equivalence.
    """
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
        or diff / denom <= 1e-5
    )


# ============================================================
# Label source discovery
# ============================================================


LABEL_COLUMN_CANDIDATES = [
    "primary_row_label",
    "row_label",
    "account_nm",
    "account_name",
    "account_label",
    "label",
]


def discover_label_column(
    df: pd.DataFrame,
):
    for col in LABEL_COLUMN_CANDIDATES:
        if col in df.columns:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if values.ne(
                ""
            ).any():
                return col

    return None


def labels_matching_signature_value(
    rows: pd.DataFrame,
    family: str,
    target_value: float,
    label_col: str | None,
) -> list[str]:

    if label_col is None:
        return []

    sub = rows.loc[
        rows[
            "account_family"
        ].eq(
            family
        )
    ].copy()

    if sub.empty:
        return []

    numeric = pd.to_numeric(
        sub[
            "selector_value_krw"
        ],
        errors="coerce",
    )

    mask = [
        near_equal(
            value,
            target_value,
        )
        if pd.notna(value)
        else False
        for value in numeric
    ]

    sub = sub.loc[
        mask
    ]

    labels = (
        sub[
            label_col
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return list(
        dict.fromkeys(
            label
            for label in labels
            if label
        )
    )


# ============================================================
# Account-label semantics
# ============================================================


def metric_label_quality(
    family: str,
    labels: list[str],
) -> tuple[int, str, str]:

    """
    Returns:
      quality_rank:
        3 = strong total label
        2 = acceptable / likely total
        1 = unresolved/generic
        0 = suspect partial/attributable metric

      class
      reason
    """

    normalized = [
        compact(
            label
        )
        for label in labels
    ]

    if not normalized:
        return (
            1,
            "LABEL_UNAVAILABLE",
            "",
        )

    joined = "|".join(
        normalized
    )

    # -----------------------------
    # Balance sheet
    # -----------------------------

    if family == "assets":

        if any(
            token in joined
            for token in [
                "자산총계",
                "총자산",
            ]
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "assets_total",
            )

        return (
            1,
            "GENERIC_OR_UNRESOLVED_LABEL",
            "assets_not_explicit_total",
        )

    if family == "liabilities":

        if any(
            token in joined
            for token in [
                "부채총계",
                "총부채",
            ]
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "liabilities_total",
            )

        return (
            1,
            "GENERIC_OR_UNRESOLVED_LABEL",
            "liabilities_not_explicit_total",
        )

    if family == "equity":

        # User's target metric is total equity.
        attributable_markers = [
            "지배기업소유주지분",
            "지배기업의소유주지분",
            "지배기업소유주에게귀속",
            "지배기업의소유주에게귀속",
        ]

        if any(
            marker in joined
            for marker in attributable_markers
        ):
            if not any(
                token in joined
                for token in [
                    "자본총계",
                    "총자본",
                ]
            ):
                return (
                    0,
                    "SUSPECT_ATTRIBUTABLE_NOT_TOTAL",
                    "equity_attributable_only",
                )

        if any(
            token in joined
            for token in [
                "자본총계",
                "총자본",
            ]
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "equity_total",
            )

        return (
            1,
            "GENERIC_OR_UNRESOLVED_LABEL",
            "equity_not_explicit_total",
        )

    # -----------------------------
    # Income statement
    # -----------------------------

    if family == "revenue":

        # Strong partial-component warning only.
        partial_markers = [
            "기타영업수익",
            "기타수익",
            "영업외수익",
        ]

        if any(
            marker in joined
            for marker in partial_markers
        ):
            return (
                0,
                "SUSPECT_PARTIAL_REVENUE_LABEL",
                "revenue_component_not_total",
            )

        total_markers = [
            "매출액",
            "매출",
            "영업수익",
            "수익합계",
            "영업수익합계",
        ]

        if any(
            marker in joined
            for marker in total_markers
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "revenue_total_like",
            )

        # Financial-sector top line can use diverse terminology.
        # Do not hard-fail unknown labels here.
        return (
            1,
            "GENERIC_OR_DOMAIN_REVIEW_LABEL",
            "revenue_requires_domain_review",
        )

    if family == "operating_income":

        if any(
            token in joined
            for token in [
                "영업이익",
                "영업손실",
            ]
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "operating_income_total",
            )

        return (
            1,
            "GENERIC_OR_UNRESOLVED_LABEL",
            "operating_income_not_explicit",
        )

    if family == "net_income":

        attributable_markers = [
            "지배기업소유주에게귀속되는당기순이익",
            "지배기업의소유주에게귀속되는당기순이익",
            "지배기업소유주지분순이익",
        ]

        if any(
            marker in joined
            for marker in attributable_markers
        ):
            if not any(
                token in joined
                for token in [
                    "당기순이익",
                    "당기순손실",
                    "분기순이익",
                    "반기순이익",
                ]
            ):
                return (
                    0,
                    "SUSPECT_ATTRIBUTABLE_NOT_TOTAL",
                    "net_income_attributable_only",
                )

        if any(
            token in joined
            for token in [
                "당기순이익",
                "당기순손실",
                "분기순이익",
                "분기순손실",
                "반기순이익",
                "반기순손실",
            ]
        ):
            return (
                3,
                "STRONG_TOTAL_LABEL",
                "net_income_total",
            )

        return (
            1,
            "GENERIC_OR_UNRESOLVED_LABEL",
            "net_income_not_explicit",
        )

    return (
        1,
        "UNKNOWN_FAMILY",
        "",
    )


# ============================================================
# Ranking evidence
# ============================================================


def basis_strength(
    selector_basis: str,
    refined_basis: str,
) -> int:

    selector_basis = str(
        selector_basis
    )

    refined_basis = str(
        refined_basis
    )

    # CFS branch
    if selector_basis == "CFS":

        if refined_basis == "CFS":
            return 4

        if refined_basis == "CFS_ENTITY_HINT":
            return 3

        if refined_basis == "UNKNOWN":
            return 2

        return 0

    # OFS fallback branch
    if selector_basis == "OFS_ONLY_NO_CFS":

        if refined_basis == "OFS_ONLY_NO_CFS":
            return 4

        if refined_basis == "OFS":
            return 3

        if refined_basis == "UNKNOWN":
            return 2

        return 0

    return 1


def entity_strength(
    scope: str,
) -> int:

    mapping = {
        "ISSUER_OR_GROUP_OWNER":
        4,

        "OTHER_ENTITY_NOTE_REFERENCE_ONLY":
        3,

        "ENTITY_SCOPE_UNKNOWN":
        2,

        "ISSUER_NAME_UNAVAILABLE":
        1,

        "MAIN_OTHER_COMPANY_AMBIGUOUS":
        0,

        "OTHER_ENTITY_OWNER_EXPLICIT":
        0,
    }

    return mapping.get(
        str(
            scope
        ),
        1,
    )


def context_specificity(
    bundle_type: str,
    main_context,
) -> int:

    text = compact(
        main_context
    )

    # Direct statement title is strongest.
    if bundle_type == "BALANCE":

        if any(
            marker in text
            for marker in [
                "연결재무상태표",
                "요약연결재무상태표",
            ]
        ):
            return 4

        if "요약연결재무정보" in text:
            return 3

        if "요약재무정보" in text:
            return 1

        return 0

    if bundle_type == "INCOME":

        if any(
            marker in text
            for marker in [
                "연결포괄손익계산서",
                "연결손익계산서",
                "요약연결포괄손익계산서",
                "요약연결손익계산서",
            ]
        ):
            return 4

        if "요약연결재무정보" in text:
            return 3

        if "요약재무정보" in text:
            return 1

        return 0

    return 0


def bundle_metric_names(
    bundle_type: str,
):
    if bundle_type == "BALANCE":
        return [
            "assets",
            "liabilities",
            "equity",
        ]

    return [
        "revenue",
        "operating_income",
        "net_income",
    ]


def signature_values(
    row: pd.Series,
):
    return [
        pd.to_numeric(
            row.get(
                "v1"
            ),
            errors="coerce",
        ),
        pd.to_numeric(
            row.get(
                "v2"
            ),
            errors="coerce",
        ),
        pd.to_numeric(
            row.get(
                "v3"
            ),
            errors="coerce",
        ),
    ]


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        TABLE_SEMANTICS,
        CONFLICT_SEMANTICS,
        POLICY_ROWS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    tables = pd.read_csv(
        TABLE_SEMANTICS,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    conflicts = pd.read_csv(
        CONFLICT_SEMANTICS,
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

    for frame in [
        tables,
        conflicts,
        rows,
    ]:
        frame[
            "rcept_no"
        ] = normalize_receipt(
            frame[
                "rcept_no"
            ]
        )

    label_col = discover_label_column(
        rows
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7M ACCOUNT-LABEL SEMANTICS + CONFLICT RANKING AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nSignature tables : "
        f"{len(tables):,}"
    )

    print(
        f"Conflict rows    : "
        f"{len(conflicts):,}"
    )

    print(
        f"Policy rows      : "
        f"{len(rows):,}"
    )

    print(
        "Label column     : "
        + (
            label_col
            if label_col
            else "NOT FOUND"
        )
    )

    # Fast lookup by receipt / table.
    row_groups = {
        key:
        group
        for key, group
        in rows.groupby(
            [
                "rcept_no",
                "table_index",
            ],
            sort=False,
        )
    }

    # --------------------------------------------------------
    # 1. Account-label audit for all signature tables
    # --------------------------------------------------------

    audit_records = []

    for _, row in tables.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        table_index = row[
            "table_index"
        ]

        bundle_type = str(
            row[
                "bundle_type"
            ]
        )

        group = row_groups.get(
            (
                rcept_no,
                table_index,
            ),
            rows.iloc[
                0:0
            ],
        )

        metrics = bundle_metric_names(
            bundle_type
        )

        values = signature_values(
            row
        )

        metric_results = {}

        for family, value in zip(
            metrics,
            values,
        ):

            labels = (
                labels_matching_signature_value(
                    group,
                    family,
                    value,
                    label_col,
                )
                if pd.notna(
                    value
                )
                else []
            )

            (
                quality_rank,
                quality_class,
                quality_reason,
            ) = metric_label_quality(
                family,
                labels,
            )

            metric_results[
                family
            ] = {
                "labels":
                labels,

                "quality_rank":
                quality_rank,

                "quality_class":
                quality_class,

                "quality_reason":
                quality_reason,
            }

        ranks = [
            result[
                "quality_rank"
            ]
            for result in metric_results.values()
        ]

        if all(
            rank == 3
            for rank in ranks
        ):
            bundle_label_class = (
                "ALL_STRONG_TOTAL_LABELS"
            )

        elif any(
            rank == 0
            for rank in ranks
        ):
            bundle_label_class = (
                "HAS_SUSPECT_PARTIAL_OR_ATTRIBUTABLE_LABEL"
            )

        else:
            bundle_label_class = (
                "LABEL_REVIEW_REQUIRED"
            )

        # Revenue magnitude sanity flag only.
        revenue_magnitude_suspect = False

        if bundle_type == "INCOME":

            revenue = values[
                0
            ]

            op = values[
                1
            ]

            net = values[
                2
            ]

            if (
                pd.notna(
                    revenue
                )
                and pd.notna(
                    op
                )
                and pd.notna(
                    net
                )
            ):
                benchmark = max(
                    abs(
                        float(
                            op
                        )
                    ),
                    abs(
                        float(
                            net
                        )
                    ),
                    1.0,
                )

                revenue_magnitude_suspect = (
                    abs(
                        float(
                            revenue
                        )
                    )
                    < 0.01
                    * benchmark
                    and benchmark
                    > 1_000_000
                )

        record = {
            "bundle_type":
            bundle_type,

            "rcept_no":
            rcept_no,

            "stock_code":
            row.get(
                "stock_code",
                "",
            ),

            "period_key":
            row.get(
                "period_key",
                "",
            ),

            "table_index":
            table_index,

            "signature_cluster":
            row.get(
                "signature_cluster",
                np.nan,
            ),

            "selector_table_basis":
            row.get(
                "selector_table_basis",
                "",
            ),

            "refined_local_basis":
            row.get(
                "refined_local_basis",
                "",
            ),

            "refined_entity_scope":
            row.get(
                "refined_entity_scope",
                "",
            ),

            "main_context":
            row.get(
                "main_context",
                "",
            ),

            "signature_text":
            row.get(
                "signature_text",
                "",
            ),

            "bundle_label_class":
            bundle_label_class,

            "bundle_label_min_rank":
            min(
                ranks
            )
            if ranks
            else 0,

            "bundle_label_sum_rank":
            sum(
                ranks
            ),

            "revenue_magnitude_suspect":
            revenue_magnitude_suspect,
        }

        for family in metrics:

            result = metric_results[
                family
            ]

            record[
                f"{family}_labels"
            ] = " || ".join(
                result[
                    "labels"
                ]
            )

            record[
                f"{family}_label_rank"
            ] = result[
                "quality_rank"
            ]

            record[
                f"{family}_label_class"
            ] = result[
                "quality_class"
            ]

            record[
                f"{family}_label_reason"
            ] = result[
                "quality_reason"
            ]

        audit_records.append(
            record
        )

    label_audit = pd.DataFrame(
        audit_records
    )

    label_audit.to_csv(
        OUT_SIGNATURE_LABELS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 2. Join labels to genuine-conflict semantic rows
    # --------------------------------------------------------

    key_cols = [
        "bundle_type",
        "rcept_no",
        "table_index",
    ]

    join_cols = [
        col
        for col in label_audit.columns
        if col not in [
            "stock_code",
            "period_key",
            "selector_table_basis",
            "refined_local_basis",
            "refined_entity_scope",
            "main_context",
            "signature_text",
            "signature_cluster",
        ]
    ]

    conflict_rank = conflicts.merge(
        label_audit[
            key_cols
            + [
                col
                for col in join_cols
                if col not in key_cols
            ]
        ],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    keep_mask = (
        conflict_rank[
            "refined_semantic_gate"
        ].eq(
            "KEEP_FOR_RANKING"
        )
    )

    # --------------------------------------------------------
    # 3. Evidence tuple
    # --------------------------------------------------------

    conflict_rank[
        "rank_basis_strength"
    ] = [
        basis_strength(
            selector,
            refined,
        )
        for selector, refined
        in zip(
            conflict_rank[
                "selector_table_basis"
            ],
            conflict_rank[
                "refined_local_basis"
            ],
        )
    ]

    conflict_rank[
        "rank_entity_strength"
    ] = conflict_rank[
        "refined_entity_scope"
    ].map(
        entity_strength
    )

    conflict_rank[
        "rank_label_strength"
    ] = pd.to_numeric(
        conflict_rank[
            "bundle_label_min_rank"
        ],
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    )

    conflict_rank[
        "rank_context_strength"
    ] = [
        context_specificity(
            bundle,
            main_context,
        )
        for bundle, main_context
        in zip(
            conflict_rank[
                "bundle_type"
            ],
            conflict_rank[
                "main_context"
            ],
        )
    ]

    conflict_rank[
        "rank_tuple"
    ] = [
        (
            int(
                b
            ),
            int(
                e
            ),
            int(
                l
            ),
            int(
                c
            ),
        )
        for b, e, l, c
        in zip(
            conflict_rank[
                "rank_basis_strength"
            ],
            conflict_rank[
                "rank_entity_strength"
            ],
            conflict_rank[
                "rank_label_strength"
            ],
            conflict_rank[
                "rank_context_strength"
            ],
        )
    ]

    conflict_rank[
        "rank_tuple_text"
    ] = conflict_rank[
        "rank_tuple"
    ].map(
        lambda x:
        "-".join(
            str(
                v
            )
            for v in x
        )
    )

    # --------------------------------------------------------
    # 4. Group-level shadow ranking
    # --------------------------------------------------------

    group_records = []

    conflict_rank[
        "shadow_rank_position"
    ] = np.nan

    conflict_rank[
        "shadow_is_top"
    ] = False

    conflict_rank[
        "shadow_group_result"
    ] = ""

    for (
        rcept_no,
        bundle_type,
    ), group in conflict_rank.loc[
        keep_mask
    ].groupby(
        [
            "rcept_no",
            "bundle_type",
        ],
        sort=False,
    ):

        tuples = group[
            "rank_tuple"
        ].tolist()

        best = max(
            tuples
        )

        top = group.loc[
            group[
                "rank_tuple"
            ].map(
                lambda value: value == best
            )
        ]

        unique_top = (
            len(
                top
            ) == 1
        )

        if len(
            group
        ) == 1:
            result = (
                "SINGLE_KEEP_TABLE"
            )

        elif unique_top:
            result = (
                "UNIQUE_SEMANTIC_TOP"
            )

        else:
            result = (
                "TIED_SEMANTIC_TOP"
            )

        # Dense lexicographic rank.
        unique_tuples = sorted(
            set(
                tuples
            ),
            reverse=True,
        )

        tuple_rank = {
            value:
            i + 1
            for i, value in enumerate(
                unique_tuples
            )
        }

        for idx in group.index:

            value = conflict_rank.at[
                idx,
                "rank_tuple",
            ]

            conflict_rank.at[
                idx,
                "shadow_rank_position",
            ] = tuple_rank[
                value
            ]

            conflict_rank.at[
                idx,
                "shadow_is_top",
            ] = (
                value == best
            )

            conflict_rank.at[
                idx,
                "shadow_group_result",
            ] = result

        top_table_indices = (
            top[
                "table_index"
            ]
            .astype(str)
            .tolist()
        )

        group_records.append(
            {
                "rcept_no":
                rcept_no,

                "bundle_type":
                bundle_type,

                "keep_table_count":
                len(
                    group
                ),

                "distinct_rank_tuple_count":
                len(
                    unique_tuples
                ),

                "shadow_group_result":
                result,

                "best_rank_tuple":
                "-".join(
                    str(
                        v
                    )
                    for v in best
                ),

                "top_table_count":
                len(
                    top
                ),

                "top_tables":
                "|".join(
                    top_table_indices
                ),

                "top_has_label_suspect":
                bool(
                    top[
                        "bundle_label_class"
                    ]
                    .eq(
                        "HAS_SUSPECT_PARTIAL_OR_ATTRIBUTABLE_LABEL"
                    )
                    .any()
                ),

                "top_has_revenue_magnitude_suspect":
                bool(
                    top[
                        "revenue_magnitude_suspect"
                    ]
                    .fillna(
                        False
                    )
                    .any()
                ),
            }
        )

    group_summary = pd.DataFrame(
        group_records
    )

    conflict_rank.to_csv(
        OUT_CONFLICT_RANKING,
        index=False,
        encoding="utf-8-sig",
    )

    group_summary.to_csv(
        OUT_GROUP_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Bundle account-label quality - all signature tables]"
    )

    print(
        label_audit[
            "bundle_label_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Revenue label class - income signatures]"
    )

    income = label_audit.loc[
        label_audit[
            "bundle_type"
        ].eq(
            "INCOME"
        )
    ]

    if income.empty:
        print(
            "None"
        )
    else:
        print(
            income[
                "revenue_label_class"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Income revenue magnitude suspect]"
    )

    print(
        int(
            income[
                "revenue_magnitude_suspect"
            ]
            .fillna(
                False
            )
            .sum()
        )
    )

    print(
        "\n[Genuine-conflict shadow ranking result]"
    )

    if group_summary.empty:
        print(
            "None"
        )
    else:
        print(
            group_summary[
                "shadow_group_result"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Groups with unique semantic top]"
    )

    unique_top_groups = group_summary.loc[
        group_summary[
            "shadow_group_result"
        ].eq(
            "UNIQUE_SEMANTIC_TOP"
        )
    ]

    print(
        f"{len(unique_top_groups):,}"
    )

    print(
        "\n[Tied semantic top groups]"
    )

    tied_groups = group_summary.loc[
        group_summary[
            "shadow_group_result"
        ].eq(
            "TIED_SEMANTIC_TOP"
        )
    ]

    print(
        f"{len(tied_groups):,}"
    )

    print(
        "\n[Unique top but label/revenue sanity suspect]"
    )

    suspect_groups = group_summary.loc[
        group_summary[
            "shadow_group_result"
        ].isin(
            [
                "SINGLE_KEEP_TABLE",
                "UNIQUE_SEMANTIC_TOP",
            ]
        )
        & (
            group_summary[
                "top_has_label_suspect"
            ]
            | group_summary[
                "top_has_revenue_magnitude_suspect"
            ]
        )
    ]

    print(
        f"{len(suspect_groups):,}"
    )

    if not suspect_groups.empty:
        print(
            suspect_groups.head(
                30
            )
            .to_string(
                index=False
            )
        )

    show_cols = [
        "bundle_type",
        "stock_code",
        "period_key",
        "rcept_no",
        "table_index",
        "selector_table_basis",
        "refined_local_basis",
        "refined_entity_scope",
        "bundle_label_class",
        "rank_basis_strength",
        "rank_entity_strength",
        "rank_label_strength",
        "rank_context_strength",
        "rank_tuple_text",
        "shadow_rank_position",
        "shadow_is_top",
        "signature_text",
        "primary_labels",
        "main_context",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in conflict_rank.columns
    ]

    print(
        "\n[Unique semantic top sample]"
    )

    unique_receipt_bundle = set(
        zip(
            unique_top_groups[
                "rcept_no"
            ],
            unique_top_groups[
                "bundle_type"
            ],
        )
    )

    unique_rows = conflict_rank.loc[
        [
            (
                r,
                b,
            )
            in unique_receipt_bundle
            for r, b
            in zip(
                conflict_rank[
                    "rcept_no"
                ],
                conflict_rank[
                    "bundle_type"
                ],
            )
        ]
    ]

    if unique_rows.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            340,
            "display.max_rows",
            50,
        ):
            print(
                unique_rows[
                    show_cols
                ]
                .head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Tied semantic top sample]"
    )

    tied_receipt_bundle = set(
        zip(
            tied_groups[
                "rcept_no"
            ],
            tied_groups[
                "bundle_type"
            ],
        )
    )

    tied_rows = conflict_rank.loc[
        [
            (
                r,
                b,
            )
            in tied_receipt_bundle
            for r, b
            in zip(
                conflict_rank[
                    "rcept_no"
                ],
                conflict_rank[
                    "bundle_type"
                ],
            )
        ]
    ]

    if tied_rows.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            340,
            "display.max_rows",
            50,
        ):
            print(
                tied_rows[
                    show_cols
                ]
                .head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[All signature tables with suspect partial/attributable labels]"
    )

    suspect_signatures = label_audit.loc[
        label_audit[
            "bundle_label_class"
        ].eq(
            "HAS_SUSPECT_PARTIAL_OR_ATTRIBUTABLE_LABEL"
        )
        | label_audit[
            "revenue_magnitude_suspect"
        ].fillna(
            False
        )
    ]

    if suspect_signatures.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            150,
            "display.width",
            320,
            "display.max_rows",
            40,
        ):
            cols = [
                col
                for col in [
                    "bundle_type",
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "table_index",
                    "bundle_label_class",
                    "revenue_labels",
                    "operating_income_labels",
                    "net_income_labels",
                    "revenue_magnitude_suspect",
                    "signature_text",
                    "main_context",
                ]
                if col in suspect_signatures.columns
            ]

            print(
                suspect_signatures[
                    cols
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
        f"- Signature label audit : "
        f"{OUT_SIGNATURE_LABELS}"
    )

    print(
        f"- Conflict ranking audit: "
        f"{OUT_CONFLICT_RANKING}"
    )

    print(
        f"- Group summary         : "
        f"{OUT_GROUP_SUMMARY}"
    )

    print(
        "\n해석 원칙:"
        "\n- ranking은 shadow audit일 뿐 production 선택이 아님"
        "\n- basis > owner > total-label quality > context specificity 순의 lexicographic evidence"
        "\n- 기타영업수익 같은 부분계정은 revenue total로 자동 인정하지 않음"
        "\n- financial-sector revenue는 별도 domain review가 필요하므로 불명확 label을 억지로 탈락시키지 않음"
        "\n- tied top은 그대로 unresolved"
    )


if __name__ == "__main__":
    main()
