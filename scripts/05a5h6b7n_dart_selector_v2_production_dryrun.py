from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7N. Selector V2 Production Dry-Run
#
# 목적
# ------------------------------------------------------------
# H6B7A~M에서 만든 모든 semantic guardrail을 합쳐
# "실제 production merge 직전" 자동채택 가능 범위를 계산한다.
#
# 아직 원본 wide dataset을 수정하지 않는다.
#
# 핵심:
# 1) economic signature cluster 단위로 중복표 collapse
# 2) basis contradiction / other-entity owner / partial label 차단
# 3) review label / magnitude sanity는 자동채택하지 않음
# 4) genuine multiple clusters는 H6B7M lexicographic evidence로 비교
# 5) tie는 unresolved 유지
# 6) economic value가 하나여도 exact source table이 tie면
#    "economic-ready"와 "source-ready"를 구분
#
# 자동채택 우선순위(lexicographic):
#   basis strength
#   > issuer/group owner evidence
#   > total-account label quality
#   > statement context specificity
#
# IMPORTANT
# ------------------------------------------------------------
# - final production selection 아님
# - 원본 noncorrected wide 파일 수정 안 함
# - tie에서 table_index 작은 것을 임의 선택하지 않음
# - suspect/review label을 억지로 채택하지 않음
#
# 실행:
# python scripts\05a5h6b7n_dart_selector_v2_production_dryrun.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TABLE_SEMANTICS = (
    INTERIM
    / "dart_noncorrected_nodata_table_refined_semantics.csv"
)

LABEL_AUDIT = (
    INTERIM
    / "dart_noncorrected_nodata_signature_account_label_audit.csv"
)

OUT_SIGNATURES = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_signature_gate.csv"
)

OUT_CLUSTERS = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_cluster_summary.csv"
)

OUT_BUNDLES = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_bundle_summary.csv"
)

OUT_RECEIPTS = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_receipt_summary.csv"
)

OUT_REVIEW = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_review_queue.csv"
)


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


def as_bool(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
    }


# ============================================================
# Evidence ranking (same philosophy as H6B7M)
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

    if selector_basis == "CFS":

        if refined_basis == "CFS":
            return 4

        if refined_basis == "CFS_ENTITY_HINT":
            return 3

        if refined_basis == "UNKNOWN":
            return 2

        return 0

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

    return {
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
    }.get(
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

    if bundle_type == "BALANCE":

        if any(
            marker in text
            for marker in [
                "연결재무상태표",
                "요약연결재무상태표",
                "별도재무상태표",
                "요약별도재무상태표",
            ]
        ):
            return 4

        if any(
            marker in text
            for marker in [
                "요약연결재무정보",
                "요약별도재무정보",
            ]
        ):
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
                "별도포괄손익계산서",
                "별도손익계산서",
            ]
        ):
            return 4

        if any(
            marker in text
            for marker in [
                "요약연결재무정보",
                "요약별도재무정보",
            ]
        ):
            return 3

        if "요약재무정보" in text:
            return 1

        return 0

    return 0


# ============================================================
# Signature semantic gate
# ============================================================


def signature_gate(row: pd.Series) -> tuple[str, str]:

    if as_bool(
        row.get(
            "refined_basis_contradiction",
            False,
        )
    ):
        return (
            "BLOCK",
            "basis_contradiction",
        )

    entity_scope = str(
        row.get(
            "refined_entity_scope",
            "",
        )
    )

    if (
        entity_scope
        == "OTHER_ENTITY_OWNER_EXPLICIT"
    ):
        return (
            "BLOCK",
            "other_entity_owner",
        )

    refined_basis = str(
        row.get(
            "refined_local_basis",
            "",
        )
    )

    if refined_basis == "MIXED_MAIN":
        return (
            "REVIEW",
            "mixed_main_basis",
        )

    if (
        entity_scope
        == "MAIN_OTHER_COMPANY_AMBIGUOUS"
    ):
        return (
            "REVIEW",
            "entity_owner_ambiguous",
        )

    label_class = str(
        row.get(
            "bundle_label_class",
            "",
        )
    )

    if (
        label_class
        == "HAS_SUSPECT_PARTIAL_OR_ATTRIBUTABLE_LABEL"
    ):
        return (
            "BLOCK",
            "suspect_partial_or_attributable_label",
        )

    if (
        label_class
        == "LABEL_REVIEW_REQUIRED"
    ):
        return (
            "REVIEW",
            "account_label_review",
        )

    if as_bool(
        row.get(
            "revenue_magnitude_suspect",
            False,
        )
    ):
        return (
            "REVIEW",
            "revenue_magnitude_sanity",
        )

    if (
        label_class
        == "ALL_STRONG_TOTAL_LABELS"
    ):
        return (
            "KEEP",
            "strong_semantics",
        )

    return (
        "REVIEW",
        "unclassified_label_semantics",
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        TABLE_SEMANTICS,
        LABEL_AUDIT,
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

    labels = pd.read_csv(
        LABEL_AUDIT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    for frame in [
        tables,
        labels,
    ]:
        frame[
            "rcept_no"
        ] = normalize_receipt(
            frame[
                "rcept_no"
            ]
        )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7N SELECTOR V2 PRODUCTION DRY-RUN"
    )

    print(
        "=" * 120
    )

    print(
        f"\nTable semantic rows : "
        f"{len(tables):,}"
    )

    print(
        f"Label audit rows    : "
        f"{len(labels):,}"
    )

    # --------------------------------------------------------
    # 1. Merge refined semantics + label semantics
    # --------------------------------------------------------

    label_cols = [
        "bundle_type",
        "rcept_no",
        "table_index",
        "bundle_label_class",
        "bundle_label_min_rank",
        "bundle_label_sum_rank",
        "revenue_magnitude_suspect",
    ]

    missing = [
        col
        for col in label_cols
        if col not in labels.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing label-audit columns: "
            + ", ".join(
                missing
            )
        )

    work = tables.merge(
        labels[
            label_cols
        ],
        on=[
            "bundle_type",
            "rcept_no",
            "table_index",
        ],
        how="left",
        validate="one_to_one",
    )

    gates = work.apply(
        signature_gate,
        axis=1,
    )

    work[
        "dryrun_signature_gate"
    ] = [
        item[
            0
        ]
        for item in gates
    ]

    work[
        "dryrun_signature_reason"
    ] = [
        item[
            1
        ]
        for item in gates
    ]

    work[
        "dryrun_basis_strength"
    ] = [
        basis_strength(
            selector,
            refined,
        )
        for selector, refined
        in zip(
            work[
                "selector_table_basis"
            ],
            work[
                "refined_local_basis"
            ],
        )
    ]

    work[
        "dryrun_entity_strength"
    ] = work[
        "refined_entity_scope"
    ].map(
        entity_strength
    )

    work[
        "dryrun_label_strength"
    ] = pd.to_numeric(
        work[
            "bundle_label_min_rank"
        ],
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    )

    work[
        "dryrun_context_strength"
    ] = [
        context_specificity(
            bundle_type,
            main_context,
        )
        for bundle_type, main_context
        in zip(
            work[
                "bundle_type"
            ],
            work[
                "main_context"
            ],
        )
    ]

    work[
        "dryrun_rank_tuple"
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
            work[
                "dryrun_basis_strength"
            ],
            work[
                "dryrun_entity_strength"
            ],
            work[
                "dryrun_label_strength"
            ],
            work[
                "dryrun_context_strength"
            ],
        )
    ]

    work[
        "dryrun_rank_tuple_text"
    ] = work[
        "dryrun_rank_tuple"
    ].map(
        lambda x:
        "-".join(
            str(
                value
            )
            for value in x
        )
    )

    work.to_csv(
        OUT_SIGNATURES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 2. Economic signature cluster-level gate
    # --------------------------------------------------------

    cluster_records = []

    for (
        rcept_no,
        bundle_type,
        signature_cluster,
    ), group in work.groupby(
        [
            "rcept_no",
            "bundle_type",
            "signature_cluster",
        ],
        sort=False,
        dropna=False,
    ):

        keep = group.loc[
            group[
                "dryrun_signature_gate"
            ].eq(
                "KEEP"
            )
        ].copy()

        review = group.loc[
            group[
                "dryrun_signature_gate"
            ].eq(
                "REVIEW"
            )
        ].copy()

        blocked = group.loc[
            group[
                "dryrun_signature_gate"
            ].eq(
                "BLOCK"
            )
        ].copy()

        if not keep.empty:

            best_tuple = max(
                keep[
                    "dryrun_rank_tuple"
                ].tolist()
            )

            top = keep.loc[
                keep[
                    "dryrun_rank_tuple"
                ].map(
                    lambda value:
                    value
                    == best_tuple
                )
            ].copy()

            if len(
                top
            ) == 1:
                source_status = (
                    "UNIQUE_SOURCE_TOP"
                )

                representative = (
                    top.iloc[
                        0
                    ]
                )

            else:
                source_status = (
                    "TIED_SOURCE_WITHIN_EQUIVALENT_CLUSTER"
                )

                representative = None

            cluster_gate = (
                "KEEP_CLUSTER"
            )

            cluster_reason = (
                "has_clean_signature_source"
            )

        elif not review.empty:

            best_tuple = None
            top = review.iloc[
                0:0
            ]

            source_status = (
                "REVIEW_SOURCE"
            )

            representative = None

            cluster_gate = (
                "REVIEW_CLUSTER"
            )

            cluster_reason = "|".join(
                sorted(
                    set(
                        review[
                            "dryrun_signature_reason"
                        ].astype(str)
                    )
                )
            )

        else:

            best_tuple = None
            top = blocked.iloc[
                0:0
            ]

            source_status = (
                "NO_VALID_SOURCE"
            )

            representative = None

            cluster_gate = (
                "BLOCK_CLUSTER"
            )

            cluster_reason = "|".join(
                sorted(
                    set(
                        blocked[
                            "dryrun_signature_reason"
                        ].astype(str)
                    )
                )
            )

        first = group.iloc[
            0
        ]

        cluster_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                first.get(
                    "stock_code",
                    "",
                ),

                "period_key":
                first.get(
                    "period_key",
                    "",
                ),

                "bundle_type":
                bundle_type,

                "signature_cluster":
                signature_cluster,

                "signature_member_table_count":
                len(
                    group
                ),

                "keep_source_count":
                len(
                    keep
                ),

                "review_source_count":
                len(
                    review
                ),

                "blocked_source_count":
                len(
                    blocked
                ),

                "cluster_gate":
                cluster_gate,

                "cluster_reason":
                cluster_reason,

                "cluster_source_status":
                source_status,

                "best_rank_tuple":
                (
                    "-".join(
                        str(
                            value
                        )
                        for value in best_tuple
                    )
                    if best_tuple
                    is not None
                    else ""
                ),

                "top_source_table_count":
                len(
                    top
                ),

                "top_source_tables":
                "|".join(
                    top[
                        "table_index"
                    ]
                    .astype(str)
                    .tolist()
                ),

                "representative_table_index":
                (
                    representative[
                        "table_index"
                    ]
                    if representative
                    is not None
                    else np.nan
                ),

                "v1":
                (
                    representative.get(
                        "v1",
                        np.nan,
                    )
                    if representative
                    is not None
                    else np.nan
                ),

                "v2":
                (
                    representative.get(
                        "v2",
                        np.nan,
                    )
                    if representative
                    is not None
                    else np.nan
                ),

                "v3":
                (
                    representative.get(
                        "v3",
                        np.nan,
                    )
                    if representative
                    is not None
                    else np.nan
                ),

                "signature_text":
                first.get(
                    "signature_text",
                    "",
                ),
            }
        )

    clusters = pd.DataFrame(
        cluster_records
    )

    clusters.to_csv(
        OUT_CLUSTERS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3. Bundle-level resolution across economic clusters
    # --------------------------------------------------------

    bundle_records = []

    for (
        rcept_no,
        bundle_type,
    ), group in clusters.groupby(
        [
            "rcept_no",
            "bundle_type",
        ],
        sort=False,
    ):

        keep_clusters = group.loc[
            group[
                "cluster_gate"
            ].eq(
                "KEEP_CLUSTER"
            )
        ].copy()

        review_clusters = group.loc[
            group[
                "cluster_gate"
            ].eq(
                "REVIEW_CLUSTER"
            )
        ].copy()

        blocked_clusters = group.loc[
            group[
                "cluster_gate"
            ].eq(
                "BLOCK_CLUSTER"
            )
        ].copy()

        selected_cluster = None
        selected_status = ""
        selected_reason = ""
        source_ready = False

        if len(
            keep_clusters
        ) == 0:

            if len(
                review_clusters
            ) > 0:
                selected_status = (
                    "REVIEW_ONLY"
                )

                selected_reason = (
                    "no_clean_cluster_but_review_exists"
                )

            else:
                selected_status = (
                    "NO_VALID_CLUSTER"
                )

                selected_reason = (
                    "all_clusters_blocked"
                )

        elif len(
            keep_clusters
        ) == 1:

            selected_cluster = (
                keep_clusters.iloc[
                    0
                ]
            )

            selected_status = (
                "ECONOMIC_CLUSTER_UNIQUE"
            )

            selected_reason = (
                "one_clean_economic_signature"
            )

        else:
            rank_values = []

            for _, cluster_row in (
                keep_clusters.iterrows()
            ):

                text = str(
                    cluster_row[
                        "best_rank_tuple"
                    ]
                ).strip()

                if text:
                    rank_tuple = tuple(
                        int(
                            piece
                        )
                        for piece in text.split(
                            "-"
                        )
                    )
                else:
                    rank_tuple = (
                        0,
                        0,
                        0,
                        0,
                    )

                rank_values.append(
                    (
                        cluster_row.name,
                        rank_tuple,
                    )
                )

            best = max(
                rank_tuple
                for _, rank_tuple
                in rank_values
            )

            top_indices = [
                idx
                for idx, rank_tuple
                in rank_values
                if rank_tuple == best
            ]

            if len(
                top_indices
            ) == 1:

                selected_cluster = (
                    keep_clusters.loc[
                        top_indices[
                            0
                        ]
                    ]
                )

                selected_status = (
                    "ECONOMIC_CLUSTER_UNIQUE_RANKED"
                )

                selected_reason = (
                    "multiple_signatures_unique_semantic_top"
                )

            else:
                selected_status = (
                    "ECONOMIC_CLUSTER_TIED"
                )

                selected_reason = (
                    "multiple_signatures_same_top_evidence"
                )

        if selected_cluster is not None:

            source_ready = (
                selected_cluster[
                    "cluster_source_status"
                ]
                == "UNIQUE_SOURCE_TOP"
            )

            selected_signature_cluster = (
                selected_cluster[
                    "signature_cluster"
                ]
            )

            selected_table = (
                selected_cluster[
                    "representative_table_index"
                ]
                if source_ready
                else np.nan
            )

            v1 = (
                selected_cluster[
                    "v1"
                ]
                if source_ready
                else np.nan
            )

            v2 = (
                selected_cluster[
                    "v2"
                ]
                if source_ready
                else np.nan
            )

            v3 = (
                selected_cluster[
                    "v3"
                ]
                if source_ready
                else np.nan
            )

        else:
            selected_signature_cluster = np.nan
            selected_table = np.nan
            v1 = np.nan
            v2 = np.nan
            v3 = np.nan

        first = group.iloc[
            0
        ]

        bundle_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                first.get(
                    "stock_code",
                    "",
                ),

                "period_key":
                first.get(
                    "period_key",
                    "",
                ),

                "bundle_type":
                bundle_type,

                "economic_cluster_count":
                len(
                    group
                ),

                "clean_cluster_count":
                len(
                    keep_clusters
                ),

                "review_cluster_count":
                len(
                    review_clusters
                ),

                "blocked_cluster_count":
                len(
                    blocked_clusters
                ),

                "bundle_resolution_status":
                selected_status,

                "bundle_resolution_reason":
                selected_reason,

                "bundle_economic_ready":
                selected_cluster
                is not None,

                "bundle_source_ready":
                source_ready,

                "selected_signature_cluster":
                selected_signature_cluster,

                "selected_table_index":
                selected_table,

                "selected_v1":
                v1,

                "selected_v2":
                v2,

                "selected_v3":
                v3,
            }
        )

    bundles = pd.DataFrame(
        bundle_records
    )

    bundles.to_csv(
        OUT_BUNDLES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4. Receipt-level core6 readiness
    # --------------------------------------------------------

    receipt_records = []

    for rcept_no, group in bundles.groupby(
        "rcept_no",
        sort=False,
    ):

        bal = group.loc[
            group[
                "bundle_type"
            ].eq(
                "BALANCE"
            )
        ]

        inc = group.loc[
            group[
                "bundle_type"
            ].eq(
                "INCOME"
            )
        ]

        bal_row = (
            bal.iloc[
                0
            ]
            if not bal.empty
            else None
        )

        inc_row = (
            inc.iloc[
                0
            ]
            if not inc.empty
            else None
        )

        balance_economic_ready = bool(
            bal_row is not None
            and bal_row[
                "bundle_economic_ready"
            ]
        )

        income_economic_ready = bool(
            inc_row is not None
            and inc_row[
                "bundle_economic_ready"
            ]
        )

        balance_source_ready = bool(
            bal_row is not None
            and bal_row[
                "bundle_source_ready"
            ]
        )

        income_source_ready = bool(
            inc_row is not None
            and inc_row[
                "bundle_source_ready"
            ]
        )

        core6_economic_ready = (
            balance_economic_ready
            and income_economic_ready
        )

        core6_source_ready = (
            balance_source_ready
            and income_source_ready
        )

        sample_row = (
            group.iloc[
                0
            ]
        )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                sample_row.get(
                    "stock_code",
                    "",
                ),

                "period_key":
                sample_row.get(
                    "period_key",
                    "",
                ),

                "balance_resolution_status":
                (
                    bal_row[
                        "bundle_resolution_status"
                    ]
                    if bal_row
                    is not None
                    else "NO_BALANCE_BUNDLE"
                ),

                "income_resolution_status":
                (
                    inc_row[
                        "bundle_resolution_status"
                    ]
                    if inc_row
                    is not None
                    else "NO_INCOME_BUNDLE"
                ),

                "balance_economic_ready":
                balance_economic_ready,

                "income_economic_ready":
                income_economic_ready,

                "balance_source_ready":
                balance_source_ready,

                "income_source_ready":
                income_source_ready,

                "core6_economic_ready":
                core6_economic_ready,

                "core6_source_ready":
                core6_source_ready,

                "balance_selected_table":
                (
                    bal_row[
                        "selected_table_index"
                    ]
                    if bal_row
                    is not None
                    else np.nan
                ),

                "income_selected_table":
                (
                    inc_row[
                        "selected_table_index"
                    ]
                    if inc_row
                    is not None
                    else np.nan
                ),

                "assets":
                (
                    bal_row[
                        "selected_v1"
                    ]
                    if bal_row
                    is not None
                    else np.nan
                ),

                "liabilities":
                (
                    bal_row[
                        "selected_v2"
                    ]
                    if bal_row
                    is not None
                    else np.nan
                ),

                "equity":
                (
                    bal_row[
                        "selected_v3"
                    ]
                    if bal_row
                    is not None
                    else np.nan
                ),

                "revenue":
                (
                    inc_row[
                        "selected_v1"
                    ]
                    if inc_row
                    is not None
                    else np.nan
                ),

                "operating_income":
                (
                    inc_row[
                        "selected_v2"
                    ]
                    if inc_row
                    is not None
                    else np.nan
                ),

                "net_income":
                (
                    inc_row[
                        "selected_v3"
                    ]
                    if inc_row
                    is not None
                    else np.nan
                ),
            }
        )

    receipts = pd.DataFrame(
        receipt_records
    )

    receipts.to_csv(
        OUT_RECEIPTS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5. Review queue
    # --------------------------------------------------------

    review_signatures = work.loc[
        work[
            "dryrun_signature_gate"
        ].ne(
            "KEEP"
        )
    ].copy()

    unresolved_bundles = bundles.loc[
        ~bundles[
            "bundle_economic_ready"
        ]
        | (
            bundles[
                "bundle_economic_ready"
            ]
            & ~bundles[
                "bundle_source_ready"
            ]
        )
    ].copy()

    review_parts = []

    if not review_signatures.empty:

        temp = review_signatures.copy()

        temp[
            "review_level"
        ] = "SIGNATURE"

        temp[
            "review_status"
        ] = temp[
            "dryrun_signature_gate"
        ]

        temp[
            "review_reason"
        ] = temp[
            "dryrun_signature_reason"
        ]

        review_parts.append(
            temp[
                [
                    "review_level",
                    "rcept_no",
                    "stock_code",
                    "period_key",
                    "bundle_type",
                    "table_index",
                    "signature_cluster",
                    "review_status",
                    "review_reason",
                    "signature_text",
                    "main_context",
                ]
            ]
        )

    if not unresolved_bundles.empty:

        temp = unresolved_bundles.copy()

        temp[
            "review_level"
        ] = "BUNDLE"

        temp[
            "table_index"
        ] = temp[
            "selected_table_index"
        ]

        temp[
            "signature_cluster"
        ] = temp[
            "selected_signature_cluster"
        ]

        temp[
            "review_status"
        ] = temp[
            "bundle_resolution_status"
        ]

        temp[
            "review_reason"
        ] = temp[
            "bundle_resolution_reason"
        ]

        temp[
            "signature_text"
        ] = ""

        temp[
            "main_context"
        ] = ""

        review_parts.append(
            temp[
                [
                    "review_level",
                    "rcept_no",
                    "stock_code",
                    "period_key",
                    "bundle_type",
                    "table_index",
                    "signature_cluster",
                    "review_status",
                    "review_reason",
                    "signature_text",
                    "main_context",
                ]
            ]
        )

    if review_parts:
        review_queue = pd.concat(
            review_parts,
            ignore_index=True,
        )
    else:
        review_queue = pd.DataFrame()

    review_queue.to_csv(
        OUT_REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Signature semantic gate]"
    )

    print(
        work[
            "dryrun_signature_gate"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Signature gate reason]"
    )

    print(
        work[
            "dryrun_signature_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Economic cluster gate]"
    )

    print(
        clusters[
            "cluster_gate"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Cluster representative source status]"
    )

    print(
        clusters[
            "cluster_source_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Bundle resolution status]"
    )

    print(
        pd.crosstab(
            bundles[
                "bundle_type"
            ],
            bundles[
                "bundle_resolution_status"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Bundle economic-ready]"
    )

    print(
        bundles.groupby(
            "bundle_type"
        )[
            "bundle_economic_ready"
        ]
        .sum()
        .to_string()
    )

    print(
        "\n[Bundle source-ready]"
    )

    print(
        bundles.groupby(
            "bundle_type"
        )[
            "bundle_source_ready"
        ]
        .sum()
        .to_string()
    )

    print(
        "\n[Core6 economic-ready receipts]"
    )

    print(
        int(
            receipts[
                "core6_economic_ready"
            ].sum()
        )
    )

    print(
        "\n[Core6 source-ready receipts]"
    )

    print(
        int(
            receipts[
                "core6_source_ready"
            ].sum()
        )
    )

    print(
        "\n[Receipt resolution combinations]"
    )

    print(
        receipts[
            [
                "balance_resolution_status",
                "income_resolution_status",
            ]
        ]
        .value_counts()
        .head(
            30
        )
        .to_string()
    )

    print(
        "\n[Unresolved economic bundle sample]"
    )

    unresolved_economic = bundles.loc[
        ~bundles[
            "bundle_economic_ready"
        ]
    ]

    if unresolved_economic.empty:
        print(
            "None"
        )
    else:
        print(
            unresolved_economic[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "bundle_type",
                    "economic_cluster_count",
                    "clean_cluster_count",
                    "review_cluster_count",
                    "blocked_cluster_count",
                    "bundle_resolution_status",
                    "bundle_resolution_reason",
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
        "\n[Economic-ready but exact source tied sample]"
    )

    source_tied = bundles.loc[
        bundles[
            "bundle_economic_ready"
        ]
        & ~bundles[
            "bundle_source_ready"
        ]
    ]

    if source_tied.empty:
        print(
            "None"
        )
    else:
        print(
            source_tied[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "bundle_type",
                    "selected_signature_cluster",
                    "bundle_resolution_status",
                    "bundle_resolution_reason",
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
        "\n[Core6 source-ready sample]"
    )

    ready = receipts.loc[
        receipts[
            "core6_source_ready"
        ]
    ]

    if ready.empty:
        print(
            "None"
        )
    else:
        print(
            ready[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "balance_selected_table",
                    "income_selected_table",
                    "assets",
                    "liabilities",
                    "equity",
                    "revenue",
                    "operating_income",
                    "net_income",
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
        f"- Signature gate : "
        f"{OUT_SIGNATURES}"
    )

    print(
        f"- Cluster summary: "
        f"{OUT_CLUSTERS}"
    )

    print(
        f"- Bundle summary : "
        f"{OUT_BUNDLES}"
    )

    print(
        f"- Receipt summary: "
        f"{OUT_RECEIPTS}"
    )

    print(
        f"- Review queue   : "
        f"{OUT_REVIEW}"
    )

    print(
        "\n해석 원칙:"
        "\n- core6_economic_ready = 경제적 signature는 balance/income 모두 하나로 결정"
        "\n- core6_source_ready = exact source table까지 balance/income 모두 유일"
        "\n- partial revenue / attributable-only label은 자동채택하지 않음"
        "\n- review label, magnitude sanity, semantic tie는 unresolved 유지"
        "\n- 이 결과가 안정적이면 다음 단계에서만 production wide merge 수행"
    )


if __name__ == "__main__":
    main()
