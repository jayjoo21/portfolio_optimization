from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7K. Genuine-Conflict Basis + Entity-Scope Audit
#
# 목적
# ------------------------------------------------------------
# H6B7J에서 duplicate-equivalent를 제거하고 남은
# genuine bundle conflicts를 바로 ranking하지 않는다.
#
# 먼저:
#   1) table local context 기준 CFS/OFS 재검증
#   2) selector_table_basis와 explicit local basis 충돌 탐지
#   3) 가능하면 issuer name을 scope에서 찾아
#      해당 table이 issuer/연결실체 표인지 다른 법인 표인지 진단
#
# IMPORTANT
# ------------------------------------------------------------
# - final selection 아님
# - ZIP 재파싱 없음
# - local basis가 UNKNOWN이면 억지 분류하지 않음
# - entity scope도 issuer name을 찾을 수 없으면 UNKNOWN 유지
#
# 실행:
# python scripts\05a5h6b7k_dart_conflict_basis_entity_scope_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CONFLICTS = (
    INTERIM
    / "dart_noncorrected_nodata_bundle_genuine_conflict_rows.csv"
)

SIGNATURES = (
    INTERIM
    / "dart_noncorrected_nodata_bundle_equivalence_table_signatures.csv"
)

SCOPE = (
    INTERIM
    / "dart_noncorrected_full_api_no_data_scope.csv"
)

OUT_CONFLICT = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_basis_entity_scope_audit.csv"
)

OUT_TABLE = (
    INTERIM
    / "dart_noncorrected_nodata_table_local_basis_audit.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_receipt_audit_summary.csv"
)


# ============================================================
# Text helpers
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
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        clean(value).lower(),
    )


def normalize_company(value) -> str:
    text = clean(value).lower()

    replacements = [
        "주식회사",
        "(주)",
        "㈜",
        "유한회사",
        "유한책임회사",
        "co.,ltd.",
        "co.ltd.",
        "corp.",
        "corporation",
        "inc.",
        "inc",
    ]

    for token in replacements:
        text = text.replace(
            token,
            "",
        )

    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        text,
    )


# ============================================================
# Local basis audit
# ============================================================


STRONG_OFS_PATTERNS = [
    "요약별도재무정보",
    "요약개별재무정보",
    "별도재무정보",
    "개별재무정보",
    "별도재무제표",
    "개별재무제표",
    "별도재무상태표",
    "별도손익계산서",
    "별도포괄손익계산서",
    "별도기준",
    "개별기준",
]

STRONG_CFS_PATTERNS = [
    "요약연결재무정보",
    "요약연결재무제표",
    "요약연결재무상태표",
    "연결재무상태표",
    "연결손익계산서",
    "연결포괄손익계산서",
    "연결재무제표",
    "연결기준",
]

NO_CFS_PATTERNS = [
    "연결대상종속기업이존재하지않",
    "연결대상종속기업은존재하지않",
    "연결대상종속기업이없",
    "연결재무제표를작성하지않",
    "연결재무제표작성의무가없",
]


def contains_any(
    text: str,
    patterns: list[str],
) -> list[str]:

    return [
        pattern
        for pattern in patterns
        if pattern in text
    ]


def local_basis(context) -> tuple[str, str]:

    text = compact(
        context
    )

    no_cfs_hits = contains_any(
        text,
        NO_CFS_PATTERNS,
    )

    ofs_hits = contains_any(
        text,
        STRONG_OFS_PATTERNS,
    )

    cfs_hits = contains_any(
        text,
        STRONG_CFS_PATTERNS,
    )

    if no_cfs_hits:
        return (
            "OFS_ONLY_NO_CFS",
            "NO_CFS:"
            + ",".join(
                no_cfs_hits
            ),
        )

    if ofs_hits and not cfs_hits:
        return (
            "OFS",
            "OFS:"
            + ",".join(
                ofs_hits
            ),
        )

    if cfs_hits and not ofs_hits:
        return (
            "CFS",
            "CFS:"
            + ",".join(
                cfs_hits
            ),
        )

    if ofs_hits and cfs_hits:
        return (
            "MIXED_EXPLICIT",
            (
                "OFS:"
                + ",".join(
                    ofs_hits
                )
                + " | CFS:"
                + ",".join(
                    cfs_hits
                )
            ),
        )

    # Company + subsidiaries is useful support, but not strong enough
    # to override explicit local titles.
    if (
        "와그종속기업" in text
        or "및그종속기업" in text
        or "와그종속회사" in text
        or "및그종속회사" in text
    ):
        return (
            "CFS_ENTITY_HINT",
            "company_and_subsidiaries",
        )

    return (
        "UNKNOWN",
        "",
    )


def selector_basis_family(
    value: str,
) -> str:

    value = str(
        value
    )

    if value == "CFS":
        return "CFS"

    if value in {
        "OFS",
        "OFS_ONLY_NO_CFS",
    }:
        return "OFS"

    return value


def local_basis_family(
    value: str,
) -> str:

    value = str(
        value
    )

    if value in {
        "CFS",
        "CFS_ENTITY_HINT",
    }:
        return "CFS"

    if value in {
        "OFS",
        "OFS_ONLY_NO_CFS",
    }:
        return "OFS"

    return value


# ============================================================
# Issuer name discovery
# ============================================================


ISSUER_NAME_CANDIDATES = [
    "corp_name",
    "corp_nm",
    "company_name",
    "stock_name",
    "stock_nm",
    "name",
    "종목명",
    "회사명",
    "법인명",
]


def discover_name_column(
    scope: pd.DataFrame,
):

    for col in ISSUER_NAME_CANDIDATES:
        if col in scope.columns:
            nonempty = (
                scope[
                    col
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if nonempty.ne(
                ""
            ).any():
                return col

    return None


def build_issuer_map(
    scope: pd.DataFrame,
):

    name_col = discover_name_column(
        scope
    )

    if name_col is None:
        return (
            {},
            None,
        )

    if "rcept_no" not in scope.columns:
        return (
            {},
            name_col,
        )

    work = scope[
        [
            "rcept_no",
            name_col,
        ]
    ].copy()

    work[
        "rcept_no"
    ] = (
        work[
            "rcept_no"
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    work = (
        work.dropna(
            subset=[
                name_col,
            ]
        )
        .drop_duplicates(
            "rcept_no"
        )
    )

    return (
        dict(
            zip(
                work[
                    "rcept_no"
                ],
                work[
                    name_col
                ].astype(str),
            )
        ),
        name_col,
    )


# ============================================================
# Entity extraction / scope audit
# ============================================================


CORP_PATTERNS = [
    re.compile(
        r"주식회사\s*([가-힣A-Za-z0-9ㆍ&]+)"
    ),
    re.compile(
        r"([가-힣A-Za-z0-9ㆍ&]+)\s*주식회사"
    ),
    re.compile(
        r"\(주\)\s*([가-힣A-Za-z0-9ㆍ&]+)"
    ),
]


OTHER_ENTITY_SECTION_PATTERNS = [
    "최대주주",
    "관계기업",
    "공동기업",
    "종속기업요약재무정보",
    "종속기업의요약재무정보",
    "주요종속기업",
]


def extract_named_companies(
    context,
) -> list[str]:

    text = clean(
        context
    )

    found = []

    for pattern in CORP_PATTERNS:
        for match in pattern.findall(
            text
        ):
            name = clean(
                match
            )

            if name and name not in found:
                found.append(
                    name
                )

    return found


def entity_scope_class(
    context,
    issuer_name,
) -> tuple[str, str, str]:

    text = clean(
        context
    )

    text_compact = compact(
        text
    )

    issuer_norm = normalize_company(
        issuer_name
    )

    companies = extract_named_companies(
        text
    )

    company_norms = [
        (
            name,
            normalize_company(
                name
            ),
        )
        for name in companies
    ]

    issuer_mentioned = bool(
        issuer_norm
        and issuer_norm in normalize_company(
            text
        )
    )

    other_named = [
        name
        for name, norm
        in company_norms
        if norm
        and (
            not issuer_norm
            or (
                norm not in issuer_norm
                and issuer_norm not in norm
            )
        )
    ]

    other_section_hits = [
        pattern
        for pattern in OTHER_ENTITY_SECTION_PATTERNS
        if pattern in text_compact
    ]

    if not issuer_norm:
        if other_section_hits:
            return (
                "OTHER_ENTITY_SECTION_SUSPECT",
                "|".join(
                    other_section_hits
                ),
                "|".join(
                    companies
                ),
            )

        return (
            "ISSUER_NAME_UNAVAILABLE",
            "",
            "|".join(
                companies
            ),
        )

    if issuer_mentioned:
        return (
            "ISSUER_OR_GROUP_MENTION",
            "",
            "|".join(
                companies
            ),
        )

    if other_section_hits:
        return (
            "OTHER_ENTITY_SECTION_SUSPECT",
            "|".join(
                other_section_hits
            ),
            "|".join(
                companies
            ),
        )

    if other_named:
        return (
            "OTHER_NAMED_COMPANY_SUSPECT",
            "",
            "|".join(
                other_named
            ),
        )

    return (
        "ENTITY_SCOPE_UNKNOWN",
        "",
        "|".join(
            companies
        ),
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        CONFLICTS,
        SIGNATURES,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    conflicts = pd.read_csv(
        CONFLICTS,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    signatures = pd.read_csv(
        SIGNATURES,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    if SCOPE.exists():
        scope = pd.read_csv(
            SCOPE,
            dtype={
                "rcept_no":
                str,

                "stock_code":
                str,
            },
            low_memory=False,
        )
    else:
        scope = pd.DataFrame()

    issuer_map, issuer_name_col = (
        build_issuer_map(
            scope
        )
        if not scope.empty
        else (
            {},
            None,
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7K GENUINE-CONFLICT BASIS + ENTITY-SCOPE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nConflict rows : "
        f"{len(conflicts):,}"
    )

    print(
        f"Signature rows: "
        f"{len(signatures):,}"
    )

    print(
        "Issuer name column: "
        + (
            issuer_name_col
            if issuer_name_col
            else "NOT FOUND"
        )
    )

    # --------------------------------------------------------
    # Audit ALL signature tables first.
    # --------------------------------------------------------

    audited = signatures.copy()

    local = audited[
        "heading_context"
    ].apply(
        lambda x:
        local_basis(
            x
        )
    )

    audited[
        "local_basis"
    ] = [
        item[
            0
        ]
        for item in local
    ]

    audited[
        "local_basis_reason"
    ] = [
        item[
            1
        ]
        for item in local
    ]

    audited[
        "selector_basis_family"
    ] = audited[
        "selector_table_basis"
    ].map(
        selector_basis_family
    )

    audited[
        "local_basis_family"
    ] = audited[
        "local_basis"
    ].map(
        local_basis_family
    )

    audited[
        "basis_explicit_contradiction"
    ] = (
        audited[
            "local_basis_family"
        ].isin(
            [
                "CFS",
                "OFS",
            ]
        )
        & audited[
            "selector_basis_family"
        ].isin(
            [
                "CFS",
                "OFS",
            ]
        )
        & audited[
            "local_basis_family"
        ].ne(
            audited[
                "selector_basis_family"
            ]
        )
    )

    audited[
        "issuer_name"
    ] = audited[
        "rcept_no"
    ].astype(str).map(
        issuer_map
    )

    entity_results = [
        entity_scope_class(
            context,
            issuer,
        )
        for context, issuer
        in zip(
            audited[
                "heading_context"
            ],
            audited[
                "issuer_name"
            ],
        )
    ]

    audited[
        "entity_scope_class"
    ] = [
        x[
            0
        ]
        for x in entity_results
    ]

    audited[
        "entity_scope_reason"
    ] = [
        x[
            1
        ]
        for x in entity_results
    ]

    audited[
        "named_companies"
    ] = [
        x[
            2
        ]
        for x in entity_results
    ]

    audited.to_csv(
        OUT_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Join audit back to genuine conflict rows.
    # --------------------------------------------------------

    key_cols = [
        "bundle_type",
        "rcept_no",
        "table_index",
    ]

    audit_cols = [
        "bundle_type",
        "rcept_no",
        "table_index",
        "issuer_name",
        "local_basis",
        "local_basis_reason",
        "selector_basis_family",
        "local_basis_family",
        "basis_explicit_contradiction",
        "entity_scope_class",
        "entity_scope_reason",
        "named_companies",
    ]

    conflict_audit = conflicts.merge(
        audited[
            audit_cols
        ],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    conflict_audit[
        "conflict_semantic_gate"
    ] = np.select(
        [
            conflict_audit[
                "basis_explicit_contradiction"
            ].fillna(
                False
            ),

            conflict_audit[
                "entity_scope_class"
            ].isin(
                [
                    "OTHER_ENTITY_SECTION_SUSPECT",
                    "OTHER_NAMED_COMPANY_SUSPECT",
                ]
            ),

            conflict_audit[
                "local_basis"
            ].eq(
                "MIXED_EXPLICIT"
            ),
        ],
        [
            "BLOCK_BASIS_CONTRADICTION",
            "BLOCK_OTHER_ENTITY_SUSPECT",
            "REVIEW_LOCAL_BASIS_MIXED",
        ],
        default=(
            "KEEP_FOR_RANKING"
        ),
    )

    conflict_audit.to_csv(
        OUT_CONFLICT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt summary
    # --------------------------------------------------------

    receipt_summary = (
        conflict_audit.groupby(
            [
                "rcept_no",
                "stock_code",
                "period_key",
            ],
            dropna=False,
        )
        .agg(
            conflict_table_rows=(
                "table_index",
                "size",
            ),

            basis_contradiction_rows=(
                "basis_explicit_contradiction",
                lambda s:
                int(
                    s.fillna(
                        False
                    ).sum()
                ),
            ),

            other_entity_suspect_rows=(
                "entity_scope_class",
                lambda s:
                int(
                    s.isin(
                        [
                            "OTHER_ENTITY_SECTION_SUSPECT",
                            "OTHER_NAMED_COMPANY_SUSPECT",
                        ]
                    ).sum()
                ),
            ),

            keep_for_ranking_rows=(
                "conflict_semantic_gate",
                lambda s:
                int(
                    s.eq(
                        "KEEP_FOR_RANKING"
                    ).sum()
                ),
            ),

            distinct_keep_tables=(
                "table_index",
                lambda s:
                int(
                    s.nunique()
                ),
            ),
        )
        .reset_index()
    )

    # distinct_keep_tables above uses all tables, so calculate exact keep-only separately.
    keep_counts = (
        conflict_audit.loc[
            conflict_audit[
                "conflict_semantic_gate"
            ].eq(
                "KEEP_FOR_RANKING"
            )
        ]
        .groupby(
            "rcept_no"
        )[
            "table_index"
        ]
        .nunique()
        .rename(
            "keep_distinct_table_count"
        )
    )

    receipt_summary = receipt_summary.merge(
        keep_counts,
        on="rcept_no",
        how="left",
    )

    receipt_summary[
        "keep_distinct_table_count"
    ] = receipt_summary[
        "keep_distinct_table_count"
    ].fillna(
        0
    ).astype(
        int
    )

    receipt_summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[All signature tables: selector basis x local basis]"
    )

    print(
        pd.crosstab(
            audited[
                "selector_table_basis"
            ],
            audited[
                "local_basis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Explicit basis contradictions - all signature tables]"
    )

    print(
        int(
            audited[
                "basis_explicit_contradiction"
            ].sum()
        )
    )

    print(
        "\n[Entity scope class - all signature tables]"
    )

    print(
        audited[
            "entity_scope_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Genuine conflict semantic gate]"
    )

    print(
        conflict_audit[
            "conflict_semantic_gate"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Genuine conflicts: selector basis x local basis]"
    )

    print(
        pd.crosstab(
            conflict_audit[
                "selector_table_basis"
            ],
            conflict_audit[
                "local_basis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Basis contradiction conflict sample]"
    )

    contradiction = (
        conflict_audit.loc[
            conflict_audit[
                "basis_explicit_contradiction"
            ].fillna(
                False
            )
        ]
    )

    if contradiction.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            50,
        ):
            print(
                contradiction[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "table_index",
                        "selector_table_basis",
                        "local_basis",
                        "local_basis_reason",
                        "signature_text",
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
        "\n[Other-entity suspect conflict sample]"
    )

    other_entity = (
        conflict_audit.loc[
            conflict_audit[
                "entity_scope_class"
            ].isin(
                [
                    "OTHER_ENTITY_SECTION_SUSPECT",
                    "OTHER_NAMED_COMPANY_SUSPECT",
                ]
            )
        ]
    )

    if other_entity.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            50,
        ):
            print(
                other_entity[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "table_index",
                        "issuer_name",
                        "entity_scope_class",
                        "entity_scope_reason",
                        "named_companies",
                        "signature_text",
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
        "\n[Remaining KEEP_FOR_RANKING sample]"
    )

    keep = (
        conflict_audit.loc[
            conflict_audit[
                "conflict_semantic_gate"
            ].eq(
                "KEEP_FOR_RANKING"
            )
        ]
    )

    if keep.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            50,
        ):
            print(
                keep[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "signature_cluster",
                        "table_index",
                        "issuer_name",
                        "selector_table_basis",
                        "local_basis",
                        "entity_scope_class",
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
        "\n[Receipts by remaining distinct conflict tables]"
    )

    print(
        receipt_summary[
            "keep_distinct_table_count"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Conflict audit : "
        f"{OUT_CONFLICT}"
    )

    print(
        f"- Table audit    : "
        f"{OUT_TABLE}"
    )

    print(
        f"- Receipt summary: "
        f"{OUT_RECEIPT}"
    )

    print(
        "\n해석 원칙:"
        "\n- explicit 별도/개별 vs CFS label 충돌은 ranking 전에 BLOCK"
        "\n- 명시적 다른 법인/타 entity 표도 ranking 전에 BLOCK 또는 REVIEW"
        "\n- local basis UNKNOWN은 억지로 CFS/OFS 재분류하지 않음"
        "\n- semantic gate를 통과한 genuine conflict만 다음 account/table ranking 대상으로 사용"
    )


if __name__ == "__main__":
    main()
