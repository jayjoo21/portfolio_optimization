from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7L. Refined Table-Semantics Audit
#
# 목적
# ------------------------------------------------------------
# H6B7K의 두 과잉탐지 문제를 수정:
#
# 1) basis:
#    표 제목/본문은 "별도"인데 뒤쪽 주석에 "연결"이 언급되어
#    MIXED_EXPLICIT로 잡히는 문제
#
# 2) entity scope:
#    issuer의 연결재무제표 주석에서 자회사/관계기업명이 언급되었을 뿐인데
#    OTHER_NAMED_COMPANY_SUSPECT로 BLOCK되는 문제
#
# 핵심 원칙
# ------------------------------------------------------------
# heading_context를
#
#   HEAD / MAIN CONTEXT
#   -------------------
#   NOTE / FOOTNOTE TAIL
#
# 로 분리한다.
#
# - main context의 직접 표 제목/회사명 증거를 우선
# - note tail의 다른 회사명 언급은 단독 BLOCK 근거로 사용하지 않음
# - 명시적 다른 회사가 표 owner인 경우만 BLOCK
# - local basis도 main context의 직접 제목을 note tail보다 우선
#
# IMPORTANT
# ------------------------------------------------------------
# - final selector 아님
# - ZIP 재파싱 없음
# - genuine conflict만 대상으로 semantic gate 재평가
#
# 실행:
# python scripts\05a5h6b7l_dart_refined_table_semantics_audit.py
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
    / "dart_noncorrected_nodata_conflict_refined_semantics.csv"
)

OUT_TABLE = (
    INTERIM
    / "dart_noncorrected_nodata_table_refined_semantics.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_conflict_refined_receipt_summary.csv"
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

    for token in [
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
    ]:
        text = text.replace(
            token,
            "",
        )

    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        text,
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


# ============================================================
# Main-context / note-tail split
# ============================================================


NOTE_MARKERS = [
    "|| 주)",
    "|| 주1)",
    "|| 주2)",
    "|| 주3)",
    "|| (*)",
    "|| ※",
    "|| 註",
    "|| 주 :",
    "|| 주:",
    "|| 참고",
]


def split_main_note(context) -> tuple[str, str]:
    text = clean(
        context
    )

    positions = []

    for marker in NOTE_MARKERS:
        pos = text.find(
            marker
        )
        if pos >= 0:
            positions.append(
                pos
            )

    if positions:
        cut = min(
            positions
        )
        return (
            text[:cut].strip(),
            text[cut:].strip(),
        )

    # No explicit footnote marker:
    # keep the first 500 chars as primary semantic context,
    # remaining tail only as weak note-like evidence.
    if len(
        text
    ) > 500:
        return (
            text[:500].strip(),
            text[500:].strip(),
        )

    return (
        text,
        "",
    )


# ============================================================
# Refined basis logic
# ============================================================


MAIN_OFS_PATTERNS = [
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

MAIN_CFS_PATTERNS = [
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


def hits(
    text: str,
    patterns: list[str],
) -> list[str]:
    return [
        p
        for p in patterns
        if p in text
    ]


def refined_local_basis(
    context,
) -> dict:

    main_text, note_text = split_main_note(
        context
    )

    main = compact(
        main_text
    )

    note = compact(
        note_text
    )

    main_no_cfs = hits(
        main,
        NO_CFS_PATTERNS,
    )

    main_ofs = hits(
        main,
        MAIN_OFS_PATTERNS,
    )

    main_cfs = hits(
        main,
        MAIN_CFS_PATTERNS,
    )

    note_ofs = hits(
        note,
        MAIN_OFS_PATTERNS,
    )

    note_cfs = hits(
        note,
        MAIN_CFS_PATTERNS,
    )

    if main_no_cfs:
        basis = (
            "OFS_ONLY_NO_CFS"
        )

        reason = (
            "MAIN_NO_CFS:"
            + ",".join(
                main_no_cfs
            )
        )

    elif main_ofs and not main_cfs:
        basis = "OFS"

        reason = (
            "MAIN_OFS:"
            + ",".join(
                main_ofs
            )
        )

    elif main_cfs and not main_ofs:
        basis = "CFS"

        reason = (
            "MAIN_CFS:"
            + ",".join(
                main_cfs
            )
        )

    elif main_ofs and main_cfs:
        # Both directly in main context.
        basis = (
            "MIXED_MAIN"
        )

        reason = (
            "MAIN_OFS:"
            + ",".join(
                main_ofs
            )
            + " | MAIN_CFS:"
            + ",".join(
                main_cfs
            )
        )

    else:
        # No direct basis title in main context.
        main_compact = main

        if (
            "와그종속기업"
            in main_compact
            or "및그종속기업"
            in main_compact
            or "와그종속회사"
            in main_compact
            or "및그종속회사"
            in main_compact
        ):
            basis = (
                "CFS_ENTITY_HINT"
            )

            reason = (
                "MAIN_company_and_subsidiaries"
            )

        else:
            basis = (
                "UNKNOWN"
            )

            reason = ""

    # Note evidence is recorded, but does not override main basis.
    note_reason = []

    if note_ofs:
        note_reason.append(
            "NOTE_OFS:"
            + ",".join(
                note_ofs
            )
        )

    if note_cfs:
        note_reason.append(
            "NOTE_CFS:"
            + ",".join(
                note_cfs
            )
        )

    return {
        "main_context":
        main_text,

        "note_context":
        note_text,

        "refined_local_basis":
        basis,

        "refined_local_basis_reason":
        reason,

        "note_basis_evidence":
        " | ".join(
            note_reason
        ),
    }


def basis_family(
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
# Issuer map
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
            values = (
                scope[
                    col
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if values.ne(
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

    if (
        name_col is None
        or "rcept_no"
        not in scope.columns
    ):
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
    ] = normalize_receipt(
        work[
            "rcept_no"
        ]
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
# Company / owner evidence
# ============================================================


COMPANY_PATTERNS = [
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


def extract_companies(
    text,
) -> list[str]:

    text = clean(
        text
    )

    found = []

    for pattern in COMPANY_PATTERNS:
        for match in pattern.findall(
            text
        ):
            value = clean(
                match
            )

            if (
                value
                and value
                not in found
            ):
                found.append(
                    value
                )

    return found


def issuer_mentioned(
    text,
    issuer_name,
) -> bool:

    issuer_norm = normalize_company(
        issuer_name
    )

    if not issuer_norm:
        return False

    return (
        issuer_norm
        in normalize_company(
            text
        )
    )


def direct_other_owner_patterns(
    main_text,
    issuer_name,
) -> tuple[
    list[str],
    list[str],
]:

    """
    True other-entity owner evidence should be direct, e.g.
      - 주식회사 우리카드와 그 종속기업
      - SK(주)의 제32기 반기 연결 재무제표 기준

    Mere footnote mentions such as
      - 자회사인 DGB생명보험의 ...
      - 관계기업인 미래에셋생명보험이 ...
    are NOT owner evidence.
    """

    text = clean(
        main_text
    )

    issuer_norm = normalize_company(
        issuer_name
    )

    companies = extract_companies(
        text
    )

    strong_owner_names = []
    owner_reasons = []

    # Pattern 1:
    # "주식회사 X와/및 그 종속기업(회사)"
    owner_pattern_1 = re.compile(
        r"(?:주식회사\s*)?"
        r"([가-힣A-Za-z0-9ㆍ&]+)"
        r"\s*(?:주식회사)?"
        r"\s*(?:와|및)\s*그\s*종속(?:기업|회사)"
    )

    for company in owner_pattern_1.findall(
        text
    ):
        company_norm = normalize_company(
            company
        )

        if (
            company_norm
            and issuer_norm
            and company_norm
            != issuer_norm
            and company_norm
            not in issuer_norm
            and issuer_norm
            not in company_norm
        ):
            strong_owner_names.append(
                company
            )

            owner_reasons.append(
                "other_company_and_subsidiaries"
            )

    # Pattern 2:
    # "SK(주)의 제32기 ... 재무제표 기준"
    owner_pattern_2 = re.compile(
        r"([가-힣A-Za-z0-9ㆍ&]+)"
        r"\s*(?:\(주\)|㈜|주식회사)?"
        r"\s*의\s*제?\s*\d+\s*기"
        r".{0,80}?"
        r"(?:연결|별도|개별)?\s*재무제표\s*기준"
    )

    for company in owner_pattern_2.findall(
        text
    ):
        company_norm = normalize_company(
            company
        )

        if (
            company_norm
            and issuer_norm
            and company_norm
            != issuer_norm
            and company_norm
            not in issuer_norm
            and issuer_norm
            not in company_norm
        ):
            strong_owner_names.append(
                company
            )

            owner_reasons.append(
                "other_company_financial_statement_basis"
            )

    return (
        list(
            dict.fromkeys(
                strong_owner_names
            )
        ),
        list(
            dict.fromkeys(
                owner_reasons
            )
        ),
    )


NOTE_RELATION_PATTERNS = [
    "자회사인",
    "종속기업인",
    "관계기업인",
    "공동기업인",
    "연결실체의관계기업",
    "연결실체의종속기업",
    "연결실체의자회사",
]


def refined_entity_scope(
    context,
    issuer_name,
) -> dict:

    main_text, note_text = split_main_note(
        context
    )

    issuer_norm = normalize_company(
        issuer_name
    )

    main_companies = extract_companies(
        main_text
    )

    note_companies = extract_companies(
        note_text
    )

    main_issuer = issuer_mentioned(
        main_text,
        issuer_name,
    )

    note_issuer = issuer_mentioned(
        note_text,
        issuer_name,
    )

    (
        other_owner_names,
        other_owner_reasons,
    ) = direct_other_owner_patterns(
        main_text,
        issuer_name,
    )

    note_compact = compact(
        note_text
    )

    relation_hits = [
        p
        for p in NOTE_RELATION_PATTERNS
        if p in note_compact
    ]

    if not issuer_norm:
        scope_class = (
            "ISSUER_NAME_UNAVAILABLE"
        )

        reason = ""

    elif other_owner_names:
        scope_class = (
            "OTHER_ENTITY_OWNER_EXPLICIT"
        )

        reason = (
            "|".join(
                other_owner_reasons
            )
            + ":"
            + ",".join(
                other_owner_names
            )
        )

    elif main_issuer:
        scope_class = (
            "ISSUER_OR_GROUP_OWNER"
        )

        reason = (
            "issuer_mentioned_in_main_context"
        )

    elif (
        note_companies
        and relation_hits
    ):
        # Other company only appears as a relation / footnote mention.
        scope_class = (
            "OTHER_ENTITY_NOTE_REFERENCE_ONLY"
        )

        reason = (
            "note_relation:"
            + ",".join(
                relation_hits
            )
        )

    elif main_companies:
        # A company is named in main context but not confidently identified
        # as owner. Review rather than auto-block.
        scope_class = (
            "MAIN_OTHER_COMPANY_AMBIGUOUS"
        )

        reason = (
            "main_named_companies:"
            + ",".join(
                main_companies
            )
        )

    else:
        scope_class = (
            "ENTITY_SCOPE_UNKNOWN"
        )

        reason = ""

    return {
        "refined_entity_scope":
        scope_class,

        "refined_entity_reason":
        reason,

        "main_named_companies":
        "|".join(
            main_companies
        ),

        "note_named_companies":
        "|".join(
            note_companies
        ),

        "main_issuer_mentioned":
        main_issuer,

        "note_issuer_mentioned":
        note_issuer,
    }


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

    conflicts[
        "rcept_no"
    ] = normalize_receipt(
        conflicts[
            "rcept_no"
        ]
    )

    signatures[
        "rcept_no"
    ] = normalize_receipt(
        signatures[
            "rcept_no"
        ]
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

        issuer_map, issuer_col = (
            build_issuer_map(
                scope
            )
        )

    else:
        issuer_map = {}
        issuer_col = None

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7L REFINED TABLE-SEMANTICS AUDIT"
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
            issuer_col
            if issuer_col
            else "NOT FOUND"
        )
    )

    # --------------------------------------------------------
    # Audit all signature rows
    # --------------------------------------------------------

    audited = signatures.copy()

    audited[
        "issuer_name"
    ] = audited[
        "rcept_no"
    ].map(
        issuer_map
    )

    basis_results = (
        audited[
            "heading_context"
        ]
        .apply(
            refined_local_basis
        )
    )

    for col in [
        "main_context",
        "note_context",
        "refined_local_basis",
        "refined_local_basis_reason",
        "note_basis_evidence",
    ]:
        audited[
            col
        ] = [
            result[
                col
            ]
            for result in basis_results
        ]

    audited[
        "selector_basis_family"
    ] = audited[
        "selector_table_basis"
    ].map(
        basis_family
    )

    audited[
        "refined_basis_family"
    ] = audited[
        "refined_local_basis"
    ].map(
        basis_family
    )

    audited[
        "refined_basis_contradiction"
    ] = (
        audited[
            "selector_basis_family"
        ].isin(
            [
                "CFS",
                "OFS",
            ]
        )
        & audited[
            "refined_basis_family"
        ].isin(
            [
                "CFS",
                "OFS",
            ]
        )
        & audited[
            "selector_basis_family"
        ].ne(
            audited[
                "refined_basis_family"
            ]
        )
    )

    entity_results = [
        refined_entity_scope(
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

    for col in [
        "refined_entity_scope",
        "refined_entity_reason",
        "main_named_companies",
        "note_named_companies",
        "main_issuer_mentioned",
        "note_issuer_mentioned",
    ]:
        audited[
            col
        ] = [
            result[
                col
            ]
            for result in entity_results
        ]

    audited.to_csv(
        OUT_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Join conflict rows
    # --------------------------------------------------------

    audit_cols = [
        "bundle_type",
        "rcept_no",
        "table_index",
        "issuer_name",
        "refined_local_basis",
        "refined_local_basis_reason",
        "note_basis_evidence",
        "selector_basis_family",
        "refined_basis_family",
        "refined_basis_contradiction",
        "refined_entity_scope",
        "refined_entity_reason",
        "main_named_companies",
        "note_named_companies",
        "main_issuer_mentioned",
        "note_issuer_mentioned",
        "main_context",
        "note_context",
    ]

    conflict_audit = conflicts.merge(
        audited[
            audit_cols
        ],
        on=[
            "bundle_type",
            "rcept_no",
            "table_index",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Refined semantic gate
    # --------------------------------------------------------

    conflict_audit[
        "refined_semantic_gate"
    ] = np.select(
        [
            conflict_audit[
                "refined_basis_contradiction"
            ].fillna(
                False
            ),

            conflict_audit[
                "refined_entity_scope"
            ].eq(
                "OTHER_ENTITY_OWNER_EXPLICIT"
            ),

            conflict_audit[
                "refined_local_basis"
            ].eq(
                "MIXED_MAIN"
            ),

            conflict_audit[
                "refined_entity_scope"
            ].eq(
                "MAIN_OTHER_COMPANY_AMBIGUOUS"
            ),
        ],
        [
            "BLOCK_BASIS_CONTRADICTION",
            "BLOCK_OTHER_ENTITY_OWNER",
            "REVIEW_MIXED_MAIN_BASIS",
            "REVIEW_ENTITY_OWNER_AMBIGUOUS",
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

    receipt_records = []

    for (
        rcept_no,
        stock_code,
        period_key,
    ), group in conflict_audit.groupby(
        [
            "rcept_no",
            "stock_code",
            "period_key",
        ],
        dropna=False,
        sort=False,
    ):

        keep = group.loc[
            group[
                "refined_semantic_gate"
            ].eq(
                "KEEP_FOR_RANKING"
            )
        ]

        blocked = group.loc[
            group[
                "refined_semantic_gate"
            ].astype(str)
            .str.startswith(
                "BLOCK_",
                na=False,
            )
        ]

        review = group.loc[
            group[
                "refined_semantic_gate"
            ].astype(str)
            .str.startswith(
                "REVIEW_",
                na=False,
            )
        ]

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                stock_code,

                "period_key":
                period_key,

                "conflict_row_count":
                len(
                    group
                ),

                "blocked_row_count":
                len(
                    blocked
                ),

                "review_row_count":
                len(
                    review
                ),

                "keep_row_count":
                len(
                    keep
                ),

                "keep_distinct_table_count":
                keep[
                    "table_index"
                ].nunique(),

                "keep_balance_table_count":
                keep.loc[
                    keep[
                        "bundle_type"
                    ].eq(
                        "BALANCE"
                    ),
                    "table_index",
                ].nunique(),

                "keep_income_table_count":
                keep.loc[
                    keep[
                        "bundle_type"
                    ].eq(
                        "INCOME"
                    ),
                    "table_index",
                ].nunique(),
            }
        )

    receipt_summary = pd.DataFrame(
        receipt_records
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
        "\n[Refined local basis - all signature tables]"
    )

    print(
        pd.crosstab(
            audited[
                "selector_table_basis"
            ],
            audited[
                "refined_local_basis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Refined explicit basis contradictions]"
    )

    print(
        int(
            audited[
                "refined_basis_contradiction"
            ].sum()
        )
    )

    print(
        "\n[Refined entity scope - all signature tables]"
    )

    print(
        audited[
            "refined_entity_scope"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Refined genuine-conflict semantic gate]"
    )

    print(
        conflict_audit[
            "refined_semantic_gate"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Note-only other-company references kept]"
    )

    note_only = conflict_audit.loc[
        conflict_audit[
            "refined_entity_scope"
        ].eq(
            "OTHER_ENTITY_NOTE_REFERENCE_ONLY"
        )
    ]

    print(
        f"{len(note_only):,}"
    )

    if not note_only.empty:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            30,
        ):
            print(
                note_only[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "table_index",
                        "issuer_name",
                        "refined_entity_scope",
                        "refined_entity_reason",
                        "signature_text",
                        "main_context",
                        "note_context",
                    ]
                ]
                .head(
                    20
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Explicit other-entity owner blocks]"
    )

    other_owner = conflict_audit.loc[
        conflict_audit[
            "refined_entity_scope"
        ].eq(
            "OTHER_ENTITY_OWNER_EXPLICIT"
        )
    ]

    print(
        f"{len(other_owner):,}"
    )

    if not other_owner.empty:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            30,
        ):
            print(
                other_owner[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "table_index",
                        "issuer_name",
                        "refined_entity_reason",
                        "signature_text",
                        "main_context",
                    ]
                ]
                .head(
                    20
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Refined basis contradiction sample]"
    )

    contradictions = conflict_audit.loc[
        conflict_audit[
            "refined_basis_contradiction"
        ].fillna(
            False
        )
    ]

    if contradictions.empty:
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
            30,
        ):
            print(
                contradictions[
                    [
                        "bundle_type",
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "table_index",
                        "selector_table_basis",
                        "refined_local_basis",
                        "refined_local_basis_reason",
                        "note_basis_evidence",
                        "signature_text",
                        "main_context",
                        "note_context",
                    ]
                ]
                .head(
                    20
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Remaining receipts by keep table count]"
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
        "\n[Receipts immediately resolved to <=1 keep table]"
    )

    print(
        int(
            receipt_summary[
                "keep_distinct_table_count"
            ]
            .le(
                1
            )
            .sum()
        )
    )

    print(
        "\n[Remaining KEEP_FOR_RANKING sample]"
    )

    keep = conflict_audit.loc[
        conflict_audit[
            "refined_semantic_gate"
        ].eq(
            "KEEP_FOR_RANKING"
        )
    ]

    if keep.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            170,
            "display.width",
            320,
            "display.max_rows",
            40,
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
                        "refined_local_basis",
                        "refined_entity_scope",
                        "signature_text",
                        "primary_labels",
                        "main_context",
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
        "\n- 표 본문/제목의 basis가 주석 basis보다 우선"
        "\n- 주석에 자회사/관계기업명이 언급된 것만으로는 다른 entity 표로 차단하지 않음"
        "\n- 다른 회사가 실제 표 owner임이 명시될 때만 BLOCK"
        "\n- 이 단계 통과 후 남은 genuine conflicts에만 ranking 적용"
    )


if __name__ == "__main__":
    main()
