from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B7D. Period V3 Context Guardrail Audit
#
# 목적
# ------------------------------------------------------------
# H6B7C에서 LOW -> context-confirmed로 승격된 1,133개 row 중
# "날짜는 맞지만 다른 회사/주석/요약표"인 false positive를 차단한다.
#
# 핵심 원칙
# ------------------------------------------------------------
# 날짜 일치만으로는 승인하지 않는다.
#
# STRONG 승인 후보:
#   - target date/year-month 일치
#   - 해당 metric family와 맞는 본 재무제표 제목이 문맥에 존재
#   - 최대주주/관계기업/종속기업 요약재무정보/주당이익 등
#     명백한 비본표 문맥이 아님
#
# IMPORTANT
# ------------------------------------------------------------
# - document ZIP 재파싱 없음
# - CFS/OFS 최종 채택 아님
# - OFS fallback 최종 결정 아님
# - period context 신뢰도만 정리
#
# 실행:
# python scripts\05a5h6b7d_dart_period_v3_context_guardrail.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_context_audit.parquet"
)

OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_guarded.parquet"
)

SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_guardrail_summary.csv"
)


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def compact(value) -> str:
    return re.sub(
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        clean(value).lower(),
    )


# ------------------------------------------------------------
# Explicit non-main-statement contexts
# ------------------------------------------------------------

BLOCK_PATTERNS = [
    ("최대주주", "major_shareholder_context"),
    ("최대주주의최대주주", "major_shareholder_parent_context"),

    ("관계기업요약재무정보", "associate_summary"),
    ("관계기업의요약재무정보", "associate_summary"),
    ("공동기업요약재무정보", "joint_venture_summary"),
    ("공동기업의요약재무정보", "joint_venture_summary"),

    ("종속기업요약재무정보", "subsidiary_summary"),
    ("종속기업의요약재무정보", "subsidiary_summary"),
    ("주요종속기업요약재무정보", "major_subsidiary_summary"),
    ("주요종속기업의요약재무정보", "major_subsidiary_summary"),

    ("관계기업투자", "associate_investment_note"),
    ("종속기업투자", "subsidiary_investment_note"),

    ("주당이익", "eps_note"),
    ("주당손익", "eps_note"),
    ("이익잉여금처분계산서", "retained_earnings_appropriation"),
    ("배당에관한사항", "dividend_note"),

    ("최근결산기재무현황", "other_entity_recent_financial_status"),
    ("최근결산기재무정보", "other_entity_recent_financial_status"),

    ("국내증권업수익성추이", "industry_statistics"),
    ("시장점유율", "market_statistics"),
]


def block_reason(context_compact: str) -> str:
    reasons = []

    for phrase, reason in BLOCK_PATTERNS:
        if phrase in context_compact:
            reasons.append(reason)

    return "|".join(dict.fromkeys(reasons))


# ------------------------------------------------------------
# Main-statement evidence
# ------------------------------------------------------------

BALANCE_TITLES = [
    "재무상태표",
    "연결재무상태표",
    "별도재무상태표",
]

INCOME_TITLES = [
    "손익계산서",
    "포괄손익계산서",
    "연결손익계산서",
    "연결포괄손익계산서",
    "별도손익계산서",
    "별도포괄손익계산서",
]

CASHFLOW_TITLES = [
    "현금흐름표",
    "연결현금흐름표",
    "별도현금흐름표",
]

EQUITY_TITLES = [
    "자본변동표",
    "연결자본변동표",
    "별도자본변동표",
]


def statement_evidence(
    account_family: str,
    context_compact: str,
) -> tuple[bool, str]:

    family = str(account_family)

    if family in {
        "assets",
        "liabilities",
        "equity",
    }:
        titles = BALANCE_TITLES

    elif family in {
        "revenue",
        "operating_income",
        "net_income",
    }:
        titles = INCOME_TITLES

    else:
        titles = []

    hits = [
        title
        for title in titles
        if title in context_compact
    ]

    return (
        bool(hits),
        "|".join(hits),
    )


def classify(row: pd.Series) -> pd.Series:

    v2_status = str(
        row.get(
            "period_v2_status",
            "",
        )
    )

    v3_status = str(
        row.get(
            "period_v3_status",
            "",
        )
    )

    v3_conf = str(
        row.get(
            "period_v3_confidence",
            "",
        )
    )

    context = compact(
        row.get(
            "heading_context",
            "",
        )
    )

    block = block_reason(
        context
    )

    has_statement, statement_titles = (
        statement_evidence(
            row.get(
                "account_family",
                "",
            ),
            context,
        )
    )

    exact_date = bool(
        row.get(
            "period_v3_target_date_in_context",
            False,
        )
    )

    target_ym = bool(
        row.get(
            "period_v3_target_ym_in_context",
            False,
        )
    )

    gate = "NOT_APPLICABLE"
    guarded_status = v3_status
    guarded_conf = v3_conf
    guarded_reason = ""

    # Only audit H6B7C upgrades.
    if (
        v2_status
        == "low_confidence_candidate"
        and v3_status
        == "selected_context_confirmed"
    ):

        if block:
            gate = "BLOCKED_NON_MAIN_CONTEXT"

            guarded_status = (
                "low_confidence_candidate"
            )

            guarded_conf = "LOW"

            guarded_reason = (
                f"blocked:{block}"
            )

        elif (
            has_statement
            and exact_date
        ):
            gate = "PASS_MAIN_STATEMENT_EXACT_DATE"

            guarded_status = (
                "selected_context_guarded"
            )

            guarded_conf = "HIGH"

            guarded_reason = (
                "main_statement"
                f":{statement_titles}"
                "|exact_target_date"
            )

        elif (
            has_statement
            and target_ym
        ):
            gate = "PASS_MAIN_STATEMENT_YEAR_MONTH"

            guarded_status = (
                "selected_context_guarded"
            )

            guarded_conf = "MEDIUM"

            guarded_reason = (
                "main_statement"
                f":{statement_titles}"
                "|target_year_month"
            )

        else:
            # Own-company MD&A / generic summary may be informative,
            # but strict recovery should not automatically adopt it.
            gate = "REVIEW_CONTEXT_ONLY"

            guarded_status = (
                "low_confidence_candidate"
            )

            guarded_conf = "LOW"

            guarded_reason = (
                "date_matches_but_no_main_statement_title"
            )

    return pd.Series(
        {
            "period_guardrail_class":
            gate,

            "period_guardrail_block_reason":
            block,

            "period_guardrail_statement_titles":
            statement_titles,

            "period_guarded_status":
            guarded_status,

            "period_guarded_confidence":
            guarded_conf,

            "period_guarded_reason":
            guarded_reason,
        }
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
        + "=" * 115
    )

    print(
        "05A5-H6B7D PERIOD V3 CONTEXT GUARDRAIL"
    )

    print(
        "=" * 115
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    audit = df.apply(
        classify,
        axis=1,
    )

    out = pd.concat(
        [
            df,
            audit,
        ],
        axis=1,
    )

    out.to_parquet(
        OUTPUT,
        index=False,
    )

    upgraded = out.loc[
        out[
            "period_v2_status"
        ].eq(
            "low_confidence_candidate"
        )
        & out[
            "period_v3_status"
        ].eq(
            "selected_context_confirmed"
        )
    ].copy()

    print(
        "\n[H6B7C context-confirmed rows]"
    )

    print(
        f"{len(upgraded):,}"
    )

    print(
        "\n[Guardrail classification]"
    )

    print(
        upgraded[
            "period_guardrail_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Guarded status]"
    )

    print(
        out[
            "period_guarded_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    blocked = upgraded.loc[
        upgraded[
            "period_guardrail_class"
        ].eq(
            "BLOCKED_NON_MAIN_CONTEXT"
        )
    ].copy()

    passed = upgraded.loc[
        upgraded[
            "period_guardrail_class"
        ].str.startswith(
            "PASS_",
            na=False,
        )
    ].copy()

    review = upgraded.loc[
        upgraded[
            "period_guardrail_class"
        ].eq(
            "REVIEW_CONTEXT_ONLY"
        )
    ].copy()

    print(
        "\n[Blocked false-positive candidates]"
    )

    print(
        f"{len(blocked):,}"
    )

    if not blocked.empty:
        print(
            blocked[
                "period_guardrail_block_reason"
            ]
            .value_counts()
            .head(
                20
            )
            .to_string()
        )

    print(
        "\n[Passed main-statement context]"
    )

    print(
        f"{len(passed):,}"
    )

    print(
        "\n[Review only]"
    )

    print(
        f"{len(review):,}"
    )

    show_cols = [
        c
        for c in [
            "stock_code",
            "period_key",
            "rcept_no",
            "account_family",
            "table_index",
            "row_index",
            "basis_v2",
            "period_v2_selected_column",
            "period_v3_confidence",
            "period_guardrail_class",
            "period_guardrail_block_reason",
            "period_guardrail_statement_titles",
            "period_guarded_status",
            "period_guarded_confidence",
            "heading_context",
        ]
        if c in out.columns
    ]

    print(
        "\n[Blocked sample]"
    )

    if blocked.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            300,
            "display.max_rows",
            40,
        ):
            print(
                blocked[
                    show_cols
                ]
                .head(
                    25
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Passed sample]"
    )

    if passed.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            300,
            "display.max_rows",
            40,
        ):
            print(
                passed[
                    show_cols
                ]
                .head(
                    25
                )
                .to_string(
                    index=False
                )
            )

    # Specific problematic / good examples
    focus = out.loc[
        out[
            "stock_code"
        ]
        .astype(
            "string"
        )
        .isin(
            [
                "003530",
                "034830",
                "001720",
            ]
        )
        & out[
            "period_v2_status"
        ].eq(
            "low_confidence_candidate"
        )
    ].copy()

    print(
        "\n[Focus: 003530 / 034830 / 001720 LOW candidates]"
    )

    if focus.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            300,
            "display.max_rows",
            60,
        ):
            print(
                focus[
                    show_cols
                ]
                .head(
                    50
                )
                .to_string(
                    index=False
                )
            )

    summary = (
        upgraded.groupby(
            [
                "period_guardrail_class",
                "period_guarded_status",
                "period_guarded_confidence",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "row_count"
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nOutput : "
        f"{OUTPUT}"
    )

    print(
        f"Summary: "
        f"{SUMMARY}"
    )

    print(
        "\n판단 기준:"
        "\n- 날짜만 맞는 최대주주/관계기업/종속기업 요약표는 차단"
        "\n- 날짜 + metric에 맞는 본 재무제표 제목이 있어야 자동 승격"
        "\n- MD&A/일반 요약은 REVIEW로 남김"
        "\n- 여기서도 아직 CFS/OFS 최종 채택은 하지 않음"
    )


if __name__ == "__main__":
    main()
