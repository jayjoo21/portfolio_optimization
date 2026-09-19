from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B6A. Basis Detector V2 Diagnostic
#
# 목적
# ------------------------------------------------------------
# H6B5의 423개 diagnostic row에 대해 기존 context_reasons를 믿지 않고,
# 실제 heading_context / primary_row_label만으로 CFS/OFS basis를 재판정한다.
#
# 핵심 수정
# ------------------------------------------------------------
# 기존 문제:
#   문맥 어딘가의 "연결" 단어만 보고 CFS로 오인 가능
#
# V2:
#   - 강한 statement phrase를 우선
#   - "개별기준", "요약개별재무정보", "별도재무제표"는 OFS 강증거
#   - "연결대상 종속기업이 존재하지 않음"은 NO_CFS/OFS-only 강증거
#   - "연결재무상태표", "연결포괄손익계산서" 등은 CFS 강증거
#   - 기존 context_reasons의 consolidated/separate tag는 사용하지 않음
#
# 이 단계는 진단만 수행한다.
# production selector / final data를 수정하지 않는다.
#
# 실행:
# python scripts\05a5h6b6a_dart_basis_detector_v2_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_h6b5_basis_context_diagnostic.csv"
)

OUTPUT = (
    INTERIM
    / "dart_noncorrected_h6b6a_basis_detector_v2_diagnostic.csv"
)


def raw_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def compact(value) -> str:
    text = raw_text(value).lower()

    # Korean statement titles often contain spaces between every syllable:
    # 연 결 포 괄 손 익 계 산 서 -> 연결포괄손익계산서
    return re.sub(
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        text,
    )


def detect_basis_v2(row: pd.Series) -> tuple[str, str, int, int]:

    # IMPORTANT:
    # Do NOT use context_reasons here.
    # H6B5 proved that field contains legacy basis-classifier output
    # and can therefore contaminate a new diagnosis.
    context = compact(
        " ".join(
            [
                raw_text(
                    row.get(
                        "heading_context"
                    )
                ),
                raw_text(
                    row.get(
                        "primary_row_label"
                    )
                ),
            ]
        )
    )

    cfs_reasons = []
    ofs_reasons = []

    # --------------------------------------------------------
    # Strong "no consolidated statements / OFS only" evidence
    # --------------------------------------------------------

    no_cfs_patterns = [
        (
            r"연결대상종속기업.*존재하지않",
            "no_consolidated_subsidiary",
        ),
        (
            r"연결대상종속기업.*없",
            "no_consolidated_subsidiary",
        ),
        (
            r"연결재무제표.*작성하지않",
            "consolidated_statements_not_prepared",
        ),
        (
            r"연결재무제표.*작성의무.*없",
            "no_consolidated_reporting_obligation",
        ),
    ]

    for pattern, reason in no_cfs_patterns:
        if re.search(
            pattern,
            context,
        ):
            ofs_reasons.append(
                (
                    reason,
                    100,
                )
            )

    # --------------------------------------------------------
    # Explicit OFS / separate / individual statement evidence
    # --------------------------------------------------------

    ofs_patterns = [
        (
            "요약개별재무정보",
            "summary_individual_financial_info",
            20,
        ),
        (
            "개별재무정보",
            "individual_financial_info",
            18,
        ),
        (
            "본자료는개별기준",
            "explicit_individual_basis",
            25,
        ),
        (
            "개별기준으로작성",
            "explicit_individual_basis",
            25,
        ),
        (
            "개별재무제표",
            "individual_financial_statements",
            25,
        ),
        (
            "별도재무제표",
            "separate_financial_statements",
            25,
        ),
        (
            "별도재무상태표",
            "separate_balance_sheet",
            30,
        ),
        (
            "별도포괄손익계산서",
            "separate_comprehensive_income_statement",
            30,
        ),
        (
            "별도손익계산서",
            "separate_income_statement",
            30,
        ),
        (
            "별도자본변동표",
            "separate_changes_in_equity",
            30,
        ),
        (
            "별도현금흐름표",
            "separate_cashflow_statement",
            30,
        ),
    ]

    for phrase, reason, score in ofs_patterns:
        if phrase in context:
            ofs_reasons.append(
                (
                    reason,
                    score,
                )
            )

    # --------------------------------------------------------
    # Explicit CFS / consolidated statement evidence
    # --------------------------------------------------------

    cfs_patterns = [
        (
            "요약연결재무정보",
            "summary_consolidated_financial_info",
            20,
        ),
        (
            "요약연결재무제표",
            "summary_consolidated_financial_statements",
            20,
        ),
        (
            "연결재무상태표",
            "consolidated_balance_sheet",
            30,
        ),
        (
            "연결포괄손익계산서",
            "consolidated_comprehensive_income_statement",
            30,
        ),
        (
            "연결손익계산서",
            "consolidated_income_statement",
            30,
        ),
        (
            "연결자본변동표",
            "consolidated_changes_in_equity",
            30,
        ),
        (
            "연결현금흐름표",
            "consolidated_cashflow_statement",
            30,
        ),
        (
            "연결재무제표기준",
            "explicit_consolidated_basis",
            25,
        ),
        (
            "연결기준으로작성",
            "explicit_consolidated_basis",
            25,
        ),
        (
            "회사와그종속기업",
            "company_and_subsidiaries",
            12,
        ),
        (
            "주식회사와그종속기업",
            "company_and_subsidiaries",
            12,
        ),
    ]

    for phrase, reason, score in cfs_patterns:
        if phrase in context:
            cfs_reasons.append(
                (
                    reason,
                    score,
                )
            )

    cfs_score = sum(
        score
        for _, score
        in cfs_reasons
    )

    ofs_score = sum(
        score
        for _, score
        in ofs_reasons
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    # Explicit no-CFS wording overrides generic "연결" mentions.
    no_cfs = any(
        reason.startswith(
            "no_consolidated"
        )
        or reason
        in {
            "consolidated_statements_not_prepared",
            "no_consolidated_reporting_obligation",
        }
        for reason, _
        in ofs_reasons
    )

    if no_cfs:
        basis = (
            "OFS_ONLY_NO_CFS"
        )

    elif (
        cfs_score > 0
        and ofs_score == 0
    ):
        basis = "CFS"

    elif (
        ofs_score > 0
        and cfs_score == 0
    ):
        basis = "OFS"

    elif (
        cfs_score > 0
        and ofs_score > 0
    ):

        # Only call a winner when evidence is substantially stronger.
        if (
            cfs_score
            >= ofs_score + 20
        ):
            basis = "CFS"

        elif (
            ofs_score
            >= cfs_score + 20
        ):
            basis = "OFS"

        else:
            basis = "MIXED"

    else:
        basis = "UNKNOWN"

    reasons = []

    if cfs_reasons:
        reasons.append(
            "CFS:"
            + ",".join(
                reason
                for reason, _
                in cfs_reasons
            )
        )

    if ofs_reasons:
        reasons.append(
            "OFS:"
            + ",".join(
                reason
                for reason, _
                in ofs_reasons
            )
        )

    return (
        basis,
        " | ".join(
            reasons
        ),
        cfs_score,
        ofs_score,
    )


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_csv(
        INPUT,
        dtype={
            "stock_code":
            str,

            "rcept_no":
            str,
        },
        low_memory=False,
    )

    detected = df.apply(
        detect_basis_v2,
        axis=1,
        result_type="expand",
    )

    detected.columns = [
        "basis_v2",
        "basis_v2_reasons",
        "basis_v2_cfs_score",
        "basis_v2_ofs_score",
    ]

    out = pd.concat(
        [
            df,
            detected,
        ],
        axis=1,
    )

    out.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "05A5-H6B6A BASIS DETECTOR V2 DIAGNOSTIC"
    )

    print(
        "=" * 110
    )

    print(
        f"\nRows: "
        f"{len(out):,}"
    )

    print(
        "\n[V2 basis]"
    )

    print(
        out[
            "basis_v2"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if (
        "diagnostic_family_evidence"
        in out.columns
    ):

        print(
            "\n[Old family evidence x V2 basis]"
        )

        print(
            pd.crosstab(
                out[
                    "diagnostic_family_evidence"
                ],
                out[
                    "basis_v2"
                ],
                dropna=False,
            )
            .to_string()
        )

    # --------------------------------------------------------
    # Decisive case: 제낙스
    # --------------------------------------------------------

    jenax = out.loc[
        out[
            "rcept_no"
        ]
        .astype(str)
        .eq(
            "20251113000087"
        )
    ].copy()

    print(
        "\n[제낙스 2025 Q3]"
    )

    if jenax.empty:
        print(
            "Not found"
        )

    else:

        cols = [
            c
            for c in [
                "account_family",
                "column_status",
                "selected_value_krw",
                "table_index",
                "row_index",
                "selected_column",
                "primary_row_label",
                "heading_context",
                "basis_v2",
                "basis_v2_reasons",
                "basis_v2_cfs_score",
                "basis_v2_ofs_score",
            ]
            if c in jenax.columns
        ]

        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            260,
            "display.max_rows",
            80,
        ):
            print(
                jenax[
                    cols
                ]
                .to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # Clear explicit consolidated statement rows
    # --------------------------------------------------------

    explicit_cfs = out.loc[
        out[
            "heading_context"
        ]
        .astype(
            "string"
        )
        .fillna("")
        .str.contains(
            r"연\s*결\s*.*"
            r"(재\s*무\s*상\s*태\s*표|"
            r"포\s*괄\s*손\s*익\s*계\s*산\s*서|"
            r"손\s*익\s*계\s*산\s*서|"
            r"자\s*본\s*변\s*동\s*표|"
            r"현\s*금\s*흐\s*름\s*표)",
            regex=True,
        )
    ].copy()

    print(
        "\n[Explicit consolidated-statement rows -> V2 basis]"
    )

    if explicit_cfs.empty:
        print(
            "None"
        )
    else:
        print(
            explicit_cfs[
                "basis_v2"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # Explicit individual/separate / no-CFS rows
    # --------------------------------------------------------

    explicit_ofs = out.loc[
        out[
            "heading_context"
        ]
        .astype(
            "string"
        )
        .fillna("")
        .str.contains(
            r"개별|별도|"
            r"연결대상\s*종속기업.*"
            r"(존재하지|없)",
            regex=True,
        )
    ].copy()

    print(
        "\n[Explicit individual/separate/no-CFS rows -> V2 basis]"
    )

    if explicit_ofs.empty:
        print(
            "None"
        )
    else:
        print(
            explicit_ofs[
                "basis_v2"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        f"\nOutput: "
        f"{OUTPUT}"
    )

    print(
        "\nPASS 기준:"
        "\n- 제낙스의 요약개별/개별기준/no-subsidiary row가 CFS로 분류되면 안 됨"
        "\n- 명시적 연결재무상태표/연결포괄손익계산서 row는 대부분 CFS여야 함"
        "\n- UNKNOWN/MIXED를 억지로 OFS/CFS로 강제하지 않음"
        "\n- PASS 후에만 production basis detector로 승격"
    )


if __name__ == "__main__":
    main()
