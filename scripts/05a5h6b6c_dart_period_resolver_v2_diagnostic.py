from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B6C. Current-Period Resolver V2 Diagnostic
#
# 목적
# ------------------------------------------------------------
# H6B6B에서 저장한 37개 진단 row만 사용하여
# current-period column 선택 규칙 V2를 검증한다.
#
# 원본 document ZIP 재파싱 없음.
# production C4 수정 없음.
# 최종 merge 없음.
#
# 검증할 핵심 케이스
# ------------------------------------------------------------
# 1) "제100(당)기" vs "제99(전)기"
#    -> 당기 선택
#
# 2) 사업보고서|2020.12인데
#    "2019년 12월말", "2018년 12월말"
#    -> 둘 다 current 아님
#
# 3) 반기보고서|2018.06인데
#    "제70기 1분기"
#    -> H1_Q1 mismatch로 거절
#
# 4) "제100기 / 제99기 / 제98기"처럼 기수만 있는 경우
#    -> highest generation을 LOW-CONFIDENCE 후보로만 표시
#       (production 자동채택 금지)
#
# INPUT
# ------------------------------------------------------------
# dart_noncorrected_h6b6b_current_period_column_diagnostic.csv
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_h6b6c_period_resolver_v2_diagnostic.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h6b6c_dart_period_resolver_v2_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_h6b6b_current_period_column_diagnostic.csv"
)

OUTPUT = (
    INTERIM
    / "dart_noncorrected_h6b6c_period_resolver_v2_diagnostic.csv"
)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def compact(value) -> str:
    return re.sub(
        r"\s+",
        "",
        clean_text(value).lower(),
    )


def marker_text(value) -> str:
    """
    DART 헤더 예:
      제76(당)기
      제 100 (당)기
      제75(전)기

    괄호/공백 때문에 단순 '당기' 검색이 실패하지 않도록
    marker 판정용 문자열만 별도로 정규화한다.
    """
    text = clean_text(value).lower()

    return re.sub(
        r"[\s\(\)\[\]\{\}〈〉《》「」『』【】]+",
        "",
        text,
    )


def parse_period_key(period_key: str) -> dict:
    text = clean_text(period_key)

    if "|" not in text:
        return {
            "report_type": "",
            "year": None,
            "month": None,
            "kind": "UNKNOWN",
        }

    report_type, ym = text.split("|", 1)

    m = re.search(
        r"(?P<year>\d{4})\.(?P<month>\d{2})",
        ym,
    )

    year = (
        int(m.group("year"))
        if m
        else None
    )

    month = (
        int(m.group("month"))
        if m
        else None
    )

    if "사업보고서" in report_type:
        kind = "FY"
    elif "반기보고서" in report_type:
        kind = "H1"
    elif "분기보고서" in report_type:
        if month == 3:
            kind = "Q1"
        elif month == 9:
            kind = "Q3"
        else:
            kind = "QUARTER"
    else:
        kind = "UNKNOWN"

    return {
        "report_type": report_type,
        "year": year,
        "month": month,
        "kind": kind,
    }


def parse_generation(label: str):
    text = compact(label)

    m = re.search(
        r"제(\d+)기",
        text,
    )

    if not m:
        return None

    return int(
        m.group(1)
    )


def extract_years(label: str) -> list[int]:
    return [
        int(x)
        for x
        in re.findall(
            r"(?<!\d)(20\d{2})(?!\d)",
            clean_text(label),
        )
    ]


def extract_months(label: str) -> list[int]:
    text = clean_text(label)

    months = []

    for m in re.finditer(
        r"20\d{2}[./\-년]\s*(\d{1,2})",
        text,
    ):
        try:
            months.append(
                int(
                    m.group(1)
                )
            )
        except Exception:
            pass

    return months


def score_candidate(
    column: str,
    period_info: dict,
    max_generation: int | None,
) -> dict:

    original = clean_text(column)
    text = compact(column)
    marker = marker_text(column)

    target_year = period_info["year"]
    target_month = period_info["month"]
    kind = period_info["kind"]

    score = 0
    reasons = []
    hard_reject = False
    confidence = "NONE"

    # --------------------------------------------------------
    # A. Explicit current / prior markers
    # --------------------------------------------------------

    current_marker = (
        "당기" in marker
        or "당분기" in marker
        or "당반기" in marker
    )

    prior_marker = (
        "전기" in marker
        or "전분기" in marker
        or "전반기" in marker
    )

    if current_marker:
        score += 100
        reasons.append(
            "explicit_current_marker"
        )

    if prior_marker:
        score -= 150
        reasons.append(
            "explicit_prior_marker"
        )
        hard_reject = True

    # --------------------------------------------------------
    # B. Explicit year check
    # --------------------------------------------------------

    years = extract_years(
        original
    )

    if (
        target_year is not None
        and years
    ):
        if target_year in years:
            score += 70
            reasons.append(
                "target_year_match"
            )
        else:
            score -= 120
            reasons.append(
                "target_year_mismatch"
            )
            hard_reject = True

    # --------------------------------------------------------
    # C. Explicit month/date check
    # --------------------------------------------------------

    months = extract_months(
        original
    )

    if (
        target_month is not None
        and months
    ):
        if target_month in months:
            score += 40
            reasons.append(
                "target_month_match"
            )
        else:
            # Only hard reject when the label clearly carries dates.
            score -= 70
            reasons.append(
                "target_month_mismatch"
            )

    # --------------------------------------------------------
    # D. Report-type semantics
    # --------------------------------------------------------

    has_q1 = (
        "1분기" in text
        or "3개월" in text
    )

    has_h1 = (
        "반기" in text
        or "6개월" in text
    )

    has_q3 = (
        "3분기" in text
        or "9개월" in text
    )

    has_fy = (
        "기말" in text
        or "12월말" in text
        or "연말" in text
    )

    if kind == "Q1":
        if has_q1:
            score += 50
            reasons.append(
                "Q1_match"
            )
        elif has_h1 or has_q3:
            score -= 100
            reasons.append(
                "Q1_period_mismatch"
            )
            hard_reject = True

    elif kind == "H1":
        if has_h1:
            score += 50
            reasons.append(
                "H1_match"
            )

        if has_q1:
            score -= 120
            reasons.append(
                "H1_Q1_mismatch"
            )
            hard_reject = True

        if has_q3:
            score -= 120
            reasons.append(
                "H1_Q3_mismatch"
            )
            hard_reject = True

    elif kind == "Q3":
        if has_q3:
            score += 50
            reasons.append(
                "Q3_match"
            )

        if has_q1:
            score -= 120
            reasons.append(
                "Q3_Q1_mismatch"
            )
            hard_reject = True

        if has_h1:
            score -= 100
            reasons.append(
                "Q3_H1_mismatch"
            )
            hard_reject = True

    elif kind == "FY":
        # '기말' 자체는 current라는 증거가 아니다.
        # 예: 2020 사업보고서 안에 2019년 기말 / 2018년 기말 표가 존재할 수 있다.
        # 따라서 FY 형식이라는 사실만 기록하고 점수는 주지 않는다.
        if has_fy:
            reasons.append(
                "FY_form_only"
            )

        if has_q1 or has_h1 or has_q3:
            score -= 120
            reasons.append(
                "FY_interim_mismatch"
            )
            hard_reject = True

    # --------------------------------------------------------
    # E. Generation-only fallback
    # --------------------------------------------------------

    generation = parse_generation(
        original
    )

    # IMPORTANT:
    # only a LOW confidence signal.
    # It must not beat explicit date/current-marker evidence.
    if (
        generation is not None
        and max_generation is not None
        and generation == max_generation
        and not years
        and not current_marker
        and not prior_marker
    ):
        score += 20
        reasons.append(
            "highest_generation_low_confidence"
        )

    # --------------------------------------------------------
    # F. Final confidence
    # --------------------------------------------------------

    if hard_reject:
        confidence = "REJECT"

    elif (
        "explicit_current_marker"
        in reasons
        or (
            "target_year_match"
            in reasons
            and "target_month_match"
            in reasons
        )
    ):
        confidence = "HIGH"

    elif (
        "target_year_match"
        in reasons
        or "Q1_match"
        in reasons
        or "H1_match"
        in reasons
        or "Q3_match"
        in reasons
    ):
        confidence = "MEDIUM"

    elif (
        "highest_generation_low_confidence"
        in reasons
    ):
        confidence = "LOW"

    else:
        confidence = "NONE"

    return {
        "column": original,
        "score": score,
        "reasons": reasons,
        "hard_reject": hard_reject,
        "confidence": confidence,
        "generation": generation,
    }


def load_candidates(value) -> list[dict]:
    if pd.isna(value):
        return []

    try:
        parsed = json.loads(
            str(value)
        )
    except Exception:
        return []

    if not isinstance(
        parsed,
        list,
    ):
        return []

    return [
        x
        for x in parsed
        if isinstance(
            x,
            dict,
        )
    ]


def resolve_row(row: pd.Series) -> dict:
    period_info = parse_period_key(
        row.get(
            "canonical_period_key",
            "",
        )
    )

    candidates = load_candidates(
        row.get(
            "raw_numeric_candidates_json"
        )
    )

    if not candidates:
        return {
            "v2_status":
            "no_numeric_candidates",

            "v2_selected_column":
            None,

            "v2_selected_numeric":
            np.nan,

            "v2_selected_score":
            np.nan,

            "v2_selected_confidence":
            "NONE",

            "v2_selected_reasons":
            "",

            "v2_all_candidates":
            "[]",
        }

    generations = [
        parse_generation(
            c.get(
                "column",
                "",
            )
        )
        for c in candidates
    ]

    generations = [
        g
        for g in generations
        if g is not None
    ]

    max_generation = (
        max(
            generations
        )
        if generations
        else None
    )

    scored = []

    for c in candidates:
        result = score_candidate(
            c.get(
                "column",
                "",
            ),
            period_info,
            max_generation,
        )

        result[
            "numeric"
        ] = pd.to_numeric(
            c.get(
                "numeric"
            ),
            errors="coerce",
        )

        result[
            "raw"
        ] = c.get(
            "raw"
        )

        scored.append(
            result
        )

    eligible = [
        x
        for x in scored
        if (
            not x[
                "hard_reject"
            ]
            and x[
                "confidence"
            ]
            in {
                "HIGH",
                "MEDIUM",
                "LOW",
            }
        )
    ]

    if not eligible:
        status = (
            "no_current_candidate"
        )

        chosen = None

    else:
        eligible = sorted(
            eligible,
            key=lambda x: (
                x[
                    "score"
                ],
                {
                    "HIGH": 3,
                    "MEDIUM": 2,
                    "LOW": 1,
                }.get(
                    x[
                        "confidence"
                    ],
                    0,
                ),
            ),
            reverse=True,
        )

        top_score = eligible[
            0
        ][
            "score"
        ]

        top = [
            x
            for x in eligible
            if x[
                "score"
            ]
            == top_score
        ]

        # Duplicate logical column names with identical numeric values
        # are treated as the same candidate.
        unique_top = {}

        for x in top:
            key = (
                x[
                    "column"
                ],
                x[
                    "numeric"
                ],
            )

            unique_top[
                key
            ] = x

        top = list(
            unique_top.values()
        )

        if len(
            top
        ) == 1:
            chosen = top[
                0
            ]

            if (
                chosen[
                    "confidence"
                ]
                == "LOW"
            ):
                status = (
                    "low_confidence_candidate"
                )
            else:
                status = (
                    "selected"
                )

        else:
            chosen = None
            status = (
                "ambiguous_top_candidates"
            )

    if chosen is None:
        return {
            "v2_status":
            status,

            "v2_selected_column":
            None,

            "v2_selected_numeric":
            np.nan,

            "v2_selected_score":
            np.nan,

            "v2_selected_confidence":
            "NONE",

            "v2_selected_reasons":
            "",

            "v2_all_candidates":
            json.dumps(
                scored,
                ensure_ascii=False,
            ),
        }

    return {
        "v2_status":
        status,

        "v2_selected_column":
        chosen[
            "column"
        ],

        "v2_selected_numeric":
        chosen[
            "numeric"
        ],

        "v2_selected_score":
        chosen[
            "score"
        ],

        "v2_selected_confidence":
        chosen[
            "confidence"
        ],

        "v2_selected_reasons":
        " | ".join(
            chosen[
                "reasons"
            ]
        ),

        "v2_all_candidates":
        json.dumps(
            scored,
            ensure_ascii=False,
        ),
    }


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

    required = [
        "canonical_period_key",
        "raw_numeric_candidates_json",
    ]

    missing_cols = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing_cols:
        raise RuntimeError(
            "H6B6B output에 필요한 컬럼이 없습니다: "
            + ", ".join(
                missing_cols
            )
        )

    resolved = df.apply(
        resolve_row,
        axis=1,
        result_type="expand",
    )

    out = pd.concat(
        [
            df,
            resolved,
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
        + "=" * 115
    )

    print(
        "05A5-H6B6C CURRENT-PERIOD RESOLVER V2 DIAGNOSTIC"
    )

    print(
        "=" * 115
    )

    print(
        f"\nRows: "
        f"{len(out):,}"
    )

    print(
        "\n[V2 status]"
    )

    print(
        out[
            "v2_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[V2 confidence]"
    )

    print(
        out[
            "v2_selected_confidence"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    show_cols = [
        c
        for c in [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "problem_class",
            "account_family",
            "raw_columns_json",
            "raw_numeric_candidates_json",
            "c4_selected_column",
            "v2_status",
            "v2_selected_column",
            "v2_selected_numeric",
            "v2_selected_score",
            "v2_selected_confidence",
            "v2_selected_reasons",
        ]
        if c in out.columns
    ]

    print(
        "\n[Rows improved from C4 None -> V2 selected]"
    )

    improved = out.loc[
        out[
            "c4_selected_column"
        ].isna()
        & out[
            "v2_status"
        ].eq(
            "selected"
        )
    ].copy()

    if improved.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            280,
            "display.max_rows",
            80,
        ):
            print(
                improved[
                    show_cols
                ].to_string(
                    index=False
                )
            )

    print(
        "\n[Low-confidence generation-only candidates]"
    )

    low = out.loc[
        out[
            "v2_status"
        ].eq(
            "low_confidence_candidate"
        )
    ].copy()

    if low.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            280,
            "display.max_rows",
            80,
        ):
            print(
                low[
                    show_cols
                ].to_string(
                    index=False
                )
            )

    print(
        "\n[Still unresolved / rejected]"
    )

    unresolved = out.loc[
        out[
            "v2_status"
        ].isin(
            [
                "no_numeric_candidates",
                "no_current_candidate",
                "ambiguous_top_candidates",
            ]
        )
    ].copy()

    if unresolved.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            140,
            "display.width",
            260,
            "display.max_rows",
            100,
        ):
            print(
                unresolved[
                    show_cols
                ].to_string(
                    index=False
                )
            )

    print(
        "\n[Decisive examples]"
    )

    decisive_mask = (
        out[
            "raw_numeric_candidates_json"
        ]
        .astype(
            "string"
        )
        .fillna("")
        .str.contains(
            r"당\)기|당기|"
            r"2019년 12월말|"
            r"1분기",
            regex=True,
        )
    )

    decisive = out.loc[
        decisive_mask
    ].copy()

    if decisive.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            280,
            "display.max_rows",
            100,
        ):
            print(
                decisive[
                    show_cols
                ].to_string(
                    index=False
                )
            )

    print(
        f"\nOutput: "
        f"{OUTPUT}"
    )

    print(
        "\nPASS 기준:"
        "\n- 제100(당)기 / 제76(당)기 -> 괄호와 무관하게 HIGH 선택"
        "\n- 제99(전)기 / 제75(전)기 -> hard reject"
        "\n- 사업보고서|2020.12에서 2019/2018 명시열 -> current로 선택 금지"
        "\n- 반기보고서에서 1분기 열 -> hard reject"
        "\n- 기수+기말만 있는 제100기/99기/98기 -> LOW confidence까지만 허용"
        "\n- LOW는 아직 production 자동채택하지 않음"
    )


if __name__ == "__main__":
    main()
