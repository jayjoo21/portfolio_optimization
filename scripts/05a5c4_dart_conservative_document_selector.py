from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05A5-C4. Conservative Document Fallback Selector Audit
#
# 핵심 변경
# ------------------------------------------------------------
# 1) substring 매칭 대신 "row label exactness"를 평가
#    - 자본총계 != 부채와자본총계
#    - 영업수익 != 기타영업수익
#    - 반기순이익 != 지배주주지분반기순이익
#
# 2) 회사 본체의 연결 재무제표 문맥을 강하게 우선
#    - 연결 / 종속기업
#    - 현재 보고기간 column
#    - 재무상태표 / 손익계산서
#
# 3) 후보가 나쁘면 억지 선택 금지
#    - negative score
#    - selected family 없음
#    - current-period column 없음
#    -> unavailable / missing
#
# 4) table 하나가 모든 계정을 포함해야 한다고 강제하지 않음
#    account family별로 가장 좋은 candidate를 고름
#
# 5) 여러 신뢰도 높은 table이 거의 같은 값을 주면 consensus QA
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/dart_document_fact_candidates.parquet
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_document_account_candidates_v2.csv
#   dart_document_selected_values_v2.csv
#   dart_document_selector_validation_v2.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

DOC_PATH = (
    INTERIM_DIR
    / "dart_document_fact_candidates.parquet"
)

CANDIDATES_OUT = (
    INTERIM_DIR
    / "dart_document_account_candidates_v2.csv"
)

SELECTED_OUT = (
    INTERIM_DIR
    / "dart_document_selected_values_v2.csv"
)

VALIDATION_OUT = (
    INTERIM_DIR
    / "dart_document_selector_validation_v2.csv"
)


UNIT_MULTIPLIER = {
    "원": 1.0,
    "천원": 1_000.0,
    "백만원": 1_000_000.0,
    "억원": 100_000_000.0,
}


# total-line aliases.
# 긴/특수 표현을 명시적으로 처리하되, 하위 계정은 제외한다.
ACCOUNT_RULES = {
    "assets": {
        "exact": [
            "자산총계",
        ],
        "exclude": [
            "부채와자본총계",
        ],
    },
    "liabilities": {
        "exact": [
            "부채총계",
        ],
        "exclude": [],
    },
    "equity": {
        "exact": [
            "자본총계",
        ],
        "exclude": [
            "부채와자본총계",
            "지배기업소유주지분",
            "지배주주지분",
            "비지배지분",
        ],
    },
    "revenue": {
        "exact": [
            "매출액",
            "영업수익",
            "수익(매출액)",
        ],
        "exclude": [
            "기타영업수익",
            "보험영업수익",
            "투자영업수익",
        ],
    },
    "operating_income": {
        "exact": [
            "영업이익",
            "영업이익(손실)",
        ],
        "exclude": [],
    },
    "net_income": {
        "exact": [
            "당기순이익",
            "당기순이익(손실)",
            "반기순이익",
            "분기순이익",
            "연결당기순이익",
            "연결반기순이익",
            "연결분기순이익",
        ],
        "exclude": [
            "지배주주지분",
            "지배기업소유주",
            "비지배주주",
            "비지배지분",
            "대손준비금",
            "비상위험준비금",
            "조정이익",
        ],
    },
}


FALSE_POSITIVE_CONTEXT = [
    "최대주주",
    "배당",
    "보수",
    "성과급",
    "영업부문",
    "종류별 영업",
    "업 종류별",
    "업종별",
    "신탁관련",
    "여신상품",
    "대출상품",
]


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def safe_json_loads(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_label(value: Any) -> str:
    text = clean_text(value)

    # 앞쪽 번호/로마숫자/괄호/대괄호 정리
    text = re.sub(
        r"^[\[\(\s]*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXLCDM0-9]+[\]\)\.\s]*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^[가-힣]\.\s*",
        "",
        text,
    )

    text = text.replace(" ", "")
    text = text.replace(":", "")
    text = text.strip("[]()")

    return text


def extract_primary_row_label(
    row_cells_json: Any,
    row_text: Any,
) -> str:

    cells = safe_json_loads(
        row_cells_json
    )

    if isinstance(cells, list):
        for cell in cells:
            text = clean_text(cell)

            if not text:
                continue

            # 숫자만 있는 cell은 label 아님
            if re.fullmatch(
                r"[\d,\.\-\(\)\s]+",
                text,
            ):
                continue

            return text

    # fallback: row_text의 첫 pipe 이전
    text = clean_text(row_text)

    if "|" in text:
        return text.split(
            "|",
            1,
        )[0].strip()

    return text


def account_label_score(
    family: str,
    label: str,
) -> tuple[float, list[str]]:

    normalized = normalize_label(
        label
    )

    rules = ACCOUNT_RULES[
        family
    ]

    reasons = []

    # exclude가 먼저
    for token in rules["exclude"]:
        if normalize_label(token) in normalized:
            return (
                -100.0,
                [
                    f"excluded={token}"
                ],
            )

    for target in rules["exact"]:
        target_norm = normalize_label(
            target
        )

        if normalized == target_norm:
            return (
                30.0,
                [
                    f"exact={target}"
                ],
            )

        # 연결 + 계정명 형태는 허용
        if normalized in (
            normalize_label(
                "연결" + target
            ),
            normalize_label(
                "연결" + target.replace(
                    "당기",
                    "반기",
                )
            ),
        ):
            return (
                28.0,
                [
                    f"consolidated_exact={target}"
                ],
            )

    # exact가 아니면 production 후보로 거의 쓰지 않음.
    # 단, row label 끝에 total label이 정확히 붙는 경우 약한 후보.
    for target in rules["exact"]:
        target_norm = normalize_label(
            target
        )

        if normalized.endswith(
            target_norm
        ):
            return (
                8.0,
                [
                    f"suffix={target}"
                ],
            )

    return (
        -30.0,
        [
            "not_total_line"
        ],
    )


def extract_period_info(
    period_key: str,
) -> dict[str, Any]:

    text = str(period_key)

    match = re.search(
        r"\((\d{4})\.(\d{2})\)",
        text,
    )

    if not match:
        return {
            "year": None,
            "month": None,
            "kind": None,
        }

    year = int(match.group(1))
    month = int(match.group(2))

    if "반기보고서" in text:
        kind = "H1"
    elif "분기보고서" in text:
        if month <= 3:
            kind = "Q1"
        elif month >= 9:
            kind = "Q3"
        else:
            kind = "Q"
    elif "사업보고서" in text:
        kind = "FY"
    else:
        kind = None

    return {
        "year": year,
        "month": month,
        "kind": kind,
    }


def normalize_col_label(
    value: Any,
) -> str:

    text = clean_text(
        value
    )

    text = re.sub(
        r"\.\d+$",
        "",
        text,
    )

    return text


def current_column_score(
    column: str,
    period_key: str,
    statement_type: str,
) -> tuple[float, list[str]]:

    col = normalize_col_label(
        column
    )

    info = extract_period_info(
        period_key
    )

    year = info["year"]
    month = info["month"]
    kind = info["kind"]

    score = 0.0
    reasons = []

    if year is not None and str(year) in col:
        score += 8
        reasons.append(
            f"year={year}"
        )

    if kind == "H1":
        if "당반기" in col:
            score += 12
            reasons.append(
                "당반기"
            )

        if "반기" in col:
            score += 5
            reasons.append(
                "반기"
            )

        if statement_type == "income":
            if "누적" in col:
                score += 15
                reasons.append(
                    "누적"
                )

            if "3개월" in col:
                score -= 10
                reasons.append(
                    "3개월_penalty"
                )

        if statement_type == "balance":
            if "반기말" in col:
                score += 12
                reasons.append(
                    "반기말"
                )

    elif kind == "Q1":
        if (
            "1분기" in col
            or "당분기" in col
            or "분기" in col
        ):
            score += 10
            reasons.append(
                "Q1"
            )

    elif kind == "Q3":
        if "당분기" in col:
            score += 12
            reasons.append(
                "당분기"
            )

        if "3분기" in col:
            score += 8
            reasons.append(
                "Q3"
            )

        if statement_type == "income":
            if "누적" in col:
                score += 15
                reasons.append(
                    "누적"
                )

            if "3개월" in col:
                score -= 10
                reasons.append(
                    "3개월_penalty"
                )

    elif kind == "FY":
        if (
            "당기" in col
            or "기말" in col
        ):
            score += 10
            reasons.append(
                "FY_current"
            )

    # exact month hint
    if (
        year is not None
        and month is not None
    ):
        tokens = [
            f"{year}년 {month}월",
            f"{year}년 {month:02d}월",
            f"{year}.{month:02d}",
        ]

        if any(
            token in col
            for token in tokens
        ):
            score += 6
            reasons.append(
                "year_month"
            )

    # prior-period penalties
    for token in [
        "전반기",
        "전분기",
        "전기",
        "전전기",
        "전년도",
        "전년",
    ]:
        if token in col:
            score -= 20
            reasons.append(
                f"{token}_penalty"
            )

    # current H1인데 Q1 column이면 강한 reject
    if (
        kind == "H1"
        and "1분기" in col
    ):
        score -= 30
        reasons.append(
            "H1_Q1_mismatch"
        )

    return (
        score,
        reasons,
    )


def context_score(
    heading_context: Any,
    statement_type: str,
) -> tuple[float, list[str]]:

    context = clean_text(
        heading_context
    )

    score = 0.0
    reasons = []

    # consolidated is preferred
    if (
        "연결" in context
        or "종속기업" in context
    ):
        score += 25
        reasons.append(
            "consolidated"
        )

    if (
        "별도" in context
        or "개별재무제표" in context
    ):
        score -= 40
        reasons.append(
            "separate_penalty"
        )

    if statement_type == "balance":
        if "연결재무상태표" in context:
            score += 25
            reasons.append(
                "연결재무상태표"
            )
        elif "재무상태표" in context:
            score += 12
            reasons.append(
                "재무상태표"
            )

    else:
        if "연결손익계산서" in context:
            score += 25
            reasons.append(
                "연결손익계산서"
            )

        if "연결포괄손익계산서" in context:
            score += 25
            reasons.append(
                "연결포괄손익계산서"
            )

        if "포괄손익계산서" in context:
            score += 12
            reasons.append(
                "포괄손익계산서"
            )
        elif "손익계산서" in context:
            score += 12
            reasons.append(
                "손익계산서"
            )

    if "요약" in context:
        # summary 자체는 허용하지만 detailed보다 낮게
        score -= 3
        reasons.append(
            "summary_penalty"
        )

    for token in FALSE_POSITIVE_CONTEXT:
        if token in context:
            score -= 50
            reasons.append(
                f"{token}_penalty"
            )

    return (
        score,
        reasons,
    )


def parse_numeric_candidates(
    value: Any,
) -> list[dict[str, Any]]:

    data = safe_json_loads(
        value
    )

    if not isinstance(
        data,
        list,
    ):
        return []

    result = []

    for item in data:
        if not isinstance(
            item,
            dict,
        ):
            continue

        numeric = item.get(
            "numeric"
        )

        if numeric is None:
            continue

        raw = clean_text(
            item.get(
                "raw"
            )
        )

        # 한 cell에 여러 재무숫자가 붙어버린 malformed case 감지
        numeric_tokens = re.findall(
            r"(?<!\d)[\(\-]?\d[\d,]*(?:\.\d+)?\)?",
            raw,
        )

        # 단순 note 번호가 앞에 있는 경우를 제외하고
        # 실제 숫자 토큰이 2개 이상이면 merged-cell로 간주
        if len(
            numeric_tokens
        ) > 1:
            item = {
                **item,
                "merged_numeric_cell": True,
            }
        else:
            item = {
                **item,
                "merged_numeric_cell": False,
            }

        result.append(
            item
        )

    return result


def unit_multiplier(
    unit_hint: Any,
) -> float | None:

    return UNIT_MULTIPLIER.get(
        clean_text(
            unit_hint
        )
    )


# ------------------------------------------------------------
# Candidate scoring
# ------------------------------------------------------------

def build_account_candidates(
    doc: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in doc.iterrows():

        family = row[
            "account_family"
        ]

        statement_type = (
            "balance"
            if family in {
                "assets",
                "liabilities",
                "equity",
            }
            else "income"
        )

        primary_label = (
            extract_primary_row_label(
                row.get(
                    "row_cells_json"
                ),
                row.get(
                    "row_text"
                ),
            )
        )

        label_score, label_reasons = (
            account_label_score(
                family=family,
                label=primary_label,
            )
        )

        ctx_score, ctx_reasons = (
            context_score(
                row.get(
                    "heading_context"
                ),
                statement_type=statement_type,
            )
        )

        numeric_candidates = (
            parse_numeric_candidates(
                row.get(
                    "numeric_candidates_json"
                )
            )
        )

        scored_numbers = []

        for item in numeric_candidates:

            col = clean_text(
                item.get(
                    "column"
                )
            )

            col_score, col_reasons = (
                current_column_score(
                    column=col,
                    period_key=row[
                        "period_key"
                    ],
                    statement_type=statement_type,
                )
            )

            # merged numeric cell은 strict selector에서 제외
            if item.get(
                "merged_numeric_cell"
            ):
                col_score -= 40
                col_reasons.append(
                    "merged_numeric_cell_penalty"
                )

            scored_numbers.append(
                {
                    **item,
                    "column_score": (
                        col_score
                    ),
                    "column_reasons": (
                        " | ".join(
                            col_reasons
                        )
                    ),
                }
            )

        # best current-period numeric cell
        valid_numbers = [
            item
            for item in scored_numbers
            if item[
                "column_score"
            ] > 0
            and not item.get(
                "merged_numeric_cell"
            )
        ]

        valid_numbers = sorted(
            valid_numbers,
            key=lambda x: (
                x[
                    "column_score"
                ],
                x.get(
                    "column",
                    "",
                ),
            ),
            reverse=True,
        )

        if valid_numbers:
            best_score = (
                valid_numbers[0][
                    "column_score"
                ]
            )

            best_numbers = [
                item
                for item in valid_numbers
                if item[
                    "column_score"
                ] == best_score
            ]

            # normalize duplicated pandas columns
            unique = {}

            for item in best_numbers:
                key = (
                    normalize_col_label(
                        item.get(
                            "column"
                        )
                    ),
                    float(
                        item[
                            "numeric"
                        ]
                    ),
                )
                unique[
                    key
                ] = item

            best_numbers = list(
                unique.values()
            )
        else:
            best_numbers = []

        if len(
            best_numbers
        ) == 1:
            number = (
                best_numbers[0]
            )

            multiplier = unit_multiplier(
                row.get(
                    "unit_hint"
                )
            )

            raw_value = float(
                number[
                    "numeric"
                ]
            )

            value_krw = (
                raw_value
                * multiplier
                if multiplier
                is not None
                else np.nan
            )

            column_status = (
                "selected"
            )

            selected_column = (
                number.get(
                    "column"
                )
            )

            selected_column_score = (
                number[
                    "column_score"
                ]
            )

            selected_column_reason = (
                number[
                    "column_reasons"
                ]
            )
        elif len(
            best_numbers
        ) == 0:
            multiplier = unit_multiplier(
                row.get(
                    "unit_hint"
                )
            )
            raw_value = np.nan
            value_krw = np.nan
            column_status = (
                "no_current_value"
            )
            selected_column = None
            selected_column_score = np.nan
            selected_column_reason = None
        else:
            multiplier = unit_multiplier(
                row.get(
                    "unit_hint"
                )
            )
            raw_value = np.nan
            value_krw = np.nan
            column_status = (
                "ambiguous_column"
            )
            selected_column = None
            selected_column_score = (
                best_numbers[0][
                    "column_score"
                ]
            )
            selected_column_reason = None

        total_score = (
            label_score
            + ctx_score
            + (
                float(
                    selected_column_score
                )
                if pd.notna(
                    selected_column_score
                )
                else -20.0
            )
        )

        rows.append(
            {
                "stock_code": row[
                    "stock_code"
                ],
                "period_key": row[
                    "period_key"
                ],
                "rcept_no": row[
                    "rcept_no"
                ],
                "rcept_dt": row[
                    "rcept_dt"
                ],
                "table_index": row[
                    "table_index"
                ],
                "row_index": row[
                    "row_index"
                ],
                "statement_type": (
                    statement_type
                ),
                "account_family": (
                    family
                ),
                "primary_row_label": (
                    primary_label
                ),
                "matched_account_name": (
                    row[
                        "matched_account_name"
                    ]
                ),
                "label_score": (
                    label_score
                ),
                "label_reasons": (
                    " | ".join(
                        label_reasons
                    )
                ),
                "context_score": (
                    ctx_score
                ),
                "context_reasons": (
                    " | ".join(
                        ctx_reasons
                    )
                ),
                "column_status": (
                    column_status
                ),
                "selected_column": (
                    selected_column
                ),
                "selected_column_score": (
                    selected_column_score
                ),
                "selected_column_reason": (
                    selected_column_reason
                ),
                "raw_numeric_value": (
                    raw_value
                ),
                "unit_hint": row.get(
                    "unit_hint"
                ),
                "unit_multiplier": (
                    multiplier
                ),
                "selected_value_krw": (
                    value_krw
                ),
                "total_score": (
                    total_score
                ),
                "heading_context": (
                    row.get(
                        "heading_context"
                    )
                ),
                "row_text": row.get(
                    "row_text"
                ),
                "numeric_candidates_debug": json.dumps(
                    scored_numbers,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# Conservative selection
# ------------------------------------------------------------

def select_best_account_values(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "account_family",
    ]

    for keys, group in candidates.groupby(
        group_cols,
        sort=False,
    ):

        (
            stock_code,
            period_key,
            rcept_no,
            rcept_dt,
            family,
        ) = keys

        base = {
            "stock_code": (
                stock_code
            ),
            "period_key": (
                period_key
            ),
            "rcept_no": (
                rcept_no
            ),
            "rcept_dt": (
                rcept_dt
            ),
            "account_family": (
                family
            ),
        }

        # strict eligibility
        eligible = group.loc[
            group[
                "column_status"
            ].eq(
                "selected"
            )
            & group[
                "label_score"
            ].ge(
                20
            )
            & group[
                "context_score"
            ].ge(
                0
            )
            & group[
                "total_score"
            ].gt(
                25
            )
            & group[
                "selected_value_krw"
            ].notna()
        ].copy()

        if eligible.empty:
            rows.append(
                {
                    **base,
                    "selection_status": (
                        "no_reliable_candidate"
                    ),
                }
            )
            continue

        eligible = (
            eligible
            .sort_values(
                [
                    "total_score",
                    "context_score",
                    "label_score",
                    "selected_column_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        best_score = float(
            eligible[
                "total_score"
            ].iloc[0]
        )

        best = eligible.loc[
            eligible[
                "total_score"
            ].eq(
                best_score
            )
        ].copy()

        # 여러 최고 후보가 있을 때 값 consensus 확인
        values = best[
            "selected_value_krw"
        ].astype(float)

        if len(
            best
        ) > 1:
            median = float(
                values.median()
            )

            if median == 0:
                rel_spread = float(
                    values.max()
                    - values.min()
                )
            else:
                rel_spread = float(
                    (
                        values.max()
                        - values.min()
                    )
                    / abs(
                        median
                    )
                )

            # 0.1% 이내면 rounding-equivalent로 consensus
            if rel_spread <= 0.001:
                chosen = (
                    best
                    .sort_values(
                        [
                            "context_score",
                            "table_index",
                        ],
                        ascending=[
                            False,
                            True,
                        ],
                    )
                    .iloc[0]
                )

                selection_status = (
                    "selected_consensus"
                )
            else:
                rows.append(
                    {
                        **base,
                        "selection_status": (
                            "ambiguous_high_quality"
                        ),
                        "candidate_count": len(
                            best
                        ),
                        "candidate_values": (
                            " | ".join(
                                f"{x:.6g}"
                                for x in values
                            )
                        ),
                        "candidate_tables": (
                            " | ".join(
                                str(x)
                                for x in best[
                                    "table_index"
                                ]
                            )
                        ),
                    }
                )
                continue
        else:
            chosen = best.iloc[
                0
            ]
            selection_status = (
                "selected"
            )

        rows.append(
            {
                **base,
                "selection_status": (
                    selection_status
                ),
                "table_index": (
                    chosen[
                        "table_index"
                    ]
                ),
                "row_index": (
                    chosen[
                        "row_index"
                    ]
                ),
                "primary_row_label": (
                    chosen[
                        "primary_row_label"
                    ]
                ),
                "selected_column": (
                    chosen[
                        "selected_column"
                    ]
                ),
                "raw_numeric_value": (
                    chosen[
                        "raw_numeric_value"
                    ]
                ),
                "unit_hint": (
                    chosen[
                        "unit_hint"
                    ]
                ),
                "selected_value_krw": (
                    chosen[
                        "selected_value_krw"
                    ]
                ),
                "total_score": (
                    chosen[
                        "total_score"
                    ]
                ),
                "label_score": (
                    chosen[
                        "label_score"
                    ]
                ),
                "context_score": (
                    chosen[
                        "context_score"
                    ]
                ),
                "column_score": (
                    chosen[
                        "selected_column_score"
                    ]
                ),
                "heading_context": (
                    chosen[
                        "heading_context"
                    ]
                ),
                "row_text": (
                    chosen[
                        "row_text"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def build_validation(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    ok = selected.loc[
        selected[
            "selection_status"
        ].isin(
            [
                "selected",
                "selected_consensus",
            ]
        )
    ].copy()

    if ok.empty:
        return pd.DataFrame()

    wide = (
        ok.pivot_table(
            index=[
                "stock_code",
                "period_key",
                "rcept_no",
                "rcept_dt",
            ],
            columns="account_family",
            values="selected_value_krw",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    for row in wide.itertuples(
        index=False
    ):
        d = row._asdict()

        assets = d.get(
            "assets"
        )
        liabilities = d.get(
            "liabilities"
        )
        equity = d.get(
            "equity"
        )

        if (
            assets is not None
            and liabilities
            is not None
            and equity is not None
            and pd.notna(
                assets
            )
            and pd.notna(
                liabilities
            )
            and pd.notna(
                equity
            )
            and float(
                assets
            ) != 0
        ):
            gap = (
                float(
                    assets
                )
                - float(
                    liabilities
                )
                - float(
                    equity
                )
            )

            rel_gap = (
                abs(
                    gap
                )
                / abs(
                    float(
                        assets
                    )
                )
            )
        else:
            gap = np.nan
            rel_gap = np.nan

        rows.append(
            {
                "stock_code": d[
                    "stock_code"
                ],
                "period_key": d[
                    "period_key"
                ],
                "rcept_no": d[
                    "rcept_no"
                ],
                "rcept_dt": d[
                    "rcept_dt"
                ],
                "assets": (
                    assets
                ),
                "liabilities": (
                    liabilities
                ),
                "equity": (
                    equity
                ),
                "balance_gap": (
                    gap
                ),
                "balance_relative_gap": (
                    rel_gap
                ),
                "balance_equation_pass": (
                    bool(
                        rel_gap
                        <= 1e-8
                    )
                    if pd.notna(
                        rel_gap
                    )
                    else None
                ),
                "revenue": d.get(
                    "revenue"
                ),
                "operating_income": (
                    d.get(
                        "operating_income"
                    )
                ),
                "net_income": d.get(
                    "net_income"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:

    if not DOC_PATH.exists():
        raise FileNotFoundError(
            f"document candidate parquet이 없습니다: {DOC_PATH}"
        )

    doc = pd.read_parquet(
        DOC_PATH
    )

    doc[
        "rcept_dt"
    ] = pd.to_datetime(
        doc[
            "rcept_dt"
        ],
        errors="coerce",
    )

    candidates = (
        build_account_candidates(
            doc
        )
    )

    candidates.to_csv(
        CANDIDATES_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    selected = (
        select_best_account_values(
            candidates
        )
    )

    selected.to_csv(
        SELECTED_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    validation = (
        build_validation(
            selected
        )
    )

    validation.to_csv(
        VALIDATION_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-C4 CONSERVATIVE DOCUMENT SELECTOR"
    )
    print(
        "=" * 80
    )

    print(
        "\n[Selection status]"
    )

    print(
        selected[
            [
                "account_family",
                "selection_status",
            ]
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Selected / unresolved]"
    )

    show_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "account_family",
        "selection_status",
        "table_index",
        "row_index",
        "primary_row_label",
        "selected_column",
        "unit_hint",
        "selected_value_krw",
        "total_score",
        "heading_context",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in selected.columns
    ]

    print(
        selected[
            show_cols
        ]
        .sort_values(
            [
                "stock_code",
                "rcept_no",
                "account_family",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n[Balance equation validation]"
    )

    if validation.empty:
        print(
            "no validation rows"
        )
    else:
        print(
            validation.to_string(
                index=False
            )
        )

    print(
        "\n[Top eligible candidates]"
    )

    eligible = candidates.loc[
        candidates[
            "column_status"
        ].eq(
            "selected"
        )
        & candidates[
            "label_score"
        ].ge(
            20
        )
        & candidates[
            "context_score"
        ].ge(
            0
        )
    ].copy()

    cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "account_family",
        "table_index",
        "row_index",
        "primary_row_label",
        "selected_column",
        "selected_value_krw",
        "label_score",
        "context_score",
        "selected_column_score",
        "total_score",
        "heading_context",
    ]

    if eligible.empty:
        print(
            "0"
        )
    else:
        print(
            eligible[
                cols
            ]
            .sort_values(
                [
                    "stock_code",
                    "rcept_no",
                    "account_family",
                    "total_score",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False,
                ],
            )
            .head(80)
            .to_string(
                index=False
            )
        )

    print(
        f"\nCandidates : {CANDIDATES_OUT}"
    )
    print(
        f"Selected   : {SELECTED_OUT}"
    )
    print(
        f"Validation : {VALIDATION_OUT}"
    )

    print(
        "\n판단 기준:"
        "\n- no_reliable_candidate는 정상적인 strict-PIT 결과일 수 있음"
        "\n- 잘못된 값을 억지 선택하는 것보다 missing이 우선"
        "\n- balance equation pass가 되면 document balance selector 신뢰도 상승"
        "\n- 금융회사 revenue/영업수익의 경제적 의미는 일반 제조업과 분리해 해석"
        "\n- 이 audit이 통과하면 XBRL + document fallback을 합쳐 PIT snapshot builder로 이동"
    )


if __name__ == "__main__":
    main()
