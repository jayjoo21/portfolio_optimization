from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05A5-C3. DART document-fallback current-period selector audit
#
# 목적
# ------------------------------------------------------------
# document.xml fallback에서
# 1) balance sheet table
# 2) income statement table
# 를 각각 고르고,
# 해당 보고기간의 current-period column을 찾아
# 핵심 계정 후보를 1개씩 선택해 본다.
#
# IMPORTANT
# ------------------------------------------------------------
# 아직 production 확정이 아니다.
# 이번 결과에서:
# - balance equation
# - table heading
# - current-period column
# 을 검증한 뒤 production으로 승격한다.
#
# INPUT
# data/interim/dart/dart_document_fact_candidates.parquet
#
# ALSO READS (있으면)
# data/interim/dart/dart_xbrl_selected_values_audit.csv
# data/interim/dart/dart_xbrl_selection_validation.csv
#
# OUTPUT
# data/interim/dart/dart_document_selected_values_audit.csv
# data/interim/dart/dart_document_selector_validation.csv
# data/interim/dart/dart_pit_selector_combined_summary.csv
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

XBRL_SELECTED_PATH = (
    INTERIM_DIR
    / "dart_xbrl_selected_values_audit.csv"
)

XBRL_VALIDATION_PATH = (
    INTERIM_DIR
    / "dart_xbrl_selection_validation.csv"
)

SELECTED_OUT = (
    INTERIM_DIR
    / "dart_document_selected_values_audit.csv"
)

VALIDATION_OUT = (
    INTERIM_DIR
    / "dart_document_selector_validation.csv"
)

COMBINED_OUT = (
    INTERIM_DIR
    / "dart_pit_selector_combined_summary.csv"
)


BALANCE_FAMILIES = {
    "assets",
    "liabilities",
    "equity",
}

INCOME_FAMILIES = {
    "revenue",
    "operating_income",
    "net_income",
}


UNIT_MULTIPLIER = {
    "원": 1.0,
    "천원": 1_000.0,
    "백만원": 1_000_000.0,
    "억원": 100_000_000.0,
}


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def safe_json_loads(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None

    try:
        return json.loads(
            str(value)
        )
    except Exception:
        return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    return text.strip()


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

    year = int(
        match.group(1)
    )
    month = int(
        match.group(2)
    )

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
    label: str,
) -> str:
    text = clean_text(
        label
    )

    # pandas read_html duplicate-column suffix
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
    reasons: list[str] = []

    # --------------------------------------------------------
    # Explicit current markers
    # --------------------------------------------------------

    if "당반기" in col:
        score += 10
        reasons.append("당반기")

    if "당분기" in col:
        score += 10
        reasons.append("당분기")

    if "당기" in col:
        score += 6
        reasons.append("당기")

    if year is not None:
        year_s = str(year)

        if year_s in col:
            score += 5
            reasons.append(
                f"year={year}"
            )

    if month is not None:
        month_tokens = {
            f"{month}월",
            f"{month:02d}월",
            f"{year}.{month:02d}"
            if year is not None
            else "",
            f"{year}년 {month}월"
            if year is not None
            else "",
            f"{year}년 {month:02d}월"
            if year is not None
            else "",
        }

        month_tokens.discard("")

        if any(
            token in col
            for token in month_tokens
        ):
            score += 4
            reasons.append(
                f"month={month}"
            )

    # --------------------------------------------------------
    # Report-kind matching
    # --------------------------------------------------------

    if kind == "H1":
        if "반기" in col:
            score += 6
            reasons.append("H1")

        if statement_type == "income":
            if "누적" in col:
                score += 8
                reasons.append("누적")
            if "3개월" in col:
                score -= 5
                reasons.append("3개월_penalty")

        if statement_type == "balance":
            if "반기말" in col:
                score += 8
                reasons.append("반기말")

    elif kind == "Q1":
        if "1분기" in col or "분기" in col:
            score += 6
            reasons.append("Q1")

    elif kind == "Q3":
        if "3분기" in col:
            score += 6
            reasons.append("Q3")
        elif "분기" in col:
            score += 3
            reasons.append("quarter")

        if statement_type == "income":
            if "누적" in col:
                score += 8
                reasons.append("누적")
            if "3개월" in col:
                score -= 5
                reasons.append("3개월_penalty")

    elif kind == "FY":
        if "기말" in col:
            score += 6
            reasons.append("FY_end")
        if "사업연도" in col:
            score += 5
            reasons.append("FY")

    # --------------------------------------------------------
    # Prior-period penalties
    # --------------------------------------------------------

    prior_tokens = [
        "전반기",
        "전분기",
        "전기",
        "전전기",
        "전년도",
        "전년",
    ]

    for token in prior_tokens:
        if token in col:
            score -= 12
            reasons.append(
                f"{token}_penalty"
            )

    # H1인데 1분기 자료면 강한 penalty
    if (
        kind == "H1"
        and "1분기" in col
    ):
        score -= 15
        reasons.append(
            "H1_vs_Q1_penalty"
        )

    return score, reasons


def table_context_score(
    group: pd.DataFrame,
    statement_type: str,
) -> tuple[float, list[str]]:

    heading = " || ".join(
        group[
            "heading_context"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    score_reason = " || ".join(
        group[
            "statement_score_reason"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    combined = (
        heading
        + " || "
        + score_reason
    )

    score = 0.0
    reasons: list[str] = []

    # --------------------------------------------------------
    # consolidated / separate
    # --------------------------------------------------------

    if (
        "연결" in combined
        or "종속기업" in combined
    ):
        score += 12
        reasons.append(
            "consolidated"
        )

    if (
        "별도" in combined
        or "개별재무제표" in combined
    ):
        score -= 15
        reasons.append(
            "separate_penalty"
        )

    # --------------------------------------------------------
    # statement type
    # --------------------------------------------------------

    if statement_type == "balance":
        if "연결재무상태표" in combined:
            score += 12
            reasons.append(
                "연결재무상태표"
            )
        elif "재무상태표" in combined:
            score += 7
            reasons.append(
                "재무상태표"
            )

    else:
        if "연결손익계산서" in combined:
            score += 12
            reasons.append(
                "연결손익계산서"
            )

        if "연결포괄손익계산서" in combined:
            score += 12
            reasons.append(
                "연결포괄손익계산서"
            )

        if "포괄손익계산서" in combined:
            score += 7
            reasons.append(
                "포괄손익계산서"
            )
        elif "손익계산서" in combined:
            score += 7
            reasons.append(
                "손익계산서"
            )

    # summary is allowed but detailed statement preferred
    if "요약" in combined:
        score -= 2
        reasons.append(
            "summary_penalty"
        )

    # --------------------------------------------------------
    # obvious false-positive sections
    # --------------------------------------------------------

    false_tokens = [
        "최대주주",
        "배당",
        "보수",
        "성과급",
        "영업부문",
        "신탁",
        "여신상품",
        "대출상품",
        "종류별 영업",
    ]

    for token in false_tokens:
        if token in combined:
            score -= 20
            reasons.append(
                f"{token}_penalty"
            )

    return score, reasons


def unit_multiplier(
    unit_hint: Any,
) -> float | None:

    unit = clean_text(
        unit_hint
    )

    return UNIT_MULTIPLIER.get(
        unit
    )


def extract_numeric_for_best_column(
    row: pd.Series,
    period_key: str,
    statement_type: str,
) -> dict[str, Any]:

    numeric_candidates = safe_json_loads(
        row.get(
            "numeric_candidates_json"
        )
    )

    if not numeric_candidates:
        return {
            "column_status": "no_numeric_candidates",
        }

    scored = []

    for candidate in numeric_candidates:

        column = clean_text(
            candidate.get(
                "column"
            )
        )

        numeric = candidate.get(
            "numeric"
        )

        if numeric is None:
            continue

        score, reasons = (
            current_column_score(
                column=column,
                period_key=period_key,
                statement_type=statement_type,
            )
        )

        scored.append(
            {
                "column": column,
                "raw": candidate.get(
                    "raw"
                ),
                "numeric": float(
                    numeric
                ),
                "score": float(
                    score
                ),
                "reasons": " | ".join(
                    reasons
                ),
            }
        )

    if not scored:
        return {
            "column_status": "no_numeric_candidates",
        }

    scored = sorted(
        scored,
        key=lambda x: (
            x["score"],
            -len(
                x["column"]
            ),
        ),
        reverse=True,
    )

    best_score = scored[0][
        "score"
    ]

    best = [
        item
        for item in scored
        if item[
            "score"
        ] == best_score
    ]

    # 같은 semantic column의 duplicated .1 등에 의해
    # 같은 숫자가 반복될 수 있으므로 value까지 같으면 하나로 취급.
    unique = {}

    for item in best:
        key = (
            normalize_col_label(
                item["column"]
            ),
            item["numeric"],
        )
        unique[key] = item

    best = list(
        unique.values()
    )

    if best_score <= 0:
        return {
            "column_status": "no_positive_current_column",
            "best_column_score": best_score,
            "column_candidates": json.dumps(
                scored,
                ensure_ascii=False,
            ),
        }

    if len(best) != 1:
        return {
            "column_status": "ambiguous_current_column",
            "best_column_score": best_score,
            "column_candidates": json.dumps(
                best,
                ensure_ascii=False,
            ),
        }

    chosen = best[0]

    multiplier = unit_multiplier(
        row.get(
            "unit_hint"
        )
    )

    standardized_value = (
        chosen["numeric"]
        * multiplier
        if multiplier is not None
        else np.nan
    )

    return {
        "column_status": "selected",
        "selected_column": chosen[
            "column"
        ],
        "selected_column_score": (
            chosen[
                "score"
            ]
        ),
        "selected_column_reason": (
            chosen[
                "reasons"
            ]
        ),
        "raw_numeric_value": (
            chosen[
                "numeric"
            ]
        ),
        "unit_hint": row.get(
            "unit_hint"
        ),
        "unit_multiplier": (
            multiplier
        ),
        "selected_value_krw": (
            standardized_value
        ),
        "column_candidates": json.dumps(
            scored,
            ensure_ascii=False,
        ),
    }


# ------------------------------------------------------------
# Table selection
# ------------------------------------------------------------

def prepare_table_scores(
    doc: pd.DataFrame,
    statement_type: str,
) -> pd.DataFrame:

    target_families = (
        BALANCE_FAMILIES
        if statement_type == "balance"
        else INCOME_FAMILIES
    )

    rows = []

    group_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "table_index",
    ]

    for keys, group in doc.groupby(
        group_cols,
        sort=False,
    ):

        families = set(
            group[
                "account_family"
            ].dropna()
        )

        relevant = (
            families
            & target_families
        )

        if not relevant:
            continue

        (
            stock_code,
            period_key,
            rcept_no,
            rcept_dt,
            table_index,
        ) = keys

        context_score, context_reasons = (
            table_context_score(
                group=group,
                statement_type=statement_type,
            )
        )

        row_results = []

        for _, row in group.iterrows():

            family = row[
                "account_family"
            ]

            if family not in target_families:
                continue

            selected = (
                extract_numeric_for_best_column(
                    row=row,
                    period_key=period_key,
                    statement_type=statement_type,
                )
            )

            row_results.append(
                {
                    "family": family,
                    "matched_account_name": row[
                        "matched_account_name"
                    ],
                    "row_index": row[
                        "row_index"
                    ],
                    "row_text": row[
                        "row_text"
                    ],
                    **selected,
                }
            )

        selected_rows = [
            row
            for row in row_results
            if row.get(
                "column_status"
            ) == "selected"
        ]

        selected_families = set(
            row[
                "family"
            ]
            for row in selected_rows
        )

        coverage = len(
            selected_families
        )

        # period-column quality
        column_score = (
            sum(
                float(
                    row.get(
                        "selected_column_score",
                        0,
                    )
                )
                for row in selected_rows
            )
            / len(
                selected_rows
            )
            if selected_rows
            else -10
        )

        completeness_bonus = (
            coverage * 5
        )

        all_three_bonus = (
            8
            if coverage == 3
            else 0
        )

        final_score = (
            context_score
            + column_score
            + completeness_bonus
            + all_three_bonus
        )

        first = group.iloc[0]

        rows.append(
            {
                "stock_code": stock_code,
                "period_key": period_key,
                "rcept_no": rcept_no,
                "rcept_dt": rcept_dt,
                "table_index": table_index,
                "statement_type": statement_type,
                "table_score": final_score,
                "context_score": context_score,
                "context_reasons": (
                    " | ".join(
                        context_reasons
                    )
                ),
                "selected_family_count": coverage,
                "selected_families": (
                    " | ".join(
                        sorted(
                            selected_families
                        )
                    )
                ),
                "column_score_mean": column_score,
                "unit_hint": first.get(
                    "unit_hint"
                ),
                "heading_context": first.get(
                    "heading_context"
                ),
                "columns_json": first.get(
                    "columns_json"
                ),
                "row_selection_json": json.dumps(
                    row_results,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "rank_within_receipt"
    ] = (
        result.groupby(
            [
                "stock_code",
                "period_key",
                "rcept_no",
                "statement_type",
            ]
        )["table_score"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return (
        result.sort_values(
            [
                "stock_code",
                "period_key",
                "rcept_no",
                "statement_type",
                "rank_within_receipt",
                "table_index",
            ]
        )
        .reset_index(drop=True)
    )


def choose_top_table(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    if scores.empty:
        return scores

    output = []

    group_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "statement_type",
    ]

    for keys, group in scores.groupby(
        group_cols,
        sort=False,
    ):

        group = group.sort_values(
            [
                "table_score",
                "table_index",
            ],
            ascending=[
                False,
                True,
            ],
        )

        best_score = group[
            "table_score"
        ].iloc[0]

        best = group.loc[
            group[
                "table_score"
            ].eq(
                best_score
            )
        ]

        base = {
            "stock_code": keys[0],
            "period_key": keys[1],
            "rcept_no": keys[2],
            "rcept_dt": keys[3],
            "statement_type": keys[4],
        }

        if len(best) != 1:
            output.append(
                {
                    **base,
                    "table_selection_status": "ambiguous_table",
                    "candidate_table_count": len(
                        best
                    ),
                    "candidate_tables": " | ".join(
                        str(x)
                        for x in best[
                            "table_index"
                        ].tolist()
                    ),
                }
            )
            continue

        row = best.iloc[0]

        output.append(
            {
                **base,
                "table_selection_status": "selected",
                "selected_table_index": row[
                    "table_index"
                ],
                "selected_table_score": row[
                    "table_score"
                ],
                "context_score": row[
                    "context_score"
                ],
                "context_reasons": row[
                    "context_reasons"
                ],
                "selected_family_count": row[
                    "selected_family_count"
                ],
                "selected_families": row[
                    "selected_families"
                ],
                "unit_hint": row[
                    "unit_hint"
                ],
                "heading_context": row[
                    "heading_context"
                ],
                "columns_json": row[
                    "columns_json"
                ],
                "row_selection_json": row[
                    "row_selection_json"
                ],
            }
        )

    return pd.DataFrame(
        output
    )


# ------------------------------------------------------------
# Expand selected table to account values
# ------------------------------------------------------------

def expand_selected_values(
    table_selection: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, table in table_selection.iterrows():

        base = {
            "stock_code": table[
                "stock_code"
            ],
            "period_key": table[
                "period_key"
            ],
            "rcept_no": table[
                "rcept_no"
            ],
            "rcept_dt": table[
                "rcept_dt"
            ],
            "statement_type": table[
                "statement_type"
            ],
            "table_selection_status": table[
                "table_selection_status"
            ],
        }

        if table[
            "table_selection_status"
        ] != "selected":
            rows.append(
                {
                    **base,
                    "account_family": None,
                    "value_selection_status": (
                        "table_not_selected"
                    ),
                }
            )
            continue

        details = safe_json_loads(
            table[
                "row_selection_json"
            ]
        )

        if not details:
            rows.append(
                {
                    **base,
                    "account_family": None,
                    "value_selection_status": (
                        "no_row_details"
                    ),
                }
            )
            continue

        # family별 selected row 후보
        by_family: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for item in details:
            family = item.get(
                "family"
            )

            if (
                not family
                or item.get(
                    "column_status"
                ) != "selected"
            ):
                continue

            by_family.setdefault(
                family,
                [],
            ).append(
                item
            )

        expected = (
            BALANCE_FAMILIES
            if table[
                "statement_type"
            ] == "balance"
            else INCOME_FAMILIES
        )

        for family in sorted(
            expected
        ):

            candidates = by_family.get(
                family,
                [],
            )

            family_base = {
                **base,
                "selected_table_index": table[
                    "selected_table_index"
                ],
                "selected_table_score": table[
                    "selected_table_score"
                ],
                "account_family": family,
                "unit_hint": table[
                    "unit_hint"
                ],
                "heading_context": table[
                    "heading_context"
                ],
                "columns_json": table[
                    "columns_json"
                ],
            }

            if not candidates:
                rows.append(
                    {
                        **family_base,
                        "value_selection_status": (
                            "missing_family"
                        ),
                    }
                )
                continue

            # 동일 family가 여러 row에서 발견되면
            # exact account name 단순성 + row text로 자동 확정하지 않음.
            # 동일 value면 하나로 묶고, 값이 다르면 ambiguous.
            unique_values = {}

            for item in candidates:
                value = item.get(
                    "selected_value_krw"
                )

                key = (
                    value
                    if value is not None
                    else item.get(
                        "raw_numeric_value"
                    )
                )

                unique_values.setdefault(
                    key,
                    [],
                ).append(
                    item
                )

            if len(
                unique_values
            ) != 1:
                rows.append(
                    {
                        **family_base,
                        "value_selection_status": (
                            "ambiguous_rows"
                        ),
                        "candidate_rows_json": json.dumps(
                            candidates,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
                continue

            item = candidates[0]

            rows.append(
                {
                    **family_base,
                    "value_selection_status": (
                        "selected"
                    ),
                    "matched_account_name": item.get(
                        "matched_account_name"
                    ),
                    "row_index": item.get(
                        "row_index"
                    ),
                    "selected_column": item.get(
                        "selected_column"
                    ),
                    "selected_column_score": item.get(
                        "selected_column_score"
                    ),
                    "selected_column_reason": item.get(
                        "selected_column_reason"
                    ),
                    "raw_numeric_value": item.get(
                        "raw_numeric_value"
                    ),
                    "unit_multiplier": item.get(
                        "unit_multiplier"
                    ),
                    "selected_value_krw": item.get(
                        "selected_value_krw"
                    ),
                    "row_text": item.get(
                        "row_text"
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
    values: pd.DataFrame,
) -> pd.DataFrame:

    selected = values.loc[
        values[
            "value_selection_status"
        ].eq(
            "selected"
        )
        & values[
            "account_family"
        ].notna()
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    wide = (
        selected.pivot_table(
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
        data = row._asdict()

        assets = data.get(
            "assets"
        )
        liabilities = data.get(
            "liabilities"
        )
        equity = data.get(
            "equity"
        )

        if (
            pd.notna(assets)
            and pd.notna(liabilities)
            and pd.notna(equity)
            and assets != 0
        ):
            gap = (
                float(assets)
                - float(liabilities)
                - float(equity)
            )

            rel_gap = (
                abs(gap)
                / abs(
                    float(assets)
                )
            )
        else:
            gap = np.nan
            rel_gap = np.nan

        rows.append(
            {
                "stock_code": data[
                    "stock_code"
                ],
                "period_key": data[
                    "period_key"
                ],
                "rcept_no": data[
                    "rcept_no"
                ],
                "rcept_dt": data[
                    "rcept_dt"
                ],
                "assets": assets,
                "liabilities": liabilities,
                "equity": equity,
                "balance_gap": gap,
                "balance_relative_gap": (
                    rel_gap
                ),
                "balance_equation_pass": (
                    bool(
                        rel_gap <= 1e-8
                    )
                    if pd.notna(
                        rel_gap
                    )
                    else None
                ),
                "revenue": data.get(
                    "revenue"
                ),
                "operating_income": data.get(
                    "operating_income"
                ),
                "net_income": data.get(
                    "net_income"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# Combined summary
# ------------------------------------------------------------

def build_combined_summary(
    doc_values: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    if XBRL_SELECTED_PATH.exists():
        xbrl = pd.read_csv(
            XBRL_SELECTED_PATH,
            dtype={
                "stock_code": str,
                "rcept_no": str,
            },
        )

        xbrl_group = (
            xbrl.groupby(
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                ]
            )
            .agg(
                roles=(
                    "role",
                    "size",
                ),
                selected_roles=(
                    "selection_status",
                    lambda s: int(
                        s.eq(
                            "selected"
                        ).sum()
                    ),
                ),
                ambiguous_roles=(
                    "selection_status",
                    lambda s: int(
                        s.eq(
                            "ambiguous"
                        ).sum()
                    ),
                ),
            )
            .reset_index()
        )

        for item in xbrl_group.to_dict(
            orient="records"
        ):
            rows.append(
                {
                    "source_type": "xbrl",
                    **item,
                }
            )

    if not doc_values.empty:
        doc_group = (
            doc_values.groupby(
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                ]
            )
            .agg(
                roles=(
                    "account_family",
                    lambda s: int(
                        s.notna().sum()
                    ),
                ),
                selected_roles=(
                    "value_selection_status",
                    lambda s: int(
                        s.eq(
                            "selected"
                        ).sum()
                    ),
                ),
                ambiguous_roles=(
                    "value_selection_status",
                    lambda s: int(
                        s.str.contains(
                            "ambiguous",
                            na=False,
                        ).sum()
                    ),
                ),
            )
            .reset_index()
        )

        for item in doc_group.to_dict(
            orient="records"
        ):
            rows.append(
                {
                    "source_type": (
                        "document_fallback"
                    ),
                    **item,
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

    balance_scores = (
        prepare_table_scores(
            doc=doc,
            statement_type="balance",
        )
    )

    income_scores = (
        prepare_table_scores(
            doc=doc,
            statement_type="income",
        )
    )

    scores = pd.concat(
        [
            balance_scores,
            income_scores,
        ],
        ignore_index=True,
        sort=False,
    )

    chosen_tables = choose_top_table(
        scores
    )

    values = expand_selected_values(
        chosen_tables
    )

    values.to_csv(
        SELECTED_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    validation = build_validation(
        values
    )

    validation.to_csv(
        VALIDATION_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    combined = build_combined_summary(
        values
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print concise QA
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-C3 DOCUMENT SELECTOR AUDIT"
    )
    print(
        "=" * 80
    )

    print(
        "\n[Selected tables]"
    )

    table_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "statement_type",
        "table_selection_status",
        "selected_table_index",
        "selected_table_score",
        "selected_family_count",
        "unit_hint",
        "context_reasons",
        "heading_context",
        "columns_json",
    ]

    table_cols = [
        col
        for col in table_cols
        if col in chosen_tables.columns
    ]

    print(
        chosen_tables[
            table_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\n[Value selection status]"
    )

    print(
        values[
            [
                "statement_type",
                "account_family",
                "value_selection_status",
            ]
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Selected values]"
    )

    show_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "statement_type",
        "account_family",
        "value_selection_status",
        "selected_table_index",
        "matched_account_name",
        "selected_column",
        "raw_numeric_value",
        "unit_hint",
        "selected_value_krw",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in values.columns
    ]

    print(
        values[
            show_cols
        ]
        .sort_values(
            [
                "stock_code",
                "rcept_no",
                "statement_type",
                "account_family",
            ],
            na_position="last",
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
        "\n[XBRL concise status]"
    )

    if XBRL_SELECTED_PATH.exists():
        xbrl = pd.read_csv(
            XBRL_SELECTED_PATH,
        )

        print(
            xbrl[
                [
                    "role",
                    "selection_status",
                ]
            ]
            .value_counts()
            .to_string()
        )

        if XBRL_VALIDATION_PATH.exists():
            xv = pd.read_csv(
                XBRL_VALIDATION_PATH
            )

            if (
                "balance_equation_pass"
                in xv.columns
            ):
                print(
                    "\nXBRL balance equation:"
                )

                print(
                    xv[
                        "balance_equation_pass"
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .to_string()
                )
    else:
        print(
            "XBRL selected audit file not found"
        )

    print(
        f"\nDocument selected : {SELECTED_OUT}"
    )
    print(
        f"Document validation: {VALIDATION_OUT}"
    )
    print(
        f"Combined summary  : {COMBINED_OUT}"
    )

    print(
        "\n판단 기준:"
        "\n- document balance table은 CFS + current period +"
        " balance equation pass가 모두 맞아야 production 승격"
        "\n- income은 반기/Q3에서 '누적' column 우선"
        "\n- separate/최대주주/배당/보수/영업부문 table은 강한 penalty"
        "\n- revenue가 금융회사에서 일반 제조업과 같은 의미라고"
        " 가정하지 않음"
        "\n- ambiguous는 절대 자동 채택하지 않음"
    )


if __name__ == "__main__":
    main()
