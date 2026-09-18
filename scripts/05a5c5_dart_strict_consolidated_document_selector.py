from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05A5-C5. Strict Consolidated Document Selector Audit
#
# C4에서 남은 문제 해결
# ------------------------------------------------------------
# 1) "연결" 근거 없는 income/revenue는 사용하지 않음.
#    예: 우리은행 가.영업실적 revenue -> reject
#
# 2) row label 자체가 [연결당기순이익]처럼 명시적 연결이면
#    consolidated evidence로 인정.
#    예: 메리츠 연결당기순이익 209,833백만원 우선
#
# 3) 별도/개별은 계속 reject
#
# 4) balance는 연결 근거 + current-period + total line exact-match
#
# 목표
# ------------------------------------------------------------
# document fallback에서 "틀린 숫자보다 missing" 원칙을 확정.
#
# INPUT
# data/interim/dart/dart_document_account_candidates_v2.csv
#
# OUTPUT
# data/interim/dart/
#   dart_document_selected_values_v3.csv
#   dart_document_selector_validation_v3.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

CANDIDATE_PATH = (
    INTERIM_DIR
    / "dart_document_account_candidates_v2.csv"
)

SELECTED_OUT = (
    INTERIM_DIR
    / "dart_document_selected_values_v3.csv"
)

VALIDATION_OUT = (
    INTERIM_DIR
    / "dart_document_selector_validation_v3.csv"
)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalized_label(value: Any) -> str:
    text = clean_text(value)

    text = text.strip("[]()")
    text = re.sub(
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXLCDM0-9]+[\.\s]*",
        "",
        text,
        flags=re.I,
    )
    text = text.replace(" ", "")
    return text


def explicit_consolidated_label(
    label: Any,
) -> bool:

    norm = normalized_label(
        label
    )

    consolidated_tokens = [
        "연결당기순이익",
        "연결반기순이익",
        "연결분기순이익",
        "연결영업이익",
        "연결영업수익",
        "연결매출액",
        "연결자산총계",
        "연결부채총계",
        "연결자본총계",
    ]

    return any(
        norm == token
        or norm.startswith(token)
        for token in consolidated_tokens
    )


def consolidated_evidence(
    row: pd.Series,
) -> tuple[float, list[str]]:

    heading = clean_text(
        row.get(
            "heading_context"
        )
    )

    label = clean_text(
        row.get(
            "primary_row_label"
        )
    )

    reasons = []
    score = 0.0

    # explicit separate evidence => hard reject
    separate_tokens = [
        "별도",
        "개별재무제표",
        "개별재무",
    ]

    if any(
        token in heading
        for token in separate_tokens
    ):
        return (
            -100.0,
            [
                "separate_heading"
            ],
        )

    if (
        "연결" in heading
        or "종속기업" in heading
    ):
        score += 40.0
        reasons.append(
            "consolidated_heading"
        )

    if explicit_consolidated_label(
        label
    ):
        score += 30.0
        reasons.append(
            "explicit_consolidated_label"
        )

    # C4 context_score가 이미 연결문맥을 포착했다면 보조 증거
    try:
        old_context = float(
            row.get(
                "context_score",
                0,
            )
        )
    except Exception:
        old_context = 0.0

    if old_context >= 20:
        score += 10.0
        reasons.append(
            "c4_context_support"
        )

    return (
        score,
        reasons,
    )


def family_priority_bonus(
    row: pd.Series,
) -> float:
    """
    동일 account family 내에서 total/consolidated 표현을 조금 더 선호.
    """
    family = clean_text(
        row.get(
            "account_family"
        )
    )

    label = normalized_label(
        row.get(
            "primary_row_label"
        )
    )

    if family == "net_income":
        if label in (
            "연결당기순이익",
            "연결반기순이익",
            "연결분기순이익",
        ):
            return 15.0

        if label in (
            "당기순이익",
            "반기순이익",
            "분기순이익",
        ):
            return 5.0

    if family == "revenue":
        if label in (
            "연결영업수익",
            "연결매출액",
        ):
            return 15.0

    return 0.0


# ------------------------------------------------------------
# selection
# ------------------------------------------------------------

def select_values(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    working = candidates.copy()

    evidence_scores = []
    evidence_reasons = []

    for _, row in working.iterrows():
        score, reasons = (
            consolidated_evidence(
                row
            )
        )

        evidence_scores.append(
            score
        )

        evidence_reasons.append(
            " | ".join(
                reasons
            )
        )

    working[
        "consolidated_evidence_score"
    ] = evidence_scores

    working[
        "consolidated_evidence_reason"
    ] = evidence_reasons

    working[
        "family_priority_bonus"
    ] = working.apply(
        family_priority_bonus,
        axis=1,
    )

    # C4 total_score + explicit consolidated evidence
    working[
        "strict_score"
    ] = (
        pd.to_numeric(
            working[
                "total_score"
            ],
            errors="coerce",
        )
        .fillna(
            -999
        )
        + working[
            "consolidated_evidence_score"
        ]
        + working[
            "family_priority_bonus"
        ]
    )

    rows = []

    group_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "account_family",
    ]

    for keys, group in working.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):

        (
            stock_code,
            period_key,
            rcept_no,
            rcept_dt,
            family,
        ) = keys

        base = {
            "stock_code": stock_code,
            "period_key": period_key,
            "rcept_no": rcept_no,
            "rcept_dt": rcept_dt,
            "account_family": family,
        }

        # ----------------------------------------------------
        # strict eligibility
        # ----------------------------------------------------
        eligible = group.loc[
            group[
                "column_status"
            ].eq(
                "selected"
            )
            & pd.to_numeric(
                group[
                    "label_score"
                ],
                errors="coerce",
            ).ge(
                20
            )
            & pd.to_numeric(
                group[
                    "consolidated_evidence_score"
                ],
                errors="coerce",
            ).ge(
                20
            )
            & pd.to_numeric(
                group[
                    "selected_value_krw"
                ],
                errors="coerce",
            ).notna()
        ].copy()

        if eligible.empty:
            rows.append(
                {
                    **base,
                    "selection_status": (
                        "no_reliable_consolidated_candidate"
                    ),
                }
            )
            continue

        eligible = eligible.sort_values(
            [
                "strict_score",
                "consolidated_evidence_score",
                "label_score",
                "selected_column_score",
                "table_index",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
        )

        best_score = float(
            eligible[
                "strict_score"
            ].iloc[0]
        )

        best = eligible.loc[
            np.isclose(
                eligible[
                    "strict_score"
                ].astype(float),
                best_score,
            )
        ].copy()

        # ----------------------------------------------------
        # multiple top candidates:
        # if values effectively identical -> consensus
        # otherwise unresolved
        # ----------------------------------------------------
        if len(best) > 1:
            values = (
                best[
                    "selected_value_krw"
                ]
                .astype(float)
            )

            median = float(
                values.median()
            )

            if median == 0:
                spread = float(
                    values.max()
                    - values.min()
                )
            else:
                spread = float(
                    (
                        values.max()
                        - values.min()
                    )
                    / abs(
                        median
                    )
                )

            if spread <= 0.001:
                chosen = best.iloc[0]
                status = (
                    "selected_consensus"
                )
            else:
                rows.append(
                    {
                        **base,
                        "selection_status": (
                            "ambiguous_consolidated_candidates"
                        ),
                        "candidate_count": len(
                            best
                        ),
                        "candidate_tables": (
                            " | ".join(
                                str(x)
                                for x in best[
                                    "table_index"
                                ]
                            )
                        ),
                        "candidate_values": (
                            " | ".join(
                                f"{x:.12g}"
                                for x in values
                            )
                        ),
                    }
                )
                continue
        else:
            chosen = best.iloc[0]
            status = "selected"

        rows.append(
            {
                **base,
                "selection_status": (
                    status
                ),
                "table_index": chosen[
                    "table_index"
                ],
                "row_index": chosen[
                    "row_index"
                ],
                "primary_row_label": chosen[
                    "primary_row_label"
                ],
                "selected_column": chosen[
                    "selected_column"
                ],
                "unit_hint": chosen[
                    "unit_hint"
                ],
                "selected_value_krw": chosen[
                    "selected_value_krw"
                ],
                "strict_score": chosen[
                    "strict_score"
                ],
                "consolidated_evidence_score": chosen[
                    "consolidated_evidence_score"
                ],
                "consolidated_evidence_reason": chosen[
                    "consolidated_evidence_reason"
                ],
                "heading_context": chosen[
                    "heading_context"
                ],
                "row_text": chosen[
                    "row_text"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# validation
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

    for item in wide.to_dict(
        orient="records"
    ):

        assets = item.get(
            "assets"
        )
        liabilities = item.get(
            "liabilities"
        )
        equity = item.get(
            "equity"
        )

        if (
            assets is not None
            and liabilities is not None
            and equity is not None
            and pd.notna(assets)
            and pd.notna(liabilities)
            and pd.notna(equity)
            and float(assets) != 0
        ):
            gap = (
                float(assets)
                - float(liabilities)
                - float(equity)
            )

            rel_gap = (
                abs(gap)
                / abs(
                    float(
                        assets
                    )
                )
            )
        else:
            gap = np.nan
            rel_gap = np.nan

        revenue = item.get(
            "revenue"
        )
        op_income = item.get(
            "operating_income"
        )
        net_income = item.get(
            "net_income"
        )

        # purely diagnostic, not an accounting identity.
        op_margin = (
            float(op_income)
            / float(revenue)
            if (
                revenue is not None
                and op_income is not None
                and pd.notna(revenue)
                and pd.notna(op_income)
                and float(revenue) != 0
            )
            else np.nan
        )

        rows.append(
            {
                "stock_code": item[
                    "stock_code"
                ],
                "period_key": item[
                    "period_key"
                ],
                "rcept_no": item[
                    "rcept_no"
                ],
                "rcept_dt": item[
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
                "revenue": revenue,
                "operating_income": (
                    op_income
                ),
                "net_income": (
                    net_income
                ),
                "diagnostic_operating_margin": (
                    op_margin
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

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"C4 candidate file not found: {CANDIDATE_PATH}"
        )

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={
            "stock_code": str,
            "rcept_no": str,
        },
    )

    if "rcept_dt" in candidates.columns:
        candidates[
            "rcept_dt"
        ] = pd.to_datetime(
            candidates[
                "rcept_dt"
            ],
            errors="coerce",
        )

    selected = select_values(
        candidates
    )

    selected.to_csv(
        SELECTED_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    validation = build_validation(
        selected
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
        "05A5-C5 STRICT CONSOLIDATED DOCUMENT SELECTOR"
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
        "strict_score",
        "consolidated_evidence_reason",
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
        "\n[Important expected behavior]"
        "\n- Woori revenue without consolidated evidence -> unresolved/missing"
        "\n- Woori balance -> selected and balance equation True"
        "\n- Meritz balance -> unresolved/missing"
        "\n- Meritz revenue / operating income -> consolidated H1 cumulative"
        "\n- Meritz net income -> 연결당기순이익 계열을 별도 당기순이익보다 우선"
    )

    print(
        f"\nSelected  : {SELECTED_OUT}"
    )
    print(
        f"Validation: {VALIDATION_OUT}"
    )


if __name__ == "__main__":
    main()
