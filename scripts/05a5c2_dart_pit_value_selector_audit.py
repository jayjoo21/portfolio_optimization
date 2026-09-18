from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05A5-C2. DART PIT Value Selector Audit
#
# 이 단계의 목표
# ------------------------------------------------------------
# A) XBRL
#    샘플 제출본에 대해 "현재기간 + 연결(CFS)" 값 1개를
#    규칙 기반으로 선택해 본다.
#
# B) document fallback
#    아직 숫자 1개를 자동 확정하지 않는다.
#    재무제표 가능성이 높은 table을 statement 단위로 ranking한다.
#
# WHY
# ------------------------------------------------------------
# XBRL은 context/dimension이 구조화되어 있어 자동 선택 가능성이 높음.
# document.xml은 같은 계정명이 사업설명/주석/요약표에도 반복되므로
# table selection을 먼저 검증해야 함.
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/dart_xbrl_fact_candidates.parquet
# data/interim/dart/dart_document_fact_candidates.parquet
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_xbrl_selected_values_audit.csv
#   dart_xbrl_selection_validation.csv
#   dart_document_table_ranking_audit.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

XBRL_PATH = (
    INTERIM_DIR
    / "dart_xbrl_fact_candidates.parquet"
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

DOC_RANKING_PATH = (
    INTERIM_DIR
    / "dart_document_table_ranking_audit.csv"
)


# ------------------------------------------------------------
# XBRL concept roles
#
# total과 parent-attributable을 분리한다.
# 나중 ROE에서 numerator/denominator 정의를 일관되게 하기 위함.
# ------------------------------------------------------------

ROLE_CONFIG: dict[str, dict[str, Any]] = {
    "assets_total": {
        "statement": "balance",
        "family": "assets",
        "concept_priority": [
            "Assets",
        ],
    },
    "liabilities_total": {
        "statement": "balance",
        "family": "liabilities",
        "concept_priority": [
            "Liabilities",
        ],
    },
    "equity_total": {
        "statement": "balance",
        "family": "equity",
        "concept_priority": [
            "Equity",
        ],
    },
    "equity_parent": {
        "statement": "balance",
        "family": "equity",
        "concept_priority": [
            "EquityAttributableToOwnersOfParent",
        ],
    },
    "revenue_cumulative": {
        "statement": "income",
        "family": "revenue",
        "concept_priority": [
            "Revenue",
            "SalesRevenue",
            "RevenueFromContractsWithCustomers",
            "OperatingRevenue",
        ],
    },
    "operating_income_cumulative": {
        "statement": "income",
        "family": "operating_income",
        "concept_priority": [
            "OperatingIncomeLoss",
            "OperatingProfitLoss",
            "OperatingIncome",
            "OperatingProfit",
        ],
    },
    "net_income_total_cumulative": {
        "statement": "income",
        "family": "net_income",
        "concept_priority": [
            "ProfitLoss",
            "NetIncomeLoss",
        ],
    },
    "net_income_parent_cumulative": {
        "statement": "income",
        "family": "net_income",
        "concept_priority": [
            "ProfitLossAttributableToOwnersOfParent",
        ],
    },
}


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def extract_period_end(
    period_key: str,
) -> pd.Timestamp | None:
    """
    period_key 예:
      반기보고서_(2022.06)
      사업보고서_(2023.12)

    월말을 해당 보고기간의 period_end로 사용.
    """
    match = re.search(
        r"\((\d{4})\.(\d{2})\)",
        str(period_key),
    )

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))

    return (
        pd.Timestamp(
            year=year,
            month=month,
            day=1,
        )
        + pd.offsets.MonthEnd(0)
    )


def is_consolidated_dimension(
    text: Any,
) -> bool:
    value = str(text or "")

    has_axis = (
        "ConsolidatedAndSeparateFinancialStatementsAxis"
        in value
    )

    has_consolidated = (
        "ConsolidatedMember"
        in value
    )

    has_separate = (
        "SeparateMember"
        in value
    )

    return (
        has_axis
        and has_consolidated
        and not has_separate
    )


def apply_xbrl_scale(
    numeric_value: Any,
    scale: Any,
) -> float | None:
    if pd.isna(numeric_value):
        return None

    value = float(numeric_value)

    if pd.isna(scale) or str(scale).strip() == "":
        return value

    try:
        scale_int = int(float(scale))
        return value * (10 ** scale_int)
    except Exception:
        return value


def concept_priority_score(
    concept: str,
    priorities: list[str],
) -> int:
    """
    exact 우선.
    RevenueFromContractsWithCustomers... 계열처럼 suffix가 있을 수 있어
    startswith도 fallback으로 허용.
    """
    concept = str(concept)

    for idx, target in enumerate(priorities):
        if concept == target:
            return 100 - idx * 10

    for idx, target in enumerate(priorities):
        if concept.startswith(target):
            return 70 - idx * 10

    return 0


def select_xbrl_role(
    group: pd.DataFrame,
    role: str,
    config: dict[str, Any],
) -> dict[str, Any]:

    period_key = str(
        group["period_key"].iloc[0]
    )

    period_end = extract_period_end(
        period_key
    )

    base = {
        "stock_code": group["stock_code"].iloc[0],
        "period_key": period_key,
        "rcept_no": group["rcept_no"].iloc[0],
        "rcept_dt": group["rcept_dt"].iloc[0],
        "is_correction": group["is_correction"].iloc[0],
        "role": role,
        "period_end_expected": period_end,
    }

    if period_end is None:
        return {
            **base,
            "selection_status": "period_end_parse_failed",
        }

    candidates = group.loc[
        group[
            "account_family"
        ].eq(
            config["family"]
        )
    ].copy()

    if candidates.empty:
        return {
            **base,
            "selection_status": "no_family_candidate",
        }

    # 연결(CFS)만
    candidates = candidates.loc[
        candidates[
            "dimension_text"
        ].map(
            is_consolidated_dimension
        )
    ].copy()

    if candidates.empty:
        return {
            **base,
            "selection_status": "no_consolidated_candidate",
        }

    # 숫자 존재
    candidates = candidates.loc[
        candidates[
            "numeric_value"
        ].notna()
    ].copy()

    if candidates.empty:
        return {
            **base,
            "selection_status": "no_numeric_candidate",
        }

    # concept priority
    candidates[
        "concept_score"
    ] = candidates[
        "concept_local_name"
    ].map(
        lambda x: concept_priority_score(
            x,
            config[
                "concept_priority"
            ],
        )
    )

    candidates = candidates.loc[
        candidates[
            "concept_score"
        ].gt(0)
    ].copy()

    if candidates.empty:
        return {
            **base,
            "selection_status": "no_priority_concept",
        }

    # period matching
    if config["statement"] == "balance":
        instant = pd.to_datetime(
            candidates["instant"],
            errors="coerce",
        )

        candidates[
            "period_match"
        ] = instant.eq(
            period_end
        )

    else:
        start = pd.to_datetime(
            candidates["start_date"],
            errors="coerce",
        )

        end = pd.to_datetime(
            candidates["end_date"],
            errors="coerce",
        )

        candidates[
            "period_match"
        ] = end.eq(
            period_end
        )

        # current period cumulative:
        # 같은 period_end로 끝나는 duration 중 가장 긴 duration을 사용.
        # H1의 181일 vs Q2 standalone 91일 문제를 해결.
        matched = candidates.loc[
            candidates[
                "period_match"
            ]
            & start.notna()
        ]

        if not matched.empty:
            max_duration = matched[
                "duration_days"
            ].max()

            # 비정상적으로 긴 비교기간이 들어오는 것을 막기 위한
            # 넓은 안전 범위.
            if (
                pd.notna(max_duration)
                and 1 <= max_duration <= 400
            ):
                candidates[
                    "cumulative_match"
                ] = (
                    candidates[
                        "period_match"
                    ]
                    & candidates[
                        "duration_days"
                    ].eq(
                        max_duration
                    )
                )
            else:
                candidates[
                    "cumulative_match"
                ] = False
        else:
            candidates[
                "cumulative_match"
            ] = False

    if config["statement"] == "balance":
        candidates = candidates.loc[
            candidates[
                "period_match"
            ]
        ].copy()
    else:
        candidates = candidates.loc[
            candidates[
                "cumulative_match"
            ]
        ].copy()

    if candidates.empty:
        return {
            **base,
            "selection_status": "no_current_period_match",
        }

    # total line item이면 추가 equity-component 등 extra dimension이
    # 없는 쪽을 선호. 최소 dimension_count 선택.
    min_dim = candidates[
        "dimension_count"
    ].min()

    candidates = candidates.loc[
        candidates[
            "dimension_count"
        ].eq(
            min_dim
        )
    ].copy()

    # concept priority 최대
    max_concept_score = candidates[
        "concept_score"
    ].max()

    candidates = candidates.loc[
        candidates[
            "concept_score"
        ].eq(
            max_concept_score
        )
    ].copy()

    # 중복 row 제거
    dedupe_cols = [
        "concept_local_name",
        "context_ref",
        "unit_ref",
        "numeric_value",
    ]

    candidates = candidates.drop_duplicates(
        subset=[
            col
            for col in dedupe_cols
            if col in candidates.columns
        ]
    )

    # 완전히 한 개로 좁혀지지 않으면 억지 선택하지 않음.
    if len(candidates) != 1:
        return {
            **base,
            "selection_status": "ambiguous",
            "candidate_count_final": len(
                candidates
            ),
            "candidate_details": " || ".join(
                (
                    f"{row.concept_local_name}"
                    f"|ctx={row.context_ref}"
                    f"|unit={row.unit_ref}"
                    f"|value={row.numeric_value}"
                    f"|dims={row.dimension_text}"
                )
                for row in candidates.itertuples()
            ),
        }

    row = candidates.iloc[0]

    selected_value = apply_xbrl_scale(
        row["numeric_value"],
        row.get("scale"),
    )

    return {
        **base,
        "selection_status": "selected",
        "concept_local_name": row[
            "concept_local_name"
        ],
        "concept_namespace": row[
            "concept_namespace"
        ],
        "context_ref": row[
            "context_ref"
        ],
        "unit_ref": row[
            "unit_ref"
        ],
        "decimals": row.get(
            "decimals"
        ),
        "scale": row.get(
            "scale"
        ),
        "raw_value": row[
            "raw_value"
        ],
        "numeric_value": row[
            "numeric_value"
        ],
        "selected_value": selected_value,
        "start_date": row[
            "start_date"
        ],
        "end_date": row[
            "end_date"
        ],
        "instant": row[
            "instant"
        ],
        "duration_days": row[
            "duration_days"
        ],
        "dimension_count": row[
            "dimension_count"
        ],
        "dimension_text": row[
            "dimension_text"
        ],
        "candidate_count_final": 1,
    }


# ------------------------------------------------------------
# XBRL validation
# ------------------------------------------------------------

def build_xbrl_selection(
    xbrl: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
    ]

    for _, group in xbrl.groupby(
        group_cols,
        sort=False,
    ):
        for role, config in ROLE_CONFIG.items():
            rows.append(
                select_xbrl_role(
                    group=group,
                    role=role,
                    config=config,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_balance_validation(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    ok = selected.loc[
        selected[
            "selection_status"
        ].eq(
            "selected"
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
            columns="role",
            values="selected_value",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []

    for row in wide.itertuples(
        index=False
    ):
        values = row._asdict()

        assets = values.get(
            "assets_total"
        )
        liabilities = values.get(
            "liabilities_total"
        )
        equity = values.get(
            "equity_total"
        )

        if (
            assets is not None
            and liabilities is not None
            and equity is not None
            and pd.notna(assets)
            and pd.notna(liabilities)
            and pd.notna(equity)
            and assets != 0
        ):
            balance_gap = (
                float(assets)
                - float(liabilities)
                - float(equity)
            )

            relative_gap = (
                abs(balance_gap)
                / abs(float(assets))
            )
        else:
            balance_gap = np.nan
            relative_gap = np.nan

        rows.append(
            {
                "stock_code": values[
                    "stock_code"
                ],
                "period_key": values[
                    "period_key"
                ],
                "rcept_no": values[
                    "rcept_no"
                ],
                "rcept_dt": values[
                    "rcept_dt"
                ],
                "assets_total": assets,
                "liabilities_total": liabilities,
                "equity_total": equity,
                "balance_gap": balance_gap,
                "balance_relative_gap": (
                    relative_gap
                ),
                "balance_equation_pass": (
                    relative_gap <= 1e-8
                    if pd.notna(
                        relative_gap
                    )
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# document table ranking
# ------------------------------------------------------------

def safe_json_loads(
    value: Any,
) -> Any:
    if pd.isna(value):
        return None

    try:
        return json.loads(
            str(value)
        )
    except Exception:
        return None


def build_document_ranking(
    doc: pd.DataFrame,
) -> pd.DataFrame:

    if doc.empty:
        return pd.DataFrame()

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
        (
            stock_code,
            period_key,
            rcept_no,
            rcept_dt,
            table_index,
        ) = keys

        families = set(
            group[
                "account_family"
            ].dropna()
        )

        balance_set = {
            "assets",
            "liabilities",
            "equity",
        }

        income_set = {
            "revenue",
            "operating_income",
            "net_income",
        }

        balance_coverage = len(
            families
            & balance_set
        )

        income_coverage = len(
            families
            & income_set
        )

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

        unit_hints = " | ".join(
            group[
                "unit_hint"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        base_score = float(
            group[
                "statement_score"
            ].max()
        )

        # 완성도 bonus
        completeness_bonus = (
            balance_coverage * 2
            + income_coverage * 2
        )

        # 명백한 false-positive 문맥 penalty
        false_positive_tokens = [
            "배당",
            "보수",
            "성과급",
            "영업부문",
            "신탁",
            "대출상품",
            "여신상품",
        ]

        false_hits = [
            token
            for token in false_positive_tokens
            if token in heading
        ]

        false_penalty = (
            5 * len(
                false_hits
            )
        )

        ranking_score = (
            base_score
            + completeness_bonus
            - false_penalty
        )

        # 계정별 대표 row text
        account_rows = {}

        for family in sorted(
            families
        ):
            subset = group.loc[
                group[
                    "account_family"
                ].eq(
                    family
                )
            ]

            account_rows[
                family
            ] = " || ".join(
                subset[
                    "row_text"
                ]
                .astype(str)
                .head(5)
                .tolist()
            )

        first = group.iloc[0]

        rows.append(
            {
                "stock_code": stock_code,
                "period_key": period_key,
                "rcept_no": rcept_no,
                "rcept_dt": rcept_dt,
                "table_index": table_index,
                "ranking_score": ranking_score,
                "base_statement_score": (
                    base_score
                ),
                "balance_coverage": (
                    balance_coverage
                ),
                "income_coverage": (
                    income_coverage
                ),
                "families": " | ".join(
                    sorted(
                        families
                    )
                ),
                "unit_hints": unit_hints,
                "false_positive_hits": (
                    " | ".join(
                        false_hits
                    )
                ),
                "score_reason": score_reason,
                "heading_context": heading,
                "columns_json": first.get(
                    "columns_json"
                ),
                "assets_rows": account_rows.get(
                    "assets"
                ),
                "liabilities_rows": (
                    account_rows.get(
                        "liabilities"
                    )
                ),
                "equity_rows": account_rows.get(
                    "equity"
                ),
                "revenue_rows": account_rows.get(
                    "revenue"
                ),
                "operating_income_rows": (
                    account_rows.get(
                        "operating_income"
                    )
                ),
                "net_income_rows": (
                    account_rows.get(
                        "net_income"
                    )
                ),
            }
        )

    ranking = pd.DataFrame(
        rows
    )

    ranking[
        "rank_within_receipt"
    ] = (
        ranking
        .groupby(
            [
                "stock_code",
                "period_key",
                "rcept_no",
            ]
        )[
            "ranking_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return (
        ranking
        .sort_values(
            [
                "stock_code",
                "period_key",
                "rcept_no",
                "rank_within_receipt",
                "table_index",
            ]
        )
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:

    if not XBRL_PATH.exists():
        raise FileNotFoundError(
            f"XBRL candidate parquet이 없습니다: {XBRL_PATH}"
        )

    if not DOC_PATH.exists():
        raise FileNotFoundError(
            f"document candidate parquet이 없습니다: {DOC_PATH}"
        )

    xbrl = pd.read_parquet(
        XBRL_PATH
    )

    doc = pd.read_parquet(
        DOC_PATH
    )

    for col in [
        "rcept_dt",
        "start_date",
        "end_date",
        "instant",
    ]:
        if col in xbrl.columns:
            xbrl[col] = pd.to_datetime(
                xbrl[col],
                errors="coerce",
            )

    if "rcept_dt" in doc.columns:
        doc["rcept_dt"] = pd.to_datetime(
            doc["rcept_dt"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # XBRL
    # --------------------------------------------------------

    selected = build_xbrl_selection(
        xbrl
    )

    selected.to_csv(
        XBRL_SELECTED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation = (
        build_balance_validation(
            selected
        )
    )

    validation.to_csv(
        XBRL_VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # document
    # --------------------------------------------------------

    ranking = build_document_ranking(
        doc
    )

    ranking.to_csv(
        DOC_RANKING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-C2 XBRL SELECTION AUDIT"
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
                "role",
                "selection_status",
            ]
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Selected values]"
    )

    selected_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "role",
        "selection_status",
        "concept_local_name",
        "selected_value",
        "unit_ref",
        "start_date",
        "end_date",
        "instant",
        "duration_days",
        "dimension_count",
    ]

    selected_cols = [
        col
        for col in selected_cols
        if col in selected.columns
    ]

    print(
        selected[
            selected_cols
        ]
        .sort_values(
            [
                "stock_code",
                "period_key",
                "rcept_no",
                "role",
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
        print("no validation rows")
    else:
        print(
            validation.to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "DOCUMENT TABLE RANKING AUDIT"
    )
    print(
        "=" * 80
    )

    if ranking.empty:
        print("no document candidates")
    else:
        display_cols = [
            "stock_code",
            "period_key",
            "rcept_no",
            "rank_within_receipt",
            "table_index",
            "ranking_score",
            "balance_coverage",
            "income_coverage",
            "families",
            "unit_hints",
            "false_positive_hits",
            "score_reason",
            "heading_context",
            "columns_json",
            "assets_rows",
            "liabilities_rows",
            "equity_rows",
            "revenue_rows",
            "operating_income_rows",
            "net_income_rows",
        ]

        print(
            ranking.loc[
                ranking[
                    "rank_within_receipt"
                ].le(8),
                display_cols,
            ].to_string(
                index=False
            )
        )

    print(
        f"\nXBRL selected : {XBRL_SELECTED_PATH}"
    )
    print(
        f"XBRL validation: {XBRL_VALIDATION_PATH}"
    )
    print(
        f"Document ranking: {DOC_RANKING_PATH}"
    )

    print(
        "\n판단 기준:"
        "\n- XBRL: selected + balance equation pass가 대부분이면"
        " 자동 selector 규칙을 production으로 승격"
        "\n- ambiguous/no_current_period_match는 절대 임의 선택하지 않음"
        "\n- document: top-ranked table의 실제 columns/rows를 보고"
        " current-period column 선택 규칙을 다음 단계에서 확정"
        "\n- document ranking_score는 품질검사용 heuristic이며"
        " 투자모델 feature가 아님"
    )


if __name__ == "__main__":
    main()
