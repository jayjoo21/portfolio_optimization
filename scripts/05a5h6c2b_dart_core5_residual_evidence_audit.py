from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6C2B. CORE5 RESIDUAL EVIDENCE AUDIT
#
# 목적
# ------------------------------------------------------------
# H6C1 core5 residual 248건의 "빠진 1개 metric"에 대해
# 기존 row-level source 3개를 이용해 왜 selector가 놓쳤는지 진단한다.
#
# Source roles
# ------------------------------------------------------------
# 1) full_fallback_selected
#    - 기존 fallback selector가 이미 metric으로 선택한 결과
#
# 2) full_fallback_rows
#    - 가장 넓은 raw account evidence
#
# 3) multi_account_rows
#    - API long-form account evidence + fs_div(CFS/OFS) 보조 확인
#
# IMPORTANT
# ------------------------------------------------------------
# - 진단 단계이며 값을 복구/merge하지 않음
# - broad lexical candidate는 "후보"일 뿐 자동 채택하지 않음
# - revenue component / financial-domain-specific account를 total revenue로
#   자동 인정하지 않음
# - net income attributable-only row를 total net income으로 자동 인정하지 않음
#
# 실행:
# python scripts\05a5h6c2b_dart_core5_residual_evidence_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

FULL_ROWS = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)

MULTI_ROWS = (
    INTERIM
    / "dart_noncorrected_multi_account_rows.parquet"
)

OUT_CANDIDATES = (
    INTERIM
    / "dart_noncorrected_core5_residual_candidate_rows.parquet"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_core5_residual_evidence_summary.csv"
)

OUT_ACCOUNT_FREQ = (
    INTERIM
    / "dart_noncorrected_core5_residual_account_frequency.csv"
)


EXPECTED_CORE5 = 248


# ============================================================
# Helpers
# ============================================================


def normalize_receipt(
    series: pd.Series,
) -> pd.Series:

    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def clean(
    value,
) -> str:

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def compact(
    value,
) -> str:

    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        clean(value).lower(),
    )


def parse_amount(
    value,
):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return float(value)

    text = str(value).strip()

    if not text:
        return np.nan

    text = (
        text.replace(
            ",",
            "",
        )
        .replace(
            " ",
            "",
        )
    )

    # Parentheses negatives.
    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    if negative:
        text = text[
            1:-1
        ]

    # Common dash / no-value forms.
    if text in {
        "-",
        "－",
        "—",
        "–",
        "",
    }:
        return np.nan

    # Keep numeric chars only, but do not attempt unit conversion here.
    text = re.sub(
        r"[^0-9eE+\-.]",
        "",
        text,
    )

    if not text:
        return np.nan

    try:
        number = float(
            text
        )
    except ValueError:
        return np.nan

    return (
        -number
        if negative
        else number
    )


def canonical_missing_metric(
    value,
) -> str:

    text = compact(
        value
    )

    if text in {
        "assets",
        "asset",
        "자산",
        "자산총계",
    }:
        return "assets"

    if "operatingincome" in text or "영업이익" in text:
        return "operating_income_cumulative"

    if "netincome" in text or "순이익" in text:
        return "net_income_total_cumulative"

    if "revenue" in text or "매출" in text or "영업수익" in text:
        return "revenue_cumulative"

    # H6C1 already stores canonical names.
    raw = clean(
        value
    )

    if raw in {
        "assets",
        "operating_income_cumulative",
        "net_income_total_cumulative",
        "revenue_cumulative",
    }:
        return raw

    return raw


def canonical_selected_metric(
    value,
) -> str:

    text = compact(
        value
    )

    if not text:
        return ""

    if (
        "asset" in text
        or "자산" in text
    ):
        return "assets"

    if (
        "operatingincome" in text
        or "영업이익" in text
        or "영업손실" in text
    ):
        return "operating_income_cumulative"

    if (
        "netincome" in text
        or "순이익" in text
        or "순손실" in text
    ):
        return "net_income_total_cumulative"

    if (
        "revenue" in text
        or "sales" in text
        or "매출" in text
        or "영업수익" in text
    ):
        return "revenue_cumulative"

    return clean(
        value
    )


def is_balance_sj(
    sj_div,
) -> bool:

    text = compact(
        sj_div
    )

    if not text:
        return True

    return (
        "bs" == text
        or "재무상태표" in text
        or "대차대조표" in text
    )


def is_income_sj(
    sj_div,
) -> bool:

    text = compact(
        sj_div
    )

    if not text:
        return True

    return (
        text in {
            "is",
            "cis",
        }
        or "손익계산서" in text
        or "포괄손익계산서" in text
    )


# ============================================================
# Account-label semantics
# ============================================================


def classify_account(
    missing_metric: str,
    account_nm,
    account_id,
    sj_div,
) -> tuple[
    bool,
    str,
    str,
]:

    """
    Returns:
      is_candidate,
      semantic_class,
      semantic_reason

    This is DIAGNOSTIC only.
    """

    name = compact(
        account_nm
    )

    aid = compact(
        account_id
    )

    if not name and not aid:
        return (
            False,
            "NO_LABEL",
            "",
        )

    # --------------------------------------------------------
    # ASSETS
    # --------------------------------------------------------

    if missing_metric == "assets":

        if not is_balance_sj(
            sj_div
        ):
            return (
                False,
                "WRONG_STATEMENT_FAMILY",
                "",
            )

        strong = (
            "자산총계" in name
            or name == "총자산"
            or aid.endswith(
                "assets"
            )
            or "ifrsfullassets" in aid
        )

        if strong:
            return (
                True,
                "STRONG_TOTAL",
                "assets_total",
            )

        return (
            False,
            "NOT_ASSETS_TOTAL",
            "",
        )

    # --------------------------------------------------------
    # OPERATING INCOME
    # --------------------------------------------------------

    if (
        missing_metric
        == "operating_income_cumulative"
    ):

        if not is_income_sj(
            sj_div
        ):
            return (
                False,
                "WRONG_STATEMENT_FAMILY",
                "",
            )

        if (
            "영업이익률" in name
            or "영업손익률" in name
        ):
            return (
                False,
                "RATIO_NOT_AMOUNT",
                "",
            )

        strong = (
            "영업이익" in name
            or "영업손실" in name
            or "operatingprofitloss" in aid
        )

        if strong:
            return (
                True,
                "STRONG_TOTAL",
                "operating_income_total",
            )

        return (
            False,
            "NOT_OPERATING_INCOME",
            "",
        )

    # --------------------------------------------------------
    # NET INCOME TOTAL
    # --------------------------------------------------------

    if (
        missing_metric
        == "net_income_total_cumulative"
    ):

        if not is_income_sj(
            sj_div
        ):
            return (
                False,
                "WRONG_STATEMENT_FAMILY",
                "",
            )

        has_profit = (
            "순이익" in name
            or "순손실" in name
            or "profitloss" in aid
        )

        if not has_profit:
            return (
                False,
                "NOT_NET_INCOME",
                "",
            )

        attributable_markers = [
            "지배기업",
            "지배주주",
            "소유주",
            "귀속",
            "비지배",
        ]

        attributable = any(
            marker in name
            for marker in attributable_markers
        )

        comprehensive = (
            "포괄손익" in name
            or "comprehensiveincome" in aid
        )

        if comprehensive:
            return (
                True,
                "COMPREHENSIVE_INCOME_NOT_NET_INCOME",
                "do_not_auto_adopt",
            )

        if attributable:
            return (
                True,
                "ATTRIBUTABLE_OR_COMPONENT",
                "not_total_net_income",
            )

        total_markers = [
            "당기순이익",
            "당기순손실",
            "분기순이익",
            "분기순손실",
            "반기순이익",
            "반기순손실",
            "연결당기순이익",
            "연결분기순이익",
            "연결반기순이익",
        ]

        if (
            any(
                marker in name
                for marker in total_markers
            )
            or aid.endswith(
                "profitloss"
            )
            or "ifrsfullprofitloss" in aid
        ):
            return (
                True,
                "STRONG_TOTAL",
                "total_net_income",
            )

        return (
            True,
            "NET_INCOME_AMBIGUOUS",
            "needs_semantic_review",
        )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    if (
        missing_metric
        == "revenue_cumulative"
    ):

        if not is_income_sj(
            sj_div
        ):
            return (
                False,
                "WRONG_STATEMENT_FAMILY",
                "",
            )

        # Exclude obvious costs / expenses / ratios first.
        exclusion_markers = [
            "매출원가",
            "영업비용",
            "판매비",
            "관리비",
            "수익률",
            "매출총이익",
            "매출총손실",
        ]

        if any(
            marker in name
            for marker in exclusion_markers
        ):
            return (
                False,
                "REVENUE_RELATED_BUT_NOT_TOPLINE",
                "",
            )

        broad_revenue = (
            "매출" in name
            or "영업수익" in name
            or "수익" in name
            or "revenue" in aid
            or "sales" in aid
        )

        if not broad_revenue:
            return (
                False,
                "NOT_REVENUE_CANDIDATE",
                "",
            )

        component_markers = [
            "기타영업수익",
            "기타수익",
            "이자수익",
            "수수료수익",
            "배당수익",
            "배당금수익",
            "금융수익",
            "투자수익",
            "평가이익",
            "처분이익",
        ]

        if any(
            marker in name
            for marker in component_markers
        ):
            return (
                True,
                "REVENUE_COMPONENT",
                "do_not_auto_adopt_as_total",
            )

        domain_markers = [
            "보험수익",
            "보험영업수익",
            "순영업수익",
            "은행업수익",
            "금융업수익",
            "증권업수익",
            "신탁업수익",
        ]

        if any(
            marker in name
            for marker in domain_markers
        ):
            return (
                True,
                "DOMAIN_SPECIFIC_REVENUE",
                "financial_sector_semantics_required",
            )

        strong_exact_like = (
            name in {
                "매출",
                "매출액",
                "영업수익",
                "영업수익합계",
                "수익",
                "수익합계",
                "매출액및영업수익",
            }
            or "영업수익매출액" in name
            or aid.endswith(
                "revenue"
            )
            or "ifrsfullrevenue" in aid
        )

        if strong_exact_like:
            return (
                True,
                "STRONG_TOTAL",
                "explicit_total_revenue_like",
            )

        return (
            True,
            "REVENUE_AMBIGUOUS",
            "needs_domain_review",
        )

    return (
        False,
        "UNKNOWN_METRIC",
        "",
    )


# ============================================================
# Source preparation
# ============================================================


def prepare_source(
    path: Path,
    source_name: str,
    target_receipts: set[str],
) -> pd.DataFrame:

    df = pd.read_parquet(
        path
    )

    if "rcept_no" not in df.columns:
        raise RuntimeError(
            f"{source_name} has no rcept_no."
        )

    df[
        "rcept_no"
    ] = normalize_receipt(
        df[
            "rcept_no"
        ]
    )

    df = df.loc[
        df[
            "rcept_no"
        ].isin(
            target_receipts
        )
    ].copy()

    df[
        "_source"
    ] = source_name

    if "account_nm" not in df.columns:
        df[
            "account_nm"
        ] = ""

    if "account_id" not in df.columns:
        df[
            "account_id"
        ] = ""

    if "sj_div" not in df.columns:
        df[
            "sj_div"
        ] = ""

    if "fs_div" not in df.columns:
        df[
            "fs_div"
        ] = ""

    if "thstrm_amount" not in df.columns:
        df[
            "thstrm_amount"
        ] = np.nan

    df[
        "_amount_numeric"
    ] = df[
        "thstrm_amount"
    ].map(
        parse_amount
    )

    return df


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        CORE5,
        SELECTED,
        FULL_ROWS,
        MULTI_ROWS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    core5 = pd.read_parquet(
        CORE5
    )

    if len(
        core5
    ) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 rows, "
            f"found {len(core5)}."
        )

    core5[
        "rcept_no"
    ] = normalize_receipt(
        core5[
            "rcept_no"
        ]
    )

    core5[
        "missing_metric"
    ] = core5[
        "missing_metric"
    ].map(
        canonical_missing_metric
    )

    target_receipts = set(
        core5[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2B CORE5 RESIDUAL EVIDENCE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCore5 receipts: "
        f"{len(core5):,}"
    )

    selected = prepare_source(
        SELECTED,
        "FULL_FALLBACK_SELECTED",
        target_receipts,
    )

    full_rows = prepare_source(
        FULL_ROWS,
        "FULL_FALLBACK_ROWS",
        target_receipts,
    )

    multi_rows = prepare_source(
        MULTI_ROWS,
        "MULTI_ACCOUNT_ROWS",
        target_receipts,
    )

    print(
        "\n[Source rows within core5 receipts]"
    )

    print(
        f"selected : "
        f"{len(selected):,}"
    )

    print(
        f"full rows: "
        f"{len(full_rows):,}"
    )

    print(
        f"multi    : "
        f"{len(multi_rows):,}"
    )

    # --------------------------------------------------------
    # Existing selected metric diagnostic
    # --------------------------------------------------------

    if "metric" in selected.columns:

        selected[
            "_metric_canonical"
        ] = selected[
            "metric"
        ].map(
            canonical_selected_metric
        )

    else:

        selected[
            "_metric_canonical"
        ] = ""

    selected = selected.merge(
        core5[
            [
                "rcept_no",
                "missing_metric",
            ]
        ],
        on="rcept_no",
        how="left",
        validate="many_to_one",
    )

    selected[
        "_matches_missing_metric"
    ] = (
        selected[
            "_metric_canonical"
        ].eq(
            selected[
                "missing_metric"
            ]
        )
    )

    # --------------------------------------------------------
    # Build broad candidate rows from raw sources.
    # --------------------------------------------------------

    raw_sources = pd.concat(
        [
            full_rows,
            multi_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    raw_sources = raw_sources.merge(
        core5[
            [
                "rcept_no",
                "stock_code",
                "corp_name",
                "period_key",
                "missing_metric",
            ]
            if "corp_name" in core5.columns
            else [
                "rcept_no",
                "stock_code",
                "period_key",
                "missing_metric",
            ]
        ],
        on="rcept_no",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_core5",
        ),
    )

    candidate_flags = [
        classify_account(
            missing_metric,
            account_nm,
            account_id,
            sj_div,
        )
        for (
            missing_metric,
            account_nm,
            account_id,
            sj_div,
        )
        in zip(
            raw_sources[
                "missing_metric"
            ],
            raw_sources[
                "account_nm"
            ],
            raw_sources[
                "account_id"
            ],
            raw_sources[
                "sj_div"
            ],
        )
    ]

    raw_sources[
        "_is_candidate"
    ] = [
        item[
            0
        ]
        for item in candidate_flags
    ]

    raw_sources[
        "_semantic_class"
    ] = [
        item[
            1
        ]
        for item in candidate_flags
    ]

    raw_sources[
        "_semantic_reason"
    ] = [
        item[
            2
        ]
        for item in candidate_flags
    ]

    candidates = raw_sources.loc[
        raw_sources[
            "_is_candidate"
        ]
    ].copy()

    # Preserve only useful columns, while keeping source-specific
    # fields such as fs_div when available.
    candidate_keep_cols = [
        col
        for col in [
            "_source",
            "rcept_no",
            "stock_code",
            "corp_name",
            "period_key",
            "missing_metric",
            "fs_div",
            "sj_div",
            "account_id",
            "account_nm",
            "thstrm_amount",
            "_amount_numeric",
            "_semantic_class",
            "_semantic_reason",
        ]
        if col in candidates.columns
    ]

    candidates = candidates[
        candidate_keep_cols
    ].copy()

    candidates.to_parquet(
        OUT_CANDIDATES,
        index=False,
    )

    # --------------------------------------------------------
    # Receipt-level evidence summary
    # --------------------------------------------------------

    selected_match_groups = (
        selected.loc[
            selected[
                "_matches_missing_metric"
            ]
        ]
        .groupby(
            "rcept_no",
            sort=False,
        )
    )

    candidate_groups = (
        candidates.groupby(
            "rcept_no",
            sort=False,
        )
    )

    receipt_records = []

    for row in core5.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        missing_metric = str(
            row.missing_metric
        )

        if rcept_no in (
            selected_match_groups.groups
        ):

            sgroup = (
                selected_match_groups.get_group(
                    rcept_no
                )
            )

        else:

            sgroup = selected.iloc[
                0:0
            ]

        if rcept_no in (
            candidate_groups.groups
        ):

            cgroup = (
                candidate_groups.get_group(
                    rcept_no
                )
            )

        else:

            cgroup = candidates.iloc[
                0:0
            ]

        strong = cgroup.loc[
            cgroup[
                "_semantic_class"
            ].eq(
                "STRONG_TOTAL"
            )
        ]

        domain = cgroup.loc[
            cgroup[
                "_semantic_class"
            ].eq(
                "DOMAIN_SPECIFIC_REVENUE"
            )
        ]

        component = cgroup.loc[
            cgroup[
                "_semantic_class"
            ].isin(
                [
                    "REVENUE_COMPONENT",
                    "ATTRIBUTABLE_OR_COMPONENT",
                    "COMPREHENSIVE_INCOME_NOT_NET_INCOME",
                ]
            )
        ]

        ambiguous = cgroup.loc[
            cgroup[
                "_semantic_class"
            ].isin(
                [
                    "REVENUE_AMBIGUOUS",
                    "NET_INCOME_AMBIGUOUS",
                ]
            )
        ]

        strong_values = (
            strong[
                "_amount_numeric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        all_values = (
            cgroup[
                "_amount_numeric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        source_names = (
            sorted(
                set(
                    cgroup[
                        "_source"
                    ].astype(str)
                )
            )
            if not cgroup.empty
            else []
        )

        fs_divs = (
            sorted(
                set(
                    value
                    for value in (
                        cgroup[
                            "fs_div"
                        ]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )
                    if value
                )
            )
            if "fs_div" in cgroup.columns
            else []
        )

        if len(
            strong_values
        ) == 1:

            diagnosis = (
                "STRONG_TOTAL_SINGLE_VALUE"
            )

        elif len(
            strong_values
        ) > 1:

            diagnosis = (
                "STRONG_TOTAL_MULTIPLE_VALUES"
            )

        elif not domain.empty:

            diagnosis = (
                "DOMAIN_SPECIFIC_ONLY_OR_WITH_AMBIGUOUS"
            )

        elif not ambiguous.empty:

            diagnosis = (
                "AMBIGUOUS_ONLY"
            )

        elif not component.empty:

            diagnosis = (
                "COMPONENT_ONLY"
            )

        elif not cgroup.empty:

            diagnosis = (
                "CANDIDATE_OTHER"
            )

        else:

            diagnosis = (
                "NO_BROAD_CANDIDATE"
            )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                getattr(
                    row,
                    "stock_code",
                    "",
                ),

                "corp_name":
                getattr(
                    row,
                    "corp_name",
                    "",
                ),

                "period_key":
                getattr(
                    row,
                    "period_key",
                    "",
                ),

                "missing_metric":
                missing_metric,

                "selected_same_metric_rows":
                len(
                    sgroup
                ),

                "selected_same_metric_accounts":
                " || ".join(
                    sgroup[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
                if "account_nm" in sgroup.columns
                else "",

                "raw_candidate_rows":
                len(
                    cgroup
                ),

                "raw_candidate_sources":
                "|".join(
                    source_names
                ),

                "raw_fs_divs":
                "|".join(
                    fs_divs
                ),

                "strong_total_rows":
                len(
                    strong
                ),

                "strong_total_unique_values":
                len(
                    strong_values
                ),

                "strong_total_accounts":
                " || ".join(
                    strong[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "domain_specific_rows":
                len(
                    domain
                ),

                "domain_specific_accounts":
                " || ".join(
                    domain[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "component_rows":
                len(
                    component
                ),

                "component_accounts":
                " || ".join(
                    component[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "ambiguous_rows":
                len(
                    ambiguous
                ),

                "ambiguous_accounts":
                " || ".join(
                    ambiguous[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "all_candidate_unique_values":
                len(
                    all_values
                ),

                "evidence_diagnosis":
                diagnosis,
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
    # Account frequency summary
    # --------------------------------------------------------

    if candidates.empty:

        account_freq = pd.DataFrame(
            columns=[
                "missing_metric",
                "_source",
                "_semantic_class",
                "account_nm",
                "receipt_count",
                "row_count",
            ]
        )

    else:

        account_freq = (
            candidates.groupby(
                [
                    "missing_metric",
                    "_source",
                    "_semantic_class",
                    "account_nm",
                ],
                dropna=False,
            )
            .agg(
                receipt_count=(
                    "rcept_no",
                    "nunique",
                ),

                row_count=(
                    "rcept_no",
                    "size",
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "missing_metric",
                    "receipt_count",
                    "row_count",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
        )

    account_freq.to_csv(
        OUT_ACCOUNT_FREQ,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Existing selected rows matching the missing metric]"
    )

    selected_counts = (
        receipt_summary[
            "selected_same_metric_rows"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        selected_counts.to_string()
    )

    print(
        "\n[Receipt evidence diagnosis]"
    )

    print(
        pd.crosstab(
            receipt_summary[
                "missing_metric"
            ],
            receipt_summary[
                "evidence_diagnosis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Strong-total unique-value count by missing metric]"
    )

    print(
        pd.crosstab(
            receipt_summary[
                "missing_metric"
            ],
            receipt_summary[
                "strong_total_unique_values"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Candidate semantic class]"
    )

    if candidates.empty:
        print(
            "None"
        )

    else:
        print(
            pd.crosstab(
                candidates[
                    "missing_metric"
                ],
                candidates[
                    "_semantic_class"
                ],
                dropna=False,
            )
            .to_string()
        )

    print(
        "\n[Top REVENUE account names]"
    )

    revenue_freq = account_freq.loc[
        account_freq[
            "missing_metric"
        ].eq(
            "revenue_cumulative"
        )
    ]

    if revenue_freq.empty:
        print(
            "None"
        )

    else:
        with pd.option_context(
            "display.max_colwidth",
            120,
            "display.width",
            300,
            "display.max_rows",
            60,
        ):
            print(
                revenue_freq[
                    [
                        "_source",
                        "_semantic_class",
                        "account_nm",
                        "receipt_count",
                        "row_count",
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
        "\n[Top NET INCOME account names]"
    )

    net_freq = account_freq.loc[
        account_freq[
            "missing_metric"
        ].eq(
            "net_income_total_cumulative"
        )
    ]

    if net_freq.empty:
        print(
            "None"
        )

    else:
        with pd.option_context(
            "display.max_colwidth",
            120,
            "display.width",
            300,
            "display.max_rows",
            60,
        ):
            print(
                net_freq[
                    [
                        "_source",
                        "_semantic_class",
                        "account_nm",
                        "receipt_count",
                        "row_count",
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
        "\n[Revenue receipt diagnosis]"
    )

    print(
        receipt_summary.loc[
            receipt_summary[
                "missing_metric"
            ].eq(
                "revenue_cumulative"
            ),
            "evidence_diagnosis",
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Net-income receipt diagnosis]"
    )

    print(
        receipt_summary.loc[
            receipt_summary[
                "missing_metric"
            ].eq(
                "net_income_total_cumulative"
            ),
            "evidence_diagnosis",
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Small residuals: operating income / assets]"
    )

    small = receipt_summary.loc[
        receipt_summary[
            "missing_metric"
        ].isin(
            [
                "operating_income_cumulative",
                "assets",
            ]
        )
    ]

    if small.empty:
        print(
            "None"
        )

    else:
        with pd.option_context(
            "display.max_colwidth",
            140,
            "display.width",
            320,
        ):
            print(
                small[
                    [
                        "stock_code",
                        "corp_name",
                        "period_key",
                        "rcept_no",
                        "missing_metric",
                        "selected_same_metric_rows",
                        "strong_total_rows",
                        "strong_total_unique_values",
                        "strong_total_accounts",
                        "evidence_diagnosis",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    print(
        "\nOutputs:"
    )

    print(
        f"- Candidate rows    : "
        f"{OUT_CANDIDATES}"
    )

    print(
        f"- Receipt summary   : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Account frequency : "
        f"{OUT_ACCOUNT_FREQ}"
    )

    print(
        "\n해석 원칙:"
        "\n- STRONG_TOTAL_SINGLE_VALUE도 아직 자동채택 아님; basis/period/YTD 검증이 다음 단계"
        "\n- revenue DOMAIN_SPECIFIC/COMPONENT는 금융업 semantics 트랙으로 분리"
        "\n- net-income ATTRIBUTABLE_OR_COMPONENT는 total net income 대체 금지"
        "\n- full_fallback_selected에 missing metric row가 이미 있다면 wide 누락 이유를 별도 추적"
        "\n- operating-income/assets 4건은 exact receipt audit 후보"
    )


if __name__ == "__main__":
    main()
