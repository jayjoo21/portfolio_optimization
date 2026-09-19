from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6C2C. CORE5 SEMANTIC CLASSIFIER V2 + SELECTED PROVENANCE
#
# 목적
# ------------------------------------------------------------
# H6C2B에서 확인된 과잉매칭을 제거한다.
#
# H6C2B v1 문제:
#   - account_id.endswith("assets") -> CurrentAssets 등까지 strong 오인
#   - account_id.endswith("profitloss") -> pretax / operating 계정까지 strong 오인
#
# V2 원칙:
#   1) account_nm exact semantic label 우선
#   2) account_id는 EXACT allowlist만 strong evidence로 사용
#   3) attributable / pretax / continuing / discontinued / component는 분리
#   4) revenue component는 절대 total revenue로 자동 승격하지 않음
#
# 추가 목적:
#   full_fallback_selected에서 "missing metric"이 이미 선택된 50 receipt를
#   따로 뽑아 selected-but-wide-missing provenance를 확인한다.
#
# 이 단계는 AUDIT ONLY.
# 값 복구/merge 없음.
#
# 실행:
# python scripts\05a5h6c2c_dart_core5_semantic_classifier_v2.py
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

COVERAGE = (
    INTERIM
    / "dart_noncorrected_major_account_coverage.parquet"
)

OUT_CANDIDATES = (
    INTERIM
    / "dart_noncorrected_core5_semantic_v2_candidate_rows.parquet"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_core5_semantic_v2_receipt_summary.csv"
)

OUT_SELECTED_MISSING = (
    INTERIM
    / "dart_noncorrected_core5_selected_but_wide_missing_audit.csv"
)

OUT_ACCOUNT_FREQ = (
    INTERIM
    / "dart_noncorrected_core5_semantic_v2_account_frequency.csv"
)


EXPECTED_CORE5 = 248


# ============================================================
# Text / amount helpers
# ============================================================


def normalize_receipt(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def compact(value) -> str:
    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        clean(value).lower(),
    )


def normalize_account_label(value) -> str:
    """
    표기용 번호/로마숫자 prefix를 제거한 뒤 compact.
    예:
      III. 영업이익 -> 영업이익
      Ⅳ. 당기순이익 -> 당기순이익
    """
    text = clean(value)

    text = re.sub(
        r"^\s*[\(\[]?\s*"
        r"(?:[ivxlcdmⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+|\d+)"
        r"\s*[\)\]\.\-:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return compact(text)


def parse_amount(value):
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    text = str(value).strip()

    if not text or text in {
        "-",
        "－",
        "—",
        "–",
    }:
        return np.nan

    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    if negative:
        text = text[1:-1]

    text = text.replace(",", "").replace(" ", "")

    text = re.sub(
        r"[^0-9eE+\-.]",
        "",
        text,
    )

    if not text:
        return np.nan

    try:
        number = float(text)
    except ValueError:
        return np.nan

    return -number if negative else number


def canonical_metric(value) -> str:
    raw = clean(value)

    if raw in {
        "assets",
        "operating_income_cumulative",
        "net_income_total_cumulative",
        "revenue_cumulative",
    }:
        return raw

    text = compact(value)

    if "asset" in text or "자산" in text:
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

    return raw


def statement_family(sj_div) -> str:
    text = compact(sj_div)

    if not text:
        return "UNKNOWN"

    if (
        text == "bs"
        or "재무상태표" in text
        or "대차대조표" in text
    ):
        return "BALANCE"

    if (
        text in {"is", "cis"}
        or "손익계산서" in text
        or "포괄손익계산서" in text
    ):
        return "INCOME"

    return "OTHER"


# ============================================================
# Exact account-id allowlists
# ============================================================

ASSET_TOTAL_IDS = {
    "ifrsfullassets",
}

OPERATING_INCOME_IDS = {
    "dartoperatingincomeloss",
    "ifrsfullprofitlossfromoperatingactivities",
}

NET_INCOME_TOTAL_IDS = {
    "ifrsfullprofitloss",
}

REVENUE_TOTAL_IDS = {
    "ifrsfullrevenue",
    "ifrsfullrevenuefromcontractswithcustomers",
}


# ============================================================
# Exact label sets
# ============================================================

ASSET_TOTAL_LABELS = {
    "자산총계",
    "총자산",
}

OPERATING_INCOME_TOTAL_LABELS = {
    "영업이익",
    "영업손실",
    "영업이익손실",
    "연결영업이익",
    "연결영업손실",
    "연결영업이익손실",
}

NET_INCOME_TOTAL_LABELS = {
    "당기순이익",
    "당기순손실",
    "당기순이익손실",
    "분기순이익",
    "분기순손실",
    "분기순이익손실",
    "반기순이익",
    "반기순손실",
    "반기순이익손실",
    "연결당기순이익",
    "연결당기순손실",
    "연결당기순이익손실",
    "연결분기순이익",
    "연결분기순손실",
    "연결분기순이익손실",
    "연결반기순이익",
    "연결반기순손실",
    "연결반기순이익손실",
}

REVENUE_TOTAL_LABELS = {
    "매출",
    "매출액",
    "영업수익",
    "영업수익합계",
    "수익",
    "수익합계",
    "매출액및영업수익",
    "영업수익매출액",
}


# ============================================================
# V2 semantic classifier
# ============================================================


def classify_v2(
    missing_metric: str,
    account_nm,
    account_id,
    sj_div,
) -> tuple[bool, str, str]:

    label = normalize_account_label(account_nm)
    aid = compact(account_id)
    family = statement_family(sj_div)

    if not label and not aid:
        return False, "NO_LABEL", ""

    # --------------------------------------------------------
    # ASSETS
    # --------------------------------------------------------
    if missing_metric == "assets":

        if family not in {
            "BALANCE",
            "UNKNOWN",
        }:
            return False, "WRONG_STATEMENT_FAMILY", ""

        if (
            label in ASSET_TOTAL_LABELS
            or aid in ASSET_TOTAL_IDS
        ):
            return True, "STRONG_TOTAL", "exact_assets_total"

        # Broad asset-related rows are recorded diagnostically
        # but never treated as totals.
        if (
            "자산" in label
            or "assets" in aid
        ):
            return True, "ASSET_COMPONENT_OR_AMBIGUOUS", "not_total_assets"

        return False, "NOT_ASSETS", ""

    # --------------------------------------------------------
    # OPERATING INCOME
    # --------------------------------------------------------
    if missing_metric == "operating_income_cumulative":

        if family not in {
            "INCOME",
            "UNKNOWN",
        }:
            return False, "WRONG_STATEMENT_FAMILY", ""

        if any(
            token in label
            for token in [
                "영업이익률",
                "영업손익률",
            ]
        ):
            return False, "RATIO_NOT_AMOUNT", ""

        if (
            label in OPERATING_INCOME_TOTAL_LABELS
            or aid in OPERATING_INCOME_IDS
        ):
            return True, "STRONG_TOTAL", "exact_operating_income"

        if (
            "영업이익" in label
            or "영업손실" in label
            or "operating" in aid
        ):
            return True, "OPERATING_INCOME_AMBIGUOUS", "needs_review"

        return False, "NOT_OPERATING_INCOME", ""

    # --------------------------------------------------------
    # NET INCOME TOTAL
    # --------------------------------------------------------
    if missing_metric == "net_income_total_cumulative":

        if family not in {
            "INCOME",
            "UNKNOWN",
        }:
            return False, "WRONG_STATEMENT_FAMILY", ""

        # Explicit pre-tax must be excluded BEFORE generic 순이익 matching.
        pretax_markers = [
            "법인세비용차감전",
            "법인세차감전",
            "세전",
        ]

        if any(
            marker in label
            for marker in pretax_markers
        ):
            return True, "PRETAX_NOT_NET_INCOME", "exclude"

        # Continuing / discontinued components are not total net income.
        component_markers = [
            "계속영업",
            "중단영업",
        ]

        if any(
            marker in label
            for marker in component_markers
        ):
            return True, "NET_INCOME_COMPONENT", "continuing_or_discontinued"

        attributable_markers = [
            "지배기업",
            "지배주주",
            "소유주",
            "귀속",
            "비지배",
        ]

        if any(
            marker in label
            for marker in attributable_markers
        ):
            return True, "ATTRIBUTABLE_OR_COMPONENT", "not_total_net_income"

        if "포괄손익" in label:
            return True, "COMPREHENSIVE_INCOME_NOT_NET_INCOME", "exclude"

        if (
            label in NET_INCOME_TOTAL_LABELS
            or aid in NET_INCOME_TOTAL_IDS
        ):
            return True, "STRONG_TOTAL", "exact_total_net_income"

        if (
            "순이익" in label
            or "순손실" in label
        ):
            return True, "NET_INCOME_AMBIGUOUS", "needs_review"

        return False, "NOT_NET_INCOME", ""

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------
    if missing_metric == "revenue_cumulative":

        if family not in {
            "INCOME",
            "UNKNOWN",
        }:
            return False, "WRONG_STATEMENT_FAMILY", ""

        exclusion_markers = [
            "매출원가",
            "영업비용",
            "판매비",
            "관리비",
            "수익률",
            "매출총이익",
            "매출총손실",
            "법인세",
        ]

        if any(
            marker in label
            for marker in exclusion_markers
        ):
            return False, "REVENUE_RELATED_BUT_NOT_TOPLINE", ""

        # Explicit components; never total.
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
            "파생상품관련수익",
            "리스수익",
        ]

        if any(
            marker in label
            for marker in component_markers
        ):
            return True, "REVENUE_COMPONENT", "do_not_auto_adopt"

        domain_markers = [
            "보험수익",
            "보험영업수익",
            "순영업수익",
            "은행업수익",
            "금융업수익",
            "증권업수익",
            "신탁업수익",
            "순이자손익",
            "순이자이익",
            "순보험손익",
        ]

        if any(
            marker in label
            for marker in domain_markers
        ):
            return True, "DOMAIN_SPECIFIC_REVENUE", "financial_semantics_required"

        if (
            label in REVENUE_TOTAL_LABELS
            or aid in REVENUE_TOTAL_IDS
        ):
            return True, "STRONG_TOTAL", "exact_total_revenue_like"

        if (
            "매출" in label
            or "영업수익" in label
            or "수익" in label
            or "revenue" in aid
        ):
            return True, "REVENUE_AMBIGUOUS", "needs_domain_review"

        return False, "NOT_REVENUE", ""

    return False, "UNKNOWN_METRIC", ""


# ============================================================
# Source prep
# ============================================================


def prepare_source(
    path: Path,
    source_name: str,
    target_receipts: set[str],
) -> pd.DataFrame:

    df = pd.read_parquet(path)

    if "rcept_no" not in df.columns:
        raise RuntimeError(
            f"{source_name} has no rcept_no."
        )

    df["rcept_no"] = normalize_receipt(
        df["rcept_no"]
    )

    df = df.loc[
        df["rcept_no"].isin(
            target_receipts
        )
    ].copy()

    df["_source"] = source_name

    for col, default in [
        ("account_nm", ""),
        ("account_id", ""),
        ("sj_div", ""),
        ("fs_div", ""),
        ("thstrm_amount", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    df["_amount_numeric"] = df[
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
            raise FileNotFoundError(path)

    core5 = pd.read_parquet(CORE5)

    if len(core5) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5}, found {len(core5)}."
        )

    core5["rcept_no"] = normalize_receipt(
        core5["rcept_no"]
    )

    core5["missing_metric"] = core5[
        "missing_metric"
    ].map(
        canonical_metric
    )

    target_receipts = set(
        core5["rcept_no"]
    )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "05A5-H6C2C CORE5 SEMANTIC CLASSIFIER V2 + SELECTED PROVENANCE"
    )
    print(
        "=" * 120
    )

    # --------------------------------------------------------
    # Selected provenance
    # --------------------------------------------------------

    selected = prepare_source(
        SELECTED,
        "FULL_FALLBACK_SELECTED",
        target_receipts,
    )

    if "metric" not in selected.columns:
        selected["metric"] = ""

    selected["_metric_canonical"] = selected[
        "metric"
    ].map(
        canonical_metric
    )

    selected = selected.merge(
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
        suffixes=("", "_core5"),
    )

    selected_same = selected.loc[
        selected[
            "_metric_canonical"
        ].eq(
            selected[
                "missing_metric"
            ]
        )
    ].copy()

    # Optional coverage/provenance fields if available.
    if COVERAGE.exists():

        coverage = pd.read_parquet(COVERAGE)

        if "rcept_no" in coverage.columns:

            coverage["rcept_no"] = normalize_receipt(
                coverage["rcept_no"]
            )

            keep_cols = [
                "rcept_no",
            ] + [
                col
                for col in coverage.columns
                if col != "rcept_no"
                and any(
                    token in col.lower()
                    for token in [
                        "status",
                        "coverage",
                        "missing",
                        "complete",
                        "core",
                    ]
                )
            ]

            # Avoid duplicate column names.
            keep_cols = list(
                dict.fromkeys(
                    keep_cols
                )
            )

            coverage_one = (
                coverage[
                    keep_cols
                ]
                .drop_duplicates(
                    subset=[
                        "rcept_no",
                    ]
                )
            )

            selected_same = selected_same.merge(
                coverage_one,
                on="rcept_no",
                how="left",
                validate="many_to_one",
                suffixes=("", "_coverage"),
            )

    selected_same.to_csv(
        OUT_SELECTED_MISSING,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Semantic V2 candidate pool
    # --------------------------------------------------------

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

    raw = pd.concat(
        [
            full_rows,
            multi_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    core5_join_cols = [
        "rcept_no",
        "stock_code",
        "period_key",
        "missing_metric",
    ]

    if "corp_name" in core5.columns:
        core5_join_cols.insert(
            2,
            "corp_name",
        )

    raw = raw.merge(
        core5[
            core5_join_cols
        ],
        on="rcept_no",
        how="left",
        validate="many_to_one",
        suffixes=("", "_core5"),
    )

    classes = [
        classify_v2(
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
            raw["missing_metric"],
            raw["account_nm"],
            raw["account_id"],
            raw["sj_div"],
        )
    ]

    raw["_is_candidate_v2"] = [
        x[0]
        for x in classes
    ]
    raw["_semantic_class_v2"] = [
        x[1]
        for x in classes
    ]
    raw["_semantic_reason_v2"] = [
        x[2]
        for x in classes
    ]

    candidates = raw.loc[
        raw["_is_candidate_v2"]
    ].copy()

    save_cols = [
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
            "_semantic_class_v2",
            "_semantic_reason_v2",
        ]
        if col in candidates.columns
    ]

    candidates = candidates[
        save_cols
    ].copy()

    candidates.to_parquet(
        OUT_CANDIDATES,
        index=False,
    )

    # --------------------------------------------------------
    # Receipt summary
    # --------------------------------------------------------

    grouped = {
        rcept_no:
        group
        for rcept_no, group
        in candidates.groupby(
            "rcept_no",
            sort=False,
        )
    }

    selected_count_map = (
        selected_same.groupby(
            "rcept_no"
        )
        .size()
        .to_dict()
    )

    receipt_records = []

    for row in core5.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        group = grouped.get(
            rcept_no,
            candidates.iloc[
                0:0
            ],
        )

        strong = group.loc[
            group[
                "_semantic_class_v2"
            ].eq(
                "STRONG_TOTAL"
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

        classes_present = (
            sorted(
                set(
                    group[
                        "_semantic_class_v2"
                    ].astype(str)
                )
            )
            if not group.empty
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

        elif any(
            cls == "DOMAIN_SPECIFIC_REVENUE"
            for cls in classes_present
        ):

            diagnosis = (
                "DOMAIN_SPECIFIC_REVENUE_ONLY_OR_MIXED"
            )

        elif any(
            cls in {
                "ATTRIBUTABLE_OR_COMPONENT",
                "NET_INCOME_COMPONENT",
                "PRETAX_NOT_NET_INCOME",
                "COMPREHENSIVE_INCOME_NOT_NET_INCOME",
                "REVENUE_COMPONENT",
                "ASSET_COMPONENT_OR_AMBIGUOUS",
            }
            for cls in classes_present
        ):

            diagnosis = (
                "ONLY_COMPONENT_OR_EXCLUDED_SEMANTICS"
            )

        elif group.empty:

            diagnosis = (
                "NO_V2_CANDIDATE"
            )

        else:

            diagnosis = (
                "AMBIGUOUS_ONLY"
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
                getattr(
                    row,
                    "missing_metric",
                    "",
                ),

                "selected_same_metric_rows":
                int(
                    selected_count_map.get(
                        rcept_no,
                        0,
                    )
                ),

                "candidate_rows_v2":
                len(
                    group
                ),

                "semantic_classes_v2":
                "|".join(
                    classes_present
                ),

                "strong_total_rows_v2":
                len(
                    strong
                ),

                "strong_total_unique_values_v2":
                len(
                    strong_values
                ),

                "strong_total_accounts_v2":
                " || ".join(
                    strong[
                        "account_nm"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "strong_total_account_ids_v2":
                " || ".join(
                    strong[
                        "account_id"
                    ]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),

                "v2_diagnosis":
                diagnosis,
            }
        )

    summary = pd.DataFrame(
        receipt_records
    )

    summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Account frequency
    # --------------------------------------------------------

    if candidates.empty:

        freq = pd.DataFrame()

    else:

        freq = (
            candidates.groupby(
                [
                    "missing_metric",
                    "_source",
                    "_semantic_class_v2",
                    "account_nm",
                    "account_id",
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

    freq.to_csv(
        OUT_ACCOUNT_FREQ,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Selected-but-wide-missing receipts]"
    )

    print(
        selected_same[
            "rcept_no"
        ].nunique()
    )

    print(
        "\n[Selected-but-wide-missing by metric]"
    )

    if selected_same.empty:
        print("None")
    else:
        print(
            selected_same.groupby(
                "missing_metric"
            )[
                "rcept_no"
            ]
            .nunique()
            .to_string()
        )

    print(
        "\n[V2 receipt diagnosis]"
    )

    print(
        pd.crosstab(
            summary[
                "missing_metric"
            ],
            summary[
                "v2_diagnosis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[V2 strong-total unique-value count]"
    )

    print(
        pd.crosstab(
            summary[
                "missing_metric"
            ],
            summary[
                "strong_total_unique_values_v2"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[V2 candidate semantic classes]"
    )

    if candidates.empty:
        print("None")
    else:
        print(
            pd.crosstab(
                candidates[
                    "missing_metric"
                ],
                candidates[
                    "_semantic_class_v2"
                ],
                dropna=False,
            )
            .to_string()
        )

    print(
        "\n[ASSETS residual exact candidates]"
    )

    assets = candidates.loc[
        candidates[
            "missing_metric"
        ].eq(
            "assets"
        )
    ]

    if assets.empty:
        print("None")
    else:
        print(
            assets[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "fs_div",
                    "sj_div",
                    "account_nm",
                    "account_id",
                    "_semantic_class_v2",
                    "_amount_numeric",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[OPERATING INCOME residual exact candidates]"
    )

    op = candidates.loc[
        candidates[
            "missing_metric"
        ].eq(
            "operating_income_cumulative"
        )
    ]

    if op.empty:
        print("None")
    else:
        print(
            op[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "fs_div",
                    "sj_div",
                    "account_nm",
                    "account_id",
                    "_semantic_class_v2",
                    "_amount_numeric",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[Selected-but-wide-missing sample]"
    )

    show_cols = [
        col
        for col in [
            "stock_code",
            "corp_name",
            "period_key",
            "rcept_no",
            "missing_metric",
            "metric",
            "account_nm",
            "account_id",
            "sj_div",
            "thstrm_amount",
        ]
        if col in selected_same.columns
    ]

    if selected_same.empty:
        print("None")
    else:
        with pd.option_context(
            "display.max_colwidth",
            140,
            "display.width",
            320,
            "display.max_rows",
            60,
        ):
            print(
                selected_same[
                    show_cols
                ]
                .head(
                    50
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[NET INCOME V2 strong-total top account names]"
    )

    if freq.empty:
        print("None")
    else:
        net = freq.loc[
            freq[
                "missing_metric"
            ].eq(
                "net_income_total_cumulative"
            )
            & freq[
                "_semantic_class_v2"
            ].eq(
                "STRONG_TOTAL"
            )
        ]

        if net.empty:
            print("None")
        else:
            print(
                net[
                    [
                        "_source",
                        "account_nm",
                        "account_id",
                        "receipt_count",
                        "row_count",
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
        "\n[REVENUE V2 semantic class by receipt]"
    )

    rev_summary = summary.loc[
        summary[
            "missing_metric"
        ].eq(
            "revenue_cumulative"
        )
    ]

    print(
        rev_summary[
            "v2_diagnosis"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- V2 candidates          : {OUT_CANDIDATES}"
    )
    print(
        f"- V2 receipt summary     : {OUT_RECEIPT}"
    )
    print(
        f"- Selected-missing audit : {OUT_SELECTED_MISSING}"
    )
    print(
        f"- V2 account frequency   : {OUT_ACCOUNT_FREQ}"
    )

    print(
        "\n해석 원칙:"
        "\n- H6C2B strong-total 통계는 V1 diagnostic으로만 보존"
        "\n- V2는 exact label / exact account-id만 total strong으로 인정"
        "\n- assets CurrentAssets 등은 component로 분리"
        "\n- net income pretax/continuing/discontinued/attributable은 total에서 제외"
        "\n- 다음 단계는 V2 결과를 기준으로 revenue / net-income / 4개 small residual을 분리"
        "\n- 아직 어떤 값도 merge하지 않음"
    )


if __name__ == "__main__":
    main()
