from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6A. Remaining Non-Corrected Fundamental Coverage Triage
#
# 목적
# ------------------------------------------------------------
# H5E 이후 남은 coverage 문제를 서로 다른 성격으로 분리한다.
#
# 1) core=6
#    → complete
#
# 2) core=5
#    → H3 full API는 available이지만 metric 하나가 여전히 missing.
#      raw full-statement 안의 후보 account_id/account_nm을 다시 집계해
#      selector 보완 가능한지 / 업종별 의미 문제인지 판단.
#
# 3) core=3 + balance_resolution =
#      cfs_fails_same_receipt_xbrl_ofs_passes_do_not_mix_basis
#    → 일진전기/드림텍처럼 "의도적으로 missing 처리한 CFS anomaly".
#      source fallback 대상에서 제외.
#
# 4) core=0 + H3 no_data
#    → Multi-account + Full API 모두 no_data.
#      receipt-specific XBRL/document source-level fallback 후보.
#
# API 호출 없음.
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_remaining_coverage_triage.parquet/csv
# dart_noncorrected_core5_candidate_labels.csv
# dart_noncorrected_full_api_no_data_scope.csv
# dart_noncorrected_source_level_fallback_plan.parquet/csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h6a_dart_remaining_coverage_triage.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

FINAL = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_final.parquet"
)

H3_ROWS = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)

H3_MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

OUT_TRIAGE_PARQUET = (
    INTERIM
    / "dart_noncorrected_remaining_coverage_triage.parquet"
)

OUT_TRIAGE_CSV = (
    INTERIM
    / "dart_noncorrected_remaining_coverage_triage.csv"
)

OUT_LABELS = (
    INTERIM
    / "dart_noncorrected_core5_candidate_labels.csv"
)

OUT_NODATA = (
    INTERIM
    / "dart_noncorrected_full_api_no_data_scope.csv"
)

OUT_PLAN_PARQUET = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_plan.parquet"
)

OUT_PLAN_CSV = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_plan.csv"
)


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]


def receipt_string(
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


def norm(
    value,
) -> str:

    if pd.isna(
        value
    ):
        return ""

    return re.sub(
        r"\s+",
        "",
        str(
            value
        ),
    ).lower()


def missing_metrics(
    row: pd.Series,
) -> list[
    str
]:

    return [
        metric
        for metric
        in CORE_METRICS
        if pd.isna(
            row.get(
                metric
            )
        )
    ]


def candidate_mask(
    df: pd.DataFrame,
    metric: str,
) -> pd.Series:

    account_nm = (
        df[
            "account_nm"
        ]
        .astype("string")
        .fillna("")
        .map(
            norm
        )
    )

    account_id = (
        df[
            "account_id"
        ]
        .astype("string")
        .fillna("")
        .map(
            norm
        )
    )

    sj = (
        df[
            "sj_div"
        ]
        .astype("string")
        .fillna("")
        .str.upper()
    )

    if metric == "revenue_cumulative":

        return (
            sj.isin(
                [
                    "IS",
                    "CIS",
                ]
            )
            & (
                account_nm.str.contains(
                    (
                        "매출|수익|보험|이자|"
                        "영업수익|순영업|"
                        "영업수익합계"
                    ),
                    regex=True,
                )
                | account_id.str.contains(
                    (
                        "revenue|"
                        "operatingincome|"
                        "insurance|"
                        "interest"
                    ),
                    regex=True,
                )
            )
        )

    if (
        metric
        == "net_income_total_cumulative"
    ):

        return (
            sj.isin(
                [
                    "IS",
                    "CIS",
                ]
            )
            & (
                account_nm.str.contains(
                    (
                        "당기순이익|"
                        "분기순이익|"
                        "반기순이익|"
                        "당기순손익|"
                        "분기순손익|"
                        "반기순손익|"
                        "연결.*순이익|"
                        "연결.*순손익"
                    ),
                    regex=True,
                )
                | account_id.str.contains(
                    "profitloss",
                    regex=False,
                )
            )
        )

    if (
        metric
        == "operating_income_cumulative"
    ):

        return (
            sj.isin(
                [
                    "IS",
                    "CIS",
                ]
            )
            & (
                account_nm.str.contains(
                    "영업이익|영업손익",
                    regex=True,
                )
                | account_id.str.contains(
                    "operatingincomeloss",
                    regex=False,
                )
            )
        )

    if metric == "assets":

        return (
            sj.eq(
                "BS"
            )
            & (
                account_nm.str.contains(
                    "자산총계",
                    regex=False,
                )
                | account_id.isin(
                    [
                        "ifrs_assets",
                        "ifrs-full_assets",
                    ]
                )
            )
        )

    if metric == "liabilities":

        return (
            sj.eq(
                "BS"
            )
            & (
                account_nm.str.contains(
                    "부채총계",
                    regex=False,
                )
                | account_id.isin(
                    [
                        "ifrs_liabilities",
                        "ifrs-full_liabilities",
                    ]
                )
            )
        )

    if metric == "equity_total":

        return (
            sj.eq(
                "BS"
            )
            & (
                account_nm.str.contains(
                    "자본총계",
                    regex=False,
                )
                | account_id.isin(
                    [
                        "ifrs_equity",
                        "ifrs-full_equity",
                    ]
                )
            )
        )

    return pd.Series(
        False,
        index=df.index,
    )


def classify_core5_case(
    metric: str,
    candidates: pd.DataFrame,
) -> str:

    if candidates.empty:
        return (
            "raw_full_statement_has_no_obvious_candidate"
        )

    names = (
        candidates[
            "account_nm"
        ]
        .astype("string")
        .fillna("")
        .map(
            norm
        )
    )

    ids = (
        candidates[
            "account_id"
        ]
        .astype("string")
        .fillna("")
        .map(
            norm
        )
    )

    if metric == "revenue_cumulative":

        # Explicit generic total revenue / sales expressions.
        safe_names = {
            "매출",
            "매출액",
            "수익(매출액)",
            "영업수익",
            "영업수익(매출액)",
        }

        if any(
            name in safe_names
            for name in names
        ):
            return (
                "possible_selector_extension_generic_revenue"
            )

        # Financial-sector component-like revenue rows should not
        # be silently promoted to total revenue.
        financial_tokens = [
            "이자수익",
            "순이자",
            "보험수익",
            "보험영업수익",
            "투자영업수익",
            "배당수익",
            "배당금수익",
            "재보험수익",
        ]

        if any(
            any(
                token in name
                for token
                in financial_tokens
            )
            for name
            in names
        ):
            return (
                "sector_revenue_semantics_review"
            )

        return (
            "revenue_candidate_present_but_not_safe_total"
        )

    if (
        metric
        == "net_income_total_cumulative"
    ):

        safe_names = {
            "당기순이익",
            "당기순이익(손실)",
            "분기순이익",
            "분기순이익(손실)",
            "반기순이익",
            "반기순이익(손실)",
            "연결당기순이익",
            "연결당기순이익(손실)",
            "연결분기순이익",
            "연결분기순이익(손실)",
            "연결반기순이익",
            "연결반기순이익(손실)",
            "당기순손익",
            "분기순손익",
            "반기순손익",
            "연결당기순손익",
            "연결분기순손익",
            "연결반기순손익",
        }

        if any(
            name in safe_names
            for name in names
        ):
            return (
                "possible_selector_extension_total_net_income"
            )

        owner_tokens = [
            "지배기업",
            "비지배",
            "ownersofparent",
            "noncontrolling",
        ]

        if all(
            any(
                token in name
                or token in aid
                for token
                in owner_tokens
            )
            for name, aid
            in zip(
                names,
                ids,
            )
        ):
            return (
                "only_parent_or_nci_profit_components"
            )

        return (
            "net_income_candidate_present_but_not_safe_total"
        )

    if (
        metric
        == "operating_income_cumulative"
    ):
        return (
            "operating_income_candidate_review"
        )

    if metric in {
        "assets",
        "liabilities",
        "equity_total",
    }:
        return (
            "balance_metric_candidate_review"
        )

    return "manual_review"


def main():

    for path in [
        FINAL,
        H3_ROWS,
        H3_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    final = pd.read_parquet(
        FINAL
    )

    h3_rows = pd.read_parquet(
        H3_ROWS
    )

    h3_manifest = pd.read_parquet(
        H3_MANIFEST
    )

    final[
        "rcept_no"
    ] = receipt_string(
        final[
            "rcept_no"
        ]
    )

    h3_rows[
        "_expected_rcept_no"
    ] = receipt_string(
        h3_rows[
            "_expected_rcept_no"
        ]
    )

    h3_manifest[
        "rcept_no"
    ] = receipt_string(
        h3_manifest[
            "rcept_no"
        ]
    )

    manifest_small = (
        h3_manifest[
            [
                "rcept_no",
                "collection_status",
                "pit_verified",
                "chosen_fs_div",
                "api_status",
                "api_message",
                "fallback_reason",
            ]
        ]
        .rename(
            columns={
                "collection_status":
                "h3_collection_status_manifest",

                "pit_verified":
                "h3_pit_verified_manifest",

                "chosen_fs_div":
                "h3_chosen_fs_div_manifest",

                "api_status":
                "h3_api_status_manifest",

                "api_message":
                "h3_api_message_manifest",

                "fallback_reason":
                "h3_fallback_reason_manifest",
            }
        )
    )

    # H4/H5 final artifact already contains H3 provenance columns.
    # Merge manifest with temporary *_manifest names, then coalesce.
    triage = final.merge(
        manifest_small,
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    provenance_pairs = [
        (
            "h3_collection_status",
            "h3_collection_status_manifest",
        ),
        (
            "h3_pit_verified",
            "h3_pit_verified_manifest",
        ),
        (
            "h3_chosen_fs_div",
            "h3_chosen_fs_div_manifest",
        ),
        (
            "h3_api_status",
            "h3_api_status_manifest",
        ),
        (
            "h3_api_message",
            "h3_api_message_manifest",
        ),
        (
            "h3_fallback_reason",
            "h3_fallback_reason_manifest",
        ),
    ]

    for base_col, manifest_col in provenance_pairs:

        if base_col in triage.columns:
            triage[
                base_col
            ] = triage[
                base_col
            ].combine_first(
                triage[
                    manifest_col
                ]
            )

        else:
            triage[
                base_col
            ] = triage[
                manifest_col
            ]

    triage = triage.drop(
        columns=[
            manifest_col
            for _, manifest_col
            in provenance_pairs
            if manifest_col
            in triage.columns
        ]
    )

    triage[
        "missing_metric_list"
    ] = triage.apply(
        lambda row:
        missing_metrics(
            row
        ),
        axis=1,
    )

    triage[
        "missing_core_metrics_h6"
    ] = triage[
        "missing_metric_list"
    ].map(
        lambda xs:
        " | ".join(
            xs
        )
    )

    # --------------------------------------------------------
    # High-level bucket
    # --------------------------------------------------------

    def bucket(
        row,
    ):

        count = int(
            row[
                "core_metric_count_final"
            ]
        )

        resolution = str(
            row.get(
                "balance_resolution",
                "",
            )
        )

        h3_status = str(
            row.get(
                "h3_collection_status",
                "",
            )
        )

        if count == 6:
            return "complete"

        if (
            count == 3
            and resolution
            == "cfs_fails_same_receipt_xbrl_ofs_passes_do_not_mix_basis"
        ):
            return (
                "closed_intentional_cfs_balance_missing"
            )

        if (
            count == 0
            and h3_status == "no_data"
        ):
            return (
                "source_level_fallback_full_api_no_data"
            )

        if (
            count == 5
            and h3_status == "available"
        ):
            return (
                "core5_selector_or_semantics_review"
            )

        return (
            "other_manual_review"
        )

    triage[
        "coverage_bucket"
    ] = triage.apply(
        bucket,
        axis=1,
    )

    # Sanity checks: these counts are implied by the final core counts.
    expected_core5 = int(
        triage[
            "core_metric_count_final"
        ]
        .eq(
            5
        )
        .sum()
    )

    bucketed_core5 = int(
        triage[
            "coverage_bucket"
        ]
        .eq(
            "core5_selector_or_semantics_review"
        )
        .sum()
    )

    expected_core0 = int(
        triage[
            "core_metric_count_final"
        ]
        .eq(
            0
        )
        .sum()
    )

    bucketed_core0 = int(
        triage[
            "coverage_bucket"
        ]
        .eq(
            "source_level_fallback_full_api_no_data"
        )
        .sum()
    )

    if (
        expected_core5
        != bucketed_core5
    ):
        raise RuntimeError(
            "H6A core=5 bucket mismatch: "
            f"expected={expected_core5}, "
            f"bucketed={bucketed_core5}. "
            "H3 provenance merge를 확인하세요."
        )

    if (
        expected_core0
        != bucketed_core0
    ):
        raise RuntimeError(
            "H6A core=0 bucket mismatch: "
            f"expected={expected_core0}, "
            f"bucketed={bucketed_core0}. "
            "H3 provenance merge를 확인하세요."
        )

    # --------------------------------------------------------
    # Core=5 detailed classification
    # --------------------------------------------------------

    core5 = triage.loc[
        triage[
            "coverage_bucket"
        ].eq(
            "core5_selector_or_semantics_review"
        )
    ].copy()

    detailed_rows = []
    label_rows = []

    for _, row in core5.iterrows():

        receipt = str(
            row[
                "rcept_no"
            ]
        )

        missing = (
            row[
                "missing_metric_list"
            ]
        )

        raw = h3_rows.loc[
            h3_rows[
                "_expected_rcept_no"
            ].eq(
                receipt
            )
        ].copy()

        case_classes = []

        for metric in missing:

            candidates = raw.loc[
                candidate_mask(
                    raw,
                    metric,
                )
            ].copy()

            classification = (
                classify_core5_case(
                    metric,
                    candidates,
                )
            )

            case_classes.append(
                (
                    metric,
                    classification,
                )
            )

            for _, cand in (
                candidates.iterrows()
            ):

                label_rows.append(
                    {
                        "missing_metric":
                        metric,

                        "classification":
                        classification,

                        "stock_code":
                        row[
                            "stock_code"
                        ],

                        "corp_name":
                        row.get(
                            "corp_name"
                        ),

                        "canonical_period_key":
                        row[
                            "canonical_period_key"
                        ],

                        "rcept_no":
                        receipt,

                        "fs_div":
                        cand.get(
                            "_requested_fs_div"
                        ),

                        "sj_div":
                        cand.get(
                            "sj_div"
                        ),

                        "account_id":
                        cand.get(
                            "account_id"
                        ),

                        "account_nm":
                        cand.get(
                            "account_nm"
                        ),

                        "thstrm_amount":
                        cand.get(
                            "thstrm_amount"
                        ),

                        "thstrm_add_amount":
                        cand.get(
                            "thstrm_add_amount"
                        ),

                        "ord":
                        cand.get(
                            "ord"
                        ),
                    }
                )

        detailed_rows.append(
            {
                "rcept_no":
                receipt,

                "core5_review_class":
                " | ".join(
                    f"{metric}:{cls}"
                    for metric, cls
                    in case_classes
                ),
            }
        )

    detail_df = pd.DataFrame(
        detailed_rows
    )

    if not detail_df.empty:
        triage = triage.merge(
            detail_df,
            on="rcept_no",
            how="left",
            validate="one_to_one",
        )

    else:
        triage[
            "core5_review_class"
        ] = pd.NA

    labels = pd.DataFrame(
        label_rows
    )

    # --------------------------------------------------------
    # No-data scope
    # --------------------------------------------------------

    nodata = triage.loc[
        triage[
            "coverage_bucket"
        ].eq(
            "source_level_fallback_full_api_no_data"
        )
    ].copy()

    # Useful time/report fields if available.
    if "bsns_year" not in nodata.columns:
        nodata[
            "bsns_year"
        ] = pd.to_numeric(
            nodata[
                "canonical_period_key"
            ]
            .astype(str)
            .str.extract(
                r"(\d{4})\."
            )[
                0
            ],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Source-level fallback plan
    # --------------------------------------------------------

    plan = triage.loc[
        triage[
            "coverage_bucket"
        ].isin(
            [
                "source_level_fallback_full_api_no_data",
                "core5_selector_or_semantics_review",
                "closed_intentional_cfs_balance_missing",
                "other_manual_review",
            ]
        )
    ].copy()

    def next_action(
        row,
    ):

        bucket_value = row[
            "coverage_bucket"
        ]

        if (
            bucket_value
            == "closed_intentional_cfs_balance_missing"
        ):
            return (
                "stop_keep_missing_do_not_use_ofs"
            )

        if (
            bucket_value
            == "source_level_fallback_full_api_no_data"
        ):
            return (
                "receipt_specific_xbrl_then_document"
            )

        if (
            bucket_value
            == "core5_selector_or_semantics_review"
        ):
            review = str(
                row.get(
                    "core5_review_class",
                    "",
                )
            )

            if (
                "possible_selector_extension"
                in review
            ):
                return (
                    "review_safe_selector_extension_before_source_fallback"
                )

            if (
                "sector_revenue_semantics_review"
                in review
            ):
                return (
                    "sector_policy_review_do_not_force_generic_revenue"
                )

            return (
                "source_level_or_manual_metric_review"
            )

        return (
            "manual_review"
        )

    plan[
        "recommended_next_action"
    ] = plan.apply(
        next_action,
        axis=1,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    triage.to_parquet(
        OUT_TRIAGE_PARQUET,
        index=False,
    )

    triage.to_csv(
        OUT_TRIAGE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    labels.to_csv(
        OUT_LABELS,
        index=False,
        encoding="utf-8-sig",
    )

    nodata.to_csv(
        OUT_NODATA,
        index=False,
        encoding="utf-8-sig",
    )

    plan.to_parquet(
        OUT_PLAN_PARQUET,
        index=False,
    )

    plan.to_csv(
        OUT_PLAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H6A REMAINING NON-CORRECTED FUNDAMENTAL COVERAGE TRIAGE"
    )

    print(
        "=" * 100
    )

    print(
        "\n[Coverage bucket]"
    )

    print(
        triage[
            "coverage_bucket"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Final core metric count]"
    )

    print(
        triage[
            "core_metric_count_final"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    print(
        "\n[Remaining missing combinations]"
    )

    remaining = triage.loc[
        triage[
            "core_metric_count_final"
        ].lt(
            6
        ),
        "missing_core_metrics_h6",
    ].value_counts()

    print(
        remaining.to_string()
    )

    print(
        "\n[Core=5 review classes]"
    )

    if core5.empty:
        print(
            "None"
        )

    else:
        review_counts = (
            triage.loc[
                triage[
                    "coverage_bucket"
                ].eq(
                    "core5_selector_or_semantics_review"
                ),
                "core5_review_class",
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            review_counts.to_string()
        )

    if not labels.empty:

        print(
            "\n[Top candidate labels]"
        )

        summary = (
            labels.groupby(
                [
                    "missing_metric",
                    "classification",
                    "sj_div",
                    "account_id",
                    "account_nm",
                ],
                dropna=False,
            )
            .size()
            .rename(
                "count"
            )
            .reset_index()
            .sort_values(
                [
                    "missing_metric",
                    "count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

        for metric in (
            summary[
                "missing_metric"
            ]
            .drop_duplicates()
            .tolist()
        ):

            print(
                "\n"
                + "-"
                * 100
            )

            print(
                f"[{metric}]"
            )

            print(
                summary.loc[
                    summary[
                        "missing_metric"
                    ].eq(
                        metric
                    )
                ]
                .head(
                    25
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Full-API no-data by business year]"
    )

    if nodata.empty:
        print(
            "None"
        )

    else:
        print(
            nodata[
                "bsns_year"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

        if "reprt_code" in nodata.columns:
            print(
                "\n[Full-API no-data by report code]"
            )

            print(
                nodata[
                    "reprt_code"
                ]
                .astype(str)
                .value_counts()
                .to_string()
            )

    print(
        "\n[Recommended next action]"
    )

    print(
        plan[
            "recommended_next_action"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        f"\nTriage : "
        f"{OUT_TRIAGE_PARQUET}"
    )

    print(
        f"Labels : "
        f"{OUT_LABELS}"
    )

    print(
        f"No-data: "
        f"{OUT_NODATA}"
    )

    print(
        f"Plan   : "
        f"{OUT_PLAN_PARQUET}"
    )

    print(
        "\n다음 단계:"
        "\n- core=5에서 안전한 selector 확장 후보가 있으면 먼저 selector 보완"
        "\n- 금융업 revenue처럼 total 의미가 불명확한 계정은 강제로 채우지 않음"
        "\n- full API no_data 890건만 receipt-specific XBRL -> document fallback 후보"
        "\n- intentional CFS balance anomaly 2건은 그대로 missing으로 고정"
    )


if __name__ == "__main__":
    main()
