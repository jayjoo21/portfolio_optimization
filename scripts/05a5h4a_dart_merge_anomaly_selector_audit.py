from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H4A. Merge Anomaly + Remaining Selector Audit
#
# 목적
# ------------------------------------------------------------
# H4 이후 바로 source-level fallback으로 가기 전에:
#
# 1) H2 major vs H3 full overlap mismatch 10건 상세 확인
# 2) hard balance failure 4건에서
#    H2 / H3 / final balance를 각각 비교
# 3) H3 API는 available이지만 최종 core 5/6인 285건에 대해
#    raw full-statement에 놓친 account_id/account_nm 후보가 있는지 집계
#
# API 호출 없음.
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_overlap_mismatch_detail.csv
#   dart_noncorrected_balance_failure_diagnostic.csv
#   dart_noncorrected_remaining_metric_candidate_labels.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h4a_dart_merge_anomaly_selector_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

FINAL_WIDE = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide.parquet"
)

OVERLAP_QA = (
    INTERIM
    / "dart_noncorrected_merge_overlap_qa.csv"
)

H2_COVERAGE = (
    INTERIM
    / "dart_noncorrected_major_account_coverage.parquet"
)

H1_ROWS = (
    INTERIM
    / "dart_noncorrected_multi_account_rows.parquet"
)

H3_ROWS = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)

H3_SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

H3_MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

OUT_OVERLAP_DETAIL = (
    INTERIM
    / "dart_noncorrected_overlap_mismatch_detail.csv"
)

OUT_BALANCE = (
    INTERIM
    / "dart_noncorrected_balance_failure_diagnostic.csv"
)

OUT_CANDIDATES = (
    INTERIM
    / "dart_noncorrected_remaining_metric_candidate_labels.csv"
)


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]

BALANCE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
]

ABS_BALANCE_TOLERANCE_KRW = 2_000_000
REL_BALANCE_TOLERANCE = 1e-6


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


def norm(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        "",
        str(value),
    ).lower()


def parse_num(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {
        "",
        "-",
        "nan",
        "None",
        "<NA>",
    }:
        return np.nan

    text = text.replace(
        ",",
        "",
    )

    if (
        text.startswith("(")
        and text.endswith(")")
    ):
        text = (
            "-"
            + text[1:-1]
        )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def balance_result(
    assets,
    liabilities,
    equity,
):
    if any(
        pd.isna(v)
        for v in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return {
            "gap": np.nan,
            "rel_gap": np.nan,
            "qa": "missing",
        }

    gap = (
        float(assets)
        - float(liabilities)
        - float(equity)
    )

    rel = (
        abs(gap)
        / abs(float(assets))
        if float(assets) != 0
        else np.nan
    )

    if gap == 0:
        qa = "exact_pass"
    elif (
        abs(gap)
        <= ABS_BALANCE_TOLERANCE_KRW
        or (
            pd.notna(rel)
            and rel
            <= REL_BALANCE_TOLERANCE
        )
    ):
        qa = "rounding_pass"
    else:
        qa = "fail"

    return {
        "gap": gap,
        "rel_gap": rel,
        "qa": qa,
    }


def h1_metric_candidate_mask(
    df: pd.DataFrame,
    metric: str,
) -> pd.Series:

    names = (
        df[
            "account_nm"
        ]
        .astype("string")
        .fillna("")
        .map(norm)
    )

    if metric == "assets":
        return names.eq(
            "자산총계"
        )

    if metric == "liabilities":
        return names.eq(
            "부채총계"
        )

    if metric == "equity_total":
        return names.eq(
            "자본총계"
        )

    if metric == "revenue_cumulative":
        return names.str.contains(
            "매출|영업수익|수익",
            regex=True,
        )

    if metric == "operating_income_cumulative":
        return names.str.contains(
            "영업이익|영업손익",
            regex=True,
        )

    if metric == "net_income_total_cumulative":
        return names.str.contains(
            "당기순이익|분기순이익|반기순이익|순손익",
            regex=True,
        )

    return pd.Series(
        False,
        index=df.index,
    )


def h3_missing_candidate_mask(
    df: pd.DataFrame,
    metric: str,
) -> pd.Series:

    account_nm = (
        df[
            "account_nm"
        ]
        .astype("string")
        .fillna("")
        .map(norm)
    )

    account_id = (
        df[
            "account_id"
        ]
        .astype("string")
        .fillna("")
        .map(norm)
    )

    sj = (
        df[
            "sj_div"
        ]
        .astype("string")
        .fillna("")
        .str.upper()
    )

    if metric == "assets":
        return (
            sj.eq("BS")
            & (
                account_nm.str.contains(
                    "자산총계",
                    regex=False,
                )
                | account_id.str.contains(
                    "assets",
                    regex=False,
                )
            )
        )

    if metric == "liabilities":
        return (
            sj.eq("BS")
            & (
                account_nm.str.contains(
                    "부채총계",
                    regex=False,
                )
                | account_id.str.contains(
                    "liabil",
                    regex=False,
                )
            )
        )

    if metric == "equity_total":
        return (
            sj.eq("BS")
            & (
                account_nm.str.contains(
                    "자본총계",
                    regex=False,
                )
                | account_id.str.contains(
                    "equity",
                    regex=False,
                )
            )
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
                    "매출|영업수익|보험수익|수익합계|수익총계|영업수익합계|순영업수익",
                    regex=True,
                )
                | account_id.str.contains(
                    "revenue|operatingrevenue|insurancecontractrevenue",
                    regex=True,
                )
            )
        )

    if metric == "operating_income_cumulative":
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

    if metric == "net_income_total_cumulative":
        return (
            sj.isin(
                [
                    "IS",
                    "CIS",
                ]
            )
            & (
                account_nm.str.contains(
                    "당기순이익|분기순이익|반기순이익|연결당기순이익|순손익",
                    regex=True,
                )
                | account_id.str.contains(
                    "profitloss",
                    regex=False,
                )
            )
        )

    return pd.Series(
        False,
        index=df.index,
    )


def main():

    required = [
        FINAL_WIDE,
        OVERLAP_QA,
        H2_COVERAGE,
        H1_ROWS,
        H3_ROWS,
        H3_SELECTED,
        H3_MANIFEST,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    final = pd.read_parquet(
        FINAL_WIDE
    )

    overlap = pd.read_csv(
        OVERLAP_QA,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    h2 = pd.read_parquet(
        H2_COVERAGE
    )

    h1_rows = pd.read_parquet(
        H1_ROWS
    )

    h3_rows = pd.read_parquet(
        H3_ROWS
    )

    h3_sel = pd.read_parquet(
        H3_SELECTED
    )

    h3_manifest = pd.read_parquet(
        H3_MANIFEST
    )

    for df, col in [
        (
            final,
            "rcept_no",
        ),
        (
            overlap,
            "rcept_no",
        ),
        (
            h2,
            "rcept_no",
        ),
        (
            h1_rows,
            "rcept_no",
        ),
        (
            h3_manifest,
            "rcept_no",
        ),
    ]:
        df[
            col
        ] = receipt_string(
            df[
                col
            ]
        )

    h3_rows[
        "_expected_rcept_no"
    ] = receipt_string(
        h3_rows[
            "_expected_rcept_no"
        ]
    )

    h3_sel[
        "_expected_rcept_no"
    ] = receipt_string(
        h3_sel[
            "_expected_rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "05A5-H4A MERGE ANOMALY + REMAINING SELECTOR AUDIT"
    )
    print(
        "=" * 100
    )

    # ========================================================
    # 1) OVERLAP MISMATCH DETAIL
    # ========================================================

    mismatches = overlap.loc[
        overlap[
            "overlap_match"
        ].astype(str)
        .str.lower()
        .isin(
            [
                "false",
                "0",
            ]
        )
    ].copy()

    print(
        "\n[Overlap mismatches]"
    )

    print(
        f"{len(mismatches):,}"
    )

    detail_records = []

    for _, mm in (
        mismatches.iterrows()
    ):

        receipt = str(
            mm[
                "rcept_no"
            ]
        )

        metric = str(
            mm[
                "metric"
            ]
        )

        h1r = h1_rows.loc[
            h1_rows[
                "rcept_no"
            ].eq(
                receipt
            )
        ].copy()

        if not h1r.empty:
            h1r = h1r.loc[
                h1_metric_candidate_mask(
                    h1r,
                    metric,
                )
            ].copy()

        h3r = h3_sel.loc[
            h3_sel[
                "_expected_rcept_no"
            ].eq(
                receipt
            )
            & h3_sel[
                "metric"
            ].eq(
                metric
            )
        ].copy()

        if h1r.empty:
            h1_candidates = [
                {}
            ]
        else:
            h1_candidates = (
                h1r.head(
                    10
                )
                .to_dict(
                    "records"
                )
            )

        if h3r.empty:
            h3_candidates = [
                {}
            ]
        else:
            h3_candidates = (
                h3r.head(
                    5
                )
                .to_dict(
                    "records"
                )
            )

        for a in h1_candidates:
            for b in h3_candidates:
                detail_records.append(
                    {
                        "stock_code":
                        mm.get(
                            "stock_code"
                        ),
                        "corp_name":
                        mm.get(
                            "corp_name"
                        ),
                        "canonical_period_key":
                        mm.get(
                            "canonical_period_key"
                        ),
                        "rcept_no":
                        receipt,
                        "metric":
                        metric,
                        "h2_major_value":
                        mm.get(
                            "h2_major_value"
                        ),
                        "h3_full_value":
                        mm.get(
                            "h3_full_value"
                        ),
                        "absolute_difference":
                        mm.get(
                            "absolute_difference"
                        ),
                        "relative_difference":
                        mm.get(
                            "relative_difference"
                        ),
                        "h1_fs_div":
                        a.get(
                            "fs_div"
                        ),
                        "h1_account_nm":
                        a.get(
                            "account_nm"
                        ),
                        "h1_thstrm_amount":
                        a.get(
                            "thstrm_amount"
                        ),
                        "h1_thstrm_add_amount":
                        a.get(
                            "thstrm_add_amount"
                        ),
                        "h3_fs_div":
                        b.get(
                            "_requested_fs_div"
                        ),
                        "h3_sj_div":
                        b.get(
                            "sj_div"
                        ),
                        "h3_account_id":
                        b.get(
                            "account_id"
                        ),
                        "h3_account_nm":
                        b.get(
                            "account_nm"
                        ),
                        "h3_thstrm_amount":
                        b.get(
                            "thstrm_amount"
                        ),
                        "h3_thstrm_add_amount":
                        b.get(
                            "thstrm_add_amount"
                        ),
                        "h3_selection_method":
                        b.get(
                            "selection_method"
                        ),
                        "h3_selection_score":
                        b.get(
                            "selection_score"
                        ),
                    }
                )

    overlap_detail = pd.DataFrame(
        detail_records
    )

    overlap_detail.to_csv(
        OUT_OVERLAP_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )

    if not mismatches.empty:
        print(
            mismatches[
                [
                    "stock_code",
                    "corp_name",
                    "canonical_period_key",
                    "rcept_no",
                    "metric",
                    "h2_major_value",
                    "h3_full_value",
                    "absolute_difference",
                    "relative_difference",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 2) BALANCE FAILURE DIAGNOSTIC
    # ========================================================

    balance_fail = final.loc[
        final[
            "balance_qa_class"
        ].eq(
            "fail"
        )
    ].copy()

    print(
        "\n[Hard balance failures]"
    )

    print(
        f"{len(balance_fail):,}"
    )

    balance_records = []

    for _, row in (
        balance_fail.iterrows()
    ):

        receipt = str(
            row[
                "rcept_no"
            ]
        )

        h2_row = h2.loc[
            h2[
                "rcept_no"
            ].eq(
                receipt
            )
        ]

        h2_row = (
            h2_row.iloc[0]
            if not h2_row.empty
            else None
        )

        h3_for_receipt = h3_sel.loc[
            h3_sel[
                "_expected_rcept_no"
            ].eq(
                receipt
            )
            & h3_sel[
                "metric"
            ].isin(
                BALANCE_METRICS
            )
        ].copy()

        h3_map = (
            h3_for_receipt.set_index(
                "metric"
            )[
                "metric_value"
            ]
            .to_dict()
            if not h3_for_receipt.empty
            else {}
        )

        h2_bal = balance_result(
            (
                h2_row[
                    "assets"
                ]
                if h2_row is not None
                else np.nan
            ),
            (
                h2_row[
                    "liabilities"
                ]
                if h2_row is not None
                else np.nan
            ),
            (
                h2_row[
                    "equity_total"
                ]
                if h2_row is not None
                else np.nan
            ),
        )

        h3_bal = balance_result(
            h3_map.get(
                "assets"
            ),
            h3_map.get(
                "liabilities"
            ),
            h3_map.get(
                "equity_total"
            ),
        )

        final_bal = balance_result(
            row[
                "assets"
            ],
            row[
                "liabilities"
            ],
            row[
                "equity_total"
            ],
        )

        h3_manifest_row = (
            h3_manifest.loc[
                h3_manifest[
                    "rcept_no"
                ].eq(
                    receipt
                )
            ]
        )

        if not h3_manifest_row.empty:
            hm = h3_manifest_row.iloc[0]
            h3_status = hm.get(
                "collection_status"
            )
            h3_fs = hm.get(
                "chosen_fs_div"
            )
        else:
            h3_status = None
            h3_fs = None

        if h3_bal[
            "qa"
        ] in {
            "exact_pass",
            "rounding_pass",
        }:
            diagnosis = (
                "h3_full_balance_pass_candidate_coherent_override"
            )

        elif h3_bal[
            "qa"
        ] == "fail":
            diagnosis = (
                "h3_full_balance_also_fails_source_level_crosscheck"
            )

        else:
            diagnosis = (
                "h3_balance_not_available_targeted_full_or_source_crosscheck"
            )

        balance_records.append(
            {
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

                "final_assets":
                row[
                    "assets"
                ],
                "final_liabilities":
                row[
                    "liabilities"
                ],
                "final_equity":
                row[
                    "equity_total"
                ],
                "final_assets_source":
                row.get(
                    "assets_source"
                ),
                "final_liabilities_source":
                row.get(
                    "liabilities_source"
                ),
                "final_equity_source":
                row.get(
                    "equity_total_source"
                ),
                "final_gap":
                final_bal[
                    "gap"
                ],
                "final_rel_gap":
                final_bal[
                    "rel_gap"
                ],

                "h2_assets":
                (
                    h2_row[
                        "assets"
                    ]
                    if h2_row is not None
                    else np.nan
                ),
                "h2_liabilities":
                (
                    h2_row[
                        "liabilities"
                    ]
                    if h2_row is not None
                    else np.nan
                ),
                "h2_equity":
                (
                    h2_row[
                        "equity_total"
                    ]
                    if h2_row is not None
                    else np.nan
                ),
                "h2_gap":
                h2_bal[
                    "gap"
                ],
                "h2_rel_gap":
                h2_bal[
                    "rel_gap"
                ],
                "h2_qa":
                h2_bal[
                    "qa"
                ],

                "h3_collection_status":
                h3_status,
                "h3_fs":
                h3_fs,
                "h3_assets":
                h3_map.get(
                    "assets"
                ),
                "h3_liabilities":
                h3_map.get(
                    "liabilities"
                ),
                "h3_equity":
                h3_map.get(
                    "equity_total"
                ),
                "h3_gap":
                h3_bal[
                    "gap"
                ],
                "h3_rel_gap":
                h3_bal[
                    "rel_gap"
                ],
                "h3_qa":
                h3_bal[
                    "qa"
                ],

                "diagnosis":
                diagnosis,
            }
        )

    balance_diag = pd.DataFrame(
        balance_records
    )

    balance_diag.to_csv(
        OUT_BALANCE,
        index=False,
        encoding="utf-8-sig",
    )

    if not balance_diag.empty:
        show_cols = [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "final_gap",
            "final_rel_gap",
            "h2_qa",
            "h3_collection_status",
            "h3_fs",
            "h3_gap",
            "h3_rel_gap",
            "h3_qa",
            "diagnosis",
        ]

        print(
            balance_diag[
                show_cols
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 3) REMAINING 5/6 SELECTOR CANDIDATES
    # ========================================================

    h3_available_receipts = set(
        h3_manifest.loc[
            h3_manifest[
                "collection_status"
            ].eq(
                "available"
            ),
            "rcept_no",
        ].astype(str)
    )

    remaining_5 = final.loc[
        final[
            "core_metric_count_final"
        ].eq(
            5
        )
        & final[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            h3_available_receipts
        )
    ].copy()

    print(
        "\n[H3 available but final core=5]"
    )

    print(
        f"{len(remaining_5):,}"
    )

    candidate_rows = []

    for _, rec in (
        remaining_5.iterrows()
    ):

        receipt = str(
            rec[
                "rcept_no"
            ]
        )

        missing_metrics = [
            metric
            for metric
            in CORE_METRICS
            if pd.isna(
                rec[
                    metric
                ]
            )
        ]

        raw = h3_rows.loc[
            h3_rows[
                "_expected_rcept_no"
            ].eq(
                receipt
            )
        ].copy()

        for metric in missing_metrics:

            if raw.empty:
                continue

            cand = raw.loc[
                h3_missing_candidate_mask(
                    raw,
                    metric,
                )
            ].copy()

            if cand.empty:
                continue

            for _, cr in (
                cand.iterrows()
            ):

                candidate_rows.append(
                    {
                        "missing_metric":
                        metric,
                        "stock_code":
                        rec[
                            "stock_code"
                        ],
                        "corp_name":
                        rec.get(
                            "corp_name"
                        ),
                        "canonical_period_key":
                        rec[
                            "canonical_period_key"
                        ],
                        "rcept_no":
                        receipt,
                        "fs_div":
                        cr.get(
                            "_requested_fs_div"
                        ),
                        "sj_div":
                        cr.get(
                            "sj_div"
                        ),
                        "account_id":
                        cr.get(
                            "account_id"
                        ),
                        "account_nm":
                        cr.get(
                            "account_nm"
                        ),
                        "thstrm_amount":
                        cr.get(
                            "thstrm_amount"
                        ),
                        "thstrm_add_amount":
                        cr.get(
                            "thstrm_add_amount"
                        ),
                        "ord":
                        cr.get(
                            "ord"
                        ),
                        "currency":
                        cr.get(
                            "currency"
                        ),
                    }
                )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates.to_csv(
        OUT_CANDIDATES,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Remaining missing metrics]"
    )

    missing_series = (
        remaining_5[
            "missing_core_metrics_final"
        ]
        .value_counts()
    )

    print(
        missing_series.to_string()
    )

    if candidates.empty:
        print(
            "\nNo candidate labels found in H3 raw rows."
        )

    else:
        print(
            "\n[Top candidate account labels by missing metric]"
        )

        summary = (
            candidates.groupby(
                [
                    "missing_metric",
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
                    30
                )
                .to_string(
                    index=False
                )
            )

    print(
        f"\nOverlap detail : "
        f"{OUT_OVERLAP_DETAIL}"
    )

    print(
        f"Balance diag   : "
        f"{OUT_BALANCE}"
    )

    print(
        f"Label candidates: "
        f"{OUT_CANDIDATES}"
    )

    print(
        "\n다음 판단:"
        "\n- overlap mismatch 원인이 selector/period-field 차이면 H2/H3 규칙 수정"
        "\n- H3 balance가 PASS면 3개 balance metric을 coherent H3 set으로 교체 후보"
        "\n- H3 balance도 FAIL/미수집이면 same-receipt source cross-check"
        "\n- 285 remaining에서 반복되는 안전한 total-account 패턴이 보이면 selector 보완"
        "\n- 그 후에만 source-level H5 대상으로 확정"
    )


if __name__ == "__main__":
    main()
