from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6A1. Safe Selector Extension - Exact 3 Case Audit
#
# 목적
# ------------------------------------------------------------
# H6A에서
#   review_safe_selector_extension_before_source_fallback = 3
# 로 분류된 receipt만 정확히 확인한다.
#
# API 호출 없음.
#
# 확인 내용
# ------------------------------------------------------------
# - 어떤 receipt / metric인지
# - H3 raw full-statement의 account_id / account_nm
# - thstrm_amount / thstrm_add_amount
# - sj_div / fs_div / ord
# - H3 selector가 현재 어떤 값을 골랐는지
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h6a1_dart_safe_selector_extension_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TRIAGE = (
    INTERIM
    / "dart_noncorrected_remaining_coverage_triage.parquet"
)

PLAN = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_plan.parquet"
)

LABELS = (
    INTERIM
    / "dart_noncorrected_core5_candidate_labels.csv"
)

H3_ROWS = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)

H3_SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

OUT = (
    INTERIM
    / "dart_noncorrected_safe_selector_extension_exact_cases.csv"
)


def rcept_str(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def main():

    for path in [
        TRIAGE,
        PLAN,
        LABELS,
        H3_ROWS,
        H3_SELECTED,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    triage = pd.read_parquet(TRIAGE)
    plan = pd.read_parquet(PLAN)

    labels = pd.read_csv(
        LABELS,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    rows = pd.read_parquet(H3_ROWS)
    selected = pd.read_parquet(H3_SELECTED)

    triage["rcept_no"] = rcept_str(triage["rcept_no"])
    plan["rcept_no"] = rcept_str(plan["rcept_no"])
    labels["rcept_no"] = rcept_str(labels["rcept_no"])
    rows["_expected_rcept_no"] = rcept_str(rows["_expected_rcept_no"])
    selected["_expected_rcept_no"] = rcept_str(
        selected["_expected_rcept_no"]
    )

    # recommended_next_action is created in H6A's PLAN artifact,
    # not in the full TRIAGE artifact.
    targets = plan.loc[
        plan[
            "recommended_next_action"
        ].eq(
            "review_safe_selector_extension_before_source_fallback"
        )
    ].copy()

    # Defensive fallback: if PLAN schema changes later, derive the same
    # target group from TRIAGE's core5_review_class.
    if targets.empty:
        if "core5_review_class" not in triage.columns:
            raise RuntimeError(
                "PLAN에서 safe-selector target을 찾지 못했고, "
                "TRIAGE에도 core5_review_class가 없습니다."
            )

        targets = triage.loc[
            triage[
                "core5_review_class"
            ]
            .astype("string")
            .fillna("")
            .str.contains(
                "possible_selector_extension",
                regex=False,
            )
        ].copy()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H6A1 SAFE SELECTOR EXTENSION EXACT CASE AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        f"\nTargets: {len(targets):,}"
    )

    if len(targets) != 3:
        print(
            "WARNING: H6A summary에서는 3건이 예상됩니다. "
            f"현재={len(targets)}"
        )

    output_rows = []

    for idx, (_, target) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        receipt = str(target["rcept_no"])

        missing_metrics = [
            metric
            for metric in [
                "assets",
                "liabilities",
                "equity_total",
                "revenue_cumulative",
                "operating_income_cumulative",
                "net_income_total_cumulative",
            ]
            if pd.isna(
                target.get(metric)
            )
        ]

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{target.get('stock_code')} | "
            f"{target.get('corp_name')} | "
            f"{target.get('canonical_period_key')} | "
            f"{receipt}"
        )

        print(
            "  missing: "
            + " | ".join(missing_metrics)
        )

        print(
            "  review class: "
            + str(
                target.get(
                    "core5_review_class"
                )
            )
        )

        label_hit = labels.loc[
            labels["rcept_no"].eq(receipt)
        ].copy()

        raw = rows.loc[
            rows["_expected_rcept_no"].eq(receipt)
        ].copy()

        sel = selected.loc[
            selected["_expected_rcept_no"].eq(receipt)
        ].copy()

        if not label_hit.empty:
            print(
                "\n  [H6A candidate rows]"
            )

            show = [
                "missing_metric",
                "classification",
                "fs_div",
                "sj_div",
                "account_id",
                "account_nm",
                "thstrm_amount",
                "thstrm_add_amount",
                "ord",
            ]

            show = [
                c
                for c in show
                if c in label_hit.columns
            ]

            print(
                label_hit[
                    show
                ].to_string(
                    index=False
                )
            )

        print(
            "\n  [H3 currently selected metrics]"
        )

        if sel.empty:
            print(
                "  None"
            )
        else:
            show_sel = [
                "metric",
                "metric_value",
                "account_id",
                "account_nm",
                "sj_div",
                "thstrm_amount",
                "thstrm_add_amount",
                "selection_method",
                "selection_score",
            ]

            show_sel = [
                c
                for c in show_sel
                if c in sel.columns
            ]

            print(
                sel[
                    show_sel
                ].to_string(
                    index=False
                )
            )

        # For each candidate selected by H6A, attach raw-row details
        # with matching account_id/account_nm where possible.
        for _, cand in label_hit.iterrows():

            matches = raw.copy()

            if "account_id" in cand.index:
                matches = matches.loc[
                    matches[
                        "account_id"
                    ].astype("string")
                    .fillna("")
                    .eq(
                        str(
                            cand.get(
                                "account_id"
                            )
                        )
                    )
                ]

            if (
                "account_nm"
                in cand.index
                and not matches.empty
            ):
                exact_name = matches[
                    "account_nm"
                ].astype("string").fillna("").eq(
                    str(
                        cand.get(
                            "account_nm"
                        )
                    )
                )

                if exact_name.any():
                    matches = matches.loc[
                        exact_name
                    ]

            if matches.empty:
                output_rows.append(
                    {
                        **cand.to_dict(),
                        "raw_match_found":
                        False,
                    }
                )
                continue

            for _, raw_row in matches.iterrows():
                record = cand.to_dict()

                record.update(
                    {
                        "raw_match_found":
                        True,

                        "raw_requested_fs_div":
                        raw_row.get(
                            "_requested_fs_div"
                        ),

                        "raw_sj_div":
                        raw_row.get(
                            "sj_div"
                        ),

                        "raw_account_id":
                        raw_row.get(
                            "account_id"
                        ),

                        "raw_account_nm":
                        raw_row.get(
                            "account_nm"
                        ),

                        "raw_thstrm_amount":
                        raw_row.get(
                            "thstrm_amount"
                        ),

                        "raw_thstrm_add_amount":
                        raw_row.get(
                            "thstrm_add_amount"
                        ),

                        "raw_ord":
                        raw_row.get(
                            "ord"
                        ),

                        "raw_currency":
                        raw_row.get(
                            "currency"
                        ),
                    }
                )

                output_rows.append(
                    record
                )

    out = pd.DataFrame(
        output_rows
    )

    out.to_csv(
        OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "H6A1 SUMMARY"
    )

    print(
        "=" * 100
    )

    if out.empty:
        print(
            "\nNo candidate rows."
        )

    else:
        summary_cols = [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "missing_metric",
            "classification",
            "raw_sj_div",
            "raw_account_id",
            "raw_account_nm",
            "raw_thstrm_amount",
            "raw_thstrm_add_amount",
            "raw_ord",
        ]

        summary_cols = [
            c
            for c in summary_cols
            if c in out.columns
        ]

        print(
            "\n[Exact safe-extension candidates]"
        )

        print(
            out[
                summary_cols
            ].to_string(
                index=False
            )
        )

    print(
        f"\nOutput: {OUT}"
    )

    print(
        "\n다음 판단:"
        "\n- 명백한 total account + 올바른 cumulative field이면 H3 selector에 최소 범위로 추가"
        "\n- parent/NCI/세전이익/부분 revenue이면 추가 금지"
        "\n- 이 3건을 닫은 뒤 890 no-data receipt-specific source fallback으로 이동"
    )


if __name__ == "__main__":
    main()
