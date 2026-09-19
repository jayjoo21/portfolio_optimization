
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

COVERAGE = INTERIM / "dart_noncorrected_nodata_document_cfs_receipt_coverage.parquet"
STATUS = INTERIM / "dart_noncorrected_nodata_document_cfs_selection_status.csv"

OUT_TRIAGE = INTERIM / "dart_noncorrected_nodata_document_recovery_triage.parquet"
OUT_TRIAGE_CSV = INTERIM / "dart_noncorrected_nodata_document_recovery_triage.csv"
OUT_ADOPTABLE = INTERIM / "dart_noncorrected_nodata_document_adoptable_strict_cfs.csv"
OUT_HARD_FAIL = INTERIM / "dart_noncorrected_nodata_document_hard_balance_fail.csv"
OUT_MISSING = INTERIM / "dart_noncorrected_nodata_document_missing_family_status.csv"

CORE = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]

METRIC_TO_FAMILY = {
    "assets": "assets",
    "liabilities": "liabilities",
    "equity_total": "equity",
    "revenue_cumulative": "revenue",
    "operating_income_cumulative": "operating_income",
    "net_income_total_cumulative": "net_income",
}

ACCEPTED = {"selected", "selected_consensus"}
AMBIGUOUS = {"ambiguous_consolidated_candidates"}
NO_RELIABLE = {"no_reliable_consolidated_candidate"}


def receipt_string(s):
    return (
        s.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def main():
    if not COVERAGE.exists():
        raise FileNotFoundError(COVERAGE)
    if not STATUS.exists():
        raise FileNotFoundError(STATUS)

    coverage = pd.read_parquet(COVERAGE)
    status = pd.read_csv(
        STATUS,
        dtype={"rcept_no": str, "stock_code": str},
        low_memory=False,
    )

    coverage["rcept_no"] = receipt_string(coverage["rcept_no"])
    status["rcept_no"] = receipt_string(status["rcept_no"])

    if coverage["rcept_no"].duplicated().any():
        raise RuntimeError("coverage duplicate rcept_no")

    dup = (
        status.groupby(["rcept_no", "account_family"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    dup = dup.loc[dup["n"] > 1]
    if not dup.empty:
        raise RuntimeError(
            "STATUS receipt-family duplicate:\n"
            + dup.head(30).to_string(index=False)
        )

    status_map = {}
    for receipt, g in status.groupby("rcept_no"):
        status_map[str(receipt)] = {
            str(r["account_family"]): str(r["selection_status"])
            for _, r in g.iterrows()
        }

    triage_records = []
    missing_records = []

    for _, row in coverage.iterrows():
        receipt = str(row["rcept_no"])
        core_count = int(row.get("strict_cfs_core_count", 0))
        balance_qa = str(row.get("balance_qa", ""))

        missing = [m for m in CORE if pd.isna(row.get(m))]
        fam_status = status_map.get(receipt, {})

        missing_details = []
        statuses_for_missing = []

        for metric in missing:
            family = METRIC_TO_FAMILY[metric]
            st = fam_status.get(family, "")
            statuses_for_missing.append(st)
            missing_details.append(f"{metric}:{st or 'status_missing'}")

            hit = status.loc[
                status["rcept_no"].eq(receipt)
                & status["account_family"].eq(family)
            ]

            rec = {
                "stock_code": row.get("stock_code"),
                "corp_name": row.get("corp_name"),
                "canonical_period_key": row.get("canonical_period_key"),
                "rcept_no": receipt,
                "missing_metric": metric,
                "account_family": family,
                "selection_status": st or "status_missing",
            }

            if not hit.empty:
                h = hit.iloc[0]
                for c in [
                    "selected_value_krw",
                    "table_index",
                    "row_index",
                    "primary_row_label",
                    "selected_column",
                    "strict_score",
                    "consolidated_evidence_score",
                    "consolidated_evidence_reason",
                    "heading_context",
                ]:
                    if c in hit.columns:
                        rec[c] = h.get(c)

            missing_records.append(rec)

        if balance_qa == "fail":
            bucket = "hard_balance_fail_manual_audit"
        elif core_count == 6 and balance_qa in {"exact_pass", "rounding_pass"}:
            bucket = "adoptable_strict_cfs_complete"
        elif core_count == 6:
            bucket = "complete_but_balance_not_pass"
        else:
            if any(s in AMBIGUOUS for s in statuses_for_missing):
                bucket = "partial_or_zero_ambiguous_cfs"
            elif statuses_for_missing and all(s in NO_RELIABLE for s in statuses_for_missing):
                bucket = "partial_or_zero_no_reliable_cfs"
            else:
                bucket = "partial_or_zero_other_status"

        rec = row.to_dict()
        rec["missing_metric_count"] = len(missing)
        rec["missing_metrics_h6b3"] = " | ".join(missing)
        rec["missing_reason_detail"] = " | ".join(missing_details)
        rec["h6b3_triage_bucket"] = bucket
        triage_records.append(rec)

    triage = pd.DataFrame(triage_records)
    missing_df = pd.DataFrame(missing_records)

    adoptable = triage.loc[
        triage["h6b3_triage_bucket"].eq("adoptable_strict_cfs_complete")
    ].copy()

    hard_fail = triage.loc[
        triage["balance_qa"].eq("fail")
    ].copy()

    triage.to_parquet(OUT_TRIAGE, index=False)
    triage.to_csv(OUT_TRIAGE_CSV, index=False, encoding="utf-8-sig")
    adoptable.to_csv(OUT_ADOPTABLE, index=False, encoding="utf-8-sig")
    hard_fail.to_csv(OUT_HARD_FAIL, index=False, encoding="utf-8-sig")
    missing_df.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("05A5-H6B3 STRICT-CFS RECOVERY TRIAGE")
    print("=" * 110)

    print(f"\nReceipts: {len(triage):,}")

    print("\n[Core count x Balance QA]")
    print(
        pd.crosstab(
            triage["strict_cfs_core_count"],
            triage["balance_qa"],
            dropna=False,
        )
        .sort_index(ascending=False)
        .to_string()
    )

    print("\n[H6B3 triage bucket]")
    print(
        triage["h6b3_triage_bucket"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Missing-family C5 status]")
    if missing_df.empty:
        print("None")
    else:
        print(
            missing_df["selection_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\n[Missing-family status by metric]")
    if missing_df.empty:
        print("None")
    else:
        print(
            pd.crosstab(
                missing_df["missing_metric"],
                missing_df["selection_status"],
                dropna=False,
            ).to_string()
        )

    print("\n[Adoptable strict CFS complete]")
    print(f"{len(adoptable):,}")

    print("\n[Hard balance fail]")
    if hard_fail.empty:
        print("None")
    else:
        cols = [
            c for c in [
                "stock_code",
                "corp_name",
                "canonical_period_key",
                "rcept_no",
                "assets",
                "liabilities",
                "equity_total",
                "balance_gap",
                "balance_relative_gap",
                "strict_cfs_core_count",
                "missing_core_metrics",
            ]
            if c in hard_fail.columns
        ]
        print(hard_fail[cols].to_string(index=False))

    print("\n[Top incomplete combinations by triage]")
    incomplete = triage.loc[
        triage["strict_cfs_core_count"].lt(6)
    ]
    combo = (
        incomplete.groupby(
            ["h6b3_triage_bucket", "missing_metrics_h6b3"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .head(30)
    )
    print(combo.to_string(index=False))

    print(f"\nTriage    : {OUT_TRIAGE}")
    print(f"Adoptable : {OUT_ADOPTABLE}")
    print(f"Hard fail : {OUT_HARD_FAIL}")
    print(f"Missing C5: {OUT_MISSING}")

    print(
        "\n다음 판단:\n"
        "- adoptable_strict_cfs_complete -> 최종 merge 후보\n"
        "- hard_balance_fail_manual_audit -> same-receipt candidate-set audit\n"
        "- partial_or_zero_ambiguous_cfs -> CFS 존재/충돌, OFS 우회 금지\n"
        "- partial_or_zero_no_reliable_cfs -> 아직 CFS 부재 단정 금지; C4 evidence로 추가 분류"
    )


if __name__ == "__main__":
    main()
