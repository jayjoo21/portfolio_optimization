
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

PARSE = INTERIM / "dart_corrected_pit_parse_manifest.parquet"
VALUES = INTERIM / "dart_corrected_pit_receipt_values.parquet"
WIDE = INTERIM / "dart_corrected_pit_receipt_wide.parquet"

OUT_SUMMARY = INTERIM / "dart_balance_failure_audit.csv"
OUT_DETAIL = INTERIM / "dart_balance_failure_detail.csv"

BALANCE_METRICS = ["assets", "liabilities", "equity_total"]


def load():
    for p in [PARSE, VALUES, WIDE]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = pd.read_parquet(PARSE)
    vals = pd.read_parquet(VALUES)
    wide = pd.read_parquet(WIDE)

    for df in [pm, vals, wide]:
        if "rcept_no" in df:
            df["rcept_no"] = df["rcept_no"].astype(str)
        if "stock_code" in df:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

    return pm, vals, wide


def reason(row):
    out = []
    gap = row["balance_relative_gap"]

    if pd.notna(gap):
        if gap > 0.5:
            out.append("very_large_gap")
        elif gap > 0.05:
            out.append("large_gap")
        else:
            out.append("small_gap")

    if row.get("unique_balance_context_count", 0) > 1:
        out.append("mixed_balance_contexts")

    units = str(row.get("selected_units", ""))
    unit_set = {x.strip() for x in units.split("|") if x.strip()}
    if len(unit_set) > 1:
        out.append("mixed_units")

    vals = [row.get("assets"), row.get("liabilities"), row.get("equity_total")]
    vals = [abs(float(x)) for x in vals if pd.notna(x) and float(x) != 0]
    if vals and max(vals) / min(vals) > 1e5:
        out.append("possible_unit_scale_mismatch")

    if str(row.get("source_type")) == "document_fallback":
        out.append("document_fallback_review")

    return " | ".join(out or ["manual_review_needed"])


def main():
    pm, vals, wide = load()

    failures = pm.loc[
        pm["balance_equation_pass"].eq(False),
        ["stock_code", "period_key", "rcept_no", "rcept_dt", "source_type"]
    ].copy()

    print("\n" + "=" * 80)
    print("05A5-F BALANCE EQUATION FAILURE AUDIT")
    print("=" * 80)
    print(f"\nBalance failures: {len(failures):,}")

    if failures.empty:
        return

    base = failures.merge(
        wide,
        on=[c for c in ["stock_code", "period_key", "rcept_no", "rcept_dt", "source_type"]
            if c in failures.columns and c in wide.columns],
        how="left"
    )

    bvals = vals.loc[
        vals["rcept_no"].isin(failures["rcept_no"])
        & vals["metric"].isin(BALANCE_METRICS)
        & vals["selection_status"].isin(["selected", "selected_consensus"])
    ].copy()

    agg = bvals.groupby("rcept_no").agg(
        selected_units=("unit", lambda s: " | ".join(sorted({str(x) for x in s.dropna()}))),
        unique_balance_context_count=("context_or_table", "nunique"),
        selected_balance_contexts=("context_or_table", lambda s: " | ".join(sorted({str(x) for x in s.dropna()}))),
        selected_balance_concepts=("concept_or_label", lambda s: " | ".join(sorted({str(x) for x in s.dropna()}))),
    ).reset_index()

    base = base.merge(agg, on="rcept_no", how="left")

    base["balance_gap"] = (
        base["assets"] - base["liabilities"] - base["equity_total"]
    )
    base["balance_relative_gap"] = (
        base["balance_gap"].abs()
        / base["assets"].abs().replace(0, np.nan)
    )
    base["gap_pct_assets"] = base["balance_relative_gap"] * 100
    base["suspect_reason"] = base.apply(reason, axis=1)

    keep = [
        "stock_code", "corp_name", "period_key", "report_nm", "rcept_no", "rcept_dt",
        "source_type", "assets", "liabilities", "equity_total",
        "balance_gap", "balance_relative_gap", "gap_pct_assets",
        "selected_units", "unique_balance_context_count",
        "selected_balance_contexts", "selected_balance_concepts",
        "suspect_reason"
    ]
    keep = [c for c in keep if c in base.columns]

    summary = base[keep].sort_values("balance_relative_gap", ascending=False)
    detail = vals.loc[
        vals["rcept_no"].isin(summary["rcept_no"])
        & vals["metric"].isin(BALANCE_METRICS)
    ].copy()

    dcols = [
        "stock_code", "corp_name", "period_key", "report_nm", "rcept_no", "rcept_dt",
        "source_type", "metric", "value", "unit", "selection_status",
        "concept_or_label", "context_or_table", "period_start", "period_end",
        "duration_days", "selection_detail"
    ]
    dcols = [c for c in dcols if c in detail.columns]
    detail = detail[dcols].sort_values(
        ["stock_code", "period_key", "rcept_dt", "rcept_no", "metric"]
    )

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    print("\n[Failure summary]")
    print(summary.to_string(index=False))

    print("\n[Source type]")
    print(summary["source_type"].value_counts(dropna=False).to_string())

    print("\n[Gap statistics]")
    print(summary["gap_pct_assets"].describe().to_string())

    print("\n[Suspect reason]")
    print(summary["suspect_reason"].value_counts(dropna=False).to_string())

    print("\n[Selected balance detail]")
    print(detail.to_string(index=False))

    print(f"\nSummary: {OUT_SUMMARY}")
    print(f"Detail : {OUT_DETAIL}")


if __name__ == "__main__":
    main()
