from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H4C. Non-Corrected Overlap Reconciliation
#
# 목적
# ------------------------------------------------------------
# H4/H4B에서 원인이 확인된 overlap mismatch를 보수적으로 정리한다.
#
# 규칙
# ------------------------------------------------------------
# A) operating_income_cumulative mismatch
#    - H2 major 값과 동일한 H3 raw row가 "영업수익"이고
#    - H3 selected row가 "영업이익"이면
#      H2 major API가 영업수익을 영업이익으로 잘못 매핑한 것으로 보고
#      H3 full-statement 영업이익으로 override.
#
# B) assets/liabilities/equity mismatch
#    - H2 balance set이 QA PASS하고
#    - H3 balance set이 FAIL이면
#      H2 값을 유지.
#
# 자동으로 애매한 mismatch를 고치지 않는다.
# 판정할 수 없는 mismatch가 남으면 RuntimeError.
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_pit_receipt_wide_reconciled.parquet/csv
# dart_noncorrected_overlap_reconciliation_log.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h4c_dart_overlap_reconciliation.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

FINAL_IN = INTERIM / "dart_noncorrected_pit_receipt_wide.parquet"
H2_IN = INTERIM / "dart_noncorrected_major_account_coverage.parquet"
H3_SELECTED_IN = INTERIM / "dart_noncorrected_full_fallback_selected.parquet"
H3_ROWS_IN = INTERIM / "dart_noncorrected_full_fallback_rows.parquet"
OVERLAP_IN = INTERIM / "dart_noncorrected_merge_overlap_qa.csv"

FINAL_OUT = INTERIM / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
FINAL_OUT_CSV = INTERIM / "dart_noncorrected_pit_receipt_wide_reconciled.csv"
LOG_OUT = INTERIM / "dart_noncorrected_overlap_reconciliation_log.csv"

ABS_BALANCE_TOL = 2_000_000
REL_BALANCE_TOL = 1e-6


def rcept_str(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def parse_num(v):
    if pd.isna(v):
        return np.nan
    text = str(v).strip().replace(",", "")
    if text in {"", "-", "nan", "None", "<NA>"}:
        return np.nan
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return pd.to_numeric(text, errors="coerce")


def norm(v) -> str:
    if pd.isna(v):
        return ""
    return "".join(str(v).split())


def balance_qa(a, l, e):
    if any(pd.isna(x) for x in [a, l, e]):
        return "missing", np.nan, np.nan

    gap = float(a) - float(l) - float(e)
    rel = abs(gap) / abs(float(a)) if float(a) != 0 else np.nan

    if gap == 0:
        qa = "exact_pass"
    elif (
        abs(gap) <= ABS_BALANCE_TOL
        or (pd.notna(rel) and rel <= REL_BALANCE_TOL)
    ):
        qa = "rounding_pass"
    else:
        qa = "fail"

    return qa, gap, rel


def close_value(a, b):
    if pd.isna(a) or pd.isna(b):
        return False
    abs_diff = abs(float(a) - float(b))
    denom = max(abs(float(a)), abs(float(b)), 1.0)
    return abs_diff <= 1 or abs_diff / denom <= 1e-9


def main():

    for p in [FINAL_IN, H2_IN, H3_SELECTED_IN, H3_ROWS_IN, OVERLAP_IN]:
        if not p.exists():
            raise FileNotFoundError(p)

    final = pd.read_parquet(FINAL_IN)
    h2 = pd.read_parquet(H2_IN)
    h3s = pd.read_parquet(H3_SELECTED_IN)
    h3r = pd.read_parquet(H3_ROWS_IN)
    overlap = pd.read_csv(
        OVERLAP_IN,
        dtype={"rcept_no": str, "stock_code": str},
        low_memory=False,
    )

    final["rcept_no"] = rcept_str(final["rcept_no"])
    h2["rcept_no"] = rcept_str(h2["rcept_no"])
    h3s["_expected_rcept_no"] = rcept_str(h3s["_expected_rcept_no"])
    h3r["_expected_rcept_no"] = rcept_str(h3r["_expected_rcept_no"])
    overlap["rcept_no"] = rcept_str(overlap["rcept_no"])

    mismatches = overlap.loc[
        overlap["overlap_match"].astype(str).str.lower().isin(["false", "0"])
    ].copy()

    out = final.copy()
    log_rows = []
    unresolved = []

    for _, mm in mismatches.iterrows():

        receipt = str(mm["rcept_no"])
        metric = str(mm["metric"])

        out_mask = out["rcept_no"].eq(receipt)
        if not out_mask.any():
            unresolved.append((receipt, metric, "receipt_not_in_final"))
            continue

        # ----------------------------------------------------
        # Case A: Operating income
        # ----------------------------------------------------
        if metric == "operating_income_cumulative":

            selected = h3s.loc[
                h3s["_expected_rcept_no"].eq(receipt)
                & h3s["metric"].eq(metric)
            ].copy()

            raw = h3r.loc[
                h3r["_expected_rcept_no"].eq(receipt)
                & h3r["sj_div"].astype(str).isin(["IS", "CIS"])
            ].copy()

            if selected.empty:
                unresolved.append((receipt, metric, "h3_selected_missing"))
                continue

            selected_row = selected.iloc[0]
            selected_name = norm(selected_row.get("account_nm"))
            h3_value = float(selected_row["metric_value"])
            h2_value = float(mm["h2_major_value"])

            h2_match_raw = raw.loc[
                raw.apply(
                    lambda r: close_value(
                        parse_num(r.get("thstrm_add_amount"))
                        if pd.notna(parse_num(r.get("thstrm_add_amount")))
                        else parse_num(r.get("thstrm_amount")),
                        h2_value,
                    ),
                    axis=1,
                )
            ].copy()

            h2_looks_like_revenue = False

            if not h2_match_raw.empty:
                h2_looks_like_revenue = any(
                    "영업수익" in norm(x)
                    for x in h2_match_raw["account_nm"].tolist()
                )

            h3_is_operating_income = (
                "영업이익" in selected_name
                or "영업손익" in selected_name
            )

            if h2_looks_like_revenue and h3_is_operating_income:

                before = out.loc[out_mask, metric].iloc[0]

                out.loc[out_mask, metric] = h3_value
                out.loc[out_mask, f"{metric}_source"] = (
                    "full_statement_qa_override"
                )

                log_rows.append(
                    {
                        "rcept_no": receipt,
                        "stock_code": mm.get("stock_code"),
                        "corp_name": mm.get("corp_name"),
                        "canonical_period_key": mm.get("canonical_period_key"),
                        "metric": metric,
                        "action": "override_h2_with_h3",
                        "before_value": before,
                        "after_value": h3_value,
                        "reason": (
                            "H2 major value matches H3 raw 영업수익; "
                            "H3 selected row is explicit 영업이익"
                        ),
                    }
                )

            else:
                unresolved.append(
                    (
                        receipt,
                        metric,
                        "operating_income_mismatch_not_proven",
                    )
                )

            continue

        # ----------------------------------------------------
        # Case B: Balance metrics
        # ----------------------------------------------------
        if metric in {"assets", "liabilities", "equity_total"}:

            h2row = h2.loc[h2["rcept_no"].eq(receipt)]
            if h2row.empty:
                unresolved.append((receipt, metric, "h2_receipt_missing"))
                continue
            h2row = h2row.iloc[0]

            h3rows = h3s.loc[
                h3s["_expected_rcept_no"].eq(receipt)
                & h3s["metric"].isin(
                    ["assets", "liabilities", "equity_total"]
                )
            ].copy()

            h3map = (
                h3rows.set_index("metric")["metric_value"].to_dict()
                if not h3rows.empty
                else {}
            )

            h2qa, h2gap, h2rel = balance_qa(
                h2row.get("assets"),
                h2row.get("liabilities"),
                h2row.get("equity_total"),
            )

            h3qa, h3gap, h3rel = balance_qa(
                h3map.get("assets"),
                h3map.get("liabilities"),
                h3map.get("equity_total"),
            )

            if h2qa in {"exact_pass", "rounding_pass"} and h3qa == "fail":

                # H4 already keeps H2 when available, so no value change.
                log_rows.append(
                    {
                        "rcept_no": receipt,
                        "stock_code": mm.get("stock_code"),
                        "corp_name": mm.get("corp_name"),
                        "canonical_period_key": mm.get("canonical_period_key"),
                        "metric": metric,
                        "action": "keep_h2",
                        "before_value": out.loc[out_mask, metric].iloc[0],
                        "after_value": out.loc[out_mask, metric].iloc[0],
                        "reason": (
                            f"H2 coherent balance={h2qa} "
                            f"(gap={h2gap}); H3 balance={h3qa} "
                            f"(gap={h3gap})"
                        ),
                    }
                )

            else:
                unresolved.append(
                    (
                        receipt,
                        metric,
                        f"balance_not_decisive_h2={h2qa}_h3={h3qa}",
                    )
                )

            continue

        unresolved.append((receipt, metric, "unsupported_metric"))

    if unresolved:
        detail = "\n".join(
            f"{r} | {m} | {reason}"
            for r, m, reason in unresolved
        )
        raise RuntimeError(
            "자동 reconciliation 할 수 없는 overlap mismatch가 남았습니다:\n"
            + detail
        )

    # Recompute balance fields because final artifact should be internally current.
    out["balance_gap"] = (
        out["assets"]
        - out["liabilities"]
        - out["equity_total"]
    )

    out["balance_relative_gap"] = (
        out["balance_gap"].abs()
        / out["assets"].abs().replace(0, np.nan)
    )

    def classify_balance(row):
        qa, _, _ = balance_qa(
            row.get("assets"),
            row.get("liabilities"),
            row.get("equity_total"),
        )
        return qa

    out["balance_qa_class"] = out.apply(
        classify_balance,
        axis=1,
    )

    out["balance_equation_pass"] = out["balance_qa_class"].isin(
        ["exact_pass", "rounding_pass"]
    )

    log = pd.DataFrame(log_rows)

    out.to_parquet(FINAL_OUT, index=False)
    out.to_csv(FINAL_OUT_CSV, index=False, encoding="utf-8-sig")
    log.to_csv(LOG_OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("05A5-H4C OVERLAP RECONCILIATION")
    print("=" * 90)

    print("\n[Actions]")
    print(
        log["action"].value_counts(dropna=False).to_string()
        if not log.empty
        else "None"
    )

    print("\n[By metric]")
    print(
        log.groupby(["metric", "action"]).size().to_string()
        if not log.empty
        else "None"
    )

    print("\n[Balance QA after reconciliation]")
    print(
        out["balance_qa_class"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nFinal reconciled: {FINAL_OUT}")
    print(f"Log             : {LOG_OUT}")


if __name__ == "__main__":
    main()
