from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H4B. Targeted Overlap / Balance Cross-Check
#
# 목적
# ------------------------------------------------------------
# H4A에서 남은:
# - overlap mismatch 10건
# - hard balance failure 4건
#
# 을 receipt 단위로 더 정밀하게 본다.
#
# 구성
# ------------------------------------------------------------
# A) overlap mismatch:
#    H1 major-account raw row와 H3 full-statement selected/raw row를 나란히 출력
#
# B) balance failure:
#    H3 full statement가 없던 receipt는 fnlttSinglAcntAll을 1회성으로 조회
#    (CFS -> no data일 때 OFS)
#    같은 receipt의 assets/liabilities/equity total을 뽑아 balance equation 비교
#
# 주의
# ------------------------------------------------------------
# - 이 스크립트는 QA용.
# - 자동 override는 하지 않는다.
# - receipt mismatch면 사용 금지.
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h4b_dart_targeted_crosscheck.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

OVERLAP = INTERIM / "dart_noncorrected_merge_overlap_qa.csv"
FINAL = INTERIM / "dart_noncorrected_pit_receipt_wide.parquet"
H1_ROWS = INTERIM / "dart_noncorrected_multi_account_rows.parquet"
H3_ROWS = INTERIM / "dart_noncorrected_full_fallback_rows.parquet"
H3_SELECTED = INTERIM / "dart_noncorrected_full_fallback_selected.parquet"
H3_MANIFEST = INTERIM / "dart_noncorrected_full_fallback_manifest.parquet"

OUT_OVERLAP = INTERIM / "dart_noncorrected_targeted_overlap_crosscheck.csv"
OUT_BALANCE = INTERIM / "dart_noncorrected_targeted_balance_crosscheck.csv"

API_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"


def receipt_str(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def parse_num(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "nan", "None", "<NA>"}:
        return np.nan
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return pd.to_numeric(s, errors="coerce")


def norm(v: Any) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", "", str(v)).lower()


def balance(a, l, e):
    if any(pd.isna(x) for x in [a, l, e]):
        return np.nan, np.nan
    gap = float(a) - float(l) - float(e)
    rel = abs(gap) / abs(float(a)) if float(a) != 0 else np.nan
    return gap, rel


def request_full(session, key, corp_code, year, reprt_code, fs_div):
    r = session.get(
        API_URL,
        params={
            "crtfc_key": key,
            "corp_code": str(corp_code).zfill(8),
            "bsns_year": str(int(year)),
            "reprt_code": str(reprt_code),
            "fs_div": fs_div,
        },
        timeout=90,
    )
    r.raise_for_status()
    payload = r.json()
    return payload


def select_balance_from_full(df: pd.DataFrame):
    if df.empty:
        return {}

    work = df.copy()
    work["aid"] = work["account_id"].map(norm)
    work["anm"] = work["account_nm"].map(norm)
    work["sj"] = work["sj_div"].astype(str).str.upper()

    specs = {
        "assets": (
            ["ifrs-full_assets", "ifrs_assets"],
            ["자산총계"],
        ),
        "liabilities": (
            ["ifrs-full_liabilities", "ifrs_liabilities"],
            ["부채총계"],
        ),
        "equity_total": (
            ["ifrs-full_equity", "ifrs_equity"],
            ["자본총계"],
        ),
    }

    out = {}

    for metric, (ids, names) in specs.items():
        cand = work.loc[
            work["sj"].eq("BS")
            & (
                work["aid"].isin(ids)
                | work["anm"].isin(names)
            )
        ].copy()

        if cand.empty:
            continue

        cand["_score"] = np.where(
            cand["aid"].isin(ids),
            100,
            80,
        )

        cand["_ord"] = pd.to_numeric(
            cand.get("ord"),
            errors="coerce",
        ).fillna(999999)

        cand = cand.sort_values(
            ["_score", "_ord"],
            ascending=[False, True],
        )

        row = cand.iloc[0]

        out[metric] = {
            "value": parse_num(row.get("thstrm_amount")),
            "account_id": row.get("account_id"),
            "account_nm": row.get("account_nm"),
            "ord": row.get("ord"),
        }

    return out


def main():

    for p in [OVERLAP, FINAL, H1_ROWS, H3_ROWS, H3_SELECTED, H3_MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(p)

    overlap = pd.read_csv(
        OVERLAP,
        dtype={"rcept_no": str, "stock_code": str},
        low_memory=False,
    )
    final = pd.read_parquet(FINAL)
    h1 = pd.read_parquet(H1_ROWS)
    h3r = pd.read_parquet(H3_ROWS)
    h3s = pd.read_parquet(H3_SELECTED)
    h3m = pd.read_parquet(H3_MANIFEST)

    overlap["rcept_no"] = receipt_str(overlap["rcept_no"])
    final["rcept_no"] = receipt_str(final["rcept_no"])
    h1["rcept_no"] = receipt_str(h1["rcept_no"])
    h3r["_expected_rcept_no"] = receipt_str(h3r["_expected_rcept_no"])
    h3s["_expected_rcept_no"] = receipt_str(h3s["_expected_rcept_no"])
    h3m["rcept_no"] = receipt_str(h3m["rcept_no"])

    print("\n" + "=" * 100)
    print("05A5-H4B TARGETED OVERLAP / BALANCE CROSS-CHECK")
    print("=" * 100)

    # --------------------------------------------------------
    # A) overlap mismatch
    # --------------------------------------------------------
    mm = overlap.loc[
        overlap["overlap_match"].astype(str).str.lower().isin(["false", "0"])
    ].copy()

    overlap_records = []

    for _, row in mm.iterrows():
        receipt = str(row["rcept_no"])
        metric = str(row["metric"])

        h1rec = h1.loc[h1["rcept_no"].eq(receipt)].copy()
        h3sel = h3s.loc[
            h3s["_expected_rcept_no"].eq(receipt)
            & h3s["metric"].eq(metric)
        ].copy()
        h3raw = h3r.loc[h3r["_expected_rcept_no"].eq(receipt)].copy()

        if metric == "operating_income_cumulative":
            h1cand = h1rec.loc[
                h1rec["account_nm"].astype("string").fillna("").map(norm)
                .isin(["영업이익", "영업이익(손실)", "영업손익"])
            ].copy()

            h3cand = h3raw.loc[
                h3raw["sj_div"].astype(str).isin(["IS", "CIS"])
                & (
                    h3raw["account_nm"].astype("string").fillna("").map(norm)
                    .str.contains("영업이익|영업손익", regex=True)
                    |
                    h3raw["account_id"].astype("string").fillna("").map(norm)
                    .str.contains("operatingincomeloss", regex=False)
                )
            ].copy()

        elif metric in {"assets", "liabilities"}:
            target_name = "자산총계" if metric == "assets" else "부채총계"
            id_token = "assets" if metric == "assets" else "liabilities"

            h1cand = h1rec.loc[
                h1rec["account_nm"].astype("string").fillna("").map(norm)
                .eq(target_name)
            ].copy()

            h3cand = h3raw.loc[
                h3raw["sj_div"].astype(str).eq("BS")
                & (
                    h3raw["account_nm"].astype("string").fillna("").map(norm)
                    .eq(target_name)
                    |
                    h3raw["account_id"].astype("string").fillna("").map(norm)
                    .str.contains(id_token, regex=False)
                )
            ].copy()
        else:
            h1cand = h1rec.head(0)
            h3cand = h3raw.head(0)

        overlap_records.append({
            "stock_code": row.get("stock_code"),
            "corp_name": row.get("corp_name"),
            "canonical_period_key": row.get("canonical_period_key"),
            "rcept_no": receipt,
            "metric": metric,
            "h2_major_value": row.get("h2_major_value"),
            "h3_full_value": row.get("h3_full_value"),
            "h1_candidates": " || ".join(
                f"{x.get('fs_div')} | {x.get('account_nm')} | "
                f"th={x.get('thstrm_amount')} | add={x.get('thstrm_add_amount')} | ord={x.get('ord')}"
                for x in h1cand.to_dict("records")
            ),
            "h3_selected": " || ".join(
                f"{x.get('sj_div')} | {x.get('account_id')} | {x.get('account_nm')} | "
                f"th={x.get('thstrm_amount')} | add={x.get('thstrm_add_amount')} | ord={x.get('ord')}"
                for x in h3sel.to_dict("records")
            ),
            "h3_candidates": " || ".join(
                f"{x.get('sj_div')} | {x.get('account_id')} | {x.get('account_nm')} | "
                f"th={x.get('thstrm_amount')} | add={x.get('thstrm_add_amount')} | ord={x.get('ord')}"
                for x in h3cand.head(20).to_dict("records")
            ),
        })

    overlap_out = pd.DataFrame(overlap_records)
    overlap_out.to_csv(
        OUT_OVERLAP,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[Overlap mismatch detail]")
    if overlap_out.empty:
        print("0")
    else:
        print(overlap_out.to_string(index=False))

    # --------------------------------------------------------
    # B) balance failure targeted full API
    # --------------------------------------------------------
    fail = final.loc[
        final["balance_qa_class"].eq("fail")
    ].copy()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise RuntimeError(".env의 DART_API_KEY가 없습니다.")

    session = requests.Session()
    balance_records = []

    for _, row in fail.iterrows():
        receipt = str(row["rcept_no"])

        existing_sel = h3s.loc[
            h3s["_expected_rcept_no"].eq(receipt)
            & h3s["metric"].isin(
                ["assets", "liabilities", "equity_total"]
            )
        ].copy()

        source = None
        fs_div = None
        selected = {}

        if not existing_sel.empty:
            source = "existing_h3"
            fs_div = existing_sel["_requested_fs_div"].dropna().astype(str).iloc[0]

            for metric in ["assets", "liabilities", "equity_total"]:
                hit = existing_sel.loc[existing_sel["metric"].eq(metric)]
                if not hit.empty:
                    x = hit.iloc[0]
                    selected[metric] = {
                        "value": float(x["metric_value"]),
                        "account_id": x.get("account_id"),
                        "account_nm": x.get("account_nm"),
                        "ord": x.get("ord"),
                    }

        else:
            source = "targeted_full_api"

            attempts = ["CFS", "OFS"]

            for fs in attempts:
                payload = request_full(
                    session,
                    api_key,
                    row["corp_code"],
                    row["bsns_year"],
                    row["reprt_code"],
                    fs,
                )

                status = str(payload.get("status", ""))

                if status == "000":
                    raw = pd.DataFrame(payload.get("list", []))
                    if raw.empty:
                        continue

                    raw["rcept_no"] = receipt_str(raw["rcept_no"])

                    matched = raw.loc[
                        raw["rcept_no"].eq(receipt)
                    ].copy()

                    if matched.empty:
                        source = "receipt_mismatch"
                        fs_div = fs
                        break

                    selected = select_balance_from_full(matched)
                    fs_div = fs
                    break

                if status in {"013", "014"}:
                    continue

                source = f"api_status_{status}"
                fs_div = fs
                break

                time.sleep(0.15)

        a = selected.get("assets", {}).get("value", np.nan)
        l = selected.get("liabilities", {}).get("value", np.nan)
        e = selected.get("equity_total", {}).get("value", np.nan)

        gap, rel = balance(a, l, e)

        balance_records.append({
            "stock_code": row["stock_code"],
            "corp_name": row.get("corp_name"),
            "canonical_period_key": row["canonical_period_key"],
            "rcept_no": receipt,

            "final_assets": row["assets"],
            "final_liabilities": row["liabilities"],
            "final_equity": row["equity_total"],
            "final_gap": row["balance_gap"],
            "final_rel_gap": row["balance_relative_gap"],

            "crosscheck_source": source,
            "crosscheck_fs": fs_div,
            "cross_assets": a,
            "cross_liabilities": l,
            "cross_equity": e,
            "cross_gap": gap,
            "cross_rel_gap": rel,

            "assets_account": selected.get("assets", {}).get("account_nm"),
            "liabilities_account": selected.get("liabilities", {}).get("account_nm"),
            "equity_account": selected.get("equity_total", {}).get("account_nm"),
        })

    balance_out = pd.DataFrame(balance_records)
    balance_out.to_csv(
        OUT_BALANCE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[Targeted balance cross-check]")
    if balance_out.empty:
        print("0")
    else:
        print(balance_out.to_string(index=False))

    print(f"\nOverlap: {OUT_OVERLAP}")
    print(f"Balance: {OUT_BALANCE}")

    print(
        "\n다음 판단:"
        "\n- overlap mismatch는 raw row를 보고 H2/H3 selector 중 잘못된 쪽을 수정"
        "\n- balance cross-check가 PASS면 coherent full-statement 3종 사용 후보"
        "\n- cross-check도 FAIL이면 same-receipt XBRL/document source-level 검증"
    )


if __name__ == "__main__":
    main()
