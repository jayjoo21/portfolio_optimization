from __future__ import annotations

import importlib.util
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H5A. Non-Corrected Hard-Balance Same-Receipt Document Audit
#
# 목적
# ------------------------------------------------------------
# H4/H4B에서 full financial API까지 동일하게 balance equation이
# 깨졌던 non-corrected receipt 4건을 "같은 rcept_no"의 document.xml로
# 교차검증한다.
#
# strict PIT:
# - 미래 보고서 사용 금지
# - 동일 receipt의 document만 사용
# - 이 단계는 audit. 자동 production override는 아직 하지 않음.
#
# 입력
# ------------------------------------------------------------
# dart_noncorrected_pit_receipt_wide_reconciled.parquet
# 없으면 dart_noncorrected_pit_receipt_wide.parquet
#
# 출력
# ------------------------------------------------------------
# dart_noncorrected_balance_document_crosscheck_selected.csv
# dart_noncorrected_balance_document_crosscheck_summary.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h5a_dart_noncorrected_balance_document_crosscheck.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

RECONCILED = INTERIM / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
ORIGINAL = INTERIM / "dart_noncorrected_pit_receipt_wide.parquet"

QA_RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "noncorrected_balance_qa"
)

OUT_SELECTED = (
    INTERIM
    / "dart_noncorrected_balance_document_crosscheck_selected.csv"
)

OUT_SUMMARY = (
    INTERIM
    / "dart_noncorrected_balance_document_crosscheck_summary.csv"
)

DART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

ABS_TOL = 2_000_000
REL_TOL = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module load 실패: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(v: str) -> str:
    text = str(v).strip()
    for old, new in [
        ("/", "_"), ("\\", "_"), (":", "_"), ("*", "_"),
        ("?", "_"), ('"', "_"), ("<", "_"), (">", "_"),
        ("|", "_"), (" ", "_"),
    ]:
        text = text.replace(old, new)
    return text


def download_document(api_key: str, rcept_no: str) -> bytes:
    response = requests.get(
        DART_DOCUMENT_URL,
        params={
            "crtfc_key": api_key,
            "rcept_no": rcept_no,
        },
        timeout=90,
    )
    response.raise_for_status()

    content = response.content

    if content[:2] != b"PK":
        preview = content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{rcept_no}: document.xml ZIP 아님\n{preview}"
        )

    return content


def qa(a, l, e):
    if any(pd.isna(x) for x in [a, l, e]):
        return "missing", np.nan, np.nan

    gap = float(a) - float(l) - float(e)
    rel = abs(gap) / abs(float(a)) if float(a) != 0 else np.nan

    if gap == 0:
        status = "exact_pass"
    elif abs(gap) <= ABS_TOL or (pd.notna(rel) and rel <= REL_TOL):
        status = "rounding_pass"
    else:
        status = "fail"

    return status, gap, rel


def main():

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DART_API_KEY")

    if not api_key:
        raise RuntimeError(".env의 DART_API_KEY가 없습니다.")

    source_path = RECONCILED if RECONCILED.exists() else ORIGINAL

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    wide = pd.read_parquet(source_path)

    wide["rcept_no"] = (
        wide["rcept_no"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
    )

    targets = wide.loc[
        wide["balance_qa_class"].eq("fail")
    ].copy()

    print("\n" + "=" * 100)
    print("05A5-H5A NON-CORRECTED HARD-BALANCE DOCUMENT CROSS-CHECK")
    print("=" * 100)

    print(f"\nHard-balance targets: {len(targets):,}")

    if targets.empty:
        print("대상 없음")
        return

    c1 = load_module(
        "dart_h5a_c1",
        SCRIPTS_DIR / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )
    c4 = load_module(
        "dart_h5a_c4",
        SCRIPTS_DIR / "05a5c4_dart_conservative_document_selector.py",
    )
    c5 = load_module(
        "dart_h5a_c5",
        SCRIPTS_DIR / "05a5c5_dart_strict_consolidated_document_selector.py",
    )

    c1.RAW_ROOT = QA_RAW_ROOT

    all_selected = []
    summaries = []

    for idx, (_, row) in enumerate(targets.iterrows(), start=1):

        stock = str(row["stock_code"]).zfill(6)
        receipt = str(row["rcept_no"])
        period = str(row["canonical_period_key"])

        rcept_dt = pd.to_datetime(
            row.get("rcept_dt"),
            errors="coerce",
        )

        if pd.isna(rcept_dt):
            # QA filename only; availability is NOT inferred from this.
            rcept_dt = pd.to_datetime(
                receipt[:8],
                format="%Y%m%d",
                errors="coerce",
            )

        period_dir = QA_RAW_ROOT / stock / safe_name(period)
        period_dir.mkdir(parents=True, exist_ok=True)

        date_text = (
            f"{rcept_dt:%Y%m%d}"
            if pd.notna(rcept_dt)
            else receipt[:8]
        )

        zip_path = (
            period_dir
            / f"{date_text}_seq01_{receipt}_document.zip"
        )

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{stock} | {period} | {receipt}"
        )

        try:
            if not zip_path.exists():
                content = download_document(api_key, receipt)
                zip_path.write_bytes(content)
                print(f"  document saved: {zip_path}")
            else:
                print(f"  existing document: {zip_path}")

            candidates = pd.DataFrame(
                c1.extract_document_candidates(zip_path)
            )

            if candidates.empty:
                summaries.append(
                    {
                        "stock_code": stock,
                        "corp_name": row.get("corp_name"),
                        "canonical_period_key": period,
                        "rcept_no": receipt,
                        "api_assets": row.get("assets"),
                        "api_liabilities": row.get("liabilities"),
                        "api_equity": row.get("equity_total"),
                        "api_gap": row.get("balance_gap"),
                        "api_rel_gap": row.get("balance_relative_gap"),
                        "document_status": "no_candidates",
                        "document_assets": np.nan,
                        "document_liabilities": np.nan,
                        "document_equity": np.nan,
                        "document_gap": np.nan,
                        "document_rel_gap": np.nan,
                        "document_qa": "missing",
                    }
                )
                continue

            c4_candidates = c4.build_account_candidates(candidates)
            selected = c5.select_values(c4_candidates)

            selected = selected.copy()
            selected["target_stock_code"] = stock
            selected["target_period_key"] = period
            selected["target_rcept_no"] = receipt

            all_selected.append(selected)

            vals = {}

            for family, metric_name in [
                ("assets", "assets"),
                ("liabilities", "liabilities"),
                ("equity", "equity"),
            ]:
                hit = selected.loc[
                    selected["account_family"].eq(family)
                    & selected["selection_status"].isin(
                        ["selected", "selected_consensus"]
                    )
                ]

                if not hit.empty:
                    vals[metric_name] = float(
                        hit.iloc[0]["selected_value_krw"]
                    )

            doc_qa, gap, rel = qa(
                vals.get("assets"),
                vals.get("liabilities"),
                vals.get("equity"),
            )

            summaries.append(
                {
                    "stock_code": stock,
                    "corp_name": row.get("corp_name"),
                    "canonical_period_key": period,
                    "rcept_no": receipt,

                    "api_assets": row.get("assets"),
                    "api_liabilities": row.get("liabilities"),
                    "api_equity": row.get("equity_total"),
                    "api_gap": row.get("balance_gap"),
                    "api_rel_gap": row.get("balance_relative_gap"),

                    "document_status": "parsed",
                    "document_assets": vals.get("assets"),
                    "document_liabilities": vals.get("liabilities"),
                    "document_equity": vals.get("equity"),
                    "document_gap": gap,
                    "document_rel_gap": rel,
                    "document_qa": doc_qa,
                }
            )

            print(
                "  document balance: "
                f"{doc_qa} | gap={gap:,.0f} | "
                f"rel={rel * 100:.9f}%"
                if pd.notna(rel)
                else f"  document balance: {doc_qa}"
            )

        except Exception as exc:
            summaries.append(
                {
                    "stock_code": stock,
                    "corp_name": row.get("corp_name"),
                    "canonical_period_key": period,
                    "rcept_no": receipt,
                    "api_assets": row.get("assets"),
                    "api_liabilities": row.get("liabilities"),
                    "api_equity": row.get("equity_total"),
                    "api_gap": row.get("balance_gap"),
                    "api_rel_gap": row.get("balance_relative_gap"),
                    "document_status": "error",
                    "document_assets": np.nan,
                    "document_liabilities": np.nan,
                    "document_equity": np.nan,
                    "document_gap": np.nan,
                    "document_rel_gap": np.nan,
                    "document_qa": "missing",
                    "error": repr(exc),
                }
            )

            print(f"  ERROR: {repr(exc)}")

    selected_all = (
        pd.concat(all_selected, ignore_index=True)
        if all_selected
        else pd.DataFrame()
    )

    summary = pd.DataFrame(summaries)

    selected_all.to_csv(
        OUT_SELECTED,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("H5A SUMMARY")
    print("=" * 100)

    print("\n[Document QA]")
    print(
        summary["document_qa"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Summary]")
    show = [
        "stock_code",
        "corp_name",
        "canonical_period_key",
        "rcept_no",
        "api_gap",
        "api_rel_gap",
        "document_assets",
        "document_liabilities",
        "document_equity",
        "document_gap",
        "document_rel_gap",
        "document_qa",
        "document_status",
    ]
    show = [c for c in show if c in summary.columns]
    print(summary[show].to_string(index=False))

    print(f"\nSelected: {OUT_SELECTED}")
    print(f"Summary : {OUT_SUMMARY}")

    print(
        "\n다음 판단:"
        "\n- document PASS → same-receipt document coherent balance override 후보"
        "\n- document도 FAIL → 원 공시 자체/태깅 구조 이슈; 해당 balance set missing 후보"
        "\n- document 선택 불가 → missing 처리 또는 receipt-specific XBRL 추가 audit"
    )


if __name__ == "__main__":
    main()
