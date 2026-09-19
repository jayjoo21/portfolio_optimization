from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

RECEIPT_SUMMARY = INTERIM / "dart_noncorrected_nodata_c4_receipt_evidence_summary.csv"
FAMILY_EVIDENCE = INTERIM / "dart_noncorrected_nodata_c4_missing_family_evidence.csv"
SOURCE_MANIFEST = INTERIM / "dart_noncorrected_nodata_source_manifest.parquet"
HARD_FAIL = INTERIM / "dart_noncorrected_nodata_document_hard_balance_fail.csv"
OUT = INTERIM / "dart_noncorrected_h6b5_basis_context_diagnostic.csv"


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module load failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rcept_str(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def pick_receipts(df: pd.DataFrame, bucket: str, n: int) -> list[str]:
    sub = df.loc[df["c4_receipt_evidence"].eq(bucket)].copy()
    if sub.empty:
        return []
    return (
        sub.sort_values(["stock_code", "canonical_period_key", "rcept_no"])
        ["rcept_no"]
        .astype(str)
        .head(n)
        .tolist()
    )


def main():
    for p in [RECEIPT_SUMMARY, FAMILY_EVIDENCE, SOURCE_MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(p)

    c1 = load_module(
        "h6b5_c1",
        SCRIPTS_DIR / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )
    c4 = load_module(
        "h6b5_c4",
        SCRIPTS_DIR / "05a5c4_dart_conservative_document_selector.py",
    )

    receipt_summary = pd.read_csv(
        RECEIPT_SUMMARY,
        dtype={"rcept_no": str, "stock_code": str},
        low_memory=False,
    )
    family_evidence = pd.read_csv(
        FAMILY_EVIDENCE,
        dtype={"rcept_no": str, "stock_code": str},
        low_memory=False,
    )
    manifest = pd.read_parquet(SOURCE_MANIFEST)

    for df in [receipt_summary, family_evidence, manifest]:
        df["rcept_no"] = rcept_str(df["rcept_no"])

    targets = []
    for bucket, n in [
        ("basis_uncertain_in_missing_families", 5),
        ("cfs_candidate_exists_selector_rejected_or_partial", 5),
        ("candidate_rows_but_no_current_value", 3),
        ("current_column_ambiguity", 3),
    ]:
        targets.extend(pick_receipts(receipt_summary, bucket, n))

    if HARD_FAIL.exists():
        hard = pd.read_csv(
            HARD_FAIL,
            dtype={"rcept_no": str, "stock_code": str},
            low_memory=False,
        )
        hard["rcept_no"] = rcept_str(hard["rcept_no"])
        targets.extend(hard["rcept_no"].astype(str).tolist())

    targets = list(dict.fromkeys(targets))

    path_map = (
        manifest[["rcept_no", "selected_path"]]
        .drop_duplicates("rcept_no")
        .set_index("rcept_no")["selected_path"]
        .to_dict()
    )

    meta_map = {
        str(r["rcept_no"]): r
        for _, r in receipt_summary.iterrows()
    }

    print("\n" + "=" * 120)
    print("05A5-H6B5 C4 BASIS CONTEXT DIAGNOSTIC")
    print("=" * 120)
    print(f"\nSample receipts: {len(targets):,}")

    collected = []

    for idx, receipt in enumerate(targets, start=1):
        meta = meta_map.get(receipt)
        if meta is None:
            print(f"\n[{idx}] {receipt} | metadata missing")
            continue

        p = path_map.get(receipt)
        if not p:
            print(f"\n[{idx}] {receipt} | selected_path missing")
            continue

        zip_path = Path(str(p))

        print("\n" + "-" * 120)
        print(
            f"[{idx}/{len(targets)}] "
            f"{meta.get('stock_code')} | {meta.get('corp_name')} | "
            f"{meta.get('canonical_period_key')} | {receipt}"
        )
        print(f"H6B4 evidence: {meta.get('c4_receipt_evidence')}")

        c1.RAW_ROOT = zip_path.parents[2]
        raw = pd.DataFrame(c1.extract_document_candidates(zip_path))
        if raw.empty:
            print("C1 raw empty")
            continue

        raw["stock_code"] = str(meta.get("stock_code", "")).zfill(6)
        raw["period_key"] = meta.get("canonical_period_key")
        raw["rcept_no"] = receipt
        raw["rcept_dt"] = pd.to_datetime(meta.get("rcept_dt"), errors="coerce")

        c4df = pd.DataFrame(c4.build_account_candidates(raw))

        missing = (
            family_evidence.loc[
                family_evidence["rcept_no"].eq(receipt),
                ["missing_metric", "account_family", "c4_evidence_class"],
            ]
            .drop_duplicates()
        )

        print("\n[Missing families]")
        print(missing.to_string(index=False))

        families = set(missing["account_family"].astype(str))
        sub = c4df.loc[
            c4df["account_family"].astype(str).isin(families)
        ].copy()

        if str(meta.get("balance_qa")) == "fail":
            bs = c4df.loc[
                c4df["account_family"].isin(["assets", "liabilities", "equity"])
            ].copy()
            sub = pd.concat([sub, bs], ignore_index=True).drop_duplicates()

        if sub.empty:
            print("No C4 candidate rows")
            continue

        sub["diagnostic_receipt_evidence"] = meta.get("c4_receipt_evidence")
        fam_class = {
            str(r["account_family"]): str(r["c4_evidence_class"])
            for _, r in missing.iterrows()
        }
        sub["diagnostic_family_evidence"] = (
            sub["account_family"].astype(str).map(fam_class)
        )
        collected.append(sub)

        cols = [
            c for c in [
                "account_family",
                "column_status",
                "selected_value_krw",
                "table_index",
                "row_index",
                "selected_column",
                "statement_type",
                "primary_row_label",
                "heading_context",
                "context_reasons",
            ]
            if c in sub.columns
        ]

        focus = sub.loc[
            sub["column_status"].astype(str).isin(
                ["selected", "ambiguous_column", "no_current_value"]
            )
        ].copy()
        if focus.empty:
            focus = sub

        print("\n[C4 focus rows]")
        with pd.option_context(
            "display.max_colwidth", 220,
            "display.width", 280,
            "display.max_rows", 80,
        ):
            print(focus[cols].head(80).to_string(index=False))

    if collected:
        out_df = pd.concat(collected, ignore_index=True)
        out_df.to_csv(OUT, index=False, encoding="utf-8-sig")
    else:
        out_df = pd.DataFrame()

    print("\n" + "=" * 120)
    print("H6B5 SUMMARY")
    print("=" * 120)
    print(f"\nDiagnostic rows: {len(out_df):,}")
    print(f"Output: {OUT}")
    print(
        "\n판단 목표:\n"
        "- UNKNOWN/MIXED가 넓은 heading_context 때문에 생기는지 확인\n"
        "- table별 제목에 연결/별도 표지가 실제로 있는지 확인\n"
        "- 제낙스 exact-pass 두 세트 중 CFS를 문맥으로 구분 가능한지 확인\n"
        "- 규칙 확인 전 OFS fallback/최종 merge 금지"
    )


if __name__ == "__main__":
    main()
