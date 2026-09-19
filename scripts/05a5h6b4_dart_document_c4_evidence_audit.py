from __future__ import annotations

import argparse
import importlib.util
import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B4. Document C4 Evidence Audit
#
# 목적
# ------------------------------------------------------------
# H6B3에서 아직 채택하지 않은 receipt를 다시 document C4 단계까지
# 내려가서 "왜 strict CFS가 확정되지 않았는지"를 분류한다.
#
# 동시에 hard balance fail receipt는 같은 table + same current column
# 안의 assets/liabilities/equity 조합을 다시 검사한다.
#
# API 호출 없음.
# 자동 OFS fallback 없음.
# 자동 production merge 없음.
#
# INPUT
# ------------------------------------------------------------
# dart_noncorrected_nodata_document_recovery_triage.parquet
# dart_noncorrected_nodata_source_manifest.parquet
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_nodata_c4_missing_family_evidence.csv
# dart_noncorrected_nodata_c4_receipt_evidence_summary.csv
# dart_noncorrected_nodata_hard_balance_candidate_sets.csv
# dart_noncorrected_nodata_hard_balance_resolution.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h6b4_dart_document_c4_evidence_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TRIAGE = (
    INTERIM
    / "dart_noncorrected_nodata_document_recovery_triage.parquet"
)

SOURCE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.parquet"
)

OUT_FAMILY = (
    INTERIM
    / "dart_noncorrected_nodata_c4_missing_family_evidence.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_c4_receipt_evidence_summary.csv"
)

OUT_BALANCE_SETS = (
    INTERIM
    / "dart_noncorrected_nodata_hard_balance_candidate_sets.csv"
)

OUT_BALANCE_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_nodata_hard_balance_resolution.csv"
)

METRIC_TO_FAMILY = {
    "assets": "assets",
    "liabilities": "liabilities",
    "equity_total": "equity",
    "revenue_cumulative": "revenue",
    "operating_income_cumulative": "operating_income",
    "net_income_total_cumulative": "net_income",
}

CORE = list(
    METRIC_TO_FAMILY.keys()
)

ABS_TOL = 2_000_000
REL_TOL = 1e-6


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


def norm(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def basis_hint(row: pd.Series) -> str:
    chunks = []

    for col in [
        "statement_type",
        "heading_context",
        "context_reasons",
        "row_text",
    ]:
        if col in row.index:
            chunks.append(norm(row.get(col)))

    text = " | ".join(chunks)

    cfs = (
        ("연결" in text)
        or ("consolidated" in text)
        or (" cfs " in f" {text} ")
    )

    ofs = (
        ("별도" in text)
        or ("개별" in text)
        or ("separate" in text)
        or (" ofs " in f" {text} ")
    )

    if cfs and not ofs:
        return "CFS_HINT"

    if ofs and not cfs:
        return "OFS_HINT"

    if cfs and ofs:
        return "MIXED_HINT"

    return "UNKNOWN"


def parse_num(value):
    return pd.to_numeric(
        value,
        errors="coerce",
    )


def balance_qa(a, l, e):
    if any(pd.isna(x) for x in [a, l, e]):
        return "missing", np.nan, np.nan

    gap = float(a) - float(l) - float(e)

    rel = (
        abs(gap) / abs(float(a))
        if float(a) != 0
        else np.nan
    )

    if gap == 0:
        qa = "exact_pass"

    elif (
        abs(gap) <= ABS_TOL
        or (
            pd.notna(rel)
            and rel <= REL_TOL
        )
    ):
        qa = "rounding_pass"

    else:
        qa = "fail"

    return qa, gap, rel


def missing_metrics(row: pd.Series):
    return [
        m
        for m in CORE
        if pd.isna(row.get(m))
    ]


def family_evidence_class(c4_family: pd.DataFrame) -> str:
    if c4_family.empty:
        return "no_c4_candidate_rows"

    statuses = set(
        c4_family["column_status"]
        .astype("string")
        .fillna("")
    )

    selected = c4_family.loc[
        c4_family["column_status"].eq("selected")
    ].copy()

    if selected.empty:
        if statuses == {"no_current_value"}:
            return "c4_rows_but_no_current_value"

        if "ambiguous_column" in statuses:
            return "c4_column_ambiguous"

        return "c4_rows_not_selected_other"

    hints = set(
        selected.apply(
            basis_hint,
            axis=1,
        )
    )

    if "CFS_HINT" in hints:
        return "selected_cfs_hint_exists"

    if hints == {"OFS_HINT"}:
        return "selected_only_ofs_hint"

    if "MIXED_HINT" in hints:
        return "selected_mixed_basis_hint"

    return "selected_basis_unknown"


def make_balance_sets(
    c4df: pd.DataFrame,
    base: dict,
) -> list[dict]:

    work = c4df.loc[
        c4df["account_family"].isin(
            ["assets", "liabilities", "equity"]
        )
        & c4df["column_status"].eq("selected")
    ].copy()

    if work.empty:
        return []

    records = []

    for (table_index, selected_column), group in work.groupby(
        ["table_index", "selected_column"],
        dropna=False,
    ):

        fam = {
            f: group.loc[
                group["account_family"].eq(f)
            ]
            for f in ["assets", "liabilities", "equity"]
        }

        if any(x.empty for x in fam.values()):
            continue

        for (_, a), (_, l), (_, e) in itertools.product(
            fam["assets"].head(10).iterrows(),
            fam["liabilities"].head(10).iterrows(),
            fam["equity"].head(10).iterrows(),
        ):

            av = parse_num(a.get("selected_value_krw"))
            lv = parse_num(l.get("selected_value_krw"))
            ev = parse_num(e.get("selected_value_krw"))

            qa, gap, rel = balance_qa(
                av,
                lv,
                ev,
            )

            hints = [
                basis_hint(a),
                basis_hint(l),
                basis_hint(e),
            ]

            if all(h == "CFS_HINT" for h in hints):
                set_basis = "CFS_HINT"

            elif all(h == "OFS_HINT" for h in hints):
                set_basis = "OFS_HINT"

            elif any(h == "MIXED_HINT" for h in hints):
                set_basis = "MIXED_HINT"

            elif len(set(hints)) == 1:
                set_basis = hints[0]

            else:
                set_basis = "MIXED_OR_UNKNOWN"

            records.append(
                {
                    **base,
                    "table_index": table_index,
                    "selected_column": selected_column,
                    "assets": av,
                    "liabilities": lv,
                    "equity_total": ev,
                    "balance_gap": gap,
                    "balance_relative_gap": rel,
                    "balance_qa": qa,
                    "basis_hint": set_basis,
                    "assets_row": a.get("row_index"),
                    "liabilities_row": l.get("row_index"),
                    "equity_row": e.get("row_index"),
                    "assets_heading": a.get("heading_context"),
                    "liabilities_heading": l.get("heading_context"),
                    "equity_heading": e.get("heading_context"),
                }
            )

    return records


def resolve_hard_balance(sets_df: pd.DataFrame) -> dict:
    if sets_df.empty:
        return {
            "hard_balance_resolution":
            "no_same_table_same_column_set"
        }

    unique = (
        sets_df.drop_duplicates(
            subset=[
                "assets",
                "liabilities",
                "equity_total",
                "table_index",
                "selected_column",
            ]
        )
        .copy()
    )

    passed = unique.loc[
        unique["balance_qa"].isin(
            ["exact_pass", "rounding_pass"]
        )
    ].copy()

    cfs_pass = passed.loc[
        passed["basis_hint"].eq(
            "CFS_HINT"
        )
    ].copy()

    if len(cfs_pass) == 1:
        r = cfs_pass.iloc[0]

        return {
            "hard_balance_resolution":
            "unique_coherent_cfs_set",

            "recommended_assets":
            r["assets"],

            "recommended_liabilities":
            r["liabilities"],

            "recommended_equity_total":
            r["equity_total"],

            "recommended_balance_qa":
            r["balance_qa"],

            "recommended_table_index":
            r["table_index"],

            "recommended_selected_column":
            r["selected_column"],
        }

    if len(cfs_pass) > 1:
        return {
            "hard_balance_resolution":
            "multiple_coherent_cfs_sets"
        }

    if len(passed) == 1:
        r = passed.iloc[0]

        return {
            "hard_balance_resolution":
            "unique_coherent_set_basis_weak",

            "recommended_assets":
            r["assets"],

            "recommended_liabilities":
            r["liabilities"],

            "recommended_equity_total":
            r["equity_total"],

            "recommended_balance_qa":
            r["balance_qa"],

            "recommended_table_index":
            r["table_index"],

            "recommended_selected_column":
            r["selected_column"],

            "recommended_basis_hint":
            r["basis_hint"],
        }

    if len(passed) > 1:
        return {
            "hard_balance_resolution":
            "multiple_coherent_sets_basis_ambiguous"
        }

    return {
        "hard_balance_resolution":
        "no_coherent_balance_set"
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="789개 전체 실행",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="진단용 처리 개수",
    )

    args = parser.parse_args()

    for p in [
        TRIAGE,
        SOURCE_MANIFEST,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    c1 = load_module(
        "h6b4_c1",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    c4 = load_module(
        "h6b4_c4",
        SCRIPTS_DIR
        / "05a5c4_dart_conservative_document_selector.py",
    )

    triage = pd.read_parquet(
        TRIAGE
    )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    triage["rcept_no"] = rcept_str(
        triage["rcept_no"]
    )

    manifest["rcept_no"] = rcept_str(
        manifest["rcept_no"]
    )

    path_map = {
        str(r["rcept_no"]):
        r["selected_path"]
        for _, r
        in manifest.iterrows()
    }

    targets_all = triage.loc[
        ~triage[
            "h6b3_triage_bucket"
        ].eq(
            "adoptable_strict_cfs_complete"
        )
    ].copy()

    if args.full:
        targets = targets_all.copy()

    elif args.limit is not None:
        targets = targets_all.head(
            max(
                int(args.limit),
                0,
            )
        ).copy()

    else:
        # 진단 기본값: 5건만 실행해서 실제 병목 위치를 바로 확인
        targets = targets_all.head(
            5
        ).copy()

    print(
        "\n"
        + "=" * 110
    )

    print(
        "05A5-H6B4 DOCUMENT C4 EVIDENCE AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        f"\nAll targets: {len(targets_all):,}"
        f"\nThis run  : {len(targets):,}"
    )

    family_records = []
    receipt_records = []
    balance_records = []
    balance_resolution_records = []
    errors = []

    for idx, (_, target) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        receipt = str(
            target["rcept_no"]
        )

        stock = str(
            target.get(
                "stock_code",
                "",
            )
        ).zfill(6)

        period = str(
            target.get(
                "canonical_period_key",
                "",
            )
        )

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{stock} | {period} | {receipt}",
            flush=True,
        )

        started = time.perf_counter()

        selected_path = path_map.get(
            receipt
        )

        if not selected_path:
            errors.append(
                (
                    receipt,
                    "selected_path_missing",
                )
            )
            continue

        zip_path = Path(
            str(selected_path)
        )

        try:
            c1.RAW_ROOT = zip_path.parents[2]

            print(
                f"  C1 extracting: {zip_path.name}",
                flush=True,
            )

            c1_started = time.perf_counter()

            raw = pd.DataFrame(
                c1.extract_document_candidates(
                    zip_path
                )
            )

            print(
                f"  C1 done: raw={len(raw):,} "
                f"| elapsed={time.perf_counter() - c1_started:.2f}s",
                flush=True,
            )

            if raw.empty:
                errors.append(
                    (
                        receipt,
                        "c1_no_candidates",
                    )
                )
                continue

            raw["stock_code"] = str(
                target.get(
                    "stock_code",
                    "",
                )
            ).zfill(6)

            raw["period_key"] = target.get(
                "canonical_period_key"
            )

            raw["rcept_no"] = receipt

            raw["rcept_dt"] = pd.to_datetime(
                target.get(
                    "rcept_dt"
                ),
                errors="coerce",
            )

            print(
                "  C4 building candidates...",
                flush=True,
            )

            c4_started = time.perf_counter()

            c4df = pd.DataFrame(
                c4.build_account_candidates(
                    raw
                )
            )

            print(
                f"  C4 done: candidates={len(c4df):,} "
                f"| elapsed={time.perf_counter() - c4_started:.2f}s",
                flush=True,
            )

            missing = missing_metrics(
                target
            )

            per_receipt_classes = []

            for metric in missing:
                family = METRIC_TO_FAMILY[
                    metric
                ]

                sub = c4df.loc[
                    c4df[
                        "account_family"
                    ].eq(
                        family
                    )
                ].copy()

                evidence_class = (
                    family_evidence_class(
                        sub
                    )
                )

                per_receipt_classes.append(
                    (
                        metric,
                        evidence_class,
                    )
                )

                selected_sub = sub.loc[
                    sub[
                        "column_status"
                    ].eq(
                        "selected"
                    )
                ].copy()

                hint_counts = (
                    selected_sub.apply(
                        basis_hint,
                        axis=1,
                    )
                    .value_counts()
                    .to_dict()
                    if not selected_sub.empty
                    else {}
                )

                family_records.append(
                    {
                        "stock_code":
                        target.get(
                            "stock_code"
                        ),

                        "corp_name":
                        target.get(
                            "corp_name"
                        ),

                        "canonical_period_key":
                        target.get(
                            "canonical_period_key"
                        ),

                        "rcept_no":
                        receipt,

                        "h6b3_triage_bucket":
                        target.get(
                            "h6b3_triage_bucket"
                        ),

                        "missing_metric":
                        metric,

                        "account_family":
                        family,

                        "c4_candidate_rows":
                        len(
                            sub
                        ),

                        "c4_selected_rows":
                        int(
                            sub[
                                "column_status"
                            ]
                            .eq(
                                "selected"
                            )
                            .sum()
                        )
                        if not sub.empty
                        else 0,

                        "c4_ambiguous_column_rows":
                        int(
                            sub[
                                "column_status"
                            ]
                            .eq(
                                "ambiguous_column"
                            )
                            .sum()
                        )
                        if not sub.empty
                        else 0,

                        "c4_no_current_value_rows":
                        int(
                            sub[
                                "column_status"
                            ]
                            .eq(
                                "no_current_value"
                            )
                            .sum()
                        )
                        if not sub.empty
                        else 0,

                        "selected_cfs_hint_rows":
                        int(
                            hint_counts.get(
                                "CFS_HINT",
                                0,
                            )
                        ),

                        "selected_ofs_hint_rows":
                        int(
                            hint_counts.get(
                                "OFS_HINT",
                                0,
                            )
                        ),

                        "selected_mixed_hint_rows":
                        int(
                            hint_counts.get(
                                "MIXED_HINT",
                                0,
                            )
                        ),

                        "selected_unknown_hint_rows":
                        int(
                            hint_counts.get(
                                "UNKNOWN",
                                0,
                            )
                        ),

                        "c4_evidence_class":
                        evidence_class,
                    }
                )

            classes = [
                c
                for _, c
                in per_receipt_classes
            ]

            if any(
                c
                == "selected_cfs_hint_exists"
                for c in classes
            ):
                receipt_evidence = (
                    "cfs_candidate_exists_selector_rejected_or_partial"
                )

            elif classes and all(
                c
                == "selected_only_ofs_hint"
                for c in classes
            ):
                receipt_evidence = (
                    "all_missing_families_only_ofs_hint"
                )

            elif any(
                c
                in {
                    "selected_mixed_basis_hint",
                    "selected_basis_unknown",
                }
                for c in classes
            ):
                receipt_evidence = (
                    "basis_uncertain_in_missing_families"
                )

            elif any(
                c
                == "c4_column_ambiguous"
                for c in classes
            ):
                receipt_evidence = (
                    "current_column_ambiguity"
                )

            elif any(
                c
                == "c4_rows_but_no_current_value"
                for c in classes
            ):
                receipt_evidence = (
                    "candidate_rows_but_no_current_value"
                )

            elif classes and all(
                c
                == "no_c4_candidate_rows"
                for c in classes
            ):
                receipt_evidence = (
                    "no_c4_candidate_rows_for_all_missing"
                )

            else:
                receipt_evidence = (
                    "mixed_other_evidence"
                )

            receipt_records.append(
                {
                    "stock_code":
                    target.get(
                        "stock_code"
                    ),

                    "corp_name":
                    target.get(
                        "corp_name"
                    ),

                    "canonical_period_key":
                    target.get(
                        "canonical_period_key"
                    ),

                    "rcept_no":
                    receipt,

                    "h6b3_triage_bucket":
                    target.get(
                        "h6b3_triage_bucket"
                    ),

                    "strict_cfs_core_count":
                    target.get(
                        "strict_cfs_core_count"
                    ),

                    "balance_qa":
                    target.get(
                        "balance_qa"
                    ),

                    "missing_metrics":
                    " | ".join(
                        missing
                    ),

                    "c4_receipt_evidence":
                    receipt_evidence,

                    "family_evidence_detail":
                    " | ".join(
                        f"{m}:{c}"
                        for m, c
                        in per_receipt_classes
                    ),
                }
            )

            if (
                target.get(
                    "h6b3_triage_bucket"
                )
                == "hard_balance_fail_manual_audit"
            ):
                base = {
                    "stock_code":
                    target.get(
                        "stock_code"
                    ),

                    "corp_name":
                    target.get(
                        "corp_name"
                    ),

                    "canonical_period_key":
                    target.get(
                        "canonical_period_key"
                    ),

                    "rcept_no":
                    receipt,
                }

                sets = make_balance_sets(
                    c4df,
                    base,
                )

                if sets:
                    temp = pd.DataFrame(
                        sets
                    )
                    balance_records.append(
                        temp
                    )
                else:
                    temp = pd.DataFrame()

                resolution = (
                    resolve_hard_balance(
                        temp
                    )
                )

                balance_resolution_records.append(
                    {
                        **base,
                        **resolution,
                    }
                )

        except Exception as exc:
            errors.append(
                (
                    receipt,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            print(
                f"  ERROR: {type(exc).__name__}: {exc}",
                flush=True,
            )

        print(
            f"  receipt done | total elapsed="
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )

    family_df = pd.DataFrame(
        family_records
    )

    receipt_df = pd.DataFrame(
        receipt_records
    )

    balance_sets_df = (
        pd.concat(
            balance_records,
            ignore_index=True,
        )
        if balance_records
        else pd.DataFrame()
    )

    balance_resolution_df = pd.DataFrame(
        balance_resolution_records
    )

    family_df.to_csv(
        OUT_FAMILY,
        index=False,
        encoding="utf-8-sig",
    )

    receipt_df.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    balance_sets_df.to_csv(
        OUT_BALANCE_SETS,
        index=False,
        encoding="utf-8-sig",
    )

    balance_resolution_df.to_csv(
        OUT_BALANCE_RESOLUTION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "H6B4 SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        "\n[C4 receipt evidence]"
    )

    if receipt_df.empty:
        print(
            "None"
        )
    else:
        print(
            receipt_df[
                "c4_receipt_evidence"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[C4 missing-family evidence]"
    )

    if family_df.empty:
        print(
            "None"
        )
    else:
        print(
            family_df[
                "c4_evidence_class"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[C4 missing-family evidence by metric]"
    )

    if family_df.empty:
        print(
            "None"
        )
    else:
        print(
            pd.crosstab(
                family_df[
                    "missing_metric"
                ],
                family_df[
                    "c4_evidence_class"
                ],
                dropna=False,
            )
            .to_string()
        )

    print(
        "\n[Hard balance resolution]"
    )

    if balance_resolution_df.empty:
        print(
            "None"
        )
    else:
        print(
            balance_resolution_df.to_string(
                index=False
            )
        )

    if not balance_sets_df.empty:
        print(
            "\n[Hard balance candidate sets]"
        )

        cols = [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "table_index",
            "selected_column",
            "assets",
            "liabilities",
            "equity_total",
            "balance_gap",
            "balance_relative_gap",
            "balance_qa",
            "basis_hint",
        ]

        print(
            balance_sets_df[
                cols
            ]
            .sort_values(
                [
                    "balance_qa",
                    "balance_relative_gap",
                ]
            )
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    print(
        f"\nErrors: "
        f"{len(errors):,}"
    )

    if errors:
        for receipt, msg in errors[
            :20
        ]:
            print(
                f"  {receipt} | {msg}"
            )

    print(
        f"\nFamily evidence : "
        f"{OUT_FAMILY}"
    )

    print(
        f"Receipt summary : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"Balance sets    : "
        f"{OUT_BALANCE_SETS}"
    )

    print(
        f"Balance resolve : "
        f"{OUT_BALANCE_RESOLUTION}"
    )

    print(
        "\n다음 판단:"
        "\n- unique_coherent_cfs_set이면 hard balance 1건은 same-receipt document CFS로 교체 후보"
        "\n- cfs_candidate_exists_selector_rejected_or_partial이면 OFS fallback 금지, selector/candidate audit"
        "\n- all_missing_families_only_ofs_hint는 OFS fallback 후보이지만 아직 자동 채택 금지"
        "\n- no_c4_candidate_rows_for_all_missing / no-current-value는 실제 source 구조/계정 부재 쪽으로 분류"
    )


if __name__ == "__main__":
    main()
