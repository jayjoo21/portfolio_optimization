from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7B. Cached C4 Replay + Basis V2 + Period V2
#
# 목적
# ------------------------------------------------------------
# H6B7A에서 만든 receipt별 C1 raw cache만 사용하여:
#
#   C1 raw cache
#      ↓
#   existing C4 build_account_candidates()
#      ↓
#   Basis Detector V2 annotation
#      +
#   Current-Period Resolver V2 annotation
#
# 을 890건 전체에 적용한다.
#
# IMPORTANT
# ------------------------------------------------------------
# - document ZIP 재파싱 없음
# - API 호출 없음
# - final merge 없음
# - 기존 값을 자동 덮어쓰지 않음
# - old C4 vs V2 충돌을 먼저 계량화
# - receipt별 output cache + resume 지원
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/nodata_c1_raw_cache/<rcept_no>.parquet
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/nodata_c4_v2_cache/<rcept_no>.parquet
# data/interim/dart/dart_noncorrected_nodata_c4_v2_cache_manifest.csv
# data/interim/dart/dart_noncorrected_nodata_c4_v2_all.parquet
#
# 실행
# ------------------------------------------------------------
# 기본 5건:
# python scripts\05a5h6b7b_dart_cached_c4_v2_replay.py
#
# 전체:
# python scripts\05a5h6b7b_dart_cached_c4_v2_replay.py --full
#
# 일부:
# python scripts\05a5h6b7b_dart_cached_c4_v2_replay.py --limit 20
#
# combined만:
# python scripts\05a5h6b7b_dart_cached_c4_v2_replay.py --combine-only
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

SCOPE = INTERIM / "dart_noncorrected_full_api_no_data_scope.csv"

C1_CACHE_DIR = INTERIM / "nodata_c1_raw_cache"

C4V2_CACHE_DIR = INTERIM / "nodata_c4_v2_cache"

CACHE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_c4_v2_cache_manifest.csv"
)

COMBINED = (
    INTERIM
    / "dart_noncorrected_nodata_c4_v2_all.parquet"
)


# ============================================================
# Utilities
# ============================================================


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


def receipt_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def compact(value) -> str:
    return re.sub(
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        clean_text(value).lower(),
    )


def marker_text(value) -> str:
    return re.sub(
        r"[\s\(\)\[\]\{\}〈〉《》「」『』【】]+",
        "",
        clean_text(value).lower(),
    )


def cache_path_for(receipt: str) -> Path:
    return C4V2_CACHE_DIR / f"{receipt}.parquet"


# ============================================================
# Basis Detector V2
# ============================================================


def detect_basis_v2(row: pd.Series) -> tuple[str, str, int, int]:

    # Do NOT use legacy context_reasons.
    context = compact(
        " ".join(
            [
                clean_text(row.get("heading_context")),
                clean_text(row.get("primary_row_label")),
            ]
        )
    )

    cfs_reasons = []
    ofs_reasons = []

    no_cfs_patterns = [
        (
            r"연결대상종속기업.*존재하지않",
            "no_consolidated_subsidiary",
        ),
        (
            r"연결대상종속기업.*없",
            "no_consolidated_subsidiary",
        ),
        (
            r"연결재무제표.*작성하지않",
            "consolidated_statements_not_prepared",
        ),
        (
            r"연결재무제표.*작성의무.*없",
            "no_consolidated_reporting_obligation",
        ),
    ]

    for pattern, reason in no_cfs_patterns:
        if re.search(pattern, context):
            ofs_reasons.append((reason, 100))

    ofs_patterns = [
        ("요약개별재무정보", "summary_individual_financial_info", 20),
        ("개별재무정보", "individual_financial_info", 18),
        ("본자료는개별기준", "explicit_individual_basis", 25),
        ("개별기준으로작성", "explicit_individual_basis", 25),
        ("개별재무제표", "individual_financial_statements", 25),
        ("별도재무제표", "separate_financial_statements", 25),
        ("별도재무상태표", "separate_balance_sheet", 30),
        ("별도포괄손익계산서", "separate_comprehensive_income_statement", 30),
        ("별도손익계산서", "separate_income_statement", 30),
        ("별도자본변동표", "separate_changes_in_equity", 30),
        ("별도현금흐름표", "separate_cashflow_statement", 30),
    ]

    for phrase, reason, score in ofs_patterns:
        if phrase in context:
            ofs_reasons.append((reason, score))

    cfs_patterns = [
        ("요약연결재무정보", "summary_consolidated_financial_info", 20),
        ("요약연결재무제표", "summary_consolidated_financial_statements", 20),
        ("연결재무상태표", "consolidated_balance_sheet", 30),
        ("연결포괄손익계산서", "consolidated_comprehensive_income_statement", 30),
        ("연결손익계산서", "consolidated_income_statement", 30),
        ("연결자본변동표", "consolidated_changes_in_equity", 30),
        ("연결현금흐름표", "consolidated_cashflow_statement", 30),
        ("연결재무제표기준", "explicit_consolidated_basis", 25),
        ("연결기준으로작성", "explicit_consolidated_basis", 25),
        ("회사와그종속기업", "company_and_subsidiaries", 12),
        ("주식회사와그종속기업", "company_and_subsidiaries", 12),
    ]

    for phrase, reason, score in cfs_patterns:
        if phrase in context:
            cfs_reasons.append((reason, score))

    cfs_score = sum(score for _, score in cfs_reasons)
    ofs_score = sum(score for _, score in ofs_reasons)

    no_cfs = any(
        reason.startswith("no_consolidated")
        or reason
        in {
            "consolidated_statements_not_prepared",
            "no_consolidated_reporting_obligation",
        }
        for reason, _ in ofs_reasons
    )

    if no_cfs:
        basis = "OFS_ONLY_NO_CFS"

    elif cfs_score > 0 and ofs_score == 0:
        basis = "CFS"

    elif ofs_score > 0 and cfs_score == 0:
        basis = "OFS"

    elif cfs_score > 0 and ofs_score > 0:
        if cfs_score >= ofs_score + 20:
            basis = "CFS"
        elif ofs_score >= cfs_score + 20:
            basis = "OFS"
        else:
            basis = "MIXED"

    else:
        basis = "UNKNOWN"

    reasons = []

    if cfs_reasons:
        reasons.append(
            "CFS:"
            + ",".join(reason for reason, _ in cfs_reasons)
        )

    if ofs_reasons:
        reasons.append(
            "OFS:"
            + ",".join(reason for reason, _ in ofs_reasons)
        )

    return (
        basis,
        " | ".join(reasons),
        cfs_score,
        ofs_score,
    )


# ============================================================
# Period Resolver V2
# ============================================================


def parse_period_key(period_key: str) -> dict:
    text = clean_text(period_key)

    if "|" not in text:
        return {
            "report_type": "",
            "year": None,
            "month": None,
            "kind": "UNKNOWN",
        }

    report_type, ym = text.split("|", 1)

    m = re.search(
        r"(?P<year>\d{4})\.(?P<month>\d{2})",
        ym,
    )

    year = int(m.group("year")) if m else None
    month = int(m.group("month")) if m else None

    if "사업보고서" in report_type:
        kind = "FY"
    elif "반기보고서" in report_type:
        kind = "H1"
    elif "분기보고서" in report_type:
        if month == 3:
            kind = "Q1"
        elif month == 9:
            kind = "Q3"
        else:
            kind = "QUARTER"
    else:
        kind = "UNKNOWN"

    return {
        "report_type": report_type,
        "year": year,
        "month": month,
        "kind": kind,
    }


def parse_generation(label: str):
    text = compact(label)

    m = re.search(r"제(\d+)기", text)

    if not m:
        return None

    return int(m.group(1))


def extract_years(label: str) -> list[int]:
    return [
        int(x)
        for x in re.findall(
            r"(?<!\d)(20\d{2})(?!\d)",
            clean_text(label),
        )
    ]


def extract_months(label: str) -> list[int]:
    text = clean_text(label)

    months = []

    for m in re.finditer(
        r"20\d{2}[./\-년]\s*(\d{1,2})",
        text,
    ):
        try:
            months.append(int(m.group(1)))
        except Exception:
            pass

    return months


def score_period_candidate(
    column: str,
    period_info: dict,
    max_generation: int | None,
) -> dict:

    original = clean_text(column)
    text = compact(column)
    marker = marker_text(column)

    target_year = period_info["year"]
    target_month = period_info["month"]
    kind = period_info["kind"]

    score = 0
    reasons = []
    hard_reject = False

    current_marker = (
        "당기" in marker
        or "당분기" in marker
        or "당반기" in marker
    )

    prior_marker = (
        "전기" in marker
        or "전분기" in marker
        or "전반기" in marker
    )

    if current_marker:
        score += 100
        reasons.append("explicit_current_marker")

    if prior_marker:
        score -= 150
        reasons.append("explicit_prior_marker")
        hard_reject = True

    years = extract_years(original)

    if target_year is not None and years:
        if target_year in years:
            score += 70
            reasons.append("target_year_match")
        else:
            score -= 120
            reasons.append("target_year_mismatch")
            hard_reject = True

    months = extract_months(original)

    if target_month is not None and months:
        if target_month in months:
            score += 40
            reasons.append("target_month_match")
        else:
            score -= 70
            reasons.append("target_month_mismatch")

    has_q1 = ("1분기" in text or "3개월" in text)
    has_h1 = ("반기" in text or "6개월" in text)
    has_q3 = ("3분기" in text or "9개월" in text)
    has_fy = (
        "기말" in text
        or "12월말" in text
        or "연말" in text
    )

    if kind == "Q1":
        if has_q1:
            score += 50
            reasons.append("Q1_match")
        elif has_h1 or has_q3:
            score -= 100
            reasons.append("Q1_period_mismatch")
            hard_reject = True

    elif kind == "H1":
        if has_h1:
            score += 50
            reasons.append("H1_match")

        if has_q1:
            score -= 120
            reasons.append("H1_Q1_mismatch")
            hard_reject = True

        if has_q3:
            score -= 120
            reasons.append("H1_Q3_mismatch")
            hard_reject = True

    elif kind == "Q3":
        if has_q3:
            score += 50
            reasons.append("Q3_match")

        if has_q1:
            score -= 120
            reasons.append("Q3_Q1_mismatch")
            hard_reject = True

        if has_h1:
            score -= 100
            reasons.append("Q3_H1_mismatch")
            hard_reject = True

    elif kind == "FY":
        if has_fy:
            reasons.append("FY_form_only")

        if has_q1 or has_h1 or has_q3:
            score -= 120
            reasons.append("FY_interim_mismatch")
            hard_reject = True

    generation = parse_generation(original)

    if (
        generation is not None
        and max_generation is not None
        and generation == max_generation
        and not years
        and not current_marker
        and not prior_marker
    ):
        score += 20
        reasons.append(
            "highest_generation_low_confidence"
        )

    if hard_reject:
        confidence = "REJECT"

    elif (
        "explicit_current_marker" in reasons
        or (
            "target_year_match" in reasons
            and "target_month_match" in reasons
        )
    ):
        confidence = "HIGH"

    elif (
        "target_year_match" in reasons
        or "Q1_match" in reasons
        or "H1_match" in reasons
        or "Q3_match" in reasons
    ):
        confidence = "MEDIUM"

    elif "highest_generation_low_confidence" in reasons:
        confidence = "LOW"

    else:
        confidence = "NONE"

    return {
        "column": original,
        "score": score,
        "reasons": reasons,
        "hard_reject": hard_reject,
        "confidence": confidence,
        "generation": generation,
    }


def load_candidates(value) -> list[dict]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [
            x for x in value
            if isinstance(x, dict)
        ]

    try:
        parsed = json.loads(str(value))
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    return [
        x for x in parsed
        if isinstance(x, dict)
    ]


def resolve_period_v2(row: pd.Series) -> dict:

    period_info = parse_period_key(
        row.get("period_key", "")
    )

    candidate_field = None

    for name in [
        "numeric_candidates_debug",
        "numeric_candidates_json",
    ]:
        if name in row.index and pd.notna(row.get(name)):
            candidate_field = row.get(name)
            break

    candidates = load_candidates(candidate_field)

    if not candidates:
        return {
            "period_v2_status": "no_numeric_candidates",
            "period_v2_selected_column": None,
            "period_v2_numeric": np.nan,
            "period_v2_score": np.nan,
            "period_v2_confidence": "NONE",
            "period_v2_reasons": "",
            "period_v2_all_candidates": "[]",
        }

    generations = [
        parse_generation(c.get("column", ""))
        for c in candidates
    ]

    generations = [
        g for g in generations
        if g is not None
    ]

    max_generation = max(generations) if generations else None

    scored = []

    for c in candidates:
        result = score_period_candidate(
            c.get("column", ""),
            period_info,
            max_generation,
        )

        result["numeric"] = pd.to_numeric(
            c.get("numeric"),
            errors="coerce",
        )

        result["raw"] = c.get("raw")

        scored.append(result)

    eligible = [
        x for x in scored
        if (
            not x["hard_reject"]
            and x["confidence"]
            in {
                "HIGH",
                "MEDIUM",
                "LOW",
            }
        )
    ]

    if not eligible:
        return {
            "period_v2_status": "no_current_candidate",
            "period_v2_selected_column": None,
            "period_v2_numeric": np.nan,
            "period_v2_score": np.nan,
            "period_v2_confidence": "NONE",
            "period_v2_reasons": "",
            "period_v2_all_candidates": json.dumps(
                scored,
                ensure_ascii=False,
            ),
        }

    eligible = sorted(
        eligible,
        key=lambda x: (
            x["score"],
            {
                "HIGH": 3,
                "MEDIUM": 2,
                "LOW": 1,
            }.get(x["confidence"], 0),
        ),
        reverse=True,
    )

    top_score = eligible[0]["score"]

    top = [
        x for x in eligible
        if x["score"] == top_score
    ]

    unique_top = {}

    for x in top:
        key = (
            x["column"],
            x["numeric"],
        )

        unique_top[key] = x

    top = list(unique_top.values())

    if len(top) != 1:
        return {
            "period_v2_status": "ambiguous_top_candidates",
            "period_v2_selected_column": None,
            "period_v2_numeric": np.nan,
            "period_v2_score": np.nan,
            "period_v2_confidence": "NONE",
            "period_v2_reasons": "",
            "period_v2_all_candidates": json.dumps(
                scored,
                ensure_ascii=False,
            ),
        }

    chosen = top[0]

    if chosen["confidence"] == "LOW":
        status = "low_confidence_candidate"
    else:
        status = "selected"

    return {
        "period_v2_status": status,
        "period_v2_selected_column": chosen["column"],
        "period_v2_numeric": chosen["numeric"],
        "period_v2_score": chosen["score"],
        "period_v2_confidence": chosen["confidence"],
        "period_v2_reasons": " | ".join(
            chosen["reasons"]
        ),
        "period_v2_all_candidates": json.dumps(
            scored,
            ensure_ascii=False,
        ),
    }


# ============================================================
# Manifest / combine
# ============================================================


def load_existing_manifest() -> pd.DataFrame:
    if not CACHE_MANIFEST.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        CACHE_MANIFEST,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    if "rcept_no" in df.columns:
        df["rcept_no"] = receipt_string(
            df["rcept_no"]
        )

    return df


def append_manifest_row(record: dict):

    existing = load_existing_manifest()

    new = pd.DataFrame([record])

    if existing.empty:
        out = new
    else:
        existing = existing.loc[
            ~existing["rcept_no"]
            .astype(str)
            .eq(str(record["rcept_no"]))
        ]

        out = pd.concat(
            [existing, new],
            ignore_index=True,
        )

    out.to_csv(
        CACHE_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )


def build_combined(expected_receipts: list[str]):

    files = []
    missing = []

    for receipt in expected_receipts:
        p = cache_path_for(receipt)

        if p.exists():
            files.append(p)
        else:
            missing.append(receipt)

    print("\n[Combine C4 V2 cache]")
    print(f"Expected receipts : {len(expected_receipts):,}")
    print(f"Cached receipts   : {len(files):,}")
    print(f"Missing receipts  : {len(missing):,}")

    if missing:
        print(
            "모든 receipt의 C4 V2 cache가 생긴 뒤 "
            "combined parquet을 생성합니다."
        )
        print("Missing sample:", missing[:20])
        return

    frames = []

    for i, p in enumerate(files, start=1):
        frames.append(
            pd.read_parquet(p)
        )

        if i % 100 == 0 or i == len(files):
            print(
                f"  read {i:,} / {len(files):,}"
            )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined.to_parquet(
        COMBINED,
        index=False,
    )

    print(f"\nCombined rows: {len(combined):,}")
    print(f"Combined file: {COMBINED}")

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    print("\n[Basis V2]")
    print(
        combined["basis_v2"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Period V2 status]")
    print(
        combined["period_v2_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Period V2 confidence]")
    print(
        combined["period_v2_confidence"]
        .value_counts(dropna=False)
        .to_string()
    )

    if "column_status" in combined.columns:
        print("\n[Old C4 column status x Period V2]")
        print(
            pd.crosstab(
                combined["column_status"],
                combined["period_v2_status"],
                dropna=False,
            ).to_string()
        )

        old_missing = (
            combined["column_status"]
            .astype(str)
            .isin(
                [
                    "no_current_value",
                    "ambiguous_column",
                ]
            )
        )

        recovered = combined.loc[
            old_missing
            & combined["period_v2_status"].eq("selected")
        ]

        print("\n[Strong period recoveries over old C4]")
        print(f"{len(recovered):,}")

        old_selected = combined[
            "column_status"
        ].astype(str).eq("selected")

        v2_no_strong = ~combined[
            "period_v2_status"
        ].eq("selected")

        conflicts = combined.loc[
            old_selected
            & v2_no_strong
        ]

        print("\n[Old selected but V2 not strong-selected]")
        print(f"{len(conflicts):,}")

        if not conflicts.empty:
            show_cols = [
                c for c in [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "account_family",
                    "table_index",
                    "row_index",
                    "selected_column",
                    "column_status",
                    "basis_v2",
                    "period_v2_status",
                    "period_v2_selected_column",
                    "period_v2_confidence",
                    "heading_context",
                ]
                if c in conflicts.columns
            ]

            print(
                conflicts[show_cols]
                .head(40)
                .to_string(index=False)
            )


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--combine-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not SCOPE.exists():
        raise FileNotFoundError(SCOPE)

    if not C1_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"C1 cache dir missing: {C1_CACHE_DIR}"
        )

    C4V2_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scope = pd.read_csv(
        SCOPE,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    scope["rcept_no"] = receipt_string(
        scope["rcept_no"]
    )

    receipts = (
        scope["rcept_no"]
        .astype(str)
        .tolist()
    )

    print("\n" + "=" * 110)
    print("05A5-H6B7B CACHED C4 V2 REPLAY")
    print("=" * 110)

    print(f"\nExpected receipts: {len(receipts):,}")

    missing_c1 = [
        r for r in receipts
        if not (
            C1_CACHE_DIR
            / f"{r}.parquet"
        ).exists()
    ]

    print(
        f"C1 cached       : "
        f"{len(receipts) - len(missing_c1):,} / "
        f"{len(receipts):,}"
    )

    if missing_c1:
        print(
            "\nC1 cache가 아직 완성되지 않았습니다."
        )
        print(
            "먼저 H6B7A --full을 완료하세요."
        )
        print(
            "Missing sample:",
            missing_c1[:20],
        )
        return

    if args.combine_only:
        build_combined(receipts)
        return

    if args.full:
        run_receipts = receipts
    elif args.limit is not None:
        run_receipts = receipts[
            :max(args.limit, 0)
        ]
    else:
        run_receipts = receipts[:5]

    print(f"This run        : {len(run_receipts):,}")

    c4 = load_module(
        "h6b7b_c4",
        SCRIPTS_DIR
        / "05a5c4_dart_conservative_document_selector.py",
    )

    cached_before = 0
    processed_now = 0
    errors = 0

    for idx, receipt in enumerate(
        run_receipts,
        start=1,
    ):

        c1_path = (
            C1_CACHE_DIR
            / f"{receipt}.parquet"
        )

        out_path = cache_path_for(
            receipt
        )

        print(
            f"\n[{idx}/{len(run_receipts)}] "
            f"{receipt}",
            flush=True,
        )

        if out_path.exists():
            try:
                nrows = len(
                    pd.read_parquet(
                        out_path,
                        columns=["rcept_no"],
                    )
                )
            except Exception:
                nrows = -1

            print(
                f"  CACHE HIT | rows={nrows:,}",
                flush=True,
            )

            cached_before += 1
            continue

        started = time.perf_counter()

        try:
            raw = pd.read_parquet(
                c1_path
            )

            print(
                f"  C1 cache rows: {len(raw):,}",
                flush=True,
            )

            c4df = pd.DataFrame(
                c4.build_account_candidates(
                    raw
                )
            )

            print(
                f"  C4 rows     : {len(c4df):,}",
                flush=True,
            )

            if c4df.empty:
                raise RuntimeError(
                    "C4 returned zero rows"
                )

            basis = c4df.apply(
                detect_basis_v2,
                axis=1,
                result_type="expand",
            )

            basis.columns = [
                "basis_v2",
                "basis_v2_reasons",
                "basis_v2_cfs_score",
                "basis_v2_ofs_score",
            ]

            period = c4df.apply(
                resolve_period_v2,
                axis=1,
                result_type="expand",
            )

            out = pd.concat(
                [
                    c4df.reset_index(drop=True),
                    basis.reset_index(drop=True),
                    period.reset_index(drop=True),
                ],
                axis=1,
            )

            # Diagnostic value only. Not final accepted value.
            out[
                "period_v2_value_krw"
            ] = (
                pd.to_numeric(
                    out[
                        "period_v2_numeric"
                    ],
                    errors="coerce",
                )
                * pd.to_numeric(
                    out.get(
                        "unit_multiplier",
                        np.nan,
                    ),
                    errors="coerce",
                )
            )

            out[
                "period_v2_strength"
            ] = np.select(
                [
                    out[
                        "period_v2_confidence"
                    ].eq(
                        "HIGH"
                    ),

                    out[
                        "period_v2_confidence"
                    ].eq(
                        "MEDIUM"
                    ),

                    out[
                        "period_v2_confidence"
                    ].eq(
                        "LOW"
                    ),
                ],
                [
                    "STRONG",
                    "REVIEW",
                    "HOLD",
                ],
                default="NONE",
            )

            out.to_parquet(
                out_path,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            append_manifest_row(
                {
                    "rcept_no": receipt,
                    "cache_status": "cached",
                    "c1_row_count": len(raw),
                    "c4_v2_row_count": len(out),
                    "elapsed_sec": round(elapsed, 4),
                    "error": "",
                }
            )

            print(
                f"  DONE | "
                f"rows={len(out):,} "
                f"| {elapsed:.2f}s",
                flush=True,
            )

            processed_now += 1

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - started
            )

            msg = (
                f"{type(exc).__name__}: {exc}"
            )

            append_manifest_row(
                {
                    "rcept_no": receipt,
                    "cache_status": "error",
                    "c1_row_count": 0,
                    "c4_v2_row_count": 0,
                    "elapsed_sec": round(elapsed, 4),
                    "error": msg,
                }
            )

            print(
                f"  ERROR | {msg}",
                flush=True,
            )

            errors += 1

    total_cached = sum(
        1 for r in receipts
        if cache_path_for(r).exists()
    )

    print("\n" + "=" * 110)
    print("H6B7B CACHE SUMMARY")
    print("=" * 110)

    print(
        f"\nCached before this run : "
        f"{cached_before:,}"
    )

    print(
        f"Processed this run     : "
        f"{processed_now:,}"
    )

    print(
        f"Errors                 : "
        f"{errors:,}"
    )

    print(
        f"Total cached / expected: "
        f"{total_cached:,} / {len(receipts):,}"
    )

    print(
        f"\nC4 V2 cache dir : "
        f"{C4V2_CACHE_DIR}"
    )

    print(
        f"Manifest        : "
        f"{CACHE_MANIFEST}"
    )

    if total_cached == len(receipts):
        build_combined(receipts)
    else:
        print(
            "\n아직 모든 receipt가 처리되지 않았으므로 "
            "combined parquet은 만들지 않았습니다."
        )


if __name__ == "__main__":
    main()
