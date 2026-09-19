from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "dart"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dart" / "pit_sources_full"

SOURCE_MANIFEST = INTERIM_DIR / "dart_pit_source_full_manifest.parquet"

PARSE_MANIFEST_PARQUET = INTERIM_DIR / "dart_corrected_pit_parse_manifest.parquet"
PARSE_MANIFEST_CSV = INTERIM_DIR / "dart_corrected_pit_parse_manifest.csv"
VALUES_PARQUET = INTERIM_DIR / "dart_corrected_pit_receipt_values.parquet"
VALUES_CSV = INTERIM_DIR / "dart_corrected_pit_receipt_values.csv"
WIDE_PARQUET = INTERIM_DIR / "dart_corrected_pit_receipt_wide.parquet"
WIDE_CSV = INTERIM_DIR / "dart_corrected_pit_receipt_wide.csv"
INTERVALS_PARQUET = INTERIM_DIR / "dart_corrected_pit_validity_intervals.parquet"
INTERVALS_CSV = INTERIM_DIR / "dart_corrected_pit_validity_intervals.csv"

CHECKPOINT_EVERY = 25

# DART document.xml은 XML 컨테이너 안에 HTML-like table markup이 섞여 있어
# validated C1 parser가 BeautifulSoup HTML parser를 의도적으로 사용한다.
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings(
        "ignore",
        category=XMLParsedAsHTMLWarning,
    )
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The behavior of DataFrame concatenation with empty or all-NA entries.*",
)


def load_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"필요한 validated script가 없습니다: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module load 실패: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_validated_parsers():
    c1 = load_module(
        "dart_c1_candidate_extractor",
        SCRIPTS_DIR / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )
    c2 = load_module(
        "dart_c2_xbrl_selector",
        SCRIPTS_DIR / "05a5c2_dart_pit_value_selector_audit.py",
    )
    c4 = load_module(
        "dart_c4_document_candidate_selector",
        SCRIPTS_DIR / "05a5c4_dart_conservative_document_selector.py",
    )
    c5 = load_module(
        "dart_c5_document_strict_selector",
        SCRIPTS_DIR / "05a5c5_dart_strict_consolidated_document_selector.py",
    )

    c1.RAW_ROOT = RAW_ROOT

    return c1, c2, c4, c5


def load_source_manifest() -> pd.DataFrame:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"full source manifest가 없습니다: {SOURCE_MANIFEST}")

    df = pd.read_parquet(SOURCE_MANIFEST)

    df["stock_code"] = (
        df["stock_code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    df["rcept_no"] = (
        df["rcept_no"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], errors="raise")

    return (
        df.sort_values(
            ["stock_code", "period_key", "rcept_dt", "rcept_no"]
        )
        .reset_index(drop=True)
    )


def load_existing_parse_manifest() -> pd.DataFrame:
    if PARSE_MANIFEST_PARQUET.exists():
        df = pd.read_parquet(PARSE_MANIFEST_PARQUET)
        if not df.empty:
            df["rcept_no"] = df["rcept_no"].astype(str)
        return df
    return pd.DataFrame()


def load_existing_values() -> pd.DataFrame:
    if VALUES_PARQUET.exists():
        df = pd.read_parquet(VALUES_PARQUET)
        if not df.empty:
            df["rcept_no"] = df["rcept_no"].astype(str)
        return df
    return pd.DataFrame()


def _normalize_parse_manifest_for_storage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    string_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "source_status",
        "source_type",
        "parse_status",
        "error",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "rcept_dt" in df.columns:
        df["rcept_dt"] = pd.to_datetime(
            df["rcept_dt"],
            errors="coerce",
        )

    for col in [
        "metric_rows",
        "selected_metric_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).astype("Int64")

    if "balance_equation_pass" in df.columns:
        df["balance_equation_pass"] = (
            df["balance_equation_pass"]
            .astype("boolean")
        )

    return df


def _normalize_values_for_storage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    pyarrow는 object 컬럼 안에 str/int가 섞이면 타입 추론에 실패할 수 있다.

    특히 context_or_table:
      XBRL     -> contextRef 문자열
      document -> table_index 숫자

    따라서 의미상 식별자인 컬럼은 모두 pandas StringDtype으로 통일한다.
    """
    df = df.copy()

    string_cols = [
        "stock_code",
        "corp_code",
        "corp_name",
        "period_key",
        "report_nm",
        "rcept_no",
        "source_type",
        "metric",
        "unit",
        "selection_status",
        "concept_or_label",
        "context_or_table",
        "selection_detail",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    datetime_cols = [
        "rcept_dt",
        "period_start",
        "period_end",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

    numeric_cols = [
        "filing_sequence",
        "value",
        "duration_days",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    if "is_correction" in df.columns:
        df["is_correction"] = (
            df["is_correction"]
            .astype("boolean")
        )

    return df


def _atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    tmp = path.with_name(
        path.name + ".tmp"
    )

    df.to_parquet(
        tmp,
        index=False,
    )

    tmp.replace(
        path
    )


def _atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    tmp = path.with_name(
        path.name + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    tmp.replace(
        path
    )


def save_state(
    parse_manifest: pd.DataFrame,
    values: pd.DataFrame,
) -> None:
    if not parse_manifest.empty:
        parse_manifest = (
            parse_manifest
            .sort_values(
                [
                    "stock_code",
                    "period_key",
                    "rcept_dt",
                    "rcept_no",
                ]
            )
            .drop_duplicates(
                subset=["rcept_no"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    if not values.empty:
        values = (
            values
            .sort_values(
                [
                    "stock_code",
                    "period_key",
                    "rcept_dt",
                    "rcept_no",
                    "metric",
                ]
            )
            .drop_duplicates(
                subset=[
                    "rcept_no",
                    "metric",
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )

    parse_manifest = (
        _normalize_parse_manifest_for_storage(
            parse_manifest
        )
    )

    values = (
        _normalize_values_for_storage(
            values
        )
    )

    # CRITICAL:
    # values를 먼저 안전하게 저장하고 마지막에 manifest를 저장한다.
    # manifest가 "완료"라고 기록됐는데 values가 없는 반쪽 checkpoint를 방지.
    _atomic_write_parquet(
        values,
        VALUES_PARQUET,
    )

    _atomic_write_csv(
        values,
        VALUES_CSV,
    )

    _atomic_write_parquet(
        parse_manifest,
        PARSE_MANIFEST_PARQUET,
    )

    _atomic_write_csv(
        parse_manifest,
        PARSE_MANIFEST_CSV,
    )


XBRL_ROLE_TO_METRIC = {
    "assets_total": "assets",
    "liabilities_total": "liabilities",
    "equity_total": "equity_total",
    "equity_parent": "equity_parent",
    "revenue_cumulative": "revenue_cumulative",
    "operating_income_cumulative": "operating_income_cumulative",
    "net_income_total_cumulative": "net_income_total_cumulative",
    "net_income_parent_cumulative": "net_income_parent_cumulative",
}

DOCUMENT_FAMILY_TO_METRIC = {
    "assets": "assets",
    "liabilities": "liabilities",
    "equity": "equity_total",
    "revenue": "revenue_cumulative",
    "operating_income": "operating_income_cumulative",
    "net_income": "net_income_total_cumulative",
}


def parse_xbrl_receipt(source_row: pd.Series, c1, c2):
    zip_path = Path(source_row["zip_path"])

    candidate_rows = c1.extract_xbrl_candidates(zip_path)

    if not candidate_rows:
        raise RuntimeError("XBRL candidate facts = 0")

    candidates = pd.DataFrame(candidate_rows)

    for col in ["rcept_dt", "start_date", "end_date", "instant"]:
        if col in candidates.columns:
            candidates[col] = pd.to_datetime(candidates[col], errors="coerce")

    selected = c2.build_xbrl_selection(candidates)

    rows = []

    for _, result in selected.iterrows():
        role = result["role"]
        metric = XBRL_ROLE_TO_METRIC.get(role)

        if metric is None:
            continue

        status = result["selection_status"]

        selected_value = (
            result.get("selected_value")
            if status == "selected"
            else np.nan
        )

        end_or_instant = result.get("end_date")
        if pd.isna(end_or_instant):
            end_or_instant = result.get("instant")

        rows.append(
            {
                "stock_code": source_row["stock_code"],
                "corp_code": source_row.get("corp_code"),
                "corp_name": source_row.get("corp_name"),
                "period_key": source_row["period_key"],
                "report_nm": source_row.get("report_nm"),
                "rcept_no": source_row["rcept_no"],
                "rcept_dt": source_row["rcept_dt"],
                "is_correction": source_row.get("is_correction"),
                "filing_sequence": source_row.get("filing_sequence"),
                "source_type": "xbrl",
                "metric": metric,
                "value": selected_value,
                "unit": result.get("unit_ref"),
                "selection_status": status,
                "concept_or_label": result.get("concept_local_name"),
                "context_or_table": result.get("context_ref"),
                "period_start": result.get("start_date"),
                "period_end": end_or_instant,
                "duration_days": result.get("duration_days"),
                "selection_detail": result.get("dimension_text"),
            }
        )

    balance_selected = {
        row["metric"]: row["value"]
        for row in rows
        if (
            row["selection_status"] == "selected"
            and row["metric"] in {"assets", "liabilities", "equity_total"}
        )
    }

    balance_pass = None

    if all(
        key in balance_selected
        for key in ["assets", "liabilities", "equity_total"]
    ):
        assets = float(balance_selected["assets"])
        gap = (
            assets
            - float(balance_selected["liabilities"])
            - float(balance_selected["equity_total"])
        )

        relative_gap = abs(gap) / abs(assets) if assets != 0 else np.nan

        balance_pass = (
            bool(relative_gap <= 1e-8)
            if pd.notna(relative_gap)
            else None
        )

    selected_count = sum(
        row["selection_status"] == "selected"
        for row in rows
    )

    parse_meta = {
        "parse_status": "parsed",
        "metric_rows": len(rows),
        "selected_metric_count": selected_count,
        "balance_equation_pass": balance_pass,
    }

    return rows, parse_meta


def parse_document_receipt(source_row: pd.Series, c1, c4, c5):
    zip_path = Path(source_row["zip_path"])

    candidate_rows = c1.extract_document_candidates(zip_path)

    if not candidate_rows:
        return (
            [],
            {
                "parse_status": "parsed_no_candidates",
                "metric_rows": 0,
                "selected_metric_count": 0,
                "balance_equation_pass": None,
            },
        )

    doc_candidates = pd.DataFrame(candidate_rows)

    if "rcept_dt" in doc_candidates.columns:
        doc_candidates["rcept_dt"] = pd.to_datetime(
            doc_candidates["rcept_dt"],
            errors="coerce",
        )

    c4_candidates = c4.build_account_candidates(doc_candidates)

    if c4_candidates.empty:
        return (
            [],
            {
                "parse_status": "parsed_no_candidates",
                "metric_rows": 0,
                "selected_metric_count": 0,
                "balance_equation_pass": None,
            },
        )

    selected = c5.select_values(c4_candidates)

    rows = []
    accepted_statuses = {"selected", "selected_consensus"}

    for _, result in selected.iterrows():
        family = result["account_family"]
        metric = DOCUMENT_FAMILY_TO_METRIC.get(family)

        if metric is None:
            continue

        status = result["selection_status"]

        value = (
            result.get("selected_value_krw")
            if status in accepted_statuses
            else np.nan
        )

        rows.append(
            {
                "stock_code": source_row["stock_code"],
                "corp_code": source_row.get("corp_code"),
                "corp_name": source_row.get("corp_name"),
                "period_key": source_row["period_key"],
                "report_nm": source_row.get("report_nm"),
                "rcept_no": source_row["rcept_no"],
                "rcept_dt": source_row["rcept_dt"],
                "is_correction": source_row.get("is_correction"),
                "filing_sequence": source_row.get("filing_sequence"),
                "source_type": "document_fallback",
                "metric": metric,
                "value": value,
                "unit": "KRW",
                "selection_status": status,
                "concept_or_label": result.get("primary_row_label"),
                "context_or_table": result.get("table_index"),
                "period_start": None,
                "period_end": None,
                "duration_days": None,
                "selection_detail": result.get("heading_context"),
            }
        )

    value_lookup = {
        row["metric"]: row["value"]
        for row in rows
        if (
            row["selection_status"] in accepted_statuses
            and row["metric"] in {"assets", "liabilities", "equity_total"}
        )
    }

    balance_pass = None

    if all(
        key in value_lookup
        for key in ["assets", "liabilities", "equity_total"]
    ):
        assets = float(value_lookup["assets"])
        gap = (
            assets
            - float(value_lookup["liabilities"])
            - float(value_lookup["equity_total"])
        )

        relative_gap = abs(gap) / abs(assets) if assets != 0 else np.nan

        balance_pass = (
            bool(relative_gap <= 1e-8)
            if pd.notna(relative_gap)
            else None
        )

    selected_count = sum(
        row["selection_status"] in accepted_statuses
        for row in rows
    )

    return (
        rows,
        {
            "parse_status": "parsed",
            "metric_rows": len(rows),
            "selected_metric_count": selected_count,
            "balance_equation_pass": balance_pass,
        },
    )


def build_wide_receipts(
    source_manifest: pd.DataFrame,
    values: pd.DataFrame,
) -> pd.DataFrame:

    base_cols = [
        "stock_code",
        "corp_code",
        "corp_name",
        "period_key",
        "report_nm",
        "rcept_no",
        "rcept_dt",
        "is_correction",
        "filing_sequence",
        "chain_submission_count",
        "status",
        "source_type",
    ]

    base_cols = [
        col
        for col in base_cols
        if col in source_manifest.columns
    ]

    base = (
        source_manifest[base_cols]
        .drop_duplicates(subset=["rcept_no"])
        .copy()
    )

    if values.empty:
        return base.copy()

    accepted = values.loc[
        values["selection_status"].isin(
            ["selected", "selected_consensus"]
        )
    ].copy()

    if accepted.empty:
        return base.copy()

    pivot = (
        accepted.pivot_table(
            index="rcept_no",
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    wide = base.merge(
        pivot,
        on="rcept_no",
        how="left",
    )

    metric_columns = [
        "assets",
        "liabilities",
        "equity_total",
        "equity_parent",
        "revenue_cumulative",
        "operating_income_cumulative",
        "net_income_total_cumulative",
        "net_income_parent_cumulative",
    ]

    for col in metric_columns:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["selected_metric_count"] = (
        wide[metric_columns]
        .notna()
        .sum(axis=1)
    )

    return (
        wide.sort_values(
            ["stock_code", "period_key", "rcept_dt", "rcept_no"]
        )
        .reset_index(drop=True)
    )


def build_validity_intervals(
    wide: pd.DataFrame,
) -> pd.DataFrame:

    if wide.empty:
        return wide.copy()

    df = wide.copy()

    df["rcept_dt"] = pd.to_datetime(
        df["rcept_dt"],
        errors="coerce",
    )

    df = df.sort_values(
        ["stock_code", "period_key", "rcept_dt", "rcept_no"]
    )

    df["same_day_sequence"] = (
        df.groupby(
            ["stock_code", "period_key", "rcept_dt"]
        )
        .cumcount()
        + 1
    )

    df["same_day_submission_count"] = (
        df.groupby(
            ["stock_code", "period_key", "rcept_dt"]
        )["rcept_no"]
        .transform("size")
    )

    df["is_eod_effective_version"] = (
        df["same_day_sequence"]
        .eq(df["same_day_submission_count"])
    )

    effective = df.loc[
        df["is_eod_effective_version"]
    ].copy()

    effective = effective.sort_values(
        ["stock_code", "period_key", "rcept_dt", "rcept_no"]
    )

    effective["valid_from_disclosure_date"] = effective["rcept_dt"]

    effective["valid_until_disclosure_date_exclusive"] = (
        effective.groupby(
            ["stock_code", "period_key"]
        )["rcept_dt"]
        .shift(-1)
    )

    effective["is_missing_barrier"] = (
        effective["status"].eq("unavailable")
    )

    effective["decision_clock_aligned"] = False

    return (
        effective.sort_values(
            [
                "stock_code",
                "period_key",
                "valid_from_disclosure_date",
                "rcept_no",
            ]
        )
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="테스트용 처리 receipt 수. --full이면 무시.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 5,625 receipt 처리",
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="기존 parse_status=error만 재처리",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    c1, c2, c4, c5 = load_validated_parsers()

    source = load_source_manifest()
    existing_manifest = load_existing_parse_manifest()
    existing_values = load_existing_values()

    print("\n" + "=" * 80)
    print("05A5-E CORRECTED-PERIOD STRICT-PIT SNAPSHOT BUILD")
    print("=" * 80)

    print(f"source receipts : {len(source):,}")

    print("\n[Source status]")
    print(
        source["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Source type]")
    print(
        source["source_type"]
        .fillna("none")
        .value_counts()
        .to_string()
    )

    if args.retry_errors:
        if existing_manifest.empty:
            print("\n기존 parse manifest가 없어 error retry 대상이 없습니다.")
            return

        error_receipts = set(
            existing_manifest.loc[
                existing_manifest["parse_status"].eq("error"),
                "rcept_no",
            ].astype(str)
        )

        targets = source.loc[
            source["rcept_no"].astype(str).isin(error_receipts)
        ].copy()

        print(f"\nRETRY ERRORS MODE: {len(targets):,}")

    else:
        completed_receipts: set[str] = set()

        if not existing_manifest.empty:
            # checkpoint consistency 검증:
            # parsed인데 metric_rows > 0이면 values 파일에도 해당 receipt가 있어야
            # 진짜 완료로 인정한다.
            value_receipts = (
                set(
                    existing_values[
                        "rcept_no"
                    ].astype(str)
                )
                if not existing_values.empty
                else set()
            )

            for record in existing_manifest.to_dict(
                orient="records"
            ):
                rcept = str(
                    record[
                        "rcept_no"
                    ]
                )

                status = record.get(
                    "parse_status"
                )

                metric_rows = pd.to_numeric(
                    pd.Series(
                        [
                            record.get(
                                "metric_rows",
                                0,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                metric_rows = (
                    int(metric_rows)
                    if pd.notna(metric_rows)
                    else 0
                )

                if status in {
                    "unavailable",
                    "parsed_no_candidates",
                }:
                    completed_receipts.add(
                        rcept
                    )

                elif (
                    status == "parsed"
                    and (
                        metric_rows == 0
                        or rcept in value_receipts
                    )
                ):
                    completed_receipts.add(
                        rcept
                    )

            orphaned = (
                set(
                    existing_manifest.loc[
                        existing_manifest[
                            "parse_status"
                        ].eq(
                            "parsed"
                        ),
                        "rcept_no",
                    ].astype(str)
                )
                - completed_receipts
            )

            if orphaned:
                print(
                    "\n[Resume repair] "
                    f"manifest에는 parsed인데 values가 없는 "
                    f"{len(orphaned):,}건을 자동 재처리합니다."
                )

        targets = source.loc[
            ~source[
                "rcept_no"
            ].astype(str)
            .isin(
                completed_receipts
            )
        ].copy()

        if not args.full:
            targets = targets.head(args.limit)
            print(f"\nTEST MODE: {len(targets):,} receipts")
        else:
            print(f"\nFULL MODE remaining: {len(targets):,}")

    manifest_state: dict[str, dict[str, Any]] = {}

    if not existing_manifest.empty:
        for record in existing_manifest.to_dict(orient="records"):
            manifest_state[str(record["rcept_no"])] = record

    values_state = (
        existing_values.copy()
        if not existing_values.empty
        else pd.DataFrame()
    )

    processed_value_frames = []

    for i, (_, source_row) in enumerate(targets.iterrows(), start=1):
        rcept_no = str(source_row["rcept_no"])
        source_status = source_row["status"]
        source_type = source_row.get("source_type")

        print(
            f"\n[{i}/{len(targets)}] "
            f"{source_row['stock_code']} | "
            f"{source_row['period_key']} | "
            f"{rcept_no} | "
            f"{source_status}/{source_type}"
        )

        manifest_base = {
            "stock_code": source_row["stock_code"],
            "period_key": source_row["period_key"],
            "rcept_no": rcept_no,
            "rcept_dt": source_row["rcept_dt"],
            "source_status": source_status,
            "source_type": source_type,
        }

        if source_status == "unavailable":
            parse_meta = {
                **manifest_base,
                "parse_status": "unavailable",
                "metric_rows": 0,
                "selected_metric_count": 0,
                "balance_equation_pass": None,
                "error": None,
            }
            metric_rows = []

        elif source_status != "available":
            parse_meta = {
                **manifest_base,
                "parse_status": "error",
                "metric_rows": 0,
                "selected_metric_count": 0,
                "balance_equation_pass": None,
                "error": f"unexpected source status: {source_status}",
            }
            metric_rows = []

        else:
            try:
                if source_type == "xbrl":
                    metric_rows, parsed = parse_xbrl_receipt(
                        source_row,
                        c1,
                        c2,
                    )

                elif source_type == "document_fallback":
                    metric_rows, parsed = parse_document_receipt(
                        source_row,
                        c1,
                        c4,
                        c5,
                    )

                else:
                    raise RuntimeError(
                        f"unexpected available source_type={source_type}"
                    )

                parse_meta = {
                    **manifest_base,
                    **parsed,
                    "error": None,
                }

            except Exception as exc:
                metric_rows = []

                parse_meta = {
                    **manifest_base,
                    "parse_status": "error",
                    "metric_rows": 0,
                    "selected_metric_count": 0,
                    "balance_equation_pass": None,
                    "error": repr(exc),
                }

        manifest_state[rcept_no] = parse_meta

        if metric_rows:
            processed_value_frames.append(pd.DataFrame(metric_rows))

        print(
            f"  -> parse={parse_meta['parse_status']} "
            f"selected={parse_meta['selected_metric_count']} "
            f"balance={parse_meta['balance_equation_pass']}"
        )

        if i % CHECKPOINT_EVERY == 0:
            current_manifest = pd.DataFrame(manifest_state.values())

            if processed_value_frames:
                new_values = pd.concat(
                    processed_value_frames,
                    ignore_index=True,
                )

                if values_state.empty:
                    values_state = new_values
                else:
                    touched = set(
                        new_values["rcept_no"].astype(str)
                    )

                    values_state = values_state.loc[
                        ~values_state["rcept_no"]
                        .astype(str)
                        .isin(touched)
                    ]

                    values_state = pd.concat(
                        [values_state, new_values],
                        ignore_index=True,
                    )

                processed_value_frames = []

            save_state(current_manifest, values_state)
            print("  checkpoint saved")

    if processed_value_frames:
        new_values = pd.concat(
            processed_value_frames,
            ignore_index=True,
        )

        if values_state.empty:
            values_state = new_values
        else:
            touched = set(
                new_values["rcept_no"].astype(str)
            )

            values_state = values_state.loc[
                ~values_state["rcept_no"]
                .astype(str)
                .isin(touched)
            ]

            values_state = pd.concat(
                [values_state, new_values],
                ignore_index=True,
            )

    final_manifest = pd.DataFrame(manifest_state.values())

    save_state(final_manifest, values_state)

    wide = build_wide_receipts(
        source_manifest=source,
        values=values_state,
    )

    for col in [
        "stock_code",
        "corp_code",
        "corp_name",
        "period_key",
        "report_nm",
        "rcept_no",
        "status",
        "source_type",
    ]:
        if col in wide.columns:
            wide[col] = wide[col].astype("string")

    _atomic_write_parquet(
        wide,
        WIDE_PARQUET,
    )
    _atomic_write_csv(
        wide,
        WIDE_CSV,
    )

    intervals = build_validity_intervals(
        wide
    )

    for col in [
        "stock_code",
        "corp_code",
        "corp_name",
        "period_key",
        "report_nm",
        "rcept_no",
        "status",
        "source_type",
    ]:
        if col in intervals.columns:
            intervals[col] = (
                intervals[col]
                .astype("string")
            )

    _atomic_write_parquet(
        intervals,
        INTERVALS_PARQUET,
    )
    _atomic_write_csv(
        intervals,
        INTERVALS_CSV,
    )

    print("\n" + "=" * 80)
    print("CORRECTED PIT SNAPSHOT SUMMARY")
    print("=" * 80)

    print("\n[Parse status]")
    print(
        final_manifest["parse_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[Selected metric coverage by source]")

    if final_manifest.empty:
        print("0")
    else:
        coverage = (
            final_manifest.groupby(
                "source_type",
                dropna=False,
            )["selected_metric_count"]
            .agg(["count", "mean", "median", "min", "max"])
        )
        print(coverage.to_string())

    print("\n[Balance QA]")

    if "balance_equation_pass" in final_manifest.columns:
        print(
            final_manifest["balance_equation_pass"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\n[Receipt-wide metric availability]")

    metric_cols = [
        "assets",
        "liabilities",
        "equity_total",
        "equity_parent",
        "revenue_cumulative",
        "operating_income_cumulative",
        "net_income_total_cumulative",
        "net_income_parent_cumulative",
    ]

    print(
        wide[metric_cols]
        .notna()
        .sum()
        .to_string()
    )

    same_day_multi = (
        intervals["same_day_submission_count"]
        .gt(1)
        .sum()
    )

    print("\n[Version intervals]")
    print(f"EOD-effective snapshots : {len(intervals):,}")
    print(
        f"same-day multi-submission EOD rows: "
        f"{same_day_multi:,}"
    )
    print(
        f"missing barriers        : "
        f"{intervals['is_missing_barrier'].sum():,}"
    )

    errors = final_manifest.loc[
        final_manifest["parse_status"].eq("error")
    ]

    print("\n[Parse errors]")

    if errors.empty:
        print("0")
    else:
        print(
            errors[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "source_type",
                    "error",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    print(f"\nParse manifest : {PARSE_MANIFEST_PARQUET}")
    print(f"Long values    : {VALUES_PARQUET}")
    print(f"Wide receipts  : {WIDE_PARQUET}")
    print(f"Intervals      : {INTERVALS_PARQUET}")

    print(
        "\n다음 단계:"
        "\n1) corrected-period parser QA 확인"
        "\n2) non-corrected periods의 일반 DART 재무 API snapshot 생성"
        "\n3) corrected + non-corrected 결합"
        "\n4) trading calendar / decision clock 기준 available_date 확정"
        "\n5) daily as-of join"
        "\n6) profitability / growth / leverage / investment / valuation feature 생성"
    )


if __name__ == "__main__":
    main()
