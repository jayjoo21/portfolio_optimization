from __future__ import annotations

import argparse
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
# 05A5-H3. Non-Corrected Selective Full-Statement Fallback
#
# 목적
# ------------------------------------------------------------
# H2에서 fallback 대상으로 확정된 receipt만
# OpenDART "단일회사 전체 재무제표(fnlttSinglAcntAll)"로 조회한다.
#
# 핵심 원칙
# ------------------------------------------------------------
# 1) H2 fallback target만 호출한다.
# 2) 기존 selected_fs_div가 CFS/OFS면 그 구분을 우선 조회한다.
# 3) H1 no_data처럼 fs_div가 없으면 CFS -> OFS 순서로 시도한다.
# 4) primary fs_div가 013(no data)일 때만 alternate fs_div를 시도한다.
#    CFS가 존재하는데 metric coverage가 부족하다고 OFS로 섞지 않는다.
# 5) API rcept_no가 expected rcept_no와 정확히 일치해야 PIT verified.
# 6) raw rows를 먼저 보존한 뒤 account_id + account_nm 기반으로
#    핵심 metric을 선택한다.
# 7) actual available_date는 아직 결정하지 않는다.
#
# 공식 API
# ------------------------------------------------------------
# GET https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json
# required:
#   crtfc_key, corp_code, bsns_year, reprt_code, fs_div
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_full_fallback_rows.parquet/csv
#   dart_noncorrected_full_fallback_manifest.parquet/csv
#   dart_noncorrected_full_fallback_selected.parquet/csv
#
# 실행
# ------------------------------------------------------------
# 테스트:
#   python scripts\05a5h3_dart_noncorrected_full_fallback.py --limit 20
#
# 전체:
#   python scripts\05a5h3_dart_noncorrected_full_fallback.py --full
#
# 오류 재시도:
#   python scripts\05a5h3_dart_noncorrected_full_fallback.py --retry-errors --full
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TARGET_PATH = (
    INTERIM
    / "dart_noncorrected_selective_fallback_targets.parquet"
)

ROWS_PARQUET = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)
ROWS_CSV = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.csv"
)

MANIFEST_PARQUET = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)
MANIFEST_CSV = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.csv"
)

SELECTED_PARQUET = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)
SELECTED_CSV = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.csv"
)

API_URL = (
    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
)

REQUEST_SLEEP_SECONDS = 0.18
MAX_RETRIES = 4
CHECKPOINT_EVERY = 25


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]

EXTRA_METRICS = [
    "equity_parent",
    "net_income_parent_cumulative",
]

ALL_METRICS = (
    CORE_METRICS
    + EXTRA_METRICS
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("\u3000", "")
    )


def canonical_account_name(value: Any) -> str:
    """
    재무제표 표시용 prefix를 제거한 보수적 계정명.

    예:
      "VIII. 당기순이익" -> "당기순이익"
      "I. 매출액"       -> "매출액"

    계정명 중간/뒤의 단어는 건드리지 않고,
    맨 앞의 로마숫자/숫자 + 구분기호만 제거한다.
    """
    text = normalize_text(value)

    # Roman numeral prefix: I., II., VIII., Ⅷ. 등
    text = re.sub(
        r"^(?:[IVXLCDM]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)[\.\)\-:]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Arabic numbering prefix: 1., 2), 3- 등
    text = re.sub(
        r"^\d+[\.\)\-:]*",
        "",
        text,
    )

    return text


def normalize_account_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(":", "_")
    )


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {
        "",
        "-",
        "nan",
        "None",
        "<NA>",
    }:
        return np.nan

    text = text.replace(
        ",",
        "",
    )

    if (
        text.startswith("(")
        and text.endswith(")")
    ):
        text = (
            "-"
            + text[1:-1]
        )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def api_value(
    row: pd.Series,
    metric: str,
) -> float:

    if metric in {
        "assets",
        "liabilities",
        "equity_total",
        "equity_parent",
    }:
        return parse_number(
            row.get(
                "thstrm_amount"
            )
        )

    cumulative = parse_number(
        row.get(
            "thstrm_add_amount"
        )
    )

    if pd.notna(
        cumulative
    ):
        return cumulative

    return parse_number(
        row.get(
            "thstrm_amount"
        )
    )


# ------------------------------------------------------------
# Conservative account classification
# ------------------------------------------------------------

def classify_account(
    row: pd.Series,
) -> tuple[
    str | None,
    int,
    str | None,
]:

    account_id = normalize_account_id(
        row.get(
            "account_id"
        )
    )

    account_nm = canonical_account_name(
        row.get(
            "account_nm"
        )
    )

    sj_div = normalize_text(
        row.get(
            "sj_div"
        )
    ).upper()

    # --------------------------------------------------------
    # BS stock variables
    # --------------------------------------------------------

    if sj_div == "BS":

        asset_ids = {
            "ifrs-full_assets",
            "ifrs_assets",
        }

        if (
            any(
                account_id.endswith(
                    token
                )
                for token in asset_ids
            )
            or account_nm == "자산총계"
        ):
            method = (
                "account_id"
                if any(
                    account_id.endswith(
                        token
                    )
                    for token in asset_ids
                )
                else "exact_account_name"
            )
            return (
                "assets",
                100 if method == "account_id" else 80,
                method,
            )

        liability_ids = {
            "ifrs-full_liabilities",
            "ifrs_liabilities",
        }

        if (
            any(
                account_id.endswith(
                    token
                )
                for token in liability_ids
            )
            or account_nm == "부채총계"
        ):
            method = (
                "account_id"
                if any(
                    account_id.endswith(
                        token
                    )
                    for token in liability_ids
                )
                else "exact_account_name"
            )
            return (
                "liabilities",
                100 if method == "account_id" else 80,
                method,
            )

        equity_ids = {
            "ifrs-full_equity",
            "ifrs_equity",
        }

        if (
            any(
                account_id.endswith(
                    token
                )
                for token in equity_ids
            )
            or account_nm == "자본총계"
        ):
            method = (
                "account_id"
                if any(
                    account_id.endswith(
                        token
                    )
                    for token in equity_ids
                )
                else "exact_account_name"
            )
            return (
                "equity_total",
                100 if method == "account_id" else 80,
                method,
            )

        if (
            "equityattributabletoownersofparent"
            in account_id.replace(
                "_",
                "",
            )
            or account_nm
            in {
                "지배기업소유주지분",
                "지배기업의소유주에게귀속되는자본",
                "지배기업소유주에게귀속되는자본",
            }
        ):
            method = (
                "account_id"
                if "equityattributabletoownersofparent"
                in account_id.replace(
                    "_",
                    "",
                )
                else "exact_account_name"
            )
            return (
                "equity_parent",
                100 if method == "account_id" else 75,
                method,
            )

    # --------------------------------------------------------
    # IS / CIS flow variables
    # --------------------------------------------------------

    if sj_div in {
        "IS",
        "CIS",
    }:

        revenue_id_tokens = {
            "ifrs-full_revenue",
            "ifrs_revenue",
            "ifrs-full_revenuefromcontractswithcustomers",
            "ifrs_revenuefromcontractswithcustomers",
            "dart_operatingrevenue",
        }

        if (
            any(
                account_id.endswith(
                    token
                )
                for token in revenue_id_tokens
            )
            or account_nm
            in {
                "매출액",
                "매출",
                "수익(매출액)",
                "영업수익",
                "영업수익(매출액)",
            }
        ):
            method = (
                "account_id"
                if any(
                    account_id.endswith(
                        token
                    )
                    for token in revenue_id_tokens
                )
                else "exact_account_name"
            )
            return (
                "revenue_cumulative",
                100 if method == "account_id" else 75,
                method,
            )

        if (
            "operatingincomeloss"
            in account_id.replace(
                "_",
                "",
            )
            or account_nm
            in {
                "영업이익",
                "영업이익(손실)",
                "영업손익",
            }
        ):
            method = (
                "account_id"
                if "operatingincomeloss"
                in account_id.replace(
                    "_",
                    "",
                )
                else "exact_account_name"
            )
            return (
                "operating_income_cumulative",
                100 if method == "account_id" else 80,
                method,
            )

        # Parent-attributable must be tested before generic ProfitLoss.
        if (
            "profitlossattributabletoownersofparent"
            in account_id.replace(
                "_",
                "",
            )
            or account_nm
            in {
                "지배기업소유주지분순이익",
                "지배기업의소유주에게귀속되는당기순이익",
                "지배기업소유주에게귀속되는당기순이익",
                "지배기업소유주지분에귀속되는당기순이익",
            }
        ):
            method = (
                "account_id"
                if "profitlossattributabletoownersofparent"
                in account_id.replace(
                    "_",
                    "",
                )
                else "exact_account_name"
            )
            return (
                "net_income_parent_cumulative",
                100 if method == "account_id" else 75,
                method,
            )

        total_profit_ids = {
            "ifrs-full_profitloss",
            "ifrs_profitloss",
        }

        if (
            any(
                account_id.endswith(
                    token
                )
                for token in total_profit_ids
            )
            or account_nm
            in {
                "당기순이익",
                "당기순이익(손실)",
                "분기순이익",
                "분기순이익(손실)",
                "반기순이익",
                "반기순이익(손실)",
                "연결당기순이익",
                "연결당기순이익(손실)",
                "연결분기순이익",
                "연결분기순이익(손실)",
                "연결반기순이익",
                "연결반기순이익(손실)",
                "당기순손익",
                "분기순손익",
                "반기순손익",
                "연결당기순손익",
                "연결분기순손익",
                "연결반기순손익",
            }
        ):
            method = (
                "account_id"
                if any(
                    account_id.endswith(
                        token
                    )
                    for token in total_profit_ids
                )
                else "exact_account_name"
            )
            return (
                "net_income_total_cumulative",
                100 if method == "account_id" else 80,
                method,
            )

    return (
        None,
        0,
        None,
    )


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

def request_full_statement(
    session: requests.Session,
    api_key: str,
    corp_code: str,
    bsns_year: int,
    reprt_code: str,
    fs_div: str,
) -> dict[str, Any]:

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):
        try:
            response = session.get(
                API_URL,
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(
                        bsns_year
                    ),
                    "reprt_code": str(
                        reprt_code
                    ),
                    "fs_div": fs_div,
                },
                timeout=90,
            )

            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as exc:
                preview = (
                    response.text[:300]
                    if response.text
                    else ""
                )

                raise RuntimeError(
                    "non_json_response | "
                    f"content_type={response.headers.get('Content-Type')} | "
                    f"preview={preview!r} | "
                    f"{type(exc).__name__}: {exc}"
                )

            status = str(
                payload.get(
                    "status",
                    "",
                )
            )

            message = payload.get(
                "message"
            )

            if status == "000":
                return {
                    "status": status,
                    "message": message,
                    "rows": payload.get(
                        "list",
                        [],
                    ),
                    "error": None,
                }

            if status in {
                "013",
                "014",
            }:
                return {
                    "status": status,
                    "message": message,
                    "rows": [],
                    "error": None,
                }

            if status == "020":
                raise RuntimeError(
                    "OpenDART request limit exceeded (020)"
                )

            return {
                "status": status,
                "message": message,
                "rows": [],
                "error": (
                    f"OpenDART status={status}: {message}"
                ),
            }

        except Exception as exc:
            last_error = exc

            if (
                isinstance(
                    exc,
                    RuntimeError,
                )
                and "020"
                in str(
                    exc
                )
            ):
                raise

            wait = min(
                2 ** attempt,
                12,
            )

            print(
                f"    retry {attempt + 1}/{MAX_RETRIES} "
                f"after {wait}s: {exc}"
            )

            time.sleep(
                wait
            )

    return {
        "status": None,
        "message": None,
        "rows": [],
        "error": repr(
            last_error
        ),
    }


# ------------------------------------------------------------
# Raw row normalization
# ------------------------------------------------------------

def normalize_api_rows(
    raw_rows: list[
        dict[str, Any]
    ],
    target: pd.Series,
    requested_fs_div: str,
) -> pd.DataFrame:

    if not raw_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        raw_rows
    )

    df[
        "_expected_rcept_no"
    ] = str(
        target[
            "rcept_no"
        ]
    )

    df[
        "_requested_fs_div"
    ] = requested_fs_div

    df[
        "_fallback_reason"
    ] = target.get(
        "fallback_reason"
    )

    for col in [
        "rcept_no",
        "corp_code",
        "reprt_code",
        "bsns_year",
        "sj_div",
        "sj_nm",
        "account_id",
        "account_nm",
        "account_detail",
        "thstrm_nm",
        "thstrm_amount",
        "thstrm_add_amount",
        "frmtrm_nm",
        "frmtrm_amount",
        "frmtrm_q_nm",
        "frmtrm_q_amount",
        "frmtrm_add_amount",
        "bfefrmtrm_nm",
        "bfefrmtrm_amount",
        "ord",
        "currency",
    ]:
        if col in df.columns:
            df[
                col
            ] = df[
                col
            ].astype(
                "string"
            )

    if "rcept_no" in df.columns:
        df[
            "rcept_no"
        ] = (
            df[
                "rcept_no"
            ]
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )

    return df


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------

def select_metrics(
    rows: pd.DataFrame,
) -> pd.DataFrame:

    if rows.empty:
        return pd.DataFrame()

    work = rows.copy()

    classified = work.apply(
        classify_account,
        axis=1,
        result_type="expand",
    )

    classified.columns = [
        "metric",
        "selection_score",
        "selection_method",
    ]

    work = pd.concat(
        [
            work,
            classified,
        ],
        axis=1,
    )

    work = work.loc[
        work[
            "metric"
        ].notna()
    ].copy()

    if work.empty:
        return pd.DataFrame()

    work[
        "metric_value"
    ] = work.apply(
        lambda r:
        api_value(
            r,
            r[
                "metric"
            ],
        ),
        axis=1,
    )

    work = work.loc[
        work[
            "metric_value"
        ].notna()
    ].copy()

    # Prefer IS over CIS for flow accounts when both carry same concept.
    work[
        "_sj_priority"
    ] = work[
        "sj_div"
    ].map(
        {
            "BS": 0,
            "IS": 0,
            "CIS": 1,
            "CF": 2,
            "SCE": 3,
        }
    ).fillna(
        9
    )

    work[
        "_ord_num"
    ] = pd.to_numeric(
        work.get(
            "ord",
            pd.Series(
                index=work.index,
                dtype="object",
            ),
        ),
        errors="coerce",
    ).fillna(
        999999
    )

    work = work.sort_values(
        [
            "_expected_rcept_no",
            "metric",
            "selection_score",
            "_sj_priority",
            "_ord_num",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
    )

    selected = (
        work.drop_duplicates(
            subset=[
                "_expected_rcept_no",
                "metric",
            ],
            keep="first",
        )
        .copy()
    )

    keep_cols = [
        "_expected_rcept_no",
        "_requested_fs_div",
        "_fallback_reason",
        "rcept_no",
        "corp_code",
        "bsns_year",
        "reprt_code",
        "sj_div",
        "sj_nm",
        "metric",
        "metric_value",
        "selection_score",
        "selection_method",
        "account_id",
        "account_nm",
        "thstrm_nm",
        "thstrm_amount",
        "thstrm_add_amount",
        "currency",
    ]

    keep_cols = [
        c
        for c in keep_cols
        if c in selected.columns
    ]

    return selected[
        keep_cols
    ].reset_index(
        drop=True
    )


# ------------------------------------------------------------
# Resume / save
# ------------------------------------------------------------

def load_existing():

    if MANIFEST_PARQUET.exists():
        manifest = pd.read_parquet(
            MANIFEST_PARQUET
        )
    else:
        manifest = pd.DataFrame()

    if ROWS_PARQUET.exists():
        rows = pd.read_parquet(
            ROWS_PARQUET
        )
    else:
        rows = pd.DataFrame()

    return (
        manifest,
        rows,
    )


def save_state(
    manifest: pd.DataFrame,
    rows: pd.DataFrame,
):

    if not manifest.empty:
        manifest = (
            manifest.sort_values(
                [
                    "rcept_no",
                "attempted_at",
            ]
        )
        .drop_duplicates(
            subset=[
                "rcept_no",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if not rows.empty:
        rows = (
            rows.drop_duplicates()
            .reset_index(
                drop=True
            )
        )

    rows.to_parquet(
        ROWS_PARQUET,
        index=False,
    )
    rows.to_csv(
        ROWS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    manifest.to_parquet(
        MANIFEST_PARQUET,
        index=False,
    )
    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    selected = select_metrics(
        rows
    )

    selected.to_parquet(
        SELECTED_PARQUET,
        index=False,
    )
    selected.to_csv(
        SELECTED_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# ------------------------------------------------------------
# Target processing
# ------------------------------------------------------------

def process_target(
    session: requests.Session,
    api_key: str,
    target: pd.Series,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:

    expected_rcept_no = str(
        target[
            "rcept_no"
        ]
    )

    selected_fs = target.get(
        "selected_fs_div"
    )

    if pd.notna(
        selected_fs
    ) and str(
        selected_fs
    ) in {
        "CFS",
        "OFS",
    }:
        fs_attempts = [
            str(
                selected_fs
            )
        ]
    else:
        fs_attempts = [
            "CFS",
            "OFS",
        ]

    all_rows = []

    chosen_fs = None
    final_status = None
    final_message = None
    final_error = None
    returned_receipts = []

    for idx, fs_div in enumerate(
        fs_attempts
    ):

        result = request_full_statement(
            session=session,
            api_key=api_key,
            corp_code=str(
                target[
                    "corp_code"
                ]
            ).zfill(
                8
            ),
            bsns_year=int(
                target[
                    "bsns_year"
                ]
            ),
            reprt_code=str(
                target[
                    "reprt_code"
                ]
            ),
            fs_div=fs_div,
        )

        final_status = result[
            "status"
        ]
        final_message = result[
            "message"
        ]
        final_error = result[
            "error"
        ]

        api_rows = normalize_api_rows(
            result[
                "rows"
            ],
            target,
            fs_div,
        )

        if not api_rows.empty:

            if "rcept_no" in api_rows.columns:
                returned_receipts = sorted(
                    set(
                        api_rows[
                            "rcept_no"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )

            matched = api_rows.loc[
                api_rows[
                    "rcept_no"
                ]
                .astype(str)
                .eq(
                    expected_rcept_no
                )
            ].copy()

            if not matched.empty:
                chosen_fs = fs_div
                all_rows.append(
                    matched
                )
                final_status = "000"
                final_error = None
                break

            # Data exists but receipt mismatch -> do not alternate silently.
            chosen_fs = fs_div
            break

        # Only no-data status may fall through to alternate FS.
        if (
            result[
                "status"
            ]
            not in {
                "013",
                "014",
            }
        ):
            break

        # If existing H2 selected fs_div was known, do not mix alternate.
        if pd.notna(
            selected_fs
        ):
            break

    combined_rows = (
        pd.concat(
            all_rows,
            ignore_index=True,
        )
        if all_rows
        else pd.DataFrame()
    )

    receipt_match = (
        bool(
            chosen_fs is not None
            and not combined_rows.empty
        )
    )

    if final_error is not None:
        collection_status = (
            "api_error"
        )
        pit_verified = False

    elif (
        returned_receipts
        and expected_rcept_no
        not in returned_receipts
    ):
        collection_status = (
            "receipt_mismatch"
        )
        pit_verified = False

    elif receipt_match:
        collection_status = (
            "available"
        )
        pit_verified = True

    else:
        collection_status = (
            "no_data"
        )
        pit_verified = False

    record = {
        "rcept_no": expected_rcept_no,
        "stock_code": str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        ),
        "corp_code": str(
            target[
                "corp_code"
            ]
        ).zfill(
            8
        ),
        "corp_name": target.get(
            "corp_name"
        ),
        "canonical_period_key": target.get(
            "canonical_period_key"
        ),
        "rcept_dt": target.get(
            "rcept_dt"
        ),
        "bsns_year": int(
            target[
                "bsns_year"
            ]
        ),
        "reprt_code": str(
            target[
                "reprt_code"
            ]
        ),
        "fallback_reason": target.get(
            "fallback_reason"
        ),
        "missing_core_metrics_before": target.get(
            "missing_core_metrics"
        ),
        "selected_fs_div_before": target.get(
            "selected_fs_div"
        ),
        "chosen_fs_div": chosen_fs,
        "collection_status": collection_status,
        "pit_verified": pit_verified,
        "returned_rcept_nos": (
            " | ".join(
                returned_receipts
            )
        ),
        "api_status": final_status,
        "api_message": final_message,
        "api_error": final_error,
        "api_row_count": int(
            len(
                combined_rows
            )
        ),
        "attempted_at": pd.Timestamp.utcnow(),
    }

    return (
        record,
        combined_rows,
    )


# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
    )

    parser.add_argument(
        "--rebuild-selected",
        action="store_true",
        help=(
            "API 호출 없이 기존 raw fallback rows에서 "
            "selector 결과만 다시 생성"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            TARGET_PATH
        )

    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "DART_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            ".env의 DART_API_KEY가 없습니다."
        )

    targets = pd.read_parquet(
        TARGET_PATH
    )

    targets[
        "rcept_no"
    ] = targets[
        "rcept_no"
    ].astype(str)

    manifest_state, row_state = (
        load_existing()
    )

    if args.rebuild_selected:
        if row_state.empty:
            raise RuntimeError(
                "기존 H3 raw fallback rows가 없습니다."
            )

        selected = select_metrics(
            row_state
        )

        selected.to_parquet(
            SELECTED_PARQUET,
            index=False,
        )
        selected.to_csv(
            SELECTED_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n"
            + "=" * 80
        )
        print(
            "05A5-H3 SELECTOR REBUILD"
        )
        print(
            "=" * 80
        )

        print(
            f"\nraw rows : {len(row_state):,}"
        )
        print(
            f"selected : {len(selected):,}"
        )

        if not selected.empty:
            wide = selected.pivot_table(
                index="_expected_rcept_no",
                columns="metric",
                values="metric_value",
                aggfunc="first",
            )

            for metric in ALL_METRICS:
                if metric not in wide.columns:
                    wide[metric] = np.nan

            print(
                "\n[Metric coverage among processed receipts]"
            )

            processed_available = int(
                manifest_state[
                    "collection_status"
                ].eq(
                    "available"
                )
                .sum()
            ) if not manifest_state.empty else wide.shape[0]

            for metric in ALL_METRICS:
                print(
                    f"{metric:<36} "
                    f"{int(wide[metric].notna().sum()):>6,} "
                    f"/ {processed_available:,} available"
                )

            core_count = (
                wide[
                    CORE_METRICS
                ]
                .notna()
                .sum(
                    axis=1
                )
            )

            print(
                "\n[Core metric count]"
            )
            print(
                core_count.value_counts()
                .sort_index(
                    ascending=False
                )
                .to_string()
            )

        print(
            f"\nSelected: {SELECTED_PARQUET}"
        )
        return

    done = set()

    retry = set()

    if not manifest_state.empty:
        manifest_state[
            "rcept_no"
        ] = manifest_state[
            "rcept_no"
        ].astype(str)

        if args.retry_errors:
            retry = set(
                manifest_state.loc[
                    manifest_state[
                        "collection_status"
                    ].isin(
                        [
                            "api_error",
                            "receipt_mismatch",
                        ]
                    ),
                    "rcept_no",
                ]
                .astype(str)
            )
        else:
            done = set(
                manifest_state.loc[
                    manifest_state[
                        "collection_status"
                    ].isin(
                        [
                            "available",
                            "no_data",
                        ]
                    ),
                    "rcept_no",
                ]
                .astype(str)
            )

    if args.retry_errors:
        work = targets.loc[
            targets[
                "rcept_no"
            ].isin(
                retry
            )
        ].copy()
    else:
        work = targets.loc[
            ~targets[
                "rcept_no"
            ].isin(
                done
            )
        ].copy()

    work = work.sort_values(
        [
            "bsns_year",
            "reprt_code",
            "stock_code",
        ]
    ).reset_index(
        drop=True
    )

    if not args.full:
        work = work.head(
            args.limit
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-H3 NON-CORRECTED SELECTIVE FULL-STATEMENT FALLBACK"
    )
    print(
        "=" * 80
    )

    print(
        f"\nfallback targets total : "
        f"{len(targets):,}"
    )
    print(
        f"targets this run       : "
        f"{len(work):,}"
    )

    if work.empty:
        print(
            "\n처리할 대상이 없습니다."
        )
        return

    session = requests.Session()

    for idx, (_, target) in enumerate(
        work.iterrows(),
        start=1,
    ):

        print(
            f"\n[{idx}/{len(work)}] "
            f"{target['stock_code']} | "
            f"{target['canonical_period_key']} | "
            f"{target['rcept_no']} | "
            f"reason={target['fallback_reason']} | "
            f"fs={target.get('selected_fs_div')}"
        )

        try:
            record, new_rows = (
                process_target(
                    session,
                    api_key,
                    target,
                )
            )

        except RuntimeError as exc:
            if "020" in str(
                exc
            ):
                save_state(
                    manifest_state,
                    row_state,
                )

                print(
                    "\nOpenDART 요청 제한(020). checkpoint 저장 후 종료."
                )
                raise

            raise

        print(
            f"  -> {record['collection_status']} "
            f"| chosen_fs={record['chosen_fs_div']} "
            f"| rows={record['api_row_count']}"
        )

        touched = str(
            record[
                "rcept_no"
            ]
        )

        if not manifest_state.empty:
            manifest_state = (
                manifest_state.loc[
                    ~manifest_state[
                        "rcept_no"
                    ]
                    .astype(str)
                    .eq(
                        touched
                    )
                ]
            )

        manifest_state = pd.concat(
            [
                manifest_state,
                pd.DataFrame(
                    [
                        record
                    ]
                ),
            ],
            ignore_index=True,
        )

        if not new_rows.empty:

            if not row_state.empty:
                row_state = (
                    row_state.loc[
                        ~row_state[
                            "_expected_rcept_no"
                        ]
                        .astype(str)
                        .eq(
                            touched
                        )
                    ]
                )

            row_state = pd.concat(
                [
                    row_state,
                    new_rows,
                ],
                ignore_index=True,
            )

        if (
            idx
            % CHECKPOINT_EVERY
            == 0
        ):
            save_state(
                manifest_state,
                row_state,
            )

            print(
                "  checkpoint saved"
            )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    save_state(
        manifest_state,
        row_state,
    )

    selected = (
        pd.read_parquet(
            SELECTED_PARQUET
        )
        if SELECTED_PARQUET.exists()
        else pd.DataFrame()
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "H3 FALLBACK SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        "\n[Collection status]"
    )
    print(
        manifest_state[
            "collection_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Chosen FS]"
    )
    print(
        manifest_state[
            "chosen_fs_div"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if not selected.empty:
        wide = (
            selected.pivot_table(
                index="_expected_rcept_no",
                columns="metric",
                values="metric_value",
                aggfunc="first",
            )
        )

        for metric in ALL_METRICS:
            if metric not in wide.columns:
                wide[
                    metric
                ] = np.nan

        print(
            "\n[Recovered metric coverage among processed available receipts]"
        )

        processed_available = int(
            manifest_state[
                "collection_status"
            ].eq(
                "available"
            )
            .sum()
        )

        for metric in ALL_METRICS:
            print(
                f"{metric:<36} "
                f"{int(wide[metric].notna().sum()):>6,} "
                f"/ {processed_available:,} available"
            )

        core_count = (
            wide[
                CORE_METRICS
            ]
            .notna()
            .sum(
                axis=1
            )
        )

        print(
            "\n[Recovered core metric count]"
        )
        print(
            core_count.value_counts()
            .sort_index(
                ascending=False
            )
            .to_string()
        )

    remaining_errors = int(
        manifest_state[
            "collection_status"
        ]
        .eq(
            "api_error"
        )
        .sum()
    )

    mismatches = int(
        manifest_state[
            "collection_status"
        ]
        .eq(
            "receipt_mismatch"
        )
        .sum()
    )

    print(
        f"\nRemaining api_error      : "
        f"{remaining_errors:,}"
    )
    print(
        f"Remaining receipt mismatch: "
        f"{mismatches:,}"
    )

    print(
        f"\nRows     : {ROWS_PARQUET}"
    )
    print(
        f"Manifest : {MANIFEST_PARQUET}"
    )
    print(
        f"Selected : {SELECTED_PARQUET}"
    )

    print(
        "\n다음 단계:"
        "\n05A5-H4 major-account + full-fallback merge QA"
        "\n→ fallback 전/후 core coverage 비교"
        "\n→ corrected reconciled와 합칠 non-corrected snapshot 확정"
    )


if __name__ == "__main__":
    main()
