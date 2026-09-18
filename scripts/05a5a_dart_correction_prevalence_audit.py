from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-A. OpenDART correction prevalence audit
#
# 목적
# ------------------------------------------------------------
# 1) 과거 KRX300 구성종목 합집합 -> DART corp_code mapping coverage 확인
# 2) 2018+ 정기보고서(분기/반기/사업) 제출이력 수집
# 3) 정정보고서가 얼마나 자주 발생하는지 측정
# 4) 원본 -> 정정 chain을 접수번호/접수일 기준으로 보존
#
# 이 단계에서는 재무제표 숫자를 수집하지 않는다.
# 먼저 strict PIT(XBRL 원본 스냅샷) 필요성을 판단한다.
#
# 실행
# ------------------------------------------------------------
# 테스트:
#   python scripts\05a5a_dart_correction_prevalence_audit.py --sample 20
#
# 전체:
#   python scripts\05a5a_dart_correction_prevalence_audit.py --full
#
# .env
#   DART_API_KEY=...
# ============================================================


BASE_URL = "https://opendart.fss.or.kr/api"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "krx"
    / "universe"
    / "krx300_monthly_membership_snapshots.csv"
)

CORP_MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "master"
    / "dart_corp_codes.parquet"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_OUT = (
    OUT_DIR
    / "dart_historical_universe_mapping_audit.csv"
)

FILINGS_OUT = (
    OUT_DIR
    / "dart_regular_filing_history_2018plus.parquet"
)

SUMMARY_OUT = (
    OUT_DIR
    / "dart_correction_prevalence_summary.csv"
)

CHAINS_OUT = (
    OUT_DIR
    / "dart_correction_chains.csv"
)


load_dotenv(PROJECT_ROOT / ".env")

DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise RuntimeError(
        "DART_API_KEY가 없습니다. 프로젝트 루트 .env를 확인하세요."
    )


START_DATE = "20180101"
END_DATE = "20260919"

REQUEST_SLEEP_SECONDS = 0.16


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def detect_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_ticker(value: Any) -> str:
    value = str(value).strip()
    value = re.sub(r"\.0$", "", value)
    return value.zfill(6)


def normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        "",
        str(value).strip(),
    )


def normalize_report_name(
    report_nm: str,
) -> str:
    """
    [기재정정], [첨부정정] 등의 prefix를 제거하여
    동일 보고기간의 원본/정정본을 같은 chain으로 묶는다.
    """
    text = str(report_nm).strip()

    # 정정 관련 [] prefix를 반복 제거
    while True:
        updated = re.sub(
            r"^\[[^\]]*정정[^\]]*\]\s*",
            "",
            text,
        )
        if updated == text:
            break
        text = updated

    # 남은 whitespace 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify_report(
    report_nm: str,
) -> str | None:
    base = normalize_report_name(
        report_nm
    )

    if "사업보고서" in base:
        return "FY"
    if "반기보고서" in base:
        return "H1"
    if "분기보고서" in base:
        # Q1/Q3는 이름 안의 기준월로 분류
        match = re.search(
            r"\((\d{4})\.(\d{2})\)",
            base,
        )
        if match:
            month = int(match.group(2))
            if month <= 3:
                return "Q1"
            if month >= 9:
                return "Q3"
        return "Q"
    return None


def extract_period_key(
    report_nm: str,
) -> str:
    """
    예:
      사업보고서 (2018.12)
      [기재정정]사업보고서 (2018.12)
    -> 사업보고서 (2018.12)
    """
    return normalize_report_name(
        report_nm
    )


# ------------------------------------------------------------
# Universe -> DART mapping
# ------------------------------------------------------------

def load_historical_universe() -> pd.DataFrame:
    if not MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"membership 파일이 없습니다: {MEMBERSHIP_PATH}"
        )

    df = pd.read_csv(
        MEMBERSHIP_PATH,
        dtype=str,
    )

    ticker_col = detect_column(
        df,
        [
            "ticker",
            "code",
            "symbol",
            "stock_code",
            "종목코드",
            "단축코드",
        ],
    )

    if ticker_col is None:
        raise RuntimeError(
            "membership CSV에서 종목코드 column을 찾지 못했습니다.\n"
            f"columns={df.columns.tolist()}"
        )

    name_col = detect_column(
        df,
        [
            "name",
            "corp_name",
            "stock_name",
            "종목명",
            "한글종목명",
        ],
    )

    out = pd.DataFrame(
        {
            "stock_code": (
                df[ticker_col]
                .dropna()
                .map(normalize_ticker)
            )
        }
    )

    if name_col is not None:
        temp = df[
            [
                ticker_col,
                name_col,
            ]
        ].dropna(
            subset=[ticker_col]
        ).copy()

        temp["stock_code"] = temp[
            ticker_col
        ].map(
            normalize_ticker
        )

        temp["historical_name"] = (
            temp[name_col]
            .astype(str)
            .str.strip()
        )

        # 동일 ticker에 여러 이름이 있으면 마지막 비어있지 않은 이름
        name_map = (
            temp.loc[
                temp["historical_name"]
                .ne("")
            ]
            .drop_duplicates(
                subset=["stock_code"],
                keep="last",
            )[
                [
                    "stock_code",
                    "historical_name",
                ]
            ]
        )

        out = (
            out
            .drop_duplicates()
            .merge(
                name_map,
                on="stock_code",
                how="left",
            )
        )
    else:
        out = out.drop_duplicates()
        out["historical_name"] = None

    return (
        out
        .sort_values("stock_code")
        .reset_index(drop=True)
    )


def map_to_dart(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    if not CORP_MASTER_PATH.exists():
        raise FileNotFoundError(
            "DART corp master가 없습니다.\n"
            "먼저 05a5_opendart_fundamental_pit_audit.py를 실행하세요.\n"
            f"expected: {CORP_MASTER_PATH}"
        )

    corp = pd.read_parquet(
        CORP_MASTER_PATH
    ).copy()

    corp["stock_code"] = (
        corp["stock_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(
            lambda x: normalize_ticker(x)
            if x
            else ""
        )
    )

    corp["corp_name_norm"] = (
        corp["corp_name"]
        .map(normalize_name)
    )

    mapped_rows = []

    by_stock = (
        corp.loc[
            corp["stock_code"].ne("")
        ]
        .drop_duplicates(
            subset=["stock_code"],
            keep="last",
        )
        .set_index("stock_code")
    )

    # 이름 fallback은 exact-normalized match만 허용
    name_counts = (
        corp.loc[
            corp["corp_name_norm"].ne("")
        ]["corp_name_norm"]
        .value_counts()
    )

    unique_name = (
        corp.loc[
            corp["corp_name_norm"].isin(
                name_counts[
                    name_counts.eq(1)
                ].index
            )
        ]
        .set_index("corp_name_norm")
    )

    for row in universe.itertuples(
        index=False
    ):
        stock_code = row.stock_code
        historical_name = getattr(
            row,
            "historical_name",
            None,
        )

        if stock_code in by_stock.index:
            hit = by_stock.loc[
                stock_code
            ]

            mapped_rows.append(
                {
                    "stock_code": stock_code,
                    "historical_name": historical_name,
                    "corp_code": hit["corp_code"],
                    "dart_corp_name": hit["corp_name"],
                    "mapping_method": "stock_code",
                    "mapped": True,
                }
            )
            continue

        name_norm = normalize_name(
            historical_name
        )

        if (
            name_norm
            and name_norm in unique_name.index
        ):
            hit = unique_name.loc[
                name_norm
            ]

            mapped_rows.append(
                {
                    "stock_code": stock_code,
                    "historical_name": historical_name,
                    "corp_code": hit["corp_code"],
                    "dart_corp_name": hit["corp_name"],
                    "mapping_method": "exact_name_fallback",
                    "mapped": True,
                }
            )
            continue

        mapped_rows.append(
            {
                "stock_code": stock_code,
                "historical_name": historical_name,
                "corp_code": None,
                "dart_corp_name": None,
                "mapping_method": "unmatched",
                "mapped": False,
            }
        )

    return pd.DataFrame(
        mapped_rows
    )


# ------------------------------------------------------------
# DART filing list
# ------------------------------------------------------------

def request_filing_page(
    session: requests.Session,
    corp_code: str,
    page_no: int,
) -> dict[str, Any]:

    response = session.get(
        f"{BASE_URL}/list.json",
        params={
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bgn_de": START_DATE,
            "end_de": END_DATE,
            "pblntf_ty": "A",       # 정기공시
            "last_reprt_at": "N",   # 정정 포함 전체
            "sort": "date",
            "sort_mth": "asc",
            "page_no": page_no,
            "page_count": 100,
        },
        timeout=45,
    )

    response.raise_for_status()

    payload = response.json()

    status = str(
        payload.get("status", "")
    )

    if status not in (
        "000",
        "013",
    ):
        raise RuntimeError(
            f"list.json status={status} "
            f"message={payload.get('message')}"
        )

    return payload


def collect_company_filings(
    session: requests.Session,
    row: pd.Series,
) -> list[dict[str, Any]]:

    corp_code = str(
        row["corp_code"]
    )

    collected = []
    page_no = 1

    while True:
        payload = request_filing_page(
            session,
            corp_code,
            page_no,
        )

        if payload.get("status") == "013":
            break

        page_rows = (
            payload.get("list")
            or []
        )

        for item in page_rows:
            report_nm = str(
                item.get(
                    "report_nm",
                    "",
                )
            )

            report_type = classify_report(
                report_nm
            )

            # 재무정보에 직접 대응하는 분기/반기/사업보고서만 남김
            if report_type is None:
                continue

            collected.append(
                {
                    "stock_code": row["stock_code"],
                    "corp_code": corp_code,
                    "corp_name": row["dart_corp_name"],
                    "rcept_no": item.get("rcept_no"),
                    "rcept_dt": item.get("rcept_dt"),
                    "report_nm": report_nm,
                    "report_type": report_type,
                    "period_key": extract_period_key(
                        report_nm
                    ),
                    "is_correction": (
                        "정정" in report_nm
                    ),
                }
            )

        total_page = int(
            payload.get(
                "total_page",
                1,
            )
        )

        if page_no >= total_page:
            break

        page_no += 1

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    return collected


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def build_summary(
    mapped: pd.DataFrame,
    filings: pd.DataFrame,
) -> pd.DataFrame:

    base = mapped.copy()

    mapped_only = base.loc[
        base["mapped"]
    ].copy()

    if filings.empty:
        mapped_only["periods"] = 0
        mapped_only["correction_periods"] = 0
        mapped_only["correction_filings"] = 0
        mapped_only["correction_period_rate"] = 0.0
        return mapped_only

    grouped = (
        filings
        .groupby(
            [
                "stock_code",
                "period_key",
            ]
        )
        .agg(
            filing_count=(
                "rcept_no",
                "size",
            ),
            correction_filings=(
                "is_correction",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped[
        "period_has_correction"
    ] = (
        grouped[
            "correction_filings"
        ] > 0
    )

    company = (
        grouped
        .groupby("stock_code")
        .agg(
            periods=(
                "period_key",
                "size",
            ),
            correction_periods=(
                "period_has_correction",
                "sum",
            ),
            correction_filings=(
                "correction_filings",
                "sum",
            ),
        )
        .reset_index()
    )

    company[
        "correction_period_rate"
    ] = (
        company[
            "correction_periods"
        ]
        / company[
            "periods"
        ].replace(0, pd.NA)
    )

    result = mapped_only.merge(
        company,
        on="stock_code",
        how="left",
    )

    for col in [
        "periods",
        "correction_periods",
        "correction_filings",
    ]:
        result[col] = (
            result[col]
            .fillna(0)
            .astype(int)
        )

    result[
        "correction_period_rate"
    ] = (
        result[
            "correction_period_rate"
        ]
        .fillna(0.0)
    )

    return result


def build_chains(
    filings: pd.DataFrame,
) -> pd.DataFrame:

    if filings.empty:
        return pd.DataFrame()

    group_info = (
        filings
        .groupby(
            [
                "stock_code",
                "period_key",
            ]
        )
        .agg(
            submission_count=(
                "rcept_no",
                "size",
            ),
            has_correction=(
                "is_correction",
                "max",
            ),
        )
        .reset_index()
    )

    interesting = group_info.loc[
        group_info[
            "has_correction"
        ]
        | group_info[
            "submission_count"
        ].gt(1)
    ][
        [
            "stock_code",
            "period_key",
        ]
    ]

    if interesting.empty:
        return pd.DataFrame()

    out = filings.merge(
        interesting,
        on=[
            "stock_code",
            "period_key",
        ],
        how="inner",
    )

    out["rcept_dt"] = pd.to_datetime(
        out["rcept_dt"],
        format="%Y%m%d",
        errors="coerce",
    )

    return (
        out
        .sort_values(
            [
                "stock_code",
                "period_key",
                "rcept_dt",
                "rcept_no",
            ]
        )
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# CLI / Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="테스트할 mapped 기업 수. 기본 20",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="mapping 가능한 전체 historical-union 기업 audit",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    universe = load_historical_universe()

    mapped = map_to_dart(
        universe
    )

    mapped.to_csv(
        MAPPING_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "DART HISTORICAL-UNIVERSE MAPPING"
    )
    print(
        "=" * 80
    )

    print(
        mapped[
            "mapping_method"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        f"\nmapping audit: {MAPPING_OUT}"
    )

    targets = (
        mapped.loc[
            mapped["mapped"]
        ]
        .sort_values(
            "stock_code"
        )
        .reset_index(drop=True)
    )

    if not args.full:
        targets = targets.head(
            args.sample
        )

        print(
            f"\nSAMPLE MODE: "
            f"{len(targets)} companies"
        )
    else:
        print(
            f"\nFULL MODE: "
            f"{len(targets)} companies"
        )

    session = requests.Session()
    filing_rows = []

    for i, row in targets.iterrows():

        print(
            f"[{i + 1}/{len(targets)}] "
            f"{row['stock_code']} "
            f"{row['dart_corp_name']}"
        )

        try:
            rows = collect_company_filings(
                session,
                row,
            )
            filing_rows.extend(
                rows
            )
        except Exception as exc:
            print(
                "  ERROR:",
                repr(exc),
            )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    filings = pd.DataFrame(
        filing_rows
    )

    if not filings.empty:
        filings[
            "rcept_dt"
        ] = pd.to_datetime(
            filings["rcept_dt"],
            format="%Y%m%d",
            errors="coerce",
        )

        filings.to_parquet(
            FILINGS_OUT,
            index=False,
        )

    summary = build_summary(
        targets,
        filings,
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    chains = build_chains(
        filings
    )

    if not chains.empty:
        chains.to_csv(
            CHAINS_OUT,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame(
            columns=[
                "stock_code",
                "period_key",
                "rcept_no",
                "rcept_dt",
                "report_nm",
                "is_correction",
            ]
        ).to_csv(
            CHAINS_OUT,
            index=False,
            encoding="utf-8-sig",
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "CORRECTION PREVALENCE SUMMARY"
    )
    print(
        "=" * 80
    )

    total_periods = int(
        summary[
            "periods"
        ].sum()
    )

    corrected_periods = int(
        summary[
            "correction_periods"
        ].sum()
    )

    companies_with_corrections = int(
        summary[
            "correction_periods"
        ].gt(0)
        .sum()
    )

    print(
        f"companies audited          : {len(summary)}"
    )
    print(
        f"companies with corrections : "
        f"{companies_with_corrections}"
    )
    print(
        f"report periods             : {total_periods}"
    )
    print(
        f"periods with corrections   : {corrected_periods}"
    )

    if total_periods > 0:
        print(
            "correction-period rate      : "
            f"{corrected_periods / total_periods:.4%}"
        )

    print(
        "\n[Top correction-heavy companies]"
    )

    print(
        summary[
            [
                "stock_code",
                "dart_corp_name",
                "periods",
                "correction_periods",
                "correction_period_rate",
            ]
        ]
        .sort_values(
            [
                "correction_periods",
                "correction_period_rate",
            ],
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    print(
        f"\nsummary : {SUMMARY_OUT}"
    )
    print(
        f"chains  : {CHAINS_OUT}"
    )

    print(
        "\n판단:"
        "\n- correction-period rate가 매우 낮더라도"
        "  corrected period에 현재 최신 숫자를 과거로 소급하면 PIT 위반"
        "\n- strict PIT가 필요하면 다음 단계에서"
        "  각 rcept_no별 fnlttXbrl.xml 원본 ZIP을 저장/파싱"
        "\n- 단순 구현을 원하면 corrected period는"
        "  correction rcept_dt 이후에만 현재 API 수치를 사용하고"
        "  그 이전은 missing으로 두는 conservative 방식도 가능"
    )


if __name__ == "__main__":
    main()
