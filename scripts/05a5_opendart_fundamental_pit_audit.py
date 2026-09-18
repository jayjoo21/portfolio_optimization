from __future__ import annotations

import io
import json
import os
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-0. OpenDART Fundamental PIT Audit
#
# 목적
# ------------------------------------------------------------
# 1) stock_code -> DART corp_code mapping 확인
# 2) 정기공시 목록에서 원본 / 정정 접수번호(rcept_no), 접수일(rcept_dt) 확인
# 3) 전체 재무제표 API가 반환하는 rcept_no가 공시목록의 어느 제출본과
#    연결되는지 확인
# 4) CFS(연결) 우선, 없으면 OFS(별도) fallback 가능성 확인
# 5) 주요 계정명/계정ID의 실제 availability 확인
#
# 아직 fundamental feature를 만들지 않는다.
# 이 audit 결과를 보고
# - simple OpenDART financial endpoint만으로 PIT가 가능한지
# - 정정 전 원본 XBRL snapshot까지 필요한지
# 를 결정한다.
#
# .env
# DART_API_KEY=발급받은_40자리_키
# ============================================================


BASE_URL = "https://opendart.fss.or.kr/api"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "audit"
)

MASTER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "master"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

RAW_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


CORP_MASTER_PATH = (
    MASTER_DIR
    / "dart_corp_codes.parquet"
)

FILINGS_PATH = (
    INTERIM_DIR
    / "dart_regular_filings_audit.parquet"
)

FS_AUDIT_PATH = (
    INTERIM_DIR
    / "dart_financial_statement_pit_audit.csv"
)


load_dotenv(PROJECT_ROOT / ".env")

DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise RuntimeError(
        "\nDART_API_KEY가 없습니다.\n"
        "프로젝트 루트 .env에 아래처럼 추가하세요.\n\n"
        "DART_API_KEY=발급받은_40자리_인증키\n"
    )


# 현재도 상장된 장기 종목 3개만 audit
TEST_TICKERS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
}

# 시작/최근 시점 둘 다 검사
AUDIT_YEARS = [2018, 2025]

REPORT_CODES = {
    "11013": "Q1",
    "11012": "H1",
    "11014": "Q3",
    "11011": "FY",
}

REQUEST_SLEEP_SECONDS = 0.20


# ------------------------------------------------------------
# Common API
# ------------------------------------------------------------

def request_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:

    response = session.get(
        f"{BASE_URL}/{endpoint}",
        params={
            "crtfc_key": DART_API_KEY,
            **params,
        },
        timeout=45,
    )

    response.raise_for_status()

    payload = response.json()

    status = str(
        payload.get("status", "")
    )

    # 000 정상 / 013 데이터 없음은 호출 자체는 정상
    if status not in ("000", "013"):
        raise RuntimeError(
            f"{endpoint}: "
            f"status={status} "
            f"message={payload.get('message')}"
        )

    time.sleep(
        REQUEST_SLEEP_SECONDS
    )

    return payload


# ------------------------------------------------------------
# Corp code master
# ------------------------------------------------------------

def download_corp_master(
    session: requests.Session,
) -> pd.DataFrame:

    print("[1] DART corpCode.xml 다운로드")

    response = session.get(
        f"{BASE_URL}/corpCode.xml",
        params={
            "crtfc_key": DART_API_KEY,
        },
        timeout=60,
    )

    response.raise_for_status()

    content = response.content

    # 공식 응답은 ZIP(binary).
    # 일부 환경에서 XML이 직접 반환될 가능성도 보수적으로 처리.
    try:
        with zipfile.ZipFile(
            io.BytesIO(content)
        ) as zf:
            names = zf.namelist()

            if not names:
                raise RuntimeError(
                    "corpCode ZIP 내부 파일 없음"
                )

            xml_bytes = zf.read(
                names[0]
            )
    except zipfile.BadZipFile:
        xml_bytes = content

    root = ET.fromstring(
        xml_bytes
    )

    rows = []

    for item in root.findall(
        ".//list"
    ):
        rows.append(
            {
                "corp_code": (
                    item.findtext("corp_code")
                    or ""
                ).strip(),
                "corp_name": (
                    item.findtext("corp_name")
                    or ""
                ).strip(),
                "stock_code": (
                    item.findtext("stock_code")
                    or ""
                ).strip(),
                "modify_date": (
                    item.findtext("modify_date")
                    or ""
                ).strip(),
            }
        )

    df = pd.DataFrame(
        rows
    )

    df["stock_code"] = (
        df["stock_code"]
        .astype(str)
        .str.strip()
    )

    df.to_parquet(
        CORP_MASTER_PATH,
        index=False,
    )

    print(
        f"corp master rows: {len(df):,}"
    )
    print(
        f"saved: {CORP_MASTER_PATH}"
    )

    return df


def load_or_download_corp_master(
    session: requests.Session,
) -> pd.DataFrame:

    if CORP_MASTER_PATH.exists():
        df = pd.read_parquet(
            CORP_MASTER_PATH
        )

        print(
            f"[1] cached corp master 사용: "
            f"{CORP_MASTER_PATH}"
        )

        return df

    return download_corp_master(
        session
    )


# ------------------------------------------------------------
# Filing list
# ------------------------------------------------------------

def collect_regular_filings(
    session: requests.Session,
    corp_code: str,
    stock_code: str,
    corp_name: str,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    page_no = 1

    while True:
        payload = request_json(
            session,
            "list.json",
            {
                "corp_code": corp_code,
                "bgn_de": "20180101",
                "end_de": "20260918",
                # A = 정기공시
                "pblntf_ty": "A",
                # N = 정정 포함 모든 제출본
                "last_reprt_at": "N",
                "sort": "date",
                "sort_mth": "asc",
                "page_no": page_no,
                "page_count": 100,
            },
        )

        if payload.get("status") == "013":
            break

        page_rows = (
            payload.get("list")
            or []
        )

        for row in page_rows:
            rows.append(
                {
                    **row,
                    "audit_stock_code": stock_code,
                    "audit_corp_name": corp_name,
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

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        df["rcept_dt"] = pd.to_datetime(
            df["rcept_dt"],
            format="%Y%m%d",
            errors="coerce",
        )

        df["is_correction"] = (
            df["report_nm"]
            .astype(str)
            .str.contains(
                "정정",
                na=False,
            )
        )

    return df


# ------------------------------------------------------------
# Financial statements
# ------------------------------------------------------------

def get_financial_statement(
    session: requests.Session,
    corp_code: str,
    year: int,
    report_code: str,
) -> tuple[pd.DataFrame, str | None]:

    # 연결(CFS) 우선
    for fs_div in (
        "CFS",
        "OFS",
    ):
        payload = request_json(
            session,
            "fnlttSinglAcntAll.json",
            {
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
                "fs_div": fs_div,
            },
        )

        if payload.get("status") == "000":
            rows = (
                payload.get("list")
                or []
            )

            if rows:
                return (
                    pd.DataFrame(rows),
                    fs_div,
                )

    return (
        pd.DataFrame(),
        None,
    )


# ------------------------------------------------------------
# Account audit
# ------------------------------------------------------------

ACCOUNT_KEYWORDS = {
    "assets": [
        "자산총계",
        "Assets",
    ],
    "liabilities": [
        "부채총계",
        "Liabilities",
    ],
    "equity": [
        "자본총계",
        "Equity",
    ],
    "revenue": [
        "매출액",
        "영업수익",
        "수익(매출액)",
        "Revenue",
    ],
    "operating_income": [
        "영업이익",
        "영업이익(손실)",
        "Operating profit",
    ],
    "net_income": [
        "당기순이익",
        "당기순이익(손실)",
        "분기순이익",
        "반기순이익",
        "Profit for the period",
    ],
}


def find_account(
    fs: pd.DataFrame,
    candidates: list[str],
) -> tuple[str | None, str | None]:

    if fs.empty:
        return (
            None,
            None,
        )

    account_nm = (
        fs.get(
            "account_nm",
            pd.Series(
                index=fs.index,
                dtype=str,
            ),
        )
        .astype(str)
        .str.strip()
    )

    for candidate in candidates:
        exact = fs.loc[
            account_nm.str.casefold()
            == candidate.casefold()
        ]

        if not exact.empty:
            row = exact.iloc[0]

            return (
                str(
                    row.get(
                        "account_id",
                        "",
                    )
                ),
                str(
                    row.get(
                        "account_nm",
                        "",
                    )
                ),
            )

    # 정확 일치가 없으면 부분 일치 여부만 audit
    for candidate in candidates:
        mask = account_nm.str.contains(
            candidate,
            case=False,
            regex=False,
            na=False,
        )

        if mask.any():
            row = fs.loc[
                mask
            ].iloc[0]

            return (
                str(
                    row.get(
                        "account_id",
                        "",
                    )
                ),
                str(
                    row.get(
                        "account_nm",
                        "",
                    )
                ),
            )

    return (
        None,
        None,
    )


# ------------------------------------------------------------
# PIT match
# ------------------------------------------------------------

def match_receipt_to_filing(
    filings: pd.DataFrame,
    rcept_no: str | None,
) -> dict[str, Any]:

    if (
        filings.empty
        or not rcept_no
    ):
        return {
            "rcept_match": False,
            "rcept_dt": pd.NaT,
            "report_nm": None,
            "is_correction": None,
        }

    match = filings.loc[
        filings["rcept_no"].astype(str)
        == str(rcept_no)
    ]

    if match.empty:
        return {
            "rcept_match": False,
            "rcept_dt": pd.NaT,
            "report_nm": None,
            "is_correction": None,
        }

    row = match.iloc[-1]

    return {
        "rcept_match": True,
        "rcept_dt": row.get(
            "rcept_dt"
        ),
        "report_nm": row.get(
            "report_nm"
        ),
        "is_correction": row.get(
            "is_correction"
        ),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    session = requests.Session()

    corp_master = (
        load_or_download_corp_master(
            session
        )
    )

    mappings = []

    for stock_code, expected_name in TEST_TICKERS.items():

        match = corp_master.loc[
            corp_master[
                "stock_code"
            ].astype(str)
            == stock_code
        ]

        if match.empty:
            raise RuntimeError(
                f"{stock_code} "
                f"{expected_name}: corp_code mapping 없음"
            )

        row = match.iloc[0]

        mappings.append(
            {
                "stock_code": stock_code,
                "expected_name": expected_name,
                "corp_code": row[
                    "corp_code"
                ],
                "dart_corp_name": row[
                    "corp_name"
                ],
            }
        )

    mapping_df = pd.DataFrame(
        mappings
    )

    print("\n[CORP CODE MAPPING]")
    print(
        mapping_df.to_string(
            index=False
        )
    )

    # -----------------------------------------
    # filings
    # -----------------------------------------

    filing_frames = []

    filings_by_stock: dict[
        str,
        pd.DataFrame,
    ] = {}

    print(
        "\n[2] 정기공시 목록 수집 "
        "(정정보고서 포함)"
    )

    for row in mappings:

        filings = collect_regular_filings(
            session=session,
            corp_code=row["corp_code"],
            stock_code=row["stock_code"],
            corp_name=row["dart_corp_name"],
        )

        filings_by_stock[
            row["stock_code"]
        ] = filings

        filing_frames.append(
            filings
        )

        corrections = (
            int(
                filings[
                    "is_correction"
                ].sum()
            )
            if (
                not filings.empty
                and "is_correction"
                in filings.columns
            )
            else 0
        )

        print(
            f"{row['stock_code']} "
            f"{row['dart_corp_name']}: "
            f"filings={len(filings)} "
            f"corrections={corrections}"
        )

    all_filings = pd.concat(
        filing_frames,
        ignore_index=True,
    )

    all_filings.to_parquet(
        FILINGS_PATH,
        index=False,
    )

    print(
        f"filings saved: {FILINGS_PATH}"
    )

    # -----------------------------------------
    # financial statement PIT audit
    # -----------------------------------------

    print(
        "\n[3] 재무제표 ↔ 접수번호 PIT audit"
    )

    audit_rows = []

    for row in mappings:

        stock_code = row[
            "stock_code"
        ]

        corp_code = row[
            "corp_code"
        ]

        filings = filings_by_stock[
            stock_code
        ]

        for year in AUDIT_YEARS:

            for report_code, period_name in REPORT_CODES.items():

                fs, fs_div = (
                    get_financial_statement(
                        session=session,
                        corp_code=corp_code,
                        year=year,
                        report_code=report_code,
                    )
                )

                if fs.empty:
                    audit_rows.append(
                        {
                            "stock_code": stock_code,
                            "corp_name": row[
                                "dart_corp_name"
                            ],
                            "year": year,
                            "report_code": report_code,
                            "period": period_name,
                            "fs_div": None,
                            "fs_rows": 0,
                            "rcept_no": None,
                            "rcept_match": False,
                            "status": "no_data",
                        }
                    )

                    print(
                        f"{stock_code} "
                        f"{year} {period_name}: "
                        "NO DATA"
                    )

                    continue

                # 한 재무제표 응답 안에서는 보통 동일 rcept_no
                rcept_values = (
                    fs["rcept_no"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                    if "rcept_no" in fs.columns
                    else []
                )

                rcept_no = (
                    rcept_values[0]
                    if len(rcept_values) == 1
                    else (
                        "MULTIPLE:"
                        + "|".join(
                            rcept_values
                        )
                        if rcept_values
                        else None
                    )
                )

                pit = match_receipt_to_filing(
                    filings=filings,
                    rcept_no=(
                        rcept_values[0]
                        if len(rcept_values) == 1
                        else None
                    ),
                )

                result: dict[str, Any] = {
                    "stock_code": stock_code,
                    "corp_name": row[
                        "dart_corp_name"
                    ],
                    "year": year,
                    "report_code": report_code,
                    "period": period_name,
                    "fs_div": fs_div,
                    "fs_rows": len(fs),
                    "rcept_no": rcept_no,
                    **pit,
                    "status": "ok",
                }

                for feature_name, keywords in ACCOUNT_KEYWORDS.items():
                    account_id, account_nm = (
                        find_account(
                            fs,
                            keywords,
                        )
                    )

                    result[
                        f"{feature_name}_account_id"
                    ] = account_id

                    result[
                        f"{feature_name}_account_nm"
                    ] = account_nm

                audit_rows.append(
                    result
                )

                print(
                    f"{stock_code} "
                    f"{year} {period_name}: "
                    f"{fs_div} "
                    f"rows={len(fs)} "
                    f"rcept_no={rcept_no} "
                    f"match={pit['rcept_match']} "
                    f"date={pit['rcept_dt']} "
                    f"correction={pit['is_correction']}"
                )

                # audit 원본 일부 저장
                raw_path = (
                    RAW_DIR
                    / (
                        f"{stock_code}_"
                        f"{year}_"
                        f"{period_name}_"
                        f"{fs_div}.json"
                    )
                )

                fs.to_json(
                    raw_path,
                    orient="records",
                    force_ascii=False,
                    indent=2,
                )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        FS_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5 FUNDAMENTAL PIT AUDIT SUMMARY"
    )
    print(
        "=" * 80
    )

    summary_cols = [
        "stock_code",
        "corp_name",
        "year",
        "period",
        "fs_div",
        "fs_rows",
        "rcept_no",
        "rcept_match",
        "rcept_dt",
        "is_correction",
        "status",
    ]

    existing_cols = [
        col
        for col in summary_cols
        if col in audit_df.columns
    ]

    print(
        audit_df[
            existing_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\n[Account availability]"
    )

    account_cols = [
        col
        for col in audit_df.columns
        if col.endswith(
            "_account_nm"
        )
    ]

    if account_cols:
        availability = (
            audit_df[
                account_cols
            ]
            .notna()
            .sum()
            .rename(
                "available_rows"
            )
        )

        print(
            availability.to_string()
        )

    print(
        f"\ncorp master : {CORP_MASTER_PATH}"
    )
    print(
        f"filings     : {FILINGS_PATH}"
    )
    print(
        f"audit       : {FS_AUDIT_PATH}"
    )

    print(
        "\n판단 기준:"
        "\n1) rcept_match=True:"
        " 재무제표 응답을 실제 공시 접수일에 연결 가능"
        "\n2) is_correction=True가 존재:"
        " 현재 재무 API가 정정본을 반환할 수 있으므로"
        " pre-correction PIT 복원 여부를 별도 확인"
        "\n3) CFS 우선 / OFS fallback:"
        " 연결재무제표가 없을 때만 별도재무제표 검토"
        "\n4) 계정명이 기업별로 다르면:"
        " account_nm 문자열만 믿지 않고"
        " account_id + 후보명 mapping 설계 필요"
    )


if __name__ == "__main__":
    main()
