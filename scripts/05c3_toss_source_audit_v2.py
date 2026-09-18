from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05C3 Toss source audit v2
#
# 목적
# 1) 현재 Toss에서 정상 지원되는 종목으로 5개 수급 API smoke test
# 2) 과거 KRX300 구성종목 합집합 중 Toss 현재 종목 마스터 지원 여부 audit
# 3) KOSPI / KOSDAQ 시장 전체 투자자 수급은 2018+ raw로 별도 수집
#
# 중요:
# - Toss가 현재 지원하지 않는 과거 상장폐지 종목을 "제외"해서
#   historical model universe를 만들면 안 된다.
# - symbol availability audit은 현재 Toss 지원범위를 파악하기 위한 것일 뿐,
#   PIT universe filter가 아니다.
# ============================================================

BASE_URL = "https://openapi.tossinvest.com"

TARGET_START_DATE = date(2018, 1, 1)
PAGE_SIZE = 100
REQUEST_SLEEP_SECONDS = 0.22
MAX_PAGES = 120

# 현재도 거래되는 대표 종목만 smoke test에 사용
ACTIVE_TEST_TICKERS = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
]

FLOW_ENDPOINTS = {
    "investor_trading": "/api/v1/stocks/{symbol}/investor-trading",
    "program_trades": "/api/v1/stocks/{symbol}/program-trades",
    "short_selling": "/api/v1/stocks/{symbol}/short-selling",
    "credit_trades": "/api/v1/stocks/{symbol}/credit-trades",
    "securities_lending": "/api/v1/stocks/{symbol}/securities-lending",
}

MARKET_SYMBOLS = ["KOSPI", "KOSDAQ"]


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "krx"
    / "universe"
    / "krx300_monthly_membership_snapshots.csv"
)

RAW_TOSS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "toss"
)

INTERIM_TOSS_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "toss"
)

RAW_TOSS_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_TOSS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Auth
# ------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "TOSS_CLIENT_ID / TOSS_CLIENT_SECRET이 없습니다.\n"
        "프로젝트 루트 .env에 추가하세요."
    )


def issue_access_token() -> str:
    response = requests.post(
        f"{BASE_URL}/oauth2/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()
    token = payload.get("access_token")

    if not token:
        raise RuntimeError(
            f"access_token이 없습니다: {payload}"
        )

    return token


def make_session(token: str) -> requests.Session:
    session = requests.Session()

    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "portfolio-optimization-research/1.0",
        }
    )

    return session


# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

def get_response(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 6,
) -> requests.Response:

    url = f"{BASE_URL}{path}"

    for attempt in range(max_retries):
        response = session.get(
            url,
            params=params,
            timeout=30,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")

            if retry_after is not None:
                wait_seconds = float(retry_after)
            else:
                wait_seconds = min(2 ** attempt, 10)

            print(
                f"[429] {wait_seconds:.1f}s 후 재시도"
            )

            time.sleep(wait_seconds)
            continue

        return response

    raise RuntimeError(
        f"{path}: 최대 재시도 횟수 초과"
    )


def get_json_or_raise(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], requests.Response]:

    response = get_response(
        session,
        path,
        params=params,
    )

    response.raise_for_status()

    return response.json(), response


# ------------------------------------------------------------
# Membership union
# ------------------------------------------------------------

def detect_ticker_column(df: pd.DataFrame) -> str:
    candidates = [
        "ticker",
        "code",
        "symbol",
        "stock_code",
        "종목코드",
        "단축코드",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "membership CSV에서 ticker column을 찾지 못했습니다.\n"
        f"columns={df.columns.tolist()}"
    )


def load_historical_union_tickers() -> list[str]:
    if not MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"membership 파일이 없습니다: {MEMBERSHIP_PATH}"
        )

    df = pd.read_csv(
        MEMBERSHIP_PATH,
        dtype=str,
    )

    ticker_col = detect_ticker_column(df)

    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return tickers


# ------------------------------------------------------------
# 1. Active symbol smoke test
# ------------------------------------------------------------

def run_active_smoke_test(
    session: requests.Session,
) -> pd.DataFrame:

    rows = []

    print("\n" + "=" * 80)
    print("ACTIVE SYMBOL FLOW SMOKE TEST")
    print("=" * 80)

    total = len(ACTIVE_TEST_TICKERS) * len(FLOW_ENDPOINTS)
    index = 0

    for ticker in ACTIVE_TEST_TICKERS:
        for endpoint_name, template in FLOW_ENDPOINTS.items():
            index += 1

            path = template.format(
                symbol=ticker
            )

            print(
                f"[{index}/{total}] "
                f"{endpoint_name} / {ticker}"
            )

            response = get_response(
                session,
                path,
                params={
                    "count": 3,
                },
            )

            status_code = response.status_code

            if status_code == 200:
                payload = response.json()
                result = payload.get("result") or {}
                records = result.get("records") or []

                rows.append(
                    {
                        "ticker": ticker,
                        "endpoint": endpoint_name,
                        "http_status": status_code,
                        "api_status": "supported",
                        "records_returned": len(records),
                    }
                )

            elif status_code == 404:
                try:
                    error_payload = response.json()
                except Exception:
                    error_payload = {}

                rows.append(
                    {
                        "ticker": ticker,
                        "endpoint": endpoint_name,
                        "http_status": status_code,
                        "api_status": "stock_not_found",
                        "records_returned": 0,
                        "error": json.dumps(
                            error_payload,
                            ensure_ascii=False,
                        ),
                    }
                )

            else:
                response.raise_for_status()

            time.sleep(REQUEST_SLEEP_SECONDS)

    result_df = pd.DataFrame(rows)

    path = (
        INTERIM_TOSS_DIR
        / "toss_active_symbol_smoke_test.csv"
    )

    result_df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print(result_df.to_string(index=False))
    print(f"\n저장: {path}")

    return result_df


# ------------------------------------------------------------
# 2. Toss current stock-master availability audit
# ------------------------------------------------------------

def audit_current_master_availability(
    session: requests.Session,
) -> pd.DataFrame:

    tickers = load_historical_union_tickers()

    print("\n" + "=" * 80)
    print("TOSS CURRENT STOCK-MASTER AVAILABILITY AUDIT")
    print("=" * 80)
    print(
        f"historical KRX300 union: {len(tickers)} tickers"
    )
    print(
        "주의: 이 audit은 Toss의 '현재' 지원 여부를 확인할 뿐,\n"
        "과거 PIT universe를 필터링하는 용도로 사용하면 안 됩니다."
    )

    rows = []

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):
        response = get_response(
            session,
            "/api/v1/stocks",
            params={
                "symbols": ticker,
            },
        )

        if response.status_code == 200:
            payload = response.json()
            result = payload.get("result")

            # API 버전에 따라 list/dict 가능성을 보수적으로 처리
            if isinstance(result, list):
                item = result[0] if result else {}
            elif isinstance(result, dict):
                item = result
            else:
                item = {}

            rows.append(
                {
                    "ticker": ticker,
                    "http_status": 200,
                    "toss_supported_now": True,
                    "name": (
                        item.get("name")
                        or item.get("stockName")
                    ),
                    "market": item.get("market"),
                    "listing_status": (
                        item.get("listingStatus")
                        or item.get("status")
                    ),
                }
            )

        elif response.status_code == 404:
            try:
                payload = response.json()
            except Exception:
                payload = {}

            rows.append(
                {
                    "ticker": ticker,
                    "http_status": 404,
                    "toss_supported_now": False,
                    "error": json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                }
            )

        else:
            response.raise_for_status()

        if (
            i == 1
            or i % 25 == 0
            or i == len(tickers)
        ):
            supported = sum(
                bool(row["toss_supported_now"])
                for row in rows
            )

            print(
                f"[{i}/{len(tickers)}] "
                f"supported={supported} "
                f"unsupported={i - supported}"
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    result_df = pd.DataFrame(rows)

    out_path = (
        INTERIM_TOSS_DIR
        / "toss_historical_union_current_availability.csv"
    )

    result_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("CURRENT MASTER SUMMARY")
    print("=" * 80)

    print(
        result_df["toss_supported_now"]
        .value_counts(dropna=False)
        .rename_axis("toss_supported_now")
        .to_string()
    )

    print(f"\n저장: {out_path}")

    return result_df


# ------------------------------------------------------------
# 3. Market-wide KOSPI/KOSDAQ investor flow collection
# ------------------------------------------------------------

def save_gzip_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
        )


def collect_market_investor_flow(
    session: requests.Session,
) -> pd.DataFrame:

    summaries = []

    print("\n" + "=" * 80)
    print("MARKET-WIDE INVESTOR FLOW COLLECTION")
    print("=" * 80)

    for market_symbol in MARKET_SYMBOLS:
        path = (
            f"/api/v1/market-indicators/"
            f"{market_symbol}/investor-trading"
        )

        until = None
        seen_until: set[str] = set()
        all_records: list[dict[str, Any]] = []

        for page_no in range(
            1,
            MAX_PAGES + 1,
        ):
            params: dict[str, Any] = {
                "interval": "1d",
                "count": PAGE_SIZE,
            }

            if until is not None:
                params["until"] = until

            payload, _ = get_json_or_raise(
                session,
                path,
                params=params,
            )

            result = payload.get("result") or {}
            records = result.get("records") or []
            next_until = result.get("nextUntil")

            all_records.extend(records)

            page_dates = []

            for record in records:
                raw_date = record.get("date")

                if raw_date:
                    page_dates.append(
                        pd.Timestamp(raw_date).date()
                    )

            oldest = (
                min(page_dates)
                if page_dates
                else None
            )

            print(
                f"{market_symbol:<6} "
                f"page={page_no:02d} "
                f"records={len(records):3d} "
                f"oldest={oldest}"
            )

            if (
                oldest is not None
                and oldest <= TARGET_START_DATE
            ):
                break

            if not next_until:
                break

            if next_until in seen_until:
                break

            seen_until.add(next_until)
            until = next_until

            time.sleep(REQUEST_SLEEP_SECONDS)

        # 날짜 중복 제거
        by_date: dict[str, dict[str, Any]] = {}

        for record in all_records:
            raw_date = record.get("date")

            if raw_date:
                by_date[raw_date] = record

        records_sorted = [
            by_date[key]
            for key in sorted(by_date)
        ]

        raw_payload = {
            "source": "Toss Securities Open API",
            "endpoint": path,
            "market": market_symbol,
            "collected_records": len(records_sorted),
            "records": records_sorted,
        }

        out_path = (
            RAW_TOSS_DIR
            / "market_flow"
            / f"{market_symbol.lower()}_investor_trading.json.gz"
        )

        save_gzip_json(
            out_path,
            raw_payload,
        )

        dates = [
            pd.Timestamp(record["date"]).date()
            for record in records_sorted
            if record.get("date")
        ]

        summaries.append(
            {
                "market": market_symbol,
                "records": len(records_sorted),
                "newest_date": max(dates) if dates else None,
                "oldest_date": min(dates) if dates else None,
                "reaches_2018": (
                    min(dates) <= TARGET_START_DATE
                    if dates
                    else False
                ),
                "raw_path": str(out_path),
            }
        )

    summary_df = pd.DataFrame(
        summaries
    )

    out_path = (
        INTERIM_TOSS_DIR
        / "toss_market_flow_manifest.csv"
    )

    summary_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print(summary_df.to_string(index=False))
    print(f"\nmanifest: {out_path}")

    return summary_df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit-universe",
        action="store_true",
        help=(
            "과거 KRX300 합집합 605종목의 "
            "현재 Toss stock-master 지원 여부를 검사"
        ),
    )

    parser.add_argument(
        "--collect-market",
        action="store_true",
        help=(
            "KOSPI/KOSDAQ 시장 전체 투자자 수급을 "
            "2018년까지 raw로 수집"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = issue_access_token()
    session = make_session(token)

    print("[AUTH OK]")

    # 아무 옵션 없이 실행하면 안전한 smoke test만 실행
    if not args.audit_universe and not args.collect_market:
        run_active_smoke_test(session)
        return

    if args.audit_universe:
        audit_current_master_availability(
            session
        )

    if args.collect_market:
        collect_market_investor_flow(
            session
        )


if __name__ == "__main__":
    main()
