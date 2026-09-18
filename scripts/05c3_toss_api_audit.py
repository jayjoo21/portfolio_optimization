from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# 05C3 Toss API historical-depth audit

# 목적
# - Toss 수급 데이터가 실제로 2018년까지 내려오는지 확인
# - 아직 full universe 수집은 하지 않음
# - 삼성전자(005930) 1종목 + KOSPI/KOSDAQ 시장수급만 audit

# 실행 위치
#   C:\code\portfolio_optimization

# 실행
#   python scripts\05c3_toss_api_audit.py



BASE_URL = "https://openapi.tossinvest.com"

# 오래 상장된 종목으로 데이터 제공 시작시점을 확인
AUDIT_SYMBOL = "005930"

# 우리 프로젝트가 필요한 historical 시작점
TARGET_START_DATE = date(2018, 1, 1)

# endpoint별 최대 100개 record
PAGE_SIZE = 100

# rate limit 10 TPS보다 여유 있게 사용
REQUEST_SLEEP_SECONDS = 0.15

# 안전장치
MAX_PAGES_PER_SERIES = 100

STOCK_ENDPOINTS = {
    "investor_trading": "/api/v1/stocks/{symbol}/investor-trading",
    "program_trades": "/api/v1/stocks/{symbol}/program-trades",
    "short_selling": "/api/v1/stocks/{symbol}/short-selling",
    "credit_trades": "/api/v1/stocks/{symbol}/credit-trades",
    "securities_lending": "/api/v1/stocks/{symbol}/securities-lending",
}

MARKET_FLOW_SYMBOLS = ["KOSPI", "KOSDAQ"]


# ------------------------------------------------------------
# 프로젝트 경로
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "toss"
    / "audit"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "toss"
)

RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 환경변수
# ------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "\nTOSS_CLIENT_ID 또는 TOSS_CLIENT_SECRET이 없습니다.\n"
        "프로젝트 루트의 .env 파일에 아래처럼 추가하세요.\n\n"
        "TOSS_CLIENT_ID=발급받은_client_id\n"
        "TOSS_CLIENT_SECRET=발급받은_client_secret\n"
    )


# ------------------------------------------------------------
# 공통 HTTP
# ------------------------------------------------------------

def issue_access_token() -> str:
    """
    OAuth 2.0 Client Credentials 방식으로 access token 발급.
    토스증권은 client당 access token 1개만 유효하므로
    프로그램 실행 중 불필요하게 재발급하지 않는다.
    """
    url = f"{BASE_URL}/oauth2/token"

    response = requests.post(
        url,
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

    if response.status_code == 403:
        raise RuntimeError(
            "\n토큰 발급이 403으로 차단되었습니다.\n"
            "Toss WTS > 설정 > Open API > 허용 IP 관리에서\n"
            "현재 PC의 외부 IP가 등록되어 있는지 확인하세요.\n\n"
            f"response: {response.text[:1000]}"
        )

    response.raise_for_status()

    payload = response.json()

    token = payload.get("access_token")

    if not token:
        raise RuntimeError(
            f"access_token이 응답에 없습니다: {payload}"
        )

    return token


def make_session(access_token: str) -> requests.Session:
    session = requests.Session()

    session.headers.update(
        {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "portfolio-optimization-research/1.0",
        }
    )

    return session


def request_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 6,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    429 rate-limit 발생 시 Retry-After를 우선 사용하고,
    없으면 지수 backoff로 재시도.
    """
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
                f"[429] rate limit → {wait_seconds:.1f}s 후 재시도"
            )

            time.sleep(wait_seconds)
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "\n401 Unauthorized입니다.\n"
                "토큰이 만료/무효화되었을 수 있습니다.\n"
                "이 스크립트를 다시 실행해 새 토큰을 발급하세요.\n\n"
                f"response: {response.text[:1000]}"
            )

        if response.status_code == 403:
            raise RuntimeError(
                "\n403 Forbidden입니다.\n"
                "허용 IP 등록 여부와 API 권한을 확인하세요.\n\n"
                f"response: {response.text[:1000]}"
            )

        response.raise_for_status()

        return response.json(), dict(response.headers)

    raise RuntimeError(
        f"{path}: 최대 재시도 횟수를 초과했습니다."
    )


# ------------------------------------------------------------
# 연결 smoke test
# ------------------------------------------------------------

def smoke_test_price(
    session: requests.Session,
    symbol: str = AUDIT_SYMBOL,
) -> None:
    """
    API 연결 자체가 정상인지 현재가 endpoint로 먼저 검사.
    """
    payload, headers = request_json(
        session,
        "/api/v1/prices",
        params={
            "symbols": symbol,
        },
    )

    result = payload.get("result", [])

    if not result:
        raise RuntimeError(
            f"현재가 smoke test에서 결과가 없습니다: {payload}"
        )

    item = result[0]

    print("\n[SMOKE TEST OK]")
    print(
        f"symbol={item.get('symbol')} "
        f"lastPrice={item.get('lastPrice')} "
        f"timestamp={item.get('timestamp')}"
    )

    print(
        "rate_limit=",
        headers.get("X-RateLimit-Limit"),
        "remaining=",
        headers.get("X-RateLimit-Remaining"),
    )


# ------------------------------------------------------------
# raw 저장
# ------------------------------------------------------------

def save_raw_page(
    category: str,
    series_name: str,
    page_no: int,
    payload: dict[str, Any],
) -> None:
    target_dir = (
        RAW_AUDIT_DIR
        / category
        / series_name
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path = (
        target_dir
        / f"page_{page_no:03d}.json"
    )

    with open(
        target_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ------------------------------------------------------------
# 공통 historical depth audit
# ------------------------------------------------------------

def audit_paginated_series(
    session: requests.Session,
    *,
    path: str,
    params: dict[str, Any],
    category: str,
    series_name: str,
) -> dict[str, Any]:

    until = None

    all_dates: list[date] = []

    seen_until: set[str] = set()

    total_records = 0

    stop_reason = "max_pages"

    reached_target = False

    newest_date: date | None = None

    oldest_date: date | None = None

    page_count = 0

    for page_no in range(
        1,
        MAX_PAGES_PER_SERIES + 1,
    ):
        query = dict(params)

        query["count"] = PAGE_SIZE

        if until is not None:
            query["until"] = until

        payload, _ = request_json(
            session,
            path,
            params=query,
        )

        page_count += 1

        save_raw_page(
            category=category,
            series_name=series_name,
            page_no=page_no,
            payload=payload,
        )

        result = payload.get("result") or {}

        records = result.get("records") or []

        next_until = result.get("nextUntil")

        dates_this_page: list[date] = []

        for record in records:
            raw_date = record.get("date")

            if raw_date:
                parsed = pd.Timestamp(
                    raw_date
                ).date()

                dates_this_page.append(
                    parsed
                )

                all_dates.append(
                    parsed
                )

        total_records += len(records)

        if all_dates:
            newest_date = max(all_dates)
            oldest_date = min(all_dates)

        print(
            f"{series_name:<25} "
            f"page={page_no:02d} "
            f"records={len(records):3d} "
            f"oldest={oldest_date} "
            f"nextUntil={next_until}"
        )

        # 우리가 필요한 2018년까지 도달했으면
        # feasibility 판단에는 충분함.
        if (
            oldest_date is not None
            and oldest_date <= TARGET_START_DATE
        ):
            reached_target = True
            stop_reason = "reached_project_start"
            break

        if not records and not next_until:
            stop_reason = "no_more_data"
            break

        if not next_until:
            stop_reason = "next_until_null"
            break

        if next_until in seen_until:
            stop_reason = "pagination_loop_detected"
            break

        seen_until.add(next_until)

        until = next_until

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    return {
        "series": series_name,
        "category": category,
        "pages": page_count,
        "records": total_records,
        "newest_date": newest_date,
        "oldest_date": oldest_date,
        "target_start_date": TARGET_START_DATE,
        "reaches_2018": reached_target,
        "stop_reason": stop_reason,
    }


# ------------------------------------------------------------
# 개별 종목 5종 수급 audit
# ------------------------------------------------------------

def audit_stock_trading_trends(
    session: requests.Session,
) -> list[dict[str, Any]]:

    print(
        "\n"
        + "=" * 80
    )
    print(
        f"STOCK FLOW AUDIT: {AUDIT_SYMBOL}"
    )
    print(
        "=" * 80
    )

    summaries = []

    for name, path_template in STOCK_ENDPOINTS.items():
        print(
            f"\n--- {name} ---"
        )

        path = path_template.format(
            symbol=AUDIT_SYMBOL
        )

        summary = audit_paginated_series(
            session,
            path=path,
            params={},
            category="stock",
            series_name=name,
        )

        summaries.append(
            summary
        )

    return summaries


# ------------------------------------------------------------
# KOSPI / KOSDAQ 시장 전체 투자자 수급 audit
# ------------------------------------------------------------

def audit_market_investor_flow(
    session: requests.Session,
) -> list[dict[str, Any]]:

    print(
        "\n"
        + "=" * 80
    )
    print(
        "MARKET INVESTOR FLOW AUDIT"
    )
    print(
        "=" * 80
    )

    summaries = []

    for market_symbol in MARKET_FLOW_SYMBOLS:
        name = (
            f"{market_symbol.lower()}_investor_trading"
        )

        print(
            f"\n--- {name} ---"
        )

        path = (
            f"/api/v1/market-indicators/"
            f"{market_symbol}/investor-trading"
        )

        summary = audit_paginated_series(
            session,
            path=path,
            params={
                "interval": "1d",
            },
            category="market",
            series_name=name,
        )

        summaries.append(
            summary
        )

    return summaries


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:
    print(
        "Toss API historical-depth audit 시작"
    )

    access_token = issue_access_token()

    session = make_session(
        access_token
    )

    # token 자체는 출력하지 않는다.
    print(
        "[AUTH OK] access token 발급 성공"
    )

    smoke_test_price(
        session
    )

    stock_summaries = (
        audit_stock_trading_trends(
            session
        )
    )

    market_summaries = (
        audit_market_investor_flow(
            session
        )
    )

    summary_df = pd.DataFrame(
        stock_summaries
        + market_summaries
    )

    summary_df["newest_date"] = (
        pd.to_datetime(
            summary_df["newest_date"]
        )
    )

    summary_df["oldest_date"] = (
        pd.to_datetime(
            summary_df["oldest_date"]
        )
    )

    summary_path = (
        REPORT_DIR
        / "toss_historical_depth_audit.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "AUDIT SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\n저장 완료: {summary_path}"
    )

    print(
        "\n판단 기준:"
    )
    print(
        "- reaches_2018=True  → 2018+ main model용 historical source 후보"
    )
    print(
        "- reaches_2018=False → 실제 oldest_date를 보고 KRX 등 대체 source 검토"
    )
    print(
        "- raw 응답은 data/raw/toss/audit/ 아래 JSON으로 보존"
    )


if __name__ == "__main__":
    main()
