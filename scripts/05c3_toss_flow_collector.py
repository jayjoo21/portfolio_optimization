from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05C3 Toss Flow & Positioning RAW Collector
#
# 목적
# - Toss audit 결과로 확인된 실제 제공기간만 수집
# - "오늘의 50종목"을 과거 전체에 쓰지 않음
# - KRX300 월별 membership snapshot의 ticker 합집합을
#   오직 RAW 수집 후보군으로만 사용
# - 최종 PIT universe / investable 50 결정은 별도 단계
#
# 저장
# data/raw/toss/flow/<endpoint>/<ticker>.json.gz
# data/interim/toss/toss_flow_collection_manifest.csv
#
# 기본 실행은 3개 종목만 검사:
#   python scripts/05c3_toss_flow_collector.py
#
# 전체 수집:
#   python scripts/05c3_toss_flow_collector.py --full
#
# 특정 endpoint만:
#   python scripts/05c3_toss_flow_collector.py --full --endpoint investor_trading
# ============================================================


BASE_URL = "https://openapi.tossinvest.com"

PROJECT_START = date(2018, 1, 1)

# 실제 audit에서 확인된 Toss의 종목별 제공 시작점.
# "문서상 추정"이 아니라 현재 프로젝트에서 005930으로 직접 측정한 값.
ENDPOINTS = {
    "investor_trading": {
        "path": "/api/v1/stocks/{symbol}/investor-trading",
        "available_from": date(2019, 4, 1),
    },
    "program_trades": {
        "path": "/api/v1/stocks/{symbol}/program-trades",
        "available_from": date(2019, 4, 1),
    },
    "short_selling": {
        "path": "/api/v1/stocks/{symbol}/short-selling",
        "available_from": date(2019, 4, 1),
    },
    "credit_trades": {
        "path": "/api/v1/stocks/{symbol}/credit-trades",
        "available_from": date(2023, 4, 17),
    },
    "securities_lending": {
        "path": "/api/v1/stocks/{symbol}/securities-lending",
        "available_from": date(2021, 4, 1),
    },
}

PAGE_SIZE = 100

# STOCK_TRADING_TREND 공식 rate-limit은 현재 최대 10 TPS.
# 운영 한도는 바뀔 수 있으므로 응답 헤더 + 429 Retry-After도 사용.
REQUEST_SLEEP_SECONDS = 0.13

MAX_PAGES = 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "krx"
    / "universe"
    / "krx300_monthly_membership_snapshots.csv"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "toss"
    / "flow"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "toss"
)

MANIFEST_PATH = (
    INTERIM_DIR
    / "toss_flow_collection_manifest.csv"
)

RAW_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "프로젝트 루트 .env에 "
        "TOSS_CLIENT_ID / TOSS_CLIENT_SECRET이 필요합니다."
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


class TossClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.reauthenticate()

    def reauthenticate(self) -> None:
        token = issue_access_token()

        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "portfolio-optimization-research/1.0",
            }
        )

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        max_retries: int = 7,
    ) -> tuple[dict[str, Any], dict[str, str]]:

        url = f"{BASE_URL}{path}"

        reauthed = False

        for attempt in range(max_retries):
            response = self.session.get(
                url,
                params=params,
                timeout=30,
            )

            if response.status_code == 401 and not reauthed:
                print("[401] token 재발급 후 재시도")
                self.reauthenticate()
                reauthed = True
                continue

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after is not None:
                    wait = float(retry_after)
                else:
                    wait = min(2 ** attempt, 10)

                print(
                    f"[429] {wait:.2f}s 후 재시도"
                )

                time.sleep(wait)
                continue

            if response.status_code == 403:
                raise RuntimeError(
                    "403 Forbidden: Toss 허용 IP 등록 여부를 확인하세요."
                )

            response.raise_for_status()

            return response.json(), dict(response.headers)

        raise RuntimeError(
            f"재시도 횟수 초과: {path}, params={params}"
        )


def load_collection_universe() -> list[str]:
    """
    중요:
    이 ticker 목록은 '최종 투자 universe'가 아니다.

    과거 KRX300 월별 membership snapshot에 한 번이라도 등장한
    ticker를 넓게 수집하는 RAW collection universe일 뿐이다.

    최종 모델에서는 PIT 검증된 membership / investable universe로
    다시 필터링해야 한다.
    """
    if not MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"membership 파일이 없습니다:\n{MEMBERSHIP_PATH}"
        )

    membership = pd.read_csv(
        MEMBERSHIP_PATH,
        dtype={
            "ticker": "string",
        },
    )

    if "ticker" not in membership.columns:
        raise KeyError(
            f"'ticker' 컬럼이 없습니다: {membership.columns.tolist()}"
        )

    if "date" in membership.columns:
        membership["date"] = pd.to_datetime(
            membership["date"],
            errors="coerce",
        )

    tickers = (
        membership["ticker"]
        .dropna()
        .astype(str)
        .str.zfill(6)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return tickers


def output_path(
    endpoint_name: str,
    symbol: str,
) -> Path:
    folder = RAW_ROOT / endpoint_name
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{symbol}.json.gz"


def write_gzip_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
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


def collect_one_series(
    client: TossClient,
    *,
    symbol: str,
    endpoint_name: str,
    overwrite: bool,
) -> dict[str, Any]:

    spec = ENDPOINTS[endpoint_name]

    path = spec["path"].format(
        symbol=symbol
    )

    target_start: date = spec["available_from"]

    out = output_path(
        endpoint_name,
        symbol,
    )

    if out.exists() and not overwrite:
        return {
            "symbol": symbol,
            "endpoint": endpoint_name,
            "status": "skipped_exists",
            "records": None,
            "newest_date": None,
            "oldest_date": None,
            "available_from": target_start.isoformat(),
            "output_path": str(out),
        }

    until: str | None = None

    seen_until: set[str] = set()

    records_all: list[dict[str, Any]] = []

    newest: date | None = None
    oldest: date | None = None

    stop_reason = "max_pages"

    pages = 0

    last_rate_limit = None

    for page_no in range(
        1,
        MAX_PAGES + 1,
    ):
        params: dict[str, Any] = {
            "count": PAGE_SIZE,
        }

        if until:
            params["until"] = until

        payload, headers = client.get_json(
            path,
            params=params,
        )

        pages += 1

        last_rate_limit = {
            "limit": headers.get("X-RateLimit-Limit"),
            "remaining": headers.get("X-RateLimit-Remaining"),
            "reset": headers.get("X-RateLimit-Reset"),
        }

        result = payload.get("result") or {}

        records = result.get("records") or []

        next_until = result.get("nextUntil")

        records_all.extend(records)

        dates = []

        for record in records:
            raw_date = record.get("date")

            if raw_date:
                d = pd.Timestamp(raw_date).date()
                dates.append(d)

        if dates:
            page_newest = max(dates)
            page_oldest = min(dates)

            newest = (
                page_newest
                if newest is None
                else max(newest, page_newest)
            )

            oldest = (
                page_oldest
                if oldest is None
                else min(oldest, page_oldest)
            )

        # Toss audit에서 확인한 endpoint 가용 시작점에 도달
        if oldest is not None and oldest <= target_start:
            stop_reason = "reached_endpoint_start"
            break

        if not records and not next_until:
            stop_reason = "no_more_data"
            break

        if not next_until:
            stop_reason = "next_until_null"
            break

        if next_until in seen_until:
            stop_reason = "pagination_loop"
            break

        seen_until.add(next_until)
        until = next_until

        time.sleep(REQUEST_SLEEP_SECONDS)

    # 혹시 마지막 page가 target_start보다 더 과거까지 포함하면
    # Toss가 제공하는 원자료는 보존하되 collection metadata에 기간 명시.
    raw_package = {
        "source": "Toss Securities Open API",
        "endpoint": endpoint_name,
        "symbol": symbol,
        "collected_at": datetime.now().astimezone().isoformat(),
        "audit_available_from": target_start.isoformat(),
        "pages": pages,
        "stop_reason": stop_reason,
        "last_rate_limit_headers": last_rate_limit,
        "records": records_all,
    }

    write_gzip_json(
        out,
        raw_package,
    )

    return {
        "symbol": symbol,
        "endpoint": endpoint_name,
        "status": "collected",
        "pages": pages,
        "records": len(records_all),
        "newest_date": newest.isoformat() if newest else None,
        "oldest_date": oldest.isoformat() if oldest else None,
        "available_from": target_start.isoformat(),
        "stop_reason": stop_reason,
        "output_path": str(out),
    }


def save_manifest(
    new_rows: list[dict[str, Any]],
) -> None:
    new_df = pd.DataFrame(new_rows)

    if MANIFEST_PATH.exists():
        old_df = pd.read_csv(
            MANIFEST_PATH,
            dtype={
                "symbol": "string",
            },
        )

        manifest = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )

        # 동일 symbol/endpoint는 최신 실행 결과 보존
        manifest = (
            manifest
            .drop_duplicates(
                subset=[
                    "symbol",
                    "endpoint",
                ],
                keep="last",
            )
        )
    else:
        manifest = new_df

    manifest = manifest.sort_values(
        [
            "endpoint",
            "symbol",
        ]
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 collection universe 수집",
    )

    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=3,
        help="--full이 아닐 때 수집할 종목 수 (기본 3)",
    )

    parser.add_argument(
        "--endpoint",
        choices=[
            "all",
            *ENDPOINTS.keys(),
        ],
        default="all",
        help="수집할 endpoint",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 json.gz 덮어쓰기",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = load_collection_universe()

    print(
        f"RAW collection universe unique tickers: {len(tickers):,}"
    )

    print(
        "주의: 이 목록은 최종 PIT investable universe가 아니라 "
        "raw 수집 후보 ticker 합집합입니다."
    )

    if not args.full:
        tickers = tickers[: args.limit_symbols]

        print(
            f"TEST MODE: 앞 {len(tickers)}개 ticker만 수집합니다."
        )
    else:
        print(
            f"FULL MODE: {len(tickers):,}개 ticker 수집"
        )

    if args.endpoint == "all":
        endpoint_names = list(
            ENDPOINTS.keys()
        )
    else:
        endpoint_names = [
            args.endpoint
        ]

    print(
        "endpoints:",
        endpoint_names,
    )

    client = TossClient()

    manifest_rows: list[dict[str, Any]] = []

    total_jobs = (
        len(tickers)
        * len(endpoint_names)
    )

    job_no = 0

    for endpoint_name in endpoint_names:
        for symbol in tickers:
            job_no += 1

            print(
                f"\n[{job_no:,}/{total_jobs:,}] "
                f"{endpoint_name} / {symbol}"
            )

            try:
                summary = collect_one_series(
                    client,
                    symbol=symbol,
                    endpoint_name=endpoint_name,
                    overwrite=args.overwrite,
                )

            except Exception as exc:
                summary = {
                    "symbol": symbol,
                    "endpoint": endpoint_name,
                    "status": "error",
                    "error": repr(exc),
                }

                print(
                    "ERROR:",
                    repr(exc),
                )

            manifest_rows.append(
                summary
            )

            # 중간 실패에도 진행상황 보존
            if len(manifest_rows) % 10 == 0:
                save_manifest(
                    manifest_rows
                )

    save_manifest(
        manifest_rows
    )

    result_df = pd.DataFrame(
        manifest_rows
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "COLLECTION SUMMARY"
    )
    print(
        "=" * 80
    )

    if "status" in result_df.columns:
        print(
            result_df["status"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        f"\nmanifest: {MANIFEST_PATH}"
    )

    print(
        "\n다음 단계:"
    )

    print(
        "1) raw JSON.GZ는 그대로 보존"
    )

    print(
        "2) clean 단계에서 long-format panel로 변환"
    )

    print(
        "3) 최종 PIT universe가 확정된 뒤 date×ticker로 필터"
    )

    print(
        "4) pre-availability 구간을 0이나 ffill로 채우지 않음"
    )


if __name__ == "__main__":
    main()
