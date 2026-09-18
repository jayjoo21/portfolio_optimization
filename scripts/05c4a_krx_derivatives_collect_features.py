from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05C4-A. KRX KOSPI200 Derivatives Collector + V1 Features
#
# 목적
# ------------------------------------------------------------
# KRX 공식 Open API에서 2018+ KOSPI200
# - 선물 (정규 코스피200 선물)
# - 옵션 (정규 월물 코스피200 옵션)
# 를 일별로 받아 market-level derivatives state를 만든다.
#
# IMPORTANT
# ------------------------------------------------------------
# [Core V1]
# - futures total volume / OI
# - futures OI change
# - futures volume activity
# - option put/call volume ratio
# - option put/call OI ratio
# - option volume activity
#
# [Candidate only]
# - liquid-contract futures basis
#   -> 만기/roll 효과가 있으므로 바로 core feature로 사용하지 않음
#
# - option volume-weighted IV
#   -> strike/maturity mix 때문에 바로 core feature로 사용하지 않음
#
# 데이터 규모 문제 때문에 하루 1.6만+ 옵션 원문 전체를 저장하지 않는다.
# 대신 exact product = "코스피200 옵션"만 즉시 집계하고,
# 수집 코드 + KRX source를 재현 근거로 남긴다.
#
# INPUT
# ------------------------------------------------------------
# data/clean/krx/krx_stock_panel_clean.parquet
# -> KRX 실제 거래일 calendar source
#
# .env
# KRX_API_KEY=...
#
# OUTPUT
# ------------------------------------------------------------
# data/clean/krx/derivatives/
#   k200_derivatives_daily_clean.parquet
#
# data/features/market/
#   k200_derivatives_features_v1.parquet
#
# data/interim/krx/derivatives/
#   k200_derivatives_collection_checkpoint.parquet
# ============================================================


BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/drv"

FUTURES_ENDPOINT = "fut_bydd_trd"
OPTIONS_ENDPOINT = "opt_bydd_trd"

FUTURES_PRODUCT = "코스피200 선물"
OPTIONS_PRODUCT = "코스피200 옵션"

REQUEST_SLEEP_SECONDS = 0.18
MAX_RETRIES = 6


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOCK_PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "krx"
    / "krx_stock_panel_clean.parquet"
)

CLEAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "krx"
    / "derivatives"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "market"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "krx"
    / "derivatives"
)

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


CHECKPOINT_PATH = (
    INTERIM_DIR
    / "k200_derivatives_collection_checkpoint.parquet"
)

CLEAN_PATH = (
    CLEAN_DIR
    / "k200_derivatives_daily_clean.parquet"
)

FEATURE_PATH = (
    FEATURE_DIR
    / "k200_derivatives_features_v1.parquet"
)


load_dotenv(PROJECT_ROOT / ".env")

KRX_API_KEY = os.getenv("KRX_API_KEY")

if not KRX_API_KEY:
    raise RuntimeError(
        "KRX_API_KEY가 없습니다.\n"
        "프로젝트 루트 .env를 확인하세요."
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def to_num(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace(
            {
                "-": np.nan,
                "": np.nan,
                "None": np.nan,
                "nan": np.nan,
            }
        ),
        errors="coerce",
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return (
        numerator
        / denominator.replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def request_krx(
    session: requests.Session,
    endpoint: str,
    bas_dd: str,
) -> list[dict[str, Any]]:

    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(MAX_RETRIES):
        response = session.get(
            url,
            params={
                "basDd": bas_dd,
            },
            headers={
                "AUTH_KEY": KRX_API_KEY,
                "Accept": "application/json",
                "User-Agent": (
                    "portfolio-optimization-research/1.0"
                ),
            },
            timeout=45,
        )

        if response.status_code == 429:
            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = min(
                    2 ** attempt,
                    15,
                )

            print(
                f"[429] {endpoint} {bas_dd}"
                f" → {wait:.1f}s 대기"
            )
            time.sleep(wait)
            continue

        if response.status_code in (401, 403):
            raise PermissionError(
                f"{endpoint} HTTP {response.status_code}\n"
                "KRX API 활용승인과 AUTH_KEY를 확인하세요.\n"
                f"{response.text[:800]}"
            )

        response.raise_for_status()

        payload = response.json()

        records = payload.get("OutBlock_1")

        if records is None:
            raise RuntimeError(
                f"{endpoint} / {bas_dd}: "
                f"OutBlock_1 없음\n"
                f"{str(payload)[:1000]}"
            )

        return records

    raise RuntimeError(
        f"{endpoint} / {bas_dd}: "
        "최대 재시도 횟수 초과"
    )


# ------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------

def detect_date_column(
    df: pd.DataFrame,
) -> str:
    candidates = [
        "date",
        "Date",
        "BAS_DD",
        "bas_dd",
        "trade_date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "KRX stock panel에서 date column을 찾지 못했습니다.\n"
        f"columns={df.columns.tolist()}"
    )


def load_trading_dates(
    start: str,
    end: str | None,
) -> list[pd.Timestamp]:

    if not STOCK_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"stock panel이 없습니다: {STOCK_PANEL_PATH}"
        )

    df = pd.read_parquet(
        STOCK_PANEL_PATH
    )

    date_col = detect_date_column(
        df
    )

    dates = pd.to_datetime(
        df[date_col],
        errors="raise",
    ).drop_duplicates()

    start_ts = pd.Timestamp(start)

    dates = dates[
        dates >= start_ts
    ]

    if end is not None:
        end_ts = pd.Timestamp(end)
        dates = dates[
            dates <= end_ts
        ]

    return (
        dates
        .sort_values()
        .tolist()
    )


# ------------------------------------------------------------
# Daily futures aggregation
# ------------------------------------------------------------

def aggregate_futures(
    bas_dd: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    df = pd.DataFrame(records)

    if df.empty:
        return {
            "date": pd.Timestamp(bas_dd),
            "fut_available": False,
        }

    if "PROD_NM" not in df.columns:
        raise RuntimeError(
            f"{bas_dd}: futures PROD_NM 없음"
        )

    df = df.loc[
        df["PROD_NM"].astype(str)
        == FUTURES_PRODUCT
    ].copy()

    if df.empty:
        return {
            "date": pd.Timestamp(bas_dd),
            "fut_available": False,
        }

    numeric_cols = [
        "TDD_CLSPRC",
        "SPOT_PRC",
        "SETL_PRC",
        "ACC_TRDVOL",
        "ACC_TRDVAL",
        "ACC_OPNINT_QTY",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_num(
                df[col]
            )

    # 가장 거래량이 큰 계약 = liquid contract
    #
    # 우리는 "최근월물"이라고 강제 해석하지 않는다.
    # market-state용 price discovery proxy로 사용.
    liquid = (
        df
        .sort_values(
            [
                "ACC_TRDVOL",
                "ACC_OPNINT_QTY",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .iloc[0]
    )

    settlement = liquid.get(
        "SETL_PRC",
        np.nan,
    )

    close = liquid.get(
        "TDD_CLSPRC",
        np.nan,
    )

    # basis 계산에는 정산가 우선.
    # 정산가 없으면 종가 fallback.
    futures_px = (
        settlement
        if pd.notna(settlement)
        else close
    )

    spot = liquid.get(
        "SPOT_PRC",
        np.nan,
    )

    basis_point = (
        futures_px - spot
        if (
            pd.notna(futures_px)
            and pd.notna(spot)
        )
        else np.nan
    )

    basis_pct = (
        futures_px / spot - 1.0
        if (
            pd.notna(futures_px)
            and pd.notna(spot)
            and spot != 0
        )
        else np.nan
    )

    return {
        "date": pd.Timestamp(bas_dd),
        "fut_available": True,
        "fut_contract_count": len(df),

        # Aggregate across all regular KOSPI200 futures maturities
        "fut_total_volume": df[
            "ACC_TRDVOL"
        ].sum(min_count=1),

        "fut_total_value": df[
            "ACC_TRDVAL"
        ].sum(min_count=1),

        "fut_total_oi": df[
            "ACC_OPNINT_QTY"
        ].sum(min_count=1),

        # Most-liquid contract diagnostics
        "fut_liquid_isu_cd": liquid.get(
            "ISU_CD"
        ),
        "fut_liquid_isu_nm": liquid.get(
            "ISU_NM"
        ),
        "fut_liquid_close": close,
        "fut_liquid_settlement": settlement,
        "fut_liquid_spot": spot,
        "fut_liquid_volume": liquid.get(
            "ACC_TRDVOL",
            np.nan,
        ),
        "fut_liquid_oi": liquid.get(
            "ACC_OPNINT_QTY",
            np.nan,
        ),

        # Candidate only
        "candidate_fut_basis_point": basis_point,
        "candidate_fut_basis_pct": basis_pct,
    }


# ------------------------------------------------------------
# Daily options aggregation
# ------------------------------------------------------------

def weighted_average(
    value: pd.Series,
    weight: pd.Series,
) -> float:

    mask = (
        value.notna()
        & weight.notna()
        & (weight > 0)
    )

    if not mask.any():
        return np.nan

    denominator = weight.loc[
        mask
    ].sum()

    if denominator == 0:
        return np.nan

    return float(
        (
            value.loc[mask]
            * weight.loc[mask]
        ).sum()
        / denominator
    )


def aggregate_options(
    bas_dd: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    df = pd.DataFrame(records)

    if df.empty:
        return {
            "date": pd.Timestamp(bas_dd),
            "opt_available": False,
        }

    required = [
        "PROD_NM",
        "RGHT_TP_NM",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"{bas_dd}: options {col} 없음"
            )

    # 정규 KOSPI200 월물 옵션만 사용.
    #
    # 미니 / 위클리(월) / 위클리(목)는 제외해서
    # product definition을 기간 전체에서 일관되게 유지.
    df = df.loc[
        df["PROD_NM"].astype(str)
        == OPTIONS_PRODUCT
    ].copy()

    if df.empty:
        return {
            "date": pd.Timestamp(bas_dd),
            "opt_available": False,
        }

    numeric_cols = [
        "TDD_CLSPRC",
        "IMP_VOLT",
        "NXTDD_BAS_PRC",
        "ACC_TRDVOL",
        "ACC_TRDVAL",
        "ACC_OPNINT_QTY",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_num(
                df[col]
            )

    rights = (
        df["RGHT_TP_NM"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    call = df.loc[
        rights == "CALL"
    ].copy()

    put = df.loc[
        rights == "PUT"
    ].copy()

    call_volume = call[
        "ACC_TRDVOL"
    ].sum(min_count=1)

    put_volume = put[
        "ACC_TRDVOL"
    ].sum(min_count=1)

    call_value = call[
        "ACC_TRDVAL"
    ].sum(min_count=1)

    put_value = put[
        "ACC_TRDVAL"
    ].sum(min_count=1)

    call_oi = call[
        "ACC_OPNINT_QTY"
    ].sum(min_count=1)

    put_oi = put[
        "ACC_OPNINT_QTY"
    ].sum(min_count=1)

    def scalar_ratio(
        numerator: float,
        denominator: float,
    ) -> float:
        if (
            pd.isna(numerator)
            or pd.isna(denominator)
            or denominator == 0
        ):
            return np.nan

        return float(
            numerator / denominator
        )

    # Candidate IV summary:
    # strike/maturity mix 문제 때문에 core feature로 쓰지 않음.
    call_iv_vw = weighted_average(
        call["IMP_VOLT"],
        call["ACC_TRDVOL"],
    )

    put_iv_vw = weighted_average(
        put["IMP_VOLT"],
        put["ACC_TRDVOL"],
    )

    all_iv_vw = weighted_average(
        df["IMP_VOLT"],
        df["ACC_TRDVOL"],
    )

    return {
        "date": pd.Timestamp(bas_dd),
        "opt_available": True,
        "opt_contract_count": len(df),

        "opt_call_volume": call_volume,
        "opt_put_volume": put_volume,
        "opt_total_volume": (
            call_volume + put_volume
            if (
                pd.notna(call_volume)
                and pd.notna(put_volume)
            )
            else np.nan
        ),

        "opt_call_value": call_value,
        "opt_put_value": put_value,

        "opt_call_oi": call_oi,
        "opt_put_oi": put_oi,
        "opt_total_oi": (
            call_oi + put_oi
            if (
                pd.notna(call_oi)
                and pd.notna(put_oi)
            )
            else np.nan
        ),

        # Core positioning summaries
        "opt_put_call_volume_ratio": scalar_ratio(
            put_volume,
            call_volume,
        ),
        "opt_put_call_oi_ratio": scalar_ratio(
            put_oi,
            call_oi,
        ),

        # Candidate only
        "candidate_opt_call_iv_vw": call_iv_vw,
        "candidate_opt_put_iv_vw": put_iv_vw,
        "candidate_opt_all_iv_vw": all_iv_vw,
    }


# ------------------------------------------------------------
# Resume / checkpoint
# ------------------------------------------------------------

def load_checkpoint() -> pd.DataFrame:
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(
        CHECKPOINT_PATH
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"]
        )

    return df


def save_checkpoint(
    df: pd.DataFrame,
) -> None:
    (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .to_parquet(
            CHECKPOINT_PATH,
            index=False,
        )
    )


# ------------------------------------------------------------
# Collection
# ------------------------------------------------------------

def collect_dates(
    dates: list[pd.Timestamp],
    resume: bool,
) -> pd.DataFrame:

    existing = (
        load_checkpoint()
        if resume
        else pd.DataFrame()
    )

    existing_dates: set[pd.Timestamp] = set()

    if not existing.empty:
        existing_dates = set(
            pd.to_datetime(
                existing["date"]
            ).dt.normalize()
        )

    rows: list[dict[str, Any]] = []

    if not existing.empty:
        rows.extend(
            existing.to_dict(
                orient="records"
            )
        )

    session = requests.Session()

    todo = [
        date
        for date in dates
        if date.normalize()
        not in existing_dates
    ]

    print(
        f"trading dates total : {len(dates)}"
    )
    print(
        f"already collected   : {len(existing_dates)}"
    )
    print(
        f"to collect          : {len(todo)}"
    )

    for i, date in enumerate(
        todo,
        start=1,
    ):
        bas_dd = date.strftime(
            "%Y%m%d"
        )

        try:
            fut_records = request_krx(
                session,
                FUTURES_ENDPOINT,
                bas_dd,
            )

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

            opt_records = request_krx(
                session,
                OPTIONS_ENDPOINT,
                bas_dd,
            )

            fut = aggregate_futures(
                bas_dd,
                fut_records,
            )

            opt = aggregate_options(
                bas_dd,
                opt_records,
            )

            row = {
                **fut,
                **{
                    key: value
                    for key, value
                    in opt.items()
                    if key != "date"
                },
                "collection_status": "ok",
            }

        except Exception as exc:
            print(
                f"\nERROR {bas_dd}: {repr(exc)}"
            )

            row = {
                "date": pd.Timestamp(
                    bas_dd
                ),
                "collection_status": "error",
                "error": repr(exc),
            }

        rows.append(row)

        if (
            i == 1
            or i % 10 == 0
            or i == len(todo)
        ):
            current = pd.DataFrame(
                rows
            )

            save_checkpoint(
                current
            )

            print(
                f"[{i}/{len(todo)}] "
                f"{bas_dd} "
                f"checkpoint saved"
            )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    result = pd.DataFrame(
        rows
    )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = (
        result
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# ------------------------------------------------------------
# Clean / feature build
# ------------------------------------------------------------

def build_features(
    clean: pd.DataFrame,
) -> pd.DataFrame:

    df = clean.loc[
        clean["collection_status"]
        == "ok"
    ].copy()

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    features = df[
        [
            "date",
            "fut_total_volume",
            "fut_total_oi",
            "opt_total_volume",
            "opt_total_oi",
            "opt_put_call_volume_ratio",
            "opt_put_call_oi_ratio",
        ]
    ].copy()

    # --------------------------------------------------------
    # Futures OI
    #
    # OI = Open Interest = 미결제약정
    # 전체 정규 KOSPI200 선물 월물의 OI 합계를 사용하여
    # 특정 월물 roll discontinuity를 줄임.
    # --------------------------------------------------------

    features[
        "k200_fut_oi_change_1d"
    ] = df[
        "fut_total_oi"
    ].pct_change(
        fill_method=None
    )

    features[
        "k200_fut_oi_change_5d"
    ] = (
        df["fut_total_oi"]
        / df["fut_total_oi"].shift(5)
        - 1.0
    )

    # 거래활동은 raw contracts보다 최근 20일 평균 대비 비율 사용
    features[
        "k200_fut_volume_ratio_20d"
    ] = safe_ratio(
        df["fut_total_volume"],
        df["fut_total_volume"].rolling(
            20,
            min_periods=20,
        ).mean(),
    )

    # --------------------------------------------------------
    # Options
    # Put/Call Ratio
    # = PUT aggregate / CALL aggregate
    #
    # 거래량 ratio: 당일 trading activity
    # OI ratio: 남아 있는 open positions의 상대 구성
    # --------------------------------------------------------

    features[
        "k200_opt_put_call_volume_ratio"
    ] = df[
        "opt_put_call_volume_ratio"
    ]

    features[
        "k200_opt_put_call_oi_ratio"
    ] = df[
        "opt_put_call_oi_ratio"
    ]

    features[
        "k200_opt_volume_ratio_20d"
    ] = safe_ratio(
        df["opt_total_volume"],
        df["opt_total_volume"].rolling(
            20,
            min_periods=20,
        ).mean(),
    )

    # --------------------------------------------------------
    # Candidate diagnostics
    #
    # 모델 입력에는 아직 자동 포함하지 않는다.
    # --------------------------------------------------------

    features[
        "candidate_k200_fut_basis_pct"
    ] = df[
        "candidate_fut_basis_pct"
    ]

    features[
        "candidate_k200_opt_iv_vw"
    ] = df[
        "candidate_opt_all_iv_vw"
    ]

    return features


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate(
    clean: pd.DataFrame,
    features: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("05C4 VALIDATION")
    print("=" * 80)

    print("\n[Collection status]")
    print(
        clean[
            "collection_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    ok = clean.loc[
        clean[
            "collection_status"
        ] == "ok"
    ].copy()

    print("\n[Date coverage]")
    if not ok.empty:
        print(
            "rows :",
            len(ok),
        )
        print(
            "start:",
            ok["date"].min(),
        )
        print(
            "end  :",
            ok["date"].max(),
        )

    print("\n[Missing core clean]")
    clean_cols = [
        "fut_total_volume",
        "fut_total_oi",
        "opt_total_volume",
        "opt_total_oi",
        "opt_put_call_volume_ratio",
        "opt_put_call_oi_ratio",
    ]

    existing_clean_cols = [
        col
        for col in clean_cols
        if col in ok.columns
    ]

    if existing_clean_cols:
        print(
            ok[
                existing_clean_cols
            ]
            .isna()
            .sum()
            .to_string()
        )

    print("\n[Feature missing]")
    core_feature_cols = [
        "k200_fut_oi_change_1d",
        "k200_fut_oi_change_5d",
        "k200_fut_volume_ratio_20d",
        "k200_opt_put_call_volume_ratio",
        "k200_opt_put_call_oi_ratio",
        "k200_opt_volume_ratio_20d",
    ]

    print(
        features[
            core_feature_cols
        ]
        .isna()
        .sum()
        .to_string()
    )

    print("\n[Latest rows]")
    show_cols = [
        "date",
        "fut_liquid_isu_nm",
        "fut_total_volume",
        "fut_total_oi",
        "candidate_fut_basis_pct",
        "opt_put_call_volume_ratio",
        "opt_put_call_oi_ratio",
        "candidate_opt_all_iv_vw",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in ok.columns
    ]

    if show_cols:
        print(
            ok[
                show_cols
            ]
            .tail(5)
            .to_string(index=False)
        )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="수집 시작일 YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "수집 종료일 YYYY-MM-DD. "
            "생략하면 stock panel의 마지막 날짜"
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="기존 checkpoint를 무시하고 다시 시작",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "테스트용: 앞 N 거래일만 수집. "
            "예: --sample 5"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dates = load_trading_dates(
        start=args.start,
        end=args.end,
    )

    if args.sample is not None:
        dates = dates[
            :args.sample
        ]

        print(
            f"TEST SAMPLE MODE: "
            f"{len(dates)} trading dates"
        )

    clean = collect_dates(
        dates=dates,
        resume=not args.no_resume,
    )

    # 최종 clean 저장
    clean.to_parquet(
        CLEAN_PATH,
        index=False,
    )

    features = build_features(
        clean
    )

    features.to_parquet(
        FEATURE_PATH,
        index=False,
    )

    validate(
        clean,
        features,
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print(
        f"checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"clean     : {CLEAN_PATH}"
    )
    print(
        f"features  : {FEATURE_PATH}"
    )

    print(
        "\nCORE V1:"
        "\n- k200_fut_oi_change_1d"
        "\n- k200_fut_oi_change_5d"
        "\n- k200_fut_volume_ratio_20d"
        "\n- k200_opt_put_call_volume_ratio"
        "\n- k200_opt_put_call_oi_ratio"
        "\n- k200_opt_volume_ratio_20d"
    )

    print(
        "\nCANDIDATE (아직 모델 입력 확정 아님):"
        "\n- candidate_k200_fut_basis_pct"
        "\n- candidate_k200_opt_iv_vw"
    )


if __name__ == "__main__":
    main()
