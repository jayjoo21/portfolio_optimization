from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 05C3-A. Toss Market-wide Investor Flow
#
# INPUT
#   data/raw/toss/market_flow/
#       kospi_investor_trading.json.gz
#       kosdaq_investor_trading.json.gz
#
# OUTPUT
#   data/clean/toss/market_flow/
#       toss_market_investor_flow_clean.parquet
#
#   data/features/market/
#       toss_market_flow_features.parquet
#
# 설계 원칙
# ------------------------------------------------------------
# 1) raw buy/sell amount는 보존
# 2) net amount = buy - sell
# 3) 시장 규모가 시간에 따라 달라지므로 raw KRW만 모델에 넣지 않고
#    market trading amount 대비 net-flow share를 주 feature로 사용
# 4) 5d/20d는 "일별 비율의 단순 합"이 아니라
#       rolling sum(net amount) / rolling sum(market amount)
#    으로 계산
# 5) 아직 common dataset에 붙이지 않음.
#    availability / decision timing 처리는 05D2 정렬 단계에서 수행
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "toss"
    / "market_flow"
)

CLEAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "toss"
    / "market_flow"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "market"
)

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


RAW_FILES = {
    "KOSPI": RAW_DIR / "kospi_investor_trading.json.gz",
    "KOSDAQ": RAW_DIR / "kosdaq_investor_trading.json.gz",
}


INVESTORS = {
    "individual": "retail",
    "foreigner": "foreign",
    "institution": "institution",
    "otherCorporation": "other_corp",
}


INSTITUTION_BREAKDOWN = {
    "financialInvestment": "financial_investment",
    "insurance": "insurance",
    "trust": "trust",
    "privateEquityFund": "private_equity_fund",
    "bank": "bank",
    "otherFinancialInstitution": "other_financial_institution",
    "pensionFund": "pension_fund",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_gzip_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"raw file이 없습니다: {path}"
        )

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def to_float(value: Any) -> float:
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------
# Raw -> Clean
# ------------------------------------------------------------

def flatten_record(
    market: str,
    record: dict[str, Any],
) -> dict[str, Any]:

    row: dict[str, Any] = {
        "date": record.get("date"),
        "updated_at": record.get("updatedAt"),
        "market": market,
    }

    # 4개 상위 투자자 분류
    for api_name, short_name in INVESTORS.items():
        block = record.get(api_name) or {}

        buy = to_float(
            block.get("buyAmount")
        )
        sell = to_float(
            block.get("sellAmount")
        )

        row[f"{short_name}_buy_amount"] = buy
        row[f"{short_name}_sell_amount"] = sell
        row[f"{short_name}_net_amount"] = buy - sell

    # 기관 세부 7개 분류도 clean에는 보존.
    # V1 model feature에는 우선 사용하지 않음.
    institution = record.get("institution") or {}
    breakdown = institution.get("breakdown") or {}

    for api_name, short_name in INSTITUTION_BREAKDOWN.items():
        block = breakdown.get(api_name) or {}

        buy = to_float(
            block.get("buyAmount")
        )
        sell = to_float(
            block.get("sellAmount")
        )

        row[f"inst_{short_name}_buy_amount"] = buy
        row[f"inst_{short_name}_sell_amount"] = sell
        row[f"inst_{short_name}_net_amount"] = buy - sell

    return row


def build_clean_panel() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for market, path in RAW_FILES.items():
        payload = read_gzip_json(path)

        records = payload.get("records") or []

        if not records:
            raise RuntimeError(
                f"{market}: records가 비어 있습니다."
            )

        for record in records:
            rows.append(
                flatten_record(
                    market=market,
                    record=record,
                )
            )

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df["updated_at"] = pd.to_datetime(
        df["updated_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("Asia/Seoul")

    numeric_cols = [
        col
        for col in df.columns
        if col.endswith(
            (
                "_buy_amount",
                "_sell_amount",
                "_net_amount",
            )
        )
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df
        .drop_duplicates(
            subset=["date", "market"],
            keep="last",
        )
        .sort_values(
            ["market", "date"]
        )
        .reset_index(drop=True)
    )

    # 시장 전체 거래대금 proxy:
    # API 문서상 4 investor categories의 buy 합계와
    # sell 합계가 시장 전체 기준으로 서로 같음.
    buy_cols = [
        "retail_buy_amount",
        "foreign_buy_amount",
        "institution_buy_amount",
        "other_corp_buy_amount",
    ]

    sell_cols = [
        "retail_sell_amount",
        "foreign_sell_amount",
        "institution_sell_amount",
        "other_corp_sell_amount",
    ]

    df["market_buy_amount"] = df[buy_cols].sum(
        axis=1,
        min_count=len(buy_cols),
    )

    df["market_sell_amount"] = df[sell_cols].sum(
        axis=1,
        min_count=len(sell_cols),
    )

    # 데이터 품질 체크용.
    # 정상적으로는 buy total == sell total이어야 함.
    df["market_amount_gap"] = (
        df["market_buy_amount"]
        - df["market_sell_amount"]
    )

    df["market_amount_gap_abs"] = (
        df["market_amount_gap"].abs()
    )

    return df


# ------------------------------------------------------------
# Clean -> Features
# ------------------------------------------------------------

def build_features(
    clean_df: pd.DataFrame,
) -> pd.DataFrame:

    keep_cols = [
        "date",
        "market",
        "updated_at",
        "market_buy_amount",
        "market_sell_amount",
        "market_amount_gap",
    ]

    feature_df = clean_df[keep_cols].copy()

    investor_names = [
        "retail",
        "foreign",
        "institution",
        "other_corp",
    ]

    # 먼저 raw net amount도 feature table에 보존.
    for investor in investor_names:
        feature_df[f"{investor}_net_amount"] = (
            clean_df[f"{investor}_net_amount"]
        )

    # --------------------------------------------------------
    # Flow imbalance
    #
    # 1d:
    #   net buy amount_t / market trading amount_t
    #
    # 5d / 20d:
    #   sum(net buy amount) / sum(market trading amount)
    #
    # 단위가 사라져 시장 규모 변화에 덜 민감해짐.
    # --------------------------------------------------------

    grouped_indices = clean_df.groupby(
        "market",
        sort=False,
    ).groups

    for market, idx in grouped_indices.items():
        sub = clean_df.loc[idx].sort_values(
            "date"
        )

        denominator = sub["market_buy_amount"]

        for investor in investor_names:
            net = sub[f"{investor}_net_amount"]

            feature_df.loc[
                sub.index,
                f"{investor}_flow_share_1d",
            ] = safe_ratio(
                net,
                denominator,
            )

            for window in (5, 20):
                rolling_net = net.rolling(
                    window=window,
                    min_periods=window,
                ).sum()

                rolling_market = denominator.rolling(
                    window=window,
                    min_periods=window,
                ).sum()

                feature_df.loc[
                    sub.index,
                    f"{investor}_flow_share_{window}d",
                ] = safe_ratio(
                    rolling_net,
                    rolling_market,
                )

        # 시장 전체 거래대금 변화:
        # flow feature 해석 시 시장 activity context로 사용 가능.
        # pct_change는 이전 값 대비 변화율.
        feature_df.loc[
            sub.index,
            "market_trading_amount_change_1d",
        ] = denominator.pct_change(
            fill_method=None
        )

        feature_df.loc[
            sub.index,
            "market_trading_amount_ratio_20d",
        ] = safe_ratio(
            denominator,
            denominator.rolling(
                window=20,
                min_periods=20,
            ).mean(),
        )

    # --------------------------------------------------------
    # Identity checks
    # 모든 investor net flow의 합은 원칙상 0이어야 함.
    # --------------------------------------------------------

    net_cols = [
        f"{name}_net_amount"
        for name in investor_names
    ]

    feature_df["net_flow_sum_check"] = (
        feature_df[net_cols]
        .sum(
            axis=1,
            min_count=len(net_cols),
        )
    )

    feature_df = (
        feature_df
        .sort_values(
            ["market", "date"]
        )
        .reset_index(drop=True)
    )

    return feature_df


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_clean(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("05C3-A VALIDATION")
    print("=" * 80)

    print("\n[Clean shape]")
    print(clean_df.shape)

    print("\n[Date coverage]")
    coverage = (
        clean_df
        .groupby("market")
        .agg(
            rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
        )
    )
    print(coverage)

    print("\n[Duplicate date-market]")
    dup_count = clean_df.duplicated(
        subset=["date", "market"]
    ).sum()
    print(dup_count)

    print("\n[Market buy/sell amount identity]")
    gap_stats = (
        clean_df
        .groupby("market")["market_amount_gap_abs"]
        .agg(
            max_gap="max",
            mean_gap="mean",
        )
    )
    print(gap_stats)

    print("\n[Investor net-flow identity]")
    net_check = (
        feature_df
        .groupby("market")["net_flow_sum_check"]
        .agg(
            max_abs=lambda x: x.abs().max(),
            mean_abs=lambda x: x.abs().mean(),
        )
    )
    print(net_check)

    print("\n[Feature missing counts]")
    feature_cols = [
        col
        for col in feature_df.columns
        if "flow_share" in col
    ]

    print(
        feature_df[
            ["market"] + feature_cols
        ]
        .groupby("market")
        .apply(
            lambda g: g[feature_cols].isna().sum()
        )
    )

    print("\n[Latest rows]")
    print(
        feature_df
        .groupby("market")
        .tail(3)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    clean_df = build_clean_panel()

    feature_df = build_features(
        clean_df
    )

    clean_path = (
        CLEAN_DIR
        / "toss_market_investor_flow_clean.parquet"
    )

    feature_path = (
        FEATURE_DIR
        / "toss_market_flow_features.parquet"
    )

    clean_df.to_parquet(
        clean_path,
        index=False,
    )

    feature_df.to_parquet(
        feature_path,
        index=False,
    )

    validate_clean(
        clean_df,
        feature_df,
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"clean   : {clean_path}")
    print(f"features: {feature_path}")

    print(
        "\n주의:"
        "\n- observation date의 t일 flow를 t일 장중 의사결정에 사용하지 않음"
        "\n- common dataset(05D2)에서 decision timing에 맞춰 lag/alignment 처리"
        "\n- 기관 세부 7개 분류는 clean에 보존하지만 V1 feature에는 아직 미사용"
    )


if __name__ == "__main__":
    main()
