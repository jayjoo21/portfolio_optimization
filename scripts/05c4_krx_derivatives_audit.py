from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05C4-0. KRX Derivatives API feasibility audit
#
# 목적
# ------------------------------------------------------------
# 1) KRX 공식 Open API의 선물/옵션 endpoint 권한 확인
# 2) 실제 응답의 상품명(PROD_NM), 종목명(ISU_NM), OI/IV 구조 확인
# 3) KOSPI200 파생상품을 어떻게 필터링할지 결정하기 위한 audit
#
# 아직 feature를 만들지 않는다.
# 실제 raw schema를 먼저 보고 다음 단계에서
# front-month / basis / OI / put-call / IV 설계를 확정한다.
#
# 필요한 KRX API 활용신청
# - 선물 일별매매정보 (주식선물外)
# - 옵션 일별매매정보 (주식옵션外)
#
# .env
# KRX_API_KEY=...
# ============================================================


BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/drv"

ENDPOINTS = {
    "futures": "fut_bydd_trd",
    "options": "opt_bydd_trd",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "krx"
    / "derivatives"
    / "audit"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "krx"
    / "derivatives"
)

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv(PROJECT_ROOT / ".env")

KRX_API_KEY = os.getenv("KRX_API_KEY")

if not KRX_API_KEY:
    raise RuntimeError(
        "\nKRX_API_KEY가 없습니다.\n"
        "프로젝트 루트 .env에 아래처럼 추가하세요.\n\n"
        "KRX_API_KEY=발급받은_인증키\n"
    )


def normalize_number(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def request_endpoint(
    endpoint: str,
    bas_dd: str,
) -> tuple[dict[str, Any], requests.Response]:

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        params={
            "basDd": bas_dd,
        },
        headers={
            "AUTH_KEY": KRX_API_KEY,
            "Accept": "application/json",
            "User-Agent": "portfolio-optimization-research/1.0",
        },
        timeout=30,
    )

    if response.status_code in (401, 403):
        raise PermissionError(
            f"\n[{endpoint}] HTTP {response.status_code}\n"
            "KRX 인증키 자체가 있어도 API별 '활용신청/승인'이 필요합니다.\n"
            "KRX Open API에서 아래 서비스를 확인하세요.\n"
            "- 선물 일별매매정보 (주식선물外)\n"
            "- 옵션 일별매매정보 (주식옵션外)\n\n"
            f"response: {response.text[:1000]}"
        )

    response.raise_for_status()

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"{endpoint}: JSON 파싱 실패\n"
            f"response={response.text[:1000]}"
        ) from exc

    return payload, response


def save_raw(
    name: str,
    bas_dd: str,
    payload: dict[str, Any],
) -> Path:

    path = (
        RAW_DIR
        / f"{name}_{bas_dd}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


def payload_to_df(
    payload: dict[str, Any],
) -> pd.DataFrame:

    records = payload.get("OutBlock_1")

    if records is None:
        # 혹시 API 오류가 JSON body 안에 들어오는 경우 진단
        raise RuntimeError(
            "OutBlock_1이 없습니다.\n"
            f"response keys={list(payload.keys())}\n"
            f"payload={str(payload)[:1500]}"
        )

    if not isinstance(records, list):
        raise RuntimeError(
            f"OutBlock_1이 list가 아닙니다: {type(records)}"
        )

    return pd.DataFrame(records)


def find_kospi200_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    text_cols = [
        col
        for col in [
            "PROD_NM",
            "ISU_NM",
            "MKT_NM",
        ]
        if col in df.columns
    ]

    if not text_cols:
        return df.iloc[0:0].copy()

    mask = pd.Series(
        False,
        index=df.index,
    )

    for col in text_cols:
        values = (
            df[col]
            .astype(str)
            .str.upper()
        )

        mask |= (
            values.str.contains(
                "KOSPI",
                na=False,
            )
            & values.str.contains(
                "200",
                na=False,
            )
        )

        mask |= values.str.contains(
            "코스피200",
            na=False,
        )

    return df.loc[mask].copy()


def audit_futures(
    df: pd.DataFrame,
) -> dict[str, Any]:

    print("\n" + "=" * 80)
    print("FUTURES")
    print("=" * 80)

    print(f"rows: {len(df):,}")

    if "PROD_NM" in df.columns:
        print("\n[unique PROD_NM]")
        print(
            df["PROD_NM"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_string(index=False)
        )

    kospi200 = find_kospi200_rows(df)

    print(
        f"\n[KOSPI200-like rows] {len(kospi200):,}"
    )

    show_cols = [
        col
        for col in [
            "BAS_DD",
            "PROD_NM",
            "MKT_NM",
            "ISU_CD",
            "ISU_NM",
            "TDD_CLSPRC",
            "SPOT_PRC",
            "SETL_PRC",
            "ACC_TRDVOL",
            "ACC_TRDVAL",
            "ACC_OPNINT_QTY",
        ]
        if col in kospi200.columns
    ]

    if not kospi200.empty:
        print(
            kospi200[show_cols]
            .head(30)
            .to_string(index=False)
        )

    numeric_candidates = [
        "TDD_CLSPRC",
        "SPOT_PRC",
        "SETL_PRC",
        "ACC_TRDVOL",
        "ACC_TRDVAL",
        "ACC_OPNINT_QTY",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = normalize_number(
                df[col]
            )

    return {
        "dataset": "futures",
        "rows": len(df),
        "kospi200_like_rows": len(kospi200),
        "products": (
            " | ".join(
                sorted(
                    df["PROD_NM"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
            if "PROD_NM" in df.columns
            else None
        ),
    }


def audit_options(
    df: pd.DataFrame,
) -> dict[str, Any]:

    print("\n" + "=" * 80)
    print("OPTIONS")
    print("=" * 80)

    print(f"rows: {len(df):,}")

    if "PROD_NM" in df.columns:
        print("\n[unique PROD_NM]")
        print(
            df["PROD_NM"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_string(index=False)
        )

    if "RGHT_TP_NM" in df.columns:
        print("\n[CALL / PUT distribution]")
        print(
            df["RGHT_TP_NM"]
            .value_counts(dropna=False)
            .to_string()
        )

    kospi200 = find_kospi200_rows(df)

    print(
        f"\n[KOSPI200-like rows] {len(kospi200):,}"
    )

    show_cols = [
        col
        for col in [
            "BAS_DD",
            "PROD_NM",
            "RGHT_TP_NM",
            "ISU_CD",
            "ISU_NM",
            "TDD_CLSPRC",
            "IMP_VOLT",
            "NXTDD_BAS_PRC",
            "ACC_TRDVOL",
            "ACC_TRDVAL",
            "ACC_OPNINT_QTY",
        ]
        if col in kospi200.columns
    ]

    if not kospi200.empty:
        print(
            kospi200[show_cols]
            .head(40)
            .to_string(index=False)
        )

    numeric_candidates = [
        "TDD_CLSPRC",
        "IMP_VOLT",
        "NXTDD_BAS_PRC",
        "ACC_TRDVOL",
        "ACC_TRDVAL",
        "ACC_OPNINT_QTY",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = normalize_number(
                df[col]
            )

    return {
        "dataset": "options",
        "rows": len(df),
        "kospi200_like_rows": len(kospi200),
        "products": (
            " | ".join(
                sorted(
                    df["PROD_NM"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
            if "PROD_NM" in df.columns
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        default="20260917",
        help="기준일자 YYYYMMDD (기본: 20260917)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bas_dd = args.date

    if len(bas_dd) != 8 or not bas_dd.isdigit():
        raise ValueError(
            "--date는 YYYYMMDD 형식이어야 합니다."
        )

    summaries = []

    for name, endpoint in ENDPOINTS.items():
        print(
            f"\n[REQUEST] {name} / {endpoint} / {bas_dd}"
        )

        payload, response = request_endpoint(
            endpoint=endpoint,
            bas_dd=bas_dd,
        )

        raw_path = save_raw(
            name=name,
            bas_dd=bas_dd,
            payload=payload,
        )

        df = payload_to_df(
            payload
        )

        print(
            f"HTTP {response.status_code}"
            f" | raw={raw_path}"
        )

        if name == "futures":
            summary = audit_futures(
                df.copy()
            )
        else:
            summary = audit_options(
                df.copy()
            )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        INTERIM_DIR
        / f"krx_derivatives_audit_{bas_dd}.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nsummary: {summary_path}"
    )

    print(
        "\n다음 단계:"
        "\n1) 실제 PROD_NM / ISU_NM 구조 확인"
        "\n2) KOSPI200 front-month 선물 식별 규칙 확정"
        "\n3) basis = futures price - spot price 설계"
        "\n4) OI / volume 변화 feature 설계"
        "\n5) 옵션 CALL/PUT volume·OI 및 IV aggregate 설계"
        "\n"
        "\n아직 투자자별 파생 포지션은 포함하지 않습니다."
    )


if __name__ == "__main__":
    main()
