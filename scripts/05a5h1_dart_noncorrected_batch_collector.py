from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H1. Non-Corrected Period Batch Financial Collector
#
# 목적
# ------------------------------------------------------------
# 정정공시가 없었던 정기보고서 기간만 골라 OpenDART의
# "다중회사 주요계정(fnlttMultiAcnt)" API로 batch 수집한다.
#
# PIT 핵심 규칙
# ------------------------------------------------------------
# 1) corrected period는 이 단계에서 절대 수집하지 않는다.
# 2) 실제 정기보고서만 universe에 포함하며 제출기한 연장신고서는 제외.
# 3) non-corrected period는 정기보고서 chain상 receipt가 1개인 기간만 사용.
# 4) API가 반환한 rcept_no가 우리가 기대한 original receipt와
#    정확히 일치할 때만 pit_verified=True.
# 5) receipt mismatch는 값을 사용하지 않고 별도 QA 대상으로 남긴다.
# 6) rcept_dt와 receipt-number 날짜를 모두 보존하고 불일치를 QA한다.
# 7) 실제 모델 사용 가능일(available_date)은 아직 확정하지 않는다.
#    trading calendar / decision clock alignment는 05D2에서 수행.
#
# 왜 fnlttMultiAcnt?
# ------------------------------------------------------------
# OpenDART 공식 API는 corp_code를 쉼표로 묶어 최대 100개 회사까지
# 주요계정을 한 번에 조회할 수 있다. non-corrected 기간 약 1.7만개를
# 단일회사 전체재무제표 API로 개별 호출하는 것보다 요청 수가 훨씬 적다.
#
# 이번 단계는 "주요계정 raw collection"까지만 한다.
# parent-attributable 등 부족한 metric은 H2 coverage audit 후
# 필요한 receipt만 fnlttSinglAcntAll로 selective fallback한다.
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/
#   - 05A5-A correction prevalence audit가 만든 전체 정기보고서 universe
#     (스크립트가 schema 기반으로 자동 탐색)
#   - dart_correction_chains.csv
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_target_periods.parquet
#   dart_noncorrected_multi_account_rows.parquet
#   dart_noncorrected_multi_account_manifest.parquet
#
# CSV도 함께 저장.
#
# 실행
# ------------------------------------------------------------
# 소량 테스트:
#   python scripts\05a5h1_dart_noncorrected_batch_collector.py --limit-batches 2
#
# 전체:
#   python scripts\05a5h1_dart_noncorrected_batch_collector.py --full
#
# API error 재시도:
#   python scripts\05a5h1_dart_noncorrected_batch_collector.py --retry-errors
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "dart"

CORRECTION_CHAIN_PATH = (
    INTERIM_DIR
    / "dart_correction_chains.csv"
)

TARGETS_PARQUET = (
    INTERIM_DIR
    / "dart_noncorrected_target_periods.parquet"
)
TARGETS_CSV = (
    INTERIM_DIR
    / "dart_noncorrected_target_periods.csv"
)

ROWS_PARQUET = (
    INTERIM_DIR
    / "dart_noncorrected_multi_account_rows.parquet"
)
ROWS_CSV = (
    INTERIM_DIR
    / "dart_noncorrected_multi_account_rows.csv"
)

MANIFEST_PARQUET = (
    INTERIM_DIR
    / "dart_noncorrected_multi_account_manifest.parquet"
)
MANIFEST_CSV = (
    INTERIM_DIR
    / "dart_noncorrected_multi_account_manifest.csv"
)

API_URL = (
    "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
)

MAX_CORPS_PER_REQUEST = 100
CHECKPOINT_EVERY_BATCHES = 5
REQUEST_SLEEP_SECONDS = 0.20
MAX_RETRIES = 5

REPORT_CODE_MAP = {
    "사업보고서": "11011",
    "반기보고서": "11012",
}

EXPECTED_UNIVERSE_PERIODS_MIN = 10_000


# ------------------------------------------------------------
# Generic file readers
# ------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype={
                "stock_code": str,
                "corp_code": str,
                "rcept_no": str,
            },
            low_memory=False,
        )

    raise ValueError(path)


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "stock_code" in df.columns:
        df["stock_code"] = (
            df["stock_code"]
            .astype("string")
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.zfill(6)
        )

    if "corp_code" in df.columns:
        df["corp_code"] = (
            df["corp_code"]
            .astype("string")
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.zfill(8)
        )

    if "rcept_no" in df.columns:
        df["rcept_no"] = (
            df["rcept_no"]
            .astype("string")
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            .str.strip()
        )

    if "rcept_dt" in df.columns:
        df["rcept_dt"] = pd.to_datetime(
            df["rcept_dt"],
            errors="coerce",
        )

    return df


# ------------------------------------------------------------
# Period canonicalization
# ------------------------------------------------------------

NON_REGULAR_REPORT_TOKENS = [
    "제출기한연장신고서",
    "제출기한연장",
]


def canonical_period_key(
    period_key: Any,
    report_nm: Any = None,
) -> str | None:
    """
    실제 정기보고서만 canonical period로 인정한다.

    허용 예:
      사업보고서 (2024.12)
      [기재정정]사업보고서 (2024.12)
      반기보고서 (2022.06)
      분기보고서 (2023.03)

    제외 예:
      사업보고서제출기한연장신고서 (2024.12)
      반기보고서제출기한연장신고서 (2022.06)

    중요한 이유:
    단순히 "사업보고서" substring만 검사하면 제출기한 연장신고서도
    같은 financial period receipt로 잘못 묶여 multi-receipt처럼 보인다.
    """

    candidates = []

    if pd.notna(report_nm):
        candidates.append(
            str(report_nm).strip()
        )

    if pd.notna(period_key):
        candidates.append(
            str(period_key).strip()
        )

    for text in candidates:

        if any(
            token in text
            for token in NON_REGULAR_REPORT_TOKENS
        ):
            continue

        # optional correction prefix 뒤에 실제 정기보고서 제목이 오는 경우만 허용.
        kind_match = re.search(
            r"(?:\[[^\]]*정정[^\]]*\]\s*)?"
            r"(사업보고서|반기보고서|분기보고서)"
            r"\s*\(",
            text,
        )

        if kind_match is None:
            continue

        date_match = re.search(
            r"\((20\d{2})\s*[.\-/]\s*(\d{1,2})\)",
            text,
        )

        if date_match is None:
            # 괄호가 없는 legacy 표현까지 최소한 지원.
            date_match = re.search(
                r"(20\d{2})\s*[.\-/]\s*(\d{1,2})",
                text,
            )

        if date_match is None:
            continue

        kind = kind_match.group(1)
        year = int(
            date_match.group(1)
        )
        month = int(
            date_match.group(2)
        )

        return (
            f"{kind}|{year:04d}.{month:02d}"
        )

    return None


def rcept_date_from_number(
    rcept_no: Any,
) -> pd.Timestamp:
    """
    receipt number 앞 8자리 날짜를 QA reference로 보존한다.
    실제 available_date로 바로 사용하지는 않는다.
    """

    if pd.isna(
        rcept_no
    ):
        return pd.NaT

    text = str(
        rcept_no
    ).strip()

    if not re.fullmatch(
        r"\d{14}",
        text,
    ):
        return pd.NaT

    return pd.to_datetime(
        text[:8],
        format="%Y%m%d",
        errors="coerce",
    )


def period_year_month(
    canonical_key: str,
) -> tuple[int, int]:

    match = re.search(
        r"\|(20\d{2})\.(\d{2})$",
        canonical_key,
    )

    if match is None:
        raise ValueError(
            canonical_key
        )

    return (
        int(
            match.group(1)
        ),
        int(
            match.group(2)
        ),
    )


def infer_fiscal_year_end_month(
    universe: pd.DataFrame,
) -> pd.Series:
    """
    각 종목의 사업보고서 period month 최빈값을 결산월 proxy로 사용.
    """

    annual = universe.loc[
        universe[
            "canonical_period_key"
        ]
        .astype("string")
        .str.startswith(
            "사업보고서|",
            na=False,
        )
    ].copy()

    if annual.empty:
        return pd.Series(
            dtype="Int64"
        )

    annual[
        "period_month"
    ] = (
        annual[
            "canonical_period_key"
        ]
        .str.extract(
            r"\|20\d{2}\.(\d{2})$"
        )[0]
        .astype(
            "Int64"
        )
    )

    fiscal = (
        annual.dropna(
            subset=[
                "period_month",
            ]
        )
        .groupby(
            "stock_code"
        )[
            "period_month"
        ]
        .agg(
            lambda s:
            int(
                s.mode().iloc[0]
            )
        )
    )

    return fiscal


def infer_reprt_code(
    canonical_key: str,
    fiscal_year_end_month: int | None,
    existing_code: Any = None,
) -> str | None:

    if pd.notna(
        existing_code
    ):
        code = str(
            existing_code
        ).replace(
            ".0",
            "",
        )

        if code in {
            "11011",
            "11012",
            "11013",
            "11014",
        }:
            return code

    kind = canonical_key.split(
        "|",
        1,
    )[0]

    if kind in REPORT_CODE_MAP:
        return REPORT_CODE_MAP[
            kind
        ]

    if kind != "분기보고서":
        return None

    _, month = period_year_month(
        canonical_key
    )

    if fiscal_year_end_month is None:
        # December fiscal-year fallback only when unambiguous.
        if month == 3:
            return "11013"
        if month == 9:
            return "11014"
        return None

    q1_month = (
        (
            int(
                fiscal_year_end_month
            )
            + 3
            - 1
        )
        % 12
        + 1
    )

    q3_month = (
        (
            int(
                fiscal_year_end_month
            )
            + 9
            - 1
        )
        % 12
        + 1
    )

    if month == q1_month:
        return "11013"

    if month == q3_month:
        return "11014"

    return None


# ------------------------------------------------------------
# Universe auto-discovery
# ------------------------------------------------------------

def discover_full_period_universe() -> tuple[
    Path,
    pd.DataFrame,
]:
    """
    05A5-A에서 만든 전체 regular-report filing universe 파일을
    파일명 대신 schema + distinct period count로 탐색한다.

    최소 필요:
      stock_code, rcept_no, rcept_dt
      + period_key 또는 report_nm
    """

    candidates = []

    paths = (
        list(
            INTERIM_DIR.glob(
                "*.parquet"
            )
        )
        + list(
            INTERIM_DIR.glob(
                "*.csv"
            )
        )
    )

    skip_tokens = [
        "noncorrected_",
        "corrected_pit_",
        "balance_",
        "source_full_manifest",
        "receipt_values",
        "validity_intervals",
    ]

    for path in paths:
        lower = path.name.lower()

        if any(
            token in lower
            for token in skip_tokens
        ):
            continue

        try:
            df = read_table(
                path
            )
        except Exception:
            continue

        cols = set(
            df.columns
        )

        if not {
            "stock_code",
            "rcept_no",
            "rcept_dt",
        }.issubset(
            cols
        ):
            continue

        if (
            "period_key" not in cols
            and "report_nm" not in cols
        ):
            continue

        work = normalize_id_columns(
            df
        )

        work[
            "canonical_period_key"
        ] = work.apply(
            lambda r:
            canonical_period_key(
                r.get(
                    "period_key"
                ),
                r.get(
                    "report_nm"
                ),
            ),
            axis=1,
        )

        valid = work.dropna(
            subset=[
                "stock_code",
                "rcept_no",
                "rcept_dt",
                "canonical_period_key",
            ]
        )

        period_count = (
            valid[
                [
                    "stock_code",
                    "canonical_period_key",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        candidates.append(
            (
                period_count,
                len(
                    valid
                ),
                path,
                valid,
            )
        )

    if not candidates:
        raise RuntimeError(
            "전체 DART report-period universe 후보 파일을 찾지 못했습니다."
        )

    candidates.sort(
        key=lambda x:
        (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    period_count, _, path, df = (
        candidates[0]
    )

    print(
        "\n[Universe auto-discovery candidates]"
    )

    for (
        count,
        rows,
        candidate_path,
        _,
    ) in candidates[:10]:
        print(
            f"{count:>7,} periods | "
            f"{rows:>7,} rows | "
            f"{candidate_path.name}"
        )

    if (
        period_count
        < EXPECTED_UNIVERSE_PERIODS_MIN
    ):
        raise RuntimeError(
            "자동 선택된 universe가 너무 작습니다. "
            f"distinct periods={period_count:,}, "
            f"path={path}"
        )

    print(
        f"\nSelected universe: "
        f"{path}"
    )

    return (
        path,
        df,
    )


def discover_corp_code_map() -> pd.DataFrame:
    """
    universe에 corp_code가 없거나 일부 missing일 때 보조 mapping 탐색.
    """

    best = None

    search_roots = [
        INTERIM_DIR,
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dart",
    ]

    for root in search_roots:
        if not root.exists():
            continue

        for pattern in [
            "*.parquet",
            "*.csv",
        ]:
            for path in root.rglob(
                pattern
            ):
                try:
                    df = read_table(
                        path
                    )
                except Exception:
                    continue

                if not {
                    "stock_code",
                    "corp_code",
                }.issubset(
                    set(
                        df.columns
                    )
                ):
                    continue

                mapping = (
                    normalize_id_columns(
                        df[
                            [
                                "stock_code",
                                "corp_code",
                            ]
                        ]
                    )
                    .dropna()
                    .drop_duplicates(
                        "stock_code",
                    )
                )

                score = (
                    mapping[
                        "stock_code"
                    ]
                    .nunique()
                )

                if (
                    best is None
                    or score
                    > best[0]
                ):
                    best = (
                        score,
                        path,
                        mapping,
                    )

    if best is None:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "corp_code",
            ]
        )

    print(
        f"\nCorp-code map: "
        f"{best[1]} "
        f"({best[0]:,} stocks)"
    )

    return best[2]


# ------------------------------------------------------------
# Target construction
# ------------------------------------------------------------

def build_targets() -> pd.DataFrame:

    if not CORRECTION_CHAIN_PATH.exists():
        raise FileNotFoundError(
            CORRECTION_CHAIN_PATH
        )

    _, universe = (
        discover_full_period_universe()
    )

    correction = normalize_id_columns(
        pd.read_csv(
            CORRECTION_CHAIN_PATH,
            dtype={
                "stock_code": str,
                "corp_code": str,
                "rcept_no": str,
            },
            low_memory=False,
        )
    )

    correction[
        "canonical_period_key"
    ] = correction.apply(
        lambda r:
        canonical_period_key(
            r.get(
                "period_key"
            ),
            r.get(
                "report_nm"
            ),
        ),
        axis=1,
    )

    corrected_keys = set(
        zip(
            correction[
                "stock_code"
            ].astype(str),
            correction[
                "canonical_period_key"
            ].astype(str),
        )
    )

    universe = normalize_id_columns(
        universe
    )

    universe[
        "canonical_period_key"
    ] = universe.apply(
        lambda r:
        canonical_period_key(
            r.get(
                "period_key"
            ),
            r.get(
                "report_nm"
            ),
        ),
        axis=1,
    )

    universe = universe.dropna(
        subset=[
            "stock_code",
            "rcept_no",
            "rcept_dt",
            "canonical_period_key",
        ]
    ).copy()

    universe[
        "is_corrected_period"
    ] = [
        (
            str(
                stock
            ),
            str(
                period
            ),
        )
        in corrected_keys
        for stock, period
        in zip(
            universe[
                "stock_code"
            ],
            universe[
                "canonical_period_key"
            ],
        )
    ]

    noncorrected = universe.loc[
        ~universe[
            "is_corrected_period"
        ]
    ].copy()

    # non-corrected라면 regular-report receipt가 하나여야 한다.
    receipt_counts = (
        noncorrected.groupby(
            [
                "stock_code",
                "canonical_period_key",
            ]
        )[
            "rcept_no"
        ]
        .nunique()
        .rename(
            "period_receipt_count"
        )
        .reset_index()
    )

    noncorrected = noncorrected.merge(
        receipt_counts,
        on=[
            "stock_code",
            "canonical_period_key",
        ],
        how="left",
    )

    ambiguous = noncorrected.loc[
        noncorrected[
            "period_receipt_count"
        ]
        .gt(
            1
        )
    ].copy()

    if not ambiguous.empty:
        ambiguous_path = (
            INTERIM_DIR
            / "dart_noncorrected_ambiguous_periods.csv"
        )

        ambiguous.to_csv(
            ambiguous_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nWARNING: corrected chain 밖에서 "
            f"multiple receipts가 있는 period "
            f"{ambiguous[['stock_code','canonical_period_key']].drop_duplicates().shape[0]:,}개."
        )
        print(
            f"수집 대상에서 제외: {ambiguous_path}"
        )

    targets = noncorrected.loc[
        noncorrected[
            "period_receipt_count"
        ]
        .eq(
            1
        )
    ].copy()

    # 동일 period의 중복 행 제거
    targets = (
        targets.sort_values(
            [
                "stock_code",
                "canonical_period_key",
                "rcept_dt",
                "rcept_no",
            ]
        )
        .drop_duplicates(
            subset=[
                "stock_code",
                "canonical_period_key",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    # corp_code 보강
    if (
        "corp_code" not in targets.columns
        or targets[
            "corp_code"
        ].isna().any()
    ):
        corp_map = (
            discover_corp_code_map()
        )

        if not corp_map.empty:
            existing = (
                targets[
                    "corp_code"
                ]
                if "corp_code"
                in targets.columns
                else pd.Series(
                    pd.NA,
                    index=targets.index,
                    dtype="string",
                )
            )

            temp = targets[
                [
                    "stock_code",
                ]
            ].merge(
                corp_map,
                on="stock_code",
                how="left",
            )[
                "corp_code"
            ]

            targets[
                "corp_code"
            ] = existing.fillna(
                temp
            )

    if "corp_code" not in targets.columns:
        raise RuntimeError(
            "corp_code를 확보하지 못했습니다."
        )

    targets[
        "corp_code"
    ] = (
        targets[
            "corp_code"
        ]
        .astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(
            8
        )
    )

    fiscal_month = (
        infer_fiscal_year_end_month(
            universe
        )
    )

    targets[
        "fiscal_year_end_month"
    ] = targets[
        "stock_code"
    ].map(
        fiscal_month
    )

    if "reprt_code" not in targets.columns:
        targets[
            "reprt_code"
        ] = pd.NA

    targets[
        "reprt_code_inferred"
    ] = targets.apply(
        lambda r:
        infer_reprt_code(
            r[
                "canonical_period_key"
            ],
            (
                int(
                    r[
                        "fiscal_year_end_month"
                    ]
                )
                if pd.notna(
                    r[
                        "fiscal_year_end_month"
                    ]
                )
                else None
            ),
            r.get(
                "reprt_code"
            ),
        ),
        axis=1,
    )

    targets[
        "bsns_year"
    ] = targets[
        "canonical_period_key"
    ].apply(
        lambda key:
        period_year_month(
            key
        )[0]
    )

    # 최종 expected receipt
    targets[
        "expected_rcept_no"
    ] = targets[
        "rcept_no"
    ].astype(
        "string"
    )

    # receipt 번호 자체의 날짜와 universe rcept_dt를 모두 보존한다.
    # 둘이 다르더라도 여기서 어느 한쪽으로 덮어쓰지 않는다.
    targets[
        "rcept_no_date"
    ] = targets[
        "expected_rcept_no"
    ].apply(
        rcept_date_from_number
    )

    targets[
        "rcept_date_mismatch"
    ] = (
        targets[
            "rcept_no_date"
        ].dt.normalize()
        != targets[
            "rcept_dt"
        ].dt.normalize()
    )

    targets.loc[
        targets[
            "rcept_no_date"
        ].isna()
        | targets[
            "rcept_dt"
        ].isna(),
        "rcept_date_mismatch",
    ] = pd.NA

    unresolved = targets.loc[
        targets[
            "reprt_code_inferred"
        ].isna()
        | targets[
            "corp_code"
        ].isna()
    ].copy()

    if not unresolved.empty:
        unresolved_path = (
            INTERIM_DIR
            / "dart_noncorrected_unresolved_target_mapping.csv"
        )

        unresolved.to_csv(
            unresolved_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nWARNING: reprt_code/corp_code unresolved "
            f"{len(unresolved):,} periods -> 제외"
        )
        print(
            unresolved_path
        )

    targets = targets.loc[
        targets[
            "reprt_code_inferred"
        ].notna()
        & targets[
            "corp_code"
        ].notna()
    ].copy()

    keep_cols = [
        "stock_code",
        "corp_code",
        "corp_name",
        "canonical_period_key",
        "period_key",
        "report_nm",
        "expected_rcept_no",
        "rcept_dt",
        "rcept_no_date",
        "rcept_date_mismatch",
        "bsns_year",
        "reprt_code_inferred",
        "fiscal_year_end_month",
        "period_receipt_count",
    ]

    keep_cols = [
        c
        for c in keep_cols
        if c in targets.columns
    ]

    targets = (
        targets[
            keep_cols
        ]
        .rename(
            columns={
                "reprt_code_inferred":
                "reprt_code",
            }
        )
        .sort_values(
            [
                "bsns_year",
                "reprt_code",
                "stock_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    targets.to_parquet(
        TARGETS_PARQUET,
        index=False,
    )

    targets.to_csv(
        TARGETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Target construction]"
    )

    print(
        f"all distinct periods : "
        f"{universe[['stock_code','canonical_period_key']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"corrected periods    : "
        f"{len(corrected_keys):,}"
    )

    print(
        f"non-corrected targets: "
        f"{len(targets):,}"
    )

    print(
        "\n[Target report codes]"
    )

    print(
        targets[
            "reprt_code"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n[Receipt-date QA]"
    )

    mismatch_count = int(
        targets[
            "rcept_date_mismatch"
        ]
        .fillna(False)
        .sum()
    )

    print(
        f"rcept_no date != universe rcept_dt: "
        f"{mismatch_count:,}"
    )

    if mismatch_count:
        print(
            targets.loc[
                targets[
                    "rcept_date_mismatch"
                ]
                .fillna(False),
                [
                    "stock_code",
                    "canonical_period_key",
                    "expected_rcept_no",
                    "rcept_dt",
                    "rcept_no_date",
                ],
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    return targets


# ------------------------------------------------------------
# Batch helpers
# ------------------------------------------------------------

def chunks(
    items: list[str],
    size: int,
) -> Iterable[
    list[str]
]:
    for i in range(
        0,
        len(
            items
        ),
        size,
    ):
        yield items[
            i:
            i + size
        ]


def build_batches(
    targets: pd.DataFrame,
) -> list[
    dict[str, Any]
]:

    batches = []

    for (
        bsns_year,
        reprt_code,
    ), group in targets.groupby(
        [
            "bsns_year",
            "reprt_code",
        ],
        sort=True,
    ):

        corp_codes = (
            group[
                "corp_code"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        for batch_seq, corp_chunk in enumerate(
            chunks(
                corp_codes,
                MAX_CORPS_PER_REQUEST,
            ),
            start=1,
        ):
            batch_targets = group.loc[
                group[
                    "corp_code"
                ].astype(str)
                .isin(
                    corp_chunk
                )
            ].copy()

            batch_id = (
                f"{int(bsns_year)}_"
                f"{reprt_code}_"
                f"{batch_seq:03d}"
            )

            batches.append(
                {
                    "batch_id": batch_id,
                    "bsns_year": int(
                        bsns_year
                    ),
                    "reprt_code": str(
                        reprt_code
                    ),
                    "corp_codes": corp_chunk,
                    "targets": batch_targets,
                }
            )

    return batches


# ------------------------------------------------------------
# OpenDART request
# ------------------------------------------------------------

def request_multi_account(
    session: requests.Session,
    api_key: str,
    corp_codes: list[str],
    bsns_year: int,
    reprt_code: str,
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
                    "corp_code": ",".join(
                        corp_codes
                    ),
                    "bsns_year": str(
                        bsns_year
                    ),
                    "reprt_code": str(
                        reprt_code
                    ),
                },
                timeout=90,
            )

            response.raise_for_status()

            payload = response.json()

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

            if status == "013":
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

            # Other API statuses are deterministic enough to preserve.
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
                20,
            )

            print(
                f"  request error -> retry {wait}s: "
                f"{repr(exc)}"
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
# Storage normalization
# ------------------------------------------------------------

def normalize_raw_rows(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
) -> pd.DataFrame:

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    df[
        "_batch_id"
    ] = batch[
        "batch_id"
    ]

    df[
        "_query_bsns_year"
    ] = batch[
        "bsns_year"
    ]

    df[
        "_query_reprt_code"
    ] = batch[
        "reprt_code"
    ]

    for col in [
        "stock_code",
        "rcept_no",
        "reprt_code",
        "bsns_year",
        "account_nm",
        "fs_div",
        "fs_nm",
        "sj_div",
        "sj_nm",
        "thstrm_nm",
        "thstrm_dt",
        "thstrm_amount",
        "thstrm_add_amount",
        "frmtrm_nm",
        "frmtrm_dt",
        "frmtrm_amount",
        "frmtrm_add_amount",
        "bfefrmtrm_nm",
        "bfefrmtrm_dt",
        "bfefrmtrm_amount",
        "ord",
        "currency",
    ]:
        if col in df.columns:
            df[
                col
            ] = (
                df[
                    col
                ]
                .astype(
                    "string"
                )
            )

    if "stock_code" in df.columns:
        df[
            "stock_code"
        ] = (
            df[
                "stock_code"
            ]
            .str.zfill(
                6
            )
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


def load_existing() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

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


def save_outputs(
    manifest: pd.DataFrame,
    rows: pd.DataFrame,
) -> None:

    if not manifest.empty:
        manifest = (
            manifest.sort_values(
                [
                    "bsns_year",
                    "reprt_code",
                    "stock_code",
                ]
            )
            .drop_duplicates(
                subset=[
                    "expected_rcept_no",
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

    manifest.to_parquet(
        MANIFEST_PARQUET,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
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


# ------------------------------------------------------------
# Batch result -> period manifest
# ------------------------------------------------------------

def evaluate_batch_targets(
    batch_targets: pd.DataFrame,
    api_rows: pd.DataFrame,
    api_status: str | None,
    api_message: str | None,
    api_error: str | None,
    batch_id: str,
) -> pd.DataFrame:

    records = []

    for _, target in (
        batch_targets.iterrows()
    ):

        expected = str(
            target[
                "expected_rcept_no"
            ]
        )

        stock_code = str(
            target[
                "stock_code"
            ]
        )

        matched_receipt_rows = (
            api_rows.loc[
                api_rows[
                    "rcept_no"
                ].astype(str)
                .eq(
                    expected
                )
            ]
            if (
                not api_rows.empty
                and "rcept_no"
                in api_rows.columns
            )
            else pd.DataFrame()
        )

        same_stock_rows = (
            api_rows.loc[
                api_rows[
                    "stock_code"
                ].astype(str)
                .eq(
                    stock_code
                )
            ]
            if (
                not api_rows.empty
                and "stock_code"
                in api_rows.columns
            )
            else pd.DataFrame()
        )

        returned_receipts = (
            sorted(
                set(
                    same_stock_rows[
                        "rcept_no"
                    ]
                    .dropna()
                    .astype(str)
                )
            )
            if (
                not same_stock_rows.empty
                and "rcept_no"
                in same_stock_rows.columns
            )
            else []
        )

        if api_error is not None:
            status = "api_error"
            pit_verified = False

        elif not matched_receipt_rows.empty:
            status = "available"
            pit_verified = True

        elif returned_receipts:
            status = "receipt_mismatch"
            pit_verified = False

        else:
            status = "no_data"
            pit_verified = False

        records.append(
            {
                "batch_id": batch_id,
                "stock_code": stock_code,
                "corp_code": target[
                    "corp_code"
                ],
                "corp_name": target.get(
                    "corp_name"
                ),
                "canonical_period_key": target[
                    "canonical_period_key"
                ],
                "period_key": target.get(
                    "period_key"
                ),
                "report_nm": target.get(
                    "report_nm"
                ),
                "expected_rcept_no": expected,
                "rcept_dt": target[
                    "rcept_dt"
                ],
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
                "collection_status": status,
                "pit_verified": pit_verified,
                "returned_rcept_nos": (
                    " | ".join(
                        returned_receipts
                    )
                ),
                "api_row_count": int(
                    len(
                        matched_receipt_rows
                    )
                ),
                "api_status": api_status,
                "api_message": api_message,
                "api_error": api_error,
            }
        )

    return pd.DataFrame(
        records
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit-batches",
        type=int,
        default=2,
        help=(
            "테스트용 batch 수. "
            "--full이면 무시."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 non-corrected target batch 수집",
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "api_error/receipt_mismatch만 다시 조회"
        ),
    )

    parser.add_argument(
        "--rebuild-targets",
        action="store_true",
        help=(
            "기존 target parquet이 있어도 period universe를 다시 계산"
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = parse_args()

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

    if (
        TARGETS_PARQUET.exists()
        and not args.rebuild_targets
    ):
        targets = normalize_id_columns(
            pd.read_parquet(
                TARGETS_PARQUET
            )
        )

        print(
            f"\n기존 target 사용: "
            f"{TARGETS_PARQUET}"
        )
        print(
            f"targets: {len(targets):,}"
        )

    else:
        targets = build_targets()

    existing_manifest, existing_rows = (
        load_existing()
    )

    if not existing_manifest.empty:
        existing_manifest[
            "expected_rcept_no"
        ] = (
            existing_manifest[
                "expected_rcept_no"
            ]
            .astype(str)
        )

    batches = build_batches(
        targets
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-H1 NON-CORRECTED BATCH FINANCIAL COLLECTION"
    )
    print(
        "=" * 80
    )

    print(
        f"\ntarget periods : "
        f"{len(targets):,}"
    )

    print(
        f"total batches  : "
        f"{len(batches):,}"
    )

    # --------------------------------------------------------
    # Resume filtering
    # --------------------------------------------------------

    done_receipts: set[str] = set()

    retry_receipts: set[str] | None = None

    if not existing_manifest.empty:

        if args.retry_errors:
            retry_receipts = set(
                existing_manifest.loc[
                    existing_manifest[
                        "collection_status"
                    ].isin(
                        [
                            "api_error",
                            "receipt_mismatch",
                        ]
                    ),
                    "expected_rcept_no",
                ].astype(str)
            )

        else:
            done_receipts = set(
                existing_manifest.loc[
                    existing_manifest[
                        "collection_status"
                    ].isin(
                        [
                            "available",
                            "no_data",
                        ]
                    ),
                    "expected_rcept_no",
                ].astype(str)
            )

    work_batches = []

    for batch in batches:
        bt = batch[
            "targets"
        ].copy()

        if retry_receipts is not None:
            bt = bt.loc[
                bt[
                    "expected_rcept_no"
                ].astype(str)
                .isin(
                    retry_receipts
                )
            ].copy()
        else:
            bt = bt.loc[
                ~bt[
                    "expected_rcept_no"
                ].astype(str)
                .isin(
                    done_receipts
                )
            ].copy()

        if bt.empty:
            continue

        batch_copy = dict(
            batch
        )

        batch_copy[
            "targets"
        ] = bt

        batch_copy[
            "corp_codes"
        ] = (
            bt[
                "corp_code"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        work_batches.append(
            batch_copy
        )

    if not args.full:
        work_batches = work_batches[
            :
            args.limit_batches
        ]

    print(
        f"remaining batches: "
        f"{len(work_batches):,}"
    )

    if not work_batches:
        print(
            "\n수집 대상이 없습니다."
        )
        return

    session = requests.Session()

    manifest_state = (
        existing_manifest.copy()
        if not existing_manifest.empty
        else pd.DataFrame()
    )

    row_state = (
        existing_rows.copy()
        if not existing_rows.empty
        else pd.DataFrame()
    )

    for batch_idx, batch in enumerate(
        work_batches,
        start=1,
    ):

        print(
            f"\n[{batch_idx}/{len(work_batches)}] "
            f"{batch['batch_id']} | "
            f"companies={len(batch['corp_codes'])} | "
            f"periods={len(batch['targets'])}"
        )

        try:
            result = request_multi_account(
                session=session,
                api_key=api_key,
                corp_codes=batch[
                    "corp_codes"
                ],
                bsns_year=batch[
                    "bsns_year"
                ],
                reprt_code=batch[
                    "reprt_code"
                ],
            )

        except RuntimeError as exc:
            if "020" in str(
                exc
            ):
                print(
                    "\nOpenDART 요청 제한(020)에 도달했습니다. "
                    "현재 checkpoint를 저장하고 종료합니다."
                )

                save_outputs(
                    manifest_state,
                    row_state,
                )

                raise

            result = {
                "status": None,
                "message": None,
                "rows": [],
                "error": repr(
                    exc
                ),
            }

        api_rows = normalize_raw_rows(
            result[
                "rows"
            ],
            batch,
        )

        period_manifest = (
            evaluate_batch_targets(
                batch_targets=batch[
                    "targets"
                ],
                api_rows=api_rows,
                api_status=result[
                    "status"
                ],
                api_message=result[
                    "message"
                ],
                api_error=result[
                    "error"
                ],
                batch_id=batch[
                    "batch_id"
                ],
            )
        )

        print(
            period_manifest[
                "collection_status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        # replace target receipts in manifest
        touched = set(
            period_manifest[
                "expected_rcept_no"
            ].astype(str)
        )

        if not manifest_state.empty:
            manifest_state = (
                manifest_state.loc[
                    ~manifest_state[
                        "expected_rcept_no"
                    ]
                    .astype(str)
                    .isin(
                        touched
                    )
                ]
            )

        manifest_state = pd.concat(
            [
                manifest_state,
                period_manifest,
            ],
            ignore_index=True,
        )

        # Keep only API rows whose receipt belongs to current target batch.
        if not api_rows.empty:
            expected_batch_receipts = set(
                batch[
                    "targets"
                ][
                    "expected_rcept_no"
                ]
                .astype(str)
            )

            api_rows = api_rows.loc[
                api_rows[
                    "rcept_no"
                ]
                .astype(str)
                .isin(
                    expected_batch_receipts
                )
            ].copy()

            if not row_state.empty:
                row_state = (
                    row_state.loc[
                        ~row_state[
                            "rcept_no"
                        ]
                        .astype(str)
                        .isin(
                            expected_batch_receipts
                        )
                    ]
                )

            row_state = pd.concat(
                [
                    row_state,
                    api_rows,
                ],
                ignore_index=True,
            )

        if (
            batch_idx
            % CHECKPOINT_EVERY_BATCHES
            == 0
        ):
            save_outputs(
                manifest_state,
                row_state,
            )

            print(
                "  checkpoint saved"
            )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    save_outputs(
        manifest_state,
        row_state,
    )

    # --------------------------------------------------------
    # QA summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "NON-CORRECTED COLLECTION SUMMARY"
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
        "\n[PIT verification]"
    )

    print(
        manifest_state[
            "pit_verified"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[API row counts]"
    )

    print(
        manifest_state[
            "api_row_count"
        ]
        .describe()
        .to_string()
    )

    mismatches = (
        manifest_state.loc[
            manifest_state[
                "collection_status"
            ].eq(
                "receipt_mismatch"
            )
        ]
    )

    print(
        "\n[Receipt mismatches]"
    )

    if mismatches.empty:
        print(
            "0"
        )
    else:
        print(
            mismatches[
                [
                    "stock_code",
                    "canonical_period_key",
                    "expected_rcept_no",
                    "returned_rcept_nos",
                    "rcept_dt",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    no_data = (
        manifest_state.loc[
            manifest_state[
                "collection_status"
            ].eq(
                "no_data"
            )
        ]
    )

    print(
        "\n[No data]"
    )

    print(
        f"{len(no_data):,}"
    )

    print(
        f"\nTargets : {TARGETS_PARQUET}"
    )
    print(
        f"Rows    : {ROWS_PARQUET}"
    )
    print(
        f"Manifest: {MANIFEST_PARQUET}"
    )

    print(
        "\n다음 단계:"
        "\n05A5-H2 major-account metric coverage audit"
        "\n→ 부족한 receipt만 fnlttSinglAcntAll selective fallback"
        "\n→ corrected reconciled + non-corrected 결합"
    )


if __name__ == "__main__":
    main()
