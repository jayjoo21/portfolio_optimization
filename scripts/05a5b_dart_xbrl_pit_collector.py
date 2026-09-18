from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-B. OpenDART Strict-PIT XBRL Source Collector
#
# 목적
# ------------------------------------------------------------
# correction chain이 존재하는 보고기간만
# 각 rcept_no(접수번호)의 원본 XBRL ZIP을 그대로 보존한다.
#
# WHY
# - OpenDART 일반 재무 API는 이후 정정 시 현재 수치가 바뀔 수 있음
# - corrected period의 최초 제출 당시 값을 복원하려면
#   접수번호별 원본 XBRL이 필요
#
# 이번 단계
# - ZIP 원본 다운로드
# - SHA256 기록
# - ZIP member 목록 audit
# - 원본/정정 제출 순서(sequence) 기록
#
# 아직 XBRL fact parsing은 하지 않는다.
# 실제 ZIP 구조를 확인한 뒤 05A5-C parser에서 수행.
#
# INPUT
# data/interim/dart/dart_correction_chains.csv
#
# OUTPUT
# data/raw/dart/xbrl_pit/<stock_code>/<period_key>/*.zip
# data/interim/dart/dart_xbrl_pit_download_manifest.csv
#
# 실행
# ------------------------------------------------------------
# 안전한 구조 audit:
#   python scripts\05a5b_dart_xbrl_pit_collector.py --sample-periods 5
#
# 전체 correction-period:
#   python scripts\05a5b_dart_xbrl_pit_collector.py --full
# ============================================================


BASE_URL = "https://opendart.fss.or.kr/api"
XBRL_ENDPOINT = "fnlttXbrl.xml"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHAINS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
    / "dart_correction_chains.csv"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "xbrl_pit"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

MANIFEST_CSV = (
    INTERIM_DIR
    / "dart_xbrl_pit_download_manifest.csv"
)

MANIFEST_PARQUET = (
    INTERIM_DIR
    / "dart_xbrl_pit_download_manifest.parquet"
)

RAW_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv(PROJECT_ROOT / ".env")

DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise RuntimeError(
        "DART_API_KEY가 없습니다. 프로젝트 루트 .env를 확인하세요."
    )


REQUEST_SLEEP_SECONDS = 0.22
MAX_RETRIES = 5


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_name(value: Any) -> str:
    text = str(value).strip()

    text = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        text,
    )

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    return text[:120]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def parse_error_payload(
    content: bytes,
) -> tuple[str | None, str | None]:
    """
    OpenDART binary API가 오류 시 XML/text를 반환하는 경우 진단.
    """
    try:
        root = ET.fromstring(content)

        status = root.findtext("status")
        message = root.findtext("message")

        return (
            status,
            message,
        )
    except Exception:
        return (
            None,
            None,
        )


def inspect_zip(
    content: bytes,
) -> dict[str, Any]:

    with zipfile.ZipFile(
        io.BytesIO(content)
    ) as zf:

        members = zf.infolist()

        names = [
            info.filename
            for info in members
        ]

        extensions: dict[str, int] = {}

        for name in names:
            suffix = (
                Path(name)
                .suffix
                .lower()
                or "<none>"
            )

            extensions[suffix] = (
                extensions.get(
                    suffix,
                    0,
                )
                + 1
            )

        # instance 후보:
        # .xbrl 또는 .xml 중 taxonomy/linkbase가 아닌 파일을 우선 표시
        instance_candidates = []

        for name in names:
            lower = name.lower()

            if lower.endswith(".xbrl"):
                instance_candidates.append(
                    name
                )
                continue

            if lower.endswith(".xml"):
                # schema/linkbase/taxonomy 느낌의 파일은 후보 우선순위에서 제외
                excluded_tokens = [
                    "label",
                    "pre",
                    "cal",
                    "def",
                    "ref",
                    "schema",
                    "taxonomy",
                ]

                if not any(
                    token in lower
                    for token in excluded_tokens
                ):
                    instance_candidates.append(
                        name
                    )

        return {
            "zip_member_count": len(names),
            "zip_members": " | ".join(
                names
            ),
            "zip_extensions": " | ".join(
                f"{key}:{value}"
                for key, value
                in sorted(
                    extensions.items()
                )
            ),
            "instance_candidates": " | ".join(
                instance_candidates
            ),
        }


# ------------------------------------------------------------
# Input chain
# ------------------------------------------------------------

def load_chains() -> pd.DataFrame:

    if not CHAINS_PATH.exists():
        raise FileNotFoundError(
            f"correction chain 파일이 없습니다: {CHAINS_PATH}"
        )

    df = pd.read_csv(
        CHAINS_PATH,
        dtype={
            "stock_code": str,
            "corp_code": str,
            "rcept_no": str,
        },
    )

    required = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "report_nm",
        "is_correction",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"chains CSV 필수 컬럼 누락: {missing}\n"
            f"columns={df.columns.tolist()}"
        )

    df["stock_code"] = (
        df["stock_code"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(6)
    )

    df["rcept_no"] = (
        df["rcept_no"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    df["rcept_dt"] = pd.to_datetime(
        df["rcept_dt"],
        errors="raise",
    )

    # CSV boolean이 문자열일 수도 있어 보수적으로 변환
    df["is_correction"] = (
        df["is_correction"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    # 동일 period chain 안에서 접수일/접수번호 순으로 sequence 부여
    df = (
        df
        .sort_values(
            [
                "stock_code",
                "period_key",
                "rcept_dt",
                "rcept_no",
            ]
        )
        .drop_duplicates(
            subset=["rcept_no"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df["filing_sequence"] = (
        df.groupby(
            [
                "stock_code",
                "period_key",
            ]
        )
        .cumcount()
        + 1
    )

    df["chain_submission_count"] = (
        df.groupby(
            [
                "stock_code",
                "period_key",
            ]
        )["rcept_no"]
        .transform("size")
    )

    return df


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

def request_xbrl(
    session: requests.Session,
    rcept_no: str,
) -> tuple[bytes, requests.Response]:

    for attempt in range(
        MAX_RETRIES
    ):
        response = session.get(
            f"{BASE_URL}/{XBRL_ENDPOINT}",
            params={
                "crtfc_key": DART_API_KEY,
                "rcept_no": rcept_no,
            },
            timeout=90,
        )

        if response.status_code == 429:
            wait = min(
                2 ** attempt,
                15,
            )

            print(
                f"[429] {rcept_no} "
                f"→ {wait}s 대기"
            )

            time.sleep(wait)
            continue

        response.raise_for_status()

        content = response.content

        # ZIP magic
        if content[:2] == b"PK":
            return (
                content,
                response,
            )

        status, message = (
            parse_error_payload(
                content
            )
        )

        raise RuntimeError(
            f"XBRL ZIP이 아닙니다. "
            f"rcept_no={rcept_no} "
            f"status={status} "
            f"message={message} "
            f"body={content[:500]!r}"
        )

    raise RuntimeError(
        f"{rcept_no}: 최대 재시도 초과"
    )


def target_zip_path(
    row: pd.Series,
) -> Path:

    period_dir = (
        RAW_ROOT
        / row["stock_code"]
        / safe_name(
            row["period_key"]
        )
    )

    period_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    date_text = (
        pd.Timestamp(
            row["rcept_dt"]
        )
        .strftime("%Y%m%d")
    )

    correction_tag = (
        "correction"
        if bool(
            row["is_correction"]
        )
        else "original"
    )

    filename = (
        f"{date_text}_"
        f"seq{int(row['filing_sequence']):02d}_"
        f"{correction_tag}_"
        f"{row['rcept_no']}.zip"
    )

    return (
        period_dir
        / filename
    )


# ------------------------------------------------------------
# Resume manifest
# ------------------------------------------------------------

def load_existing_manifest() -> pd.DataFrame:

    if MANIFEST_PARQUET.exists():
        df = pd.read_parquet(
            MANIFEST_PARQUET
        )

        if "rcept_no" in df.columns:
            df["rcept_no"] = (
                df["rcept_no"]
                .astype(str)
            )

        return df

    return pd.DataFrame()


def save_manifest(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        df = (
            df
            .sort_values(
                [
                    "stock_code",
                    "period_key",
                    "rcept_dt",
                    "rcept_no",
                ]
            )
            .drop_duplicates(
                subset=["rcept_no"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    df.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    df.to_parquet(
        MANIFEST_PARQUET,
        index=False,
    )

    return df


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-periods",
        type=int,
        default=5,
        help=(
            "안전한 ZIP 구조 audit용 correction period 수. "
            "기본 5"
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "모든 correction-period의 모든 rcept_no XBRL을 다운로드"
        ),
    )

    parser.add_argument(
        "--redownload",
        action="store_true",
        help="이미 정상 저장된 ZIP도 다시 다운로드",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    args = parse_args()

    chains = load_chains()

    period_keys = (
        chains[
            [
                "stock_code",
                "period_key",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "stock_code",
                "period_key",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STRICT-PIT XBRL SOURCE COLLECTION"
    )
    print(
        "=" * 80
    )

    print(
        f"correction periods available : "
        f"{len(period_keys):,}"
    )

    print(
        f"filing submissions available : "
        f"{len(chains):,}"
    )

    if not args.full:
        selected_periods = (
            period_keys
            .head(
                args.sample_periods
            )
        )

        targets = chains.merge(
            selected_periods,
            on=[
                "stock_code",
                "period_key",
            ],
            how="inner",
        )

        print(
            f"SAMPLE MODE                  : "
            f"{len(selected_periods)} periods / "
            f"{len(targets)} submissions"
        )

    else:
        targets = chains.copy()

        print(
            f"FULL MODE                    : "
            f"{len(period_keys)} periods / "
            f"{len(targets)} submissions"
        )

    existing = load_existing_manifest()

    existing_ok: dict[
        str,
        dict[str, Any],
    ] = {}

    if not existing.empty:
        ok_rows = existing.loc[
            existing[
                "status"
            ].eq(
                "downloaded"
            )
        ]

        for row in ok_rows.to_dict(
            orient="records"
        ):
            existing_ok[
                str(
                    row["rcept_no"]
                )
            ] = row

    manifest_rows: list[
        dict[str, Any]
    ] = (
        existing.to_dict(
            orient="records"
        )
        if not existing.empty
        else []
    )

    session = requests.Session()

    for idx, row in targets.reset_index(
        drop=True
    ).iterrows():

        rcept_no = str(
            row["rcept_no"]
        )

        zip_path = target_zip_path(
            row
        )

        print(
            f"\n[{idx + 1}/{len(targets)}] "
            f"{row['stock_code']} | "
            f"{row['period_key']} | "
            f"seq={int(row['filing_sequence'])} | "
            f"{rcept_no}"
        )

        # 정상 manifest + 실제 파일 존재하면 skip
        previous = existing_ok.get(
            rcept_no
        )

        if (
            not args.redownload
            and previous is not None
            and zip_path.exists()
        ):
            print(
                f"SKIP existing: {zip_path}"
            )
            continue

        base_manifest = {
            "stock_code": row[
                "stock_code"
            ],
            "corp_code": row.get(
                "corp_code"
            ),
            "corp_name": row.get(
                "corp_name"
            ),
            "period_key": row[
                "period_key"
            ],
            "report_nm": row[
                "report_nm"
            ],
            "rcept_no": rcept_no,
            "rcept_dt": row[
                "rcept_dt"
            ],
            "is_correction": bool(
                row["is_correction"]
            ),
            "filing_sequence": int(
                row[
                    "filing_sequence"
                ]
            ),
            "chain_submission_count": int(
                row[
                    "chain_submission_count"
                ]
            ),
        }

        try:
            content, response = (
                request_xbrl(
                    session,
                    rcept_no,
                )
            )

            zip_info = inspect_zip(
                content
            )

            zip_path.write_bytes(
                content
            )

            item = {
                **base_manifest,
                "status": "downloaded",
                "http_status": (
                    response.status_code
                ),
                "file_bytes": len(
                    content
                ),
                "sha256": sha256_bytes(
                    content
                ),
                "zip_path": str(
                    zip_path
                ),
                **zip_info,
                "error": None,
            }

            manifest_rows.append(
                item
            )

            print(
                f"OK "
                f"bytes={len(content):,} "
                f"members={zip_info['zip_member_count']}"
            )

            if zip_info[
                "instance_candidates"
            ]:
                print(
                    "instance candidates:",
                    zip_info[
                        "instance_candidates"
                    ],
                )

        except Exception as exc:

            item = {
                **base_manifest,
                "status": "error",
                "http_status": None,
                "file_bytes": None,
                "sha256": None,
                "zip_path": str(
                    zip_path
                ),
                "zip_member_count": None,
                "zip_members": None,
                "zip_extensions": None,
                "instance_candidates": None,
                "error": repr(exc),
            }

            manifest_rows.append(
                item
            )

            print(
                "ERROR:",
                repr(exc),
            )

        save_manifest(
            manifest_rows
        )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    manifest = save_manifest(
        manifest_rows
    )

    # 이번 targets에 해당하는 최신 manifest만 summary
    target_receipts = set(
        targets[
            "rcept_no"
        ].astype(str)
    )

    subset = manifest.loc[
        manifest[
            "rcept_no"
        ].astype(str)
        .isin(
            target_receipts
        )
    ].copy()

    print(
        "\n"
        + "=" * 80
    )
    print(
        "XBRL DOWNLOAD SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        subset[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[ZIP structure examples]"
    )

    example_cols = [
        "stock_code",
        "period_key",
        "rcept_dt",
        "is_correction",
        "filing_sequence",
        "zip_member_count",
        "zip_extensions",
        "instance_candidates",
    ]

    if not subset.empty:
        print(
            subset[
                example_cols
            ]
            .sort_values(
                [
                    "stock_code",
                    "period_key",
                    "rcept_dt",
                ]
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    errors = subset.loc[
        subset[
            "status"
        ].ne(
            "downloaded"
        )
    ]

    print(
        "\n[Errors]"
    )

    if errors.empty:
        print("0")
    else:
        print(
            errors[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "error",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        f"\nmanifest CSV    : {MANIFEST_CSV}"
    )
    print(
        f"manifest parquet: {MANIFEST_PARQUET}"
    )

    print(
        "\n다음 단계:"
        "\n- sample ZIP 구조에서 실제 instance file/context/unit 구조 확인"
        "\n- 05A5-C에서 Assets/Equity/Revenue/Operating income/"
        "Net income를 접수번호별로 파싱"
        "\n- 각 snapshot의 valid_from = rcept_dt"
        "\n- 다음 정정 rcept_dt 직전까지 유효하도록 validity interval 생성"
    )


if __name__ == "__main__":
    main()
