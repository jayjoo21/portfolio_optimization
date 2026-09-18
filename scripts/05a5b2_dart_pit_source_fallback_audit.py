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
# 05A5-B2. OpenDART Strict-PIT Source Audit with fallback
#
# source priority
# 1) fnlttXbrl.xml  -> 재무제표 XBRL 원본 ZIP
# 2) document.xml   -> 공시서류 전체 원본 ZIP fallback
#
# 목적
# - XBRL이 없는 과거/특정 회사 보고서에서도
#   접수 당시 원본 공시를 보존할 수 있는지 확인
# - sample이 한 종목에 몰리지 않도록 서로 다른 회사에서
#   correction-period를 1개씩 선택
#
# 아직 숫자 parsing은 하지 않는다.
# ============================================================


BASE_URL = "https://opendart.fss.or.kr/api"

XBRL_ENDPOINT = "fnlttXbrl.xml"
DOCUMENT_ENDPOINT = "document.xml"

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
    / "pit_source_audit"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

MANIFEST_PATH = (
    INTERIM_DIR
    / "dart_pit_source_fallback_audit.csv"
)

RAW_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv(PROJECT_ROOT / ".env")

DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise RuntimeError(
        "DART_API_KEY가 없습니다. .env를 확인하세요."
    )


REQUEST_SLEEP_SECONDS = 0.20
MAX_RETRIES = 5


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def safe_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_error_xml(
    content: bytes,
) -> tuple[str | None, str | None]:
    try:
        root = ET.fromstring(content)
        return (
            root.findtext("status"),
            root.findtext("message"),
        )
    except Exception:
        return (None, None)


def inspect_zip(
    content: bytes,
) -> dict[str, Any]:

    with zipfile.ZipFile(
        io.BytesIO(content)
    ) as zf:

        infos = zf.infolist()

        names = [
            info.filename
            for info in infos
        ]

        extensions: dict[str, int] = {}

        for name in names:
            suffix = (
                Path(name).suffix.lower()
                or "<none>"
            )
            extensions[suffix] = (
                extensions.get(suffix, 0)
                + 1
            )

        xml_like = [
            name
            for name in names
            if name.lower().endswith(
                (
                    ".xml",
                    ".xbrl",
                    ".html",
                    ".htm",
                )
            )
        ]

        largest = sorted(
            infos,
            key=lambda x: x.file_size,
            reverse=True,
        )[:10]

        largest_members = " | ".join(
            f"{info.filename}:{info.file_size}"
            for info in largest
        )

        return {
            "zip_member_count": len(names),
            "zip_extensions": " | ".join(
                f"{k}:{v}"
                for k, v in sorted(
                    extensions.items()
                )
            ),
            "xml_like_members": " | ".join(
                xml_like[:30]
            ),
            "largest_members": largest_members,
        }


def load_chains() -> pd.DataFrame:

    if not CHAINS_PATH.exists():
        raise FileNotFoundError(
            f"correction chain 파일이 없습니다: {CHAINS_PATH}"
        )

    df = pd.read_csv(
        CHAINS_PATH,
        dtype=str,
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
            f"필수 컬럼 누락: {missing}"
        )

    df["stock_code"] = (
        df["stock_code"]
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(6)
    )

    df["rcept_no"] = (
        df["rcept_no"]
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

    df["is_correction"] = (
        df["is_correction"]
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    return (
        df
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
# diversified sampling
# ------------------------------------------------------------

def select_diverse_periods(
    chains: pd.DataFrame,
    n_companies: int,
) -> pd.DataFrame:

    # correction period 단위
    periods = (
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
    )

    # 회사당 1개 period만 선택해서 특정 회사 쏠림 방지
    selected = (
        periods
        .groupby(
            "stock_code",
            as_index=False,
        )
        .first()
        .head(
            n_companies
        )
    )

    return chains.merge(
        selected,
        on=[
            "stock_code",
            "period_key",
        ],
        how="inner",
    )


# ------------------------------------------------------------
# API request
# ------------------------------------------------------------

def request_binary_api(
    session: requests.Session,
    endpoint: str,
    rcept_no: str,
) -> tuple[
    bytes | None,
    str | None,
    str | None,
]:

    for attempt in range(
        MAX_RETRIES
    ):
        response = session.get(
            f"{BASE_URL}/{endpoint}",
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
            time.sleep(wait)
            continue

        response.raise_for_status()

        content = response.content

        if content[:2] == b"PK":
            return (
                content,
                "000",
                "정상",
            )

        status, message = (
            parse_error_xml(
                content
            )
        )

        return (
            None,
            status,
            message,
        )

    raise RuntimeError(
        f"{endpoint} {rcept_no}: 최대 재시도 초과"
    )


# ------------------------------------------------------------
# save
# ------------------------------------------------------------

def build_save_path(
    row: pd.Series,
    source_type: str,
) -> Path:

    folder = (
        RAW_ROOT
        / row["stock_code"]
        / safe_name(
            row["period_key"]
        )
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt = pd.Timestamp(
        row["rcept_dt"]
    ).strftime("%Y%m%d")

    correction_tag = (
        "correction"
        if bool(
            row["is_correction"]
        )
        else "original"
    )

    return (
        folder
        / (
            f"{dt}_"
            f"{correction_tag}_"
            f"{row['rcept_no']}_"
            f"{source_type}.zip"
        )
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-companies",
        type=int,
        default=8,
        help="서로 다른 회사 수. 기본 8",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    chains = load_chains()

    targets = select_diverse_periods(
        chains,
        args.sample_companies,
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "DART PIT SOURCE FALLBACK AUDIT"
    )
    print(
        "=" * 80
    )

    print(
        f"companies sampled : "
        f"{targets['stock_code'].nunique()}"
    )
    print(
        f"periods sampled   : "
        f"{targets[['stock_code', 'period_key']].drop_duplicates().shape[0]}"
    )
    print(
        f"submissions       : "
        f"{len(targets)}"
    )

    print(
        "\n[selected periods]"
    )

    print(
        targets[
            [
                "stock_code",
                "corp_name",
                "period_key",
                "rcept_dt",
                "is_correction",
                "rcept_no",
            ]
        ].to_string(
            index=False
        )
    )

    session = requests.Session()

    rows = []

    for i, row in targets.reset_index(
        drop=True
    ).iterrows():

        rcept_no = str(
            row["rcept_no"]
        )

        print(
            f"\n[{i + 1}/{len(targets)}] "
            f"{row['stock_code']} | "
            f"{row['period_key']} | "
            f"{rcept_no}"
        )

        base = {
            "stock_code": row[
                "stock_code"
            ],
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
        }

        # -----------------------------------------
        # 1) XBRL first
        # -----------------------------------------

        xbrl_content, xbrl_status, xbrl_message = (
            request_binary_api(
                session,
                XBRL_ENDPOINT,
                rcept_no,
            )
        )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

        if xbrl_content is not None:
            path = build_save_path(
                row,
                "xbrl",
            )

            path.write_bytes(
                xbrl_content
            )

            info = inspect_zip(
                xbrl_content
            )

            rows.append(
                {
                    **base,
                    "source_type": "xbrl",
                    "xbrl_status": "000",
                    "xbrl_message": "정상",
                    "document_status": None,
                    "document_message": None,
                    "status": "available",
                    "file_bytes": len(
                        xbrl_content
                    ),
                    "sha256": sha256_bytes(
                        xbrl_content
                    ),
                    "zip_path": str(
                        path
                    ),
                    **info,
                }
            )

            print(
                f"XBRL OK "
                f"bytes={len(xbrl_content):,} "
                f"members={info['zip_member_count']}"
            )

            continue

        print(
            f"XBRL unavailable: "
            f"status={xbrl_status} "
            f"message={xbrl_message}"
        )

        # -----------------------------------------
        # 2) full document fallback
        # -----------------------------------------

        doc_content, doc_status, doc_message = (
            request_binary_api(
                session,
                DOCUMENT_ENDPOINT,
                rcept_no,
            )
        )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

        if doc_content is not None:
            path = build_save_path(
                row,
                "document",
            )

            path.write_bytes(
                doc_content
            )

            info = inspect_zip(
                doc_content
            )

            rows.append(
                {
                    **base,
                    "source_type": "document_fallback",
                    "xbrl_status": xbrl_status,
                    "xbrl_message": xbrl_message,
                    "document_status": "000",
                    "document_message": "정상",
                    "status": "available",
                    "file_bytes": len(
                        doc_content
                    ),
                    "sha256": sha256_bytes(
                        doc_content
                    ),
                    "zip_path": str(
                        path
                    ),
                    **info,
                }
            )

            print(
                f"DOCUMENT FALLBACK OK "
                f"bytes={len(doc_content):,} "
                f"members={info['zip_member_count']}"
            )

        else:
            rows.append(
                {
                    **base,
                    "source_type": None,
                    "xbrl_status": xbrl_status,
                    "xbrl_message": xbrl_message,
                    "document_status": doc_status,
                    "document_message": doc_message,
                    "status": "unavailable",
                    "file_bytes": None,
                    "sha256": None,
                    "zip_path": None,
                    "zip_member_count": None,
                    "zip_extensions": None,
                    "xml_like_members": None,
                    "largest_members": None,
                }
            )

            print(
                "BOTH UNAVAILABLE: "
                f"document status={doc_status} "
                f"message={doc_message}"
            )

    manifest = pd.DataFrame(
        rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "PIT SOURCE FALLBACK SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        manifest[
            "source_type"
        ]
        .fillna(
            "unavailable"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\n[Per submission]"
    )

    show_cols = [
        "stock_code",
        "period_key",
        "rcept_dt",
        "is_correction",
        "source_type",
        "xbrl_status",
        "document_status",
        "zip_member_count",
        "zip_extensions",
    ]

    print(
        manifest[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n[Document fallback structure]"
    )

    fallback = manifest.loc[
        manifest[
            "source_type"
        ].eq(
            "document_fallback"
        )
    ]

    if fallback.empty:
        print("0")
    else:
        print(
            fallback[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "xml_like_members",
                    "largest_members",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print(
        "\n[Unavailable]"
    )

    unavailable = manifest.loc[
        manifest[
            "status"
        ].eq(
            "unavailable"
        )
    ]

    if unavailable.empty:
        print("0")
    else:
        print(
            unavailable[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "xbrl_status",
                    "xbrl_message",
                    "document_status",
                    "document_message",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        f"\nmanifest: {MANIFEST_PATH}"
    )

    print(
        "\n판단 기준:"
        "\n- xbrl: 가장 좋은 strict-PIT source"
        "\n- document_fallback: XBRL ZIP이 없어도 접수 당시 공시 원문을 복원 가능"
        "\n- unavailable: 두 공식 원본 API 모두 실패한 경우 별도 처리 필요"
        "\n"
        "\n다음 단계는 이 결과의 ZIP 구조를 보고"
        " XBRL parser와 document fallback parser를 각각 설계"
    )


if __name__ == "__main__":
    main()
