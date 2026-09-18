from __future__ import annotations

import argparse
import hashlib
import io
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
# 05A5-D. OpenDART Full Strict-PIT Source Collector
#
# 목적
# ------------------------------------------------------------
# 전체 correction-period chain에 대해 각 rcept_no별 원본을 수집한다.
#
# Source priority
#   1) fnlttXbrl.xml   -> XBRL ZIP
#   2) document.xml    -> 공시원문 ZIP fallback
#   3) 둘 다 없음      -> unavailable
#
# IMPORTANT
# ------------------------------------------------------------
# - corrected period만 대상이다.
# - non-corrected period는 이후 단계에서 fnlttSinglAcntAll API를 사용.
# - resume 가능.
# - 이미 정상 저장된 rcept_no는 다시 받지 않는다.
# - unavailable(014)은 정상적인 strict-PIT 결과일 수 있다.
# - temporary/network/API 오류는 error로 별도 보존한다.
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/dart_correction_chains.csv
#
# OUTPUT
# ------------------------------------------------------------
# data/raw/dart/pit_sources_full/
#   <stock_code>/<period_key>/
#     <date>_<rcept_no>_xbrl.zip
#     또는
#     <date>_<rcept_no>_document.zip
#
# data/interim/dart/
#   dart_pit_source_full_manifest.csv
#   dart_pit_source_full_manifest.parquet
#
# 실행
# ------------------------------------------------------------
# 먼저 소량 테스트:
#   python scripts\05a5d_dart_full_strict_pit_source_collector.py --limit 20
#
# 전체:
#   python scripts\05a5d_dart_full_strict_pit_source_collector.py --full
#
# 특정 실패만 재시도:
#   python scripts\05a5d_dart_full_strict_pit_source_collector.py --retry-errors
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
    / "pit_sources_full"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

MANIFEST_CSV = (
    INTERIM_DIR
    / "dart_pit_source_full_manifest.csv"
)

MANIFEST_PARQUET = (
    INTERIM_DIR
    / "dart_pit_source_full_manifest.parquet"
)

RAW_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise RuntimeError(
        "DART_API_KEY가 없습니다. 프로젝트 루트 .env를 확인하세요."
    )


REQUEST_SLEEP_SECONDS = 0.25
MAX_RETRIES = 5
CHECKPOINT_EVERY = 20


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def safe_name(value: Any) -> str:
    text = str(value).strip()

    for old, new in [
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("*", "_"),
        ("?", "_"),
        ('"', "_"),
        ("<", "_"),
        (">", "_"),
        ("|", "_"),
        (" ", "_"),
    ]:
        text = text.replace(old, new)

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

        infos = zf.infolist()

        extensions: dict[str, int] = {}

        for info in infos:
            suffix = (
                Path(info.filename).suffix.lower()
                or "<none>"
            )

            extensions[suffix] = (
                extensions.get(suffix, 0)
                + 1
            )

        return {
            "zip_member_count": len(infos),
            "zip_extensions": " | ".join(
                f"{key}:{value}"
                for key, value
                in sorted(
                    extensions.items()
                )
            ),
        }


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
            f"필수 컬럼 누락: {missing}\n"
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

    # 같은 접수번호 중복 제거
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

    # chain sequence
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


def load_manifest() -> pd.DataFrame:
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

    if MANIFEST_CSV.exists():
        return pd.read_csv(
            MANIFEST_CSV,
            dtype={
                "stock_code": str,
                "rcept_no": str,
            },
        )

    return pd.DataFrame()


def save_manifest(
    df: pd.DataFrame,
) -> None:
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


def build_save_path(
    row: pd.Series,
    source_type: str,
) -> Path:

    period_dir = (
        RAW_ROOT
        / str(row["stock_code"])
        / safe_name(
            row["period_key"]
        )
    )

    period_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    date_text = pd.Timestamp(
        row["rcept_dt"]
    ).strftime("%Y%m%d")

    filename = (
        f"{date_text}_"
        f"seq{int(row['filing_sequence']):02d}_"
        f"{row['rcept_no']}_"
        f"{source_type}.zip"
    )

    return (
        period_dir
        / filename
    )


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

def request_binary(
    session: requests.Session,
    endpoint: str,
    rcept_no: str,
) -> dict[str, Any]:

    last_exception = None

    for attempt in range(
        MAX_RETRIES
    ):
        try:
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
                    20,
                )

                print(
                    f"  HTTP 429 -> {wait}s retry"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            content = response.content

            if content[:2] == b"PK":
                return {
                    "ok": True,
                    "content": content,
                    "status": "000",
                    "message": "정상",
                    "http_status": response.status_code,
                }

            status, message = (
                parse_error_xml(
                    content
                )
            )

            return {
                "ok": False,
                "content": None,
                "status": status,
                "message": message,
                "http_status": response.status_code,
            }

        except (
            requests.RequestException,
            TimeoutError,
        ) as exc:
            last_exception = exc

            wait = min(
                2 ** attempt,
                20,
            )

            print(
                f"  network error -> {wait}s retry: {repr(exc)}"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"{endpoint} {rcept_no}: "
        f"retry exceeded; "
        f"last={repr(last_exception)}"
    )


def collect_one(
    session: requests.Session,
    row: pd.Series,
) -> dict[str, Any]:

    base = {
        "stock_code": row["stock_code"],
        "corp_code": row.get("corp_code"),
        "corp_name": row.get("corp_name"),
        "period_key": row["period_key"],
        "report_nm": row["report_nm"],
        "rcept_no": row["rcept_no"],
        "rcept_dt": row["rcept_dt"],
        "is_correction": bool(
            row["is_correction"]
        ),
        "filing_sequence": int(
            row["filing_sequence"]
        ),
        "chain_submission_count": int(
            row["chain_submission_count"]
        ),
    }

    rcept_no = str(
        row["rcept_no"]
    )

    # --------------------------------------------------------
    # 1) XBRL
    # --------------------------------------------------------

    xbrl = request_binary(
        session=session,
        endpoint=XBRL_ENDPOINT,
        rcept_no=rcept_no,
    )

    time.sleep(
        REQUEST_SLEEP_SECONDS
    )

    if xbrl["ok"]:
        content = xbrl["content"]

        path = build_save_path(
            row,
            "xbrl",
        )

        path.write_bytes(
            content
        )

        zip_info = inspect_zip(
            content
        )

        return {
            **base,
            "status": "available",
            "source_type": "xbrl",
            "xbrl_status": "000",
            "xbrl_message": "정상",
            "document_status": None,
            "document_message": None,
            "http_status": xbrl[
                "http_status"
            ],
            "file_bytes": len(
                content
            ),
            "sha256": sha256_bytes(
                content
            ),
            "zip_path": str(
                path
            ),
            **zip_info,
            "error": None,
        }

    # quota / permission / auth 계열은 document fallback로
    # 덮어버리면 원인 파악이 어려우므로 014만 정상 fallback.
    if xbrl["status"] != "014":
        return {
            **base,
            "status": "error",
            "source_type": None,
            "xbrl_status": xbrl[
                "status"
            ],
            "xbrl_message": xbrl[
                "message"
            ],
            "document_status": None,
            "document_message": None,
            "http_status": xbrl[
                "http_status"
            ],
            "file_bytes": None,
            "sha256": None,
            "zip_path": None,
            "zip_member_count": None,
            "zip_extensions": None,
            "error": (
                "XBRL API returned non-014 error; "
                "document fallback intentionally not attempted"
            ),
        }

    # --------------------------------------------------------
    # 2) document fallback
    # --------------------------------------------------------

    document = request_binary(
        session=session,
        endpoint=DOCUMENT_ENDPOINT,
        rcept_no=rcept_no,
    )

    time.sleep(
        REQUEST_SLEEP_SECONDS
    )

    if document["ok"]:
        content = document[
            "content"
        ]

        path = build_save_path(
            row,
            "document",
        )

        path.write_bytes(
            content
        )

        zip_info = inspect_zip(
            content
        )

        return {
            **base,
            "status": "available",
            "source_type": "document_fallback",
            "xbrl_status": xbrl[
                "status"
            ],
            "xbrl_message": xbrl[
                "message"
            ],
            "document_status": "000",
            "document_message": "정상",
            "http_status": document[
                "http_status"
            ],
            "file_bytes": len(
                content
            ),
            "sha256": sha256_bytes(
                content
            ),
            "zip_path": str(
                path
            ),
            **zip_info,
            "error": None,
        }

    if document[
        "status"
    ] == "014":
        return {
            **base,
            "status": "unavailable",
            "source_type": None,
            "xbrl_status": xbrl[
                "status"
            ],
            "xbrl_message": xbrl[
                "message"
            ],
            "document_status": document[
                "status"
            ],
            "document_message": document[
                "message"
            ],
            "http_status": document[
                "http_status"
            ],
            "file_bytes": None,
            "sha256": None,
            "zip_path": None,
            "zip_member_count": None,
            "zip_extensions": None,
            "error": None,
        }

    return {
        **base,
        "status": "error",
        "source_type": None,
        "xbrl_status": xbrl[
            "status"
        ],
        "xbrl_message": xbrl[
            "message"
        ],
        "document_status": document[
            "status"
        ],
        "document_message": document[
            "message"
        ],
        "http_status": document[
            "http_status"
        ],
        "file_bytes": None,
        "sha256": None,
        "zip_path": None,
        "zip_member_count": None,
        "zip_extensions": None,
        "error": (
            "document fallback returned non-014 error"
        ),
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "테스트용 접수번호 수. "
            "--full이면 무시."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 correction-chain submissions 수집",
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "기존 manifest의 status=error인 접수번호만 재시도"
        ),
    )

    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help=(
            "기존 unavailable도 다시 확인. "
            "기본은 unavailable 재호출 안 함."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:

    args = parse_args()

    chains = load_chains()

    manifest = load_manifest()

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-D FULL STRICT-PIT SOURCE COLLECTION"
    )
    print(
        "=" * 80
    )

    print(
        f"corrected periods : "
        f"{chains[['stock_code', 'period_key']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"submissions       : "
        f"{len(chains):,}"
    )

    if not manifest.empty:
        print(
            "\n[Existing manifest]"
        )

        print(
            manifest[
                "status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # target selection
    # --------------------------------------------------------

    if args.retry_errors:

        if manifest.empty:
            print(
                "\n기존 manifest가 없어 retry-errors 대상이 없습니다."
            )
            return

        error_receipts = set(
            manifest.loc[
                manifest[
                    "status"
                ].eq(
                    "error"
                ),
                "rcept_no",
            ].astype(str)
        )

        targets = chains.loc[
            chains[
                "rcept_no"
            ].astype(str)
            .isin(
                error_receipts
            )
        ].copy()

        print(
            f"\nRETRY ERRORS MODE: "
            f"{len(targets):,}"
        )

    else:
        done_receipts: set[str] = set()

        if not manifest.empty:
            completed_status = [
                "available",
            ]

            if not args.retry_unavailable:
                completed_status.append(
                    "unavailable"
                )

            done_receipts = set(
                manifest.loc[
                    manifest[
                        "status"
                    ].isin(
                        completed_status
                    ),
                    "rcept_no",
                ].astype(str)
            )

        targets = chains.loc[
            ~chains[
                "rcept_no"
            ].astype(str)
            .isin(
                done_receipts
            )
        ].copy()

        if not args.full:
            targets = targets.head(
                args.limit
            )

            print(
                f"\nTEST MODE: "
                f"{len(targets):,} submissions"
            )
        else:
            print(
                f"\nFULL MODE remaining: "
                f"{len(targets):,} submissions"
            )

    if targets.empty:
        print(
            "\n수집 대상이 없습니다."
        )
        return

    # --------------------------------------------------------
    # manifest state dict
    # --------------------------------------------------------

    state: dict[
        str,
        dict[str, Any],
    ] = {}

    if not manifest.empty:
        for record in manifest.to_dict(
            orient="records"
        ):
            state[
                str(
                    record[
                        "rcept_no"
                    ]
                )
            ] = record

    session = requests.Session()

    for i, (_, row) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        print(
            f"\n[{i}/{len(targets)}] "
            f"{row['stock_code']} | "
            f"{row['period_key']} | "
            f"seq={int(row['filing_sequence'])} | "
            f"{row['rcept_no']}"
        )

        try:
            result = collect_one(
                session=session,
                row=row,
            )

        except Exception as exc:
            result = {
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
                "rcept_no": row[
                    "rcept_no"
                ],
                "rcept_dt": row[
                    "rcept_dt"
                ],
                "is_correction": bool(
                    row[
                        "is_correction"
                    ]
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
                "status": "error",
                "source_type": None,
                "xbrl_status": None,
                "xbrl_message": None,
                "document_status": None,
                "document_message": None,
                "http_status": None,
                "file_bytes": None,
                "sha256": None,
                "zip_path": None,
                "zip_member_count": None,
                "zip_extensions": None,
                "error": repr(
                    exc
                ),
            }

        state[
            str(
                row[
                    "rcept_no"
                ]
            )
        ] = result

        print(
            f"  -> status={result['status']} "
            f"source={result.get('source_type')} "
            f"xbrl={result.get('xbrl_status')} "
            f"document={result.get('document_status')}"
        )

        if (
            i % CHECKPOINT_EVERY
            == 0
        ):
            checkpoint = pd.DataFrame(
                state.values()
            )

            save_manifest(
                checkpoint
            )

            print(
                "  checkpoint saved"
            )

    final_manifest = pd.DataFrame(
        state.values()
    )

    save_manifest(
        final_manifest
    )

    # --------------------------------------------------------
    # final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "FULL PIT SOURCE COLLECTION SUMMARY"
    )
    print(
        "=" * 80
    )

    print(
        final_manifest[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Source type]"
    )

    print(
        final_manifest[
            "source_type"
        ]
        .fillna(
            "none"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\n[Coverage]"
    )

    total_chain_receipts = chains[
        "rcept_no"
    ].nunique()

    manifest_receipts = (
        final_manifest[
            "rcept_no"
        ].astype(str)
        .nunique()
    )

    print(
        f"chain receipts    : "
        f"{total_chain_receipts:,}"
    )

    print(
        f"manifest receipts : "
        f"{manifest_receipts:,}"
    )

    print(
        f"coverage          : "
        f"{manifest_receipts / total_chain_receipts:.2%}"
    )

    errors = final_manifest.loc[
        final_manifest[
            "status"
        ].eq(
            "error"
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
                    "xbrl_status",
                    "xbrl_message",
                    "document_status",
                    "document_message",
                    "error",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    unavailable = final_manifest.loc[
        final_manifest[
            "status"
        ].eq(
            "unavailable"
        )
    ]

    print(
        "\n[Unavailable official originals]"
    )

    if unavailable.empty:
        print("0")
    else:
        print(
            unavailable[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "rcept_dt",
                ]
            ]
            .head(50)
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
        "\n1) available XBRL -> production XBRL selector"
        "\n2) document_fallback -> C5 strict consolidated selector"
        "\n3) unavailable/error -> missing"
        "\n4) corrected-period PIT snapshots 생성"
        "\n5) non-corrected periods의 일반 DART 재무 API 결과와 결합"
    )


if __name__ == "__main__":
    main()
