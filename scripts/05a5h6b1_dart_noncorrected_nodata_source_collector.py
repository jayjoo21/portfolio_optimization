from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H6B1. Non-Corrected No-Data Same-Receipt Source Collector
#
# 목적
# ------------------------------------------------------------
# H6A에서
#   source_level_fallback_full_api_no_data = 890
# 로 분류된 receipt만 대상으로,
#
#   1) same-receipt XBRL 원본
#   2) XBRL이 없으면 same-receipt document 원본
#
# 순서로 수집한다.
#
# IMPORTANT
# ------------------------------------------------------------
# - 미래/최신 공시로 backfill하지 않는다.
# - expected rcept_no 그 자체만 요청한다.
# - 이 단계에서는 "값 선택/파싱"을 하지 않는다.
# - source availability manifest를 만드는 단계다.
# - resume 가능.
#
# OpenDART endpoints
# ------------------------------------------------------------
# XBRL:
#   https://opendart.fss.or.kr/api/fnlttXbrl.xml
#   params: crtfc_key, rcept_no, reprt_code
#
# Document:
#   https://opendart.fss.or.kr/api/document.xml
#   params: crtfc_key, rcept_no
#
# 실행
# ------------------------------------------------------------
# test 20:
#   python scripts\05a5h6b1_dart_noncorrected_nodata_source_collector.py
#
# full:
#   python scripts\05a5h6b1_dart_noncorrected_nodata_source_collector.py --full
#
# 특정 개수:
#   python scripts\05a5h6b1_dart_noncorrected_nodata_source_collector.py --limit 100
#
# 처음부터 재수집:
#   python scripts\05a5h6b1_dart_noncorrected_nodata_source_collector.py --full --reset
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_full_api_no_data_scope.csv"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "noncorrected_pit_sources"
)

MANIFEST_PARQUET = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.parquet"
)

MANIFEST_CSV = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.csv"
)

XBRL_URL = (
    "https://opendart.fss.or.kr/api/fnlttXbrl.xml"
)

DOCUMENT_URL = (
    "https://opendart.fss.or.kr/api/document.xml"
)

TERMINAL_STATUSES = {
    "available_xbrl",
    "available_document",
    "unavailable",
}

RETRYABLE_HTTP = {
    429,
    500,
    502,
    503,
    504,
}


def receipt_string(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def safe_name(
    value: Any,
) -> str:

    text = str(
        value
    ).strip()

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
        text = text.replace(
            old,
            new,
        )

    return text


def parse_error_xml(
    content: bytes,
) -> tuple[
    str | None,
    str | None,
]:

    try:
        root = ET.fromstring(
            content
        )
    except Exception:
        text = content[
            :1000
        ].decode(
            "utf-8",
            errors="replace",
        )

        return (
            None,
            text,
        )

    status = None
    message = None

    for elem in root.iter():

        tag = (
            elem.tag.rsplit(
                "}",
                1,
            )[-1]
            if "}" in elem.tag
            else elem.tag
        )

        if tag == "status":
            status = (
                elem.text.strip()
                if elem.text
                else None
            )

        elif tag == "message":
            message = (
                elem.text.strip()
                if elem.text
                else None
            )

    return (
        status,
        message,
    )


def request_binary(
    session: requests.Session,
    url: str,
    params: dict[
        str,
        Any,
    ],
    max_retries: int = 4,
) -> dict[
    str,
    Any,
]:

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:
            response = session.get(
                url,
                params=params,
                timeout=120,
            )

            if (
                response.status_code
                in RETRYABLE_HTTP
            ):
                last_error = (
                    f"HTTP {response.status_code}"
                )

                if attempt < max_retries:
                    time.sleep(
                        min(
                            2 ** attempt,
                            10,
                        )
                    )
                    continue

            response.raise_for_status()

            content = response.content

            if content[
                :2
            ] == b"PK":

                return {
                    "kind":
                    "zip",

                    "content":
                    content,

                    "status":
                    "000",

                    "message":
                    "정상",
                }

            status, message = (
                parse_error_xml(
                    content
                )
            )

            return {
                "kind":
                "api_message",

                "content":
                None,

                "status":
                status,

                "message":
                message,
            }

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_retries:
                time.sleep(
                    min(
                        2 ** attempt,
                        10,
                    )
                )
                continue

    return {
        "kind":
        "error",

        "content":
        None,

        "status":
        None,

        "message":
        last_error,
    }


def load_manifest() -> pd.DataFrame:

    if MANIFEST_PARQUET.exists():

        manifest = pd.read_parquet(
            MANIFEST_PARQUET
        )

        if not manifest.empty:
            manifest[
                "rcept_no"
            ] = receipt_string(
                manifest[
                    "rcept_no"
                ]
            )

        return manifest

    return pd.DataFrame()


def save_manifest(
    records: list[
        dict[
            str,
            Any,
        ]
    ],
):

    if not records:
        return

    new = pd.DataFrame(
        records
    )

    old = load_manifest()

    if old.empty:
        merged = new.copy()

    else:
        merged = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    merged[
        "rcept_no"
    ] = receipt_string(
        merged[
            "rcept_no"
        ]
    )

    merged = (
        merged.sort_values(
            [
                "rcept_no",
                "attempted_at",
            ]
        )
        .drop_duplicates(
            subset=[
                "rcept_no",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    merged.to_parquet(
        MANIFEST_PARQUET,
        index=False,
    )

    merged.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 890건 처리",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 receipt 수",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 manifest를 무시하고 다시 수집",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.20,
        help="API 요청 사이 대기 초",
    )

    args = parser.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

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

    targets = pd.read_csv(
        INPUT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
            "reprt_code":
            str,
            "corp_code":
            str,
        },
        low_memory=False,
    )

    targets[
        "rcept_no"
    ] = receipt_string(
        targets[
            "rcept_no"
        ]
    )

    if targets[
        "rcept_no"
    ].duplicated().any():

        dupes = (
            targets.loc[
                targets[
                    "rcept_no"
                ].duplicated(
                    keep=False
                ),
                [
                    "stock_code",
                    "corp_name",
                    "canonical_period_key",
                    "rcept_no",
                ],
            ]
        )

        raise RuntimeError(
            "INPUT에 duplicate rcept_no가 있습니다:\n"
            + dupes.head(
                30
            ).to_string(
                index=False
            )
        )

    if len(
        targets
    ) != 890:
        print(
            "WARNING: H6A에서 예상한 no-data target은 890건입니다. "
            f"현재={len(targets):,}"
        )

    existing = load_manifest()

    done_receipts = set()

    if (
        not args.reset
        and not existing.empty
    ):

        done_receipts = set(
            existing.loc[
                existing[
                    "collection_status"
                ].isin(
                    TERMINAL_STATUSES
                ),
                "rcept_no",
            ].astype(str)
        )

    todo = targets.loc[
        ~targets[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            done_receipts
        )
    ].copy()

    if args.full:
        limit = None

    elif (
        args.limit
        is not None
    ):
        limit = max(
            int(
                args.limit
            ),
            0,
        )

    else:
        limit = 20

    if (
        limit is not None
        and limit > 0
    ):
        todo = todo.head(
            limit
        ).copy()

    elif limit == 0:
        todo = todo.head(
            0
        ).copy()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H6B1 NON-CORRECTED NO-DATA SAME-RECEIPT SOURCE COLLECTOR"
    )

    print(
        "=" * 100
    )

    print(
        f"\nAll targets : "
        f"{len(targets):,}"
    )

    print(
        f"Already done: "
        f"{len(done_receipts):,}"
    )

    print(
        f"This run    : "
        f"{len(todo):,}"
    )

    print(
        "\nSource priority:"
        "\n1) same-receipt XBRL"
        "\n2) same-receipt document"
        "\n3) otherwise unavailable"
    )

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = (
        requests.Session()
    )

    buffer = []

    for idx, (_, row) in enumerate(
        todo.iterrows(),
        start=1,
    ):

        receipt = str(
            row[
                "rcept_no"
            ]
        )

        stock = str(
            row.get(
                "stock_code",
                "",
            )
        ).zfill(
            6
        )

        period = str(
            row.get(
                "canonical_period_key",
                "",
            )
        )

        reprt_code = str(
            row.get(
                "reprt_code",
                "",
            )
        )

        period_dir = (
            RAW_ROOT
            / stock
            / safe_name(
                period
            )
        )

        period_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        xbrl_path = (
            period_dir
            / f"{receipt}_xbrl.zip"
        )

        document_path = (
            period_dir
            / f"{receipt}_document.zip"
        )

        print(
            f"\n[{idx}/{len(todo)}] "
            f"{stock} | "
            f"{period} | "
            f"{receipt}"
        )

        # ----------------------------------------------------
        # 1) XBRL
        # ----------------------------------------------------

        xbrl_result = (
            request_binary(
                session,
                XBRL_URL,
                {
                    "crtfc_key":
                    api_key,

                    "rcept_no":
                    receipt,

                    "reprt_code":
                    reprt_code,
                },
            )
        )

        time.sleep(
            args.sleep
        )

        selected_source = None
        collection_status = None
        selected_path = None

        if (
            xbrl_result[
                "kind"
            ]
            == "zip"
        ):

            xbrl_path.write_bytes(
                xbrl_result[
                    "content"
                ]
            )

            selected_source = (
                "xbrl"
            )

            collection_status = (
                "available_xbrl"
            )

            selected_path = str(
                xbrl_path
            )

            print(
                "  XBRL: available"
            )

            document_result = {
                "status":
                None,
                "message":
                "not_requested_xbrl_available",
                "kind":
                "not_requested",
            }

        else:

            print(
                "  XBRL: "
                f"{xbrl_result.get('status')} | "
                f"{xbrl_result.get('message')}"
            )

            # ------------------------------------------------
            # 2) document fallback
            # ------------------------------------------------

            document_result = (
                request_binary(
                    session,
                    DOCUMENT_URL,
                    {
                        "crtfc_key":
                        api_key,

                        "rcept_no":
                        receipt,
                    },
                )
            )

            time.sleep(
                args.sleep
            )

            if (
                document_result[
                    "kind"
                ]
                == "zip"
            ):

                document_path.write_bytes(
                    document_result[
                        "content"
                    ]
                )

                selected_source = (
                    "document"
                )

                collection_status = (
                    "available_document"
                )

                selected_path = str(
                    document_path
                )

                print(
                    "  Document: available"
                )

            elif (
                xbrl_result[
                    "kind"
                ]
                == "error"
                or document_result[
                    "kind"
                ]
                == "error"
            ):

                collection_status = (
                    "request_error"
                )

                print(
                    "  Document: ERROR | "
                    f"{document_result.get('message')}"
                )

            else:

                collection_status = (
                    "unavailable"
                )

                print(
                    "  Document: "
                    f"{document_result.get('status')} | "
                    f"{document_result.get('message')}"
                )

        record = {
            "stock_code":
            stock,

            "corp_code":
            row.get(
                "corp_code"
            ),

            "corp_name":
            row.get(
                "corp_name"
            ),

            "canonical_period_key":
            period,

            "rcept_no":
            receipt,

            "reprt_code":
            reprt_code,

            "rcept_dt":
            row.get(
                "rcept_dt"
            ),

            "selected_source":
            selected_source,

            "collection_status":
            collection_status,

            "selected_path":
            selected_path,

            "xbrl_status":
            xbrl_result.get(
                "status"
            ),

            "xbrl_message":
            xbrl_result.get(
                "message"
            ),

            "document_status":
            document_result.get(
                "status"
            ),

            "document_message":
            document_result.get(
                "message"
            ),

            "attempted_at":
            pd.Timestamp.now(
                tz="Asia/Seoul"
            ),
        }

        buffer.append(
            record
        )

        # Persist frequently so an interrupted run resumes safely.
        if len(
            buffer
        ) >= 25:

            save_manifest(
                buffer
            )

            buffer = []

    if buffer:
        save_manifest(
            buffer
        )

    manifest = (
        load_manifest()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "H6B1 SOURCE COLLECTION SUMMARY"
    )

    print(
        "=" * 100
    )

    if manifest.empty:
        print(
            "\nManifest empty."
        )
        return

    manifest_scope = manifest.loc[
        manifest[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            set(
                targets[
                    "rcept_no"
                ].astype(str)
            )
        )
    ].copy()

    print(
        "\n[Collection status]"
    )

    print(
        manifest_scope[
            "collection_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Selected source]"
    )

    print(
        manifest_scope[
            "selected_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[XBRL status]"
    )

    print(
        manifest_scope[
            "xbrl_status"
        ]
        .value_counts(
            dropna=False
        )
        .head(
            20
        )
        .to_string()
    )

    fallback_doc = manifest_scope.loc[
        manifest_scope[
            "collection_status"
        ].eq(
            "available_document"
        )
    ]

    unavailable = manifest_scope.loc[
        manifest_scope[
            "collection_status"
        ].eq(
            "unavailable"
        )
    ]

    errors = manifest_scope.loc[
        manifest_scope[
            "collection_status"
        ].eq(
            "request_error"
        )
    ]

    print(
        f"\nXBRL -> document fallback success: "
        f"{len(fallback_doc):,}"
    )

    print(
        f"Unavailable both sources         : "
        f"{len(unavailable):,}"
    )

    print(
        f"Request errors                   : "
        f"{len(errors):,}"
    )

    print(
        f"\nManifest: "
        f"{MANIFEST_PARQUET}"
    )

    print(
        f"Raw root: "
        f"{RAW_ROOT}"
    )

    print(
        "\n다음 단계:"
        "\n- test run에서 request_error가 0인지 확인"
        "\n- 정상이라면 --full로 890건 수집"
        "\n- full 완료 후 H6B2에서 XBRL/document parser + core-6 recovery QA"
    )


if __name__ == "__main__":
    main()
