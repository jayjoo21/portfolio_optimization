from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H1C. OpenDART API Error Adaptive Recovery
#
# 목적
# ------------------------------------------------------------
# H1 fnlttMultiAcnt에서 api_error로 남은 receipt만 복구한다.
#
# 문제 패턴
# ------------------------------------------------------------
# 특정 100-company batch에서 OpenDART 응답이 JSON이 아니어서
# JSONDecodeError가 반복되는 경우:
#
#   100 companies
#       ↓ fail
#    50 + 50
#       ↓ fail이면
#    25 + 25 ...
#       ↓
#     1 company
#
# 까지 자동 분할한다.
#
# PIT 규칙
# ------------------------------------------------------------
# - 기존 expected_rcept_no와 API returned rcept_no가 일치할 때만
#   pit_verified=True.
# - receipt mismatch는 사용하지 않는다.
# - api_error가 끝까지 남으면 해당 receipt는 이후 full API fallback
#   대상으로 별도 처리한다.
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h1c_dart_api_error_adaptive_recovery.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

H1_PATH = (
    SCRIPTS_DIR
    / "05a5h1_dart_noncorrected_batch_collector.py"
)

API_URL = (
    "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
)

FATAL_STATUSES = {
    "010",  # unregistered key
    "011",  # unusable key
    "012",  # IP restriction
    "020",  # request limit
    "901",  # account/key issue
}

MAX_RETRIES_PER_CHUNK = 2
REQUEST_SLEEP_SECONDS = 0.20


def load_h1():
    spec = importlib.util.spec_from_file_location(
        "dart_h1_recovery_base",
        H1_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"H1 module load 실패: {H1_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "dart_h1_recovery_base"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def response_preview(
    response: requests.Response,
    limit: int = 300,
) -> str:
    try:
        text = response.text
    except Exception:
        return "<response text unavailable>"

    text = (
        text.replace(
            "\r",
            " ",
        )
        .replace(
            "\n",
            " ",
        )
        .strip()
    )

    return text[:limit]


def request_chunk_once(
    session: requests.Session,
    api_key: str,
    corp_codes: list[str],
    bsns_year: int,
    reprt_code: str,
) -> dict[str, Any]:

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

    http_status = response.status_code
    content_type = response.headers.get(
        "Content-Type",
        "",
    )

    response.raise_for_status()

    try:
        payload = response.json()

    except Exception as exc:
        return {
            "kind": "non_json",
            "status": None,
            "message": None,
            "rows": [],
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "http_status": http_status,
            "content_type": content_type,
            "preview": response_preview(
                response
            ),
        }

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
            "kind": "success",
            "status": status,
            "message": message,
            "rows": payload.get(
                "list",
                [],
            ),
            "error": None,
            "http_status": http_status,
            "content_type": content_type,
            "preview": "",
        }

    if status == "013":
        return {
            "kind": "no_data",
            "status": status,
            "message": message,
            "rows": [],
            "error": None,
            "http_status": http_status,
            "content_type": content_type,
            "preview": "",
        }

    if status in FATAL_STATUSES:
        return {
            "kind": "fatal",
            "status": status,
            "message": message,
            "rows": [],
            "error": (
                f"OpenDART status={status}: {message}"
            ),
            "http_status": http_status,
            "content_type": content_type,
            "preview": "",
        }

    # 021 (too many companies), 100, 900, etc.
    # A bad member may poison the whole chunk, so split is useful.
    return {
        "kind": "split_candidate",
        "status": status,
        "message": message,
        "rows": [],
        "error": (
            f"OpenDART status={status}: {message}"
        ),
        "http_status": http_status,
        "content_type": content_type,
        "preview": "",
    }


def request_chunk_with_short_retry(
    session: requests.Session,
    api_key: str,
    corp_codes: list[str],
    bsns_year: int,
    reprt_code: str,
) -> dict[str, Any]:

    last = None

    for attempt in range(
        MAX_RETRIES_PER_CHUNK
    ):
        try:
            result = request_chunk_once(
                session=session,
                api_key=api_key,
                corp_codes=corp_codes,
                bsns_year=bsns_year,
                reprt_code=reprt_code,
            )

            last = result

            if result[
                "kind"
            ] in {
                "success",
                "no_data",
                "fatal",
                "split_candidate",
            }:
                return result

            # non_json: one brief retry before splitting
            if (
                result[
                    "kind"
                ] == "non_json"
                and attempt
                < MAX_RETRIES_PER_CHUNK - 1
            ):
                time.sleep(
                    1
                )
                continue

            return result

        except requests.RequestException as exc:
            last = {
                "kind": "transport_error",
                "status": None,
                "message": None,
                "rows": [],
                "error": repr(
                    exc
                ),
                "http_status": None,
                "content_type": None,
                "preview": "",
            }

            if attempt < (
                MAX_RETRIES_PER_CHUNK - 1
            ):
                time.sleep(
                    1
                )
                continue

    return last


def split_target_frame(
    targets: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    corp_codes = (
        targets[
            "corp_code"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    midpoint = max(
        1,
        len(
            corp_codes
        )
        // 2,
    )

    left_codes = set(
        corp_codes[
            :midpoint
        ]
    )

    right_codes = set(
        corp_codes[
            midpoint:
        ]
    )

    left = targets.loc[
        targets[
            "corp_code"
        ]
        .astype(str)
        .isin(
            left_codes
        )
    ].copy()

    right = targets.loc[
        targets[
            "corp_code"
        ]
        .astype(str)
        .isin(
            right_codes
        )
    ].copy()

    return (
        left,
        right,
    )


def merge_state(
    h1,
    manifest_state: pd.DataFrame,
    row_state: pd.DataFrame,
    period_manifest: pd.DataFrame,
    api_rows: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    touched = set(
        period_manifest[
            "expected_rcept_no"
        ]
        .astype(str)
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

    if not row_state.empty:
        row_state = (
            row_state.loc[
                ~row_state[
                    "rcept_no"
                ]
                .astype(str)
                .isin(
                    touched
                )
            ]
        )

    if not api_rows.empty:
        api_rows = api_rows.loc[
            api_rows[
                "rcept_no"
            ]
            .astype(str)
            .isin(
                touched
            )
        ].copy()

        row_state = pd.concat(
            [
                row_state,
                api_rows,
            ],
            ignore_index=True,
        )

    return (
        manifest_state,
        row_state,
    )


def make_terminal_error_manifest(
    targets: pd.DataFrame,
    batch_id: str,
    result: dict[str, Any],
) -> pd.DataFrame:

    records = []

    for _, target in (
        targets.iterrows()
    ):

        records.append(
            {
                "batch_id": batch_id,
                "stock_code": target[
                    "stock_code"
                ],
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
                "expected_rcept_no": str(
                    target[
                        "expected_rcept_no"
                    ]
                ),
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
                "collection_status": "api_error",
                "pit_verified": False,
                "returned_rcept_nos": "",
                "api_row_count": 0,
                "api_status": result.get(
                    "status"
                ),
                "api_message": result.get(
                    "message"
                ),
                "api_error": (
                    f"{result.get('error')} | "
                    f"http={result.get('http_status')} | "
                    f"content_type={result.get('content_type')} | "
                    f"preview={result.get('preview')}"
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def recover_recursive(
    *,
    h1,
    session: requests.Session,
    api_key: str,
    targets: pd.DataFrame,
    bsns_year: int,
    reprt_code: str,
    logical_batch_id: str,
    depth: int = 0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    corp_codes = (
        targets[
            "corp_code"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    indent = (
        "  "
        * depth
    )

    print(
        f"{indent}try "
        f"companies={len(corp_codes)} "
        f"periods={len(targets)}"
    )

    result = request_chunk_with_short_retry(
        session=session,
        api_key=api_key,
        corp_codes=corp_codes,
        bsns_year=bsns_year,
        reprt_code=reprt_code,
    )

    if result[
        "kind"
    ] == "fatal":
        raise RuntimeError(
            result[
                "error"
            ]
        )

    if result[
        "kind"
    ] in {
        "success",
        "no_data",
    }:

        batch = {
            "batch_id": (
                f"{logical_batch_id}"
                f"_d{depth}"
                f"_n{len(corp_codes)}"
            ),
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
            "corp_codes": corp_codes,
            "targets": targets,
        }

        api_rows = h1.normalize_raw_rows(
            result[
                "rows"
            ],
            batch,
        )

        period_manifest = (
            h1.evaluate_batch_targets(
                batch_targets=targets,
                api_rows=api_rows,
                api_status=result[
                    "status"
                ],
                api_message=result[
                    "message"
                ],
                api_error=None,
                batch_id=batch[
                    "batch_id"
                ],
            )
        )

        print(
            f"{indent}  -> "
            + ", ".join(
                f"{k}={v}"
                for k, v
                in period_manifest[
                    "collection_status"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
                .items()
            )
        )

        return (
            period_manifest,
            api_rows,
        )

    # Non-JSON / server-ish error / other split-worthy status.
    if len(
        corp_codes
    ) > 1:

        print(
            f"{indent}  -> split "
            f"because {result['kind']}: "
            f"{result.get('error')}"
        )

        left, right = split_target_frame(
            targets
        )

        manifests = []
        rows_list = []

        for suffix, part in [
            (
                "L",
                left,
            ),
            (
                "R",
                right,
            ),
        ]:
            if part.empty:
                continue

            pm, ar = recover_recursive(
                h1=h1,
                session=session,
                api_key=api_key,
                targets=part,
                bsns_year=bsns_year,
                reprt_code=reprt_code,
                logical_batch_id=(
                    f"{logical_batch_id}_{suffix}"
                ),
                depth=depth + 1,
            )

            manifests.append(
                pm
            )

            if not ar.empty:
                rows_list.append(
                    ar
                )

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

        return (
            pd.concat(
                manifests,
                ignore_index=True,
            )
            if manifests
            else pd.DataFrame(),
            pd.concat(
                rows_list,
                ignore_index=True,
            )
            if rows_list
            else pd.DataFrame(),
        )

    # One company and still broken: preserve as terminal api_error.
    print(
        f"{indent}  -> terminal api_error "
        f"corp={corp_codes[0] if corp_codes else 'NA'} "
        f"error={result.get('error')} "
        f"preview={result.get('preview')}"
    )

    period_manifest = (
        make_terminal_error_manifest(
            targets=targets,
            batch_id=(
                f"{logical_batch_id}"
                f"_terminal"
            ),
            result=result,
        )
    )

    return (
        period_manifest,
        pd.DataFrame(),
    )


def main():

    if not H1_PATH.exists():
        raise FileNotFoundError(
            H1_PATH
        )

    h1 = load_h1()

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

    if not h1.TARGETS_PARQUET.exists():
        raise FileNotFoundError(
            h1.TARGETS_PARQUET
        )

    if not h1.MANIFEST_PARQUET.exists():
        raise FileNotFoundError(
            h1.MANIFEST_PARQUET
        )

    targets = h1.normalize_id_columns(
        pd.read_parquet(
            h1.TARGETS_PARQUET
        )
    )

    manifest_state = pd.read_parquet(
        h1.MANIFEST_PARQUET
    )

    if h1.ROWS_PARQUET.exists():
        row_state = pd.read_parquet(
            h1.ROWS_PARQUET
        )
    else:
        row_state = pd.DataFrame()

    manifest_state[
        "expected_rcept_no"
    ] = (
        manifest_state[
            "expected_rcept_no"
        ]
        .astype(str)
    )

    error_receipts = set(
        manifest_state.loc[
            manifest_state[
                "collection_status"
            ].eq(
                "api_error"
            ),
            "expected_rcept_no",
        ]
        .astype(str)
    )

    error_targets = targets.loc[
        targets[
            "expected_rcept_no"
        ]
        .astype(str)
        .isin(
            error_receipts
        )
    ].copy()

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-H1C API ERROR ADAPTIVE RECOVERY"
    )
    print(
        "=" * 80
    )

    print(
        f"\napi_error targets: "
        f"{len(error_targets):,}"
    )

    if error_targets.empty:
        print(
            "복구할 api_error가 없습니다."
        )
        return

    session = requests.Session()

    grouped = list(
        error_targets.groupby(
            [
                "bsns_year",
                "reprt_code",
            ],
            sort=True,
        )
    )

    for idx, (
        (
            bsns_year,
            reprt_code,
        ),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        print(
            f"\n[{idx}/{len(grouped)}] "
            f"year={bsns_year} "
            f"reprt={reprt_code} "
            f"targets={len(group)}"
        )

        period_manifest, api_rows = (
            recover_recursive(
                h1=h1,
                session=session,
                api_key=api_key,
                targets=group,
                bsns_year=int(
                    bsns_year
                ),
                reprt_code=str(
                    reprt_code
                ),
                logical_batch_id=(
                    f"recovery_"
                    f"{int(bsns_year)}_"
                    f"{reprt_code}"
                ),
            )
        )

        manifest_state, row_state = (
            merge_state(
                h1=h1,
                manifest_state=manifest_state,
                row_state=row_state,
                period_manifest=period_manifest,
                api_rows=api_rows,
            )
        )

        h1.save_outputs(
            manifest_state,
            row_state,
        )

        print(
            "  checkpoint saved"
        )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "RECOVERY SUMMARY"
    )
    print(
        "=" * 80
    )

    final_manifest = pd.read_parquet(
        h1.MANIFEST_PARQUET
    )

    print(
        "\n[Collection status]"
    )

    print(
        final_manifest[
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
        final_manifest[
            "pit_verified"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    remaining = final_manifest.loc[
        final_manifest[
            "collection_status"
        ].eq(
            "api_error"
        )
    ].copy()

    print(
        f"\nRemaining api_error: "
        f"{len(remaining):,}"
    )

    if not remaining.empty:
        cols = [
            "stock_code",
            "corp_code",
            "canonical_period_key",
            "expected_rcept_no",
            "api_status",
            "api_error",
        ]

        cols = [
            c
            for c in cols
            if c in remaining.columns
        ]

        print(
            remaining[
                cols
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n다음 단계:"
        "\napi_error=0이면 05A5-H2 coverage audit 실행."
        "\n개별 회사 api_error가 남으면 해당 receipt만"
        "\nfnlttSinglAcntAll 또는 source-level fallback 대상으로 넘긴다."
    )


if __name__ == "__main__":
    main()
