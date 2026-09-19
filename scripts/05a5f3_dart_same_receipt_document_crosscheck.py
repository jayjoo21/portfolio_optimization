from __future__ import annotations

import importlib.util
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-F3. Same-Receipt Document Cross-Check
#
# 목적
# ------------------------------------------------------------
# XBRL balance outlier에 대해 "같은 rcept_no"의 document.xml을
# 별도로 받아 C4/C5 strict document selector로 교차검증한다.
#
# IMPORTANT
# ------------------------------------------------------------
# - 다음 해 보고서 숫자로 과거를 backfill하지 않는다.
# - 같은 접수번호에서 당시 공개된 document.xml만 사용한다.
# - 이 스크립트는 audit만 수행하며 production 값을 아직 수정하지 않는다.
#
# 기본 대상
# ------------------------------------------------------------
# 종근당홀딩스 2019 FY
# rcept_no = 20200330003558
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5f3_dart_same_receipt_document_crosscheck.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "dart"

SOURCE_MANIFEST = (
    INTERIM_DIR
    / "dart_pit_source_full_manifest.parquet"
)

TARGET_RCEPT_NO = "20200330003558"

QA_RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "pit_balance_qa"
)

OUT_CANDIDATES = (
    INTERIM_DIR
    / f"dart_balance_same_receipt_document_candidates_{TARGET_RCEPT_NO}.csv"
)

OUT_SELECTED = (
    INTERIM_DIR
    / f"dart_balance_same_receipt_document_selected_{TARGET_RCEPT_NO}.csv"
)

DART_DOCUMENT_URL = (
    "https://opendart.fss.or.kr/api/document.xml"
)


def load_module(
    module_name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"module load 실패: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def safe_name(value: str) -> str:
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
        text = text.replace(
            old,
            new,
        )

    return text


def download_document_zip(
    api_key: str,
    rcept_no: str,
) -> bytes:

    response = requests.get(
        DART_DOCUMENT_URL,
        params={
            "crtfc_key": api_key,
            "rcept_no": rcept_no,
        },
        timeout=90,
    )

    response.raise_for_status()

    content = response.content

    if content[:2] != b"PK":
        preview = content[:500].decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            "document.xml이 ZIP을 반환하지 않았습니다.\n"
            f"{preview}"
        )

    with zipfile.ZipFile(
        BytesIO(content)
    ) as zf:
        print(
            "\n[document.xml ZIP members]"
        )

        for info in zf.infolist():
            print(
                f"{info.filename} "
                f"({info.file_size / 1024 / 1024:.2f} MB)"
            )

    return content


def main():

    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "DART_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            ".env의 DART_API_KEY를 찾지 못했습니다."
        )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    manifest[
        "rcept_no"
    ] = (
        manifest[
            "rcept_no"
        ]
        .astype(str)
    )

    target = manifest.loc[
        manifest[
            "rcept_no"
        ].eq(
            TARGET_RCEPT_NO
        )
    ].copy()

    if target.empty:
        raise RuntimeError(
            f"source manifest에서 {TARGET_RCEPT_NO}를 찾지 못했습니다."
        )

    row = target.iloc[0]

    stock_code = str(
        row[
            "stock_code"
        ]
    ).zfill(
        6
    )

    period_key = str(
        row[
            "period_key"
        ]
    )

    rcept_dt = pd.Timestamp(
        row[
            "rcept_dt"
        ]
    )

    filing_sequence = int(
        row.get(
            "filing_sequence",
            1,
        )
    )

    period_dir = (
        QA_RAW_ROOT
        / stock_code
        / safe_name(
            period_key
        )
    )

    period_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_zip = (
        period_dir
        / (
            f"{rcept_dt:%Y%m%d}_"
            f"seq{filing_sequence:02d}_"
            f"{TARGET_RCEPT_NO}_"
            f"document.zip"
        )
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "05A5-F3 SAME-RECEIPT DOCUMENT CROSS-CHECK"
    )
    print(
        "=" * 100
    )

    print(
        f"\nstock_code : {stock_code}"
    )
    print(
        f"period_key : {period_key}"
    )
    print(
        f"rcept_no   : {TARGET_RCEPT_NO}"
    )
    print(
        f"rcept_dt   : {rcept_dt.date()}"
    )

    if out_zip.exists():
        print(
            f"\n기존 QA document 사용: {out_zip}"
        )
    else:
        content = download_document_zip(
            api_key,
            TARGET_RCEPT_NO,
        )

        out_zip.write_bytes(
            content
        )

        print(
            f"\n저장: {out_zip}"
        )

    c1 = load_module(
        "dart_c1_balance_crosscheck",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    c4 = load_module(
        "dart_c4_balance_crosscheck",
        SCRIPTS_DIR
        / "05a5c4_dart_conservative_document_selector.py",
    )

    c5 = load_module(
        "dart_c5_balance_crosscheck",
        SCRIPTS_DIR
        / "05a5c5_dart_strict_consolidated_document_selector.py",
    )

    # C1 path metadata 기준을 QA root로 교체
    c1.RAW_ROOT = QA_RAW_ROOT

    candidate_rows = (
        c1.extract_document_candidates(
            out_zip
        )
    )

    if not candidate_rows:
        raise RuntimeError(
            "document candidate rows = 0"
        )

    document_candidates = pd.DataFrame(
        candidate_rows
    )

    document_candidates.to_csv(
        OUT_CANDIDATES,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nRaw document candidates: "
        f"{len(document_candidates):,}"
    )

    c4_candidates = (
        c4.build_account_candidates(
            document_candidates
        )
    )

    print(
        f"C4 account candidates : "
        f"{len(c4_candidates):,}"
    )

    selected = (
        c5.select_values(
            c4_candidates
        )
    )

    selected.to_csv(
        OUT_SELECTED,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[C5 selected / unresolved]"
    )

    wanted = [
        "account_family",
        "selection_status",
        "table_index",
        "row_index",
        "primary_row_label",
        "selected_column",
        "unit_hint",
        "selected_value_krw",
        "strict_score",
        "consolidated_evidence_reason",
        "heading_context",
    ]

    wanted = [
        col
        for col in wanted
        if col in selected.columns
    ]

    print(
        selected[
            wanted
        ].to_string(
            index=False
        )
    )

    balance = {}

    for family in [
        "assets",
        "liabilities",
        "equity",
    ]:
        hit = selected.loc[
            selected[
                "account_family"
            ].eq(
                family
            )
            & selected[
                "selection_status"
            ].isin(
                [
                    "selected",
                    "selected_consensus",
                ]
            )
        ]

        if not hit.empty:
            balance[
                family
            ] = float(
                hit.iloc[0][
                    "selected_value_krw"
                ]
            )

    print(
        "\n[Same-receipt document balance equation]"
    )

    if all(
        key in balance
        for key in [
            "assets",
            "liabilities",
            "equity",
        ]
    ):
        assets = balance[
            "assets"
        ]

        liabilities = balance[
            "liabilities"
        ]

        equity = balance[
            "equity"
        ]

        gap = (
            assets
            - liabilities
            - equity
        )

        relative_gap = (
            abs(
                gap
            )
            / abs(
                assets
            )
            if assets != 0
            else np.nan
        )

        print(
            f"assets      : {assets:,.0f}"
        )
        print(
            f"liabilities : {liabilities:,.0f}"
        )
        print(
            f"equity      : {equity:,.0f}"
        )
        print(
            f"gap         : {gap:,.0f}"
        )
        print(
            f"gap_pct     : "
            f"{relative_gap * 100:.9f}%"
        )

        if relative_gap <= 1e-6:
            print(
                "result      : PASS"
            )
        else:
            print(
                "result      : FAIL"
            )

    else:
        print(
            "3개 balance 계정을 모두 strict하게 선택하지 못했습니다."
        )

    print(
        f"\nCandidates: {OUT_CANDIDATES}"
    )

    print(
        f"Selected  : {OUT_SELECTED}"
    )

    print(
        "\n해석:"
        "\n- document가 방정식을 만족하면 XBRL tagging anomaly 가능성이 매우 큼"
        "\n- document도 같은 gap이면 원 공시 자체 숫자/표 구조 이슈 가능성"
        "\n- 둘 다 불확실하면 해당 receipt의 balance metrics를 missing 처리"
    )


if __name__ == "__main__":
    main()
