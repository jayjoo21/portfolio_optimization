from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H1B. Multi-Receipt Outside Correction-Chain Audit
#
# 목적
# ------------------------------------------------------------
# H1 target에서 제외된 15개 period처럼,
# "같은 stock_code + canonical_period_key에 receipt가 2개 이상인데
# 기존 correction chain에는 포함되지 않은" 케이스를 전수 펼쳐본다.
#
# 이 단계에서는 값을 쓰지 않는다.
# 각 receipt의 날짜/보고서명/정정표시/순서를 확인해
# 1) correction detector 누락
# 2) same-day duplicate
# 3) 서로 다른 regular filing이 잘못 같은 period로 canonicalize
# 4) 기타 특수공시
# 로 분류하기 위한 audit이다.
#
# API 호출 없음.
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h1b_dart_multireceipt_outside_chain_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

H1_PATH = (
    SCRIPTS_DIR
    / "05a5h1_dart_noncorrected_batch_collector.py"
)

MISSING_AUDIT_PATH = (
    INTERIM
    / "dart_noncorrected_target_missing_periods_audit.csv"
)

OUT_DETAIL = (
    INTERIM
    / "dart_multireceipt_outside_chain_detail.csv"
)

OUT_PERIOD_SUMMARY = (
    INTERIM
    / "dart_multireceipt_outside_chain_summary.csv"
)


CORRECTION_MARKERS = [
    "기재정정",
    "첨부정정",
    "정정",
    "변경등록",
]


def load_h1():
    spec = importlib.util.spec_from_file_location(
        "dart_h1_base_for_multireceipt_audit",
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
        "dart_h1_base_for_multireceipt_audit"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def has_correction_marker(text) -> bool:
    if pd.isna(text):
        return False

    s = str(text)

    return any(
        marker in s
        for marker in CORRECTION_MARKERS
    )


def rcept_date_from_number(rcept_no: str):
    text = str(rcept_no)

    if re.fullmatch(
        r"\d{14}",
        text,
    ):
        try:
            return pd.to_datetime(
                text[:8],
                format="%Y%m%d",
            )
        except Exception:
            return pd.NaT

    return pd.NaT


def classify_period(group: pd.DataFrame) -> str:

    g = group.sort_values(
        [
            "rcept_dt",
            "rcept_no",
        ]
    ).copy()

    correction_flags = (
        g[
            "has_correction_marker"
        ]
        .fillna(False)
        .astype(bool)
    )

    unique_dates = (
        g[
            "rcept_dt"
        ]
        .dropna()
        .dt.normalize()
        .nunique()
    )

    if correction_flags.any():
        return (
            "explicit_correction_marker_missed_by_chain"
        )

    if unique_dates == 1:
        return (
            "same_day_multiple_receipts_no_marker"
        )

    return (
        "later_multiple_receipts_no_marker"
    )


def main():

    if not H1_PATH.exists():
        raise FileNotFoundError(
            H1_PATH
        )

    if not MISSING_AUDIT_PATH.exists():
        raise FileNotFoundError(
            MISSING_AUDIT_PATH
        )

    h1 = load_h1()

    _, universe = (
        h1.discover_full_period_universe()
    )

    universe = (
        h1.normalize_id_columns(
            universe
        )
    )

    universe[
        "canonical_period_key"
    ] = universe.apply(
        lambda r:
        h1.canonical_period_key(
            r.get(
                "period_key"
            ),
            r.get(
                "report_nm"
            ),
        ),
        axis=1,
    )

    missing = pd.read_csv(
        MISSING_AUDIT_PATH,
        dtype={
            "stock_code": str,
            "corp_code": str,
            "rcept_no": str,
        },
        low_memory=False,
    )

    missing[
        "stock_code"
    ] = (
        missing[
            "stock_code"
        ]
        .astype(str)
        .str.zfill(6)
    )

    missing_keys = set(
        zip(
            missing[
                "stock_code"
            ].astype(str),
            missing[
                "canonical_period_key"
            ].astype(str),
        )
    )

    detail = universe.loc[
        [
            (
                str(stock),
                str(period),
            )
            in missing_keys
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
    ].copy()

    if detail.empty:
        raise RuntimeError(
            "15개 multi-receipt period의 원본 rows를 찾지 못했습니다."
        )

    detail[
        "has_correction_marker"
    ] = detail.apply(
        lambda r:
        (
            has_correction_marker(
                r.get(
                    "report_nm"
                )
            )
            or has_correction_marker(
                r.get(
                    "period_key"
                )
            )
        ),
        axis=1,
    )

    detail[
        "rcept_no_date"
    ] = detail[
        "rcept_no"
    ].apply(
        rcept_date_from_number
    )

    detail[
        "rcept_date_mismatch"
    ] = (
        detail[
            "rcept_no_date"
        ].dt.normalize()
        != detail[
            "rcept_dt"
        ].dt.normalize()
    )

    detail = detail.sort_values(
        [
            "stock_code",
            "canonical_period_key",
            "rcept_dt",
            "rcept_no",
        ]
    ).copy()

    detail[
        "filing_sequence_within_period"
    ] = (
        detail.groupby(
            [
                "stock_code",
                "canonical_period_key",
            ]
        )
        .cumcount()
        + 1
    )

    summary_records = []

    for (
        stock_code,
        canonical_period_key,
    ), group in detail.groupby(
        [
            "stock_code",
            "canonical_period_key",
        ],
        sort=True,
    ):

        g = group.sort_values(
            [
                "rcept_dt",
                "rcept_no",
            ]
        ).copy()

        receipt_count = (
            g[
                "rcept_no"
            ]
            .nunique()
        )

        first_dt = (
            g[
                "rcept_dt"
            ]
            .min()
        )

        last_dt = (
            g[
                "rcept_dt"
            ]
            .max()
        )

        days_between = (
            (
                last_dt.normalize()
                - first_dt.normalize()
            ).days
            if (
                pd.notna(
                    first_dt
                )
                and pd.notna(
                    last_dt
                )
            )
            else pd.NA
        )

        report_names = (
            " || ".join(
                g[
                    "report_nm"
                ]
                .astype("string")
                .fillna("")
                .tolist()
            )
            if "report_nm"
            in g.columns
            else ""
        )

        receipt_nos = (
            " || ".join(
                g[
                    "rcept_no"
                ]
                .astype(str)
                .tolist()
            )
        )

        receipt_dates = (
            " || ".join(
                g[
                    "rcept_dt"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .fillna("")
                .tolist()
            )
        )

        summary_records.append(
            {
                "stock_code": stock_code,
                "corp_name": (
                    g[
                        "corp_name"
                    ]
                    .dropna()
                    .iloc[0]
                    if (
                        "corp_name"
                        in g.columns
                        and not g[
                            "corp_name"
                        ].dropna().empty
                    )
                    else pd.NA
                ),
                "canonical_period_key": (
                    canonical_period_key
                ),
                "receipt_count": int(
                    receipt_count
                ),
                "first_rcept_dt": (
                    first_dt
                ),
                "last_rcept_dt": (
                    last_dt
                ),
                "days_between_receipts": (
                    days_between
                ),
                "has_any_correction_marker": (
                    bool(
                        g[
                            "has_correction_marker"
                        ].any()
                    )
                ),
                "has_rcept_date_mismatch": (
                    bool(
                        g[
                            "rcept_date_mismatch"
                        ].any()
                    )
                ),
                "classification": (
                    classify_period(
                        g
                    )
                ),
                "receipt_nos": (
                    receipt_nos
                ),
                "receipt_dates": (
                    receipt_dates
                ),
                "report_names": (
                    report_names
                ),
            }
        )

    summary = pd.DataFrame(
        summary_records
    )

    detail_cols = [
        "stock_code",
        "corp_code",
        "corp_name",
        "canonical_period_key",
        "period_key",
        "report_nm",
        "rcept_no",
        "rcept_dt",
        "rcept_no_date",
        "rcept_date_mismatch",
        "has_correction_marker",
        "filing_sequence_within_period",
        "reprt_code",
    ]

    detail_cols = [
        c
        for c in detail_cols
        if c in detail.columns
    ]

    detail[
        detail_cols
    ].to_csv(
        OUT_DETAIL,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_PERIOD_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H1B MULTI-RECEIPT OUTSIDE CORRECTION-CHAIN AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        f"\nperiods : "
        f"{len(summary):,}"
    )

    print(
        f"receipts: "
        f"{detail['rcept_no'].nunique():,}"
    )

    print(
        "\n[Classification]"
    )

    print(
        summary[
            "classification"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Correction markers]"
    )

    print(
        summary[
            "has_any_correction_marker"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Receipt-number date mismatch]"
    )

    print(
        summary[
            "has_rcept_date_mismatch"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Period summary]"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n[Receipt detail]"
    )

    print(
        detail[
            detail_cols
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSummary: "
        f"{OUT_PERIOD_SUMMARY}"
    )

    print(
        f"Detail : "
        f"{OUT_DETAIL}"
    )

    print(
        "\n다음 판단:"
        "\n- explicit_correction_marker_missed_by_chain"
        "\n  → correction chain detector 보완 대상"
        "\n- later_multiple_receipts_no_marker"
        "\n  → 두 receipt의 공시 성격/본문 차이 추가 audit"
        "\n- same_day_multiple_receipts_no_marker"
        "\n  → 동일일 제출 순서 및 EOD version rule 검토"
        "\n- rcept_date_mismatch=True"
        "\n  → universe의 rcept_dt 생성/파싱 로직 별도 점검"
    )


if __name__ == "__main__":
    main()
