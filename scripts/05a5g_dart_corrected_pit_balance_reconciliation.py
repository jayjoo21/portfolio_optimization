from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 05A5-G. Corrected PIT Balance Reconciliation
#
# 목적
# ------------------------------------------------------------
# corrected-period strict PIT 결과의 balance QA를 최종 정리한다.
#
# 규칙
# 1) 반올림 허용오차:
#       abs_gap <= 2,000,000 KRW
#       OR relative_gap <= 1e-6
#    이면 QA PASS
#
# 2) 허용오차를 넘는 XBRL anomaly는
#    "같은 rcept_no"의 document.xml 교차검증 결과가 PASS인 경우에만
#    balance metrics를 document 값으로 대체한다.
#
# 3) 원본 파일은 절대 덮어쓰지 않는다.
#    reconciled 별도 산출물을 만든다.
#
# 현재 확인된 anomaly:
#   종근당홀딩스 2019 FY
#   rcept_no = 20200330003558
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_corrected_pit_receipt_values.parquet
#   dart_corrected_pit_receipt_wide.parquet
#   dart_corrected_pit_parse_manifest.parquet
#   dart_balance_same_receipt_document_selected_20200330003558.csv
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_corrected_pit_receipt_values_reconciled.parquet
#   dart_corrected_pit_receipt_wide_reconciled.parquet
#   dart_corrected_pit_parse_manifest_reconciled.parquet
#   dart_corrected_pit_balance_reconciliation_log.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5g_dart_corrected_pit_balance_reconciliation.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

VALUES_IN = INTERIM / "dart_corrected_pit_receipt_values.parquet"
WIDE_IN = INTERIM / "dart_corrected_pit_receipt_wide.parquet"
MANIFEST_IN = INTERIM / "dart_corrected_pit_parse_manifest.parquet"

VALUES_OUT = INTERIM / "dart_corrected_pit_receipt_values_reconciled.parquet"
WIDE_OUT = INTERIM / "dart_corrected_pit_receipt_wide_reconciled.parquet"
MANIFEST_OUT = INTERIM / "dart_corrected_pit_parse_manifest_reconciled.parquet"

LOG_OUT = INTERIM / "dart_corrected_pit_balance_reconciliation_log.csv"

ABS_TOLERANCE_KRW = 2_000_000
REL_TOLERANCE = 1e-6

TARGET_RECEIPT = "20200330003558"

TARGET_DOC_SELECTION = (
    INTERIM
    / f"dart_balance_same_receipt_document_selected_{TARGET_RECEIPT}.csv"
)

BALANCE_METRICS = {
    "assets": "assets",
    "liabilities": "liabilities",
    "equity": "equity_total",
}


def qa_pass(
    assets: float,
    liabilities: float,
    equity: float,
) -> tuple[bool, float, float]:

    gap = (
        float(assets)
        - float(liabilities)
        - float(equity)
    )

    rel_gap = (
        abs(gap) / abs(float(assets))
        if float(assets) != 0
        else np.nan
    )

    passed = (
        abs(gap) <= ABS_TOLERANCE_KRW
        or (
            pd.notna(rel_gap)
            and rel_gap <= REL_TOLERANCE
        )
    )

    return bool(passed), gap, rel_gap


def main():

    for path in [
        VALUES_IN,
        WIDE_IN,
        MANIFEST_IN,
        TARGET_DOC_SELECTION,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"필수 파일 없음: {path}"
            )

    values = pd.read_parquet(
        VALUES_IN
    )

    wide = pd.read_parquet(
        WIDE_IN
    )

    manifest = pd.read_parquet(
        MANIFEST_IN
    )

    doc_selected = pd.read_csv(
        TARGET_DOC_SELECTION,
        dtype={
            "rcept_no": str,
        },
    )

    for df in [
        values,
        wide,
        manifest,
    ]:
        df["rcept_no"] = (
            df["rcept_no"]
            .astype(str)
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-G CORRECTED PIT BALANCE RECONCILIATION"
    )
    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 1) 현재 balance QA 재계산
    # --------------------------------------------------------

    wide["balance_gap_raw"] = (
        wide["assets"]
        - wide["liabilities"]
        - wide["equity_total"]
    )

    wide["balance_relative_gap_raw"] = (
        wide["balance_gap_raw"]
        .abs()
        / wide["assets"]
        .abs()
        .replace(
            0,
            np.nan,
        )
    )

    wide["balance_pass_tolerant_raw"] = (
        wide.apply(
            lambda r:
            (
                bool(
                    abs(
                        r["balance_gap_raw"]
                    )
                    <= ABS_TOLERANCE_KRW
                    or (
                        pd.notna(
                            r[
                                "balance_relative_gap_raw"
                            ]
                        )
                        and r[
                            "balance_relative_gap_raw"
                        ]
                        <= REL_TOLERANCE
                    )
                )
                if (
                    pd.notna(
                        r.get(
                            "assets"
                        )
                    )
                    and pd.notna(
                        r.get(
                            "liabilities"
                        )
                    )
                    and pd.notna(
                        r.get(
                            "equity_total"
                        )
                    )
                )
                else pd.NA
            ),
            axis=1,
        )
    )

    hard_fail = wide.loc[
        wide[
            "balance_pass_tolerant_raw"
        ].eq(
            False
        )
    ].copy()

    print(
        f"\nHard failures after tolerance: "
        f"{len(hard_fail):,}"
    )

    if not hard_fail.empty:
        print(
            hard_fail[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "assets",
                    "liabilities",
                    "equity_total",
                    "balance_gap_raw",
                    "balance_relative_gap_raw",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # 현재 audit 결과상 1건만 남아야 한다.
    unexpected = hard_fail.loc[
        ~hard_fail[
            "rcept_no"
        ].eq(
            TARGET_RECEIPT
        )
    ]

    if not unexpected.empty:
        raise RuntimeError(
            "예상하지 못한 hard balance failure가 있습니다.\n"
            + unexpected[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "balance_gap_raw",
                    "balance_relative_gap_raw",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 2) same-receipt document cross-check 값 확인
    # --------------------------------------------------------

    selected = doc_selected.loc[
        doc_selected[
            "selection_status"
        ].isin(
            [
                "selected",
                "selected_consensus",
            ]
        )
    ].copy()

    replacement = {}

    for doc_family, metric in (
        BALANCE_METRICS.items()
    ):

        hit = selected.loc[
            selected[
                "account_family"
            ].eq(
                doc_family
            )
        ]

        if hit.empty:
            raise RuntimeError(
                f"document cross-check에서 {doc_family}를 선택하지 못했습니다."
            )

        replacement[
            metric
        ] = float(
            hit.iloc[0][
                "selected_value_krw"
            ]
        )

    doc_pass, doc_gap, doc_rel_gap = (
        qa_pass(
            replacement[
                "assets"
            ],
            replacement[
                "liabilities"
            ],
            replacement[
                "equity_total"
            ],
        )
    )

    if not doc_pass:
        raise RuntimeError(
            "same-receipt document cross-check 자체가 balance QA를 통과하지 못했습니다."
        )

    print(
        "\n[Same-receipt document fallback]"
    )

    print(
        f"rcept_no   : {TARGET_RECEIPT}"
    )
    print(
        f"assets     : {replacement['assets']:,.0f}"
    )
    print(
        f"liabilities: {replacement['liabilities']:,.0f}"
    )
    print(
        f"equity     : {replacement['equity_total']:,.0f}"
    )
    print(
        f"gap        : {doc_gap:,.0f}"
    )
    print(
        f"gap_pct    : {doc_rel_gap * 100:.9f}%"
    )

    # --------------------------------------------------------
    # 3) 원본 보존 + reconciled copy 생성
    # --------------------------------------------------------

    values_out = values.copy()
    wide_out = wide.copy()
    manifest_out = manifest.copy()

    # provenance columns
    for col, default in [
        (
            "qa_reconciled",
            False,
        ),
        (
            "qa_reconciliation_reason",
            pd.NA,
        ),
        (
            "qa_reconciliation_source",
            pd.NA,
        ),
    ]:
        if col not in values_out.columns:
            values_out[
                col
            ] = default

    target_mask = (
        values_out[
            "rcept_no"
        ].eq(
            TARGET_RECEIPT
        )
        & values_out[
            "metric"
        ].isin(
            [
                "assets",
                "liabilities",
                "equity_total",
            ]
        )
    )

    # original XBRL value 보존
    if (
        "value_before_reconciliation"
        not in values_out.columns
    ):
        values_out[
            "value_before_reconciliation"
        ] = np.nan

    for metric, new_value in (
        replacement.items()
    ):

        mask = (
            values_out[
                "rcept_no"
            ].eq(
                TARGET_RECEIPT
            )
            & values_out[
                "metric"
            ].eq(
                metric
            )
        )

        if not mask.any():
            raise RuntimeError(
                f"long values에서 target {metric} row를 찾지 못했습니다."
            )

        values_out.loc[
            mask,
            "value_before_reconciliation",
        ] = values_out.loc[
            mask,
            "value",
        ]

        values_out.loc[
            mask,
            "value",
        ] = new_value

        values_out.loc[
            mask,
            "qa_reconciled",
        ] = True

        values_out.loc[
            mask,
            "qa_reconciliation_reason",
        ] = (
            "xbrl_balance_tagging_anomaly_same_receipt_document_fallback"
        )

        values_out.loc[
            mask,
            "qa_reconciliation_source",
        ] = (
            "same_receipt_document"
        )

    # wide patch
    wide_target = (
        wide_out[
            "rcept_no"
        ].eq(
            TARGET_RECEIPT
        )
    )

    for metric, new_value in (
        replacement.items()
    ):
        wide_out.loc[
            wide_target,
            metric,
        ] = new_value

    wide_out[
        "balance_gap_final"
    ] = (
        wide_out[
            "assets"
        ]
        - wide_out[
            "liabilities"
        ]
        - wide_out[
            "equity_total"
        ]
    )

    wide_out[
        "balance_relative_gap_final"
    ] = (
        wide_out[
            "balance_gap_final"
        ]
        .abs()
        / wide_out[
            "assets"
        ]
        .abs()
        .replace(
            0,
            np.nan,
        )
    )

    wide_out[
        "balance_equation_pass_final"
    ] = (
        wide_out.apply(
            lambda r:
            (
                bool(
                    abs(
                        r[
                            "balance_gap_final"
                        ]
                    )
                    <= ABS_TOLERANCE_KRW
                    or (
                        pd.notna(
                            r[
                                "balance_relative_gap_final"
                            ]
                        )
                        and r[
                            "balance_relative_gap_final"
                        ]
                        <= REL_TOLERANCE
                    )
                )
                if (
                    pd.notna(
                        r.get(
                            "assets"
                        )
                    )
                    and pd.notna(
                        r.get(
                            "liabilities"
                        )
                    )
                    and pd.notna(
                        r.get(
                            "equity_total"
                        )
                    )
                )
                else pd.NA
            ),
            axis=1,
        )
    )

    # manifest QA flag update
    if (
        "balance_equation_pass_final"
        not in manifest_out.columns
    ):
        manifest_out[
            "balance_equation_pass_final"
        ] = pd.NA

    pass_map = (
        wide_out[
            [
                "rcept_no",
                "balance_equation_pass_final",
            ]
        ]
        .drop_duplicates(
            "rcept_no"
        )
        .set_index(
            "rcept_no"
        )[
            "balance_equation_pass_final"
        ]
    )

    manifest_out[
        "balance_equation_pass_final"
    ] = (
        manifest_out[
            "rcept_no"
        ]
        .map(
            pass_map
        )
    )

    # --------------------------------------------------------
    # 4) reconciliation log
    # --------------------------------------------------------

    original_row = wide.loc[
        wide[
            "rcept_no"
        ].eq(
            TARGET_RECEIPT
        )
    ].iloc[
        0
    ]

    final_row = wide_out.loc[
        wide_out[
            "rcept_no"
        ].eq(
            TARGET_RECEIPT
        )
    ].iloc[
        0
    ]

    log = pd.DataFrame(
        [
            {
                "rcept_no": (
                    TARGET_RECEIPT
                ),
                "stock_code": (
                    original_row[
                        "stock_code"
                    ]
                ),
                "period_key": (
                    original_row[
                        "period_key"
                    ]
                ),
                "reason": (
                    "xbrl_balance_tagging_anomaly"
                ),
                "fallback_source": (
                    "same_receipt_document"
                ),
                "original_assets": (
                    original_row[
                        "assets"
                    ]
                ),
                "original_liabilities": (
                    original_row[
                        "liabilities"
                    ]
                ),
                "original_equity": (
                    original_row[
                        "equity_total"
                    ]
                ),
                "original_gap": (
                    original_row[
                        "balance_gap_raw"
                    ]
                ),
                "original_relative_gap": (
                    original_row[
                        "balance_relative_gap_raw"
                    ]
                ),
                "final_assets": (
                    final_row[
                        "assets"
                    ]
                ),
                "final_liabilities": (
                    final_row[
                        "liabilities"
                    ]
                ),
                "final_equity": (
                    final_row[
                        "equity_total"
                    ]
                ),
                "final_gap": (
                    final_row[
                        "balance_gap_final"
                    ]
                ),
                "final_relative_gap": (
                    final_row[
                        "balance_relative_gap_final"
                    ]
                ),
                "strict_pit_safe": True,
            }
        ]
    )

    # --------------------------------------------------------
    # 5) save
    # --------------------------------------------------------

    values_out.to_parquet(
        VALUES_OUT,
        index=False,
    )

    wide_out.to_parquet(
        WIDE_OUT,
        index=False,
    )

    manifest_out.to_parquet(
        MANIFEST_OUT,
        index=False,
    )

    log.to_csv(
        LOG_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6) final QA
    # --------------------------------------------------------

    final_with_balance = (
        wide_out[
            [
                "assets",
                "liabilities",
                "equity_total",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    final_qa = (
        wide_out.loc[
            final_with_balance,
            "balance_equation_pass_final",
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n[Final balance QA]"
    )

    print(
        final_qa.to_string()
    )

    remaining_fail = wide_out.loc[
        wide_out[
            "balance_equation_pass_final"
        ].eq(
            False
        )
    ]

    print(
        f"\nRemaining hard failures: "
        f"{len(remaining_fail):,}"
    )

    if not remaining_fail.empty:
        print(
            remaining_fail[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "balance_gap_final",
                    "balance_relative_gap_final",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        f"\nValues   : {VALUES_OUT}"
    )
    print(
        f"Wide     : {WIDE_OUT}"
    )
    print(
        f"Manifest : {MANIFEST_OUT}"
    )
    print(
        f"Log      : {LOG_OUT}"
    )

    print(
        "\n다음 단계:"
        "\n05A5-H non-corrected period financial snapshot collection"
    )


if __name__ == "__main__":
    main()
