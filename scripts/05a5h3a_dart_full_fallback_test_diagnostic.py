from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H3A. Full-Fallback Test Diagnostic
#
# 목적
# ------------------------------------------------------------
# H3 소량 테스트 결과를 API 재호출 없이 진단한다.
#
# 확인:
# 1) available인데 core 6개가 덜 잡힌 receipt
# 2) 그 receipt의 BS/IS/CIS account_id + account_nm 후보
# 3) no_data의 API status/message
# 4) selector가 놓친 계정 패턴
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h3a_dart_full_fallback_test_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

TARGETS = (
    INTERIM
    / "dart_noncorrected_selective_fallback_targets.parquet"
)

MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

ROWS = (
    INTERIM
    / "dart_noncorrected_full_fallback_rows.parquet"
)

SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

OUT = (
    INTERIM
    / "dart_noncorrected_full_fallback_test_diagnostic.csv"
)


CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]


def main():

    for p in [
        TARGETS,
        MANIFEST,
        ROWS,
        SELECTED,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    targets = pd.read_parquet(
        TARGETS
    )

    manifest = pd.read_parquet(
        MANIFEST
    )

    rows = pd.read_parquet(
        ROWS
    )

    selected = pd.read_parquet(
        SELECTED
    )

    manifest[
        "rcept_no"
    ] = manifest[
        "rcept_no"
    ].astype(str)

    if (
        "_expected_rcept_no"
        in rows.columns
    ):
        rows[
            "_expected_rcept_no"
        ] = rows[
            "_expected_rcept_no"
        ].astype(str)

    if (
        "_expected_rcept_no"
        in selected.columns
    ):
        selected[
            "_expected_rcept_no"
        ] = selected[
            "_expected_rcept_no"
        ].astype(str)

    print(
        "\n"
        + "=" * 90
    )
    print(
        "05A5-H3A FULL-FALLBACK TEST DIAGNOSTIC"
    )
    print(
        "=" * 90
    )

    processed = len(
        manifest
    )

    print(
        f"\nprocessed receipts: "
        f"{processed:,}"
    )

    # --------------------------------------------------------
    # no_data diagnostics
    # --------------------------------------------------------

    no_data = manifest.loc[
        manifest[
            "collection_status"
        ].eq(
            "no_data"
        )
    ].copy()

    print(
        "\n[No-data API status]"
    )

    if no_data.empty:
        print(
            "None"
        )
    else:
        cols = [
            c
            for c in [
                "api_status",
                "api_message",
                "chosen_fs_div",
            ]
            if c in no_data.columns
        ]

        if cols:
            print(
                no_data[
                    cols
                ]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )

        show = [
            c
            for c in [
                "stock_code",
                "corp_name",
                "canonical_period_key",
                "rcept_no",
                "fallback_reason",
                "selected_fs_div_before",
                "api_status",
                "api_message",
            ]
            if c in no_data.columns
        ]

        print(
            "\n[No-data receipts]"
        )
        print(
            no_data[
                show
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # available coverage
    # --------------------------------------------------------

    available = manifest.loc[
        manifest[
            "collection_status"
        ].eq(
            "available"
        )
    ].copy()

    if available.empty:
        print(
            "\nAvailable receipts: 0"
        )
        return

    selected_core = selected.loc[
        selected[
            "metric"
        ].isin(
            CORE_METRICS
        )
    ].copy()

    core_presence = (
        selected_core.assign(
            present=True
        )
        .pivot_table(
            index="_expected_rcept_no",
            columns="metric",
            values="present",
            aggfunc="max",
            fill_value=False,
        )
    )

    for metric in CORE_METRICS:
        if metric not in core_presence.columns:
            core_presence[
                metric
            ] = False

    core_presence[
        "core_metric_count"
    ] = (
        core_presence[
            CORE_METRICS
        ]
        .sum(
            axis=1
        )
    )

    core_presence[
        "missing_core_metrics"
    ] = core_presence.apply(
        lambda r:
        " | ".join(
            metric
            for metric
            in CORE_METRICS
            if not bool(
                r[
                    metric
                ]
            )
        ),
        axis=1,
    )

    available = available.merge(
        core_presence[
            [
                "core_metric_count",
                "missing_core_metrics",
            ]
        ],
        left_on="rcept_no",
        right_index=True,
        how="left",
    )

    available[
        "core_metric_count"
    ] = (
        available[
            "core_metric_count"
        ]
        .fillna(
            0
        )
        .astype(int)
    )

    available[
        "missing_core_metrics"
    ] = (
        available[
            "missing_core_metrics"
        ]
        .fillna(
            " | ".join(
                CORE_METRICS
            )
        )
    )

    print(
        "\n[Available receipt core coverage]"
    )

    show = [
        c
        for c in [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "fallback_reason",
            "chosen_fs_div",
            "api_row_count",
            "core_metric_count",
            "missing_core_metrics",
        ]
        if c in available.columns
    ]

    print(
        available[
            show
        ]
        .sort_values(
            [
                "core_metric_count",
                "stock_code",
            ]
        )
        .to_string(
            index=False
        )
    )

    incomplete = available.loc[
        available[
            "core_metric_count"
        ].lt(
            len(
                CORE_METRICS
            )
        )
    ].copy()

    print(
        f"\nIncomplete available receipts: "
        f"{len(incomplete):,}"
    )

    # --------------------------------------------------------
    # Candidate rows for incomplete available receipts
    # --------------------------------------------------------

    diagnostic_rows = []

    for _, rec in (
        incomplete.iterrows()
    ):

        rcept_no = str(
            rec[
                "rcept_no"
            ]
        )

        rr = rows.loc[
            rows[
                "_expected_rcept_no"
            ].astype(str)
            .eq(
                rcept_no
            )
        ].copy()

        rr = rr.loc[
            rr[
                "sj_div"
            ]
            .astype(str)
            .isin(
                [
                    "BS",
                    "IS",
                    "CIS",
                ]
            )
        ].copy()

        # keep financially relevant-looking rows;
        # if that would become empty, keep all BS/IS/CIS rows.
        text = (
            rr[
                "account_nm"
            ]
            .astype("string")
            .fillna("")
            + " "
            + rr[
                "account_id"
            ]
            .astype("string")
            .fillna("")
        )

        keywords = [
            "매출",
            "수익",
            "영업",
            "순이익",
            "순손익",
            "당기",
            "분기",
            "반기",
            "profit",
            "revenue",
            "income",
            "operating",
            "asset",
            "liabil",
            "equity",
        ]

        mask = pd.Series(
            False,
            index=rr.index,
        )

        lower = text.str.lower()

        for keyword in keywords:
            mask = (
                mask
                | lower.str.contains(
                    keyword.lower(),
                    regex=False,
                )
            )

        candidate = rr.loc[
            mask
        ].copy()

        if candidate.empty:
            candidate = rr

        keep_cols = [
            c
            for c in [
                "_expected_rcept_no",
                "_requested_fs_div",
                "sj_div",
                "sj_nm",
                "account_id",
                "account_nm",
                "account_detail",
                "thstrm_nm",
                "thstrm_amount",
                "thstrm_add_amount",
                "ord",
                "currency",
            ]
            if c in candidate.columns
        ]

        candidate = candidate[
            keep_cols
        ].copy()

        candidate[
            "missing_core_metrics"
        ] = rec[
            "missing_core_metrics"
        ]

        diagnostic_rows.append(
            candidate
        )

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"[Candidates for {rec['stock_code']} "
            f"| {rcept_no} "
            f"| missing={rec['missing_core_metrics']}]"
        )

        print(
            candidate.to_string(
                index=False
            )
        )

    if diagnostic_rows:
        diagnostic = pd.concat(
            diagnostic_rows,
            ignore_index=True,
        )

        diagnostic.to_csv(
            OUT,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nSaved diagnostic: "
            f"{OUT}"
        )

    print(
        "\n다음 판단:"
        "\n- missing metric에 대응하는 account_id/name이 raw rows에 있으면 selector 보완"
        "\n- raw rows에도 없으면 해당 full statement 자체 미제공/업종 특수구조"
        "\n- no_data가 013/014면 API 미제공으로 분류하고 source-level fallback 후보"
    )


if __name__ == "__main__":
    main()
