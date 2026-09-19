from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 05A5-H6C2A. CORE5 SOURCE MAP / EVIDENCE DISCOVERY
#
# 목적
# ------------------------------------------------------------
# H6C1 core5 residual 248건에 대해,
# 이미 data/interim/dart 아래에 존재하는 parquet 중
# 어떤 파일이 실제 row-level account evidence를 보유하는지 찾는다.
#
# 이유
# ------------------------------------------------------------
# H6B7A~M 캐시는 "API full no_data 890건"용이므로
# core5 248건에 그대로 적용하면 안 된다.
#
# 이 단계에서는:
# - API 호출 없음
# - ZIP 파싱 없음
# - 값 복구 없음
# - 파일 수정 없음
#
# 단지 기존 parquet의 schema와 rcept_no overlap을 검사한다.
#
# 실행:
# python scripts\05a5h6c2a_dart_core5_source_map.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

OUT_SOURCE_MAP = (
    INTERIM
    / "dart_noncorrected_core5_source_map.csv"
)

OUT_RECEIPT_COVERAGE = (
    INTERIM
    / "dart_noncorrected_core5_source_receipt_coverage.csv"
)


EXPECTED_CORE5 = 248


# Columns that suggest a file contains useful account-level evidence.
EVIDENCE_COLUMN_HINTS = {
    "rcept_no",
    "stock_code",
    "corp_code",
    "corp_name",
    "period_key",
    "reprt_code",
    "bsns_year",
    "fs_div",
    "sj_div",
    "account_id",
    "account_nm",
    "account_name",
    "account_label",
    "row_label",
    "primary_row_label",
    "account_family",
    "metric",
    "metric_name",
    "selector_value_krw",
    "value_krw",
    "thstrm_amount",
    "frmtrm_amount",
    "bfefrmtrm_amount",
    "table_index",
    "row_index",
    "heading_context",
    "period_v2_selected_column",
}


STRONG_ACCOUNT_HINTS = {
    "account_id",
    "account_nm",
    "account_name",
    "account_label",
    "row_label",
    "primary_row_label",
    "account_family",
    "metric",
    "metric_name",
}


VALUE_HINTS = {
    "selector_value_krw",
    "value_krw",
    "thstrm_amount",
    "frmtrm_amount",
    "bfefrmtrm_amount",
    "value",
    "amount",
}


def normalize_receipt(
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


def safe_relative(
    path: Path,
) -> str:

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except Exception:
        return str(
            path
        )


def candidate_priority(
    schema_names: list[str],
    overlap_rows: int,
    overlap_receipts: int,
) -> tuple[int, int, int, int]:

    names = set(
        schema_names
    )

    account_score = len(
        names
        & STRONG_ACCOUNT_HINTS
    )

    value_score = len(
        names
        & VALUE_HINTS
    )

    # More rows than receipts usually means row-level long-form evidence.
    long_form_score = int(
        overlap_rows
        > overlap_receipts
    )

    return (
        long_form_score,
        account_score,
        value_score,
        overlap_rows,
    )


def main():

    if not CORE5.exists():
        raise FileNotFoundError(
            CORE5
        )

    core5 = pd.read_parquet(
        CORE5
    )

    if len(
        core5
    ) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 rows, "
            f"found {len(core5)}."
        )

    if "rcept_no" not in core5.columns:
        raise RuntimeError(
            "CORE5 file has no rcept_no."
        )

    core5[
        "rcept_no"
    ] = normalize_receipt(
        core5[
            "rcept_no"
        ]
    )

    target_receipts = set(
        core5[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2A CORE5 SOURCE MAP / EVIDENCE DISCOVERY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCore5 receipts: "
        f"{len(target_receipts):,}"
    )

    parquet_files = sorted(
        INTERIM.glob(
            "*.parquet"
        )
    )

    # Avoid scanning our own core5 output as a "source".
    exclude_exact = {
        CORE5.resolve(),
    }

    parquet_files = [
        path
        for path in parquet_files
        if path.resolve()
        not in exclude_exact
    ]

    print(
        f"Parquet files to inspect: "
        f"{len(parquet_files):,}"
    )

    records = []
    receipt_source_rows = []

    for i, path in enumerate(
        parquet_files,
        start=1,
    ):

        try:
            parquet_file = pq.ParquetFile(
                path
            )

            schema_names = (
                parquet_file.schema_arrow.names
            )

        except Exception as exc:

            records.append(
                {
                    "file":
                    safe_relative(
                        path
                    ),

                    "status":
                    "SCHEMA_ERROR",

                    "error":
                    repr(
                        exc
                    ),

                    "file_rows":
                    None,

                    "overlap_rows":
                    0,

                    "overlap_receipts":
                    0,

                    "coverage_pct":
                    0.0,

                    "has_rcept_no":
                    False,

                    "looks_row_level":
                    False,

                    "account_hint_count":
                    0,

                    "value_hint_count":
                    0,

                    "useful_columns":
                    "",
                }
            )

            continue

        names = set(
            schema_names
        )

        if "rcept_no" not in names:

            records.append(
                {
                    "file":
                    safe_relative(
                        path
                    ),

                    "status":
                    "NO_RCEPT_NO",

                    "error":
                    "",

                    "file_rows":
                    parquet_file.metadata.num_rows,

                    "overlap_rows":
                    0,

                    "overlap_receipts":
                    0,

                    "coverage_pct":
                    0.0,

                    "has_rcept_no":
                    False,

                    "looks_row_level":
                    False,

                    "account_hint_count":
                    len(
                        names
                        & STRONG_ACCOUNT_HINTS
                    ),

                    "value_hint_count":
                    len(
                        names
                        & VALUE_HINTS
                    ),

                    "useful_columns":
                    "|".join(
                        sorted(
                            names
                            & EVIDENCE_COLUMN_HINTS
                        )
                    ),
                }
            )

            continue

        # Read only rcept_no first.
        try:
            receipt_col = pd.read_parquet(
                path,
                columns=[
                    "rcept_no",
                ],
            )

        except Exception as exc:

            records.append(
                {
                    "file":
                    safe_relative(
                        path
                    ),

                    "status":
                    "RCEPT_READ_ERROR",

                    "error":
                    repr(
                        exc
                    ),

                    "file_rows":
                    parquet_file.metadata.num_rows,

                    "overlap_rows":
                    0,

                    "overlap_receipts":
                    0,

                    "coverage_pct":
                    0.0,

                    "has_rcept_no":
                    True,

                    "looks_row_level":
                    False,

                    "account_hint_count":
                    len(
                        names
                        & STRONG_ACCOUNT_HINTS
                    ),

                    "value_hint_count":
                    len(
                        names
                        & VALUE_HINTS
                    ),

                    "useful_columns":
                    "|".join(
                        sorted(
                            names
                            & EVIDENCE_COLUMN_HINTS
                        )
                    ),
                }
            )

            continue

        receipt_col[
            "rcept_no"
        ] = normalize_receipt(
            receipt_col[
                "rcept_no"
            ]
        )

        matched_mask = (
            receipt_col[
                "rcept_no"
            ].isin(
                target_receipts
            )
        )

        overlap_rows = int(
            matched_mask.sum()
        )

        if overlap_rows:

            matched_receipts = set(
                receipt_col.loc[
                    matched_mask,
                    "rcept_no",
                ]
            )

        else:
            matched_receipts = set()

        overlap_receipts = len(
            matched_receipts
        )

        coverage_pct = (
            100.0
            * overlap_receipts
            / len(
                target_receipts
            )
        )

        account_hint_count = len(
            names
            & STRONG_ACCOUNT_HINTS
        )

        value_hint_count = len(
            names
            & VALUE_HINTS
        )

        looks_row_level = (
            overlap_rows
            > overlap_receipts
            and account_hint_count
            > 0
        )

        status = (
            "OVERLAP"
            if overlap_receipts
            else "NO_OVERLAP"
        )

        records.append(
            {
                "file":
                safe_relative(
                    path
                ),

                "status":
                status,

                "error":
                "",

                "file_rows":
                parquet_file.metadata.num_rows,

                "overlap_rows":
                overlap_rows,

                "overlap_receipts":
                overlap_receipts,

                "coverage_pct":
                round(
                    coverage_pct,
                    2,
                ),

                "has_rcept_no":
                True,

                "looks_row_level":
                looks_row_level,

                "account_hint_count":
                account_hint_count,

                "value_hint_count":
                value_hint_count,

                "useful_columns":
                "|".join(
                    sorted(
                        names
                        & EVIDENCE_COLUMN_HINTS
                    )
                ),

                "all_columns":
                "|".join(
                    schema_names
                ),
            }
        )

        if overlap_receipts:

            for rcept_no in sorted(
                matched_receipts
            ):

                receipt_source_rows.append(
                    {
                        "rcept_no":
                        rcept_no,

                        "file":
                        safe_relative(
                            path
                        ),

                        "looks_row_level":
                        looks_row_level,

                        "account_hint_count":
                        account_hint_count,

                        "value_hint_count":
                        value_hint_count,
                    }
                )

    source_map = pd.DataFrame(
        records
    )

    if source_map.empty:
        raise RuntimeError(
            "No parquet files inspected."
        )

    # Priority columns for human review.
    overlap = source_map.loc[
        source_map[
            "overlap_receipts"
        ].fillna(
            0
        ).gt(
            0
        )
    ].copy()

    if not overlap.empty:

        priority = [
            candidate_priority(
                str(
                    row.get(
                        "all_columns",
                        ""
                    )
                ).split(
                    "|"
                ),
                int(
                    row[
                        "overlap_rows"
                    ]
                ),
                int(
                    row[
                        "overlap_receipts"
                    ]
                ),
            )
            for _, row
            in overlap.iterrows()
        ]

        overlap[
            "priority_long_form"
        ] = [
            item[
                0
            ]
            for item in priority
        ]

        overlap[
            "priority_account_score"
        ] = [
            item[
                1
            ]
            for item in priority
        ]

        overlap[
            "priority_value_score"
        ] = [
            item[
                2
            ]
            for item in priority
        ]

        overlap = overlap.sort_values(
            [
                "priority_long_form",
                "priority_account_score",
                "priority_value_score",
                "overlap_receipts",
                "overlap_rows",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )

    source_map.to_csv(
        OUT_SOURCE_MAP,
        index=False,
        encoding="utf-8-sig",
    )

    receipt_coverage = pd.DataFrame(
        receipt_source_rows
    )

    if not receipt_coverage.empty:

        receipt_coverage = (
            receipt_coverage.merge(
                core5[
                    [
                        "rcept_no",
                        "stock_code",
                        "period_key",
                        "missing_metric",
                    ]
                ],
                on="rcept_no",
                how="left",
                validate="many_to_one",
            )
        )

    receipt_coverage.to_csv(
        OUT_RECEIPT_COVERAGE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Files with core5 overlap]"
    )

    if overlap.empty:
        print(
            "None"
        )

    else:
        show_cols = [
            "file",
            "file_rows",
            "overlap_rows",
            "overlap_receipts",
            "coverage_pct",
            "looks_row_level",
            "account_hint_count",
            "value_hint_count",
            "useful_columns",
        ]

        with pd.option_context(
            "display.max_colwidth",
            160,
            "display.width",
            360,
            "display.max_rows",
            60,
        ):

            print(
                overlap[
                    show_cols
                ]
                .head(
                    40
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Best row-level candidate sources]"
    )

    if overlap.empty:

        print(
            "None"
        )

    else:

        row_level = overlap.loc[
            overlap[
                "looks_row_level"
            ].fillna(
                False
            )
        ]

        if row_level.empty:
            print(
                "None"
            )

        else:
            print(
                row_level[
                    [
                        "file",
                        "overlap_rows",
                        "overlap_receipts",
                        "coverage_pct",
                        "account_hint_count",
                        "value_hint_count",
                    ]
                ]
                .head(
                    20
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Core5 receipts with NO overlapping interim parquet source]"
    )

    covered_receipts = set(
        receipt_coverage[
            "rcept_no"
        ]
    ) if not receipt_coverage.empty else set()

    uncovered = core5.loc[
        ~core5[
            "rcept_no"
        ].isin(
            covered_receipts
        )
    ]

    print(
        f"{len(uncovered):,}"
    )

    if not uncovered.empty:
        print(
            uncovered[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "missing_metric",
                ]
            ]
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

    print(
        "\nOutputs:"
    )

    print(
        f"- Source map       : "
        f"{OUT_SOURCE_MAP}"
    )

    print(
        f"- Receipt coverage : "
        f"{OUT_RECEIPT_COVERAGE}"
    )

    print(
        "\n해석 원칙:"
        "\n- looks_row_level=True + account/value columns가 많은 파일이 H6C2B 후보"
        "\n- validated/wide 같은 1 receipt = 1 row 파일은 source evidence로 쓰지 않음"
        "\n- 다음 단계에서 best existing row-level source를 정확히 고른 뒤 revenue/net-income semantics를 분리"
        "\n- 아직 API/ZIP 재수집이나 값 merge는 전혀 하지 않음"
    )


if __name__ == "__main__":
    main()
