from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H1A. Non-Corrected Target Coverage Audit
#
# 목적
# ------------------------------------------------------------
# 전체 regular-report period universe에서
#
#   all periods
# - corrected periods
# = expected non-corrected periods
#
# 와 H1 target parquet을 비교해, target에서 빠진 period를
# 정확한 이유별로 분류한다.
#
# API 호출 없음.
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h1a_dart_noncorrected_target_coverage_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

H1_PATH = (
    SCRIPTS_DIR
    / "05a5h1_dart_noncorrected_batch_collector.py"
)

CORRECTION_CHAIN_PATH = (
    INTERIM
    / "dart_correction_chains.csv"
)

TARGETS_PATH = (
    INTERIM
    / "dart_noncorrected_target_periods.parquet"
)

OUT_MISSING = (
    INTERIM
    / "dart_noncorrected_target_missing_periods_audit.csv"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dart_h1_target_audit_base",
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
        "dart_h1_target_audit_base"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def main():

    if not H1_PATH.exists():
        raise FileNotFoundError(
            H1_PATH
        )

    if not CORRECTION_CHAIN_PATH.exists():
        raise FileNotFoundError(
            CORRECTION_CHAIN_PATH
        )

    if not TARGETS_PATH.exists():
        raise FileNotFoundError(
            TARGETS_PATH
        )

    h1 = load_module()

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

    universe = universe.dropna(
        subset=[
            "stock_code",
            "rcept_no",
            "rcept_dt",
            "canonical_period_key",
        ]
    ).copy()

    correction = (
        h1.normalize_id_columns(
            pd.read_csv(
                CORRECTION_CHAIN_PATH,
                dtype={
                    "stock_code": str,
                    "corp_code": str,
                    "rcept_no": str,
                },
                low_memory=False,
            )
        )
    )

    correction[
        "canonical_period_key"
    ] = correction.apply(
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

    corrected_keys = set(
        zip(
            correction[
                "stock_code"
            ].astype(str),
            correction[
                "canonical_period_key"
            ].astype(str),
        )
    )

    universe_keys = (
        universe[
            [
                "stock_code",
                "canonical_period_key",
            ]
        ]
        .drop_duplicates()
    )

    universe_keys[
        "is_corrected_period"
    ] = [
        (
            str(stock),
            str(period),
        )
        in corrected_keys
        for stock, period
        in zip(
            universe_keys[
                "stock_code"
            ],
            universe_keys[
                "canonical_period_key"
            ],
        )
    ]

    expected_noncorrected = (
        universe_keys.loc[
            ~universe_keys[
                "is_corrected_period"
            ]
        ]
        [
            [
                "stock_code",
                "canonical_period_key",
            ]
        ]
        .copy()
    )

    targets = (
        pd.read_parquet(
            TARGETS_PATH
        )
    )

    targets[
        "stock_code"
    ] = (
        targets[
            "stock_code"
        ]
        .astype(str)
        .str.zfill(6)
    )

    target_keys = set(
        zip(
            targets[
                "stock_code"
            ].astype(str),
            targets[
                "canonical_period_key"
            ].astype(str),
        )
    )

    missing = (
        expected_noncorrected.loc[
            [
                (
                    str(stock),
                    str(period),
                )
                not in target_keys
                for stock, period
                in zip(
                    expected_noncorrected[
                        "stock_code"
                    ],
                    expected_noncorrected[
                        "canonical_period_key"
                    ],
                )
            ]
        ]
        .copy()
    )

    if missing.empty:
        print(
            "\nMissing target periods: 0"
        )
        return

    # period receipt counts
    counts = (
        universe.groupby(
            [
                "stock_code",
                "canonical_period_key",
            ]
        )[
            "rcept_no"
        ]
        .nunique()
        .rename(
            "period_receipt_count"
        )
        .reset_index()
    )

    missing = missing.merge(
        counts,
        on=[
            "stock_code",
            "canonical_period_key",
        ],
        how="left",
    )

    # representative metadata
    meta_cols = [
        "stock_code",
        "canonical_period_key",
        "corp_code",
        "corp_name",
        "period_key",
        "report_nm",
        "rcept_no",
        "rcept_dt",
        "reprt_code",
    ]

    meta_cols = [
        c
        for c in meta_cols
        if c in universe.columns
    ]

    meta = (
        universe[
            meta_cols
        ]
        .sort_values(
            [
                "stock_code",
                "canonical_period_key",
                "rcept_dt",
                "rcept_no",
            ]
        )
        .drop_duplicates(
            [
                "stock_code",
                "canonical_period_key",
            ],
            keep="last",
        )
    )

    missing = missing.merge(
        meta,
        on=[
            "stock_code",
            "canonical_period_key",
        ],
        how="left",
    )

    fiscal_month = (
        h1.infer_fiscal_year_end_month(
            universe
        )
    )

    missing[
        "fiscal_year_end_month"
    ] = missing[
        "stock_code"
    ].map(
        fiscal_month
    )

    if "reprt_code" not in missing.columns:
        missing[
            "reprt_code"
        ] = pd.NA

    missing[
        "reprt_code_inferred"
    ] = missing.apply(
        lambda r:
        h1.infer_reprt_code(
            r[
                "canonical_period_key"
            ],
            (
                int(
                    r[
                        "fiscal_year_end_month"
                    ]
                )
                if pd.notna(
                    r[
                        "fiscal_year_end_month"
                    ]
                )
                else None
            ),
            r.get(
                "reprt_code"
            ),
        ),
        axis=1,
    )

    # corp code mapping assist
    corp_map = (
        h1.discover_corp_code_map()
    )

    if not corp_map.empty:
        mapped = (
            missing[
                [
                    "stock_code",
                ]
            ]
            .merge(
                corp_map,
                on="stock_code",
                how="left",
            )[
                "corp_code"
            ]
        )

        if "corp_code" not in missing.columns:
            missing[
                "corp_code"
            ] = mapped
        else:
            missing[
                "corp_code"
            ] = (
                missing[
                    "corp_code"
                ]
                .astype("string")
                .fillna(
                    mapped
                )
            )

    def classify(row):

        reasons = []

        if (
            pd.notna(
                row.get(
                    "period_receipt_count"
                )
            )
            and int(
                row[
                    "period_receipt_count"
                ]
            )
            > 1
        ):
            reasons.append(
                "multiple_receipts_outside_corrected_chain"
            )

        corp_code = row.get(
            "corp_code"
        )

        if (
            pd.isna(
                corp_code
            )
            or str(
                corp_code
            ).strip()
            in {
                "",
                "<NA>",
                "nan",
            }
        ):
            reasons.append(
                "corp_code_missing"
            )

        if pd.isna(
            row.get(
                "reprt_code_inferred"
            )
        ):
            reasons.append(
                "reprt_code_unresolved"
            )

        if not reasons:
            reasons.append(
                "other_target_construction_exclusion"
            )

        return " | ".join(
            reasons
        )

    missing[
        "exclusion_reason"
    ] = missing.apply(
        classify,
        axis=1,
    )

    missing = missing.sort_values(
        [
            "exclusion_reason",
            "stock_code",
            "canonical_period_key",
        ]
    )

    missing.to_csv(
        OUT_MISSING,
        index=False,
        encoding="utf-8-sig",
    )

    all_periods = len(
        universe_keys
    )

    corrected_periods = int(
        universe_keys[
            "is_corrected_period"
        ].sum()
    )

    expected_count = len(
        expected_noncorrected
    )

    target_count = len(
        target_keys
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "05A5-H1A NON-CORRECTED TARGET COVERAGE AUDIT"
    )
    print(
        "=" * 80
    )

    print(
        f"\nall distinct periods       : "
        f"{all_periods:,}"
    )

    print(
        f"corrected periods          : "
        f"{corrected_periods:,}"
    )

    print(
        f"expected non-corrected     : "
        f"{expected_count:,}"
    )

    print(
        f"H1 target periods          : "
        f"{target_count:,}"
    )

    print(
        f"missing from H1 targets    : "
        f"{len(missing):,}"
    )

    print(
        "\n[Exclusion reasons]"
    )

    print(
        missing[
            "exclusion_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Missing periods]"
    )

    show_cols = [
        "stock_code",
        "corp_name",
        "canonical_period_key",
        "period_key",
        "report_nm",
        "rcept_no",
        "rcept_dt",
        "period_receipt_count",
        "corp_code",
        "fiscal_year_end_month",
        "reprt_code_inferred",
        "exclusion_reason",
    ]

    show_cols = [
        c
        for c in show_cols
        if c in missing.columns
    ]

    print(
        missing[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUT_MISSING}"
    )


if __name__ == "__main__":
    main()
