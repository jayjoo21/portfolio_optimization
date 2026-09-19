from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-F2. Targeted XBRL Balance Failure Audit
#
# 목적
# ------------------------------------------------------------
# Balance QA에서 실질적인 outlier로 남은 특정 receipt의
# Assets / Liabilities / Equity 후보 fact 전체를 확인한다.
#
# 기본 대상:
#   종근당홀딩스 2019 FY
#   rcept_no = 20200330003558
#
# 이 스크립트는 값을 수정하지 않는다.
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_balance_targeted_xbrl_candidates_20200330003558.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5f2_dart_balance_targeted_xbrl_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "dart"

SOURCE_MANIFEST = INTERIM_DIR / "dart_pit_source_full_manifest.parquet"

TARGET_RCEPT_NO = "20200330003558"

OUT_PATH = (
    INTERIM_DIR
    / f"dart_balance_targeted_xbrl_candidates_{TARGET_RCEPT_NO}.csv"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"module load 실패: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def main():

    c1 = load_module(
        "dart_c1_targeted_audit",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    # C1은 원래 sample audit 폴더를 RAW_ROOT로 사용하므로,
    # 이번 production PIT source 폴더로 반드시 교체한다.
    c1.RAW_ROOT = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dart"
        / "pit_sources_full"
    )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    manifest["rcept_no"] = (
        manifest["rcept_no"]
        .astype(str)
    )

    target = manifest.loc[
        manifest["rcept_no"].eq(
            TARGET_RCEPT_NO
        )
    ].copy()

    if target.empty:
        raise RuntimeError(
            f"manifest에서 receipt를 찾지 못했습니다: {TARGET_RCEPT_NO}"
        )

    row = target.iloc[0]

    print(
        "\n"
        + "=" * 100
    )
    print(
        "05A5-F2 TARGETED XBRL BALANCE AUDIT"
    )
    print(
        "=" * 100
    )

    print(
        f"\nstock_code : {row.get('stock_code')}"
    )
    print(
        f"period_key : {row.get('period_key')}"
    )
    print(
        f"rcept_no   : {row.get('rcept_no')}"
    )
    print(
        f"rcept_dt   : {row.get('rcept_dt')}"
    )
    print(
        f"source     : {row.get('source_type')}"
    )
    print(
        f"zip        : {row.get('zip_path')}"
    )

    zip_path = Path(
        str(
            row["zip_path"]
        )
    )

    candidates = pd.DataFrame(
        c1.extract_xbrl_candidates(
            zip_path
        )
    )

    if candidates.empty:
        raise RuntimeError(
            "XBRL candidates = 0"
        )

    print(
        f"\nall candidate facts: {len(candidates):,}"
    )

    concept_col = (
        "concept_local_name"
        if "concept_local_name" in candidates.columns
        else None
    )

    if concept_col is None:
        raise RuntimeError(
            f"concept_local_name 컬럼이 없습니다. columns={candidates.columns.tolist()}"
        )

    concept_text = (
        candidates[
            concept_col
        ]
        .astype(str)
        .str.lower()
    )

    mask = (
        concept_text.str.contains(
            "asset",
            regex=False,
        )
        | concept_text.str.contains(
            "liabil",
            regex=False,
        )
        | concept_text.str.contains(
            "equity",
            regex=False,
        )
    )

    balance = candidates.loc[
        mask
    ].copy()

    preferred_cols = [
        "stock_code",
        "period_key",
        "rcept_no",
        "rcept_dt",
        "concept_local_name",
        "value",
        "numeric_value",
        "unit_ref",
        "decimals",
        "context_ref",
        "instant",
        "start_date",
        "end_date",
        "duration_days",
        "dimension_count",
        "dimension_text",
        "is_consolidated",
        "is_separate",
    ]

    cols = [
        c
        for c in preferred_cols
        if c in balance.columns
    ]

    extra_cols = [
        c
        for c in balance.columns
        if c not in cols
        and c in {
            "concept_qname",
            "namespace",
            "precision",
            "scale",
            "sign",
        }
    ]

    cols += extra_cols

    balance = (
        balance[
            cols
        ]
        .sort_values(
            [
                c
                for c in [
                    "concept_local_name",
                    "instant",
                    "context_ref",
                ]
                if c in cols
            ]
        )
        .reset_index(
            drop=True
        )
    )

    balance.to_csv(
        OUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Concept counts]"
    )
    print(
        balance[
            "concept_local_name"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if "context_ref" in balance.columns:
        print(
            "\n[Contexts containing balance facts]"
        )

        print(
            balance.groupby(
                "context_ref",
                dropna=False,
            )
            .agg(
                fact_count=(
                    "concept_local_name",
                    "size",
                ),
                concepts=(
                    "concept_local_name",
                    lambda s:
                    " | ".join(
                        sorted(
                            set(
                                s.astype(str)
                            )
                        )
                    ),
                ),
            )
            .sort_values(
                "fact_count",
                ascending=False,
            )
            .head(30)
            .to_string()
        )

    exact = balance.loc[
        balance[
            "concept_local_name"
        ].isin(
            [
                "Assets",
                "Liabilities",
                "Equity",
                "EquityAttributableToOwnersOfParent",
                "NoncontrollingInterestsInConsolidatedEntity",
                "NoncontrollingInterests",
            ]
        )
    ].copy()

    print(
        "\n[Core balance facts]"
    )

    if exact.empty:
        print(
            "No exact core concepts."
        )
    else:
        print(
            exact.to_string(
                index=False
            )
        )

    print(
        "\n[Same-context equation check]"
    )

    value_col = None

    for candidate in [
        "numeric_value",
        "value",
    ]:
        if candidate in exact.columns:
            value_col = candidate
            break

    if (
        value_col is not None
        and "context_ref" in exact.columns
    ):
        temp = exact.copy()

        temp[
            value_col
        ] = pd.to_numeric(
            temp[
                value_col
            ],
            errors="coerce",
        )

        pivot = (
            temp.pivot_table(
                index="context_ref",
                columns="concept_local_name",
                values=value_col,
                aggfunc="first",
            )
        )

        if all(
            c in pivot.columns
            for c in [
                "Assets",
                "Liabilities",
                "Equity",
            ]
        ):
            pivot[
                "equation_gap"
            ] = (
                pivot[
                    "Assets"
                ]
                - pivot[
                    "Liabilities"
                ]
                - pivot[
                    "Equity"
                ]
            )

            pivot[
                "gap_pct_assets"
            ] = (
                pivot[
                    "equation_gap"
                ]
                .abs()
                / pivot[
                    "Assets"
                ]
                .abs()
                * 100
            )

        print(
            pivot.to_string()
        )

    print(
        f"\nSaved: {OUT_PATH}"
    )

    print(
        "\n확인 포인트:"
        "\n1) 같은 context에 Assets/Liabilities/Equity 중복 fact가 있는지"
        "\n2) decimals/scale이 서로 다른지"
        "\n3) EquityAttributableToOwnersOfParent 또는 NCI가 별도로 존재하는지"
        "\n4) 다른 consolidated context에서는 방정식이 맞는지"
    )


if __name__ == "__main__":
    main()
