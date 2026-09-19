from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B6B. Current-Period Column Resolver Diagnostic
#
# 목적
# ------------------------------------------------------------
# 명백한 CFS 문맥인데도 C4에서
#   - no_current_value
#   - ambiguous_column
# 이 나온 대표 receipt를 소수 샘플링하여,
#
# 같은 table_index / row_index의
#   1) C1 raw candidate 전체 필드
#   2) C4 candidate 전체 필드
# 를 함께 출력한다.
#
# 이 단계는 진단만 수행한다.
# resolver 수정 / production merge 없음.
#
# 실행:
# python scripts\05a5h6b6b_dart_current_period_column_diagnostic.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

FAMILY_EVIDENCE = (
    INTERIM
    / "dart_noncorrected_nodata_c4_missing_family_evidence.csv"
)

RECEIPT_SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_c4_receipt_evidence_summary.csv"
)

SOURCE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.parquet"
)

OUTPUT = (
    INTERIM
    / "dart_noncorrected_h6b6b_current_period_column_diagnostic.csv"
)

# Known decisive example from H6B5.
FORCE_RECEIPTS = [
    "20210310000259",  # 한화손해보험 FY2020
]


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module load failed: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rcept_str(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def print_record(title: str, record: pd.Series):
    print("\n" + title)
    print("-" * 120)

    for key, value in record.items():
        text = str(value)

        if len(text) > 600:
            text = text[:600] + " ...[truncated]"

        print(f"{key}: {text}")


def main():

    for p in [
        FAMILY_EVIDENCE,
        RECEIPT_SUMMARY,
        SOURCE_MANIFEST,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    c1 = load_module(
        "h6b6b_c1",
        SCRIPTS_DIR / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    c4 = load_module(
        "h6b6b_c4",
        SCRIPTS_DIR / "05a5c4_dart_conservative_document_selector.py",
    )

    fam = pd.read_csv(
        FAMILY_EVIDENCE,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    receipt_summary = pd.read_csv(
        RECEIPT_SUMMARY,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    for df in [
        fam,
        receipt_summary,
        manifest,
    ]:
        df["rcept_no"] = rcept_str(
            df["rcept_no"]
        )

    path_map = (
        manifest[
            [
                "rcept_no",
                "selected_path",
            ]
        ]
        .drop_duplicates(
            "rcept_no"
        )
        .set_index(
            "rcept_no"
        )[
            "selected_path"
        ]
        .to_dict()
    )

    meta_map = {
        str(r["rcept_no"]): r
        for _, r
        in receipt_summary.iterrows()
    }

    # --------------------------------------------------------
    # Deterministic sample:
    #   no-current-value 3 receipts
    #   ambiguous-column 3 receipts
    # plus forced decisive example
    # --------------------------------------------------------

    target_receipts = list(
        FORCE_RECEIPTS
    )

    for cls, n in [
        (
            "c4_rows_but_no_current_value",
            3,
        ),
        (
            "c4_column_ambiguous",
            3,
        ),
    ]:

        sub = fam.loc[
            fam[
                "c4_evidence_class"
            ].eq(
                cls
            )
        ].copy()

        # Prefer rows where CFS hint was already found,
        # because these isolate period-column logic from basis logic.
        if (
            "selected_cfs_hint_rows"
            in sub.columns
        ):
            preferred = sub.loc[
                pd.to_numeric(
                    sub[
                        "selected_cfs_hint_rows"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
            ]

            if not preferred.empty:
                sub = preferred

        receipts = (
            sub.sort_values(
                [
                    "stock_code",
                    "canonical_period_key",
                    "rcept_no",
                ]
            )[
                "rcept_no"
            ]
            .astype(str)
            .drop_duplicates()
            .head(n)
            .tolist()
        )

        target_receipts.extend(
            receipts
        )

    target_receipts = list(
        dict.fromkeys(
            target_receipts
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B6B CURRENT-PERIOD COLUMN RESOLVER DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        f"\nSample receipts: "
        f"{len(target_receipts):,}"
    )

    output_rows = []

    for idx, receipt in enumerate(
        target_receipts,
        start=1,
    ):

        meta = meta_map.get(
            receipt
        )

        selected_path = path_map.get(
            receipt
        )

        if (
            meta is None
            or not selected_path
        ):
            print(
                f"\n[{idx}] {receipt} | metadata/path missing"
            )
            continue

        zip_path = Path(
            str(
                selected_path
            )
        )

        print(
            "\n"
            + "=" * 120
        )

        print(
            f"[{idx}/{len(target_receipts)}] "
            f"{meta.get('stock_code')} | "
            f"{meta.get('corp_name')} | "
            f"{meta.get('canonical_period_key')} | "
            f"{receipt}"
        )

        c1.RAW_ROOT = (
            zip_path.parents[
                2
            ]
        )

        raw = pd.DataFrame(
            c1.extract_document_candidates(
                zip_path
            )
        )

        if raw.empty:
            print(
                "C1 raw empty"
            )
            continue

        raw[
            "stock_code"
        ] = str(
            meta.get(
                "stock_code",
                "",
            )
        ).zfill(
            6
        )

        raw[
            "period_key"
        ] = meta.get(
            "canonical_period_key"
        )

        raw[
            "rcept_no"
        ] = receipt

        raw[
            "rcept_dt"
        ] = pd.to_datetime(
            meta.get(
                "rcept_dt"
            ),
            errors="coerce",
        )

        c4df = pd.DataFrame(
            c4.build_account_candidates(
                raw
            )
        )

        target_families = fam.loc[
            fam[
                "rcept_no"
            ].eq(
                receipt
            )
            & fam[
                "c4_evidence_class"
            ].isin(
                [
                    "c4_rows_but_no_current_value",
                    "c4_column_ambiguous",
                ]
            ),
            [
                "missing_metric",
                "account_family",
                "c4_evidence_class",
            ],
        ].drop_duplicates()

        if target_families.empty:
            print(
                "No target family rows"
            )
            continue

        print(
            "\n[Target families]"
        )

        print(
            target_families.to_string(
                index=False
            )
        )

        for _, target_family in (
            target_families.iterrows()
        ):

            family = str(
                target_family[
                    "account_family"
                ]
            )

            problem_class = str(
                target_family[
                    "c4_evidence_class"
                ]
            )

            c4_problem = c4df.loc[
                c4df[
                    "account_family"
                ]
                .astype(str)
                .eq(
                    family
                )
                & c4df[
                    "column_status"
                ]
                .astype(str)
                .isin(
                    [
                        "no_current_value",
                        "ambiguous_column",
                    ]
                )
            ].copy()

            # Prefer rows that visibly look like financial statements.
            statement_mask = (
                c4_problem[
                    "heading_context"
                ]
                .astype(
                    "string"
                )
                .fillna("")
                .str.contains(
                    r"연\s*결|"
                    r"재\s*무\s*상\s*태\s*표|"
                    r"포\s*괄\s*손\s*익\s*계\s*산\s*서|"
                    r"손\s*익\s*계\s*산\s*서",
                    regex=True,
                )
            )

            if statement_mask.any():
                c4_problem = (
                    c4_problem.loc[
                        statement_mask
                    ]
                )

            c4_problem = (
                c4_problem.head(
                    3
                )
            )

            for _, c4row in (
                c4_problem.iterrows()
            ):

                table_index = (
                    c4row.get(
                        "table_index"
                    )
                )

                row_index = (
                    c4row.get(
                        "row_index"
                    )
                )

                print_record(
                    (
                        f"[C4 PROBLEM] "
                        f"family={family} | "
                        f"class={problem_class} | "
                        f"table={table_index} | "
                        f"row={row_index}"
                    ),
                    c4row,
                )

                raw_match = raw.copy()

                if (
                    "table_index"
                    in raw_match.columns
                    and pd.notna(
                        table_index
                    )
                ):
                    raw_match = (
                        raw_match.loc[
                            pd.to_numeric(
                                raw_match[
                                    "table_index"
                                ],
                                errors="coerce",
                            )
                            .eq(
                                pd.to_numeric(
                                    table_index,
                                    errors="coerce",
                                )
                            )
                        ]
                    )

                if (
                    "row_index"
                    in raw_match.columns
                    and pd.notna(
                        row_index
                    )
                ):
                    exact_raw = (
                        raw_match.loc[
                            pd.to_numeric(
                                raw_match[
                                    "row_index"
                                ],
                                errors="coerce",
                            )
                            .eq(
                                pd.to_numeric(
                                    row_index,
                                    errors="coerce",
                                )
                            )
                        ]
                    )

                else:
                    exact_raw = (
                        pd.DataFrame()
                    )

                if (
                    exact_raw.empty
                    and not raw_match.empty
                ):
                    # If exact row id is unavailable, still print nearby
                    # candidate rows from the same table.
                    exact_raw = (
                        raw_match.head(
                            5
                        )
                    )

                for raw_idx, (_, rawrow) in enumerate(
                    exact_raw.head(
                        5
                    ).iterrows(),
                    start=1,
                ):

                    print_record(
                        (
                            f"[C1 RAW {raw_idx}] "
                            f"table={table_index} | "
                            f"row={row_index}"
                        ),
                        rawrow,
                    )

                    out = {
                        "stock_code":
                        meta.get(
                            "stock_code"
                        ),

                        "corp_name":
                        meta.get(
                            "corp_name"
                        ),

                        "canonical_period_key":
                        meta.get(
                            "canonical_period_key"
                        ),

                        "rcept_no":
                        receipt,

                        "problem_class":
                        problem_class,

                        "account_family":
                        family,

                        "table_index":
                        table_index,

                        "row_index":
                        row_index,

                        "c4_selected_column":
                        c4row.get(
                            "selected_column"
                        ),

                        "c4_heading_context":
                        c4row.get(
                            "heading_context"
                        ),

                        "c4_primary_row_label":
                        c4row.get(
                            "primary_row_label"
                        ),

                        "c4_context_reasons":
                        c4row.get(
                            "context_reasons"
                        ),
                    }

                    # Preserve every C1 field as raw_* for offline inspection.
                    for col, value in (
                        rawrow.items()
                    ):
                        out[
                            f"raw_{col}"
                        ] = value

                    output_rows.append(
                        out
                    )

    out_df = pd.DataFrame(
        output_rows
    )

    out_df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "H6B6B SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nDiagnostic rows: "
        f"{len(out_df):,}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    print(
        "\n판단 목표:"
        "\n- C1 raw에 실제 column/header/value 배열이 어떤 필드로 보존되는지 확인"
        "\n- 당기와 전기 헤더가 중복/다중행인지 확인"
        "\n- current period regex가 보고서 period_key와 어떻게 불일치하는지 확인"
        "\n- 문제 원인 확인 후에만 C4 resolver를 수정"
    )


if __name__ == "__main__":
    main()
