from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H5E. Final Non-Corrected Balance Reconciliation
#
# 목적
# ------------------------------------------------------------
# H5C/H5D 결과를 이용해 hard balance failure 4건을 최종 처리한다.
#
# 규칙
# ------------------------------------------------------------
# 1) H5C에서 unique_coherent_consolidated_set
#    → same-receipt document의 연결 balance 3종을 통째로 override
#
# 2) H5D XBRL에서
#      SeparateMember = balance PASS
#      ConsolidatedMember = balance FAIL
#    → OFS 값을 CFS 대신 쓰지 않는다.
#    → assets/liabilities/equity_total 3개를 모두 missing 처리
#
# 3) 나머지 애매한 케이스
#    → 자동 수정 금지 + RuntimeError
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_pit_receipt_wide_final.parquet/csv
# dart_noncorrected_balance_final_reconciliation_log.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h5e_dart_noncorrected_balance_final_reconciliation.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

BASE = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
)

H5C_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_balance_candidate_set_resolution.csv"
)

H5D_SETS = (
    INTERIM
    / "dart_noncorrected_ambiguous_xbrl_balance_sets.csv"
)

H5D_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_ambiguous_xbrl_balance_resolution.csv"
)

OUT_PARQUET = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_final.parquet"
)

OUT_CSV = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_final.csv"
)

OUT_LOG = (
    INTERIM
    / "dart_noncorrected_balance_final_reconciliation_log.csv"
)

ABS_TOL = 2_000_000
REL_TOL = 1e-6

CORE_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]


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


def balance_qa(
    assets,
    liabilities,
    equity,
):

    if any(
        pd.isna(x)
        for x in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return (
            "missing",
            np.nan,
            np.nan,
        )

    gap = (
        float(assets)
        - float(liabilities)
        - float(equity)
    )

    rel = (
        abs(gap)
        / abs(float(assets))
        if float(assets) != 0
        else np.nan
    )

    if gap == 0:
        qa = "exact_pass"

    elif (
        abs(gap) <= ABS_TOL
        or (
            pd.notna(rel)
            and rel <= REL_TOL
        )
    ):
        qa = "rounding_pass"

    else:
        qa = "fail"

    return (
        qa,
        gap,
        rel,
    )


def xbrl_basis(
    context_dimensions,
) -> str:

    text = str(
        context_dimensions
    ).lower()

    # IMPORTANT:
    # Do not inspect the axis name itself because
    # "ConsolidatedAndSeparateFinancialStatementsAxis"
    # contains both words.
    #
    # The member after "=" determines the actual basis.
    if "separatemember" in text:
        return "OFS"

    if "consolidatedmember" in text:
        return "CFS"

    return "UNKNOWN"


def main():

    for path in [
        BASE,
        H5C_RESOLUTION,
        H5D_SETS,
        H5D_RESOLUTION,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    final = pd.read_parquet(
        BASE
    )

    h5c = pd.read_csv(
        H5C_RESOLUTION,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    h5d_sets = pd.read_csv(
        H5D_SETS,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    h5d_resolution = pd.read_csv(
        H5D_RESOLUTION,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    for df in [
        final,
        h5c,
        h5d_sets,
        h5d_resolution,
    ]:
        df[
            "rcept_no"
        ] = receipt_string(
            df[
                "rcept_no"
            ]
        )

    if final[
        "rcept_no"
    ].duplicated().any():
        raise RuntimeError(
            "BASE final wide에 duplicate rcept_no가 있습니다."
        )

    out = final.copy()
    audit = []
    unresolved = []

    print(
        "\n"
        + "=" * 100
    )

    print(
        "05A5-H5E FINAL NON-CORRECTED BALANCE RECONCILIATION"
    )

    print(
        "=" * 100
    )

    hard_before = out.loc[
        out[
            "balance_qa_class"
        ].eq(
            "fail"
        )
    ].copy()

    print(
        f"\nHard balance failures before: "
        f"{len(hard_before):,}"
    )

    # --------------------------------------------------------
    # A) H5C unique coherent consolidated document sets
    # --------------------------------------------------------

    doc_good = h5c.loc[
        h5c[
            "resolution"
        ].eq(
            "unique_coherent_consolidated_set"
        )
    ].copy()

    for _, row in doc_good.iterrows():

        receipt = str(
            row[
                "rcept_no"
            ]
        )

        mask = out[
            "rcept_no"
        ].eq(
            receipt
        )

        if not mask.any():
            unresolved.append(
                (
                    receipt,
                    "document_override_receipt_missing",
                )
            )
            continue

        before = out.loc[
            mask,
            [
                "assets",
                "liabilities",
                "equity_total",
            ],
        ].iloc[
            0
        ]

        assets = pd.to_numeric(
            row.get(
                "recommended_assets"
            ),
            errors="coerce",
        )

        liabilities = pd.to_numeric(
            row.get(
                "recommended_liabilities"
            ),
            errors="coerce",
        )

        equity = pd.to_numeric(
            row.get(
                "recommended_equity"
            ),
            errors="coerce",
        )

        qa, gap, rel = balance_qa(
            assets,
            liabilities,
            equity,
        )

        if qa not in {
            "exact_pass",
            "rounding_pass",
        }:
            unresolved.append(
                (
                    receipt,
                    f"H5C recommended set not passing: {qa}",
                )
            )
            continue

        out.loc[
            mask,
            "assets",
        ] = assets

        out.loc[
            mask,
            "liabilities",
        ] = liabilities

        out.loc[
            mask,
            "equity_total",
        ] = equity

        out.loc[
            mask,
            "assets_source",
        ] = (
            "same_receipt_document_consolidated_override"
        )

        out.loc[
            mask,
            "liabilities_source",
        ] = (
            "same_receipt_document_consolidated_override"
        )

        out.loc[
            mask,
            "equity_total_source",
        ] = (
            "same_receipt_document_consolidated_override"
        )

        out.loc[
            mask,
            "balance_resolution",
        ] = (
            "document_unique_coherent_consolidated_set"
        )

        audit.append(
            {
                "rcept_no":
                receipt,

                "stock_code":
                row.get(
                    "stock_code"
                ),

                "corp_name":
                row.get(
                    "corp_name"
                ),

                "canonical_period_key":
                row.get(
                    "canonical_period_key"
                ),

                "action":
                "override_with_same_receipt_document_cfs",

                "old_assets":
                before[
                    "assets"
                ],

                "old_liabilities":
                before[
                    "liabilities"
                ],

                "old_equity_total":
                before[
                    "equity_total"
                ],

                "new_assets":
                assets,

                "new_liabilities":
                liabilities,

                "new_equity_total":
                equity,

                "new_balance_qa":
                qa,

                "reason":
                "H5C unique coherent same-table/same-column consolidated balance set",
            }
        )

    # --------------------------------------------------------
    # B) H5D: only OFS passes; CFS exists and fails
    # --------------------------------------------------------

    h5d_sets[
        "xbrl_basis"
    ] = h5d_sets[
        "context_dimensions"
    ].map(
        xbrl_basis
    )

    ambiguous_receipts = (
        h5d_resolution[
            "rcept_no"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for receipt in ambiguous_receipts:

        mask = out[
            "rcept_no"
        ].eq(
            receipt
        )

        if not mask.any():
            unresolved.append(
                (
                    receipt,
                    "xbrl_receipt_missing",
                )
            )
            continue

        sets = h5d_sets.loc[
            h5d_sets[
                "rcept_no"
            ].eq(
                receipt
            )
        ].copy()

        if sets.empty:
            unresolved.append(
                (
                    receipt,
                    "no_h5d_sets",
                )
            )
            continue

        cfs_sets = sets.loc[
            sets[
                "xbrl_basis"
            ].eq(
                "CFS"
            )
        ]

        ofs_sets = sets.loc[
            sets[
                "xbrl_basis"
            ].eq(
                "OFS"
            )
        ]

        cfs_pass = cfs_sets[
            "balance_qa"
        ].isin(
            [
                "exact_pass",
                "rounding_pass",
            ]
        ).any()

        ofs_pass = ofs_sets[
            "balance_qa"
        ].isin(
            [
                "exact_pass",
                "rounding_pass",
            ]
        ).any()

        cfs_exists = (
            not cfs_sets.empty
        )

        if (
            cfs_exists
            and not cfs_pass
            and ofs_pass
        ):

            before = out.loc[
                mask,
                [
                    "assets",
                    "liabilities",
                    "equity_total",
                ],
            ].iloc[
                0
            ]

            out.loc[
                mask,
                [
                    "assets",
                    "liabilities",
                    "equity_total",
                ],
            ] = np.nan

            for metric in [
                "assets",
                "liabilities",
                "equity_total",
            ]:
                source_col = (
                    f"{metric}_source"
                )

                out.loc[
                    mask,
                    source_col,
                ] = (
                    "missing_cfs_balance_anomaly_same_receipt_xbrl"
                )

            out.loc[
                mask,
                "balance_resolution",
            ] = (
                "cfs_fails_same_receipt_xbrl_ofs_passes_do_not_mix_basis"
            )

            row = out.loc[
                mask
            ].iloc[
                0
            ]

            audit.append(
                {
                    "rcept_no":
                    receipt,

                    "stock_code":
                    row.get(
                        "stock_code"
                    ),

                    "corp_name":
                    row.get(
                        "corp_name"
                    ),

                    "canonical_period_key":
                    row.get(
                        "canonical_period_key"
                    ),

                    "action":
                    "set_cfs_balance_triplet_missing",

                    "old_assets":
                    before[
                        "assets"
                    ],

                    "old_liabilities":
                    before[
                        "liabilities"
                    ],

                    "old_equity_total":
                    before[
                        "equity_total"
                    ],

                    "new_assets":
                    np.nan,

                    "new_liabilities":
                    np.nan,

                    "new_equity_total":
                    np.nan,

                    "new_balance_qa":
                    "missing",

                    "reason":
                    (
                        "Same-receipt XBRL explicitly shows "
                        "SeparateMember balance PASS but "
                        "ConsolidatedMember balance FAIL; "
                        "OFS not substituted for CFS."
                    ),
                }
            )

        else:
            unresolved.append(
                (
                    receipt,
                    (
                        "H5D not decisive: "
                        f"cfs_exists={cfs_exists}, "
                        f"cfs_pass={cfs_pass}, "
                        f"ofs_pass={ofs_pass}"
                    ),
                )
            )

    if unresolved:
        detail = "\n".join(
            f"{rcept} | {reason}"
            for rcept, reason in unresolved
        )

        raise RuntimeError(
            "자동 final reconciliation 할 수 없는 balance case가 남았습니다:\n"
            + detail
        )

    # --------------------------------------------------------
    # C) Recompute final QA / core coverage
    # --------------------------------------------------------

    out[
        "balance_gap"
    ] = (
        out[
            "assets"
        ]
        - out[
            "liabilities"
        ]
        - out[
            "equity_total"
        ]
    )

    out[
        "balance_relative_gap"
    ] = (
        out[
            "balance_gap"
        ]
        .abs()
        / out[
            "assets"
        ]
        .abs()
        .replace(
            0,
            np.nan,
        )
    )

    def classify_row(
        row,
    ):
        qa, _, _ = balance_qa(
            row.get(
                "assets"
            ),
            row.get(
                "liabilities"
            ),
            row.get(
                "equity_total"
            ),
        )

        return qa

    out[
        "balance_qa_class"
    ] = out.apply(
        classify_row,
        axis=1,
    )

    out[
        "balance_equation_pass"
    ] = out[
        "balance_qa_class"
    ].isin(
        [
            "exact_pass",
            "rounding_pass",
        ]
    )

    out[
        "core_metric_count_final"
    ] = (
        out[
            CORE_METRICS
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    out[
        "missing_core_metrics_final"
    ] = out.apply(
        lambda r:
        " | ".join(
            metric
            for metric
            in CORE_METRICS
            if pd.isna(
                r[
                    metric
                ]
            )
        ),
        axis=1,
    )

    out[
        "needs_source_level_fallback"
    ] = (
        out[
            "core_metric_count_final"
        ]
        .lt(
            len(
                CORE_METRICS
            )
        )
    )

    audit_df = pd.DataFrame(
        audit
    )

    out.to_parquet(
        OUT_PARQUET,
        index=False,
    )

    out.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    audit_df.to_csv(
        OUT_LOG,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # D) Summary
    # --------------------------------------------------------

    print(
        "\n[Actions]"
    )

    print(
        audit_df[
            "action"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Balance QA final]"
    )

    print(
        out[
            "balance_qa_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Core metric count final]"
    )

    print(
        out[
            "core_metric_count_final"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    hard_after = int(
        out[
            "balance_qa_class"
        ]
        .eq(
            "fail"
        )
        .sum()
    )

    print(
        f"\nHard balance failures after: "
        f"{hard_after:,}"
    )

    print(
        "\n[Reconciled balance cases]"
    )

    print(
        audit_df[
            [
                "stock_code",
                "corp_name",
                "canonical_period_key",
                "rcept_no",
                "action",
                "new_balance_qa",
                "reason",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nFinal : "
        f"{OUT_PARQUET}"
    )

    print(
        f"Log   : "
        f"{OUT_LOG}"
    )

    print(
        "\n다음 단계:"
        "\n- Hard balance failure가 0이면 balance QA 종료"
        "\n- 이후 remaining core=5 selector audit + 890 full-API no_data source-level fallback 범위를 정리"
        "\n- 마지막에 corrected reconciled + non-corrected final을 통합"
    )


if __name__ == "__main__":
    main()
