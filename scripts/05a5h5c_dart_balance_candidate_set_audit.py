from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H5C. Same-Receipt Document Balance Candidate-Set Audit
#
# 목적
# ------------------------------------------------------------
# H5B의 C4 candidate rows를 이용해 같은 receipt 안에서
# assets / liabilities / equity 후보를 "같은 표 + 같은 선택 열"로 묶고,
#
#     Assets = Liabilities + Equity
#
# 를 만족하는 coherent balance set이 존재하는지 확인한다.
#
# 중요:
# - API 호출 없음
# - document 재다운로드 없음
# - 자동 production override 없음
# - C4가 이미 current-period column으로 selected한 후보만 사용
# - 서로 다른 table/column을 섞지 않는 strict mode를 우선
#
# OUTPUT
# ------------------------------------------------------------
# dart_noncorrected_balance_candidate_sets_strict.csv
# dart_noncorrected_balance_candidate_sets_relaxed.csv
# dart_noncorrected_balance_candidate_set_resolution.csv
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h5c_dart_balance_candidate_set_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

C4_DEBUG = (
    INTERIM
    / "dart_noncorrected_balance_document_c4_candidates_debug.csv"
)

FINAL = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
)

OUT_STRICT = (
    INTERIM
    / "dart_noncorrected_balance_candidate_sets_strict.csv"
)

OUT_RELAXED = (
    INTERIM
    / "dart_noncorrected_balance_candidate_sets_relaxed.csv"
)

OUT_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_balance_candidate_set_resolution.csv"
)

ABS_TOL = 2_000_000
REL_TOL = 1e-6

FAMILIES = [
    "assets",
    "liabilities",
    "equity",
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


def parse_num(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {
        "",
        "-",
        "nan",
        "None",
        "<NA>",
    }:
        return np.nan

    text = text.replace(
        ",",
        "",
    )

    if (
        text.startswith("(")
        and text.endswith(")")
    ):
        text = (
            "-"
            + text[1:-1]
        )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def norm(value) -> str:

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("\u3000", "")
        .lower()
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
        abs(gap)
        <= ABS_TOL
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


def context_evidence(
    rows: list[
        pd.Series
    ],
):

    chunks = []

    for row in rows:
        for col in [
            "statement_type",
            "heading_context",
            "row_text",
            "context_reasons",
        ]:
            if col in row.index:
                chunks.append(
                    norm(
                        row.get(
                            col
                        )
                    )
                )

    text = " | ".join(
        chunks
    )

    consolidated_tokens = [
        "연결",
        "consolidated",
        "cfs",
    ]

    separate_tokens = [
        "별도",
        "개별",
        "separate",
        "ofs",
    ]

    consolidated_hits = [
        token
        for token in consolidated_tokens
        if token in text
    ]

    separate_hits = [
        token
        for token in separate_tokens
        if token in text
    ]

    score = (
        2
        * len(
            consolidated_hits
        )
        - 2
        * len(
            separate_hits
        )
    )

    return {
        "context_score":
        score,
        "consolidated_hits":
        " | ".join(
            consolidated_hits
        ),
        "separate_hits":
        " | ".join(
            separate_hits
        ),
    }


def make_record(
    target: pd.Series,
    arow: pd.Series,
    lrow: pd.Series,
    erow: pd.Series,
    mode: str,
):

    assets = parse_num(
        arow.get(
            "selected_value_krw"
        )
    )

    liabilities = parse_num(
        lrow.get(
            "selected_value_krw"
        )
    )

    equity = parse_num(
        erow.get(
            "selected_value_krw"
        )
    )

    qa, gap, rel = balance_qa(
        assets,
        liabilities,
        equity,
    )

    evidence = context_evidence(
        [
            arow,
            lrow,
            erow,
        ]
    )

    scores = []

    for row in [
        arow,
        lrow,
        erow,
    ]:
        score = parse_num(
            row.get(
                "total_score"
            )
        )

        if pd.notna(
            score
        ):
            scores.append(
                float(
                    score
                )
            )

    total_candidate_score = (
        sum(
            scores
        )
        if scores
        else np.nan
    )

    return {
        "mode":
        mode,

        "stock_code":
        str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        ),

        "corp_name":
        target.get(
            "corp_name"
        ),

        "canonical_period_key":
        target[
            "canonical_period_key"
        ],

        "rcept_no":
        str(
            target[
                "rcept_no"
            ]
        ),

        "table_index":
        arow.get(
            "table_index"
        ),

        "selected_column_assets":
        arow.get(
            "selected_column"
        ),

        "selected_column_liabilities":
        lrow.get(
            "selected_column"
        ),

        "selected_column_equity":
        erow.get(
            "selected_column"
        ),

        "assets":
        assets,

        "liabilities":
        liabilities,

        "equity":
        equity,

        "balance_gap":
        gap,

        "balance_relative_gap":
        rel,

        "balance_qa":
        qa,

        "context_score":
        evidence[
            "context_score"
        ],

        "consolidated_hits":
        evidence[
            "consolidated_hits"
        ],

        "separate_hits":
        evidence[
            "separate_hits"
        ],

        "candidate_score_sum":
        total_candidate_score,

        "assets_row":
        arow.get(
            "row_index"
        ),

        "liabilities_row":
        lrow.get(
            "row_index"
        ),

        "equity_row":
        erow.get(
            "row_index"
        ),

        "assets_label":
        arow.get(
            "primary_row_label"
        ),

        "liabilities_label":
        lrow.get(
            "primary_row_label"
        ),

        "equity_label":
        erow.get(
            "primary_row_label"
        ),

        "assets_heading":
        arow.get(
            "heading_context"
        ),

        "liabilities_heading":
        lrow.get(
            "heading_context"
        ),

        "equity_heading":
        erow.get(
            "heading_context"
        ),
    }


def enumerate_strict_sets(
    candidates: pd.DataFrame,
    target: pd.Series,
):

    records = []

    # Strict:
    # same table + same selected current-period column
    group_cols = [
        "table_index",
        "selected_column",
    ]

    for _, group in candidates.groupby(
        group_cols,
        dropna=False,
    ):

        fam = {
            family:
            group.loc[
                group[
                    "account_family"
                ].eq(
                    family
                )
            ]
            for family in FAMILIES
        }

        if any(
            x.empty
            for x in fam.values()
        ):
            continue

        # normally very small, but cap pathological tables
        arows = list(
            fam[
                "assets"
            ]
            .head(
                10
            )
            .iterrows()
        )

        lrows = list(
            fam[
                "liabilities"
            ]
            .head(
                10
            )
            .iterrows()
        )

        erows = list(
            fam[
                "equity"
            ]
            .head(
                10
            )
            .iterrows()
        )

        for (
            (_, arow),
            (_, lrow),
            (_, erow),
        ) in itertools.product(
            arows,
            lrows,
            erows,
        ):

            records.append(
                make_record(
                    target,
                    arow,
                    lrow,
                    erow,
                    "strict_same_table_same_column",
                )
            )

    return records


def enumerate_relaxed_sets(
    candidates: pd.DataFrame,
    target: pd.Series,
):

    records = []

    # Relaxed diagnostic only:
    # same table, but selected_column may differ.
    # NEVER auto-select from this mode.
    for _, group in candidates.groupby(
        [
            "table_index",
        ],
        dropna=False,
    ):

        fam = {
            family:
            group.loc[
                group[
                    "account_family"
                ].eq(
                    family
                )
            ]
            for family in FAMILIES
        }

        if any(
            x.empty
            for x in fam.values()
        ):
            continue

        arows = list(
            fam[
                "assets"
            ]
            .head(
                10
            )
            .iterrows()
        )

        lrows = list(
            fam[
                "liabilities"
            ]
            .head(
                10
            )
            .iterrows()
        )

        erows = list(
            fam[
                "equity"
            ]
            .head(
                10
            )
            .iterrows()
        )

        for (
            (_, arow),
            (_, lrow),
            (_, erow),
        ) in itertools.product(
            arows,
            lrows,
            erows,
        ):

            records.append(
                make_record(
                    target,
                    arow,
                    lrow,
                    erow,
                    "relaxed_same_table",
                )
            )

    return records


def unique_value_sets(
    df: pd.DataFrame,
):

    if df.empty:
        return df.copy()

    work = df.copy()

    # Values are integer KRW in practice.
    for col in [
        "assets",
        "liabilities",
        "equity",
    ]:
        work[
            col
        ] = pd.to_numeric(
            work[
                col
            ],
            errors="coerce",
        )

    work = (
        work.sort_values(
            [
                "context_score",
                "candidate_score_sum",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "assets",
                "liabilities",
                "equity",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return work


def resolve_receipt(
    strict_sets: pd.DataFrame,
):

    if strict_sets.empty:
        return {
            "resolution":
            "no_strict_candidate_set",

            "unique_strict_sets":
            0,

            "unique_pass_sets":
            0,

            "recommended_assets":
            np.nan,

            "recommended_liabilities":
            np.nan,

            "recommended_equity":
            np.nan,

            "recommended_qa":
            None,
        }

    unique = unique_value_sets(
        strict_sets
    )

    passed = unique.loc[
        unique[
            "balance_qa"
        ].isin(
            [
                "exact_pass",
                "rounding_pass",
            ]
        )
    ].copy()

    # Prefer sets with positive consolidated evidence and no
    # explicit separate evidence.
    passed_consolidated = passed.loc[
        passed[
            "context_score"
        ].gt(
            0
        )
        & passed[
            "separate_hits"
        ].fillna(
            ""
        )
        .eq(
            ""
        )
    ].copy()

    if len(
        passed_consolidated
    ) == 1:

        row = (
            passed_consolidated.iloc[
                0
            ]
        )

        resolution = (
            "unique_coherent_consolidated_set"
        )

    elif len(
        passed_consolidated
    ) > 1:

        return {
            "resolution":
            "multiple_coherent_consolidated_sets",

            "unique_strict_sets":
            len(
                unique
            ),

            "unique_pass_sets":
            len(
                passed
            ),

            "recommended_assets":
            np.nan,

            "recommended_liabilities":
            np.nan,

            "recommended_equity":
            np.nan,

            "recommended_qa":
            None,
        }

    elif len(
        passed
    ) == 1:

        row = passed.iloc[
            0
        ]

        resolution = (
            "unique_coherent_set_context_weak"
        )

    elif len(
        passed
    ) > 1:

        return {
            "resolution":
            "multiple_coherent_sets_context_ambiguous",

            "unique_strict_sets":
            len(
                unique
            ),

            "unique_pass_sets":
            len(
                passed
            ),

            "recommended_assets":
            np.nan,

            "recommended_liabilities":
            np.nan,

            "recommended_equity":
            np.nan,

            "recommended_qa":
            None,
        }

    else:

        return {
            "resolution":
            "no_coherent_strict_set",

            "unique_strict_sets":
            len(
                unique
            ),

            "unique_pass_sets":
            0,

            "recommended_assets":
            np.nan,

            "recommended_liabilities":
            np.nan,

            "recommended_equity":
            np.nan,

            "recommended_qa":
            None,
        }

    return {
        "resolution":
        resolution,

        "unique_strict_sets":
        len(
            unique
        ),

        "unique_pass_sets":
        len(
            passed
        ),

        "recommended_assets":
        row[
            "assets"
        ],

        "recommended_liabilities":
        row[
            "liabilities"
        ],

        "recommended_equity":
        row[
            "equity"
        ],

        "recommended_qa":
        row[
            "balance_qa"
        ],

        "recommended_table_index":
        row[
            "table_index"
        ],

        "recommended_selected_column":
        row[
            "selected_column_assets"
        ],

        "recommended_context_score":
        row[
            "context_score"
        ],

        "recommended_consolidated_hits":
        row[
            "consolidated_hits"
        ],

        "recommended_separate_hits":
        row[
            "separate_hits"
        ],
    }


def main():

    for path in [
        C4_DEBUG,
        FINAL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    c4 = pd.read_csv(
        C4_DEBUG,
        dtype={
            "_target_rcept_no":
            str,
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    final = pd.read_parquet(
        FINAL
    )

    final[
        "rcept_no"
    ] = receipt_string(
        final[
            "rcept_no"
        ]
    )

    target_col = (
        "_target_rcept_no"
        if "_target_rcept_no"
        in c4.columns
        else "rcept_no"
    )

    c4[
        target_col
    ] = receipt_string(
        c4[
            target_col
        ]
    )

    targets = final.loc[
        final[
            "balance_qa_class"
        ].eq(
            "fail"
        )
    ].copy()

    print(
        "\n"
        + "="
        * 110
    )

    print(
        "05A5-H5C SAME-RECEIPT DOCUMENT BALANCE CANDIDATE-SET AUDIT"
    )

    print(
        "="
        * 110
    )

    print(
        f"\nTargets: "
        f"{len(targets):,}"
    )

    strict_records = []
    relaxed_records = []
    resolutions = []

    for idx, (_, target) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        receipt = str(
            target[
                "rcept_no"
            ]
        )

        stock = str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        )

        candidates = c4.loc[
            c4[
                target_col
            ].eq(
                receipt
            )
            & c4[
                "account_family"
            ].isin(
                FAMILIES
            )
            & c4[
                "column_status"
            ].eq(
                "selected"
            )
        ].copy()

        candidates[
            "selected_value_krw"
        ] = pd.to_numeric(
            candidates[
                "selected_value_krw"
            ],
            errors="coerce",
        )

        candidates = candidates.loc[
            candidates[
                "selected_value_krw"
            ].notna()
        ].copy()

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{stock} | "
            f"{target['canonical_period_key']} | "
            f"{receipt}"
        )

        print(
            "  selected candidates: "
            f"{len(candidates):,}"
        )

        if not candidates.empty:

            family_counts = (
                candidates[
                    "account_family"
                ]
                .value_counts()
            )

            print(
                "  family counts:"
            )

            print(
                family_counts.to_string()
            )

        strict = enumerate_strict_sets(
            candidates,
            target,
        )

        relaxed = enumerate_relaxed_sets(
            candidates,
            target,
        )

        strict_df = pd.DataFrame(
            strict
        )

        relaxed_df = pd.DataFrame(
            relaxed
        )

        if not strict_df.empty:
            strict_records.append(
                strict_df
            )

        if not relaxed_df.empty:
            relaxed_records.append(
                relaxed_df
            )

        resolution = resolve_receipt(
            strict_df
        )

        resolution.update(
            {
                "stock_code":
                stock,

                "corp_name":
                target.get(
                    "corp_name"
                ),

                "canonical_period_key":
                target[
                    "canonical_period_key"
                ],

                "rcept_no":
                receipt,

                "api_assets":
                target.get(
                    "assets"
                ),

                "api_liabilities":
                target.get(
                    "liabilities"
                ),

                "api_equity":
                target.get(
                    "equity_total"
                ),

                "api_gap":
                target.get(
                    "balance_gap"
                ),

                "api_rel_gap":
                target.get(
                    "balance_relative_gap"
                ),
            }
        )

        resolutions.append(
            resolution
        )

        print(
            "  resolution: "
            f"{resolution['resolution']}"
        )

        if not strict_df.empty:

            strict_unique = (
                unique_value_sets(
                    strict_df
                )
            )

            strict_unique[
                "_qa_priority"
            ] = strict_unique[
                "balance_qa"
            ].map(
                {
                    "exact_pass":
                    0,
                    "rounding_pass":
                    1,
                    "fail":
                    2,
                    "missing":
                    3,
                }
            ).fillna(
                9
            )

            strict_unique = (
                strict_unique.sort_values(
                    [
                        "_qa_priority",
                        "context_score",
                        "balance_relative_gap",
                        "candidate_score_sum",
                    ],
                    ascending=[
                        True,
                        False,
                        True,
                        False,
                    ],
                )
            )

            print(
                "\n  [Top strict candidate sets]"
            )

            show_cols = [
                "table_index",
                "selected_column_assets",
                "assets",
                "liabilities",
                "equity",
                "balance_gap",
                "balance_relative_gap",
                "balance_qa",
                "context_score",
                "consolidated_hits",
                "separate_hits",
                "candidate_score_sum",
            ]

            print(
                strict_unique[
                    show_cols
                ]
                .head(
                    15
                )
                .to_string(
                    index=False
                )
            )

        else:
            print(
                "  No strict same-table/same-column set."
            )

    strict_all = (
        pd.concat(
            strict_records,
            ignore_index=True,
        )
        if strict_records
        else pd.DataFrame()
    )

    relaxed_all = (
        pd.concat(
            relaxed_records,
            ignore_index=True,
        )
        if relaxed_records
        else pd.DataFrame()
    )

    resolution_df = pd.DataFrame(
        resolutions
    )

    strict_all.to_csv(
        OUT_STRICT,
        index=False,
        encoding="utf-8-sig",
    )

    relaxed_all.to_csv(
        OUT_RELAXED,
        index=False,
        encoding="utf-8-sig",
    )

    resolution_df.to_csv(
        OUT_RESOLUTION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 110
    )

    print(
        "H5C SUMMARY"
    )

    print(
        "="
        * 110
    )

    print(
        "\n[Resolution]"
    )

    print(
        resolution_df[
            [
                "stock_code",
                "corp_name",
                "canonical_period_key",
                "rcept_no",
                "resolution",
                "unique_strict_sets",
                "unique_pass_sets",
                "recommended_assets",
                "recommended_liabilities",
                "recommended_equity",
                "recommended_qa",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nStrict sets : "
        f"{OUT_STRICT}"
    )

    print(
        f"Relaxed sets: "
        f"{OUT_RELAXED}"
    )

    print(
        f"Resolution  : "
        f"{OUT_RESOLUTION}"
    )

    print(
        "\n판정 원칙:"
        "\n- unique_coherent_consolidated_set"
        "\n  → 동일 표/동일 열의 유일한 연결 balance set; override 후보"
        "\n- unique_coherent_set_context_weak"
        "\n  → balance는 유일하게 맞지만 연결 근거 약함; 자동 override 금지, 추가 확인"
        "\n- multiple_*"
        "\n  → 여러 세트가 맞음; 자동 선택 금지"
        "\n- no_coherent_strict_set"
        "\n  → document 안에서도 일관된 현재기간 balance set 없음; missing 후보"
    )


if __name__ == "__main__":
    main()
