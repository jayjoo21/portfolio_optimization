from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7F. Receipt × Metric Selector V2 Candidate Audit
#
# 목적
# ------------------------------------------------------------
# H6B7E까지 통과한 cached candidate rows를
# receipt × metric 단위로 압축하기 전 마지막 진단.
#
# 하는 일:
# 1) 모든 strong period candidate에 non-main-context guard 적용
# 2) table 단위 Basis(CFS/OFS) 전파
# 3) receipt 단위 strict basis policy 결정
#       CFS가 있으면 CFS only
#       CFS가 없고 exact receipt에 no-CFS affirmative evidence가 있으면
#       OFS fallback eligible
#       그 외는 basis unresolved
# 4) metric별 unique numeric value 개수 집계
# 5) assets = liabilities + equity balance coherent 조합 검사
#
# IMPORTANT
# ------------------------------------------------------------
# - 아직 최종 값을 production에 채택하지 않음
# - multiple values를 임의 점수로 하나 고르지 않음
# - ZIP 재파싱 없음
# - CFS/OFS를 한 receipt 안에서 섞지 않음
#
# 실행:
# python scripts\05a5h6b7f_dart_selector_v2_candidate_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_column_consistent.parquet"
)

CANDIDATE_OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_candidate_audit.parquet"
)

METRIC_OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_metric_summary.csv"
)

RECEIPT_OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_receipt_summary.csv"
)

BALANCE_OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_balance_coherence.csv"
)


CORE = [
    "assets",
    "liabilities",
    "equity",
    "revenue",
    "operating_income",
    "net_income",
]


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def compact(value) -> str:
    return re.sub(
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        clean(value).lower(),
    )


# ============================================================
# Context guard for ALL candidates
# ============================================================

BLOCK_PATTERNS = [
    "최대주주",
    "관계기업요약재무정보",
    "관계기업의요약재무정보",
    "공동기업요약재무정보",
    "공동기업의요약재무정보",
    "종속기업요약재무정보",
    "종속기업의요약재무정보",
    "주요종속기업요약재무정보",
    "주요종속기업의요약재무정보",
    "관계기업투자",
    "종속기업투자",
    "주당이익",
    "주당손익",
    "이익잉여금처분계산서",
    "배당에관한사항",
    "최근결산기재무현황",
    "최근결산기재무정보",
    "국내증권업수익성추이",
    "시장점유율",
]


def non_main_context(context) -> bool:
    text = compact(context)
    return any(
        phrase in text
        for phrase in BLOCK_PATTERNS
    )


BALANCE_TITLES = [
    "재무상태표",
    "연결재무상태표",
    "별도재무상태표",
]

INCOME_TITLES = [
    "손익계산서",
    "포괄손익계산서",
    "연결손익계산서",
    "연결포괄손익계산서",
    "별도손익계산서",
    "별도포괄손익계산서",
]


def main_statement_match(
    family,
    context,
) -> bool:

    text = compact(context)

    if family in {
        "assets",
        "liabilities",
        "equity",
    }:
        titles = BALANCE_TITLES

    elif family in {
        "revenue",
        "operating_income",
        "net_income",
    }:
        titles = INCOME_TITLES

    else:
        return False

    return any(
        title in text
        for title in titles
    )


# ============================================================
# Basis propagation
# ============================================================

def resolve_table_basis(
    values: pd.Series,
) -> str:

    bases = set(
        values
        .dropna()
        .astype(str)
        .tolist()
    )

    bases.discard("")
    bases.discard("UNKNOWN")

    if not bases:
        return "UNKNOWN"

    # Any explicit conflicting basis in the same table -> conservative MIXED.
    if "MIXED" in bases:
        return "MIXED"

    has_cfs = "CFS" in bases

    has_ofs = bool(
        bases.intersection(
            {
                "OFS",
                "OFS_ONLY_NO_CFS",
            }
        )
    )

    if has_cfs and has_ofs:
        return "MIXED"

    if has_cfs:
        return "CFS"

    if "OFS_ONLY_NO_CFS" in bases:
        return "OFS_ONLY_NO_CFS"

    if "OFS" in bases:
        return "OFS"

    return "UNKNOWN"


def basis_family(value) -> str:
    if value == "CFS":
        return "CFS"

    if value in {
        "OFS",
        "OFS_ONLY_NO_CFS",
    }:
        return "OFS"

    return value


# ============================================================
# Balance QA
# ============================================================

def balance_pass(
    assets: float,
    liabilities: float,
    equity: float,
) -> bool:

    gap = (
        assets
        - liabilities
        - equity
    )

    abs_gap = abs(
        gap
    )

    denom = max(
        abs(assets),
        abs(liabilities)
        + abs(equity),
        1.0,
    )

    rel_gap = (
        abs_gap
        / denom
    )

    return (
        abs_gap
        <= 2_000_000
        or rel_gap
        <= 1e-6
    )


def balance_gap(
    assets: float,
    liabilities: float,
    equity: float,
):
    gap = (
        assets
        - liabilities
        - equity
    )

    abs_gap = abs(
        gap
    )

    denom = max(
        abs(assets),
        abs(liabilities)
        + abs(equity),
        1.0,
    )

    return (
        gap,
        abs_gap,
        abs_gap / denom,
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_parquet(
        INPUT
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7F RECEIPT × METRIC SELECTOR V2 CANDIDATE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # 1. Candidate numeric value
    # --------------------------------------------------------

    if "period_v2_value_krw" in df.columns:
        df[
            "selector_value_krw"
        ] = pd.to_numeric(
            df[
                "period_v2_value_krw"
            ],
            errors="coerce",
        )

    else:
        numeric = pd.to_numeric(
            df.get(
                "period_v2_numeric"
            ),
            errors="coerce",
        )

        unit = pd.to_numeric(
            df.get(
                "unit_multiplier",
                1.0,
            ),
            errors="coerce",
        ).fillna(
            1.0
        )

        df[
            "selector_value_krw"
        ] = numeric * unit

    # --------------------------------------------------------
    # 2. Strong-period candidate
    # --------------------------------------------------------

    df[
        "selector_period_ok"
    ] = (
        df[
            "period_final_status"
        ]
        .astype(str)
        .isin(
            {
                "selected",
                "selected_context_column_confirmed",
            }
        )
    )

    df[
        "selector_non_main_context"
    ] = df[
        "heading_context"
    ].map(
        non_main_context
    )

    df[
        "selector_main_statement"
    ] = [
        main_statement_match(
            family,
            context,
        )
        for family, context
        in zip(
            df[
                "account_family"
            ],
            df[
                "heading_context"
            ],
        )
    ]

    # AUTO-eligible is intentionally strict.
    df[
        "selector_row_eligible"
    ] = (
        df[
            "selector_period_ok"
        ]
        & ~df[
            "selector_non_main_context"
        ]
        & df[
            "selector_main_statement"
        ]
        & df[
            "account_family"
        ].isin(
            CORE
        )
        & df[
            "selector_value_krw"
        ].notna()
    )

    # --------------------------------------------------------
    # 3. Table basis propagation
    # --------------------------------------------------------

    table_basis = (
        df.groupby(
            [
                "rcept_no",
                "table_index",
            ],
            dropna=False,
        )[
            "basis_v2"
        ]
        .agg(
            resolve_table_basis
        )
        .rename(
            "selector_table_basis"
        )
        .reset_index()
    )

    df = df.merge(
        table_basis,
        on=[
            "rcept_no",
            "table_index",
        ],
        how="left",
        validate="many_to_one",
    )

    df[
        "selector_basis_family"
    ] = df[
        "selector_table_basis"
    ].map(
        basis_family
    )

    # --------------------------------------------------------
    # 4. Receipt-level strict basis policy
    # --------------------------------------------------------

    receipt_records = []

    for rcept_no, group in (
        df.groupby(
            "rcept_no",
            sort=False,
        )
    ):

        eligible = group.loc[
            group[
                "selector_row_eligible"
            ]
        ]

        cfs_exists = (
            eligible[
                "selector_table_basis"
            ]
            .eq(
                "CFS"
            )
            .any()
        )

        # Exact receipt affirmative no-CFS evidence can occur even
        # outside a currently eligible metric row, so inspect all rows.
        no_cfs_affirmed = (
            group[
                "basis_v2"
            ]
            .eq(
                "OFS_ONLY_NO_CFS"
            )
            .any()
            or group[
                "selector_table_basis"
            ]
            .eq(
                "OFS_ONLY_NO_CFS"
            )
            .any()
        )

        ofs_exists = (
            eligible[
                "selector_basis_family"
            ]
            .eq(
                "OFS"
            )
            .any()
        )

        if cfs_exists:
            policy = "CFS_ONLY"

        elif (
            no_cfs_affirmed
            and ofs_exists
        ):
            policy = (
                "OFS_FALLBACK_EXPLICIT_NO_CFS"
            )

        elif ofs_exists:
            policy = (
                "OFS_PRESENT_BUT_NO_FALLBACK_PROOF"
            )

        else:
            policy = (
                "BASIS_UNRESOLVED"
            )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "selector_receipt_basis_policy":
                policy,

                "selector_receipt_cfs_exists":
                cfs_exists,

                "selector_receipt_no_cfs_affirmed":
                no_cfs_affirmed,

                "selector_receipt_ofs_exists":
                ofs_exists,
            }
        )

    receipt_policy = pd.DataFrame(
        receipt_records
    )

    df = df.merge(
        receipt_policy,
        on="rcept_no",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 5. Apply receipt basis policy
    # --------------------------------------------------------

    df[
        "selector_basis_eligible"
    ] = False

    mask_cfs = (
        df[
            "selector_receipt_basis_policy"
        ].eq(
            "CFS_ONLY"
        )
        & df[
            "selector_table_basis"
        ].eq(
            "CFS"
        )
    )

    mask_ofs = (
        df[
            "selector_receipt_basis_policy"
        ].eq(
            "OFS_FALLBACK_EXPLICIT_NO_CFS"
        )
        & df[
            "selector_basis_family"
        ].eq(
            "OFS"
        )
    )

    df.loc[
        mask_cfs | mask_ofs,
        "selector_basis_eligible",
    ] = True

    df[
        "selector_auto_pool"
    ] = (
        df[
            "selector_row_eligible"
        ]
        & df[
            "selector_basis_eligible"
        ]
    )

    # Save row-level audit.
    df.to_parquet(
        CANDIDATE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Receipt × metric unique-value summary
    # --------------------------------------------------------

    metric_records = []

    auto = df.loc[
        df[
            "selector_auto_pool"
        ]
    ].copy()

    for (
        rcept_no,
        family,
    ), group in auto.groupby(
        [
            "rcept_no",
            "account_family",
        ],
        sort=False,
    ):

        values = (
            pd.to_numeric(
                group[
                    "selector_value_krw"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )

        values = sorted(
            values
        )

        if len(values) == 1:
            status = (
                "UNIQUE_VALUE"
            )

        elif len(values) > 1:
            status = (
                "MULTIPLE_VALUES"
            )

        else:
            status = (
                "NO_VALUE"
            )

        metric_records.append(
            {
                "rcept_no":
                rcept_no,

                "account_family":
                family,

                "candidate_row_count":
                len(group),

                "unique_value_count":
                len(values),

                "metric_candidate_status":
                status,

                "unique_values_krw":
                "|".join(
                    f"{x:.10g}"
                    for x in values
                ),
            }
        )

    metric_summary = pd.DataFrame(
        metric_records
    )

    # Add missing receipt × core families explicitly.
    receipts = (
        df[
            "rcept_no"
        ]
        .drop_duplicates()
        .tolist()
    )

    grid = pd.MultiIndex.from_product(
        [
            receipts,
            CORE,
        ],
        names=[
            "rcept_no",
            "account_family",
        ],
    ).to_frame(
        index=False
    )

    metric_summary = grid.merge(
        metric_summary,
        on=[
            "rcept_no",
            "account_family",
        ],
        how="left",
    )

    metric_summary[
        "candidate_row_count"
    ] = metric_summary[
        "candidate_row_count"
    ].fillna(
        0
    ).astype(
        int
    )

    metric_summary[
        "unique_value_count"
    ] = metric_summary[
        "unique_value_count"
    ].fillna(
        0
    ).astype(
        int
    )

    metric_summary[
        "metric_candidate_status"
    ] = metric_summary[
        "metric_candidate_status"
    ].fillna(
        "NO_AUTO_CANDIDATE"
    )

    metric_summary[
        "unique_values_krw"
    ] = metric_summary[
        "unique_values_krw"
    ].fillna(
        ""
    )

    metric_summary.to_csv(
        METRIC_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 7. Balance coherent combination audit
    # --------------------------------------------------------

    balance_records = []

    def values_for(
        receipt_metric: pd.DataFrame,
        family: str,
    ):
        rows = receipt_metric.loc[
            receipt_metric[
                "account_family"
            ].eq(
                family
            )
        ]

        if rows.empty:
            return []

        text = rows.iloc[
            0
        ][
            "unique_values_krw"
        ]

        if not text:
            return []

        out = []

        for piece in str(
            text
        ).split(
            "|"
        ):
            try:
                out.append(
                    float(
                        piece
                    )
                )
            except Exception:
                pass

        return list(
            dict.fromkeys(
                out
            )
        )

    for rcept_no, group in (
        metric_summary.groupby(
            "rcept_no",
            sort=False,
        )
    ):

        assets_values = values_for(
            group,
            "assets",
        )

        liabilities_values = values_for(
            group,
            "liabilities",
        )

        equity_values = values_for(
            group,
            "equity",
        )

        # Protect against pathological candidate explosion.
        cap_exceeded = any(
            len(values) > 15
            for values in [
                assets_values,
                liabilities_values,
                equity_values,
            ]
        )

        coherent = []

        if (
            not cap_exceeded
            and assets_values
            and liabilities_values
            and equity_values
        ):
            for a, l, e in itertools.product(
                assets_values,
                liabilities_values,
                equity_values,
            ):
                if balance_pass(
                    a,
                    l,
                    e,
                ):
                    gap, abs_gap, rel_gap = (
                        balance_gap(
                            a,
                            l,
                            e,
                        )
                    )

                    coherent.append(
                        (
                            a,
                            l,
                            e,
                            gap,
                            abs_gap,
                            rel_gap,
                        )
                    )

        if cap_exceeded:
            status = (
                "CANDIDATE_CAP_EXCEEDED"
            )

        elif (
            not assets_values
            or not liabilities_values
            or not equity_values
        ):
            status = (
                "BALANCE_METRIC_MISSING"
            )

        elif len(
            coherent
        ) == 0:
            status = (
                "NO_COHERENT_COMBINATION"
            )

        elif len(
            coherent
        ) == 1:
            status = (
                "UNIQUE_COHERENT_COMBINATION"
            )

        else:
            status = (
                "MULTIPLE_COHERENT_COMBINATIONS"
            )

        chosen = (
            coherent[
                0
            ]
            if len(
                coherent
            ) == 1
            else (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        )

        balance_records.append(
            {
                "rcept_no":
                rcept_no,

                "assets_unique_count":
                len(
                    assets_values
                ),

                "liabilities_unique_count":
                len(
                    liabilities_values
                ),

                "equity_unique_count":
                len(
                    equity_values
                ),

                "balance_coherent_combo_count":
                len(
                    coherent
                ),

                "balance_candidate_status":
                status,

                "balance_selected_assets":
                chosen[
                    0
                ],

                "balance_selected_liabilities":
                chosen[
                    1
                ],

                "balance_selected_equity":
                chosen[
                    2
                ],

                "balance_gap":
                chosen[
                    3
                ],

                "balance_abs_gap":
                chosen[
                    4
                ],

                "balance_rel_gap":
                chosen[
                    5
                ],
            }
        )

    balance_summary = pd.DataFrame(
        balance_records
    )

    balance_summary.to_csv(
        BALANCE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 8. Receipt summary
    # --------------------------------------------------------

    receipt_summary = (
        receipt_policy.copy()
    )

    metric_counts = (
        metric_summary.groupby(
            "rcept_no"
        )
        .agg(
            auto_metric_count=(
                "unique_value_count",
                lambda s:
                int(
                    (
                        s > 0
                    ).sum()
                ),
            ),

            unique_metric_count=(
                "metric_candidate_status",
                lambda s:
                int(
                    (
                        s
                        == "UNIQUE_VALUE"
                    ).sum()
                ),
            ),

            ambiguous_metric_count=(
                "metric_candidate_status",
                lambda s:
                int(
                    (
                        s
                        == "MULTIPLE_VALUES"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    receipt_summary = (
        receipt_summary.merge(
            metric_counts,
            on="rcept_no",
            how="left",
            validate="one_to_one",
        )
        .merge(
            balance_summary[
                [
                    "rcept_no",
                    "balance_candidate_status",
                    "balance_coherent_combo_count",
                ]
            ],
            on="rcept_no",
            how="left",
            validate="one_to_one",
        )
    )

    receipt_summary[
        "strict_core6_unique"
    ] = receipt_summary[
        "unique_metric_count"
    ].eq(
        6
    )

    receipt_summary.to_csv(
        RECEIPT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Receipt basis policy]"
    )

    print(
        receipt_summary[
            "selector_receipt_basis_policy"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Metric candidate status]"
    )

    print(
        metric_summary[
            "metric_candidate_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Unique value count distribution]"
    )

    print(
        metric_summary[
            "unique_value_count"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .head(
            20
        )
        .to_string()
    )

    print(
        "\n[Balance candidate status]"
    )

    print(
        balance_summary[
            "balance_candidate_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Receipt metric coverage]"
    )

    print(
        receipt_summary[
            [
                "auto_metric_count",
                "unique_metric_count",
                "ambiguous_metric_count",
            ]
        ]
        .value_counts()
        .head(
            25
        )
        .to_string()
    )

    print(
        "\n[Strict core6 unique receipts]"
    )

    print(
        int(
            receipt_summary[
                "strict_core6_unique"
            ].sum()
        )
    )

    # Ambiguity sample
    ambiguous = (
        metric_summary.loc[
            metric_summary[
                "metric_candidate_status"
            ].eq(
                "MULTIPLE_VALUES"
            )
        ]
    )

    print(
        "\n[Multiple-value metric sample]"
    )

    if ambiguous.empty:
        print(
            "None"
        )
    else:
        print(
            ambiguous.head(
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
        f"- Candidate audit : "
        f"{CANDIDATE_OUTPUT}"
    )

    print(
        f"- Metric summary  : "
        f"{METRIC_OUTPUT}"
    )

    print(
        f"- Receipt summary : "
        f"{RECEIPT_OUTPUT}"
    )

    print(
        f"- Balance audit   : "
        f"{BALANCE_OUTPUT}"
    )

    print(
        "\n판단 원칙:"
        "\n- CFS candidate가 있으면 receipt 전체에서 CFS만 사용"
        "\n- exact receipt에 no-CFS affirmative evidence가 있을 때만 OFS fallback 허용"
        "\n- CFS/OFS 혼합 금지"
        "\n- multiple values는 아직 임의 선택 금지"
        "\n- balance equation으로 coherent combination만 별도 표시"
    )


if __name__ == "__main__":
    main()
