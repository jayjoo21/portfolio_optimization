from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6C2E. CORE5 EXISTING-FIVE BASIS PROVENANCE AUDIT
#
# 목적
# ------------------------------------------------------------
# core5 248 receipt에 이미 들어가 있는 "나머지 5개 metric"의
# CFS/OFS basis provenance를 확인한다.
#
# 동시에:
# - H6C2D missing strong candidate의 basis와 비교
# - source-level fallback plan / manifest에서 exact no-CFS proof로
#   쓸 수 있을 만한 필드/문구가 실제로 존재하는지 진단
#
# IMPORTANT
# ------------------------------------------------------------
# - AUDIT ONLY
# - merge 없음
# - 기존 5개가 CFS인데 missing 후보가 OFS이면 혼합 금지
# - 기존 5개가 OFS여도 explicit no-CFS proof 없이는 OFS fallback 확정 금지
# - label-only basis 추정은 production proof가 아님
#
# 실행:
# python scripts\05a5h6c2e_dart_core5_existing_basis_provenance_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

SELECTED = (
    INTERIM
    / "dart_noncorrected_full_fallback_selected.parquet"
)

MULTI_ROWS = (
    INTERIM
    / "dart_noncorrected_multi_account_rows.parquet"
)

STRONG_AUDIT = (
    INTERIM
    / "dart_noncorrected_core5_strong_candidate_basis_id_audit.csv"
)

SOURCE_PLAN = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_plan.parquet"
)

FALLBACK_MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

OUT_SELECTED_BASIS = (
    INTERIM
    / "dart_noncorrected_core5_existing_selected_metric_basis.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_core5_existing_basis_receipt_summary.csv"
)

OUT_MISSING_COMPARE = (
    INTERIM
    / "dart_noncorrected_core5_missing_candidate_vs_existing_basis.csv"
)

OUT_FALLBACK_EVIDENCE = (
    INTERIM
    / "dart_noncorrected_core5_fallback_proof_evidence.csv"
)


EXPECTED_CORE5 = 248


CORE6_METRICS = [
    "assets",
    "liabilities",
    "equity_total",
    "revenue_cumulative",
    "operating_income_cumulative",
    "net_income_total_cumulative",
]


# ============================================================
# Helpers
# ============================================================


def normalize_receipt(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def compact(value) -> str:
    return re.sub(
        r"[^0-9a-z가-힣]",
        "",
        clean(value).lower(),
    )


def norm_label(value) -> str:
    text = clean(value)

    text = re.sub(
        r"^\s*[\(\[]?\s*"
        r"(?:[ivxlcdmⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+|\d+)"
        r"\s*[\)\]\.\-:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return compact(text)


def parse_amount(value):
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    text = str(value).strip()

    if not text or text in {
        "-",
        "－",
        "—",
        "–",
    }:
        return np.nan

    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    if negative:
        text = text[1:-1]

    text = text.replace(",", "").replace(" ", "")

    text = re.sub(
        r"[^0-9eE+\-.]",
        "",
        text,
    )

    if not text:
        return np.nan

    try:
        number = float(text)
    except ValueError:
        return np.nan

    return -number if negative else number


def near_equal(a, b) -> bool:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return False

    if not (
        math.isfinite(a)
        and math.isfinite(b)
    ):
        return False

    diff = abs(a - b)

    denom = max(
        abs(a),
        abs(b),
        1.0,
    )

    return (
        diff <= 1_000_000
        or diff / denom <= 1e-6
    )


def normalize_fs_div(value) -> str:
    text = compact(value)

    if text in {
        "cfs",
        "연결",
        "연결재무제표",
    }:
        return "CFS"

    if text in {
        "ofs",
        "별도",
        "개별",
        "별도재무제표",
        "개별재무제표",
    }:
        return "OFS"

    return ""


def canonical_selected_metric(value) -> str:
    """
    Conservative exact mapping.
    parent-net-income is deliberately NOT mapped to total net income.
    """
    raw = clean(value)
    low = raw.lower()

    exact = {
        "assets":
        "assets",

        "liabilities":
        "liabilities",

        "equity":
        "equity_total",

        "equity_total":
        "equity_total",

        "revenue":
        "revenue_cumulative",

        "revenue_cumulative":
        "revenue_cumulative",

        "operating_income":
        "operating_income_cumulative",

        "operating_income_cumulative":
        "operating_income_cumulative",

        "net_income_total":
        "net_income_total_cumulative",

        "net_income_total_cumulative":
        "net_income_total_cumulative",
    }

    if low in exact:
        return exact[low]

    # Explicitly exclude owner-attributable metric.
    if "parent" in low or "attributable" in low:
        return ""

    return ""


# ============================================================
# Multi source index
# ============================================================


def build_multi_index(
    multi: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[str, float]]]:

    index = {}

    for _, row in multi.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        label = norm_label(
            row.get(
                "account_nm",
                "",
            )
        )

        basis = normalize_fs_div(
            row.get(
                "fs_div",
                "",
            )
        )

        amount = parse_amount(
            row.get(
                "thstrm_amount",
                np.nan,
            )
        )

        if (
            not label
            or not basis
            or pd.isna(amount)
        ):
            continue

        key = (
            rcept_no,
            label,
        )

        index.setdefault(
            key,
            [],
        ).append(
            (
                basis,
                float(amount),
            )
        )

    return index


def infer_selected_basis(
    rcept_no: str,
    account_nm,
    amount,
    multi_index,
) -> tuple[str, str]:

    label = norm_label(
        account_nm
    )

    if not label or pd.isna(amount):
        return (
            "UNKNOWN",
            "NO_LABEL_OR_AMOUNT",
        )

    matches = multi_index.get(
        (
            str(rcept_no),
            label,
        ),
        [],
    )

    exact_basis = sorted(
        set(
            basis
            for basis, candidate_amount
            in matches
            if near_equal(
                amount,
                candidate_amount,
            )
        )
    )

    if len(exact_basis) == 1:
        return (
            exact_basis[0],
            "SAME_LABEL_SAME_VALUE",
        )

    if len(exact_basis) > 1:
        return (
            "MIXED",
            "SAME_VALUE_IN_CFS_AND_OFS",
        )

    # label-only basis is diagnostic, not production proof.
    label_basis = sorted(
        set(
            basis
            for basis, _
            in matches
        )
    )

    if len(label_basis) == 1:
        return (
            f"{label_basis[0]}_LABEL_ONLY",
            "SAME_LABEL_BUT_VALUE_NOT_MATCHED",
        )

    if len(label_basis) > 1:
        return (
            "MIXED_LABEL_ONLY",
            "LABEL_EXISTS_IN_CFS_AND_OFS_NO_VALUE_MATCH",
        )

    return (
        "UNKNOWN",
        "NO_MULTI_MATCH",
    )


# ============================================================
# Explicit no-CFS evidence diagnostic
# ============================================================


PROOF_COLUMN_PATTERN = re.compile(
    r"(?:cfs|ofs|fs[_]?div|fallback|reason|status|source|"
    r"available|availability|no[_]?data|document|xbrl|api|"
    r"consolidat|연결|별도|개별|종속)",
    flags=re.IGNORECASE,
)


STRONG_NO_CFS_PATTERNS = [
    re.compile(
        r"\bofs[_\s-]*only[_\s-]*no[_\s-]*cfs\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bno[_\s-]*cfs\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"no\s+consolidated\s+(?:financial\s+statements?|subsidiar(?:y|ies))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"연결재무제표.{0,20}(?:없|미작성|작성하지|해당없)",
    ),
    re.compile(
        r"(?:연결대상\s*)?종속기업.{0,20}(?:없|존재하지|해당없)",
    ),
]


def candidate_proof_columns(
    frame: pd.DataFrame,
) -> list[str]:

    return [
        col
        for col in frame.columns
        if col != "rcept_no"
        and PROOF_COLUMN_PATTERN.search(
            str(col)
        )
    ]


def build_proof_text(
    row: pd.Series,
    cols: list[str],
) -> str:

    pieces = []

    for col in cols:

        value = row.get(
            col,
            np.nan,
        )

        if pd.isna(value):
            continue

        text = clean(
            value
        )

        if not text:
            continue

        pieces.append(
            f"{col}={text}"
        )

    return " || ".join(
        pieces
    )


def has_strong_no_cfs_phrase(text: str) -> bool:
    return any(
        pattern.search(
            text
        )
        for pattern in STRONG_NO_CFS_PATTERNS
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        CORE5,
        SELECTED,
        MULTI_ROWS,
        STRONG_AUDIT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    core5 = pd.read_parquet(
        CORE5
    )

    if len(core5) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 rows, got {len(core5)}."
        )

    selected = pd.read_parquet(
        SELECTED
    )

    multi = pd.read_parquet(
        MULTI_ROWS
    )

    strong_audit = pd.read_csv(
        STRONG_AUDIT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    for frame in [
        core5,
        selected,
        multi,
        strong_audit,
    ]:
        frame[
            "rcept_no"
        ] = normalize_receipt(
            frame[
                "rcept_no"
            ]
        )

    target_receipts = set(
        core5[
            "rcept_no"
        ]
    )

    selected = selected.loc[
        selected[
            "rcept_no"
        ].isin(
            target_receipts
        )
    ].copy()

    multi = multi.loc[
        multi[
            "rcept_no"
        ].isin(
            target_receipts
        )
    ].copy()

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2E CORE5 EXISTING-FIVE BASIS PROVENANCE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCore5 receipts: "
        f"{len(core5):,}"
    )

    # --------------------------------------------------------
    # Selected metric distribution first.
    # --------------------------------------------------------

    print(
        "\n[Raw selected metric distribution]"
    )

    if "metric" not in selected.columns:
        raise RuntimeError(
            "full_fallback_selected has no metric column."
        )

    print(
        selected[
            "metric"
        ]
        .value_counts(
            dropna=False
        )
        .head(
            30
        )
        .to_string()
    )

    selected[
        "canonical_metric"
    ] = selected[
        "metric"
    ].map(
        canonical_selected_metric
    )

    selected[
        "_amount_numeric"
    ] = selected[
        "thstrm_amount"
    ].map(
        parse_amount
    )

    core6_selected = selected.loc[
        selected[
            "canonical_metric"
        ].isin(
            CORE6_METRICS
        )
    ].copy()

    core6_selected = core6_selected.merge(
        core5[
            [
                "rcept_no",
                "stock_code",
                "corp_name",
                "period_key",
                "missing_metric",
            ]
            if "corp_name" in core5.columns
            else [
                "rcept_no",
                "stock_code",
                "period_key",
                "missing_metric",
            ]
        ],
        on="rcept_no",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_core5",
        ),
    )

    core6_selected[
        "is_missing_metric"
    ] = (
        core6_selected[
            "canonical_metric"
        ]
        .eq(
            core6_selected[
                "missing_metric"
            ]
        )
    )

    existing_five = core6_selected.loc[
        ~core6_selected[
            "is_missing_metric"
        ]
    ].copy()

    # --------------------------------------------------------
    # Infer selected basis from MULTI_ACCOUNT_ROWS.
    # --------------------------------------------------------

    multi_index = build_multi_index(
        multi
    )

    basis_info = [
        infer_selected_basis(
            row[
                "rcept_no"
            ],
            row.get(
                "account_nm",
                "",
            ),
            row[
                "_amount_numeric"
            ],
            multi_index,
        )
        for _, row
        in existing_five.iterrows()
    ]

    existing_five[
        "inferred_basis"
    ] = [
        item[
            0
        ]
        for item in basis_info
    ]

    existing_five[
        "basis_evidence"
    ] = [
        item[
            1
        ]
        for item in basis_info
    ]

    existing_five.to_csv(
        OUT_SELECTED_BASIS,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Existing-five inferred basis rows]"
    )

    print(
        pd.crosstab(
            existing_five[
                "canonical_metric"
            ],
            existing_five[
                "inferred_basis"
            ],
            dropna=False,
        )
        .to_string()
    )

    # --------------------------------------------------------
    # Receipt-level existing basis profile
    # --------------------------------------------------------

    selected_groups = {
        rcept_no:
        group
        for rcept_no, group
        in existing_five.groupby(
            "rcept_no",
            sort=False,
        )
    }

    receipt_records = []

    for _, row in core5.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        group = selected_groups.get(
            rcept_no,
            existing_five.iloc[
                0:0
            ],
        )

        per_metric_basis = {}

        for metric in CORE6_METRICS:

            if metric == row[
                "missing_metric"
            ]:
                continue

            metric_group = group.loc[
                group[
                    "canonical_metric"
                ].eq(
                    metric
                )
            ]

            basis_set = sorted(
                set(
                    metric_group[
                        "inferred_basis"
                    ].astype(str)
                )
            )

            per_metric_basis[
                metric
            ] = "|".join(
                basis_set
            )

        strong_basis_values = []

        for basis_text in (
            group[
                "inferred_basis"
            ].astype(str)
        ):

            if basis_text == "CFS":
                strong_basis_values.append(
                    "CFS"
                )

            elif basis_text == "OFS":
                strong_basis_values.append(
                    "OFS"
                )

            elif basis_text == "MIXED":
                strong_basis_values.extend(
                    [
                        "CFS",
                        "OFS",
                    ]
                )

        unique_strong = set(
            strong_basis_values
        )

        has_unknown = (
            group[
                "inferred_basis"
            ]
            .astype(str)
            .isin(
                [
                    "UNKNOWN",
                    "CFS_LABEL_ONLY",
                    "OFS_LABEL_ONLY",
                    "MIXED_LABEL_ONLY",
                ]
            )
            .any()
        )

        present_metric_count = (
            group[
                "canonical_metric"
            ]
            .nunique()
        )

        if (
            unique_strong
            == {
                "CFS"
            }
            and not has_unknown
        ):
            profile = (
                "EXISTING_FIVE_CFS_ONLY"
            )

        elif (
            unique_strong
            == {
                "OFS"
            }
            and not has_unknown
        ):
            profile = (
                "EXISTING_FIVE_OFS_ONLY"
            )

        elif unique_strong == {
            "CFS",
            "OFS",
        }:
            profile = (
                "EXISTING_FIVE_MIXED_CFS_OFS"
            )

        elif not group.empty:
            profile = (
                "EXISTING_FIVE_PARTIAL_OR_UNKNOWN"
            )

        else:
            profile = (
                "NO_SELECTED_EXISTING_FIVE_EVIDENCE"
            )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                row.get(
                    "stock_code",
                    "",
                ),

                "corp_name":
                row.get(
                    "corp_name",
                    "",
                ),

                "period_key":
                row.get(
                    "period_key",
                    "",
                ),

                "missing_metric":
                row[
                    "missing_metric"
                ],

                "existing_selected_rows":
                len(
                    group
                ),

                "existing_selected_metric_count":
                int(
                    present_metric_count
                ),

                "existing_basis_profile":
                profile,

                "assets_basis":
                per_metric_basis.get(
                    "assets",
                    "",
                ),

                "liabilities_basis":
                per_metric_basis.get(
                    "liabilities",
                    "",
                ),

                "equity_total_basis":
                per_metric_basis.get(
                    "equity_total",
                    "",
                ),

                "revenue_cumulative_basis":
                per_metric_basis.get(
                    "revenue_cumulative",
                    "",
                ),

                "operating_income_cumulative_basis":
                per_metric_basis.get(
                    "operating_income_cumulative",
                    "",
                ),

                "net_income_total_cumulative_basis":
                per_metric_basis.get(
                    "net_income_total_cumulative",
                    "",
                ),
            }
        )

    receipt_summary = pd.DataFrame(
        receipt_records
    )

    receipt_summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Existing-five receipt basis profile]"
    )

    print(
        receipt_summary[
            "existing_basis_profile"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Existing-five selected metric count]"
    )

    print(
        receipt_summary[
            "existing_selected_metric_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Compare H6C2D missing candidate basis to existing profile
    # --------------------------------------------------------

    strong_numeric = strong_audit.loc[
        strong_audit[
            "id_consistency"
        ].ne(
            "CONTRADICTION"
        )
        & pd.to_numeric(
            strong_audit[
                "_amount_numeric"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    candidate_basis_by_receipt = (
        strong_numeric.groupby(
            [
                "rcept_no",
                "missing_metric",
            ],
            dropna=False,
        )[
            "basis_v2"
        ]
        .agg(
            lambda s:
            "|".join(
                sorted(
                    set(
                        s.astype(str)
                    )
                )
            )
        )
        .reset_index(
            name="missing_candidate_basis"
        )
    )

    compare = receipt_summary.merge(
        candidate_basis_by_receipt,
        on=[
            "rcept_no",
            "missing_metric",
        ],
        how="left",
        validate="one_to_one",
    )

    compare[
        "missing_candidate_basis"
    ] = compare[
        "missing_candidate_basis"
    ].fillna(
        ""
    )

    def compare_status(
        row,
    ) -> str:

        existing = row[
            "existing_basis_profile"
        ]

        missing = row[
            "missing_candidate_basis"
        ]

        if not missing:
            return (
                "NO_NUMERIC_STRONG_MISSING_CANDIDATE"
            )

        if (
            existing
            == "EXISTING_FIVE_CFS_ONLY"
            and "OFS" in missing
            and "CFS" not in missing
        ):
            return (
                "BLOCK_WOULD_MIX_OFS_INTO_CFS"
            )

        if (
            existing
            == "EXISTING_FIVE_OFS_ONLY"
            and "OFS" in missing
            and "CFS" not in missing
        ):
            return (
                "OFS_MATCHES_EXISTING_BUT_NEEDS_EXPLICIT_NO_CFS_PROOF"
            )

        if (
            existing
            == "EXISTING_FIVE_CFS_ONLY"
            and "CFS" in missing
        ):
            return (
                "CFS_BASIS_COMPATIBLE_NEEDS_PERIOD_YTD"
            )

        if (
            existing
            == "EXISTING_FIVE_OFS_ONLY"
            and "CFS" in missing
        ):
            return (
                "REVIEW_EXISTING_OFS_BUT_MISSING_CFS"
            )

        return (
            "REVIEW_BASIS_PROFILE_INCOMPLETE_OR_MIXED"
        )

    compare[
        "basis_compatibility"
    ] = compare.apply(
        compare_status,
        axis=1,
    )

    compare.to_csv(
        OUT_MISSING_COMPARE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Missing candidate vs existing-five basis]"
    )

    print(
        pd.crosstab(
            compare[
                "missing_metric"
            ],
            compare[
                "basis_compatibility"
            ],
            dropna=False,
        )
        .to_string()
    )

    # --------------------------------------------------------
    # Source-plan / manifest no-CFS evidence discovery
    # --------------------------------------------------------

    fallback_frames = []

    for source_name, path in [
        (
            "SOURCE_LEVEL_FALLBACK_PLAN",
            SOURCE_PLAN,
        ),
        (
            "FULL_FALLBACK_MANIFEST",
            FALLBACK_MANIFEST,
        ),
    ]:

        if not path.exists():
            continue

        frame = pd.read_parquet(
            path
        )

        if "rcept_no" not in frame.columns:
            continue

        frame[
            "rcept_no"
        ] = normalize_receipt(
            frame[
                "rcept_no"
            ]
        )

        frame = frame.loc[
            frame[
                "rcept_no"
            ].isin(
                target_receipts
            )
        ].copy()

        proof_cols = candidate_proof_columns(
            frame
        )

        print(
            f"\n[{source_name} candidate proof columns]"
        )

        if not proof_cols:
            print(
                "None"
            )

        else:
            print(
                " | ".join(
                    proof_cols
                )
            )

            for col in proof_cols:

                values = (
                    frame[
                        col
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                if values.empty:
                    continue

                vc = values.value_counts().head(
                    12
                )

                print(
                    f"\n{col}:"
                )

                print(
                    vc.to_string()
                )

        if not proof_cols:
            frame[
                "_proof_text"
            ] = ""
        else:
            frame[
                "_proof_text"
            ] = frame.apply(
                lambda row:
                build_proof_text(
                    row,
                    proof_cols,
                ),
                axis=1,
            )

        frame[
            "_potential_explicit_no_cfs_phrase"
        ] = frame[
            "_proof_text"
        ].map(
            has_strong_no_cfs_phrase
        )

        temp_cols = [
            "rcept_no",
            "_proof_text",
            "_potential_explicit_no_cfs_phrase",
        ]

        temp = frame[
            temp_cols
        ].copy()

        temp[
            "_proof_source"
        ] = source_name

        fallback_frames.append(
            temp
        )

    if fallback_frames:

        proof = pd.concat(
            fallback_frames,
            ignore_index=True,
        )

    else:

        proof = pd.DataFrame(
            columns=[
                "rcept_no",
                "_proof_text",
                "_potential_explicit_no_cfs_phrase",
                "_proof_source",
            ]
        )

    proof = proof.merge(
        core5[
            [
                "rcept_no",
                "stock_code",
                "corp_name",
                "period_key",
                "missing_metric",
            ]
            if "corp_name" in core5.columns
            else [
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

    proof.to_csv(
        OUT_FALLBACK_EVIDENCE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Potential explicit no-CFS phrase hits]"
    )

    if proof.empty:
        print(
            "0"
        )
    else:
        print(
            int(
                proof[
                    "_potential_explicit_no_cfs_phrase"
                ].sum()
            )
        )

        hits = proof.loc[
            proof[
                "_potential_explicit_no_cfs_phrase"
            ]
        ]

        if not hits.empty:

            with pd.option_context(
                "display.max_colwidth",
                180,
                "display.width",
                360,
                "display.max_rows",
                40,
            ):

                print(
                    hits[
                        [
                            "_proof_source",
                            "stock_code",
                            "corp_name",
                            "period_key",
                            "rcept_no",
                            "missing_metric",
                            "_proof_text",
                        ]
                    ]
                    .head(
                        30
                    )
                    .to_string(
                        index=False
                    )
                )

    print(
        "\nOutputs:"
    )

    print(
        f"- Existing selected metric basis : "
        f"{OUT_SELECTED_BASIS}"
    )

    print(
        f"- Receipt basis summary          : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Missing-vs-existing comparison : "
        f"{OUT_MISSING_COMPARE}"
    )

    print(
        f"- Fallback proof evidence        : "
        f"{OUT_FALLBACK_EVIDENCE}"
    )

    print(
        "\n해석 원칙:"
        "\n- SAME_LABEL_SAME_VALUE basis만 강한 provenance로 봄"
        "\n- *_LABEL_ONLY는 진단용이며 production proof 아님"
        "\n- 기존 5개 CFS + missing OFS는 즉시 BLOCK"
        "\n- 기존 5개 OFS + missing OFS도 explicit no-CFS proof 없이는 자동채택 금지"
        "\n- no-CFS phrase hit는 '후보 evidence'일 뿐 다음 단계에서 exact receipt 문맥 확인 필요"
        "\n- 아직 값 merge 없음"
    )


if __name__ == "__main__":
    main()
