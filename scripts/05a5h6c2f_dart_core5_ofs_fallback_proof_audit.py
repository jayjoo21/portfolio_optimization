from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6C2F. OFS FALLBACK PROOF AUDIT
#
# 목적
# ------------------------------------------------------------
# H6C2E에서 확인된 receipt-level selected_fs_div:
#   CFS 227
#   OFS  21
#
# 중 OFS 21건만 exact-receipt 단위로 재검증한다.
#
# 확인:
# 1) source plan / fallback manifest의 basis 선택이 서로 일치하는가
# 2) MULTI_ACCOUNT_ROWS에 같은 receipt의 CFS row가 실제 존재하는가
# 3) CFS/OFS 각각 row 수 / numeric row 수
# 4) missing metric의 strong numeric 후보 basis
# 5) API-level exact receipt에서 CFS row 자체가 0인지
#
# IMPORTANT
# ------------------------------------------------------------
# - API-level "CFS row = 0"은 강한 진단 증거지만
#   문서 자체에서 CFS가 없다는 최종 production proof와 동일시하지 않는다.
# - CFS row가 존재하는데 selected_fs_div=OFS면 자동 fallback 금지.
# - merge 없음.
#
# 실행:
# python scripts\05a5h6c2f_dart_core5_ofs_fallback_proof_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

SOURCE_PLAN = (
    INTERIM
    / "dart_noncorrected_source_level_fallback_plan.parquet"
)

FALLBACK_MANIFEST = (
    INTERIM
    / "dart_noncorrected_full_fallback_manifest.parquet"
)

MULTI_ROWS = (
    INTERIM
    / "dart_noncorrected_multi_account_rows.parquet"
)

STRONG_AUDIT = (
    INTERIM
    / "dart_noncorrected_core5_strong_candidate_basis_id_audit.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_core5_ofs21_fallback_proof_audit.csv"
)

OUT_ROWS = (
    INTERIM
    / "dart_noncorrected_core5_ofs21_multi_account_rows.csv"
)

OUT_REVIEW = (
    INTERIM
    / "dart_noncorrected_core5_ofs21_review_queue.csv"
)


EXPECTED_CORE5 = 248
EXPECTED_OFS = 21


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

    text = (
        text.replace(",", "")
        .replace(" ", "")
    )

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


def get_basis_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for col in candidates:
        if col in frame.columns:
            return col

    return None


def metric_candidate_basis_summary(
    strong_audit: pd.DataFrame,
    rcept_no: str,
) -> tuple[str, int, int, int]:

    group = strong_audit.loc[
        strong_audit[
            "rcept_no"
        ].eq(
            rcept_no
        )
        & strong_audit[
            "id_consistency"
        ].ne(
            "CONTRADICTION"
        )
    ].copy()

    if group.empty:
        return (
            "",
            0,
            0,
            0,
        )

    numeric = pd.to_numeric(
        group[
            "_amount_numeric"
        ],
        errors="coerce",
    )

    group = group.loc[
        numeric.notna()
    ].copy()

    if group.empty:
        return (
            "",
            0,
            0,
            0,
        )

    bases = sorted(
        set(
            group[
                "basis_v2"
            ]
            .fillna("")
            .astype(str)
        )
    )

    return (
        "|".join(
            bases
        ),
        int(
            group[
                "basis_v2"
            ].eq(
                "CFS"
            ).sum()
        ),
        int(
            group[
                "basis_v2"
            ].eq(
                "OFS"
            ).sum()
        ),
        int(
            group[
                "basis_v2"
            ].isin(
                [
                    "UNKNOWN",
                    "MIXED",
                ]
            ).sum()
        ),
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        CORE5,
        SOURCE_PLAN,
        FALLBACK_MANIFEST,
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

    source_plan = pd.read_parquet(
        SOURCE_PLAN
    )

    manifest = pd.read_parquet(
        FALLBACK_MANIFEST
    )

    multi = pd.read_parquet(
        MULTI_ROWS
    )

    strong = pd.read_csv(
        STRONG_AUDIT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    if len(core5) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 rows, got {len(core5)}."
        )

    for frame in [
        core5,
        source_plan,
        manifest,
        multi,
        strong,
    ]:
        if "rcept_no" in frame.columns:
            frame[
                "rcept_no"
            ] = normalize_receipt(
                frame[
                    "rcept_no"
                ]
            )

    # --------------------------------------------------------
    # Resolve receipt-level selected basis columns.
    # --------------------------------------------------------

    plan_basis_col = get_basis_column(
        source_plan,
        [
            "selected_fs_div",
            "h3_chosen_fs_div",
        ],
    )

    manifest_basis_col = get_basis_column(
        manifest,
        [
            "chosen_fs_div",
            "selected_fs_div_before",
        ],
    )

    if plan_basis_col is None:
        raise RuntimeError(
            "No receipt-level basis column found in source plan."
        )

    if manifest_basis_col is None:
        raise RuntimeError(
            "No receipt-level basis column found in fallback manifest."
        )

    plan = source_plan[
        [
            col
            for col in [
                "rcept_no",
                "selected_fs_div",
                "h3_chosen_fs_div",
                "collection_status",
                "api_row_count",
                "h3_collection_status",
                "h3_api_status",
                "h3_api_message",
                "fallback_reason",
                "source_level_fallback_reason",
            ]
            if col in source_plan.columns
        ]
    ].drop_duplicates(
        subset=[
            "rcept_no",
        ]
    )

    mani = manifest[
        [
            col
            for col in [
                "rcept_no",
                "selected_fs_div_before",
                "chosen_fs_div",
                "collection_status",
                "api_status",
                "api_message",
                "api_row_count",
                "fallback_reason",
            ]
            if col in manifest.columns
        ]
    ].drop_duplicates(
        subset=[
            "rcept_no",
        ]
    )

    basis = core5.merge(
        plan,
        on="rcept_no",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_plan",
        ),
    )

    basis = basis.merge(
        mani,
        on="rcept_no",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_manifest",
        ),
    )

    selected_basis = basis[
        plan_basis_col
    ].map(
        normalize_fs_div
    )

    ofs21 = basis.loc[
        selected_basis.eq(
            "OFS"
        )
    ].copy()

    if len(ofs21) != EXPECTED_OFS:
        raise RuntimeError(
            f"Expected {EXPECTED_OFS} OFS receipts, found {len(ofs21)}."
        )

    ofs_receipts = set(
        ofs21[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2F CORE5 OFS FALLBACK PROOF AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCore5 receipts : "
        f"{len(core5):,}"
    )

    print(
        f"OFS receipts   : "
        f"{len(ofs21):,}"
    )

    print(
        "\n[OFS 21 by missing metric]"
    )

    print(
        ofs21[
            "missing_metric"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Multi-account rows for exact OFS receipts
    # --------------------------------------------------------

    multi21 = multi.loc[
        multi[
            "rcept_no"
        ].isin(
            ofs_receipts
        )
    ].copy()

    multi21[
        "_basis"
    ] = multi21[
        "fs_div"
    ].map(
        normalize_fs_div
    )

    multi21[
        "_amount_numeric"
    ] = multi21[
        "thstrm_amount"
    ].map(
        parse_amount
    )

    multi21.to_csv(
        OUT_ROWS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt audit
    # --------------------------------------------------------

    records = []

    for _, row in ofs21.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        m = multi21.loc[
            multi21[
                "rcept_no"
            ].eq(
                rcept_no
            )
        ]

        cfs = m.loc[
            m[
                "_basis"
            ].eq(
                "CFS"
            )
        ]

        ofs = m.loc[
            m[
                "_basis"
            ].eq(
                "OFS"
            )
        ]

        cfs_numeric = cfs.loc[
            cfs[
                "_amount_numeric"
            ].notna()
        ]

        ofs_numeric = ofs.loc[
            ofs[
                "_amount_numeric"
            ].notna()
        ]

        (
            missing_basis,
            strong_numeric_cfs_rows,
            strong_numeric_ofs_rows,
            strong_numeric_unknown_rows,
        ) = metric_candidate_basis_summary(
            strong,
            rcept_no,
        )

        selected_plan = normalize_fs_div(
            row.get(
                "selected_fs_div",
                row.get(
                    "h3_chosen_fs_div",
                    "",
                ),
            )
        )

        selected_h3 = normalize_fs_div(
            row.get(
                "h3_chosen_fs_div",
                "",
            )
        )

        manifest_before = normalize_fs_div(
            row.get(
                "selected_fs_div_before",
                "",
            )
        )

        manifest_chosen = normalize_fs_div(
            row.get(
                "chosen_fs_div",
                "",
            )
        )

        upstream_values = [
            value
            for value in [
                selected_plan,
                selected_h3,
                manifest_before,
                manifest_chosen,
            ]
            if value
        ]

        upstream_basis_consistent = (
            len(
                set(
                    upstream_values
                )
            )
            <= 1
        )

        if len(cfs) == 0:

            api_cfs_state = (
                "NO_CFS_ROWS_IN_MULTI_ACCOUNT_API"
            )

        elif len(cfs_numeric) == 0:

            api_cfs_state = (
                "CFS_ROWS_EXIST_BUT_NO_NUMERIC_AMOUNT"
            )

        else:

            api_cfs_state = (
                "CFS_NUMERIC_ROWS_EXIST"
            )

        if (
            api_cfs_state
            == "NO_CFS_ROWS_IN_MULTI_ACCOUNT_API"
            and upstream_basis_consistent
            and selected_plan
            == "OFS"
        ):

            fallback_audit = (
                "API_LEVEL_NO_CFS_EVIDENCE_ONLY"
            )

        elif (
            api_cfs_state
            != "NO_CFS_ROWS_IN_MULTI_ACCOUNT_API"
        ):

            fallback_audit = (
                "BLOCK_OR_REVIEW_CFS_ROWS_EXIST"
            )

        else:

            fallback_audit = (
                "REVIEW_UPSTREAM_BASIS_INCONSISTENT"
            )

        records.append(
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
                row.get(
                    "missing_metric",
                    "",
                ),

                "selected_fs_div":
                selected_plan,

                "h3_chosen_fs_div":
                selected_h3,

                "manifest_selected_fs_div_before":
                manifest_before,

                "manifest_chosen_fs_div":
                manifest_chosen,

                "upstream_basis_consistent":
                upstream_basis_consistent,

                "multi_total_rows":
                len(
                    m
                ),

                "multi_cfs_rows":
                len(
                    cfs
                ),

                "multi_ofs_rows":
                len(
                    ofs
                ),

                "multi_cfs_numeric_rows":
                len(
                    cfs_numeric
                ),

                "multi_ofs_numeric_rows":
                len(
                    ofs_numeric
                ),

                "api_cfs_state":
                api_cfs_state,

                "missing_strong_numeric_basis":
                missing_basis,

                "missing_strong_numeric_cfs_rows":
                strong_numeric_cfs_rows,

                "missing_strong_numeric_ofs_rows":
                strong_numeric_ofs_rows,

                "missing_strong_numeric_unknown_rows":
                strong_numeric_unknown_rows,

                "fallback_audit_status":
                fallback_audit,
            }
        )

    audit = pd.DataFrame(
        records
    )

    audit.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    review = audit.loc[
        ~audit[
            "fallback_audit_status"
        ].eq(
            "API_LEVEL_NO_CFS_EVIDENCE_ONLY"
        )
    ].copy()

    review.to_csv(
        OUT_REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Upstream basis consistency]"
    )

    print(
        audit[
            "upstream_basis_consistent"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Exact-receipt CFS state among OFS 21]"
    )

    print(
        audit[
            "api_cfs_state"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Fallback audit status]"
    )

    print(
        audit[
            "fallback_audit_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[OFS21 by missing metric x CFS state]"
    )

    print(
        pd.crosstab(
            audit[
                "missing_metric"
            ],
            audit[
                "api_cfs_state"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[OFS21 missing strong numeric candidate basis]"
    )

    print(
        pd.crosstab(
            audit[
                "missing_metric"
            ],
            audit[
                "missing_strong_numeric_basis"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[OFS21 detailed audit]"
    )

    with pd.option_context(
        "display.max_colwidth",
        120,
        "display.width",
        360,
        "display.max_rows",
        40,
    ):

        print(
            audit[
                [
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "missing_metric",
                    "selected_fs_div",
                    "manifest_chosen_fs_div",
                    "multi_cfs_rows",
                    "multi_ofs_rows",
                    "multi_cfs_numeric_rows",
                    "multi_ofs_numeric_rows",
                    "api_cfs_state",
                    "missing_strong_numeric_basis",
                    "fallback_audit_status",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[Potential next-step document-proof targets]"
    )

    proof_targets = audit.loc[
        audit[
            "fallback_audit_status"
        ].eq(
            "API_LEVEL_NO_CFS_EVIDENCE_ONLY"
        )
    ]

    print(
        f"{len(proof_targets):,}"
    )

    if not proof_targets.empty:

        print(
            proof_targets[
                [
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "missing_metric",
                    "missing_strong_numeric_basis",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Inspect any hidden cfs/ofs-related columns in upstream files
    # --------------------------------------------------------

    print(
        "\n[Additional source-plan columns containing CFS/OFS]"
    )

    extra_cols = [
        col
        for col in source_plan.columns
        if re.search(
            r"(?:cfs|ofs)",
            str(col),
            flags=re.IGNORECASE,
        )
    ]

    if not extra_cols:
        print(
            "None"
        )

    else:

        print(
            " | ".join(
                extra_cols
            )
        )

        source_ofs = source_plan.loc[
            source_plan[
                "rcept_no"
            ].isin(
                ofs_receipts
            ),
            [
                "rcept_no",
            ]
            + extra_cols,
        ].copy()

        with pd.option_context(
            "display.max_colwidth",
            100,
            "display.width",
            320,
            "display.max_rows",
            25,
        ):

            print(
                source_ofs.head(
                    21
                ).to_string(
                    index=False
                )
            )

    print(
        "\nOutputs:"
    )

    print(
        f"- OFS21 receipt audit : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- OFS21 raw rows      : "
        f"{OUT_ROWS}"
    )

    print(
        f"- OFS21 review queue  : "
        f"{OUT_REVIEW}"
    )

    print(
        "\n해석 원칙:"
        "\n- API_LEVEL_NO_CFS_EVIDENCE_ONLY = exact receipt의 multi-account API에서 CFS row가 0"
        "\n- 이것만으로 production OFS fallback을 확정하지 않음"
        "\n- CFS rows가 존재하면 OFS 자동 fallback 금지"
        "\n- 다음 단계에서 API-level no-CFS 대상만 document-level no-CFS proof 필요 여부 결정"
        "\n- merge 없음"
    )


if __name__ == "__main__":
    main()
