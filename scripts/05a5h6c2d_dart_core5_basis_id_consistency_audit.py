from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6C2D. CORE5 BASIS + LABEL/ACCOUNT-ID CONSISTENCY AUDIT
#
# 목적
# ------------------------------------------------------------
# H6C2C V2 candidate 중 STRONG_TOTAL 후보에 대해:
#
# 1) CFS/OFS basis를 확인
# 2) account_nm과 account_id가 같은 의미인지 확인
# 3) OFS-only 후보를 CFS metric에 섞지 않도록 차단
# 4) assets 후보는 기존 liabilities + equity와 balance coherence도 진단
# 5) flow metric은 여기서 값 채택하지 않고 YTD audit 대상으로만 넘김
#
# IMPORTANT
# ------------------------------------------------------------
# - AUDIT ONLY
# - merge 없음
# - CFS 우선
# - OFS는 exact receipt에서 no-CFS가 명시적으로 증명된 경우만 fallback 가능
# - UNKNOWN basis는 자동채택 금지
# - label이 맞아도 account_id가 다른 개념이면 CONTRADICTION
#
# 실행:
# python scripts\05a5h6c2d_dart_core5_basis_id_consistency_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

CORE5 = (
    INTERIM
    / "dart_noncorrected_core5_residual_248.parquet"
)

CANDIDATES = (
    INTERIM
    / "dart_noncorrected_core5_semantic_v2_candidate_rows.parquet"
)

MULTI_ROWS = (
    INTERIM
    / "dart_noncorrected_multi_account_rows.parquet"
)

VALIDATED_WIDE = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_validated.parquet"
)

SELECTED_MISSING = (
    INTERIM
    / "dart_noncorrected_core5_selected_but_wide_missing_audit.csv"
)

OUT_ROWS = (
    INTERIM
    / "dart_noncorrected_core5_strong_candidate_basis_id_audit.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_core5_basis_id_receipt_summary.csv"
)

OUT_REVIEW = (
    INTERIM
    / "dart_noncorrected_core5_basis_id_review_queue.csv"
)


EXPECTED_CORE5 = 248


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
        value = float(text)
    except ValueError:
        return np.nan

    return -value if negative else value


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


# ============================================================
# account_nm ↔ account_id consistency
# ============================================================


def id_consistency(
    missing_metric: str,
    account_nm,
    account_id,
) -> tuple[str, str]:

    label = norm_label(account_nm)
    aid = compact(account_id)

    # Empty / custom IDs: label evidence exists, but ID does not
    # independently confirm it.
    if (
        not aid
        or "표준계정코드미사용" in aid
        or aid in {
            "-",
            "na",
            "none",
        }
    ):
        return (
            "LABEL_ONLY_NO_STANDARD_ID",
            "no_standard_id_confirmation",
        )

    # --------------------------------------------------------
    # ASSETS
    # --------------------------------------------------------

    if missing_metric == "assets":

        if aid == "ifrsfullassets":
            return (
                "CONSISTENT",
                "ifrs_full_assets",
            )

        contradiction_markers = [
            "currentassets",
            "noncurrentassets",
            "cashandcashequivalents",
            "inventories",
            "financialassets",
            "deferredtaxassets",
            "propertyplantandequipment",
            "intangibleassets",
            "rightofuseassets",
        ]

        if any(
            marker in aid
            for marker in contradiction_markers
        ):
            return (
                "CONTRADICTION",
                "account_id_is_asset_component_not_total",
            )

        return (
            "REVIEW_UNKNOWN_ID",
            "unrecognized_assets_account_id",
        )

    # --------------------------------------------------------
    # OPERATING INCOME
    # --------------------------------------------------------

    if missing_metric == "operating_income_cumulative":

        allowed = {
            "dartoperatingincomeloss",
            "ifrsfullprofitlossfromoperatingactivities",
        }

        if aid in allowed:
            return (
                "CONSISTENT",
                "operating_income_id",
            )

        contradiction_markers = [
            "grossprofit",
            "profitlossbeforetax",
            "financeincome",
            "financecosts",
            "revenue",
        ]

        if any(
            marker in aid
            for marker in contradiction_markers
        ):
            return (
                "CONTRADICTION",
                "account_id_semantics_not_operating_income",
            )

        return (
            "REVIEW_UNKNOWN_ID",
            "unrecognized_operating_income_id",
        )

    # --------------------------------------------------------
    # NET INCOME TOTAL
    # --------------------------------------------------------

    if missing_metric == "net_income_total_cumulative":

        if aid == "ifrsfullprofitloss":
            return (
                "CONSISTENT",
                "ifrs_full_profit_loss_total",
            )

        contradiction_markers = [
            "beforetax",
            "continuingoperations",
            "discontinuedoperations",
            "attributabletoowners",
            "attributabletononcontrolling",
            "comprehensiveincome",
            "operating",
            "grossprofit",
        ]

        if any(
            marker in aid
            for marker in contradiction_markers
        ):
            return (
                "CONTRADICTION",
                "account_id_is_not_total_net_income",
            )

        return (
            "REVIEW_UNKNOWN_ID",
            "unrecognized_net_income_id",
        )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    if missing_metric == "revenue_cumulative":

        allowed = {
            "ifrsfullrevenue",
            "ifrsfullrevenuefromcontractswithcustomers",
        }

        if aid in allowed:
            return (
                "CONSISTENT",
                "ifrs_revenue_total",
            )

        contradiction_markers = [
            "interestincome",
            "feeincome",
            "commissionincome",
            "dividendincome",
            "otherincome",
            "financeincome",
            "gain",
            "insurancefinanceincome",
        ]

        if any(
            marker in aid
            for marker in contradiction_markers
        ):
            return (
                "CONTRADICTION",
                "account_id_is_revenue_component",
            )

        return (
            "REVIEW_UNKNOWN_ID",
            "unrecognized_revenue_id",
        )

    return (
        "REVIEW_UNKNOWN_ID",
        "unknown_metric",
    )


# ============================================================
# Infer basis for rows lacking fs_div using MULTI_ACCOUNT_ROWS
# ============================================================


def build_multi_match_index(
    multi: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[str, float]]]:

    result = {}

    for row in multi.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        label = norm_label(
            getattr(
                row,
                "account_nm",
                "",
            )
        )

        if not label:
            continue

        basis = normalize_fs_div(
            getattr(
                row,
                "fs_div",
                "",
            )
        )

        if not basis:
            continue

        amount = parse_amount(
            getattr(
                row,
                "thstrm_amount",
                np.nan,
            )
        )

        if pd.isna(amount):
            continue

        key = (
            rcept_no,
            label,
        )

        result.setdefault(
            key,
            [],
        ).append(
            (
                basis,
                float(amount),
            )
        )

    return result


def infer_basis(
    rcept_no: str,
    account_nm,
    amount,
    explicit_fs_div,
    multi_index,
) -> tuple[str, str]:

    explicit = normalize_fs_div(
        explicit_fs_div
    )

    if explicit:
        return (
            explicit,
            "EXPLICIT_FS_DIV",
        )

    label = norm_label(
        account_nm
    )

    if not label or pd.isna(amount):
        return (
            "UNKNOWN",
            "NO_BASIS_MATCH_EVIDENCE",
        )

    matches = multi_index.get(
        (
            str(rcept_no),
            label,
        ),
        [],
    )

    matched_basis = sorted(
        set(
            basis
            for basis, multi_amount
            in matches
            if near_equal(
                amount,
                multi_amount,
            )
        )
    )

    if len(matched_basis) == 1:
        return (
            matched_basis[0],
            "INFERRED_FROM_MULTI_SAME_LABEL_VALUE",
        )

    if len(matched_basis) > 1:
        return (
            "MIXED",
            "MATCHES_BOTH_CFS_OFS",
        )

    return (
        "UNKNOWN",
        "NO_MULTI_SAME_LABEL_VALUE_MATCH",
    )


# ============================================================
# Balance coherence helper
# ============================================================


def balance_gap(
    assets,
    liabilities,
    equity,
) -> tuple[float, float, str]:

    if any(
        pd.isna(x)
        for x in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return (
            np.nan,
            np.nan,
            "MISSING",
        )

    a = float(assets)
    l = float(liabilities)
    e = float(equity)

    gap = a - l - e

    rel = abs(gap) / max(
        abs(a),
        abs(l) + abs(e),
        1.0,
    )

    if abs(gap) == 0:
        status = "EXACT"

    elif (
        abs(gap) <= 2_000_000
        or rel <= 1e-6
    ):
        status = "ROUNDING"

    else:
        status = "HARD_FAIL"

    return (
        gap,
        rel,
        status,
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        CORE5,
        CANDIDATES,
        MULTI_ROWS,
        VALIDATED_WIDE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    core5 = pd.read_parquet(
        CORE5
    )

    if len(core5) != EXPECTED_CORE5:
        raise RuntimeError(
            f"Expected {EXPECTED_CORE5} core5 receipts, got {len(core5)}."
        )

    candidates = pd.read_parquet(
        CANDIDATES
    )

    multi = pd.read_parquet(
        MULTI_ROWS
    )

    wide = pd.read_parquet(
        VALIDATED_WIDE
    )

    for frame in [
        core5,
        candidates,
        multi,
        wide,
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
        "05A5-H6C2D CORE5 BASIS + LABEL/ACCOUNT-ID CONSISTENCY AUDIT"
    )
    print(
        "=" * 120
    )

    # --------------------------------------------------------
    # Strong-total candidate rows only
    # --------------------------------------------------------

    strong = candidates.loc[
        candidates[
            "_semantic_class_v2"
        ].eq(
            "STRONG_TOTAL"
        )
    ].copy()

    strong[
        "_amount_numeric"
    ] = strong[
        "_amount_numeric"
    ].map(
        parse_amount
    )

    multi_index = build_multi_match_index(
        multi
    )

    basis_info = [
        infer_basis(
            rcept_no,
            account_nm,
            amount,
            fs_div,
            multi_index,
        )
        for (
            rcept_no,
            account_nm,
            amount,
            fs_div,
        )
        in zip(
            strong[
                "rcept_no"
            ],
            strong[
                "account_nm"
            ],
            strong[
                "_amount_numeric"
            ],
            strong[
                "fs_div"
            ],
        )
    ]

    strong[
        "basis_v2"
    ] = [
        x[0]
        for x in basis_info
    ]

    strong[
        "basis_evidence"
    ] = [
        x[1]
        for x in basis_info
    ]

    id_info = [
        id_consistency(
            missing_metric,
            account_nm,
            account_id,
        )
        for (
            missing_metric,
            account_nm,
            account_id,
        )
        in zip(
            strong[
                "missing_metric"
            ],
            strong[
                "account_nm"
            ],
            strong[
                "account_id"
            ],
        )
    ]

    strong[
        "id_consistency"
    ] = [
        x[0]
        for x in id_info
    ]

    strong[
        "id_consistency_reason"
    ] = [
        x[1]
        for x in id_info
    ]

    # --------------------------------------------------------
    # Asset balance-coherence diagnostic
    # --------------------------------------------------------

    wide_lookup = wide.set_index(
        "rcept_no",
        drop=False,
    )

    asset_gap = []
    asset_rel = []
    asset_bal_status = []

    # NOTE:
    # itertuples() renames columns that are not valid namedtuple field names.
    # In particular, a column beginning with "_" such as _amount_numeric
    # cannot safely be accessed as row._amount_numeric. Use iterrows() here
    # so the original DataFrame column name is preserved exactly.
    for _, row in strong.iterrows():

        if (
            row["missing_metric"]
            != "assets"
        ):
            asset_gap.append(
                np.nan
            )
            asset_rel.append(
                np.nan
            )
            asset_bal_status.append(
                ""
            )
            continue

        rcept_no = str(
            row["rcept_no"]
        )

        wide_row = wide_lookup.loc[
            rcept_no
        ]

        (
            gap,
            rel,
            status,
        ) = balance_gap(
            row["_amount_numeric"],
            wide_row.get(
                "liabilities",
                np.nan,
            ),
            wide_row.get(
                "equity_total",
                np.nan,
            ),
        )

        asset_gap.append(
            gap
        )
        asset_rel.append(
            rel
        )
        asset_bal_status.append(
            status
        )

    strong[
        "asset_balance_gap"
    ] = asset_gap

    strong[
        "asset_balance_rel_gap"
    ] = asset_rel

    strong[
        "asset_balance_status"
    ] = asset_bal_status

    strong.to_csv(
        OUT_ROWS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt-level decision
    # --------------------------------------------------------

    strong_groups = {
        rcept_no:
        group
        for rcept_no, group
        in strong.groupby(
            "rcept_no",
            sort=False,
        )
    }

    receipt_records = []

    for row in core5.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        metric = str(
            row.missing_metric
        )

        group = strong_groups.get(
            rcept_no,
            strong.iloc[
                0:0
            ],
        )

        contradiction = group.loc[
            group[
                "id_consistency"
            ].eq(
                "CONTRADICTION"
            )
        ]

        semantically_usable = group.loc[
            ~group[
                "id_consistency"
            ].eq(
                "CONTRADICTION"
            )
        ]

        cfs = semantically_usable.loc[
            semantically_usable[
                "basis_v2"
            ].eq(
                "CFS"
            )
        ]

        ofs = semantically_usable.loc[
            semantically_usable[
                "basis_v2"
            ].eq(
                "OFS"
            )
        ]

        unknown = semantically_usable.loc[
            semantically_usable[
                "basis_v2"
            ].isin(
                [
                    "UNKNOWN",
                    "MIXED",
                ]
            )
        ]

        cfs_values = (
            cfs[
                "_amount_numeric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        ofs_values = (
            ofs[
                "_amount_numeric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        unknown_values = (
            unknown[
                "_amount_numeric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if len(
            cfs_values
        ) == 1:

            decision = (
                "CFS_STRONG_SINGLE_VALUE"
            )

        elif len(
            cfs_values
        ) > 1:

            decision = (
                "CFS_STRONG_MULTIPLE_VALUES"
            )

        elif len(
            ofs_values
        ) > 0:

            decision = (
                "OFS_STRONG_NO_CFS_FALLBACK_PROOF"
            )

        elif len(
            unknown_values
        ) > 0:

            decision = (
                "STRONG_VALUE_BASIS_UNRESOLVED"
            )

        elif not contradiction.empty:

            decision = (
                "ONLY_LABEL_ID_CONTRADICTION"
            )

        else:

            decision = (
                "NO_STRONG_TOTAL_CANDIDATE"
            )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                getattr(
                    row,
                    "stock_code",
                    "",
                ),

                "corp_name":
                getattr(
                    row,
                    "corp_name",
                    "",
                ),

                "period_key":
                getattr(
                    row,
                    "period_key",
                    "",
                ),

                "missing_metric":
                metric,

                "strong_rows":
                len(
                    group
                ),

                "id_contradiction_rows":
                len(
                    contradiction
                ),

                "cfs_strong_rows":
                len(
                    cfs
                ),

                "cfs_unique_values":
                len(
                    cfs_values
                ),

                "ofs_strong_rows":
                len(
                    ofs
                ),

                "ofs_unique_values":
                len(
                    ofs_values
                ),

                "unknown_basis_rows":
                len(
                    unknown
                ),

                "unknown_basis_unique_values":
                len(
                    unknown_values
                ),

                "basis_id_decision":
                decision,
            }
        )

    summary = pd.DataFrame(
        receipt_records
    )

    summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    review = summary.loc[
        ~summary[
            "basis_id_decision"
        ].eq(
            "CFS_STRONG_SINGLE_VALUE"
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
        "\n[Strong-total rows by missing metric]"
    )

    print(
        strong[
            "missing_metric"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Strong-total basis]"
    )

    print(
        pd.crosstab(
            strong[
                "missing_metric"
            ],
            strong[
                "basis_v2"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Label ↔ account-id consistency]"
    )

    print(
        pd.crosstab(
            strong[
                "missing_metric"
            ],
            strong[
                "id_consistency"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Receipt decision]"
    )

    print(
        pd.crosstab(
            summary[
                "missing_metric"
            ],
            summary[
                "basis_id_decision"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[ASSETS strong candidate detail]"
    )

    assets = strong.loc[
        strong[
            "missing_metric"
        ].eq(
            "assets"
        )
    ]

    if assets.empty:
        print("None")
    else:
        print(
            assets[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "fs_div",
                    "basis_v2",
                    "basis_evidence",
                    "account_nm",
                    "account_id",
                    "id_consistency",
                    "_amount_numeric",
                    "asset_balance_status",
                    "asset_balance_gap",
                    "asset_balance_rel_gap",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[OPERATING INCOME strong candidate detail]"
    )

    op = strong.loc[
        strong[
            "missing_metric"
        ].eq(
            "operating_income_cumulative"
        )
    ]

    if op.empty:
        print("None")
    else:
        print(
            op[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "basis_v2",
                    "basis_evidence",
                    "account_nm",
                    "account_id",
                    "id_consistency",
                    "id_consistency_reason",
                    "_amount_numeric",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[NET INCOME receipt decision]"
    )

    net = summary.loc[
        summary[
            "missing_metric"
        ].eq(
            "net_income_total_cumulative"
        )
    ]

    print(
        net[
            "basis_id_decision"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[NET INCOME CFS strong sample]"
    )

    net_cfs = strong.loc[
        strong[
            "missing_metric"
        ].eq(
            "net_income_total_cumulative"
        )
        & strong[
            "basis_v2"
        ].eq(
            "CFS"
        )
        & ~strong[
            "id_consistency"
        ].eq(
            "CONTRADICTION"
        )
    ]

    if net_cfs.empty:
        print("None")
    else:
        print(
            net_cfs[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "account_nm",
                    "account_id",
                    "id_consistency",
                    "_amount_numeric",
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
        "\n[REVENUE strong candidate detail]"
    )

    revenue = strong.loc[
        strong[
            "missing_metric"
        ].eq(
            "revenue_cumulative"
        )
    ]

    if revenue.empty:
        print("None")
    else:
        print(
            revenue[
                [
                    "_source",
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "basis_v2",
                    "basis_evidence",
                    "account_nm",
                    "account_id",
                    "id_consistency",
                    "_amount_numeric",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if SELECTED_MISSING.exists():

        selected_missing = pd.read_csv(
            SELECTED_MISSING,
            dtype={
                "rcept_no":
                str,
                "stock_code":
                str,
            },
            low_memory=False,
        )

        print(
            "\n[Selected parent-net-income rows are target-compatible?]"
        )

        if "missing_metric" in selected_missing.columns:

            print(
                selected_missing[
                    "missing_metric"
                ]
                .value_counts()
                .to_string()
            )

        print(
            "\nInterpretation: these are provenance evidence only; "
            "net_income_parent_cumulative is NOT a substitute for total net income."
        )

    print(
        "\nOutputs:"
    )

    print(
        f"- Strong row audit : "
        f"{OUT_ROWS}"
    )

    print(
        f"- Receipt summary  : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Review queue     : "
        f"{OUT_REVIEW}"
    )

    print(
        "\n해석 원칙:"
        "\n- CFS_STRONG_SINGLE_VALUE도 아직 flow YTD 검증 전이라 merge 금지"
        "\n- OFS strong만 있는 경우 no-CFS fallback proof 없이는 채택 금지"
        "\n- LABEL_ONLY_NO_STANDARD_ID는 context/period 검증 필요"
        "\n- CONTRADICTION은 자동채택 금지"
        "\n- assets는 balance coherence가 맞아도 basis가 OFS면 CFS와 섞지 않음"
        "\n- 다음 단계는 이 결과로 net-income CFS 후보 / revenue domain-specific / small residual을 분리"
    )


if __name__ == "__main__":
    main()
