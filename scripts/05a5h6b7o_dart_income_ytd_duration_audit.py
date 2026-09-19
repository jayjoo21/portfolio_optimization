from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7O. Income YTD Duration Semantic Audit
#
# 목적
# ------------------------------------------------------------
# H6B7N core6_source_ready receipt의 income values가
# 목표 정의인 "YTD cumulative flow"인지 마지막으로 검증한다.
#
# 특히:
#   Q1  : 3개월 = YTD 이므로 first-quarter 증거가 있으면 허용
#   H1  : 6개월/반기/누적 증거 필요
#   Q3  : 9개월/누적 증거 필요. 3개월-only는 차단
#   FY  : 연간값이므로 duration 측면 PASS
#
# IMPORTANT
# ------------------------------------------------------------
# - 분기보고서의 period_key month만 보고 Q1/Q3를 단정하지 않는다.
#   비12월 결산법인이 있을 수 있기 때문.
# - 실제 selected column / table context / cached header evidence를 우선한다.
# - 애매하면 REVIEW.
# - production merge 없음.
# - ZIP/API 없음.
#
# 실행:
# python scripts\05a5h6b7o_dart_income_ytd_duration_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

DRYRUN_RECEIPTS = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_dryrun_receipt_summary.csv"
)

CANDIDATES = (
    INTERIM
    / "dart_noncorrected_nodata_selector_v2_candidate_audit.parquet"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_receipt_audit.csv"
)

OUT_ROW = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_row_evidence.csv"
)

OUT_REVIEW = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_review_queue.csv"
)


INCOME_METRICS = [
    "revenue",
    "operating_income",
    "net_income",
]


# ============================================================
# General helpers
# ============================================================


def normalize_receipt(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def compact(value) -> str:
    return re.sub(
        r"[\s\|\[\]\(\)\{\}:;,_./\\\-]+",
        "",
        clean(value).lower(),
    )


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

    diff = abs(
        a - b
    )

    denom = max(
        abs(a),
        abs(b),
        1.0,
    )

    return (
        diff <= 1_000_000
        or diff / denom <= 1e-5
    )


def report_type(period_key) -> str:
    text = clean(
        period_key
    )

    if "사업보고서" in text:
        return "FY"

    if "반기보고서" in text:
        return "H1"

    if "분기보고서" in text:
        return "QUARTER"

    return "UNKNOWN"


# ============================================================
# Evidence extraction
# ============================================================


CUMULATIVE_MARKERS = [
    "누적",
    "누계",
    "누계액",
    "누적액",
    "누적금액",
    "누적손익",
    "누적실적",
]

SIX_MONTH_MARKERS = [
    "6개월",
    "육개월",
    "반기누적",
    "당반기누적",
]

NINE_MONTH_MARKERS = [
    "9개월",
    "구개월",
    "3분기누적",
    "삼분기누적",
]

THREE_MONTH_MARKERS = [
    "3개월",
    "삼개월",
]

FIRST_QUARTER_MARKERS = [
    "1분기",
    "제1분기",
    "일분기",
]

THIRD_QUARTER_MARKERS = [
    "3분기",
    "제3분기",
    "삼분기",
]

HALF_MARKERS = [
    "반기",
    "당반기",
    "상반기",
]


def contains_any(
    text: str,
    markers: list[str],
) -> bool:
    return any(
        marker in text
        for marker in markers
    )


def safe_json_text(value) -> str:
    """
    columns_json / row_cells_json 등의 JSON을 사람이 읽을 수 있는
    단순 문자열 evidence로 펼친다.
    """
    if pd.isna(value):
        return ""

    raw = str(
        value
    ).strip()

    if not raw:
        return ""

    try:
        obj = json.loads(
            raw
        )

        if isinstance(
            obj,
            dict,
        ):
            pieces = []

            for key, val in obj.items():
                pieces.append(
                    str(
                        key
                    )
                )
                pieces.append(
                    str(
                        val
                    )
                )

            return " | ".join(
                pieces
            )

        if isinstance(
            obj,
            list,
        ):
            return " | ".join(
                str(
                    x
                )
                for x in obj
            )

        return str(
            obj
        )

    except Exception:
        return raw


def row_evidence_text(
    row: pd.Series,
) -> tuple[str, str, str]:

    selected_col = clean(
        row.get(
            "period_v2_selected_column",
            "",
        )
    )

    heading = clean(
        row.get(
            "heading_context",
            "",
        )
    )

    extra_parts = []

    for col in [
        "columns_json",
        "column_headers_json",
        "header_json",
        "row_cells_json",
        "numeric_candidates_json",
        "period_v2_reason",
        "period_v3_reason",
        "column_consistency_reason",
    ]:
        if col in row.index:
            value = row.get(
                col
            )

            if col.endswith(
                "_json"
            ):
                text = safe_json_text(
                    value
                )
            else:
                text = clean(
                    value
                )

            if text:
                extra_parts.append(
                    f"{col}={text}"
                )

    extra = " || ".join(
        extra_parts
    )

    combined = " || ".join(
        part
        for part in [
            selected_col,
            heading,
            extra,
        ]
        if part
    )

    return (
        selected_col,
        heading,
        combined,
    )


def quarter_position(
    evidence_compact: str,
) -> tuple[str, str]:

    first = contains_any(
        evidence_compact,
        FIRST_QUARTER_MARKERS,
    )

    third = contains_any(
        evidence_compact,
        THIRD_QUARTER_MARKERS,
    )

    if first and not third:
        return (
            "Q1",
            "explicit_first_quarter_marker",
        )

    if third and not first:
        return (
            "Q3",
            "explicit_third_quarter_marker",
        )

    if first and third:
        return (
            "AMBIGUOUS",
            "both_first_and_third_quarter_markers",
        )

    return (
        "UNKNOWN",
        "",
    )


def duration_classify(
    period_key,
    selected_col,
    heading,
    combined,
) -> tuple[str, str, str]:

    rtype = report_type(
        period_key
    )

    selected_text = compact(
        selected_col
    )

    # selected column is primary evidence.
    combined_text = compact(
        combined
    )

    heading_text = compact(
        heading
    )

    cumulative_selected = contains_any(
        selected_text,
        CUMULATIVE_MARKERS,
    )

    six_selected = contains_any(
        selected_text,
        SIX_MONTH_MARKERS,
    )

    nine_selected = contains_any(
        selected_text,
        NINE_MONTH_MARKERS,
    )

    three_selected = contains_any(
        selected_text,
        THREE_MONTH_MARKERS,
    )

    half_selected = contains_any(
        selected_text,
        HALF_MARKERS,
    )

    quarter_pos, quarter_reason = (
        quarter_position(
            selected_text
        )
    )

    if quarter_pos == "UNKNOWN":
        # table context can establish quarter ordinal,
        # but only after selected-column evidence failed.
        quarter_pos, quarter_reason = (
            quarter_position(
                heading_text
            )
        )

    cumulative_any = contains_any(
        combined_text,
        CUMULATIVE_MARKERS,
    )

    six_any = contains_any(
        combined_text,
        SIX_MONTH_MARKERS,
    )

    nine_any = contains_any(
        combined_text,
        NINE_MONTH_MARKERS,
    )

    three_any = contains_any(
        combined_text,
        THREE_MONTH_MARKERS,
    )

    # --------------------------------------------------------
    # FY
    # --------------------------------------------------------

    if rtype == "FY":
        return (
            "PASS",
            "FY_ANNUAL_IS_YTD",
            quarter_pos,
        )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    if rtype == "H1":

        if (
            cumulative_selected
            or six_selected
            or half_selected
        ):
            return (
                "PASS",
                "H1_SELECTED_COLUMN_CUMULATIVE_OR_HALF",
                quarter_pos,
            )

        if (
            three_selected
            and not cumulative_selected
        ):
            return (
                "BLOCK",
                "H1_SELECTED_COLUMN_3MONTH_ONLY",
                quarter_pos,
            )

        if (
            cumulative_any
            or six_any
        ):
            return (
                "REVIEW",
                "H1_CUMULATIVE_EVIDENCE_ONLY_IN_CONTEXT_NOT_SELECTED_COLUMN",
                quarter_pos,
            )

        if (
            "반기" in heading_text
            and not three_any
        ):
            return (
                "REVIEW",
                "H1_GENERIC_HALF_CONTEXT_WITHOUT_SELECTED_DURATION_MARKER",
                quarter_pos,
            )

        return (
            "REVIEW",
            "H1_DURATION_UNRESOLVED",
            quarter_pos,
        )

    # --------------------------------------------------------
    # QUARTER
    # --------------------------------------------------------

    if rtype == "QUARTER":

        # Explicit Q1:
        # 3-month flow is YTD by definition.
        if quarter_pos == "Q1":

            if (
                cumulative_selected
                or three_selected
                or contains_any(
                    selected_text,
                    FIRST_QUARTER_MARKERS,
                )
            ):
                return (
                    "PASS",
                    "Q1_3MONTH_EQUALS_YTD",
                    quarter_pos,
                )

            if (
                cumulative_any
                or three_any
            ):
                return (
                    "REVIEW",
                    "Q1_DURATION_EVIDENCE_CONTEXT_ONLY",
                    quarter_pos,
                )

            return (
                "REVIEW",
                "Q1_ORDINAL_KNOWN_BUT_SELECTED_DURATION_UNRESOLVED",
                quarter_pos,
            )

        # Explicit Q3:
        # Must be cumulative / 9-month.
        if quarter_pos == "Q3":

            if (
                cumulative_selected
                or nine_selected
            ):
                return (
                    "PASS",
                    "Q3_SELECTED_COLUMN_9MONTH_OR_CUMULATIVE",
                    quarter_pos,
                )

            if (
                three_selected
                and not cumulative_selected
            ):
                return (
                    "BLOCK",
                    "Q3_SELECTED_COLUMN_3MONTH_ONLY",
                    quarter_pos,
                )

            if (
                cumulative_any
                or nine_any
            ):
                return (
                    "REVIEW",
                    "Q3_CUMULATIVE_EVIDENCE_CONTEXT_ONLY",
                    quarter_pos,
                )

            return (
                "REVIEW",
                "Q3_DURATION_UNRESOLVED",
                quarter_pos,
            )

        # Quarter ordinal unknown.
        if cumulative_selected:
            return (
                "PASS",
                "QUARTER_ORDINAL_UNKNOWN_BUT_SELECTED_COLUMN_EXPLICIT_CUMULATIVE",
                quarter_pos,
            )

        if nine_selected:
            return (
                "PASS",
                "QUARTER_SELECTED_COLUMN_EXPLICIT_9MONTH",
                "Q3",
            )

        if three_selected:
            # Could be Q1 YTD or Q3 current-quarter.
            return (
                "REVIEW",
                "QUARTER_SELECTED_COLUMN_3MONTH_BUT_Q1_VS_Q3_UNKNOWN",
                quarter_pos,
            )

        if cumulative_any:
            return (
                "REVIEW",
                "QUARTER_CUMULATIVE_EVIDENCE_CONTEXT_ONLY",
                quarter_pos,
            )

        return (
            "REVIEW",
            "QUARTER_DURATION_UNRESOLVED",
            quarter_pos,
        )

    return (
        "REVIEW",
        "REPORT_TYPE_UNKNOWN",
        quarter_pos,
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        DRYRUN_RECEIPTS,
        CANDIDATES,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    receipts = pd.read_csv(
        DRYRUN_RECEIPTS,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    candidates = pd.read_parquet(
        CANDIDATES
    )

    receipts[
        "rcept_no"
    ] = normalize_receipt(
        receipts[
            "rcept_no"
        ]
    )

    candidates[
        "rcept_no"
    ] = normalize_receipt(
        candidates[
            "rcept_no"
        ]
    )

    ready = receipts.loc[
        receipts[
            "core6_source_ready"
        ].fillna(
            False
        ).astype(
            bool
        )
    ].copy()

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7O INCOME YTD DURATION SEMANTIC AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        f"\nCore6 source-ready receipts: "
        f"{len(ready):,}"
    )

    # --------------------------------------------------------
    # Candidate row lookup for selected income table
    # --------------------------------------------------------

    selected = ready[
        [
            "rcept_no",
            "stock_code",
            "period_key",
            "income_selected_table",
            "revenue",
            "operating_income",
            "net_income",
        ]
    ].copy()

    selected[
        "income_selected_table"
    ] = pd.to_numeric(
        selected[
            "income_selected_table"
        ],
        errors="coerce",
    )

    candidate_income = candidates.loc[
        candidates[
            "account_family"
        ].isin(
            INCOME_METRICS
        )
    ].copy()

    candidate_income[
        "table_index_num"
    ] = pd.to_numeric(
        candidate_income[
            "table_index"
        ],
        errors="coerce",
    )

    candidate_income = candidate_income.merge(
        selected[
            [
                "rcept_no",
                "income_selected_table",
            ]
        ],
        on="rcept_no",
        how="inner",
        validate="many_to_one",
    )

    candidate_income = candidate_income.loc[
        candidate_income[
            "table_index_num"
        ].eq(
            candidate_income[
                "income_selected_table"
            ]
        )
    ].copy()

    target_value_map = {}

    for row in selected.itertuples(
        index=False
    ):
        target_value_map[
            (
                str(
                    row.rcept_no
                ),
                "revenue",
            )
        ] = row.revenue

        target_value_map[
            (
                str(
                    row.rcept_no
                ),
                "operating_income",
            )
        ] = row.operating_income

        target_value_map[
            (
                str(
                    row.rcept_no
                ),
                "net_income",
            )
        ] = row.net_income

    candidate_income[
        "dryrun_target_value"
    ] = [
        target_value_map.get(
            (
                str(
                    rcept_no
                ),
                family,
            ),
            np.nan,
        )
        for rcept_no, family
        in zip(
            candidate_income[
                "rcept_no"
            ],
            candidate_income[
                "account_family"
            ],
        )
    ]

    candidate_income[
        "matches_dryrun_value"
    ] = [
        near_equal(
            candidate,
            target,
        )
        for candidate, target
        in zip(
            pd.to_numeric(
                candidate_income[
                    "selector_value_krw"
                ],
                errors="coerce",
            ),
            pd.to_numeric(
                candidate_income[
                    "dryrun_target_value"
                ],
                errors="coerce",
            ),
        )
    ]

    matched = candidate_income.loc[
        candidate_income[
            "matches_dryrun_value"
        ]
    ].copy()

    # --------------------------------------------------------
    # Row-level duration evidence
    # --------------------------------------------------------

    row_records = []

    for _, row in matched.iterrows():

        (
            selected_col,
            heading,
            combined,
        ) = row_evidence_text(
            row
        )

        (
            status,
            reason,
            qpos,
        ) = duration_classify(
            row.get(
                "period_key",
                "",
            ),
            selected_col,
            heading,
            combined,
        )

        row_records.append(
            {
                "rcept_no":
                row[
                    "rcept_no"
                ],

                "stock_code":
                row.get(
                    "stock_code",
                    "",
                ),

                "period_key":
                row.get(
                    "period_key",
                    "",
                ),

                "account_family":
                row.get(
                    "account_family",
                    "",
                ),

                "table_index":
                row.get(
                    "table_index",
                    np.nan,
                ),

                "row_index":
                row.get(
                    "row_index",
                    np.nan,
                ),

                "selector_value_krw":
                row.get(
                    "selector_value_krw",
                    np.nan,
                ),

                "period_v2_selected_column":
                selected_col,

                "quarter_position_evidence":
                qpos,

                "duration_status":
                status,

                "duration_reason":
                reason,

                "heading_context":
                heading,

                "duration_combined_evidence":
                combined[
                    :2500
                ],
            }
        )

    row_audit = pd.DataFrame(
        row_records
    )

    row_audit.to_csv(
        OUT_ROW,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt-level aggregation
    # --------------------------------------------------------

    receipt_records = []

    for row in ready.itertuples(
        index=False
    ):

        rcept_no = str(
            row.rcept_no
        )

        sub = row_audit.loc[
            row_audit[
                "rcept_no"
            ].eq(
                rcept_no
            )
        ].copy()

        metric_status = {}

        for family in INCOME_METRICS:

            fam = sub.loc[
                sub[
                    "account_family"
                ].eq(
                    family
                )
            ]

            statuses = set(
                fam[
                    "duration_status"
                ].astype(str)
            )

            if not statuses:
                final = (
                    "REVIEW"
                )

                reason = (
                    "NO_MATCHED_SOURCE_ROW"
                )

            elif statuses == {
                "PASS"
            }:
                final = "PASS"

                reason = "|".join(
                    sorted(
                        set(
                            fam[
                                "duration_reason"
                            ].astype(str)
                        )
                    )
                )

            elif "BLOCK" in statuses:
                final = "BLOCK"

                reason = "|".join(
                    sorted(
                        set(
                            fam[
                                "duration_reason"
                            ].astype(str)
                        )
                    )
                )

            else:
                final = "REVIEW"

                reason = "|".join(
                    sorted(
                        set(
                            fam[
                                "duration_reason"
                            ].astype(str)
                        )
                    )
                )

            metric_status[
                family
            ] = (
                final,
                reason,
                len(
                    fam
                ),
            )

        statuses = [
            metric_status[
                family
            ][
                0
            ]
            for family in INCOME_METRICS
        ]

        if all(
            status == "PASS"
            for status in statuses
        ):
            receipt_status = (
                "PASS_ALL_INCOME_YTD"
            )

        elif any(
            status == "BLOCK"
            for status in statuses
        ):
            receipt_status = (
                "BLOCK_INCOME_DURATION"
            )

        else:
            receipt_status = (
                "REVIEW_INCOME_DURATION"
            )

        receipt_records.append(
            {
                "rcept_no":
                rcept_no,

                "stock_code":
                row.stock_code,

                "period_key":
                row.period_key,

                "income_selected_table":
                row.income_selected_table,

                "income_ytd_audit_status":
                receipt_status,

                "revenue_duration_status":
                metric_status[
                    "revenue"
                ][
                    0
                ],

                "revenue_duration_reason":
                metric_status[
                    "revenue"
                ][
                    1
                ],

                "operating_income_duration_status":
                metric_status[
                    "operating_income"
                ][
                    0
                ],

                "operating_income_duration_reason":
                metric_status[
                    "operating_income"
                ][
                    1
                ],

                "net_income_duration_status":
                metric_status[
                    "net_income"
                ][
                    0
                ],

                "net_income_duration_reason":
                metric_status[
                    "net_income"
                ][
                    1
                ],

                "revenue_matched_row_count":
                metric_status[
                    "revenue"
                ][
                    2
                ],

                "operating_income_matched_row_count":
                metric_status[
                    "operating_income"
                ][
                    2
                ],

                "net_income_matched_row_count":
                metric_status[
                    "net_income"
                ][
                    2
                ],
            }
        )

    receipt_audit = pd.DataFrame(
        receipt_records
    )

    receipt_audit.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    review_queue = receipt_audit.loc[
        ~receipt_audit[
            "income_ytd_audit_status"
        ].eq(
            "PASS_ALL_INCOME_YTD"
        )
    ].copy()

    review_queue.to_csv(
        OUT_REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Matched source rows]"
    )

    print(
        f"{len(row_audit):,}"
    )

    print(
        "\n[Row duration status]"
    )

    if row_audit.empty:
        print(
            "None"
        )
    else:
        print(
            row_audit[
                "duration_status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Row duration reason]"
    )

    if row_audit.empty:
        print(
            "None"
        )
    else:
        print(
            row_audit[
                "duration_reason"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Receipt income YTD audit]"
    )

    print(
        receipt_audit[
            "income_ytd_audit_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Audit by report type]"
    )

    temp = receipt_audit.copy()

    temp[
        "report_type"
    ] = temp[
        "period_key"
    ].map(
        report_type
    )

    print(
        pd.crosstab(
            temp[
                "report_type"
            ],
            temp[
                "income_ytd_audit_status"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Metric duration status]"
    )

    for family in INCOME_METRICS:

        col = (
            f"{family}_duration_status"
        )

        print(
            f"\n{family}:"
        )

        print(
            receipt_audit[
                col
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[BLOCK sample]"
    )

    blocked = receipt_audit.loc[
        receipt_audit[
            "income_ytd_audit_status"
        ].eq(
            "BLOCK_INCOME_DURATION"
        )
    ]

    if blocked.empty:
        print(
            "None"
        )
    else:
        print(
            blocked[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "income_selected_table",
                    "revenue_duration_status",
                    "revenue_duration_reason",
                    "operating_income_duration_status",
                    "operating_income_duration_reason",
                    "net_income_duration_status",
                    "net_income_duration_reason",
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
        "\n[REVIEW sample]"
    )

    review = receipt_audit.loc[
        receipt_audit[
            "income_ytd_audit_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
    ]

    if review.empty:
        print(
            "None"
        )
    else:
        print(
            review[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "income_selected_table",
                    "revenue_duration_status",
                    "revenue_duration_reason",
                    "operating_income_duration_status",
                    "operating_income_duration_reason",
                    "net_income_duration_status",
                    "net_income_duration_reason",
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
        "\n[Q3 / third-quarter matched-row sample]"
    )

    q3 = row_audit.loc[
        row_audit[
            "quarter_position_evidence"
        ].eq(
            "Q3"
        )
    ]

    if q3.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            340,
            "display.max_rows",
            40,
        ):
            print(
                q3[
                    [
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "account_family",
                        "table_index",
                        "period_v2_selected_column",
                        "duration_status",
                        "duration_reason",
                        "heading_context",
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
        "\n[H1 matched-row sample]"
    )

    h1_receipts = set(
        temp.loc[
            temp[
                "report_type"
            ].eq(
                "H1"
            ),
            "rcept_no",
        ]
    )

    h1_rows = row_audit.loc[
        row_audit[
            "rcept_no"
        ].isin(
            h1_receipts
        )
    ]

    if h1_rows.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            340,
            "display.max_rows",
            40,
        ):
            print(
                h1_rows[
                    [
                        "stock_code",
                        "period_key",
                        "rcept_no",
                        "account_family",
                        "table_index",
                        "period_v2_selected_column",
                        "duration_status",
                        "duration_reason",
                        "heading_context",
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
        f"- Receipt audit : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Row evidence  : "
        f"{OUT_ROW}"
    )

    print(
        f"- Review queue  : "
        f"{OUT_REVIEW}"
    )

    print(
        "\n해석 원칙:"
        "\n- FY는 annual flow이므로 duration PASS"
        "\n- Q1은 3개월 자체가 YTD"
        "\n- H1은 반기/6개월/누적 evidence 필요"
        "\n- Q3은 누적/9개월 evidence 필요; 3개월-only면 BLOCK"
        "\n- period_key의 달력 월만으로 Q1/Q3를 확정하지 않음"
        "\n- 이 audit까지 PASS한 source-ready receipt만 production merge 후보"
    )


if __name__ == "__main__":
    main()
