from __future__ import annotations

import calendar
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 05A5-H6B7P. Income YTD Explicit-Range Resolver
#
# 목적
# ------------------------------------------------------------
# H6B7O REVIEW 40건 중,
# selected column 자체에 명시된 날짜/월 범위로
# YTD duration을 직접 확인할 수 있는 경우만 추가 PASS한다.
#
# 예:
#   FY2018 3분기 (2018.1~9월)
#       -> inclusive 9-month span -> Q3 YTD PASS
#
#   2019.01.01 ~ 2019.09.30
#       -> 약 9개월 -> Q3 YTD PASS
#
# 반대로:
#   제50기 3분기 | -
#   제32기 2분기말 (2018년 6월말) | -
#       -> 시작점 없음 -> REVIEW 유지
#
# IMPORTANT
# ------------------------------------------------------------
# - calendar month 자체로 Q1/Q3를 추정하지 않음
# - selected-column explicit range만 strong evidence로 사용
# - heading의 여러 날짜 범위는 선택 열과 연결이 불확실하므로 자동 PASS 근거로 사용하지 않음
# - ZIP/API 없음
# - production merge 없음
#
# 실행:
# python scripts\05a5h6b7p_dart_income_ytd_explicit_range_resolver.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

H6B7O_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_receipt_audit.csv"
)

H6B7O_ROWS = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_row_evidence.csv"
)

OUT_RECEIPT = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_duration_final_audit.csv"
)

OUT_RANGE = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_explicit_range_evidence.csv"
)

OUT_REVIEW = (
    INTERIM
    / "dart_noncorrected_nodata_income_ytd_final_review_queue.csv"
)


INCOME_METRICS = [
    "revenue",
    "operating_income",
    "net_income",
]


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


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
# Explicit range parsers
# ============================================================


def month_span(
    y1: int,
    m1: int,
    y2: int,
    m2: int,
) -> int | None:

    if not (
        1 <= m1 <= 12
        and 1 <= m2 <= 12
    ):
        return None

    months = (
        (
            y2 * 12
            + m2
        )
        - (
            y1 * 12
            + m1
        )
        + 1
    )

    if months <= 0:
        return None

    return months


def parse_month_range(
    text: str,
) -> list[dict]:

    """
    Examples:
      2018.1~9월
      2018.01~2018.09
      2018년 1월~9월
      2018년 1월 ~ 2018년 9월
    """

    out = []

    patterns = [
        re.compile(
            r"(?P<y1>20\d{2})\s*[.\-/년]\s*"
            r"(?P<m1>\d{1,2})\s*월?"
            r"\s*(?:~|～|−|–|—|부터)\s*"
            r"(?:(?P<y2>20\d{2})\s*[.\-/년]\s*)?"
            r"(?P<m2>\d{1,2})\s*월"
        ),

        re.compile(
            r"(?P<y1>20\d{2})\s*년\s*"
            r"(?P<m1>\d{1,2})\s*월"
            r"\s*(?:~|～|−|–|—|부터)\s*"
            r"(?:(?P<y2>20\d{2})\s*년\s*)?"
            r"(?P<m2>\d{1,2})\s*월"
        ),
    ]

    for pattern in patterns:

        for match in pattern.finditer(
            text
        ):

            y1 = int(
                match.group(
                    "y1"
                )
            )

            m1 = int(
                match.group(
                    "m1"
                )
            )

            y2_raw = match.group(
                "y2"
            )

            y2 = (
                int(
                    y2_raw
                )
                if y2_raw
                else y1
            )

            m2 = int(
                match.group(
                    "m2"
                )
            )

            span = month_span(
                y1,
                m1,
                y2,
                m2,
            )

            if span is None:
                continue

            out.append(
                {
                    "kind":
                    "MONTH_RANGE",

                    "raw":
                    match.group(
                        0
                    ),

                    "start_year":
                    y1,

                    "start_month":
                    m1,

                    "end_year":
                    y2,

                    "end_month":
                    m2,

                    "span_months":
                    span,

                    "span_days":
                    np.nan,
                }
            )

    # de-duplicate
    unique = []

    seen = set()

    for item in out:

        key = (
            item[
                "raw"
            ],
            item[
                "start_year"
            ],
            item[
                "start_month"
            ],
            item[
                "end_year"
            ],
            item[
                "end_month"
            ],
        )

        if key not in seen:
            seen.add(
                key
            )
            unique.append(
                item
            )

    return unique


def parse_full_date_range(
    text: str,
) -> list[dict]:

    """
    Examples:
      2018년 1월 1일부터 2018년 9월 30일까지
      2018.01.01 ~ 2018.09.30
      2018-01-01 ~ 2018-09-30
    """

    out = []

    patterns = [
        re.compile(
            r"(?P<y1>20\d{2})\s*년\s*"
            r"(?P<m1>\d{1,2})\s*월\s*"
            r"(?P<d1>\d{1,2})\s*일"
            r"\s*(?:부터|~|～|−|–|—)\s*"
            r"(?P<y2>20\d{2})\s*년\s*"
            r"(?P<m2>\d{1,2})\s*월\s*"
            r"(?P<d2>\d{1,2})\s*일"
            r"\s*(?:까지)?"
        ),

        re.compile(
            r"(?P<y1>20\d{2})[.\-/]"
            r"(?P<m1>\d{1,2})[.\-/]"
            r"(?P<d1>\d{1,2})"
            r"\s*(?:부터|~|～|−|–|—)\s*"
            r"(?P<y2>20\d{2})[.\-/]"
            r"(?P<m2>\d{1,2})[.\-/]"
            r"(?P<d2>\d{1,2})"
        ),
    ]

    for pattern in patterns:

        for match in pattern.finditer(
            text
        ):

            try:
                start = date(
                    int(
                        match.group(
                            "y1"
                        )
                    ),
                    int(
                        match.group(
                            "m1"
                        )
                    ),
                    int(
                        match.group(
                            "d1"
                        )
                    ),
                )

                end = date(
                    int(
                        match.group(
                            "y2"
                        )
                    ),
                    int(
                        match.group(
                            "m2"
                        )
                    ),
                    int(
                        match.group(
                            "d2"
                        )
                    ),
                )

            except ValueError:
                continue

            if end < start:
                continue

            span_days = (
                end
                - start
            ).days + 1

            span_months = month_span(
                start.year,
                start.month,
                end.year,
                end.month,
            )

            out.append(
                {
                    "kind":
                    "FULL_DATE_RANGE",

                    "raw":
                    match.group(
                        0
                    ),

                    "start_year":
                    start.year,

                    "start_month":
                    start.month,

                    "end_year":
                    end.year,

                    "end_month":
                    end.month,

                    "span_months":
                    span_months,

                    "span_days":
                    span_days,
                }
            )

    return out


def all_explicit_ranges(
    selected_column,
) -> list[dict]:

    text = clean(
        selected_column
    )

    ranges = (
        parse_full_date_range(
            text
        )
        + parse_month_range(
            text
        )
    )

    unique = []

    seen = set()

    for item in ranges:

        key = (
            item[
                "kind"
            ],
            item[
                "raw"
            ],
        )

        if key not in seen:
            seen.add(
                key
            )

            unique.append(
                item
            )

    return unique


# ============================================================
# Duration decision from explicit range
# ============================================================


def range_duration_class(
    rtype: str,
    ranges: list[dict],
) -> tuple[
    str,
    str,
    int | None,
]:

    if not ranges:
        return (
            "NO_UPGRADE",
            "NO_SELECTED_COLUMN_EXPLICIT_RANGE",
            None,
        )

    spans = sorted(
        set(
            int(
                item[
                    "span_months"
                ]
            )
            for item in ranges
            if pd.notna(
                item[
                    "span_months"
                ]
            )
        )
    )

    # If selected-column text somehow contains several conflicting
    # ranges, do not infer.
    if len(
        spans
    ) != 1:
        return (
            "REVIEW",
            "MULTIPLE_SELECTED_COLUMN_RANGE_SPANS",
            None,
        )

    span = spans[
        0
    ]

    if rtype == "FY":

        if span >= 12:
            return (
                "PASS",
                "FY_EXPLICIT_12M_OR_LONGER_RANGE",
                span,
            )

        return (
            "REVIEW",
            f"FY_EXPLICIT_RANGE_{span}M",
            span,
        )

    if rtype == "H1":

        if span == 6:
            return (
                "PASS",
                "H1_EXPLICIT_6M_RANGE",
                span,
            )

        if span == 3:
            return (
                "BLOCK",
                "H1_EXPLICIT_3M_ONLY_RANGE",
                span,
            )

        return (
            "REVIEW",
            f"H1_EXPLICIT_RANGE_{span}M",
            span,
        )

    if rtype == "QUARTER":

        if span == 3:
            # Could be Q1 YTD or a current-quarter-only Q3.
            return (
                "REVIEW",
                "QUARTER_EXPLICIT_3M_RANGE_Q1_VS_Q3_UNRESOLVED",
                span,
            )

        if span == 9:
            return (
                "PASS",
                "Q3_EXPLICIT_9M_RANGE",
                span,
            )

        if span == 6:
            return (
                "REVIEW",
                "QUARTER_EXPLICIT_6M_RANGE_UNEXPECTED",
                span,
            )

        return (
            "REVIEW",
            f"QUARTER_EXPLICIT_RANGE_{span}M",
            span,
        )

    return (
        "REVIEW",
        "UNKNOWN_REPORT_TYPE",
        span,
    )


# ============================================================
# Main
# ============================================================


def main():

    for path in [
        H6B7O_RECEIPT,
        H6B7O_ROWS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    receipts = pd.read_csv(
        H6B7O_RECEIPT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    rows = pd.read_csv(
        H6B7O_ROWS,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    receipts[
        "rcept_no"
    ] = normalize_receipt(
        receipts[
            "rcept_no"
        ]
    )

    rows[
        "rcept_no"
    ] = normalize_receipt(
        rows[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7P INCOME YTD EXPLICIT-RANGE RESOLVER"
    )

    print(
        "=" * 120
    )

    review_receipts = receipts.loc[
        receipts[
            "income_ytd_audit_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
    ].copy()

    print(
        f"\nH6B7O REVIEW receipts: "
        f"{len(review_receipts):,}"
    )

    # --------------------------------------------------------
    # Build receipt-level selected-column evidence.
    # All three metrics normally share the same selected column,
    # but we keep ambiguity explicit if they do not.
    # --------------------------------------------------------

    range_records = []

    for rcept_no, group in rows.loc[
        rows[
            "rcept_no"
        ].isin(
            set(
                review_receipts[
                    "rcept_no"
                ]
            )
        )
    ].groupby(
        "rcept_no",
        sort=False,
    ):

        cols = (
            group[
                "period_v2_selected_column"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        cols = [
            value
            for value in dict.fromkeys(
                cols.tolist()
            )
            if value
        ]

        period_values = (
            group[
                "period_key"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        period_key = (
            period_values[
                0
            ]
            if period_values
            else ""
        )

        rtype = report_type(
            period_key
        )

        if len(
            cols
        ) == 0:

            range_records.append(
                {
                    "rcept_no":
                    rcept_no,

                    "period_key":
                    period_key,

                    "report_type":
                    rtype,

                    "selected_column_count":
                    0,

                    "selected_column":
                    "",

                    "explicit_ranges":
                    "",

                    "explicit_span_months":
                    np.nan,

                    "range_resolution":
                    "NO_UPGRADE",

                    "range_resolution_reason":
                    "NO_SELECTED_COLUMN",
                }
            )

            continue

        if len(
            cols
        ) > 1:

            # If all three metric rows duplicated but selected column
            # differs, do not silently choose one.
            range_records.append(
                {
                    "rcept_no":
                    rcept_no,

                    "period_key":
                    period_key,

                    "report_type":
                    rtype,

                    "selected_column_count":
                    len(
                        cols
                    ),

                    "selected_column":
                    " || ".join(
                        cols
                    ),

                    "explicit_ranges":
                    "",

                    "explicit_span_months":
                    np.nan,

                    "range_resolution":
                    "REVIEW",

                    "range_resolution_reason":
                    "MULTIPLE_SELECTED_COLUMNS_WITHIN_RECEIPT",
                }
            )

            continue

        selected_col = cols[
            0
        ]

        ranges = all_explicit_ranges(
            selected_col
        )

        (
            resolution,
            reason,
            span,
        ) = range_duration_class(
            rtype,
            ranges,
        )

        range_records.append(
            {
                "rcept_no":
                rcept_no,

                "period_key":
                period_key,

                "report_type":
                rtype,

                "selected_column_count":
                1,

                "selected_column":
                selected_col,

                "explicit_ranges":
                " || ".join(
                    item[
                        "raw"
                    ]
                    for item in ranges
                ),

                "explicit_span_months":
                span,

                "range_resolution":
                resolution,

                "range_resolution_reason":
                reason,
            }
        )

    range_audit = pd.DataFrame(
        range_records
    )

    range_audit.to_csv(
        OUT_RANGE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Final receipt audit
    # --------------------------------------------------------

    final = receipts.merge(
        range_audit[
            [
                "rcept_no",
                "selected_column",
                "explicit_ranges",
                "explicit_span_months",
                "range_resolution",
                "range_resolution_reason",
            ]
        ],
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    final[
        "income_ytd_final_status"
    ] = final[
        "income_ytd_audit_status"
    ]

    final[
        "income_ytd_final_reason"
    ] = np.where(
        final[
            "income_ytd_audit_status"
        ].eq(
            "PASS_ALL_INCOME_YTD"
        ),
        "H6B7O_PASS",
        "",
    )

    upgrade_mask = (
        final[
            "income_ytd_audit_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
        & final[
            "range_resolution"
        ].eq(
            "PASS"
        )
    )

    block_mask = (
        final[
            "income_ytd_audit_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
        & final[
            "range_resolution"
        ].eq(
            "BLOCK"
        )
    )

    final.loc[
        upgrade_mask,
        "income_ytd_final_status",
    ] = "PASS_ALL_INCOME_YTD"

    final.loc[
        upgrade_mask,
        "income_ytd_final_reason",
    ] = (
        "H6B7P_EXPLICIT_RANGE_PASS:"
        + final.loc[
            upgrade_mask,
            "range_resolution_reason",
        ].astype(str)
    )

    final.loc[
        block_mask,
        "income_ytd_final_status",
    ] = "BLOCK_INCOME_DURATION"

    final.loc[
        block_mask,
        "income_ytd_final_reason",
    ] = (
        "H6B7P_EXPLICIT_RANGE_BLOCK:"
        + final.loc[
            block_mask,
            "range_resolution_reason",
        ].astype(str)
    )

    remaining_review = (
        final[
            "income_ytd_audit_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
        & ~upgrade_mask
        & ~block_mask
    )

    final.loc[
        remaining_review,
        "income_ytd_final_status",
    ] = "REVIEW_INCOME_DURATION"

    final.loc[
        remaining_review,
        "income_ytd_final_reason",
    ] = (
        "H6B7P_REMAINS_REVIEW:"
        + final.loc[
            remaining_review,
            "range_resolution_reason",
        ]
        .fillna(
            "NO_RANGE_EVIDENCE"
        )
        .astype(str)
    )

    final.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    review_queue = final.loc[
        ~final[
            "income_ytd_final_status"
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
        "\n[Explicit range resolution among H6B7O reviews]"
    )

    if range_audit.empty:
        print(
            "None"
        )
    else:
        print(
            range_audit[
                "range_resolution"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Explicit range reason]"
    )

    if range_audit.empty:
        print(
            "None"
        )
    else:
        print(
            range_audit[
                "range_resolution_reason"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print(
        "\n[Final income YTD status]"
    )

    print(
        final[
            "income_ytd_final_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Newly upgraded PASS]"
    )

    upgraded = final.loc[
        upgrade_mask
    ]

    print(
        f"{len(upgraded):,}"
    )

    if not upgraded.empty:
        print(
            upgraded[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "income_selected_table",
                    "selected_column",
                    "explicit_ranges",
                    "explicit_span_months",
                    "income_ytd_final_reason",
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
        "\n[New explicit-range BLOCK]"
    )

    blocked = final.loc[
        block_mask
    ]

    print(
        f"{len(blocked):,}"
    )

    if not blocked.empty:
        print(
            blocked[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "income_selected_table",
                    "selected_column",
                    "explicit_ranges",
                    "explicit_span_months",
                    "income_ytd_final_reason",
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
        "\n[Remaining REVIEW]"
    )

    remaining = final.loc[
        final[
            "income_ytd_final_status"
        ].eq(
            "REVIEW_INCOME_DURATION"
        )
    ]

    print(
        f"{len(remaining):,}"
    )

    if not remaining.empty:
        print(
            remaining[
                [
                    "stock_code",
                    "period_key",
                    "rcept_no",
                    "income_selected_table",
                    "selected_column",
                    "range_resolution_reason",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n[Final status by report type]"
    )

    temp = final.copy()

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
                "income_ytd_final_status"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Final receipt audit: "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Range evidence     : "
        f"{OUT_RANGE}"
    )

    print(
        f"- Final review queue : "
        f"{OUT_REVIEW}"
    )

    print(
        "\n해석 원칙:"
        "\n- selected column 자체의 explicit duration range만 추가 PASS 근거로 사용"
        "\n- 9개월 explicit range는 Q3 YTD로 PASS"
        "\n- H1 6개월 explicit range는 PASS"
        "\n- endpoint-only / generic '3분기' / '2분기말'은 REVIEW 유지"
        "\n- 이 단계 후 PASS receipt만 production merge 후보"
    )


if __name__ == "__main__":
    main()
