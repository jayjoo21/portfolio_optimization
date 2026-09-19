from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B7E. Selected-Column ↔ Context Period Consistency Audit
#
# 목적
# ------------------------------------------------------------
# H6B7D에서 PASS_MAIN_STATEMENT_* 로 통과한 후보라도,
# heading_context 안에 여러 회계기간이 동시에 적혀 있을 수 있다.
#
# 예:
#   제20기 2015.01.01 ~ 2015.12.31
#   제21기 2016.01.01 ~ 2016.12.31
#   제22기 2017.01.01 ~ 2017.12.31
#
# target = 2017.12 인데 selected_column = 제21기 라면
# 표 전체에는 2017.12가 존재하더라도 선택 열은 전기이므로 BLOCK.
#
# 이 단계는:
#   selected_column의 기수
#        ↕
#   heading_context의 "기수 -> 종료기간" mapping
# 을 비교한다.
#
# IMPORTANT
# ------------------------------------------------------------
# - document ZIP 재파싱 없음
# - H6B7D PASS 후보만 정밀검증
# - mapping 불충분하면 자동 승인하지 않고 REVIEW
# - final CFS/OFS selector는 아직 아님
#
# 실행:
# python scripts\05a5h6b7e_dart_selected_column_context_consistency.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_guarded.parquet"
)

OUTPUT = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_column_consistent.parquet"
)

SUMMARY = (
    INTERIM
    / "dart_noncorrected_nodata_period_v3_column_consistency_summary.csv"
)


GEN_PATTERN = re.compile(
    r"제\s*(\d+)\s*"
    r"(?:\(\s*(?:당|전)\s*\)\s*)?"
    r"기(?:말)?",
    re.IGNORECASE,
)


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def parse_target(period_key: str):
    text = clean(period_key)

    m = re.search(
        r"\|(?P<year>\d{4})\.(?P<month>\d{2})$",
        text,
    )

    if not m:
        return None, None

    return (
        int(m.group("year")),
        int(m.group("month")),
    )


def parse_generation(value):
    text = clean(value)

    m = GEN_PATTERN.search(
        text
    )

    if not m:
        return None

    return int(
        m.group(1)
    )


def extract_year_months(text: str):
    """
    텍스트에 나타나는 날짜를 등장 순서대로 추출.
    income statement:
      2017년 1월 1일부터 2017년 12월 31일까지
    에서는 마지막 날짜인 2017.12가 period endpoint.
    """

    found = []

    token_pattern = re.compile(
        r"(?<!\d)"
        r"(20\d{2})"
        r"(?:\s*년\s*|[./\-])"
        r"(\d{1,2})"
        r"(?:\s*월|(?=[./\-]|\D|$))"
    )

    for m in token_pattern.finditer(
        text
    ):
        try:
            year = int(
                m.group(1)
            )

            month = int(
                m.group(2)
            )

            if 1 <= month <= 12:
                found.append(
                    (
                        year,
                        month,
                    )
                )
        except Exception:
            pass

    return found


def generation_period_map(
    heading_context,
):
    """
    heading_context를 generation별 segment로 나누고,
    각 segment에서 마지막 year-month를 endpoint로 사용한다.

    반환 예:
      {
        20: (2015, 12),
        21: (2016, 12),
        22: (2017, 12)
      }
    """

    text = clean(
        heading_context
    )

    matches = list(
        GEN_PATTERN.finditer(
            text
        )
    )

    mapping = {}
    evidence = {}

    for i, match in enumerate(
        matches
    ):
        generation = int(
            match.group(1)
        )

        start = match.end()

        end = (
            matches[
                i + 1
            ].start()
            if i + 1 < len(
                matches
            )
            else len(
                text
            )
        )

        segment = text[
            start:end
        ]

        dates = (
            extract_year_months(
                segment
            )
        )

        if dates:
            endpoint = dates[
                -1
            ]

            # 동일 generation이 반복될 경우 target-like 최신 endpoint를
            # 무작정 덮지 않고 evidence에 모두 남긴다.
            evidence.setdefault(
                generation,
                [],
            ).append(
                {
                    "endpoint":
                    endpoint,

                    "segment":
                    segment[
                        :300
                    ],
                }
            )

    for generation, items in (
        evidence.items()
    ):
        endpoints = [
            tuple(
                item[
                    "endpoint"
                ]
            )
            for item in items
        ]

        unique = list(
            dict.fromkeys(
                endpoints
            )
        )

        if len(
            unique
        ) == 1:
            mapping[
                generation
            ] = unique[
                0
            ]

    return (
        mapping,
        evidence,
    )


def classify(
    row: pd.Series,
) -> pd.Series:

    guard_class = str(
        row.get(
            "period_guardrail_class",
            "",
        )
    )

    # H6B7D에서 자동 PASS가 아니면 현 상태 유지.
    if not guard_class.startswith(
        "PASS_"
    ):
        return pd.Series(
            {
                "period_column_gate":
                "NOT_APPLICABLE",

                "period_column_selected_generation":
                None,

                "period_column_target_generations":
                "",

                "period_column_selected_endpoint":
                "",

                "period_column_generation_map":
                "{}",

                "period_final_status":
                row.get(
                    "period_guarded_status"
                ),

                "period_final_confidence":
                row.get(
                    "period_guarded_confidence"
                ),

                "period_final_reason":
                row.get(
                    "period_guarded_reason",
                    "",
                ),
            }
        )

    target_year, target_month = (
        parse_target(
            row.get(
                "period_key",
                "",
            )
        )
    )

    selected_col = (
        row.get(
            "period_v2_selected_column"
        )
    )

    selected_generation = (
        parse_generation(
            selected_col
        )
    )

    mapping, evidence = (
        generation_period_map(
            row.get(
                "heading_context",
                "",
            )
        )
    )

    target_generations = [
        generation
        for generation, endpoint
        in mapping.items()
        if (
            target_year is not None
            and target_month is not None
            and endpoint
            == (
                target_year,
                target_month,
            )
        )
    ]

    selected_endpoint = (
        mapping.get(
            selected_generation
        )
        if selected_generation
        is not None
        else None
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        selected_generation
        is not None
        and selected_endpoint
        == (
            target_year,
            target_month,
        )
    ):
        gate = (
            "PASS_SELECTED_COLUMN_TARGET_PERIOD"
        )

        final_status = (
            "selected_context_column_confirmed"
        )

        final_conf = "HIGH"

        reason = (
            "selected_generation_maps_to_target_period"
        )

    elif (
        selected_generation
        is not None
        and selected_endpoint
        is not None
        and selected_endpoint
        != (
            target_year,
            target_month,
        )
        and target_generations
    ):
        gate = (
            "BLOCK_SELECTED_COLUMN_PRIOR_OR_OTHER_PERIOD"
        )

        final_status = (
            "low_confidence_candidate"
        )

        final_conf = "LOW"

        reason = (
            "selected_generation_maps_to_non_target_period"
        )

    elif (
        selected_generation
        is not None
        and target_generations
        and selected_generation
        not in target_generations
    ):
        # selected generation mapping 자체는 불완전하더라도
        # context에서 target generation이 따로 명확히 보이면 보수적으로 차단.
        gate = (
            "BLOCK_SELECTED_GENERATION_NOT_TARGET"
        )

        final_status = (
            "low_confidence_candidate"
        )

        final_conf = "LOW"

        reason = (
            "target_generation_exists_but_selected_generation_differs"
        )

    else:
        gate = (
            "REVIEW_GENERATION_MAPPING_INSUFFICIENT"
        )

        final_status = (
            "low_confidence_candidate"
        )

        final_conf = "LOW"

        reason = (
            "main_statement_date_match_but_selected_column_mapping_unresolved"
        )

    mapping_json = {
        str(k):
        f"{v[0]:04d}.{v[1]:02d}"
        for k, v in sorted(
            mapping.items()
        )
    }

    return pd.Series(
        {
            "period_column_gate":
            gate,

            "period_column_selected_generation":
            selected_generation,

            "period_column_target_generations":
            "|".join(
                str(x)
                for x in sorted(
                    target_generations
                )
            ),

            "period_column_selected_endpoint":
            (
                f"{selected_endpoint[0]:04d}."
                f"{selected_endpoint[1]:02d}"
                if selected_endpoint
                is not None
                else ""
            ),

            "period_column_generation_map":
            json.dumps(
                mapping_json,
                ensure_ascii=False,
            ),

            "period_final_status":
            final_status,

            "period_final_confidence":
            final_conf,

            "period_final_reason":
            reason,
        }
    )


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
        "05A5-H6B7E SELECTED-COLUMN / CONTEXT PERIOD CONSISTENCY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    audit = df.apply(
        classify,
        axis=1,
    )

    out = pd.concat(
        [
            df,
            audit,
        ],
        axis=1,
    )

    out.to_parquet(
        OUTPUT,
        index=False,
    )

    passed_guardrail = out.loc[
        out[
            "period_guardrail_class"
        ]
        .astype(str)
        .str.startswith(
            "PASS_",
            na=False,
        )
    ].copy()

    print(
        "\n[H6B7D PASS rows]"
    )

    print(
        f"{len(passed_guardrail):,}"
    )

    print(
        "\n[Selected-column consistency gate]"
    )

    print(
        passed_guardrail[
            "period_column_gate"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n[Final period status after column consistency]"
    )

    print(
        out[
            "period_final_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    blocked = passed_guardrail.loc[
        passed_guardrail[
            "period_column_gate"
        ]
        .astype(str)
        .str.startswith(
            "BLOCK_",
            na=False,
        )
    ].copy()

    confirmed = passed_guardrail.loc[
        passed_guardrail[
            "period_column_gate"
        ].eq(
            "PASS_SELECTED_COLUMN_TARGET_PERIOD"
        )
    ].copy()

    review = passed_guardrail.loc[
        passed_guardrail[
            "period_column_gate"
        ].eq(
            "REVIEW_GENERATION_MAPPING_INSUFFICIENT"
        )
    ].copy()

    print(
        "\n[Confirmed selected-column target period]"
    )

    print(
        f"{len(confirmed):,}"
    )

    print(
        "\n[Blocked selected-column mismatch]"
    )

    print(
        f"{len(blocked):,}"
    )

    print(
        "\n[Review: mapping insufficient]"
    )

    print(
        f"{len(review):,}"
    )

    show_cols = [
        c
        for c in [
            "stock_code",
            "period_key",
            "rcept_no",
            "account_family",
            "table_index",
            "row_index",
            "basis_v2",
            "period_v2_selected_column",
            "period_column_selected_generation",
            "period_column_target_generations",
            "period_column_selected_endpoint",
            "period_column_generation_map",
            "period_column_gate",
            "period_final_status",
            "heading_context",
        ]
        if c in out.columns
    ]

    print(
        "\n[Blocked sample]"
    )

    if blocked.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            50,
        ):
            print(
                blocked[
                    show_cols
                ]
                .head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Confirmed sample]"
    )

    if confirmed.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            40,
        ):
            print(
                confirmed[
                    show_cols
                ]
                .head(
                    25
                )
                .to_string(
                    index=False
                )
            )

    # Explicitly inspect the discovered 한국토지신탁 row.
    focus = out.loc[
        out[
            "rcept_no"
        ]
        .astype(
            "string"
        )
        .eq(
            "20180402002531"
        )
        & out[
            "table_index"
        ]
        .eq(
            258
        )
    ].copy()

    print(
        "\n[Focus: 한국토지신탁 table 258]"
    )

    if focus.empty:
        print(
            "None"
        )
    else:
        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            320,
            "display.max_rows",
            50,
        ):
            print(
                focus[
                    show_cols
                ]
                .to_string(
                    index=False
                )
            )

    summary = (
        passed_guardrail.groupby(
            [
                "period_column_gate",
                "period_final_status",
                "period_final_confidence",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "row_count"
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nOutput : "
        f"{OUTPUT}"
    )

    print(
        f"Summary: "
        f"{SUMMARY}"
    )

    print(
        "\n판단 기준:"
        "\n- 표 전체에 target 날짜가 있다는 것만으로는 부족"
        "\n- selected_column의 기수가 target endpoint에 직접 연결돼야 PASS"
        "\n- selected_column이 전기/다른 기수면 BLOCK"
        "\n- mapping을 만들 수 없으면 REVIEW"
        "\n- 이 검증 뒤에 receipt × metric Selector V2로 진행"
    )


if __name__ == "__main__":
    main()
