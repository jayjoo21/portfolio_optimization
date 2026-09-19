from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

INPUT = INTERIM / "dart_noncorrected_nodata_c4_v2_all.parquet"
OUTPUT = INTERIM / "dart_noncorrected_nodata_period_v3_context_audit.parquet"
SUMMARY_CSV = INTERIM / "dart_noncorrected_nodata_period_v3_context_summary.csv"


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_target(period_key: str):
    text = clean_text(period_key)
    m = re.search(r"\|(?P<year>\d{4})\.(?P<month>\d{2})$", text)
    if not m:
        return None, None
    return int(m.group("year")), int(m.group("month"))


def context_period_evidence(context, target_year, target_month):
    if target_year is None or target_month is None:
        return {
            "target_ym_in_context": False,
            "target_date_in_context": False,
            "other_year_months": [],
        }

    text = clean_text(context)
    compact = re.sub(r"\s+", "", text)

    ym_patterns = [
        rf"(?<!\d){target_year}\.{target_month:02d}(?!\d)",
        rf"(?<!\d){target_year}-{target_month:02d}(?!\d)",
        rf"(?<!\d){target_year}/{target_month:02d}(?!\d)",
        rf"(?<!\d){target_year}년{target_month:02d}월",
        rf"(?<!\d){target_year}년{target_month}월",
    ]

    date_patterns = [
        rf"(?<!\d){target_year}\.{target_month:02d}\.\d{{1,2}}(?!\d)",
        rf"(?<!\d){target_year}-{target_month:02d}-\d{{1,2}}(?!\d)",
        rf"(?<!\d){target_year}/{target_month:02d}/\d{{1,2}}(?!\d)",
        rf"(?<!\d){target_year}년{target_month:02d}월\d{{1,2}}일",
        rf"(?<!\d){target_year}년{target_month}월\d{{1,2}}일",
    ]

    ym_match = any(re.search(p, compact) for p in ym_patterns)
    date_match = any(re.search(p, compact) for p in date_patterns)

    found = []
    patterns = [
        r"(?<!\d)(20\d{2})[.\-/](\d{1,2})(?:[.\-/]\d{1,2})?",
        r"(?<!\d)(20\d{2})년\s*(\d{1,2})월",
    ]

    for pattern in patterns:
        for y, m in re.findall(pattern, text):
            try:
                pair = (int(y), int(m))
                if pair not in found:
                    found.append(pair)
            except Exception:
                pass

    other = [
        f"{y:04d}.{m:02d}"
        for y, m in found
        if y != target_year or m != target_month
    ]

    return {
        "target_ym_in_context": ym_match,
        "target_date_in_context": date_match,
        "other_year_months": other,
    }


def classify_row(row: pd.Series) -> pd.Series:
    old_status = str(row.get("period_v2_status", ""))
    old_conf = str(row.get("period_v2_confidence", ""))

    target_year, target_month = parse_target(row.get("period_key", ""))

    evidence = context_period_evidence(
        row.get("heading_context", ""),
        target_year,
        target_month,
    )

    new_status = old_status
    new_conf = old_conf

    base_reason = row.get("period_v2_reasons", "")
    reason = "" if pd.isna(base_reason) else str(base_reason)

    if old_status == "low_confidence_candidate":
        if evidence["target_date_in_context"]:
            new_status = "selected_context_confirmed"
            new_conf = "HIGH"
            reason += (" | " if reason else "") + "heading_exact_target_date"

        elif evidence["target_ym_in_context"]:
            new_status = "selected_context_confirmed"
            new_conf = "MEDIUM"
            reason += (" | " if reason else "") + "heading_target_year_month"

    return pd.Series(
        {
            "period_v3_status": new_status,
            "period_v3_confidence": new_conf,
            "period_v3_reasons": reason,
            "period_v3_target_year": target_year,
            "period_v3_target_month": target_month,
            "period_v3_target_ym_in_context": evidence["target_ym_in_context"],
            "period_v3_target_date_in_context": evidence["target_date_in_context"],
            "period_v3_other_year_months": "|".join(evidence["other_year_months"]),
        }
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_parquet(INPUT)

    print("\n" + "=" * 115)
    print("05A5-H6B7C PERIOD V3 CONTEXT-DATE AUDIT")
    print("=" * 115)
    print(f"\nRows: {len(df):,}")

    v3 = df.apply(classify_row, axis=1)

    out = pd.concat([df, v3], axis=1)
    out.to_parquet(OUTPUT, index=False)

    print("\n[V2 status]")
    print(out["period_v2_status"].value_counts(dropna=False).to_string())

    print("\n[V3 status]")
    print(out["period_v3_status"].value_counts(dropna=False).to_string())

    upgraded = out.loc[
        out["period_v2_status"].eq("low_confidence_candidate")
        & out["period_v3_status"].eq("selected_context_confirmed")
    ].copy()

    print("\n[LOW -> context-confirmed]")
    print(f"{len(upgraded):,}")

    print("\n[Upgrade confidence]")
    if upgraded.empty:
        print("None")
    else:
        print(
            upgraded["period_v3_confidence"]
            .value_counts(dropna=False)
            .to_string()
        )

    quarter_nonstandard_month = out.loc[
        out["period_key"].astype("string").str.startswith("분기보고서|", na=False)
        & ~out["period_v3_target_month"].isin([3, 9])
    ].copy()

    print("\n[Quarter reports whose period month is not Mar/Sep]")
    print(f"Rows     : {len(quarter_nonstandard_month):,}")
    print(f"Receipts : {quarter_nonstandard_month['rcept_no'].nunique():,}")

    if not quarter_nonstandard_month.empty:
        print("\n[V3 status in those rows]")
        print(
            quarter_nonstandard_month["period_v3_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    show_cols = [
        c for c in [
            "stock_code",
            "period_key",
            "rcept_no",
            "account_family",
            "table_index",
            "row_index",
            "selected_column",
            "period_v2_status",
            "period_v2_confidence",
            "period_v2_selected_column",
            "period_v3_status",
            "period_v3_confidence",
            "period_v3_reasons",
            "heading_context",
        ]
        if c in out.columns
    ]

    print("\n[Context-confirmed sample]")
    if upgraded.empty:
        print("None")
    else:
        with pd.option_context(
            "display.max_colwidth", 180,
            "display.width", 300,
            "display.max_rows", 50,
        ):
            print(
                upgraded[show_cols]
                .head(30)
                .to_string(index=False)
            )

    special = out.loc[
        out["period_key"].astype("string").eq("분기보고서|2017.12")
    ].copy()

    print("\n[분기보고서|2017.12 sample]")
    if special.empty:
        print("None")
    else:
        with pd.option_context(
            "display.max_colwidth", 180,
            "display.width", 300,
            "display.max_rows", 40,
        ):
            print(
                special[show_cols]
                .head(25)
                .to_string(index=False)
            )

    summary = (
        out.groupby(
            [
                "period_v2_status",
                "period_v3_status",
                "period_v3_confidence",
            ],
            dropna=False,
        )
        .size()
        .rename("row_count")
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nOutput : {OUTPUT}")
    print(f"Summary: {SUMMARY_CSV}")

    print(
        "\n판단 기준:"
        "\n- LOW + heading exact target date -> context-confirmed HIGH"
        "\n- LOW + heading target year/month -> context-confirmed MEDIUM"
        "\n- ambiguous_top_candidates는 그대로 보류"
        "\n- no_current_candidate는 그대로 보류"
        "\n- 달력 월만으로 Q1/Q3를 강제하지 않음"
    )


if __name__ == "__main__":
    main()
