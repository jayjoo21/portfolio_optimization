from __future__ import annotations

import runpy
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLLECTOR_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "05c4a_krx_derivatives_collect_features.py"
)

if not COLLECTOR_PATH.exists():
    raise FileNotFoundError(
        f"기존 collector가 없습니다: {COLLECTOR_PATH}"
    )

m = runpy.run_path(str(COLLECTOR_PATH))

CHECKPOINT_PATH = m["CHECKPOINT_PATH"]
CLEAN_PATH = m["CLEAN_PATH"]
FEATURE_PATH = m["FEATURE_PATH"]
STOCK_PANEL_PATH = m["STOCK_PANEL_PATH"]

request_krx = m["request_krx"]
aggregate_futures = m["aggregate_futures"]
aggregate_options = m["aggregate_options"]
build_features = m["build_features"]
validate = m["validate"]

FUTURES_ENDPOINT = m["FUTURES_ENDPOINT"]
OPTIONS_ENDPOINT = m["OPTIONS_ENDPOINT"]
REQUEST_SLEEP_SECONDS = m["REQUEST_SLEEP_SECONDS"]


def print_stock_panel_coverage() -> None:
    panel = pd.read_parquet(STOCK_PANEL_PATH)

    date_candidates = [
        "date", "Date", "BAS_DD", "bas_dd", "trade_date"
    ]

    date_col = next(
        (col for col in date_candidates if col in panel.columns),
        None,
    )

    if date_col is None:
        print("\n[Stock panel coverage] date column을 찾지 못했습니다.")
        return

    dates = pd.to_datetime(
        panel[date_col],
        errors="coerce",
    ).dropna()

    print("\n[Stock panel coverage]")
    print("start:", dates.min())
    print("end  :", dates.max())
    print("unique trading dates:", dates.nunique())


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"checkpoint가 없습니다: {CHECKPOINT_PATH}"
        )

    print_stock_panel_coverage()

    df = pd.read_parquet(CHECKPOINT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    error_mask = (
        df["collection_status"]
        .fillna("error")
        .ne("ok")
    )
    errors = df.loc[error_mask].copy()

    print("\n" + "=" * 80)
    print("FAILED DATES")
    print("=" * 80)

    if errors.empty:
        print("재시도할 error row가 없습니다.")
    else:
        show_cols = [
            col
            for col in ["date", "collection_status", "error"]
            if col in errors.columns
        ]
        print(
            errors[show_cols]
            .sort_values("date")
            .to_string(index=False)
        )

    session = requests.Session()
    repaired_rows = []

    for i, row in enumerate(
        errors.sort_values("date").itertuples(index=False),
        start=1,
    ):
        date = pd.Timestamp(getattr(row, "date"))
        bas_dd = date.strftime("%Y%m%d")

        print(f"\n[{i}/{len(errors)}] retry {bas_dd}")

        try:
            fut_records = request_krx(
                session,
                FUTURES_ENDPOINT,
                bas_dd,
            )

            time.sleep(REQUEST_SLEEP_SECONDS)

            opt_records = request_krx(
                session,
                OPTIONS_ENDPOINT,
                bas_dd,
            )

            fut = aggregate_futures(
                bas_dd,
                fut_records,
            )

            opt = aggregate_options(
                bas_dd,
                opt_records,
            )

            repaired = {
                **fut,
                **{
                    key: value
                    for key, value in opt.items()
                    if key != "date"
                },
                "collection_status": "ok",
                "error": None,
            }

            repaired_rows.append(repaired)
            print("OK")

        except Exception as exc:
            repaired_rows.append(
                {
                    "date": date,
                    "collection_status": "error",
                    "error": repr(exc),
                }
            )
            print("FAILED:", repr(exc))

        time.sleep(REQUEST_SLEEP_SECONDS)

    if repaired_rows:
        repaired_df = pd.DataFrame(repaired_rows)
        repaired_df["date"] = pd.to_datetime(repaired_df["date"])

        error_dates = set(errors["date"])

        base = df.loc[
            ~df["date"].isin(error_dates)
        ].copy()

        df = pd.concat(
            [base, repaired_df],
            ignore_index=True,
            sort=False,
        )

    df = (
        df.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df.to_parquet(CHECKPOINT_PATH, index=False)
    df.to_parquet(CLEAN_PATH, index=False)

    features = build_features(df)
    features.to_parquet(FEATURE_PATH, index=False)

    validate(df, features)

    print("\n" + "=" * 80)
    print("REPAIR SUMMARY")
    print("=" * 80)
    print(
        df["collection_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    remaining = df.loc[
        df["collection_status"]
        .fillna("error")
        .ne("ok"),
        [
            col
            for col in ["date", "collection_status", "error"]
            if col in df.columns
        ],
    ]

    print("\n[Remaining errors]")
    if remaining.empty:
        print("0")
    else:
        print(remaining.to_string(index=False))

    print(f"\ncheckpoint: {CHECKPOINT_PATH}")
    print(f"clean     : {CLEAN_PATH}")
    print(f"features  : {FEATURE_PATH}")


if __name__ == "__main__":
    main()
