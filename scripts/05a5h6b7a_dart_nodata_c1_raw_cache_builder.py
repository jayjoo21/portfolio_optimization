from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B7A. Non-corrected No-data C1 Raw Cache Builder
#
# 목적
# ------------------------------------------------------------
# 890개 same-receipt document를 C1 단계까지만 "마지막으로 한 번"
# 파싱하여 receipt별 parquet cache를 만든다.
#
# 이후 Basis / Period / C4 / C5 규칙을 수정할 때는
# document ZIP을 다시 열지 않고 이 cache만 사용한다.
#
# 왜 890건 전체인가?
# ------------------------------------------------------------
# H6B6A에서 기존 basis detector의 false-CFS 가능성을 확인했다.
# 따라서 기존 strict CFS 6/6 PASS 101건도 새 basis 규칙으로
# 한 번은 재검증해야 한다.
#
# 특징
# ------------------------------------------------------------
# - API 호출 없음
# - receipt별 즉시 저장
# - resume 가능
# - 이미 cache가 있으면 skip
# - 중간에 Ctrl+C해도 기존 cache 유지
# - 마지막에 combined parquet 생성
#
# INPUT
# ------------------------------------------------------------
# data/interim/dart/dart_noncorrected_full_api_no_data_scope.csv
# data/interim/dart/dart_noncorrected_nodata_source_manifest.parquet
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/nodata_c1_raw_cache/<rcept_no>.parquet
# data/interim/dart/dart_noncorrected_nodata_c1_cache_manifest.csv
# data/interim/dart/dart_noncorrected_nodata_c1_raw_cache.parquet
#
# 실행
# ------------------------------------------------------------
# 기본 5건 테스트:
# python scripts\05a5h6b7a_dart_nodata_c1_raw_cache_builder.py
#
# 전체:
# python scripts\05a5h6b7a_dart_nodata_c1_raw_cache_builder.py --full
#
# 일부:
# python scripts\05a5h6b7a_dart_nodata_c1_raw_cache_builder.py --limit 20
#
# combined parquet만 다시 만들기:
# python scripts\05a5h6b7a_dart_nodata_c1_raw_cache_builder.py --combine-only
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

SCOPE = (
    INTERIM
    / "dart_noncorrected_full_api_no_data_scope.csv"
)

SOURCE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_source_manifest.parquet"
)

CACHE_DIR = (
    INTERIM
    / "nodata_c1_raw_cache"
)

CACHE_MANIFEST = (
    INTERIM
    / "dart_noncorrected_nodata_c1_cache_manifest.csv"
)

COMBINED = (
    INTERIM
    / "dart_noncorrected_nodata_c1_raw_cache.parquet"
)


def load_module(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"module load failed: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def receipt_string(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(
            "string"
        )
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def first_existing(
    row: pd.Series,
    names: list[str],
):
    for name in names:
        if (
            name in row.index
            and pd.notna(
                row.get(
                    name
                )
            )
        ):
            return row.get(
                name
            )

    return None


def safe_stock_code(
    value,
) -> str:

    if pd.isna(
        value
    ):
        return ""

    text = str(
        value
    ).strip()

    if text.endswith(
        ".0"
    ):
        text = text[
            :-2
        ]

    return text.zfill(
        6
    )


def cache_path_for(
    receipt: str,
) -> Path:

    return (
        CACHE_DIR
        / f"{receipt}.parquet"
    )


def load_existing_manifest() -> pd.DataFrame:

    if not CACHE_MANIFEST.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        CACHE_MANIFEST,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    if (
        "rcept_no"
        in df.columns
    ):
        df[
            "rcept_no"
        ] = receipt_string(
            df[
                "rcept_no"
            ]
        )

    return df


def append_manifest_row(
    record: dict,
):

    existing = (
        load_existing_manifest()
    )

    new = pd.DataFrame(
        [
            record
        ]
    )

    if existing.empty:
        out = new

    else:
        # 최신 시도 1개만 남긴다.
        existing = existing.loc[
            ~existing[
                "rcept_no"
            ]
            .astype(str)
            .eq(
                str(
                    record[
                        "rcept_no"
                    ]
                )
            )
        ]

        out = pd.concat(
            [
                existing,
                new,
            ],
            ignore_index=True,
        )

    out.to_csv(
        CACHE_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )


def build_combined(
    expected_receipts: list[str],
):

    files = []

    missing = []

    for receipt in (
        expected_receipts
    ):

        p = cache_path_for(
            receipt
        )

        if p.exists():
            files.append(
                p
            )
        else:
            missing.append(
                receipt
            )

    print(
        "\n[Combine cache]"
    )

    print(
        f"Expected receipts : "
        f"{len(expected_receipts):,}"
    )

    print(
        f"Cached receipts   : "
        f"{len(files):,}"
    )

    print(
        f"Missing receipts  : "
        f"{len(missing):,}"
    )

    if missing:
        print(
            "Combined parquet은 "
            "모든 receipt cache가 생긴 뒤 생성합니다."
        )

        print(
            "Missing sample:",
            missing[
                :20
            ],
        )

        return

    frames = []

    for i, p in enumerate(
        files,
        start=1,
    ):
        frames.append(
            pd.read_parquet(
                p
            )
        )

        if (
            i % 100 == 0
            or i == len(
                files
            )
        ):
            print(
                f"  read "
                f"{i:,} / "
                f"{len(files):,}"
            )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined.to_parquet(
        COMBINED,
        index=False,
    )

    print(
        f"\nCombined rows: "
        f"{len(combined):,}"
    )

    print(
        f"Combined file: "
        f"{COMBINED}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="890건 전체 cache 구축",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트 처리 개수",
    )

    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="기존 receipt cache들을 combined parquet으로만 합침",
    )

    args = parser.parse_args()

    for p in [
        SCOPE,
        SOURCE_MANIFEST,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scope = pd.read_csv(
        SCOPE,
        dtype={
            "rcept_no":
            str,

            "stock_code":
            str,
        },
        low_memory=False,
    )

    manifest = pd.read_parquet(
        SOURCE_MANIFEST
    )

    scope[
        "rcept_no"
    ] = receipt_string(
        scope[
            "rcept_no"
        ]
    )

    manifest[
        "rcept_no"
    ] = receipt_string(
        manifest[
            "rcept_no"
        ]
    )

    # H6B1 manifest가 source path의 source-of-truth.
    source_cols = [
        c
        for c in [
            "rcept_no",
            "collection_status",
            "selected_source",
            "selected_path",
        ]
        if c in manifest.columns
    ]

    manifest_small = (
        manifest[
            source_cols
        ]
        .drop_duplicates(
            "rcept_no"
        )
        .copy()
    )

    rename = {}

    for c in [
        "collection_status",
        "selected_source",
        "selected_path",
    ]:
        if c in manifest_small.columns:
            rename[
                c
            ] = (
                f"h6b1_{c}"
            )

    manifest_small = (
        manifest_small.rename(
            columns=rename
        )
    )

    targets = scope.merge(
        manifest_small,
        on="rcept_no",
        how="left",
        validate="one_to_one",
    )

    selected_path_col = (
        "h6b1_selected_path"
        if "h6b1_selected_path"
        in targets.columns
        else None
    )

    if selected_path_col is None:
        raise RuntimeError(
            "H6B1 manifest에 selected_path가 없습니다."
        )

    all_receipts = (
        targets[
            "rcept_no"
        ]
        .astype(str)
        .tolist()
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "05A5-H6B7A C1 RAW CACHE BUILDER"
    )

    print(
        "=" * 110
    )

    print(
        f"\nAll targets: "
        f"{len(targets):,}"
    )

    if args.combine_only:
        build_combined(
            all_receipts
        )
        return

    if args.full:
        run_targets = (
            targets.copy()
        )

    elif (
        args.limit
        is not None
    ):
        run_targets = (
            targets.head(
                max(
                    args.limit,
                    0,
                )
            )
            .copy()
        )

    else:
        run_targets = (
            targets.head(
                5
            )
            .copy()
        )

    print(
        f"This run  : "
        f"{len(run_targets):,}"
    )

    c1 = load_module(
        "h6b7a_c1",
        SCRIPTS_DIR
        / "05a5c1_dart_pit_candidate_fact_extractor.py",
    )

    cached_before = 0
    parsed_now = 0
    errors = 0

    for idx, (
        _,
        target,
    ) in enumerate(
        run_targets.iterrows(),
        start=1,
    ):

        receipt = str(
            target[
                "rcept_no"
            ]
        )

        cache_path = (
            cache_path_for(
                receipt
            )
        )

        stock_code = (
            safe_stock_code(
                first_existing(
                    target,
                    [
                        "stock_code",
                        "stock_code_x",
                        "stock_code_y",
                    ],
                )
            )
        )

        period_key = (
            first_existing(
                target,
                [
                    "canonical_period_key",
                    "period_key",
                    "canonical_period",
                ],
            )
        )

        rcept_dt = (
            first_existing(
                target,
                [
                    "rcept_dt",
                    "rcept_no_date",
                ],
            )
        )

        print(
            f"\n[{idx}/{len(run_targets)}] "
            f"{stock_code} | "
            f"{period_key} | "
            f"{receipt}",
            flush=True,
        )

        if cache_path.exists():

            try:
                cached_rows = len(
                    pd.read_parquet(
                        cache_path,
                        columns=[
                            "rcept_no"
                        ],
                    )
                )
            except Exception:
                cached_rows = -1

            print(
                f"  CACHE HIT "
                f"| rows={cached_rows:,}",
                flush=True,
            )

            cached_before += 1

            continue

        selected_path = (
            target.get(
                selected_path_col
            )
        )

        if (
            pd.isna(
                selected_path
            )
            or not str(
                selected_path
            ).strip()
        ):
            print(
                "  ERROR: selected_path missing",
                flush=True,
            )

            append_manifest_row(
                {
                    "rcept_no":
                    receipt,

                    "stock_code":
                    stock_code,

                    "period_key":
                    period_key,

                    "cache_status":
                    "error",

                    "row_count":
                    0,

                    "elapsed_sec":
                    0.0,

                    "error":
                    "selected_path_missing",
                }
            )

            errors += 1

            continue

        zip_path = Path(
            str(
                selected_path
            )
        )

        started = (
            time.perf_counter()
        )

        try:

            c1.RAW_ROOT = (
                zip_path.parents[
                    2
                ]
            )

            print(
                f"  parsing "
                f"{zip_path.name} ...",
                flush=True,
            )

            raw = pd.DataFrame(
                c1.extract_document_candidates(
                    zip_path
                )
            )

            if raw.empty:
                raise RuntimeError(
                    "C1 returned zero rows"
                )

            # Canonical target metadata로 강제 덮어쓴다.
            raw[
                "stock_code"
            ] = stock_code

            raw[
                "period_key"
            ] = period_key

            raw[
                "rcept_no"
            ] = receipt

            raw[
                "rcept_dt"
            ] = pd.to_datetime(
                rcept_dt,
                errors="coerce",
            )

            raw[
                "cache_source_path"
            ] = str(
                zip_path
            )

            raw.to_parquet(
                cache_path,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            append_manifest_row(
                {
                    "rcept_no":
                    receipt,

                    "stock_code":
                    stock_code,

                    "period_key":
                    period_key,

                    "cache_status":
                    "cached",

                    "row_count":
                    len(
                        raw
                    ),

                    "elapsed_sec":
                    round(
                        elapsed,
                        4,
                    ),

                    "error":
                    "",
                }
            )

            print(
                f"  DONE "
                f"| rows={len(raw):,} "
                f"| {elapsed:.2f}s",
                flush=True,
            )

            parsed_now += 1

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            msg = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            append_manifest_row(
                {
                    "rcept_no":
                    receipt,

                    "stock_code":
                    stock_code,

                    "period_key":
                    period_key,

                    "cache_status":
                    "error",

                    "row_count":
                    0,

                    "elapsed_sec":
                    round(
                        elapsed,
                        4,
                    ),

                    "error":
                    msg,
                }
            )

            print(
                f"  ERROR "
                f"| {msg}",
                flush=True,
            )

            errors += 1

    cached_total = sum(
        1
        for receipt
        in all_receipts
        if cache_path_for(
            receipt
        ).exists()
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "H6B7A CACHE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"\nCached before this run : "
        f"{cached_before:,}"
    )

    print(
        f"Parsed this run        : "
        f"{parsed_now:,}"
    )

    print(
        f"Errors                 : "
        f"{errors:,}"
    )

    print(
        f"Total cached / expected: "
        f"{cached_total:,} / "
        f"{len(all_receipts):,}"
    )

    print(
        f"\nCache dir      : "
        f"{CACHE_DIR}"
    )

    print(
        f"Cache manifest : "
        f"{CACHE_MANIFEST}"
    )

    if (
        cached_total
        == len(
            all_receipts
        )
    ):
        build_combined(
            all_receipts
        )

    else:
        print(
            "\n아직 모든 receipt가 cache되지 않았으므로 "
            "combined parquet은 만들지 않았습니다."
        )

        print(
            "전체 실행 후 --combine-only 또는 "
            "--full을 다시 실행하면 "
            "기존 cache는 skip됩니다."
        )


if __name__ == "__main__":
    main()
