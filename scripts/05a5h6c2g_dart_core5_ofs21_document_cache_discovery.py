from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 05A5-H6C2G. OFS21 DOCUMENT CACHE / SOURCE DISCOVERY
#
# 목적
# ------------------------------------------------------------
# H6C2F에서 document-level no-CFS proof가 필요한 OFS 21 receipt에 대해
# 이미 프로젝트 안에 exact-receipt 문서/원문 캐시가 존재하는지 먼저 찾는다.
#
# 이 단계에서는:
# - API 호출 없음
# - DART 다운로드 없음
# - HTML/XML/ZIP 파싱 없음
# - 값 복구/merge 없음
#
# 찾는 것:
# 1) 파일명/경로에 rcept_no가 포함된 raw/interim 파일
# 2) parquet/csv manifest 안에서 rcept_no와 path-like column 연결
# 3) receipt별 CACHE_FOUND / MANIFEST_PATH_FOUND / NOT_FOUND
#
# 실행:
# python scripts\05a5h6c2g_dart_core5_ofs21_document_cache_discovery.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DART = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
)

INTERIM_DART = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

OFS21_AUDIT = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_fallback_proof_audit.csv"
)

OUT_RECEIPT = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_document_cache_discovery.csv"
)

OUT_HITS = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_document_cache_hits.csv"
)

OUT_COLLECTION_TARGETS = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_document_collection_targets.csv"
)


EXPECTED_OFS = 21


# ============================================================
# Helpers
# ============================================================


def normalize_receipt(
    series: pd.Series,
) -> pd.Series:

    return (
        series.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def clean(
    value,
) -> str:

    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip()


def safe_relative(
    path: Path,
) -> str:

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except Exception:
        return str(
            path
        )


PATH_COLUMN_PATTERN = re.compile(
    r"(?:"
    r"path|file|filename|filepath|"
    r"source|cache|local|download|"
    r"html|xml|zip|document|doc|"
    r"content|raw"
    r")",
    flags=re.IGNORECASE,
)


DOCUMENT_EXTENSIONS = {
    ".html",
    ".htm",
    ".xml",
    ".xhtml",
    ".txt",
    ".zip",
    ".json",
    ".csv",
    ".parquet",
}


def looks_like_existing_path(
    text: str,
) -> tuple[
    bool,
    str,
]:

    if not text:
        return (
            False,
            "",
        )

    candidate = Path(
        text
    )

    # Absolute path in manifest.
    if candidate.is_absolute():

        return (
            candidate.exists(),
            str(
                candidate
            ),
        )

    # Relative to project root.
    project_candidate = (
        PROJECT_ROOT
        / candidate
    )

    if project_candidate.exists():

        return (
            True,
            str(
                project_candidate
            ),
        )

    # Relative to data/raw/dart.
    raw_candidate = (
        RAW_DART
        / candidate
    )

    if raw_candidate.exists():

        return (
            True,
            str(
                raw_candidate
            ),
        )

    # Relative to data/interim/dart.
    interim_candidate = (
        INTERIM_DART
        / candidate
    )

    if interim_candidate.exists():

        return (
            True,
            str(
                interim_candidate
            ),
        )

    return (
        False,
        "",
    )


# ============================================================
# Filesystem filename scan
# ============================================================


def scan_filesystem_by_receipt(
    receipts: set[str],
) -> list[dict]:

    records = []

    roots = [
        RAW_DART,
        INTERIM_DART,
    ]

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            suffix = (
                path.suffix.lower()
            )

            if suffix not in (
                DOCUMENT_EXTENSIONS
            ):
                continue

            path_text = str(
                path
            )

            name_text = (
                path.name
            )

            matched = [
                rcept_no
                for rcept_no
                in receipts
                if (
                    rcept_no
                    in name_text
                    or rcept_no
                    in path_text
                )
            ]

            for rcept_no in matched:

                records.append(
                    {
                        "rcept_no":
                        rcept_no,

                        "hit_type":
                        "FILESYSTEM_FILENAME_MATCH",

                        "source_container":
                        safe_relative(
                            root
                        ),

                        "manifest_file":
                        "",

                        "manifest_column":
                        "",

                        "manifest_value":
                        "",

                        "resolved_existing_path":
                        str(
                            path
                        ),

                        "file_suffix":
                        suffix,

                        "file_size_bytes":
                        path.stat().st_size,
                    }
                )

    return records


# ============================================================
# Manifest schema/content scan
# ============================================================


def candidate_manifest_files() -> list[Path]:

    files = []

    for root in [
        RAW_DART,
        INTERIM_DART,
    ]:

        if not root.exists():
            continue

        files.extend(
            root.glob(
                "*.parquet"
            )
        )

        files.extend(
            root.glob(
                "*.csv"
            )
        )

    # Unique sorted paths.
    unique = sorted(
        {
            path.resolve():
            path
            for path in files
        }.values(),
        key=lambda p:
        str(
            p
        ),
    )

    return unique


def inspect_parquet_manifest(
    path: Path,
    receipts: set[str],
) -> list[dict]:

    try:
        schema_names = (
            pq.ParquetFile(
                path
            )
            .schema_arrow
            .names
        )

    except Exception:
        return []

    if "rcept_no" not in schema_names:
        return []

    path_cols = [
        col
        for col in schema_names
        if (
            col != "rcept_no"
            and PATH_COLUMN_PATTERN.search(
                str(
                    col
                )
            )
        )
    ]

    if not path_cols:
        return []

    read_cols = [
        "rcept_no",
    ] + path_cols

    try:
        frame = pd.read_parquet(
            path,
            columns=read_cols,
        )

    except Exception:
        return []

    return inspect_manifest_frame(
        path=path,
        frame=frame,
        path_cols=path_cols,
        receipts=receipts,
    )


def inspect_csv_manifest(
    path: Path,
    receipts: set[str],
) -> list[dict]:

    try:
        head = pd.read_csv(
            path,
            nrows=5,
            low_memory=False,
        )

    except Exception:
        return []

    if "rcept_no" not in head.columns:
        return []

    path_cols = [
        col
        for col in head.columns
        if (
            col != "rcept_no"
            and PATH_COLUMN_PATTERN.search(
                str(
                    col
                )
            )
        )
    ]

    if not path_cols:
        return []

    try:
        frame = pd.read_csv(
            path,
            usecols=[
                "rcept_no",
            ] + path_cols,
            dtype={
                "rcept_no":
                str,
            },
            low_memory=False,
        )

    except Exception:
        return []

    return inspect_manifest_frame(
        path=path,
        frame=frame,
        path_cols=path_cols,
        receipts=receipts,
    )


def inspect_manifest_frame(
    path: Path,
    frame: pd.DataFrame,
    path_cols: list[str],
    receipts: set[str],
) -> list[dict]:

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
            receipts
        )
    ].copy()

    if frame.empty:
        return []

    records = []

    for _, row in frame.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        for col in path_cols:

            value = clean(
                row.get(
                    col,
                    "",
                )
            )

            if not value:
                continue

            (
                exists,
                resolved_path,
            ) = looks_like_existing_path(
                value
            )

            # Keep all path-like manifest evidence, even when path
            # no longer exists, because it is useful provenance.
            records.append(
                {
                    "rcept_no":
                    rcept_no,

                    "hit_type":
                    (
                        "MANIFEST_PATH_EXISTS"
                        if exists
                        else "MANIFEST_PATH_REFERENCE_ONLY"
                    ),

                    "source_container":
                    "",

                    "manifest_file":
                    safe_relative(
                        path
                    ),

                    "manifest_column":
                    col,

                    "manifest_value":
                    value,

                    "resolved_existing_path":
                    (
                        resolved_path
                        if exists
                        else ""
                    ),

                    "file_suffix":
                    (
                        Path(
                            resolved_path
                        ).suffix.lower()
                        if exists
                        else ""
                    ),

                    "file_size_bytes":
                    (
                        Path(
                            resolved_path
                        ).stat().st_size
                        if exists
                        else np.nan
                    ),
                }
            )

    return records


# ============================================================
# Main
# ============================================================


def main():

    if not OFS21_AUDIT.exists():
        raise FileNotFoundError(
            OFS21_AUDIT
        )

    audit = pd.read_csv(
        OFS21_AUDIT,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    audit[
        "rcept_no"
    ] = normalize_receipt(
        audit[
            "rcept_no"
        ]
    )

    if len(
        audit
    ) != EXPECTED_OFS:
        raise RuntimeError(
            f"Expected {EXPECTED_OFS} OFS receipts, "
            f"found {len(audit)}."
        )

    receipts = set(
        audit[
            "rcept_no"
        ]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2G OFS21 DOCUMENT CACHE / SOURCE DISCOVERY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nOFS receipts: "
        f"{len(receipts):,}"
    )

    # --------------------------------------------------------
    # 1. Direct filesystem filename/path scan
    # --------------------------------------------------------

    hits = scan_filesystem_by_receipt(
        receipts
    )

    print(
        "\n[Direct filesystem receipt-number hits]"
    )

    direct_count = len(
        {
            row[
                "rcept_no"
            ]
            for row in hits
        }
    )

    print(
        f"receipts with direct hit: "
        f"{direct_count:,}"
    )

    print(
        f"total direct-hit files : "
        f"{len(hits):,}"
    )

    # --------------------------------------------------------
    # 2. Manifest discovery
    # --------------------------------------------------------

    manifest_files = (
        candidate_manifest_files()
    )

    print(
        "\n[Manifest files inspected]"
    )

    print(
        len(
            manifest_files
        )
    )

    for path in manifest_files:

        if path.suffix.lower() == ".parquet":

            hits.extend(
                inspect_parquet_manifest(
                    path,
                    receipts,
                )
            )

        elif path.suffix.lower() == ".csv":

            hits.extend(
                inspect_csv_manifest(
                    path,
                    receipts,
                )
            )

    hits_df = pd.DataFrame(
        hits
    )

    if hits_df.empty:

        hits_df = pd.DataFrame(
            columns=[
                "rcept_no",
                "hit_type",
                "source_container",
                "manifest_file",
                "manifest_column",
                "manifest_value",
                "resolved_existing_path",
                "file_suffix",
                "file_size_bytes",
            ]
        )

    # Remove exact duplicate evidence rows.
    hits_df = hits_df.drop_duplicates()

    hits_df.to_csv(
        OUT_HITS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt-level summary
    # --------------------------------------------------------

    receipt_records = []

    for _, row in audit.iterrows():

        rcept_no = str(
            row[
                "rcept_no"
            ]
        )

        group = hits_df.loc[
            hits_df[
                "rcept_no"
            ].eq(
                rcept_no
            )
        ]

        existing = group.loc[
            group[
                "resolved_existing_path"
            ].astype(str)
            .str.len()
            .gt(
                0
            )
        ]

        direct = group.loc[
            group[
                "hit_type"
            ].eq(
                "FILESYSTEM_FILENAME_MATCH"
            )
        ]

        manifest_exists = group.loc[
            group[
                "hit_type"
            ].eq(
                "MANIFEST_PATH_EXISTS"
            )
        ]

        reference_only = group.loc[
            group[
                "hit_type"
            ].eq(
                "MANIFEST_PATH_REFERENCE_ONLY"
            )
        ]

        existing_paths = (
            existing[
                "resolved_existing_path"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if existing_paths:

            cache_status = (
                "EXISTING_CACHE_FOUND"
            )

        elif not reference_only.empty:

            cache_status = (
                "MANIFEST_REFERENCE_BUT_FILE_NOT_FOUND"
            )

        else:

            cache_status = (
                "NO_EXISTING_DOCUMENT_CACHE_FOUND"
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
                row.get(
                    "missing_metric",
                    "",
                ),

                "cache_status":
                cache_status,

                "direct_filename_hits":
                len(
                    direct
                ),

                "manifest_existing_path_hits":
                len(
                    manifest_exists
                ),

                "manifest_reference_only_hits":
                len(
                    reference_only
                ),

                "existing_path_count":
                len(
                    existing_paths
                ),

                "existing_paths":
                " || ".join(
                    existing_paths
                ),
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

    collection_targets = summary.loc[
        ~summary[
            "cache_status"
        ].eq(
            "EXISTING_CACHE_FOUND"
        )
    ].copy()

    collection_targets.to_csv(
        OUT_COLLECTION_TARGETS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[Receipt cache status]"
    )

    print(
        summary[
            "cache_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[Cache status by missing metric]"
    )

    print(
        pd.crosstab(
            summary[
                "missing_metric"
            ],
            summary[
                "cache_status"
            ],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[Existing cache receipts]"
    )

    cached = summary.loc[
        summary[
            "cache_status"
        ].eq(
            "EXISTING_CACHE_FOUND"
        )
    ]

    print(
        f"{len(cached):,}"
    )

    if not cached.empty:

        with pd.option_context(
            "display.max_colwidth",
            180,
            "display.width",
            360,
            "display.max_rows",
            30,
        ):

            print(
                cached[
                    [
                        "stock_code",
                        "corp_name",
                        "period_key",
                        "rcept_no",
                        "missing_metric",
                        "existing_path_count",
                        "existing_paths",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Receipts requiring document collection]"
    )

    print(
        f"{len(collection_targets):,}"
    )

    if not collection_targets.empty:

        print(
            collection_targets[
                [
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "missing_metric",
                    "cache_status",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[Hit types]"
    )

    if hits_df.empty:
        print(
            "None"
        )

    else:
        print(
            hits_df[
                "hit_type"
            ]
            .value_counts()
            .to_string()
        )

    print(
        "\nOutputs:"
    )

    print(
        f"- Receipt discovery : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- All cache hits    : "
        f"{OUT_HITS}"
    )

    print(
        f"- Collection targets: "
        f"{OUT_COLLECTION_TARGETS}"
    )

    print(
        "\n해석 원칙:"
        "\n- EXISTING_CACHE_FOUND면 다음 단계에서 그 exact-receipt source 재사용"
        "\n- NO_EXISTING_DOCUMENT_CACHE_FOUND면 그 receipt만 신규 document 수집"
        "\n- MANIFEST_REFERENCE_BUT_FILE_NOT_FOUND는 stale provenance이므로 재수집 후보"
        "\n- 아직 문서 파싱/값 복구/merge 없음"
    )


if __name__ == "__main__":
    main()
