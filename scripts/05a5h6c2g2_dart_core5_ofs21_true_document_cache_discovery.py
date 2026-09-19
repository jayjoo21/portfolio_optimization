from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 05A5-H6C2G2. OFS21 TRUE DOCUMENT CACHE DISCOVERY
#
# H6C2G 수정판
# ------------------------------------------------------------
# H6C2G 문제:
#   aggregate parquet/csv 경로까지 "document cache"로 오인했다.
#
# 수정 원칙:
#   - .parquet / .csv는 절대 document cache로 인정하지 않음
#   - 실제 원문 가능성이 있는 확장자만 인정
#   - filesystem direct scan은 receipt number가 filename/path에 있을 때만
#   - manifest path는 exact receipt row와 연결되고 실제 document-like
#     file로 resolve될 때만 cache hit
#
# API 호출 / 다운로드 / 문서 파싱 / merge 없음.
#
# 실행:
# python scripts\05a5h6c2g2_dart_core5_ofs21_true_document_cache_discovery.py
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
    / "dart_noncorrected_core5_ofs21_true_document_cache_discovery.csv"
)

OUT_HITS = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_true_document_cache_hits.csv"
)

OUT_COLLECTION_TARGETS = (
    INTERIM_DART
    / "dart_noncorrected_core5_ofs21_true_document_collection_targets.csv"
)


EXPECTED_OFS = 21


# Only plausible document/source artifacts.
DOCUMENT_EXTENSIONS = {
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".zip",
    ".txt",
}

# Explicitly NEVER document cache.
AGGREGATE_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".feather",
    ".pkl",
    ".pickle",
}


PATH_COLUMN_PATTERN = re.compile(
    r"(?:"
    r"path|filepath|filename|file_name|"
    r"local_file|local_path|cache_path|"
    r"download_path|document_path|doc_path|"
    r"html_path|xml_path|zip_path|txt_path|"
    r"source_path|raw_path"
    r")",
    flags=re.IGNORECASE,
)


def normalize_receipt(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def is_document_like(path: Path) -> bool:
    suffix = path.suffix.lower()

    if suffix in AGGREGATE_EXTENSIONS:
        return False

    return suffix in DOCUMENT_EXTENSIONS


def resolve_path_text(text: str) -> Path | None:
    if not text:
        return None

    candidates = []

    raw = Path(text)

    if raw.is_absolute():
        candidates.append(raw)

    else:
        candidates.extend(
            [
                PROJECT_ROOT / raw,
                RAW_DART / raw,
                INTERIM_DART / raw,
            ]
        )

    for candidate in candidates:
        try:
            if (
                candidate.exists()
                and candidate.is_file()
                and is_document_like(candidate)
            ):
                return candidate.resolve()
        except OSError:
            continue

    return None


def scan_direct_files(
    receipts: set[str],
) -> list[dict]:

    records = []

    # Raw is primary. Interim is included only for document-like caches.
    for root in [
        RAW_DART,
        INTERIM_DART,
    ]:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if not is_document_like(path):
                continue

            text = str(path)

            matched = [
                rcept_no
                for rcept_no in receipts
                if rcept_no in text
            ]

            for rcept_no in matched:
                records.append(
                    {
                        "rcept_no": rcept_no,
                        "hit_type": "DIRECT_RECEIPT_DOCUMENT_FILE",
                        "manifest_file": "",
                        "manifest_column": "",
                        "manifest_value": "",
                        "resolved_path": str(path.resolve()),
                        "suffix": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                    }
                )

    return records


def inspect_manifest_frame(
    manifest_path: Path,
    frame: pd.DataFrame,
    receipts: set[str],
    path_cols: list[str],
) -> list[dict]:

    if "rcept_no" not in frame.columns:
        return []

    frame = frame.copy()

    frame["rcept_no"] = normalize_receipt(
        frame["rcept_no"]
    )

    frame = frame.loc[
        frame["rcept_no"].isin(receipts)
    ]

    if frame.empty:
        return []

    records = []

    for _, row in frame.iterrows():

        rcept_no = str(
            row["rcept_no"]
        )

        for col in path_cols:

            value = clean(
                row.get(col, "")
            )

            if not value:
                continue

            resolved = resolve_path_text(
                value
            )

            if resolved is not None:

                records.append(
                    {
                        "rcept_no": rcept_no,
                        "hit_type": "MANIFEST_DOCUMENT_PATH_EXISTS",
                        "manifest_file": safe_relative(manifest_path),
                        "manifest_column": col,
                        "manifest_value": value,
                        "resolved_path": str(resolved),
                        "suffix": resolved.suffix.lower(),
                        "size_bytes": resolved.stat().st_size,
                    }
                )

                continue

            # Stale/reference-only evidence is retained ONLY if the
            # value itself looks like a document filename/path.
            suffix = Path(value).suffix.lower()

            if suffix in DOCUMENT_EXTENSIONS:

                records.append(
                    {
                        "rcept_no": rcept_no,
                        "hit_type": "MANIFEST_DOCUMENT_REFERENCE_ONLY",
                        "manifest_file": safe_relative(manifest_path),
                        "manifest_column": col,
                        "manifest_value": value,
                        "resolved_path": "",
                        "suffix": suffix,
                        "size_bytes": np.nan,
                    }
                )

    return records


def inspect_parquet_manifest(
    path: Path,
    receipts: set[str],
) -> list[dict]:

    try:
        schema = pq.ParquetFile(
            path
        ).schema_arrow.names
    except Exception:
        return []

    if "rcept_no" not in schema:
        return []

    path_cols = [
        col
        for col in schema
        if (
            col != "rcept_no"
            and PATH_COLUMN_PATTERN.search(
                str(col)
            )
        )
    ]

    if not path_cols:
        return []

    try:
        frame = pd.read_parquet(
            path,
            columns=[
                "rcept_no",
            ]
            + path_cols,
        )
    except Exception:
        return []

    return inspect_manifest_frame(
        manifest_path=path,
        frame=frame,
        receipts=receipts,
        path_cols=path_cols,
    )


def inspect_csv_manifest(
    path: Path,
    receipts: set[str],
) -> list[dict]:

    try:
        head = pd.read_csv(
            path,
            nrows=3,
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
                str(col)
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
            ]
            + path_cols,
            dtype={
                "rcept_no": str,
            },
            low_memory=False,
        )
    except Exception:
        return []

    return inspect_manifest_frame(
        manifest_path=path,
        frame=frame,
        receipts=receipts,
        path_cols=path_cols,
    )


def manifest_files() -> list[Path]:

    result = []

    for root in [
        RAW_DART,
        INTERIM_DART,
    ]:

        if not root.exists():
            continue

        result.extend(
            root.rglob(
                "*.parquet"
            )
        )

        result.extend(
            root.rglob(
                "*.csv"
            )
        )

    # Avoid scanning outputs from this script.
    excluded = {
        OUT_RECEIPT.resolve(),
        OUT_HITS.resolve(),
        OUT_COLLECTION_TARGETS.resolve(),
    }

    unique = []

    seen = set()

    for path in sorted(
        result,
        key=lambda x:
        str(x),
    ):

        resolved = path.resolve()

        if resolved in excluded:
            continue

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        unique.append(
            path
        )

    return unique


def main():

    if not OFS21_AUDIT.exists():
        raise FileNotFoundError(
            OFS21_AUDIT
        )

    audit = pd.read_csv(
        OFS21_AUDIT,
        dtype={
            "rcept_no": str,
            "stock_code": str,
        },
        low_memory=False,
    )

    audit["rcept_no"] = normalize_receipt(
        audit["rcept_no"]
    )

    if len(audit) != EXPECTED_OFS:
        raise RuntimeError(
            f"Expected {EXPECTED_OFS} OFS receipts, "
            f"found {len(audit)}."
        )

    receipts = set(
        audit["rcept_no"]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6C2G2 OFS21 TRUE DOCUMENT CACHE DISCOVERY"
    )

    print(
        "=" * 120
    )

    print(
        f"\nOFS receipts: "
        f"{len(receipts):,}"
    )

    # --------------------------------------------------------
    # Direct exact-receipt file scan
    # --------------------------------------------------------

    hits = scan_direct_files(
        receipts
    )

    direct_receipts = {
        row["rcept_no"]
        for row in hits
    }

    print(
        "\n[Direct exact-receipt document files]"
    )

    print(
        f"receipts: "
        f"{len(direct_receipts):,}"
    )

    print(
        f"files   : "
        f"{len(hits):,}"
    )

    # --------------------------------------------------------
    # Strict manifest document-path scan
    # --------------------------------------------------------

    manifests = manifest_files()

    print(
        "\n[Manifest aggregate files inspected]"
    )

    print(
        len(
            manifests
        )
    )

    for path in manifests:

        suffix = path.suffix.lower()

        if suffix == ".parquet":

            hits.extend(
                inspect_parquet_manifest(
                    path,
                    receipts,
                )
            )

        elif suffix == ".csv":

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
                "manifest_file",
                "manifest_column",
                "manifest_value",
                "resolved_path",
                "suffix",
                "size_bytes",
            ]
        )

    hits_df = hits_df.drop_duplicates()

    hits_df.to_csv(
        OUT_HITS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Receipt summary
    # --------------------------------------------------------

    records = []

    for _, row in audit.iterrows():

        rcept_no = str(
            row["rcept_no"]
        )

        group = hits_df.loc[
            hits_df["rcept_no"].eq(
                rcept_no
            )
        ]

        existing = group.loc[
            group["resolved_path"]
            .fillna("")
            .astype(str)
            .str.len()
            .gt(0)
        ]

        stale = group.loc[
            group["hit_type"].eq(
                "MANIFEST_DOCUMENT_REFERENCE_ONLY"
            )
        ]

        paths = (
            existing["resolved_path"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if paths:
            status = "TRUE_DOCUMENT_CACHE_FOUND"

        elif not stale.empty:
            status = "DOCUMENT_REFERENCE_STALE"

        else:
            status = "NO_TRUE_DOCUMENT_CACHE_FOUND"

        records.append(
            {
                "rcept_no": rcept_no,
                "stock_code": row.get(
                    "stock_code",
                    "",
                ),
                "corp_name": row.get(
                    "corp_name",
                    "",
                ),
                "period_key": row.get(
                    "period_key",
                    "",
                ),
                "missing_metric": row.get(
                    "missing_metric",
                    "",
                ),
                "true_cache_status": status,
                "existing_document_count": len(
                    paths
                ),
                "existing_document_paths": " || ".join(
                    paths
                ),
                "stale_document_reference_count": len(
                    stale
                ),
            }
        )

    summary = pd.DataFrame(
        records
    )

    summary.to_csv(
        OUT_RECEIPT,
        index=False,
        encoding="utf-8-sig",
    )

    collection = summary.loc[
        ~summary[
            "true_cache_status"
        ].eq(
            "TRUE_DOCUMENT_CACHE_FOUND"
        )
    ].copy()

    collection.to_csv(
        OUT_COLLECTION_TARGETS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prints
    # --------------------------------------------------------

    print(
        "\n[TRUE receipt cache status]"
    )

    print(
        summary[
            "true_cache_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n[TRUE cache status by missing metric]"
    )

    print(
        pd.crosstab(
            summary["missing_metric"],
            summary["true_cache_status"],
            dropna=False,
        )
        .to_string()
    )

    print(
        "\n[True cached receipts]"
    )

    cached = summary.loc[
        summary[
            "true_cache_status"
        ].eq(
            "TRUE_DOCUMENT_CACHE_FOUND"
        )
    ]

    print(
        len(cached)
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
                        "existing_document_count",
                        "existing_document_paths",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    print(
        "\n[Receipts requiring real document collection]"
    )

    print(
        len(collection)
    )

    if not collection.empty:

        print(
            collection[
                [
                    "stock_code",
                    "corp_name",
                    "period_key",
                    "rcept_no",
                    "missing_metric",
                    "true_cache_status",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n[Strict hit types]"
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
        f"- True cache summary : "
        f"{OUT_RECEIPT}"
    )

    print(
        f"- Strict cache hits  : "
        f"{OUT_HITS}"
    )

    print(
        f"- Collection targets : "
        f"{OUT_COLLECTION_TARGETS}"
    )

    print(
        "\n해석 원칙:"
        "\n- parquet/csv는 document cache로 절대 인정하지 않음"
        "\n- TRUE_DOCUMENT_CACHE_FOUND만 원문 재사용 가능"
        "\n- NO_TRUE_DOCUMENT_CACHE_FOUND면 H6C2H 신규 exact-receipt document 수집 대상"
        "\n- DOCUMENT_REFERENCE_STALE도 재수집 대상"
        "\n- 아직 다운로드/파싱/merge 없음"
    )


if __name__ == "__main__":
    main()
