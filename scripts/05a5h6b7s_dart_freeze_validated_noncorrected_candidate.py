from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# 05A5-H6B7S. Freeze Validated Non-corrected Candidate
#
# 목적
# ------------------------------------------------------------
# H6B7R FINAL QA: PASS를 통과한 H6B7Q recovered parquet을
# "validated non-corrected candidate"로 고정한다.
#
# 원칙
# ------------------------------------------------------------
# - 기존 원본 dart_noncorrected_pit_receipt_wide_final.parquet은 보존
# - recovered parquet도 보존
# - 새 stable candidate 파일을 별도로 생성
# - SHA256 / row count / core-count distribution / QA 상태를 manifest에 기록
# - 예상 QA가 하나라도 다르면 promotion 중단
#
# 실행:
# python scripts\05a5h6b7s_dart_freeze_validated_noncorrected_candidate.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

SOURCE = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7q_ytd_recovered.parquet"
)

RECEIPT_QA = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7r_receipt_qa.csv"
)

CELL_DIFF = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_h6b7r_cell_diff.csv"
)

TARGET = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_validated.parquet"
)

MANIFEST_JSON = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_validated_manifest.json"
)

MANIFEST_CSV = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_validated_manifest.csv"
)


EXPECTED_ROWS = 16_869
EXPECTED_CHANGED_RECEIPTS = 142
EXPECTED_CHANGED_CELLS = 852
EXPECTED_CORE_DISTRIBUTION = {
    0: 748,
    3: 2,
    5: 248,
    6: 15_871,
}
EXPECTED_HARD_BALANCE_FAIL = 0
EXPECTED_YTD_REVIEW_CHANGED = 0


TARGET_ALIASES = {
    "assets": [
        "assets",
        "assets_total",
    ],
    "liabilities": [
        "liabilities",
        "liabilities_total",
    ],
    "equity": [
        "equity",
        "equity_total",
    ],
    "revenue": [
        "revenue",
        "revenue_cumulative",
    ],
    "operating_income": [
        "operating_income",
        "operating_income_cumulative",
    ],
    "net_income": [
        "net_income",
        "net_income_total",
        "net_income_cumulative",
        "net_income_total_cumulative",
    ],
}


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
            f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def resolve_metric_mapping(
    frame: pd.DataFrame,
) -> dict[str, str]:

    mapping = {}

    for source, candidates in (
        TARGET_ALIASES.items()
    ):

        hits = [
            col
            for col in candidates
            if col in frame.columns
        ]

        if len(
            hits
        ) != 1:
            raise RuntimeError(
                f"Metric mapping failed for {source}: {hits}"
            )

        mapping[
            source
        ] = hits[
            0
        ]

    return mapping


def main():

    for path in [
        SOURCE,
        RECEIPT_QA,
        CELL_DIFF,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "05A5-H6B7S FREEZE VALIDATED NON-CORRECTED CANDIDATE"
    )

    print(
        "=" * 120
    )

    frame = pd.read_parquet(
        SOURCE
    )

    qa = pd.read_csv(
        RECEIPT_QA,
        low_memory=False,
    )

    diff = pd.read_csv(
        CELL_DIFF,
        low_memory=False,
    )

    print(
        f"\nSource rows: "
        f"{len(frame):,}"
    )

    if len(
        frame
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, got {len(frame):,}"
        )

    mapping = resolve_metric_mapping(
        frame
    )

    target_cols = list(
        mapping.values()
    )

    core_count = (
        frame[
            target_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    distribution = {
        int(
            k
        ):
        int(
            v
        )
        for k, v in (
            core_count
            .value_counts()
            .sort_index()
            .items()
        )
    }

    print(
        "\n[Core-count distribution]"
    )

    print(
        pd.Series(
            distribution
        ).to_string()
    )

    if (
        distribution
        != EXPECTED_CORE_DISTRIBUTION
    ):
        raise RuntimeError(
            "Core-count distribution mismatch.\n"
            f"Expected: {EXPECTED_CORE_DISTRIBUTION}\n"
            f"Actual:   {distribution}"
        )

    changed_receipts = int(
        diff[
            "rcept_no"
        ].nunique()
    )

    changed_cells = len(
        diff
    )

    if (
        changed_receipts
        != EXPECTED_CHANGED_RECEIPTS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_CHANGED_RECEIPTS} changed receipts, "
            f"got {changed_receipts}"
        )

    if (
        changed_cells
        != EXPECTED_CHANGED_CELLS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_CHANGED_CELLS} changed cells, "
            f"got {changed_cells}"
        )

    hard_balance_fail = int(
        qa[
            "balance_status_after"
        ]
        .eq(
            "HARD_FAIL"
        )
        .sum()
    )

    if (
        hard_balance_fail
        != EXPECTED_HARD_BALANCE_FAIL
    ):
        raise RuntimeError(
            f"Hard balance fail count = {hard_balance_fail}"
        )

    review_changed = int(
        (
            qa[
                "ytd_review_unmerged"
            ].fillna(
                False
            )
            & qa[
                "h6b7q_validated_merge"
            ].fillna(
                False
            )
        ).sum()
    )

    if (
        review_changed
        != EXPECTED_YTD_REVIEW_CHANGED
    ):
        raise RuntimeError(
            "A YTD review receipt appears in the validated merge set."
        )

    source_sha256 = sha256_file(
        SOURCE
    )

    # --------------------------------------------------------
    # Freeze by byte-copy.
    # This avoids rewriting parquet representation.
    # --------------------------------------------------------

    if TARGET.exists():

        existing_hash = sha256_file(
            TARGET
        )

        if (
            existing_hash
            != source_sha256
        ):
            raise RuntimeError(
                f"Target already exists with different content:\n{TARGET}\n"
                "Delete/rename it manually only after inspection."
            )

        copy_status = (
            "ALREADY_EXISTS_IDENTICAL"
        )

    else:

        shutil.copy2(
            SOURCE,
            TARGET,
        )

        copy_status = (
            "COPIED"
        )

    target_sha256 = sha256_file(
        TARGET
    )

    if (
        target_sha256
        != source_sha256
    ):
        raise RuntimeError(
            "SHA256 mismatch after promotion."
        )

    manifest = {
        "stage":
        "05A5-H6B7S",

        "status":
        "VALIDATED_NONCORRECTED_CANDIDATE",

        "promotion_status":
        copy_status,

        "source_file":
        str(
            SOURCE
        ),

        "target_file":
        str(
            TARGET
        ),

        "sha256":
        target_sha256,

        "rows":
        len(
            frame
        ),

        "validated_merged_receipts":
        changed_receipts,

        "changed_core_cells":
        changed_cells,

        "non_core_changed_cells":
        0,

        "ytd_review_receipts_changed":
        review_changed,

        "hard_balance_failures":
        hard_balance_fail,

        "core0":
        distribution.get(
            0,
            0,
        ),

        "core3":
        distribution.get(
            3,
            0,
        ),

        "core5":
        distribution.get(
            5,
            0,
        ),

        "core6":
        distribution.get(
            6,
            0,
        ),

        "metric_mapping":
        mapping,

        "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

        "notes": [
            "Original dart_noncorrected_pit_receipt_wide_final.parquet preserved.",
            "H6B7Q recovered parquet preserved.",
            "Only H6B7P PASS_ALL_INCOME_YTD receipts were merged.",
            "35 YTD-review receipts remain unresolved.",
            "This file is the validated candidate for downstream non-corrected work.",
        ],
    }

    MANIFEST_JSON.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    flat_manifest = {
        key:
        (
            json.dumps(
                value,
                ensure_ascii=False,
            )
            if isinstance(
                value,
                (
                    dict,
                    list,
                ),
            )
            else value
        )
        for key, value
        in manifest.items()
    }

    pd.DataFrame(
        [
            flat_manifest
        ]
    ).to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[Promotion]"
    )

    print(
        f"status : "
        f"{copy_status}"
    )

    print(
        f"source : "
        f"{SOURCE}"
    )

    print(
        f"target : "
        f"{TARGET}"
    )

    print(
        f"sha256 : "
        f"{target_sha256}"
    )

    print(
        "\n[Validated state]"
    )

    print(
        f"core6 : "
        f"{distribution.get(6, 0):,}"
    )

    print(
        f"core5 : "
        f"{distribution.get(5, 0):,}"
    )

    print(
        f"core3 : "
        f"{distribution.get(3, 0):,}"
    )

    print(
        f"core0 : "
        f"{distribution.get(0, 0):,}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"- Validated candidate: "
        f"{TARGET}"
    )

    print(
        f"- Manifest JSON      : "
        f"{MANIFEST_JSON}"
    )

    print(
        f"- Manifest CSV       : "
        f"{MANIFEST_CSV}"
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "H6B NO-DATA RECOVERY: CLOSED / VALIDATED"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
