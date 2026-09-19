from __future__ import annotations

import io
import itertools
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 05A5-H5D. Ambiguous Balance Same-Receipt XBRL Audit
#
# 목적
# ------------------------------------------------------------
# H5C에서 document HTML 기준으로도 balance set이 여러 개라
# 자동 선택할 수 없었던 receipt만 대상으로,
# OpenDART same-receipt XBRL 원본 ZIP을 내려받아
#
#   같은 XBRL instance file
#   + 같은 contextRef
#   + 현재 보고기간 말일
#
# 조건에서 Assets / Liabilities / Equity fact를 묶어
# balance equation을 검사한다.
#
# 이 스크립트는 QA용이며 자동 production override를 하지 않는다.
#
# 공식 endpoint:
#   https://opendart.fss.or.kr/api/fnlttXbrl.xml
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_noncorrected_ambiguous_xbrl_balance_facts.csv
#   dart_noncorrected_ambiguous_xbrl_balance_sets.csv
#   dart_noncorrected_ambiguous_xbrl_balance_resolution.csv
#
# RAW
# ------------------------------------------------------------
# data/raw/dart/noncorrected_balance_qa_xbrl/
#
# 실행
# ------------------------------------------------------------
# python scripts\05a5h5d_dart_ambiguous_balance_xbrl_audit.py
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim" / "dart"

FINAL = (
    INTERIM
    / "dart_noncorrected_pit_receipt_wide_reconciled.parquet"
)

H5C_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_balance_candidate_set_resolution.csv"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "noncorrected_balance_qa_xbrl"
)

OUT_FACTS = (
    INTERIM
    / "dart_noncorrected_ambiguous_xbrl_balance_facts.csv"
)

OUT_SETS = (
    INTERIM
    / "dart_noncorrected_ambiguous_xbrl_balance_sets.csv"
)

OUT_RESOLUTION = (
    INTERIM
    / "dart_noncorrected_ambiguous_xbrl_balance_resolution.csv"
)

API_URL = (
    "https://opendart.fss.or.kr/api/fnlttXbrl.xml"
)

ABS_TOL = 2_000_000
REL_TOL = 1e-6

STRICT_CONCEPTS = {
    "assets": {
        "assets",
    },
    "liabilities": {
        "liabilities",
    },
    "equity": {
        "equity",
    },
}


def receipt_string(
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


def local_name(
    tag: str,
) -> str:
    if "}" in tag:
        return tag.rsplit(
            "}",
            1,
        )[-1]
    if ":" in tag:
        return tag.rsplit(
            ":",
            1,
        )[-1]
    return tag


def normalize_token(
    value: Any,
) -> str:
    if value is None:
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def parse_number(
    text: Any,
    scale: Any = None,
) -> float:
    if text is None:
        return np.nan

    raw = str(text).strip()

    if raw in {
        "",
        "-",
        "nan",
        "None",
        "<NA>",
    }:
        return np.nan

    raw = raw.replace(
        ",",
        "",
    )

    if (
        raw.startswith("(")
        and raw.endswith(")")
    ):
        raw = (
            "-"
            + raw[1:-1]
        )

    value = pd.to_numeric(
        raw,
        errors="coerce",
    )

    if pd.isna(
        value
    ):
        return np.nan

    if scale not in {
        None,
        "",
    }:
        try:
            value = (
                float(value)
                * (
                    10
                    ** int(
                        scale
                    )
                )
            )
        except Exception:
            pass

    return float(
        value
    )


def period_end_from_key(
    period_key: str,
) -> str | None:

    match = re.search(
        r"(\d{4})\.(\d{2})",
        str(
            period_key
        ),
    )

    if not match:
        return None

    year = int(
        match.group(
            1
        )
    )

    month = int(
        match.group(
            2
        )
    )

    end_day = {
        3: 31,
        6: 30,
        9: 30,
        12: 31,
    }.get(
        month
    )

    if end_day is None:
        return None

    return (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{end_day:02d}"
    )


def balance_qa(
    assets,
    liabilities,
    equity,
):
    if any(
        pd.isna(
            x
        )
        for x in [
            assets,
            liabilities,
            equity,
        ]
    ):
        return (
            "missing",
            np.nan,
            np.nan,
        )

    gap = (
        float(
            assets
        )
        - float(
            liabilities
        )
        - float(
            equity
        )
    )

    rel = (
        abs(
            gap
        )
        / abs(
            float(
                assets
            )
        )
        if float(
            assets
        ) != 0
        else np.nan
    )

    if gap == 0:
        qa = (
            "exact_pass"
        )

    elif (
        abs(
            gap
        )
        <= ABS_TOL
        or (
            pd.notna(
                rel
            )
            and rel
            <= REL_TOL
        )
    ):
        qa = (
            "rounding_pass"
        )

    else:
        qa = (
            "fail"
        )

    return (
        qa,
        gap,
        rel,
    )


def download_xbrl(
    session: requests.Session,
    api_key: str,
    rcept_no: str,
    reprt_code: str,
) -> bytes:

    response = session.get(
        API_URL,
        params={
            "crtfc_key":
            api_key,
            "rcept_no":
            rcept_no,
            "reprt_code":
            str(
                reprt_code
            ),
        },
        timeout=120,
    )

    response.raise_for_status()

    content = (
        response.content
    )

    if content[:2] != b"PK":
        preview = (
            content[
                :1000
            ]
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            "XBRL ZIP 응답이 아닙니다: "
            + preview
        )

    return content


def parse_contexts(
    root: ET.Element,
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:

    contexts = {}

    for elem in root.iter():
        if (
            local_name(
                elem.tag
            )
            != "context"
        ):
            continue

        context_id = elem.attrib.get(
            "id"
        )

        if not context_id:
            continue

        instant = None
        start_date = None
        end_date = None
        dimensions = []

        for child in elem.iter():

            lname = local_name(
                child.tag
            )

            text = (
                child.text.strip()
                if child.text
                else ""
            )

            if lname == "instant":
                instant = text

            elif lname == "startDate":
                start_date = text

            elif lname == "endDate":
                end_date = text

            elif lname in {
                "explicitMember",
                "typedMember",
            }:
                dim = (
                    child.attrib.get(
                        "dimension",
                        "",
                    )
                )

                dimensions.append(
                    f"{dim}={text}"
                )

        contexts[
            context_id
        ] = {
            "instant":
            instant,
            "start_date":
            start_date,
            "end_date":
            end_date,
            "dimensions":
            " | ".join(
                dimensions
            ),
        }

    return contexts


def classify_concept(
    concept_local: str,
) -> str | None:

    token = normalize_token(
        concept_local
    )

    for metric, names in (
        STRICT_CONCEPTS.items()
    ):
        if token in names:
            return metric

    return None


def file_context_hint(
    filename: str,
    dimensions: str,
) -> tuple[
    int,
    str,
]:

    text = (
        str(
            filename
        )
        + " "
        + str(
            dimensions
        )
    ).lower()

    consolidated_tokens = [
        "consolidated",
        "consolidation",
        "연결",
    ]

    separate_tokens = [
        "separate",
        "individual",
        "별도",
        "개별",
    ]

    chits = [
        token
        for token
        in consolidated_tokens
        if token
        in text
    ]

    shits = [
        token
        for token
        in separate_tokens
        if token
        in text
    ]

    score = (
        len(
            chits
        )
        - len(
            shits
        )
    )

    reason = (
        "consolidated="
        + ",".join(
            chits
        )
        + " | separate="
        + ",".join(
            shits
        )
    )

    return (
        score,
        reason,
    )


def parse_instance_member(
    member_name: str,
    data: bytes,
    target: pd.Series,
) -> pd.DataFrame:

    try:
        root = (
            ET.fromstring(
                data
            )
        )
    except Exception:
        return pd.DataFrame()

    if (
        local_name(
            root.tag
        ).lower()
        != "xbrl"
    ):
        return pd.DataFrame()

    contexts = (
        parse_contexts(
            root
        )
    )

    target_end = (
        period_end_from_key(
            target[
                "canonical_period_key"
            ]
        )
    )

    rows = []

    for elem in list(
        root
    ):

        context_ref = (
            elem.attrib.get(
                "contextRef"
            )
        )

        if not context_ref:
            continue

        concept_local = (
            local_name(
                elem.tag
            )
        )

        metric = (
            classify_concept(
                concept_local
            )
        )

        if metric is None:
            continue

        context = (
            contexts.get(
                context_ref,
                {},
            )
        )

        instant = (
            context.get(
                "instant"
            )
        )

        if (
            target_end
            and instant
            and instant
            != target_end
        ):
            continue

        value = (
            parse_number(
                elem.text,
                elem.attrib.get(
                    "scale"
                ),
            )
        )

        if pd.isna(
            value
        ):
            continue

        dimensions = (
            context.get(
                "dimensions",
                "",
            )
        )

        hint_score, hint_reason = (
            file_context_hint(
                member_name,
                dimensions,
            )
        )

        rows.append(
            {
                "stock_code":
                str(
                    target[
                        "stock_code"
                    ]
                ).zfill(
                    6
                ),

                "corp_name":
                target.get(
                    "corp_name"
                ),

                "canonical_period_key":
                target[
                    "canonical_period_key"
                ],

                "rcept_no":
                str(
                    target[
                        "rcept_no"
                    ]
                ),

                "reprt_code":
                target[
                    "reprt_code"
                ],

                "target_period_end":
                target_end,

                "member_name":
                member_name,

                "context_ref":
                context_ref,

                "context_instant":
                instant,

                "context_start_date":
                context.get(
                    "start_date"
                ),

                "context_end_date":
                context.get(
                    "end_date"
                ),

                "context_dimensions":
                dimensions,

                "context_hint_score":
                hint_score,

                "context_hint_reason":
                hint_reason,

                "metric":
                metric,

                "concept_local_name":
                concept_local,

                "concept_tag":
                elem.tag,

                "value":
                value,

                "unit_ref":
                elem.attrib.get(
                    "unitRef"
                ),

                "decimals":
                elem.attrib.get(
                    "decimals"
                ),

                "scale":
                elem.attrib.get(
                    "scale"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_sets(
    facts: pd.DataFrame,
) -> pd.DataFrame:

    if facts.empty:
        return pd.DataFrame()

    records = []

    group_cols = [
        "stock_code",
        "corp_name",
        "canonical_period_key",
        "rcept_no",
        "reprt_code",
        "target_period_end",
        "member_name",
        "context_ref",
        "context_instant",
        "context_dimensions",
        "context_hint_score",
        "context_hint_reason",
    ]

    for key, group in (
        facts.groupby(
            group_cols,
            dropna=False,
        )
    ):

        metric_rows = {}

        for metric in [
            "assets",
            "liabilities",
            "equity",
        ]:
            sub = group.loc[
                group[
                    "metric"
                ].eq(
                    metric
                )
            ].copy()

            if sub.empty:
                metric_rows = {}
                break

            metric_rows[
                metric
            ] = (
                sub.drop_duplicates(
                    subset=[
                        "value",
                        "concept_local_name",
                    ]
                )
                .head(
                    10
                )
            )

        if not metric_rows:
            continue

        for (
            (_, a),
            (_, l),
            (_, e),
        ) in itertools.product(
            metric_rows[
                "assets"
            ].iterrows(),
            metric_rows[
                "liabilities"
            ].iterrows(),
            metric_rows[
                "equity"
            ].iterrows(),
        ):

            qa, gap, rel = (
                balance_qa(
                    a[
                        "value"
                    ],
                    l[
                        "value"
                    ],
                    e[
                        "value"
                    ],
                )
            )

            base = dict(
                zip(
                    group_cols,
                    key
                    if isinstance(
                        key,
                        tuple,
                    )
                    else [
                        key
                    ],
                )
            )

            base.update(
                {
                    "assets":
                    a[
                        "value"
                    ],

                    "liabilities":
                    l[
                        "value"
                    ],

                    "equity":
                    e[
                        "value"
                    ],

                    "assets_concept":
                    a[
                        "concept_local_name"
                    ],

                    "liabilities_concept":
                    l[
                        "concept_local_name"
                    ],

                    "equity_concept":
                    e[
                        "concept_local_name"
                    ],

                    "balance_gap":
                    gap,

                    "balance_relative_gap":
                    rel,

                    "balance_qa":
                    qa,
                }
            )

            records.append(
                base
            )

    return pd.DataFrame(
        records
    )


def resolve(
    sets_df: pd.DataFrame,
    target: pd.Series,
) -> dict[
    str,
    Any,
]:

    base = {
        "stock_code":
        str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        ),

        "corp_name":
        target.get(
            "corp_name"
        ),

        "canonical_period_key":
        target[
            "canonical_period_key"
        ],

        "rcept_no":
        str(
            target[
                "rcept_no"
            ]
        ),
    }

    if sets_df.empty:
        base.update(
            {
                "resolution":
                "no_same_context_balance_set",

                "unique_sets":
                0,

                "pass_sets":
                0,
            }
        )

        return base

    unique = (
        sets_df.sort_values(
            [
                "context_hint_score",
                "balance_relative_gap",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "assets",
                "liabilities",
                "equity",
                "member_name",
                "context_ref",
            ]
        )
    )

    passed = unique.loc[
        unique[
            "balance_qa"
        ].isin(
            [
                "exact_pass",
                "rounding_pass",
            ]
        )
    ].copy()

    positive = passed.loc[
        passed[
            "context_hint_score"
        ].gt(
            0
        )
    ].copy()

    if len(
        positive
    ) == 1:
        chosen = (
            positive.iloc[
                0
            ]
        )

        resolution = (
            "unique_xbrl_consolidated_pass"
        )

    elif len(
        positive
    ) > 1:
        base.update(
            {
                "resolution":
                "multiple_xbrl_consolidated_pass",

                "unique_sets":
                len(
                    unique
                ),

                "pass_sets":
                len(
                    passed
                ),
            }
        )

        return base

    elif len(
        passed
    ) == 1:
        chosen = (
            passed.iloc[
                0
            ]
        )

        resolution = (
            "unique_xbrl_pass_context_weak"
        )

    elif len(
        passed
    ) > 1:
        base.update(
            {
                "resolution":
                "multiple_xbrl_pass_context_ambiguous",

                "unique_sets":
                len(
                    unique
                ),

                "pass_sets":
                len(
                    passed
                ),
            }
        )

        return base

    else:
        base.update(
            {
                "resolution":
                "no_xbrl_balance_pass",

                "unique_sets":
                len(
                    unique
                ),

                "pass_sets":
                0,
            }
        )

        return base

    base.update(
        {
            "resolution":
            resolution,

            "unique_sets":
            len(
                unique
            ),

            "pass_sets":
            len(
                passed
            ),

            "recommended_assets":
            chosen[
                "assets"
            ],

            "recommended_liabilities":
            chosen[
                "liabilities"
            ],

            "recommended_equity":
            chosen[
                "equity"
            ],

            "recommended_qa":
            chosen[
                "balance_qa"
            ],

            "member_name":
            chosen[
                "member_name"
            ],

            "context_ref":
            chosen[
                "context_ref"
            ],

            "context_dimensions":
            chosen[
                "context_dimensions"
            ],

            "context_hint_score":
            chosen[
                "context_hint_score"
            ],

            "context_hint_reason":
            chosen[
                "context_hint_reason"
            ],
        }
    )

    return base


def main():

    for path in [
        FINAL,
        H5C_RESOLUTION,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "DART_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            ".env의 DART_API_KEY가 없습니다."
        )

    final = pd.read_parquet(
        FINAL
    )

    resolution = pd.read_csv(
        H5C_RESOLUTION,
        dtype={
            "rcept_no":
            str,
            "stock_code":
            str,
        },
        low_memory=False,
    )

    final[
        "rcept_no"
    ] = receipt_string(
        final[
            "rcept_no"
        ]
    )

    resolution[
        "rcept_no"
    ] = receipt_string(
        resolution[
            "rcept_no"
        ]
    )

    ambiguous_receipts = set(
        resolution.loc[
            resolution[
                "resolution"
            ].astype(str)
            .str.startswith(
                "multiple_"
            ),
            "rcept_no",
        ].astype(str)
    )

    targets = final.loc[
        final[
            "rcept_no"
        ]
        .astype(str)
        .isin(
            ambiguous_receipts
        )
    ].copy()

    print(
        "\n"
        + "="
        * 110
    )

    print(
        "05A5-H5D AMBIGUOUS BALANCE SAME-RECEIPT XBRL AUDIT"
    )

    print(
        "="
        * 110
    )

    print(
        f"\nTargets: "
        f"{len(targets):,}"
    )

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = (
        requests.Session()
    )

    all_facts = []
    all_sets = []
    resolutions = []

    for idx, (_, target) in enumerate(
        targets.iterrows(),
        start=1,
    ):

        receipt = str(
            target[
                "rcept_no"
            ]
        )

        stock = str(
            target[
                "stock_code"
            ]
        ).zfill(
            6
        )

        reprt_code = str(
            target[
                "reprt_code"
            ]
        )

        zip_path = (
            RAW_ROOT
            / (
                f"{stock}_"
                f"{receipt}_"
                f"{reprt_code}_xbrl.zip"
            )
        )

        print(
            f"\n[{idx}/{len(targets)}] "
            f"{stock} | "
            f"{target['canonical_period_key']} | "
            f"{receipt}"
        )

        try:
            if not zip_path.exists():
                content = (
                    download_xbrl(
                        session,
                        api_key,
                        receipt,
                        reprt_code,
                    )
                )

                zip_path.write_bytes(
                    content
                )

                print(
                    f"  XBRL saved: "
                    f"{zip_path}"
                )

            else:
                print(
                    f"  existing XBRL: "
                    f"{zip_path}"
                )

            facts_for_receipt = []

            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as zf:

                members = [
                    name
                    for name
                    in zf.namelist()
                    if name.lower().endswith(
                        (
                            ".xbrl",
                            ".xml",
                        )
                    )
                ]

                for member in members:

                    try:
                        data = zf.read(
                            member
                        )

                        parsed = (
                            parse_instance_member(
                                member,
                                data,
                                target,
                            )
                        )

                        if not parsed.empty:
                            facts_for_receipt.append(
                                parsed
                            )

                    except Exception as exc:
                        print(
                            "  member parse skip: "
                            f"{member} | "
                            f"{type(exc).__name__}: {exc}"
                        )

            facts = (
                pd.concat(
                    facts_for_receipt,
                    ignore_index=True,
                )
                if facts_for_receipt
                else pd.DataFrame()
            )

            sets_df = (
                build_sets(
                    facts
                )
            )

            print(
                f"  facts={len(facts):,} "
                f"| same-context sets={len(sets_df):,}"
            )

            if not sets_df.empty:

                unique = (
                    sets_df.sort_values(
                        [
                            "context_hint_score",
                            "balance_relative_gap",
                        ],
                        ascending=[
                            False,
                            True,
                        ],
                    )
                    .drop_duplicates(
                        subset=[
                            "assets",
                            "liabilities",
                            "equity",
                            "member_name",
                            "context_ref",
                        ]
                    )
                )

                print(
                    "\n  [Top XBRL balance sets]"
                )

                cols = [
                    "member_name",
                    "context_ref",
                    "context_dimensions",
                    "context_hint_score",
                    "assets",
                    "liabilities",
                    "equity",
                    "balance_gap",
                    "balance_relative_gap",
                    "balance_qa",
                ]

                print(
                    unique[
                        cols
                    ]
                    .head(
                        20
                    )
                    .to_string(
                        index=False
                    )
                )

            result = (
                resolve(
                    sets_df,
                    target,
                )
            )

            print(
                "\n  resolution: "
                f"{result['resolution']}"
            )

            if not facts.empty:
                all_facts.append(
                    facts
                )

            if not sets_df.empty:
                all_sets.append(
                    sets_df
                )

            resolutions.append(
                result
            )

        except Exception as exc:

            print(
                "  ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            resolutions.append(
                {
                    "stock_code":
                    stock,

                    "corp_name":
                    target.get(
                        "corp_name"
                    ),

                    "canonical_period_key":
                    target[
                        "canonical_period_key"
                    ],

                    "rcept_no":
                    receipt,

                    "resolution":
                    "xbrl_error",

                    "error":
                    repr(
                        exc
                    ),
                }
            )

    facts_all = (
        pd.concat(
            all_facts,
            ignore_index=True,
        )
        if all_facts
        else pd.DataFrame()
    )

    sets_all = (
        pd.concat(
            all_sets,
            ignore_index=True,
        )
        if all_sets
        else pd.DataFrame()
    )

    resolution_all = (
        pd.DataFrame(
            resolutions
        )
    )

    facts_all.to_csv(
        OUT_FACTS,
        index=False,
        encoding="utf-8-sig",
    )

    sets_all.to_csv(
        OUT_SETS,
        index=False,
        encoding="utf-8-sig",
    )

    resolution_all.to_csv(
        OUT_RESOLUTION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 110
    )

    print(
        "H5D SUMMARY"
    )

    print(
        "="
        * 110
    )

    show = [
        c
        for c in [
            "stock_code",
            "corp_name",
            "canonical_period_key",
            "rcept_no",
            "resolution",
            "unique_sets",
            "pass_sets",
            "recommended_assets",
            "recommended_liabilities",
            "recommended_equity",
            "recommended_qa",
            "member_name",
            "context_ref",
            "context_dimensions",
        ]
        if c
        in resolution_all.columns
    ]

    print(
        "\n[Resolution]"
    )

    print(
        resolution_all[
            show
        ].to_string(
            index=False
        )
    )

    print(
        f"\nFacts      : "
        f"{OUT_FACTS}"
    )

    print(
        f"Balance sets: "
        f"{OUT_SETS}"
    )

    print(
        f"Resolution : "
        f"{OUT_RESOLUTION}"
    )

    print(
        "\n판정:"
        "\n- unique_xbrl_consolidated_pass → XBRL 원본에서 유일하게 연결 근거가 있는 coherent set"
        "\n- unique_xbrl_pass_context_weak → XBRL balance는 유일하지만 연결 근거 약함"
        "\n- multiple_* → XBRL도 자동선택 금지"
        "\n- no_xbrl_balance_pass/no_same_context_balance_set → 보수적으로 unresolved"
    )


if __name__ == "__main__":
    main()
