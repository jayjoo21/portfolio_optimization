from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree


# ============================================================
# 05A5-C0. DART PIT source structure audit
#
# 목적
# ------------------------------------------------------------
# 1) XBRL ZIP:
#    - instance .xbrl 파일
#    - namespace
#    - contextRef / unitRef
#    - 핵심 재무 concept 후보
#
# 2) document fallback ZIP:
#    - 원문 XML/HTML 구조
#    - 재무상태표 / 손익계산서 후보 table
#    - 자산총계/매출액/영업이익 등 키워드가 실제 어디에 있는지
#
# 이 단계는 "구조 파악"만 수행한다.
# 아직 production fundamental value를 만들지 않는다.
#
# INPUT
# data/raw/dart/pit_source_audit/**/*.zip
#
# OUTPUT
# data/interim/dart/dart_pit_source_structure_audit.csv
# data/interim/dart/dart_xbrl_context_audit.csv
# data/interim/dart/dart_document_table_audit.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dart"
    / "pit_source_audit"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dart"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY_PATH = (
    OUT_DIR
    / "dart_pit_source_structure_audit.csv"
)

CONTEXT_PATH = (
    OUT_DIR
    / "dart_xbrl_context_audit.csv"
)

DOCUMENT_TABLE_PATH = (
    OUT_DIR
    / "dart_document_table_audit.csv"
)


# 핵심 concept 후보.
# 아직 최종 account mapping이 아니라 structure audit용.
CONCEPT_KEYWORDS = [
    "Assets",
    "Liabilities",
    "Equity",
    "Revenue",
    "Sales",
    "OperatingIncome",
    "OperatingProfit",
    "ProfitLoss",
    "NetIncome",
]

KOREAN_KEYWORDS = [
    "자산총계",
    "부채총계",
    "자본총계",
    "매출액",
    "영업수익",
    "수익(매출액)",
    "영업이익",
    "영업이익(손실)",
    "당기순이익",
    "당기순이익(손실)",
    "분기순이익",
    "반기순이익",
]


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


def decode_bytes(content: bytes) -> str:
    encodings = [
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    for encoding in encodings:
        try:
            return content.decode(
                encoding
            )
        except UnicodeDecodeError:
            pass

    return content.decode(
        "utf-8",
        errors="replace",
    )


def extract_metadata_from_path(
    path: Path,
) -> dict[str, Any]:

    relative = path.relative_to(
        RAW_ROOT
    )

    parts = relative.parts

    stock_code = (
        parts[0]
        if len(parts) >= 1
        else None
    )

    period_key = (
        parts[1]
        if len(parts) >= 2
        else None
    )

    filename = path.stem

    # filename 끝의 _xbrl / _document 제거 전
    source_type = (
        "xbrl"
        if "_xbrl" in filename
        else (
            "document"
            if "_document" in filename
            else "unknown"
        )
    )

    match = re.search(
        r"(\d{14})",
        filename,
    )

    rcept_no = (
        match.group(1)
        if match
        else None
    )

    date_match = re.match(
        r"(\d{8})_",
        filename,
    )

    rcept_dt = (
        pd.to_datetime(
            date_match.group(1),
            format="%Y%m%d",
        )
        if date_match
        else pd.NaT
    )

    return {
        "stock_code": stock_code,
        "period_key": period_key,
        "rcept_no": rcept_no,
        "rcept_dt": rcept_dt,
        "source_type": source_type,
        "zip_path": str(path),
    }


def find_instance_member(
    zf: zipfile.ZipFile,
) -> str | None:

    names = zf.namelist()

    xbrl = [
        name
        for name in names
        if name.lower().endswith(
            ".xbrl"
        )
    ]

    if xbrl:
        # 일반적으로 하나.
        # 여러 개면 큰 파일 우선.
        return max(
            xbrl,
            key=lambda name: (
                zf.getinfo(name).file_size
            ),
        )

    xml = [
        name
        for name in names
        if name.lower().endswith(
            ".xml"
        )
    ]

    excluded = [
        "_lab",
        "_pre",
        "_cal",
        "_def",
        "label",
        "presentation",
        "calculation",
        "definition",
        "reference",
    ]

    candidates = [
        name
        for name in xml
        if not any(
            token in name.lower()
            for token in excluded
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda name: (
            zf.getinfo(name).file_size
        ),
    )


# ------------------------------------------------------------
# XBRL audit
# ------------------------------------------------------------

def parse_context(
    node: etree._Element,
) -> dict[str, Any]:

    context_id = node.get("id")

    entity_identifier = None
    start_date = None
    end_date = None
    instant = None

    dimension_members: list[str] = []

    for child in node.iter():

        name = local_name(
            child.tag
        )

        if name == "identifier":
            entity_identifier = (
                child.text
            )

        elif name == "startDate":
            start_date = child.text

        elif name == "endDate":
            end_date = child.text

        elif name == "instant":
            instant = child.text

        elif name in (
            "explicitMember",
            "typedMember",
        ):
            dim = child.get(
                "dimension"
            )

            text = (
                "".join(
                    child.itertext()
                ).strip()
            )

            dimension_members.append(
                f"{dim}={text}"
            )

    return {
        "context_id": context_id,
        "entity_identifier": entity_identifier,
        "start_date": start_date,
        "end_date": end_date,
        "instant": instant,
        "dimensions": " | ".join(
            dimension_members
        ),
        "dimension_count": len(
            dimension_members
        ),
    }


def audit_xbrl_zip(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    metadata = extract_metadata_from_path(
        path
    )

    with zipfile.ZipFile(
        path
    ) as zf:

        instance_member = (
            find_instance_member(
                zf
            )
        )

        if instance_member is None:
            return (
                {
                    **metadata,
                    "status": (
                        "no_instance_candidate"
                    ),
                },
                [],
            )

        content = zf.read(
            instance_member
        )

    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
    )

    root = etree.fromstring(
        content,
        parser=parser,
    )

    namespaces = {
        key or "default": value
        for key, value
        in (
            root.nsmap
            or {}
        ).items()
    }

    context_nodes = [
        node
        for node in root.iter()
        if local_name(
            node.tag
        ) == "context"
    ]

    contexts = [
        parse_context(
            node
        )
        for node in context_nodes
    ]

    context_lookup = {
        row["context_id"]: row
        for row in contexts
        if row[
            "context_id"
        ] is not None
    }

    unit_ids = []

    fact_count = 0
    numeric_fact_count = 0
    concept_counter = Counter()
    context_counter = Counter()

    key_fact_examples: list[str] = []

    for node in root.iter():

        name = local_name(
            node.tag
        )

        if name in (
            "context",
            "unit",
            "schemaRef",
        ):
            if name == "unit":
                unit_id = node.get(
                    "id"
                )
                if unit_id:
                    unit_ids.append(
                        unit_id
                    )
            continue

        context_ref = node.get(
            "contextRef"
        )

        if not context_ref:
            continue

        fact_count += 1

        concept_counter[
            name
        ] += 1

        context_counter[
            context_ref
        ] += 1

        unit_ref = node.get(
            "unitRef"
        )

        value = (
            node.text.strip()
            if node.text
            else ""
        )

        if unit_ref:
            numeric_fact_count += 1

        if any(
            keyword.lower()
            in name.lower()
            for keyword
            in CONCEPT_KEYWORDS
        ):
            ctx = context_lookup.get(
                context_ref,
                {}
            )

            key_fact_examples.append(
                (
                    f"{name}"
                    f"|ctx={context_ref}"
                    f"|unit={unit_ref}"
                    f"|value={value[:80]}"
                    f"|start={ctx.get('start_date')}"
                    f"|end={ctx.get('end_date')}"
                    f"|instant={ctx.get('instant')}"
                    f"|dims={ctx.get('dimension_count')}"
                )
            )

    summary = {
        **metadata,
        "status": "ok",
        "instance_member": (
            instance_member
        ),
        "instance_bytes": len(
            content
        ),
        "namespace_count": len(
            namespaces
        ),
        "namespaces": json.dumps(
            namespaces,
            ensure_ascii=False,
        ),
        "context_count": len(
            contexts
        ),
        "unit_count": len(
            set(unit_ids)
        ),
        "fact_count": fact_count,
        "numeric_fact_count": (
            numeric_fact_count
        ),
        "top_concepts": " | ".join(
            f"{name}:{count}"
            for name, count
            in concept_counter.most_common(
                25
            )
        ),
        "top_contexts": " | ".join(
            f"{name}:{count}"
            for name, count
            in context_counter.most_common(
                20
            )
        ),
        "key_fact_examples": " || ".join(
            key_fact_examples[:50]
        ),
    }

    context_rows = [
        {
            **metadata,
            **row,
        }
        for row in contexts
    ]

    return (
        summary,
        context_rows,
    )


# ------------------------------------------------------------
# document fallback audit
# ------------------------------------------------------------

def score_table_text(
    text: str,
) -> tuple[int, list[str]]:

    found = [
        keyword
        for keyword in KOREAN_KEYWORDS
        if keyword in text
    ]

    return (
        len(
            set(found)
        ),
        sorted(
            set(found)
        ),
    )


def audit_document_zip(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    metadata = extract_metadata_from_path(
        path
    )

    with zipfile.ZipFile(
        path
    ) as zf:

        members = [
            name
            for name in zf.namelist()
            if name.lower().endswith(
                (
                    ".xml",
                    ".html",
                    ".htm",
                )
            )
        ]

        if not members:
            return (
                {
                    **metadata,
                    "status": (
                        "no_xml_html_member"
                    ),
                },
                [],
            )

        member = max(
            members,
            key=lambda name: (
                zf.getinfo(
                    name
                ).file_size
            ),
        )

        raw = zf.read(
            member
        )

    text = decode_bytes(
        raw
    )

    # DART 원문은 XML이지만 HTML성 table markup을 포함.
    soup = BeautifulSoup(
        text,
        "lxml",
    )

    tables = soup.find_all(
        "table"
    )

    table_rows = []

    for index, table in enumerate(
        tables
    ):

        table_text = " ".join(
            table.stripped_strings
        )

        score, found = (
            score_table_text(
                table_text
            )
        )

        if score == 0:
            continue

        # 앞뒤 heading 후보
        previous_texts = []

        node = table

        for _ in range(6):
            node = (
                node.find_previous()
                if node
                else None
            )

            if node is None:
                break

            candidate = (
                " ".join(
                    node.stripped_strings
                )
                if hasattr(
                    node,
                    "stripped_strings",
                )
                else ""
            )

            candidate = re.sub(
                r"\s+",
                " ",
                candidate,
            ).strip()

            if (
                candidate
                and len(candidate) <= 250
            ):
                previous_texts.append(
                    candidate
                )

        # pandas로 table 구조가 읽히는지도 검사
        try:
            frames = pd.read_html(
                io.StringIO(
                    str(table)
                )
            )

            parsed_shape = (
                str(
                    frames[0].shape
                )
                if frames
                else None
            )

            sample_csv = (
                frames[0]
                .head(8)
                .to_csv(
                    index=False,
                )[:3000]
                if frames
                else None
            )
        except Exception as exc:
            parsed_shape = None
            sample_csv = (
                f"READ_HTML_ERROR:"
                f"{type(exc).__name__}:"
                f"{exc}"
            )

        table_rows.append(
            {
                **metadata,
                "document_member": (
                    member
                ),
                "table_index": index,
                "keyword_score": score,
                "keywords": " | ".join(
                    found
                ),
                "preceding_text": (
                    " || ".join(
                        previous_texts
                    )
                ),
                "table_text_sample": (
                    table_text[:3000]
                ),
                "parsed_shape": (
                    parsed_shape
                ),
                "parsed_sample_csv": (
                    sample_csv
                ),
            }
        )

    table_rows = sorted(
        table_rows,
        key=lambda row: (
            -row[
                "keyword_score"
            ],
            row[
                "table_index"
            ],
        ),
    )

    summary = {
        **metadata,
        "status": "ok",
        "document_member": member,
        "document_bytes": len(
            raw
        ),
        "table_count": len(
            tables
        ),
        "financial_candidate_tables": len(
            table_rows
        ),
        "top_candidate_tables": (
            " || ".join(
                (
                    f"idx={row['table_index']}"
                    f",score={row['keyword_score']}"
                    f",kw={row['keywords']}"
                    f",shape={row['parsed_shape']}"
                )
                for row
                in table_rows[:15]
            )
        ),
    }

    return (
        summary,
        table_rows,
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "audit할 ZIP 최대 개수. "
            "기본은 현재 sample 전체"
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"PIT source 폴더가 없습니다: {RAW_ROOT}"
        )

    zip_paths = sorted(
        RAW_ROOT.rglob(
            "*.zip"
        )
    )

    if args.max_files is not None:
        zip_paths = zip_paths[
            :args.max_files
        ]

    print(
        "\n"
        + "=" * 80
    )
    print(
        "DART PIT SOURCE STRUCTURE AUDIT"
    )
    print(
        "=" * 80
    )

    print(
        f"zip files: {len(zip_paths)}"
    )

    source_summaries = []
    context_rows = []
    document_table_rows = []

    for i, path in enumerate(
        zip_paths,
        start=1,
    ):

        metadata = (
            extract_metadata_from_path(
                path
            )
        )

        print(
            f"\n[{i}/{len(zip_paths)}] "
            f"{metadata['stock_code']} | "
            f"{metadata['period_key']} | "
            f"{metadata['source_type']}"
        )

        try:
            if (
                metadata[
                    "source_type"
                ] == "xbrl"
            ):
                (
                    summary,
                    contexts,
                ) = audit_xbrl_zip(
                    path
                )

                source_summaries.append(
                    summary
                )

                context_rows.extend(
                    contexts
                )

                print(
                    "  XBRL:",
                    f"contexts={summary.get('context_count')}",
                    f"facts={summary.get('fact_count')}",
                    f"instance={summary.get('instance_member')}",
                )

            elif (
                metadata[
                    "source_type"
                ] == "document"
            ):
                (
                    summary,
                    tables,
                ) = audit_document_zip(
                    path
                )

                source_summaries.append(
                    summary
                )

                document_table_rows.extend(
                    tables
                )

                print(
                    "  DOCUMENT:",
                    f"tables={summary.get('table_count')}",
                    f"candidates={summary.get('financial_candidate_tables')}",
                )

            else:
                source_summaries.append(
                    {
                        **metadata,
                        "status": (
                            "unknown_source_type"
                        ),
                    }
                )

        except Exception as exc:
            source_summaries.append(
                {
                    **metadata,
                    "status": "error",
                    "error": repr(
                        exc
                    ),
                }
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary_df = pd.DataFrame(
        source_summaries
    )

    context_df = pd.DataFrame(
        context_rows
    )

    document_df = pd.DataFrame(
        document_table_rows
    )

    summary_df.to_csv(
        SOURCE_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    context_df.to_csv(
        CONTEXT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    document_df.to_csv(
        DOCUMENT_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "SOURCE STRUCTURE SUMMARY"
    )
    print(
        "=" * 80
    )

    show_cols = [
        col
        for col in [
            "stock_code",
            "period_key",
            "rcept_dt",
            "source_type",
            "status",
            "instance_member",
            "context_count",
            "fact_count",
            "table_count",
            "financial_candidate_tables",
        ]
        if col in summary_df.columns
    ]

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    if not context_df.empty:
        print(
            "\n[XBRL context patterns]"
        )

        pattern = (
            context_df
            .groupby(
                [
                    "start_date",
                    "end_date",
                    "instant",
                    "dimension_count",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
            .head(30)
        )

        print(
            pattern.to_string(
                index=False
            )
        )

    if not document_df.empty:
        print(
            "\n[Top document financial-table candidates]"
        )

        cols = [
            "stock_code",
            "period_key",
            "rcept_no",
            "table_index",
            "keyword_score",
            "keywords",
            "parsed_shape",
            "preceding_text",
        ]

        print(
            document_df[
                cols
            ]
            .sort_values(
                [
                    "keyword_score",
                    "stock_code",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(30)
            .to_string(
                index=False
            )
        )

    print(
        f"\nsummary : {SOURCE_SUMMARY_PATH}"
    )
    print(
        f"contexts: {CONTEXT_PATH}"
    )
    print(
        f"tables  : {DOCUMENT_TABLE_PATH}"
    )

    print(
        "\n다음 단계:"
        "\n1) XBRL context 패턴을 이용해 current-period CFS fact 선택 규칙 확정"
        "\n2) document fallback에서 재무제표 표 식별 규칙 확정"
        "\n3) strict-PIT snapshot parser 작성"
        "\n4) source unavailable interval은 절대 최신값/0/ffill로 메우지 않음"
    )


if __name__ == "__main__":
    main()
