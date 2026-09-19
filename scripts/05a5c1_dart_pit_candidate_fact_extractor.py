from __future__ import annotations

import io
import json
import re
import zipfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from lxml import etree


warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


# ============================================================
# 05A5-C1. DART PIT Candidate Fact Extractor
#
# 목적
# ------------------------------------------------------------
# 이번 단계에서는 "최종 재무값"을 확정하지 않는다.
#
# XBRL:
#   핵심 계정 후보 fact를 모두 추출
#   + context 기간
#   + dimension/member
#   + unit
#   + concept namespace
#
# document fallback:
#   실제 재무제표 가능성이 있는 table에서
#   핵심 계정 row 후보를 모두 추출
#   + table heading
#   + 연결/별도 힌트
#   + 단위
#   + 숫자 후보
#
# WHY
# ------------------------------------------------------------
# - 하나의 XBRL concept에 당기/전기/누적/3개월/연결/별도 context가
#   동시에 존재할 수 있음.
# - document.xml은 회사 소개/주석/요약표에도 같은 계정명이 등장함.
# - 따라서 지금 자동으로 "정답 숫자 1개"를 고르면 PIT보다 더 큰
#   statement-selection 오류를 만들 수 있음.
#
# INPUT
# ------------------------------------------------------------
# data/raw/dart/pit_source_audit/**/*.zip
#
# OUTPUT
# ------------------------------------------------------------
# data/interim/dart/
#   dart_xbrl_fact_candidates.csv
#   dart_xbrl_fact_candidates.parquet
#   dart_document_fact_candidates.csv
#   dart_document_fact_candidates.parquet
#   dart_pit_candidate_extraction_summary.csv
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

XBRL_CSV = OUT_DIR / "dart_xbrl_fact_candidates.csv"
XBRL_PARQUET = OUT_DIR / "dart_xbrl_fact_candidates.parquet"

DOC_CSV = OUT_DIR / "dart_document_fact_candidates.csv"
DOC_PARQUET = OUT_DIR / "dart_document_fact_candidates.parquet"

SUMMARY_CSV = OUT_DIR / "dart_pit_candidate_extraction_summary.csv"


# ------------------------------------------------------------
# Target account families
# ------------------------------------------------------------

# XBRL local-name 후보.
# 여기서는 넓게 가져오고 다음 단계에서 실제 concept/context를 확정한다.
XBRL_CONCEPT_PATTERNS: dict[str, list[str]] = {
    "assets": [
        r"^Assets$",
    ],
    "liabilities": [
        r"^Liabilities$",
    ],
    "equity": [
        r"^Equity$",
        r"^EquityAttributableToOwnersOfParent$",
    ],
    "revenue": [
        r"^Revenue$",
        r"^SalesRevenue$",
        r"^RevenueFromContractsWithCustomers.*$",
        r"^OperatingRevenue$",
    ],
    "operating_income": [
        r"^OperatingIncomeLoss$",
        r"^OperatingProfitLoss$",
        r"^OperatingIncome$",
        r"^OperatingProfit$",
    ],
    "net_income": [
        r"^ProfitLoss$",
        r"^NetIncomeLoss$",
        r"^ProfitLossAttributableToOwnersOfParent$",
    ],
}


DOCUMENT_ACCOUNT_PATTERNS: dict[str, list[str]] = {
    "assets": [
        "자산총계",
    ],
    "liabilities": [
        "부채총계",
    ],
    "equity": [
        "자본총계",
    ],
    "revenue": [
        "매출액",
        "영업수익",
        "수익(매출액)",
    ],
    "operating_income": [
        "영업이익",
        "영업이익(손실)",
    ],
    "net_income": [
        "당기순이익",
        "당기순이익(손실)",
        "분기순이익",
        "반기순이익",
    ],
}


STATEMENT_HEADINGS = [
    "연결재무상태표",
    "연결손익계산서",
    "연결포괄손익계산서",
    "연결재무제표",
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "재무제표",
]

CONSOLIDATED_HINTS = [
    "연결",
    "종속기업",
    "연결재무",
]

SEPARATE_HINTS = [
    "별도",
    "개별",
]

NOTE_HINTS = [
    "주석",
    "영업부문",
    "부문별",
    "신탁",
    "주요재무",
    "요약재무",
]


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------

def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


def namespace_uri(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None


def decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    return content.decode("utf-8", errors="replace")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"&cr;", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata_from_path(path: Path) -> dict[str, Any]:
    rel = path.relative_to(RAW_ROOT)
    parts = rel.parts

    stock_code = parts[0] if len(parts) >= 1 else None
    period_key = parts[1] if len(parts) >= 2 else None

    filename = path.stem

    source_type = (
        "xbrl"
        if "_xbrl" in filename
        else (
            "document"
            if "_document" in filename
            else "unknown"
        )
    )

    receipt_match = re.search(r"(\d{14})", filename)
    rcept_no = receipt_match.group(1) if receipt_match else None

    date_match = re.match(r"(\d{8})_", filename)
    rcept_dt = (
        pd.to_datetime(date_match.group(1), format="%Y%m%d")
        if date_match
        else pd.NaT
    )

    is_correction = "_correction_" in filename

    return {
        "stock_code": stock_code,
        "period_key": period_key,
        "rcept_no": rcept_no,
        "rcept_dt": rcept_dt,
        "is_correction": is_correction,
        "source_type": source_type,
        "zip_path": str(path),
    }


def find_instance_member(zf: zipfile.ZipFile) -> str | None:
    names = zf.namelist()

    xbrl_files = [
        name
        for name in names
        if name.lower().endswith(".xbrl")
    ]

    if xbrl_files:
        return max(
            xbrl_files,
            key=lambda n: zf.getinfo(n).file_size,
        )

    return None


# ------------------------------------------------------------
# XBRL
# ------------------------------------------------------------

def classify_xbrl_concept(local: str) -> str | None:
    for family, patterns in XBRL_CONCEPT_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, local, flags=re.I):
                return family
    return None


def parse_xbrl_context(node: etree._Element) -> dict[str, Any]:
    context_id = node.get("id")

    start_date = None
    end_date = None
    instant = None
    entity_identifier = None

    dimensions: list[dict[str, str]] = []

    for child in node.iter():
        name = local_name(child.tag)

        if name == "identifier":
            entity_identifier = clean_text(child.text)

        elif name == "startDate":
            start_date = clean_text(child.text)

        elif name == "endDate":
            end_date = clean_text(child.text)

        elif name == "instant":
            instant = clean_text(child.text)

        elif name == "explicitMember":
            dimensions.append(
                {
                    "type": "explicit",
                    "dimension": child.get("dimension") or "",
                    "member": clean_text(child.text),
                }
            )

        elif name == "typedMember":
            typed_value = clean_text(
                " ".join(child.itertext())
            )
            dimensions.append(
                {
                    "type": "typed",
                    "dimension": child.get("dimension") or "",
                    "member": typed_value,
                }
            )

    duration_days = None
    if start_date and end_date:
        try:
            duration_days = (
                pd.Timestamp(end_date)
                - pd.Timestamp(start_date)
            ).days + 1
        except Exception:
            duration_days = None

    dimension_text = " | ".join(
        f"{d['dimension']}={d['member']}"
        for d in dimensions
    )

    return {
        "context_id": context_id,
        "entity_identifier": entity_identifier,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "instant": instant or None,
        "duration_days": duration_days,
        "dimension_count": len(dimensions),
        "dimension_text": dimension_text,
        "dimensions_json": json.dumps(
            dimensions,
            ensure_ascii=False,
        ),
    }


def parse_numeric_text(value: str) -> float | None:
    text = clean_text(value)

    if not text:
        return None

    # XBRL numeric facts are normally plain numeric strings.
    text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def extract_xbrl_candidates(path: Path) -> list[dict[str, Any]]:
    metadata = extract_metadata_from_path(path)

    with zipfile.ZipFile(path) as zf:
        member = find_instance_member(zf)

        if member is None:
            return []

        content = zf.read(member)

    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
    )

    root = etree.fromstring(
        content,
        parser=parser,
    )

    contexts = {}

    for node in root.iter():
        if local_name(node.tag) == "context":
            parsed = parse_xbrl_context(node)
            if parsed["context_id"]:
                contexts[parsed["context_id"]] = parsed

    rows = []

    for node in root.iter():
        context_ref = node.get("contextRef")

        if not context_ref:
            continue

        local = local_name(node.tag)

        family = classify_xbrl_concept(local)

        if family is None:
            continue

        ctx = contexts.get(
            context_ref,
            {},
        )

        raw_value = clean_text(node.text)

        rows.append(
            {
                **metadata,
                "instance_member": member,
                "account_family": family,
                "concept_local_name": local,
                "concept_namespace": namespace_uri(node.tag),
                "context_ref": context_ref,
                "unit_ref": node.get("unitRef"),
                "decimals": node.get("decimals"),
                "scale": node.get("scale"),
                "raw_value": raw_value,
                "numeric_value": parse_numeric_text(raw_value),
                **ctx,
            }
        )

    return rows


# ------------------------------------------------------------
# document.xml fallback
# ------------------------------------------------------------

def flatten_columns(columns: Any) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        result = []
        for tup in columns.to_list():
            parts = [
                clean_text(x)
                for x in tup
                if clean_text(x)
                and not clean_text(x).startswith("Unnamed")
            ]
            result.append(" | ".join(parts))
        return result

    return [
        clean_text(x)
        for x in list(columns)
    ]


def identify_account_family(text: str) -> tuple[str | None, str | None]:
    normalized = clean_text(text)

    for family, names in DOCUMENT_ACCOUNT_PATTERNS.items():
        # 긴 표현 먼저
        for name in sorted(names, key=len, reverse=True):
            if name in normalized:
                return family, name

    return None, None


def parse_korean_number(value: Any) -> float | None:
    text = clean_text(value)

    if not text:
        return None

    # 괄호 음수
    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    text = text.strip("()")
    text = text.replace(",", "")
    text = text.replace("△", "-")
    text = text.replace("−", "-")

    # 숫자 아닌 annotation 제거
    text = re.sub(
        r"[^0-9.\-]",
        "",
        text,
    )

    if text in ("", "-", ".", "-."):
        return None

    try:
        num = float(text)
        return -abs(num) if negative else num
    except ValueError:
        return None


def find_unit_hint(text: str) -> str | None:
    patterns = [
        r"단위\s*[:：]?\s*([가-힣A-Za-z]+)",
        r"\(단위\s*[:：]?\s*([가-힣A-Za-z]+)\)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def surrounding_heading(table) -> str:
    candidates = []

    node = table

    for _ in range(12):
        node = node.find_previous()

        if node is None:
            break

        if getattr(node, "name", None) in (
            "p", "div", "span", "title",
            "h1", "h2", "h3", "h4",
            "tu", "te", "td",
        ):
            text = clean_text(
                " ".join(node.stripped_strings)
            )

            if 0 < len(text) <= 300:
                candidates.append(text)

        if len(candidates) >= 8:
            break

    return " || ".join(candidates)


def statement_score(
    heading: str,
    table_text: str,
) -> tuple[int, str]:
    combined = f"{heading} {table_text}"

    score = 0
    reasons = []

    heading_hits = [
        token
        for token in STATEMENT_HEADINGS
        if token in combined
    ]

    if heading_hits:
        score += 5
        reasons.append(
            "statement_heading="
            + ",".join(sorted(set(heading_hits)))
        )

    consolidated_hits = [
        token
        for token in CONSOLIDATED_HINTS
        if token in combined
    ]

    if consolidated_hits:
        score += 3
        reasons.append(
            "consolidated_hint="
            + ",".join(sorted(set(consolidated_hits)))
        )

    separate_hits = [
        token
        for token in SEPARATE_HINTS
        if token in combined
    ]

    if separate_hits:
        score -= 2
        reasons.append(
            "separate_hint="
            + ",".join(sorted(set(separate_hits)))
        )

    note_hits = [
        token
        for token in NOTE_HINTS
        if token in heading
    ]

    if note_hits:
        score -= 4
        reasons.append(
            "note_penalty="
            + ",".join(sorted(set(note_hits)))
        )

    return score, " | ".join(reasons)


def _direct_table_rows(table) -> list[Any]:
    """
    nested table의 <tr>까지 섞이지 않게 현재 table에 직접 속한 row만 반환한다.
    """
    return [
        tr
        for tr in table.find_all("tr")
        if tr.find_parent("table") is table
    ]


def _direct_row_cells(tr) -> list[Any]:
    """
    nested element 내부의 td/th를 중복 집계하지 않고 현재 row cell만 반환한다.
    """
    return [
        cell
        for cell in tr.find_all(["th", "td"])
        if cell.find_parent("tr") is tr
    ]


def _expand_table_grid(table) -> list[list[str]]:
    """
    DART HTML-like table을 pandas.read_html 없이 직접 grid로 펼친다.

    rowspan / colspan을 지원한다.
    값 자체를 반복해서 넣는 이유는 column header 조합에 필요하기 때문이다.
    """
    trs = _direct_table_rows(table)

    grid: list[list[str]] = []
    spans: dict[int, tuple[int, str]] = {}

    for tr in trs:
        row_map: dict[int, str] = {}

        # 이전 row에서 내려온 rowspan 적용
        next_spans: dict[int, tuple[int, str]] = {}

        for col_idx, (remaining, value) in spans.items():
            row_map[col_idx] = value

            if remaining > 1:
                next_spans[col_idx] = (
                    remaining - 1,
                    value,
                )

        cells = _direct_row_cells(tr)
        col_idx = 0

        for cell in cells:
            while col_idx in row_map:
                col_idx += 1

            value = clean_text(
                " ".join(cell.stripped_strings)
            )

            try:
                rowspan = max(
                    int(cell.get("rowspan", 1)),
                    1,
                )
            except Exception:
                rowspan = 1

            try:
                colspan = max(
                    int(cell.get("colspan", 1)),
                    1,
                )
            except Exception:
                colspan = 1

            for offset in range(colspan):
                target_col = col_idx + offset
                row_map[target_col] = value

                if rowspan > 1:
                    next_spans[target_col] = (
                        rowspan - 1,
                        value,
                    )

            col_idx += colspan

        spans = next_spans

        if row_map:
            max_col = max(row_map)
            row_values = [
                row_map.get(i, "")
                for i in range(max_col + 1)
            ]
        else:
            row_values = []

        grid.append(row_values)

    if not grid:
        return []

    width = max(len(row) for row in grid)

    return [
        row + [""] * (width - len(row))
        for row in grid
    ]


def _is_plain_numeric_cell(value: Any) -> bool:
    """
    header를 구성할 때 실제 숫자 cell을 제외하기 위한 가벼운 판정.
    날짜/기간 문구(예: 2023년 반기)는 header로 남긴다.
    """
    text = clean_text(value)

    if not text:
        return False

    normalized = (
        text.replace(",", "")
        .replace("△", "-")
        .replace("−", "-")
        .strip()
    )

    if (
        normalized.startswith("(")
        and normalized.endswith(")")
    ):
        normalized = "-" + normalized[1:-1]

    return bool(
        re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?",
            normalized,
        )
    )


def _build_lightweight_column_labels(
    grid: list[list[str]],
    first_target_row: int,
) -> list[str]:
    """
    target 계정이 처음 등장하기 전 row를 header 후보로 보고
    각 column별 semantic label을 만든다.

    예:
      당반기 / 누적 -> "당반기 | 누적"
      2018년 반기말 / 제185기 반기 -> 그대로 결합
    """
    if not grid:
        return []

    width = len(grid[0])
    labels: list[str] = []

    header_rows = grid[:first_target_row]

    for col_idx in range(width):
        parts: list[str] = []

        for row in header_rows:
            if col_idx >= len(row):
                continue

            value = clean_text(row[col_idx])

            if not value:
                continue

            if _is_plain_numeric_cell(value):
                continue

            # 지나치게 긴 설명문은 column header로 부적절
            if len(value) > 120:
                continue

            if not parts or parts[-1] != value:
                parts.append(value)

        # 가장 가까운 header 정보 위주로 유지
        parts = parts[-4:]

        labels.append(
            " | ".join(parts)
            if parts
            else f"col_{col_idx}"
        )

    return labels


def extract_document_candidates(path: Path) -> list[dict[str, Any]]:
    metadata = extract_metadata_from_path(path)

    with zipfile.ZipFile(path) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.lower().endswith(
                (".xml", ".html", ".htm")
            )
        ]

        if not members:
            return []

        member = max(
            members,
            key=lambda n: zf.getinfo(n).file_size,
        )

        raw = zf.read(member)

    text = decode_bytes(raw)

    # DART document.xml은 XML container이지만 내부 table markup은
    # HTML-like structure가 많다. 기존 audit과 동일한 lxml HTML parser를
    # 사용하되 pd.read_html은 사용하지 않는다.
    soup = BeautifulSoup(
        text,
        "lxml",
    )

    tables = soup.find_all("table")

    rows: list[dict[str, Any]] = []

    target_names = [
        name
        for names in DOCUMENT_ACCOUNT_PATTERNS.values()
        for name in names
    ]

    for table_index, table in enumerate(tables):
        table_text = clean_text(
            table.get_text(
                " ",
                strip=True,
            )
        )

        # 매우 싼 pre-filter.
        # 핵심 계정명이 전혀 없으면 grid 자체를 만들지 않는다.
        if not any(
            name in table_text
            for name in target_names
        ):
            continue

        heading = surrounding_heading(table)

        unit_hint = find_unit_hint(
            f"{heading} {table_text[:1000]}"
        )

        score, score_reason = statement_score(
            heading,
            table_text,
        )

        # pandas.read_html 대신 직접 lightweight grid 구성
        try:
            grid = _expand_table_grid(table)
        except Exception:
            continue

        if not grid:
            continue

        # 첫 target 계정 row를 찾아 그 이전 row를 header 후보로 사용
        first_target_row = len(grid)

        for idx, grid_row in enumerate(grid):
            joined = " | ".join(
                clean_text(value)
                for value in grid_row
            )

            family, _ = identify_account_family(joined)

            if family is not None:
                first_target_row = idx
                break

        flat_cols = _build_lightweight_column_labels(
            grid,
            first_target_row,
        )

        if not flat_cols:
            flat_cols = [
                f"col_{i}"
                for i in range(len(grid[0]))
            ]

        for row_index, grid_row in enumerate(grid):
            cell_texts = [
                clean_text(value)
                for value in grid_row
            ]

            joined = " | ".join(cell_texts)

            family, matched_name = (
                identify_account_family(joined)
            )

            if family is None:
                continue

            numeric_candidates = []

            for col_idx, cell in enumerate(cell_texts):
                num = parse_korean_number(cell)

                if num is None:
                    continue

                col_name = (
                    flat_cols[col_idx]
                    if col_idx < len(flat_cols)
                    else f"col_{col_idx}"
                )

                numeric_candidates.append(
                    {
                        "column": col_name,
                        "raw": cell,
                        "numeric": num,
                    }
                )

            rows.append(
                {
                    **metadata,
                    "document_member": member,
                    "table_index": table_index,
                    "row_index": int(row_index),
                    "account_family": family,
                    "matched_account_name": matched_name,
                    "statement_score": score,
                    "statement_score_reason": score_reason,
                    "unit_hint": unit_hint,
                    "heading_context": heading,
                    "columns_json": json.dumps(
                        flat_cols,
                        ensure_ascii=False,
                    ),
                    "row_cells_json": json.dumps(
                        cell_texts,
                        ensure_ascii=False,
                    ),
                    "numeric_candidates_json": json.dumps(
                        numeric_candidates,
                        ensure_ascii=False,
                    ),
                    "row_text": joined,
                    "document_parse_engine": (
                        "beautifulsoup_lightweight_grid_v2"
                    ),
                }
            )

    return rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"PIT source audit 폴더가 없습니다: {RAW_ROOT}"
        )

    zip_paths = sorted(
        RAW_ROOT.rglob("*.zip")
    )

    xbrl_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []

    summary_rows = []

    print("\n" + "=" * 80)
    print("05A5-C1 PIT CANDIDATE FACT EXTRACTION")
    print("=" * 80)
    print(f"zip files: {len(zip_paths)}")

    for i, path in enumerate(zip_paths, start=1):
        metadata = extract_metadata_from_path(path)

        print(
            f"\n[{i}/{len(zip_paths)}] "
            f"{metadata['stock_code']} | "
            f"{metadata['period_key']} | "
            f"{metadata['source_type']} | "
            f"{metadata['rcept_no']}"
        )

        try:
            if metadata["source_type"] == "xbrl":
                rows = extract_xbrl_candidates(path)
                xbrl_rows.extend(rows)

                print(
                    f"  XBRL candidate facts: {len(rows)}"
                )

                summary_rows.append(
                    {
                        **metadata,
                        "status": "ok",
                        "candidate_count": len(rows),
                    }
                )

            elif metadata["source_type"] == "document":
                rows = extract_document_candidates(path)
                doc_rows.extend(rows)

                print(
                    f"  document candidate rows: {len(rows)}"
                )

                summary_rows.append(
                    {
                        **metadata,
                        "status": "ok",
                        "candidate_count": len(rows),
                    }
                )

            else:
                summary_rows.append(
                    {
                        **metadata,
                        "status": "unknown_source",
                        "candidate_count": 0,
                    }
                )

        except Exception as exc:
            print(
                "  ERROR:",
                repr(exc),
            )

            summary_rows.append(
                {
                    **metadata,
                    "status": "error",
                    "candidate_count": 0,
                    "error": repr(exc),
                }
            )

    xbrl_df = pd.DataFrame(xbrl_rows)
    doc_df = pd.DataFrame(doc_rows)
    summary_df = pd.DataFrame(summary_rows)

    xbrl_df.to_csv(
        XBRL_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    xbrl_df.to_parquet(
        XBRL_PARQUET,
        index=False,
    )

    doc_df.to_csv(
        DOC_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    doc_df.to_parquet(
        DOC_PARQUET,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    print(
        summary_df[
            [
                "stock_code",
                "period_key",
                "rcept_no",
                "source_type",
                "status",
                "candidate_count",
            ]
        ].to_string(index=False)
    )

    if not xbrl_df.empty:
        print("\n[XBRL account families]")
        print(
            xbrl_df[
                "account_family"
            ]
            .value_counts()
            .to_string()
        )

        print("\n[XBRL concept names]")
        print(
            xbrl_df[
                [
                    "account_family",
                    "concept_local_name",
                ]
            ]
            .value_counts()
            .head(50)
            .to_string()
        )

        print("\n[XBRL dimension patterns for candidate facts]")
        dims = (
            xbrl_df[
                [
                    "dimension_count",
                    "dimension_text",
                ]
            ]
            .value_counts()
            .head(30)
        )

        print(
            dims.to_string()
        )

        print("\n[XBRL period patterns for candidate facts]")
        periods = (
            xbrl_df[
                [
                    "start_date",
                    "end_date",
                    "instant",
                    "duration_days",
                ]
            ]
            .value_counts(
                dropna=False
            )
            .head(30)
        )

        print(
            periods.to_string()
        )

    if not doc_df.empty:
        print("\n[Document candidate account families]")
        print(
            doc_df[
                "account_family"
            ]
            .value_counts()
            .to_string()
        )

        print("\n[Highest-ranked document candidates]")
        cols = [
            "stock_code",
            "period_key",
            "rcept_no",
            "table_index",
            "row_index",
            "account_family",
            "matched_account_name",
            "statement_score",
            "unit_hint",
            "statement_score_reason",
            "heading_context",
            "row_text",
        ]

        print(
            doc_df[
                cols
            ]
            .sort_values(
                [
                    "statement_score",
                    "stock_code",
                    "table_index",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .head(40)
            .to_string(
                index=False
            )
        )

    print(f"\nXBRL candidates : {XBRL_CSV}")
    print(f"Document candidates: {DOC_CSV}")
    print(f"Summary         : {SUMMARY_CSV}")

    print(
        "\n다음 판단:"
        "\n1) XBRL에서 CFS를 나타내는 dimension/member 문자열 확인"
        "\n2) H1에서 6개월 누적 vs 3개월 context를 구분"
        "\n3) document fallback에서 연결재무제표 heading이 있는 table만 우선"
        "\n4) 이 규칙을 검증한 뒤에만 최종 snapshot 값 1개를 선택"
    )


if __name__ == "__main__":
    main()
