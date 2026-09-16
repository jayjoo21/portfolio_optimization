from pathlib import Path
import os

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("KRX_API_KEY")

if not api_key:
    raise ValueError(
        "KRX_API_KEY가 .env에 없습니다."
    )


url = (
    "https://data-dbg.krx.co.kr"
    "/svc/apis/idx/krx_dd_trd"
)

headers = {
    "AUTH_KEY": api_key
}

params = {
    "basDd": "20260915"
}


response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
)

print(
    "API key loaded:",
    bool(api_key)
)

print(
    "HTTP status:",
    response.status_code
)

response.raise_for_status()

data = response.json()

rows = data.get(
    "OutBlock_1",
    []
)

print(
    "Rows:",
    len(rows)
)

krx300_rows = [
    row
    for row in rows
    if row.get("IDX_NM") == "KRX 300"
]

print(
    "KRX300 rows:",
    len(krx300_rows)
)

for row in krx300_rows:

    print({
        "date": row.get("BAS_DD"),
        "name": row.get("IDX_NM"),
        "close": row.get("CLSPRC_IDX"),
        "change_pct": row.get("FLUC_RT")
    })
