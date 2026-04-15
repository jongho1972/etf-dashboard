"""ETF 목록 스냅샷 갱신 스크립트.

Streamlit Cloud에서 finance.naver.com이 차단되므로 로컬에서 이 스크립트를
실행해 etf_list.csv 를 갱신한 뒤 커밋/푸시한다.

사용법:
    python update_etf_list.py
"""
from pathlib import Path
import requests
import pandas as pd

URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.naver.com/sise/etf.naver",
}
OUT = Path(__file__).parent / "etf_list.csv"


def main():
    r = requests.get(URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    rows = r.json()["result"]["etfItemList"]
    df = (
        pd.DataFrame(rows)
        .rename(columns={"itemcode": "Symbol", "itemname": "Name", "marketSum": "MarCap"})
        [["Symbol", "Name", "MarCap"]]
    )
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {len(df)} rows -> {OUT.name}")


if __name__ == "__main__":
    main()
