"""ETF 목록 스냅샷 갱신 스크립트.

snapshot.yml이 스냅샷 빌드 전에 매일 실행해 시총·신규 상장·상장폐지를 반영한다
(실패해도 기존 csv로 계속 진행하고, select_default.py가 메일로 경고한다).
응답이 비정상적으로 적으면 기존 목록을 덮어쓰지 않는다.

사용법:
    python update_etf_list.py
"""
import sys
from pathlib import Path
import requests
import pandas as pd

URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.naver.com/sise/etf.naver",
}
OUT = Path(__file__).parent / "etf_list.csv"
MIN_LIST_ROWS = 500  # 이보다 적으면 응답 이상으로 보고 기존 목록 유지
# 네이버 ETF 화면의 분류 탭(etfTabCode). expert_picks.py가 분류별로 후보를 고른다
CATEGORIES = {1: "국내지수", 2: "국내테마", 3: "국내파생", 4: "해외주식", 5: "원자재", 6: "채권", 7: "기타"}


def main() -> int:
    r = requests.get(URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    rows = r.json()["result"]["etfItemList"]
    df = pd.DataFrame(rows).rename(columns={"itemcode": "Symbol", "itemname": "Name", "marketSum": "MarCap"})
    unknown = sorted(set(df["etfTabCode"]) - set(CATEGORIES))
    if unknown:   # 네이버가 탭 체계를 바꾸면 분류별 후보 선정이 조용히 무너지므로 알린다
        print(f"경고: 모르는 분류 코드 {unknown} — '기타'로 처리", file=sys.stderr)
    df = df.assign(분류=df["etfTabCode"].map(CATEGORIES).fillna("기타"))[["Symbol", "Name", "MarCap", "분류"]]
    if len(df) < MIN_LIST_ROWS:
        print(f"응답 {len(df)}행 < {MIN_LIST_ROWS} — 기존 목록 유지", file=sys.stderr)
        return 1
    old = set(pd.read_csv(OUT, dtype={"Symbol": str})["Symbol"]) if OUT.exists() else set()
    new = set(df["Symbol"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {len(df)} rows -> {OUT.name} (신규 {len(new - old)} · 제외 {len(old - new)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
