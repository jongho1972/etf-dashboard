"""ETF 데이터 스냅샷 빌더.

snapshot.yml(평일 16:00 KST)에서 실행되어 yfinance로 전종목 2년치를 수집하고
`etf_data.parquet`와 수집 결과 `snapshot_report.json`을 저장한다. KRX 장마감(15:30 KST)
이후 실행 시 당일 종가를 포함하고, 그 전에는 D-1까지만 사용해 장중 미확정 종가를 배제한다.

지표 정의 (2026-10-31 정정)
- 주가·3M/1Y수익률: 실제 종가(auto_adjust=False) 기준 순수 가격 변동. 배당은 별도로 더한다.
  yfinance 기본값(수정주가)을 쓰면 수익률에 배당이 이미 들어가 What-if에서 배당이 이중 합산된다.
- 3M/1Y총수익률: 배당 재투자 포함(Adj Close). 디폴트 교체 비교용.
- 연간배당금: 배당 주기(최근 지급 간격 중앙값)로 연 지급 횟수 N을 정하고 최근 N회 합계.
  기간(365일)으로 자르면 지급일이 하루만 밀려도 13회·5회가 잡혀 과대 계산된다.
  상장 1년 미만이라 N회가 안 되면 평균 × N으로 연환산하고 `상장1Y미만`으로 표시한다.
- 월배당금: 월배당 종목은 최근 1회, 그 외는 연간배당금 / 12.
- 1Y배당률: 연간배당금 / 현재 주가 (What-if의 매수 기준 가격과 동일).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from warnings import filterwarnings

import pandas as pd
import yfinance as yf

filterwarnings("ignore")

ROOT = Path(__file__).parent
ETF_LIST_CSV = ROOT / "etf_list.csv"
ETF_DATA_PARQUET = ROOT / "etf_data.parquet"
REPORT_JSON = ROOT / "snapshot_report.json"

EXCLUDE_SYMBOLS = {"265690"}
MIN_ROWS = 20
MIN_SUCCESS_RATE = 0.95   # 이보다 적게 수집되면 parquet을 덮어쓰지 않고 실패 처리
# 지급 간격 중앙값(일) 상한 → 연 지급 횟수
FREQUENCIES = ((10, 52), (45, 12), (120, 4), (240, 2), (10_000, 1))


def _trim_to_confirmed_close(hist: pd.DataFrame) -> pd.DataFrame:
    """KRX 장마감(15:30 KST) 이후면 당일 종가까지 포함, 그 전이면 D-1까지만.
    app.py 비교 차트도 이 함수를 쓴다. hist 인덱스는 tz 없는 KST 날짜여야 한다."""
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    kst_today = now_kst.normalize()
    market_closed = now_kst >= (kst_today + pd.Timedelta(hours=15, minutes=30))
    cutoff = kst_today if market_closed else kst_today - pd.Timedelta(days=1)
    return hist.loc[hist.index.normalize() <= cutoff]


def _history(item: str) -> pd.DataFrame:
    """실제 종가 + Adj Close + 배당. 예외·빈 응답만 1회 재시도(짧은 이력은 정상 응답).
    2년치를 받는 이유: 연 1·2회 지급 종목의 배당 주기 판단."""
    hist = pd.DataFrame()
    for attempt in range(2):
        try:
            hist = yf.Ticker(f"{item}.KS").history(period="2y", actions=True, auto_adjust=False)
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty:
            break
        if attempt == 0:
            time.sleep(2)
    if hist.empty:
        return hist
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    return _trim_to_confirmed_close(hist)


def _price_on_or_before(s: pd.Series, date: pd.Timestamp) -> float:
    """date 이전 마지막 값. 상장 전이면 첫 값."""
    before = s.loc[s.index <= date]
    return before.iloc[-1] if not before.empty else s.iloc[0]


def _dividend_stats(div: pd.Series) -> tuple[float, float, str]:
    """(연간배당금, 월배당금, 배당일)"""
    if div.empty:
        return 0.0, 0.0, "기타"
    if len(div) == 1:
        # 주기를 알 수 없음: 상장 1년 이상이면 연 1회 지급으로 본다
        return float(div.iloc[0]), float(div.iloc[0]) / 12, "기타"

    # 최근 6개 간격만 본다 (분기→월배당 전환 종목 대응)
    gap = div.index[-7:].to_series().diff().dt.days.dropna().median()
    per_year = next(n for limit, n in FREQUENCIES if gap <= limit)
    recent = div.iloc[-per_year:]
    annual = float(recent.sum()) if len(recent) == per_year else float(recent.mean() * per_year)
    if per_year == 12:
        label = "월초" if div.index[-1].day > 20 else "월중"
        return annual, float(div.iloc[-1]), label
    return annual, annual / 12, "기타"


def fetch_etf_data(item: str) -> dict | str:
    """성공 시 지표 dict, 실패 시 사유 문자열."""
    hist = _history(item)
    if len(hist) < MIN_ROWS:
        return "yfinance 시세 없음" if len(hist) <= 1 else f"상장 {len(hist)}거래일 (최소 {MIN_ROWS}일)"

    valid = hist["Close"].notna() & (hist["Close"] > 0)
    close, adj = hist.loc[valid, "Close"], hist.loc[valid, "Adj Close"]
    if len(close) < MIN_ROWS:
        return "유효 종가 부족"

    latest = close.index.max()
    d_1y, d_3m = latest - pd.DateOffset(years=1), latest - pd.DateOffset(months=3)
    short_history = close.index.min() > d_1y + pd.Timedelta(days=7)

    div = hist.loc[hist["Dividends"] > 0, "Dividends"]
    annual, monthly, label = _dividend_stats(div)

    def ret(s: pd.Series, date: pd.Timestamp) -> float:
        return round((s.iloc[-1] / _price_on_or_before(s, date) - 1) * 100, 2)

    return {
        "Symbol": item,
        "주가": close.iloc[-1],
        "주가_3M": _price_on_or_before(close, d_3m),   # portfolio-advisor가 참조
        "주가_1Y": _price_on_or_before(close, d_1y),
        "3M수익률": ret(close, d_3m),
        "1Y수익률": ret(close, d_1y),
        "월배당금": monthly,
        "연간배당금": annual,
        "배당일": label,
        "3M총수익률": ret(adj, d_3m),
        "1Y총수익률": ret(adj, d_1y),
        "상장1Y미만": bool(short_history),
        "기준일": latest.strftime("%Y-%m-%d"),
        "기준일_배당": div.index[-1].strftime("%Y-%m-%d") if not div.empty else "",
    }


def build() -> tuple[pd.DataFrame, dict]:
    etfs_list = pd.read_csv(ETF_LIST_CSV, dtype={"Symbol": str})
    etfs_list = etfs_list[~etfs_list["Symbol"].isin(EXCLUDE_SYMBOLS)]
    symbols = etfs_list["Symbol"].tolist()

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = dict(zip(symbols, executor.map(fetch_etf_data, symbols)))

    result = pd.DataFrame([r for r in results.values() if isinstance(r, dict)])
    result["1M배당률"] = (result["월배당금"] / result["주가"] * 100).round(2)
    result["1Y배당률"] = (result["연간배당금"] / result["주가"] * 100).round(2)

    final = pd.merge(etfs_list, result, on="Symbol", how="inner").fillna({"MarCap": 0})
    final["주가"] = final["주가"].round().astype("int64")
    final["월배당금"] = final["월배당금"].round().astype("int64")
    final = final.rename(columns={"MarCap": "시총"})

    failed = etfs_list.assign(reason=etfs_list["Symbol"].map(results))
    failed = failed[failed["reason"].map(lambda r: isinstance(r, str))].sort_values("MarCap", ascending=False)
    report = {
        "기준일": final["기준일"].max(),
        "목록": len(etfs_list),
        "수집": len(final),
        "누락": [{"Symbol": r.Symbol, "Name": r.Name, "시총": int(r.MarCap), "사유": r.reason}
                for r in failed.itertuples()],
    }
    return final, report


def main() -> int:
    df, report = build()
    rate = report["수집"] / report["목록"]
    print(f"rows={report['수집']:,}/{report['목록']:,} ({rate:.1%}) · 기준일={report['기준일']}")
    for m in report["누락"][:10]:
        print(f"  누락: {m['Name']} ({m['Symbol']}, 시총 {m['시총']:,}억) — {m['사유']}")
    if rate < MIN_SUCCESS_RATE:
        print(f"수집률 {rate:.1%} < {MIN_SUCCESS_RATE:.0%} — 기존 스냅샷 유지", file=sys.stderr)
        return 1
    df.to_parquet(ETF_DATA_PARQUET, compression="snappy", index=False)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
