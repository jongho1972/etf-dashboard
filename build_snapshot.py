"""ETF 데이터 스냅샷 빌더.

GitHub Actions cron(매일 06:00 KST)에서 실행되어 yfinance로 전종목 1년치를
수집하고 `etf_data.parquet`로 저장한다. 장중 미확정 종가를 배제하기 위해
KST 기준 D-1 종가까지만 사용한다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from warnings import filterwarnings

import pandas as pd
import yfinance as yf

filterwarnings("ignore")

ROOT = Path(__file__).parent
ETF_LIST_CSV = ROOT / "etf_list.csv"
ETF_DATA_PARQUET = ROOT / "etf_data.parquet"


def _trim_to_dminus1(hist: pd.DataFrame) -> pd.DataFrame:
    kst_today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    idx = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    return hist.loc[idx.normalize() < kst_today]


def fetch_etf_data(item: str):
    try:
        ticker = yf.Ticker(f"{item}.KS")
        hist = ticker.history(period="1y", actions=True)
        hist = _trim_to_dminus1(hist)

        if hist.empty or len(hist) < 20:
            return None

        close = hist["Close"].dropna()
        close = close[close > 0]
        if len(close) < 20:
            return None

        latest_date = close.index.max()
        three_months_ago_date = latest_date - pd.DateOffset(months=3)
        one_year_ago_date = close.index.min()

        latest_price = close.loc[close.index <= latest_date].iloc[-1]
        three_months_ago_price = close.loc[close.index <= three_months_ago_date].iloc[-1]
        one_year_ago_price = close.loc[close.index <= one_year_ago_date].iloc[-1]

        if not (latest_price > 0 and three_months_ago_price > 0 and one_year_ago_price > 0):
            return None

        dividends = hist[hist["Dividends"] > 0]["Dividends"]
        dividend_day = "기타"

        if not dividends.empty:
            annual_dividend = dividends.sum()
            month_dividend = dividends.iloc[-1]
            last_dividend_date = dividends.index[-1]

            if len(dividends) >= 2:
                dividend_term = dividends.index[-1] - dividends.index[-2]
                if 25 <= dividend_term.days <= 35:
                    dividend_day = "월초" if dividends.index[-1].day > 20 else "월중"

            return {
                "Symbol": item,
                "주가": latest_price,
                "주가_3M": three_months_ago_price,
                "주가_1Y": one_year_ago_price,
                "월배당금": month_dividend,
                "연간배당금": annual_dividend,
                "배당일": dividend_day,
                "기준일": latest_date.strftime("%Y-%m-%d"),
                "기준일_배당": last_dividend_date.strftime("%Y-%m-%d"),
            }

        return {
            "Symbol": item,
            "주가": latest_price,
            "주가_3M": three_months_ago_price,
            "주가_1Y": one_year_ago_price,
            "월배당금": 0,
            "연간배당금": 0,
            "배당일": dividend_day,
            "기준일": latest_date.strftime("%Y-%m-%d"),
            "기준일_배당": "",
        }
    except Exception:
        return None


def build() -> pd.DataFrame:
    etfs_list = pd.read_csv(ETF_LIST_CSV, dtype={"Symbol": str})
    symbol_list = [s for s in etfs_list["Symbol"].tolist() if s not in {"265690"}]

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(fetch_etf_data, symbol_list))

    data = [r for r in results if r is not None]
    result = pd.DataFrame(data)

    result["1M배당률"] = (result["월배당금"] / result["주가"] * 100).round(2)
    result["1Y배당률"] = (result["연간배당금"] / result["주가_1Y"] * 100).round(2)
    result["3M수익률"] = ((result["주가"] - result["주가_3M"]) / result["주가_3M"] * 100).round(2)
    result["1Y수익률"] = ((result["주가"] - result["주가_1Y"]) / result["주가_1Y"] * 100).round(2)

    final = pd.merge(etfs_list, result, on="Symbol", how="inner")
    final = final.fillna(0)
    final = final.astype({"주가": "int32", "월배당금": "int32"})
    final = final.rename(columns={"MarCap": "시총"})

    return final


def main() -> None:
    df = build()
    df.to_parquet(ETF_DATA_PARQUET, compression="snappy", index=False)
    print(f"saved {ETF_DATA_PARQUET.name} · rows={len(df):,} · 기준일(D-1)={df['기준일'].max()}")


if __name__ == "__main__":
    main()
