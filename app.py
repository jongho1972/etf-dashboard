import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from warnings import filterwarnings

filterwarnings("ignore")

st.set_page_config(
    page_title="ETF 투자 대시보드",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────

def fetch_etf_data(item):
    try:
        ticker = yf.Ticker(f"{item}.KS")
        hist = ticker.history(period="1y", actions=True)

        if hist.empty or len(hist) < 20:
            return None

        latest_date = hist.index.max()
        three_months_ago_date = latest_date - pd.DateOffset(months=3)
        one_year_ago_date = hist.index.min()

        latest_price = hist.loc[hist.index <= latest_date, "Close"].iloc[-1]
        three_months_ago_price = hist.loc[hist.index <= three_months_ago_date, "Close"].iloc[-1]
        one_year_ago_price = hist.loc[hist.index <= one_year_ago_date, "Close"].iloc[-1]

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
        else:
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_etf_data():
    etfs = fdr.StockListing("ETF/KR")
    etfs_list = etfs[["Symbol", "Name", "MarCap"]]
    symbol_list = etfs_list["Symbol"].tolist()
    symbol_list = [s for s in symbol_list if s not in {"265690"}]

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


# ─────────────────────────────────────────────
# 시뮬레이션 함수
# ─────────────────────────────────────────────

def simul(final_df, etf_name, cash):
    target = final_df[final_df["Name"] == etf_name]
    if target.empty or cash <= 0:
        return None

    price = target["주가"].iloc[0]
    price_pct_3m = target["3M수익률"].iloc[0] / 100
    price_pct_1y = target["1Y수익률"].iloc[0] / 100
    div_pct_1y = target["1Y배당률"].iloc[0] / 100
    share_cnt = int(cash // price)

    raw_dividend = price * div_pct_1y * share_cnt
    dividend_annual = int(raw_dividend * (1 - 0.154))
    dividend_monthly = dividend_annual // 12

    raw_profit_3m = price * price_pct_3m * share_cnt
    profit_3m = int(raw_profit_3m * (1 - 0.154)) if raw_profit_3m > 0 else int(raw_profit_3m)

    raw_profit_1y = price * price_pct_1y * share_cnt
    profit_1y = int(raw_profit_1y * (1 - 0.154)) if raw_profit_1y > 0 else int(raw_profit_1y)

    배당일 = target["배당일"].iloc[0]

    return {
        "종목": etf_name,
        "배당일": 배당일,
        "예상 투자금": cash,
        "주식수": share_cnt,
        "연배당금": dividend_annual,
        "월배당금": dividend_monthly,
        "주가차익(3M)": profit_3m,
        "총수익(3M)": dividend_annual + profit_3m,
        "주가차익(1Y)": profit_1y,
        "총수익(1Y)": dividend_annual + profit_1y,
    }


# ─────────────────────────────────────────────
# Name 컬럼 고정 테이블 (모바일 지원)
# ─────────────────────────────────────────────

def sticky_dataframe(df, fmt=None, height=760):
    disp = df.copy().reset_index(drop=True)
    if fmt:
        for col, pattern in fmt.items():
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: pattern.format(x) if pd.notna(x) else "")
    disp = disp.rename(columns={"시총": "시총(억원)", "주가": "주가(원)", "월배당금": "월배당금(원)"})

    def th(col, i):
        pin = "position:sticky;left:0;z-index:3;" if i == 0 else "z-index:1;"
        align = "left" if i == 0 else "right"
        return (f'<th style="position:sticky;top:0;{pin}'
                f'background:#f0f2f6;color:#31333f;padding:6px 12px;'
                f'text-align:{align};white-space:nowrap;border-bottom:2px solid #ccc;">'
                f'{col}</th>')

    headers = "".join(th(col, i) for i, col in enumerate(disp.columns))
    rows = ""
    for idx, row in disp.iterrows():
        bg = "#ffffff" if idx % 2 == 0 else "#f5f7fb"
        tds = ""
        for i, val in enumerate(row):
            if i == 0:
                tds += (f'<td style="position:sticky;left:0;background:{bg};'
                        f'padding:6px 12px;white-space:nowrap;'
                        f'border-right:2px solid #ccc;font-weight:500;">{val}</td>')
            else:
                tds += f'<td style="padding:6px 12px;white-space:nowrap;background:{bg};text-align:right;font-variant-numeric:tabular-nums;font-family:monospace;">{val}</td>'
        rows += f"<tr>{tds}</tr>"

    html = (f'<div style="overflow-x:auto;overflow-y:auto;max-height:{height}px;'
            f'border:1px solid #e0e0e0;border-radius:4px;">'
            f'<table style="border-collapse:collapse;font-size:13px;width:100%;">'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div>')
    components.html(html, height=height + 4, scrolling=False)


# ─────────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────────

st.title("ETF 투자 대시보드")
st.caption("한국 ETF 실시간 데이터 | yfinance + FinanceDataReader | 배당소득세 15.4% 적용 | 개발: 이종호 (jongho1972@gmail.com)")

with st.spinner("ETF 데이터 불러오는 중... (첫 실행 시 최대 1분 소요)"):
    final = load_etf_data()

st.success(f"총 {len(final):,}개 ETF 로드 완료  |  기준일: {final['기준일'].max()}")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 3px solid #e0e0e0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 8px 8px 0 0;
    color: #666;
    background: #f0f2f6;
    border: 1px solid #ddd;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: #ff4b4b !important;
    border-color: #ff4b4b !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["ETF 조회", "What-if 분석", "ETF 비교 차트"])


# ── Tab 1: ETF 조회 ──────────────────────────

with tab1:
    st.subheader("ETF 목록 조회")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_marcap = st.number_input("최소 시총 (억원)", value=1000, step=500, min_value=0)
    with col2:
        div_filter = st.selectbox("배당일", ["전체", "월초", "월중", "기타"])
    with col3:
        keyword = st.text_input("종목명 검색")

    cols_show = ["Symbol", "Name", "시총", "주가", "월배당금", "1M배당률", "1Y배당률", "1Y수익률", "3M수익률", "배당일"]
    filtered = final[final["시총"] > min_marcap][cols_show].copy()

    if div_filter != "전체":
        filtered = filtered[filtered["배당일"] == div_filter]
    if keyword:
        filtered = filtered[filtered["Name"].str.contains(keyword, na=False)]

    fmt = {
        "시총": "{:,.0f}",
        "주가": "{:,.0f}",
        "월배당금": "{:,.0f}",
        "1M배당률": "{:.1f}%",
        "1Y배당률": "{:.1f}%",
        "1Y수익률": "{:.1f}%",
        "3M수익률": "{:.1f}%",
    }

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1M 배당률 Top 20**")
        cols_div = ["Name", "Symbol", "시총", "주가", "월배당금", "1M배당률", "3M수익률", "배당일"]
        fmt_div = {k: v for k, v in fmt.items() if k in cols_div}
        top_div = (
            filtered[filtered["배당일"] != "기타"]
            .sort_values("1M배당률", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        sticky_dataframe(top_div[cols_div], fmt=fmt_div)

    with col_b:
        st.markdown("**3M 수익률 Top 20**")
        cols_ret = ["Name", "Symbol", "시총", "주가", "3M수익률", "1Y수익률"]
        fmt_ret = {k: v for k, v in fmt.items() if k in cols_ret}
        top_ret = (
            filtered[~filtered["Name"].str.contains("레버리지", na=False)]
            .sort_values("3M수익률", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        sticky_dataframe(top_ret[cols_ret], fmt=fmt_ret)


# ── Tab 2: 투자 시뮬레이션 ───────────────────

with tab2:
    st.subheader("What-if 분석")
    st.caption("배당소득세 15.4% 적용 / 증권사 수수료 및 매매차익 소득세 미적용")

    etf_names = sorted(final["Name"].tolist())

    default_rows = pd.DataFrame([
        {"ETF 종목": "TIGER 배당커버드콜액티브",               "투자금 (원)": "100,000,000"},
        {"ETF 종목": "KODEX 미국나스닥100데일리커버드콜OTM",   "투자금 (원)": "200,000,000"},
        {"ETF 종목": "ACE 미국반도체데일리타겟커버드콜(합성)", "투자금 (원)": "100,000,000"},
        {"ETF 종목": "KODEX 미국S&P500데일리커버드콜OTM",      "투자금 (원)": "100,000,000"},
        {"ETF 종목": "ACE KRX금현물",                          "투자금 (원)": "100,000,000"},
    ])

    st.info("ETF 종목과 투자금을 직접 수정하거나 행을 추가/삭제할 수 있습니다. 셀을 클릭하면 편집됩니다.")

    edited = st.data_editor(
        default_rows,
        column_config={
            "ETF 종목": st.column_config.SelectboxColumn(
                "ETF 종목 (수정 가능)", options=etf_names, required=True,
                help="클릭하여 ETF 종목을 선택하세요",
            ),
            "투자금 (원)": st.column_config.TextColumn(
                "투자금 (원, 수정 가능)",
                help="콤마 포함 입력 가능 (예: 100,000,000)",
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
    )

    if st.button("What-if 분석 실행", type="primary"):
        results_list = []
        for _, row in edited.iterrows():
            try:
                cash = int(str(row["투자금 (원)"]).replace(",", ""))
            except ValueError:
                continue
            r = simul(final, row["ETF 종목"], cash)
            if r:
                results_list.append(r)

        if results_list:
            st.session_state["simul_result"] = pd.DataFrame(results_list)
        else:
            st.session_state.pop("simul_result", None)
            st.warning("유효한 종목이 없습니다. ETF 종목명을 확인해주세요.")

    if "simul_result" in st.session_state:
        df_result = st.session_state["simul_result"]

        st.info("📌 **1년 전 투자했다면 현재 수익** — 1년 전 동일 금액을 투자했을 때 현재 시점 기준 배당금 및 주가차익을 추정한 결과입니다.")

        total_invest = df_result["예상 투자금"].sum()
        total_annual_div = df_result["연배당금"].sum()
        monthly_div = total_annual_div // 12
        total_profit_3m = df_result["주가차익(3M)"].sum()

        # 요약 지표
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 투자금", f"{total_invest:,.0f}원")
        c2.metric("연간 배당금", f"{total_annual_div:,.0f}원")
        c3.metric("월 배당금", f"{monthly_div:,.0f}원")
        c4.metric("연간 주가차익", f"{total_profit_3m:,.0f}원")

        st.divider()

        st.caption("3M (최근 3개월 수익률 기준) / 1Y (최근 1년 수익률 기준) — 주가차익과 총수익은 이 두 기준의 범위 안에서 실현될 수 있습니다.")

        # 상세 테이블
        fmt2 = {c: "{:,.0f}" for c in ["예상 투자금", "주식수", "연배당금", "월배당금", "주가차익(3M)", "총수익(3M)", "주가차익(1Y)", "총수익(1Y)"]}
        st.dataframe(df_result.style.format(fmt2), use_container_width=True)

        # 배당금 비중 파이 차트
        fig_pie = go.Figure(go.Pie(
            labels=df_result["종목"],
            values=df_result["연배당금"],
            hole=0.4,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title="종목별 연배당금 비중", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)


# ── Tab 3: ETF 비교 차트 ─────────────────────

with tab3:
    st.subheader("ETF 주가 추이 비교")
    st.caption("시작값 100 기준 정규화 (1년)")

    compare_list = st.multiselect(
        "비교할 ETF 선택 (최대 8개)",
        options=sorted(final["Name"].tolist()),
        default=[
            "KODEX 미국나스닥100데일리커버드콜OTM",
            "KODEX 미국S&P500데일리커버드콜OTM",
            "KODEX 미국배당다우존스타겟커버드콜",
        ],
        max_selections=8,
    )

    if compare_list:
        for i, name in enumerate(compare_list, 1):
            st.caption(f"{i}. {name}")

    period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
    period_label = st.radio("기간", list(period_map.keys()), index=3, horizontal=True)

    if compare_list and st.button("차트 그리기", type="primary"):
        fig = go.Figure()
        with st.spinner("주가 데이터 불러오는 중..."):
            for etf_name in compare_list:
                row = final[final["Name"] == etf_name]
                if row.empty:
                    continue
                symbol = row["Symbol"].iloc[0]
                hist = yf.Ticker(f"{symbol}.KS").history(period=period_map[period_label])
                if hist.empty:
                    st.warning(f"{etf_name}: 데이터 없음")
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    mode="lines",
                    name=etf_name,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + etf_name + "</extra>",
                ))

        if fig.data:
            fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
            fig.update_layout(
                title=f"ETF 주가 추이 비교 ({period_label}, 시작=100)",
                xaxis_title="날짜",
                yaxis_title="정규화 주가",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("차트에 표시할 데이터가 없습니다.")
