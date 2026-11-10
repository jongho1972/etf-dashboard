import html
import json

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from urllib.parse import urlparse
from warnings import filterwarnings

from build_snapshot import ETF_DATA_PARQUET, ROOT, _trim_to_confirmed_close
from select_default import DEFAULT_JSON, exposure_shares

filterwarnings("ignore")

# 앱 본문 폰트는 .streamlit/config.toml, iframe 표와 차트는 아래 값으로 지정한다.
FONT_STACK = "Pretendard, -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
# 다크 고정 팔레트 — 홈페이지(jhawk.kr index.css)와 같은 값. 상승 빨강·하락 파랑(한국식)은 다크용 밝은 톤
# config.toml [theme]의 bg·surface·text·border·primary(accent_fill)도 같은 값으로 맞춘다
C = {
    "bg": "#0b0c0e", "surface": "#15171a", "surface_2": "#1d2025", "row_alt": "#121417", "total": "#1a2433",
    "text": "#f2f3f5", "text_2": "#c9ccd2", "text_muted": "#8f949c", "icon": "#aeb3bb",
    "accent": "#3d9bff", "accent_fill": "#0a6fe0", "accent_fill_hover": "#1a7cf0",
    "gain": "#ff6b6b", "loss": "#5b9dff", "border": "#24272d", "border_strong": "rgba(255,255,255,0.16)",
}
ROOT_CSS = ":root{color-scheme:dark;" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in C.items()) + "}"
# iframe(표)은 부모와 color-scheme이 달라 브라우저 기본 캔버스가 칠해지므로 페이지 배경을 직접 채운다
IFRAME_BASE = (f'<style>{ROOT_CSS}html,body{{margin:0;background:{C["bg"]};color:{C["text"]};}}'
               f'::-webkit-scrollbar{{height:8px;width:8px}}::-webkit-scrollbar-thumb{{background:{C["surface_2"]};border-radius:4px}}</style>')
FONT_LINK = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard-dynamic-subset.min.css">'

st.set_page_config(
    page_title="ETF 투자 가이드",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# 데이터 로드 (build_snapshot.py가 생성한 parquet 스냅샷)
# ─────────────────────────────────────────────



@st.cache_data(ttl=3600, show_spinner=False)
def load_etf_data():
    return pd.read_parquet(ETF_DATA_PARQUET)


@st.cache_data(ttl=3600, show_spinner=False)
def load_default_portfolio() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


EXPERT_JSON = ROOT / "expert_picks.json"


@st.cache_data(ttl=3600, show_spinner=False)
def load_expert_picks() -> dict | None:
    try:
        return json.loads(EXPERT_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _signed_pct(v: float) -> str:
    if pd.isna(v):
        return "-"
    color = "var(--gain)" if v > 0 else "var(--loss)" if v < 0 else "inherit"
    return f'<span style="color:{color}">{v:+.1f}%</span>'


SOURCE_NAMES = {
    "bok.or.kr": "한국은행", "yna.co.kr": "연합뉴스", "mt.co.kr": "머니투데이", "mk.co.kr": "매일경제",
    "hankyung.com": "한국경제", "asiae.co.kr": "아시아경제", "fnnews.com": "파이낸셜뉴스",
    "edaily.co.kr": "이데일리", "bizwatch.co.kr": "비즈워치", "newsway.co.kr": "뉴스웨이",
    "fntimes.com": "한국금융신문", "truefriend.com": "한국투자증권", "chosun.com": "조선일보",
    "joongang.co.kr": "중앙일보", "donga.com": "동아일보", "sedaily.com": "서울경제",
    "heraldcorp.com": "헤럴드경제", "news1.kr": "뉴스1", "newsis.com": "뉴시스",
    "kbthink.com": "KB Think", "reuters.com": "Reuters", "bloomberg.com": "Bloomberg",
}


def _source_label(src: dict) -> str:
    """출처 표시명: 아는 도메인은 매체명, 그다음 기사 제목 끝의 ' - 매체명', 없으면 도메인."""
    host = (urlparse(src["url"]).hostname or "").lower()
    for dom, label in SOURCE_NAMES.items():
        if host == dom or host.endswith("." + dom):
            return label
    for sep in (" - ", " | ", " : "):
        if sep in src.get("title", ""):
            tail = src["title"].rsplit(sep, 1)[1].strip()
            if 0 < len(tail) <= 14:
                return tail
    for pre in ("www.", "m.", "news.", "view."):
        host = host.removeprefix(pre)
    return host or "출처"


XP_CSS = """
.xp-wrap { background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:20px 20px 16px; margin:0.5rem 0 2rem; word-break:keep-all; }
.xp-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 10px; }
.xp-wrap h2.xp-h { margin:0; padding:0; font-size:1.35rem; font-weight:800; letter-spacing:-0.01em; color:var(--text); }
.xp-date { font-size:0.8rem; color:var(--text-muted); }
.xp-summary { font-size:0.92rem; font-weight:500; color:var(--text); margin:10px 0 4px; line-height:1.55;
              border-left:3px solid var(--accent); padding-left:10px; }
.xp-disclaimer { font-size:0.8rem; color:var(--text-muted); margin:0 0 12px; line-height:1.5; }
.xp-nav { display:none; }
.xp-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; align-items:start; }
.xp-wrap a:focus-visible, .xp-wrap summary:focus-visible {
  outline:2px solid var(--text); outline-offset:2px; border-radius:6px; }
.xp-card { background:var(--surface-2); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }
.xp-wrap h3.xp-title { margin:0; padding:0; font-size:1.05rem; font-weight:800; color:var(--text); }
.xp-target { font-size:0.8rem; color:var(--text-muted); margin:2px 0 4px; }
.xp-card ul.xp-picks { list-style:none; margin:0; padding:0; }
.xp-wrap li.xp-pick { border-top:1px solid var(--border); padding:12px 0 10px !important; margin:0 !important; }
.xp-wrap [data-testid="stHeaderActionElements"] { display:none !important; }   /* Streamlit 제목 앵커 아이콘 */
.xp-pick:first-child { border-top:1px solid var(--border-strong); }
.xp-wrap h4.xp-name { margin:0; padding:0; font-size:0.95rem; font-weight:700; color:var(--text); line-height:1.4; }
.xp-badge { display:inline-block; margin-left:6px; font-size:0.72rem; font-weight:600; padding:1px 7px;
            border-radius:999px; background:var(--surface); border:1px solid var(--border-strong); color:var(--text); vertical-align:1px; white-space:nowrap; }
.xp-reason { font-size:0.86rem; color:var(--text-2); line-height:1.55; margin-top:4px; }
.xp-caution, .xp-dissent { font-size:0.8rem; color:var(--text-muted); line-height:1.5; margin-top:3px; }
.xp-caution b, .xp-dissent b { font-weight:600; color:var(--text); margin-right:4px; }
.xp-metric { font-size:0.8rem; color:var(--text-muted); margin-top:6px; font-variant-numeric:tabular-nums; }
.xp-metric b { font-weight:600; color:var(--text); }
.xp-metric b span { font-weight:600; }
.xp-metric .m { white-space:nowrap; }
.xp-wrap details { margin-top:14px; font-size:0.85rem; color:var(--text); }
.xp-wrap summary { list-style:none; display:inline-flex; align-items:center; gap:6px; cursor:pointer;
                   font-size:0.85rem; font-weight:600; color:var(--text-muted); padding:6px 0; }
.xp-wrap summary::-webkit-details-marker { display:none; }
.xp-wrap summary::after { content:"›"; display:inline-block; transform:rotate(90deg); transition:transform .15s; }
.xp-wrap details[open] summary::after { transform:rotate(-90deg); }
.xp-wrap details ul { list-style:none; margin:4px 0 0; padding:0; }
.xp-wrap details li { margin:0 !important; padding:8px 0 !important; border-top:1px solid var(--border); line-height:1.6; }
.xp-src { display:inline-flex; flex-wrap:wrap; gap:6px; margin-left:6px; vertical-align:middle; }
.xp-wrap .xp-src a { display:inline-block; min-height:24px; line-height:22px; font-size:0.74rem; color:var(--text-muted);
                     text-decoration:none; border:1px solid var(--border-strong); border-radius:6px; padding:0 7px; }
.xp-wrap .xp-src a:hover { color:var(--text); border-color:var(--text-muted); }
.xp-note { font-size:0.75rem; color:var(--text-muted); line-height:1.6; max-width:68ch;
           margin-top:12px; padding-top:10px; border-top:1px dashed var(--border-strong); }
@media (max-width: 900px) {
  .xp-wrap { padding:16px 14px 14px; }
  .xp-nav { display:flex; gap:6px; margin:0 0 10px; overflow-x:auto; scrollbar-width:none; }
  .xp-nav::-webkit-scrollbar { display:none; }
  .xp-wrap .xp-nav a { flex:0 0 auto; font-size:0.8rem; font-weight:600; color:var(--text); text-decoration:none;
                       background:var(--surface-2); border:1px solid var(--border-strong); border-radius:999px; padding:6px 12px; }
  .xp-grid { grid-template-columns:1fr; gap:12px; }   /* 모바일은 세로로 쌓고 칩은 해당 카드로 이동 */
  .xp-card { padding:16px; scroll-margin-top:16px; }
}
"""


def render_expert_picks(data: dict, df: pd.DataFrame) -> None:
    """전문가 3인 합의 추천 — 문구는 json(expert_picks.py), 수치는 스냅샷에서 직접."""
    e = html.escape
    by_sym = df.set_index("Symbol")
    n_experts = len(data["experts"])
    votes = [p["votes"] for b in data["bundles"] for p in b["picks"]]
    n_full = votes.count(n_experts)
    consensus = (f"{len(votes)}종목 모두 {n_experts}명 전원 찬성" if n_full == len(votes)
                 else f"전원 찬성 {n_full} · 다수 찬성 {len(votes) - n_full}")
    cards, nav = [], []
    for b in data["bundles"]:
        items = []
        for p in b["picks"]:
            if p["symbol"] in by_sym.index:
                row = by_sym.loc[p["symbol"]]
                name = row["Name"]
                # 1Y배당률 0 = 최근 1년 분배 기록 없음(TR형 재투자 또는 무분배)
                div = (f'배당률 <b>{row["1Y배당률"]:.1f}%</b>' if row["1Y배당률"] > 0
                       else "<b>1년 분배 없음</b>")
                metrics = " · ".join(f'<span class="m">{m}</span>' for m in (
                    div, f'1년 총수익(배당 포함) <b>{_signed_pct(row["1Y총수익률"])}</b>',
                    f'시총 <b>{row["시총"]:,.0f}억</b>'))
            else:
                name, metrics = p["name"], "스냅샷 없음"
            # 배지는 이견이 있을 때만 — 전원 찬성은 제목 옆 요약 한 번으로 충분
            badge = ("" if p["votes"] == n_experts
                     else f'<span class="xp-badge">{p["votes"]}/{n_experts} 찬성</span>')
            extra = "".join(
                f'<div class="xp-{cls}"><b>{label}</b>{e(p[key])}</div>'
                for cls, label, key in (("caution", "유의", "caution"), ("dissent", "반대 의견", "dissent"))
                if p.get(key))
            items.append(
                f'<li class="xp-pick"><h4 class="xp-name">{e(name)}{badge}</h4>'
                f'<div class="xp-reason">{e(p["reason"])}</div>{extra}'
                f'<div class="xp-metric">{metrics}</div></li>'
            )
        cid = f'xp-{e(b["key"])}'
        nav.append(f'<a href="#{cid}">{e(b["title"])}</a>')
        cards.append(
            f'<article class="xp-card" id="{cid}"><h3 class="xp-title">{e(b["title"])}</h3>'
            f'<div class="xp-target">{e(b["target"])}</div><ul class="xp-picks">{"".join(items)}</ul></article>'
        )
    views = "".join(
        f'<li><b>{e(x["name"])}</b> {e(x["view"])}<span class="xp-src">'
        + "".join(
            f'<a href="{e(s["url"])}" target="_blank" rel="noopener" title="{e(s.get("title", ""))}"'
            f' aria-label="{e(x["name"])} 출처: {e(_source_label(s))} (새 창)">{e(_source_label(s))} ↗</a>'
            for s in x.get("sources", [])[:3])
        + "</span></li>"
        for x in data["experts"]
    )
    n_src = sum(min(len(x.get("sources", [])), 3) for x in data["experts"])
    st.markdown(f"""<style>{XP_CSS}</style>
<section class="xp-wrap" aria-label="AI 전문가 3인 합의 추천">
<div class="xp-head"><h2 class="xp-h">AI 전문가 3인 합의 추천</h2>
<span class="xp-date">주 1회 갱신 · {e(data["generated_at"][:10])} 논의 · {consensus}</span></div>
<div class="xp-summary">{e(data["market_summary"])}</div>
<div class="xp-disclaimer">거시·배당·위험관리 관점의 AI 전문가 3명이 제안·투표한 결과입니다. 정보 제공용이며 투자 권유가 아닙니다.</div>
<nav class="xp-nav" aria-label="추천 묶음 바로가기">{"".join(nav)}</nav>
<div class="xp-grid" role="region" aria-label="추천 묶음 {len(cards)}개">{"".join(cards)}</div>
<details><summary>전문가별 의견과 참고 출처 {n_src}건</summary><ul>{views}</ul></details>
<div class="xp-note">선정 기준: 3명 중 2명 이상 찬성한 종목만 싣습니다. 수치는 최신 스냅샷 기준이며, 1년 총수익은 분배금을 포함합니다(아래 표의 1Y수익률은 가격 기준).</div>
</section>""", unsafe_allow_html=True)


def snapshot_mtime_kst() -> str:
    try:
        ts = ETF_DATA_PARQUET.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except FileNotFoundError:
        return "N/A"


def _history_confirmed_close(symbol: str, period: str) -> pd.DataFrame:
    """KRX 장마감(15:30 KST) 이후면 당일 종가까지, 그 전이면 D-1까지만."""
    h = yf.Ticker(f"{symbol}.KS").history(period=period)
    if h.empty:
        return h
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    return _trim_to_confirmed_close(h)


# ─────────────────────────────────────────────
# 시뮬레이션 함수
# ─────────────────────────────────────────────

# 매매차익 비과세인 국내주식형 ETF 추정 — 종목명에 해외/기타 자산 키워드가 없으면 True
_NON_DOMESTIC_KW = (
    "미국", "S&P500", "나스닥", "차이나", "중국", "일본", "유럽", "글로벌",
    "인도", "대만", "베트남", "신흥국", "선진국", "월드",
    "미국채", "국고채", "회사채", "채권", "금현물", "은현물", "골드", "원유",
    "천연가스", "리츠", "TIPS", "달러", "엔화", "위안",
    "커버드콜", "레버리지", "인버스", "선물", "혼합", "TRF",
)


def is_domestic_stock_etf(name: str) -> bool:
    return not any(k in name for k in _NON_DOMESTIC_KW)


def simul(final_df, etf_name, cash):
    target = final_df[final_df["Name"] == etf_name]
    if target.empty or cash <= 0:
        return None

    price = target["주가"].iloc[0]
    price_pct_3m = target["3M수익률"].iloc[0] / 100
    price_pct_1y = target["1Y수익률"].iloc[0] / 100
    annual_div_per_share = target["연간배당금"].iloc[0]
    share_cnt = int(cash // price)

    raw_dividend = annual_div_per_share * share_cnt
    dividend_annual = int(raw_dividend * (1 - 0.154))
    dividend_monthly = dividend_annual // 12

    cg_tax = 0.0 if is_domestic_stock_etf(etf_name) else 0.154

    raw_profit_3m = price * price_pct_3m * share_cnt
    profit_3m = int(raw_profit_3m * (1 - cg_tax)) if raw_profit_3m > 0 else int(raw_profit_3m)

    raw_profit_1y = price * price_pct_1y * share_cnt
    profit_1y = int(raw_profit_1y * (1 - cg_tax)) if raw_profit_1y > 0 else int(raw_profit_1y)

    배당일 = target["배당일"].iloc[0]

    return {
        "종목": etf_name,
        "배당일": 배당일,
        "예상 투자금": cash,
        "주식수": share_cnt,
        "연배당금": dividend_annual,
        "월배당금": dividend_monthly,
        "주가차익 (3M 추세 가정)": profit_3m,
        "총수익 (3M 추세 가정)": dividend_annual + profit_3m,
        "주가차익 (1Y 추세 가정)": profit_1y,
        "총수익 (1Y 추세 가정)": dividend_annual + profit_1y,
    }


# ─────────────────────────────────────────────
# Name 컬럼 고정 테이블 (모바일 지원)
# ─────────────────────────────────────────────

RET_COLS = {"3M수익률", "1Y수익률"}

def sticky_dataframe(df, fmt=None, height=None, mobile_hide=None):
    """mobile_hide: 모바일(<=600px)에서 숨길 원본 컬럼명 리스트.
    height=None이면 행 수에 맞춰 자동 계산(데이터 적을 때 빈 iframe 공간 제거).
    """
    orig = df.copy().reset_index(drop=True)
    if height is None:
        # 헤더 50px + 행당 ~34px, 최대 760px
        height = min(50 + len(orig) * 34, 760)
    disp = df.copy().reset_index(drop=True)
    if fmt:
        for col, pattern in fmt.items():
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: pattern.format(x) if pd.notna(x) else "")
    disp = disp.rename(columns={"시총": "시총(억원)", "주가": "주가(원)", "월배당금": "월배당(원)"})

    orig_cols = list(orig.columns)
    has_symbol = "Symbol" in orig_cols
    hide_idx = set()
    if mobile_hide:
        hide_idx = {orig_cols.index(c) for c in mobile_hide if c in orig_cols}

    def col_class(i, orig_col):
        parts = [f"col-{i}"]
        if orig_col == "Symbol":
            parts.append("etf-symbol-col")
        if i in hide_idx:
            parts.append("mobile-hide-col")
        return " ".join(parts)

    def th(col, i, orig_col):
        pin = "position:sticky;left:0;z-index:3;" if i == 0 else "z-index:1;"
        align = "left" if i == 0 else "right"
        inner = f'<div class="etf-name-text">{col}</div>' if i == 0 else col
        return (f'<th class="{col_class(i, orig_col)}" style="position:sticky;top:0;{pin}'
                f'background:{C["surface_2"]};color:{C["text_2"]};font-weight:600;padding:6px 12px;'
                f'text-align:{align};white-space:nowrap;border-bottom:1px solid {C["border_strong"]};">'
                f'{inner}</th>')

    headers = "".join(th(col, i, orig_cols[i] if i < len(orig_cols) else "") for i, col in enumerate(disp.columns))
    rows = ""
    for idx, row in disp.iterrows():
        bg = C["surface"] if idx % 2 == 0 else C["row_alt"]
        symbol = orig.at[idx, "Symbol"] if has_symbol else ""
        tds = ""
        for i, val in enumerate(row):
            orig_col = orig_cols[i] if i < len(orig_cols) else ""
            cls = col_class(i, orig_col)
            if i == 0:
                if symbol:
                    onclick = f"showEtfModal('{val}','{symbol}',event.clientY)"
                else:
                    onclick = f"alert('{val}')"
                tds += (f'<td class="{cls}" onclick="{onclick}" '
                        f'style="position:sticky;left:0;background:{bg};'
                        f'padding:6px 12px;'
                        f'border-right:1px solid {C["border_strong"]};font-weight:500;cursor:pointer;" title="{val}">'
                        f'<div class="etf-name-text">{val}</div></td>')
            else:
                color = ""
                if orig_col in RET_COLS:
                    orig_val = orig.at[idx, orig_col]
                    if pd.notna(orig_val):
                        color = f'color:{C["loss"]};' if orig_val < 0 else f'color:{C["gain"]};'
                tds += f'<td class="{cls}" style="padding:6px 12px;white-space:nowrap;background:{bg};text-align:right;font-variant-numeric:tabular-nums;{color}">{val}</td>'
        rows += f"<tr>{tds}</tr>"

    modal = (
        '<div id="etf-modal" onclick="if(event.target===this)closeEtfModal()" '
        'style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;'
        'background:rgba(0,0,0,0.6);z-index:9999;">'
        '<div id="etf-modal-box" style="position:absolute;left:50%;transform:translateX(-50%);top:80px;'
        f'background:{C["surface"]};border:1px solid {C["border_strong"]};padding:20px 18px 16px;border-radius:14px;min-width:260px;max-width:80%;'
        f'box-shadow:0 12px 40px rgba(0,0,0,0.5);font-family:{FONT_STACK};">'
        f'<div id="etf-modal-name" style="font-size:15px;font-weight:600;color:{C["text"]};margin-bottom:16px;'
        'text-align:center;word-break:keep-all;line-height:1.4;"></div>'
        '<div style="display:flex;gap:8px;">'
        f'<button onclick="openEtfInfo()" style="flex:1;padding:10px;background:{C["accent_fill"]};color:#fff;border:none;'
        'border-radius:8px;font-size:14px;font-weight:500;cursor:pointer;">상품정보</button>'
        f'<button onclick="closeEtfModal()" style="flex:1;padding:10px;background:{C["surface_2"]};color:{C["text"]};border:none;'
        'border-radius:8px;font-size:14px;font-weight:500;cursor:pointer;">닫기</button>'
        '</div></div></div>'
        '<script>'
        'let _etfSymbol="";'
        'function showEtfModal(n,s,y){_etfSymbol=s;'
        'document.getElementById("etf-modal-name").textContent=n;'
        'var box=document.getElementById("etf-modal-box");'
        'box.style.top=Math.max(20,(y||80)-110)+"px";'
        'document.getElementById("etf-modal").style.display="block";}'
        'function closeEtfModal(){document.getElementById("etf-modal").style.display="none";}'
        'function openEtfInfo(){window.open("https://www.k-etf.com/etf/"+_etfSymbol,"_blank");closeEtfModal();}'
        '</script>'
    )
    style = (
        '<style>'
        '.etf-name-text{width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}'
        '.etf-symbol-col{display:none;}'
        '@media (max-width:600px){'
        '.etf-name-text{width:170px;}'
        '.mobile-hide-col{display:none;}'
        '.etf-table td,.etf-table th{padding:6px 8px !important;font-size:12.5px !important;}'
        '}'
        '</style>'
    )
    html = (f'{FONT_LINK}{IFRAME_BASE}{style}{modal}'
            f'<div style="overflow-x:auto;overflow-y:auto;max-height:{height}px;'
            f'border:1px solid {C["border"]};border-radius:10px;">'
            f'<table class="etf-table" style="border-collapse:collapse;font-size:13px;width:100%;font-family:{FONT_STACK};">'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div>')
    components.html(html, height=height + 4, scrolling=False)


# ─────────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────────

st.markdown(f"<style>{ROOT_CSS}</style>" + """
<style>
/* ── 다크 고정: 배경·글자는 .streamlit/config.toml 테마, 토큰은 파이썬 C → ROOT_CSS ── */

/* ── Streamlit 기본 헤더/데코/툴바 숨김 → 커스텀 헤더 사용 ── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
.main .block-container,
[data-testid="stMainBlockContainer"],
section.main > div.block-container {
    padding-top: 0 !important;
    padding-bottom: 1rem !important;
}
/* 컬럼 stack 간 모바일 여백 축소 */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
}

/* ── 커스텀 헤더 ── */
.app-header {
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 18px 0 14px;
    margin: 0 -4rem 1.2rem;
    padding-left: 4rem; padding-right: 4rem;
}
.app-header h1 {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.6rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.5px; margin: 0; padding: 0;
}
.app-header .hd-icon {
    flex: 0 0 auto; width: 34px; height: 34px; border-radius: 10px;
    background: var(--surface-2); display: inline-flex; align-items: center; justify-content: center;
}
.app-header .hd-icon svg { width: 20px; height: 20px; fill: none; stroke: var(--icon); stroke-width: 1.6;
    stroke-linecap: round; stroke-linejoin: round; }
.app-header [data-testid="stHeaderActionElements"] { display: none !important; }

/* ── 모바일 컴팩트화 ── */
@media (max-width: 640px) {
    .app-header {
        padding: 10px 0 10px;
        margin: 0 -1rem 0.8rem;
        padding-left: 1rem; padding-right: 1rem;
    }
    .app-header h1 { font-size: 1.3rem; }
    .app-header .hd-icon { width: 30px; height: 30px; }
}

/* ── 탭 스타일 (칩 CTA) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: none !important;
    margin-bottom: 14px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px; font-weight: 600;
    padding: 12px 22px;
    min-height: 44px;
    border-radius: 999px;
    color: var(--text-2);
    background: var(--surface);
    border: 1px solid var(--border) !important;
    transition: all 0.15s ease;
    cursor: pointer;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--surface-2);
    color: var(--text);
    transform: translateY(-1px);
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: var(--accent-fill) !important;
    border-color: var(--accent-fill) !important;
    box-shadow: 0 4px 14px color-mix(in srgb, var(--accent-fill) 35%, transparent);
}
.stTabs [aria-selected="true"]:hover {
    background: var(--accent-fill-hover) !important;
}
/* baseweb 기본 underline indicator 숨김 */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── multiselect 칩: 터치 타겟 + 표시 폭 ── */
.stMultiSelect [data-baseweb="tag"] {
    max-width: 200px;
    padding: 4px 6px;
}
.stMultiSelect [data-baseweb="tag"] [role="button"],
.stMultiSelect [data-baseweb="tag"] span[role="button"] {
    min-width: 28px; min-height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    padding: 4px;
}
@media (max-width: 640px) {
    .stMultiSelect [data-baseweb="tag"] { max-width: 170px; }
}

/* ── 스냅샷 배너 (st.success 대체, 모바일 의미 단위 줄바꿈) ── */
.snapshot-banner {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-2);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 0.25rem 0 1rem;
    font-size: 0.9rem;
    line-height: 1.55;
}
.snapshot-banner span { display: inline-block; white-space: nowrap; }
.snapshot-banner .sep { color: var(--text-muted); padding: 0 4px; }
</style>

<div class="app-header">
  <h1><span class="hd-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg></span>ETF 투자 가이드</h1>
</div>
""", unsafe_allow_html=True)

final = load_etf_data()

st.markdown(
    f"""<div class="snapshot-banner">
<span>총 {len(final):,}개 ETF</span><span class="sep">·</span>
<span>종가 기준일 {final['기준일'].max()}</span><span class="sep">·</span>
<span>스냅샷 {snapshot_mtime_kst()}</span>
</div>""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["ETF 상품 조회", "What-if 분석", "ETF 비교 차트"])


# ── Tab 1: ETF 상품 조회 ──────────────────────────

with tab1:
    expert = load_expert_picks()
    if expert:
        render_expert_picks(expert, final)

    st.subheader("ETF 상품 조회")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_marcap = st.number_input("최소 시총 (억원)", value=1000, step=500, min_value=0)
    with col2:
        div_filter = st.selectbox("배당일", ["전체", "월초", "월중", "기타"])
    with col3:
        keyword = st.text_input("종목명 키워드")

    cols_show = ["Symbol", "Name", "시총", "주가", "월배당금", "1M배당률", "1Y배당률", "1Y수익률", "3M수익률", "배당일", "상장1Y미만"]
    filtered = final[final["시총"] > min_marcap][cols_show].copy()

    if div_filter != "전체":
        filtered = filtered[filtered["배당일"] == div_filter]
    if keyword:
        filtered = filtered[filtered["Name"].str.contains(keyword, na=False)]
    # 상장 1년 미만: 1Y 값이 상장일 이후 기준·배당은 연환산이라 표시 (화면용 이름만 변경)
    filtered.loc[filtered["상장1Y미만"], "Name"] = "* " + filtered.loc[filtered["상장1Y미만"], "Name"]

    fmt = {
        "시총": "{:,.0f}",
        "주가": "{:,.0f}",
        "월배당금": "{:,.0f}",
        "1M배당률": "{:.1f}%",
        "1Y배당률": "{:.1f}%",
        "1Y수익률": "{:+.1f}%",
        "3M수익률": "{:+.1f}%",
    }

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1M 배당률 Top 20**")
        cols_div = ["Name", "월배당금", "1M배당률", "3M수익률", "시총", "주가", "Symbol"]
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
        cols_ret = ["Name", "3M수익률", "1Y수익률", "시총", "주가", "Symbol"]
        fmt_ret = {k: v for k, v in fmt.items() if k in cols_ret}
        top_ret = (
            filtered[~filtered["Name"].str.contains("레버리지", na=False)]
            .sort_values("3M수익률", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        sticky_dataframe(top_ret[cols_ret], fmt=fmt_ret)

    st.caption(
        "수익률은 배당을 뺀 가격 변동, 배당률은 최근 1년 지급액 ÷ 현재 주가입니다. "
        "\\* 상장 1년 미만 종목은 1Y 값이 상장일 이후 기준이고 연간배당은 연환산입니다."
    )


# ── Tab 2: 투자 시뮬레이션 ───────────────────

with tab2:
    st.subheader("What-if 분석")

    etf_names = sorted(final["Name"].tolist())

    default_cfg = load_default_portfolio()
    default_rows = pd.DataFrame([
        {"ETF 종목": slot["name"], "예상 투자금 (원)": f"{slot['amount']:,}"}
        for slot in default_cfg["slots"]
    ])
    last_swap = default_cfg["history"][-1]["date"] if default_cfg["history"] else "없음"
    # 안내 문구 수치는 스냅샷으로 매일 다시 계산 (하드코딩하면 데이터·교체와 어긋남)
    default_monthly = sum(r["월배당금"] for s in default_cfg["slots"]
                          if (r := simul(final, s["name"], s["amount"])))
    group_desc = default_cfg.get("group_desc", {})
    composition = " · ".join(
        f"{g} {p:.0f}%" + (f" ({html.escape(group_desc[g])})" if g in group_desc else "")
        for g, p in exposure_shares(default_cfg, key="group").items()
    )

    st.markdown(
        f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);padding:14px 18px;border-radius:10px;margin:8px 0 12px 0;word-break:keep-all;overflow-wrap:break-word;">
          <div style="font-size:1.05rem;font-weight:700;color:var(--text);margin-bottom:8px;">클로드가 추천하는 포트폴리오</div>
          <div style="font-size:0.9rem;color:var(--text-2);line-height:1.6;">
            <b>목표 10억</b> — 은퇴 후 <b>현금흐름</b> · <b>하락장 원금 방어</b><br/>
            <b>구성</b>: {composition}<br/>
            <b>운용</b>: 세후 월 약 {default_monthly / 10_000:,.0f}만원 분배금(최근 1년 지급액 기준)을 생활비로 쓰고, 하락장엔 현금 버퍼 사용<br/>
            <b>점검</b>: 매 영업일 자동 점검, 기준 미달 시에만 교체 (최근 교체: {last_swap})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("ETF 종목과 투자금을 직접 수정하거나 행을 추가/삭제할 수 있습니다. 셀을 클릭하면 편집됩니다.")

    edited = st.data_editor(
        default_rows,
        column_config={
            "ETF 종목": st.column_config.SelectboxColumn(
                "ETF 종목 (수정 가능)", options=etf_names, required=True,
                help="클릭하여 ETF 종목을 선택하세요",
                width="medium",
            ),
            "예상 투자금 (원)": st.column_config.TextColumn(
                "예상 투자금(원, 수정가능)",
                help="콤마 포함 입력 가능 (예: 100,000,000)",
                width="small",
            ),
        },
        num_rows="dynamic",
        width="stretch",
    )

    # What-if 종목 목록이 바뀌면 비교 차트 탭 디폴트도 동기화
    whatif_etf_names = [n for n in edited["ETF 종목"].dropna().tolist() if n in final["Name"].values][:8]
    if whatif_etf_names != st.session_state.get("_prev_whatif"):
        st.session_state["_prev_whatif"] = whatif_etf_names
        st.session_state["compare_list"] = whatif_etf_names

    if st.button("What-if 분석 실행", type="primary"):
        results_list = []
        for _, row in edited.iterrows():
            try:
                cash = int(str(row["예상 투자금 (원)"]).replace(",", ""))
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

        st.info(
            "1년 보유 가정 시뮬레이션 — 최근 3개월·1년 수익률을 기준으로 한 예상 결과입니다.\n\n"
            "- 배당금은 최근 1년(연 지급 횟수만큼) 실제 지급액, 주가차익은 배당을 뺀 순수 가격 변동\n"
            "- 배당소득세 15.4% 적용\n"
            "- 매매차익에 15.4% 차감 (국내주식형 ETF는 매매차익 비과세 자동 반영)\n"
            "- 증권사 매매 수수료 미적용\n"
            "- 배당금 재투자 복리효과 미반영"
        )

        total_invest = df_result["예상 투자금"].sum()
        total_annual_div = df_result["연배당금"].sum()
        monthly_div = total_annual_div // 12
        total_profit_3m = df_result["주가차익 (3M 추세 가정)"].sum()

        # 요약 지표 (모바일 2x2 wrap, 핵심=배당금에 액센트 강조)
        label_st = "font-size:0.82rem;color:var(--text-muted);margin-bottom:4px;line-height:1.4;"
        val_base = "font-weight:700;white-space:nowrap;line-height:1.2;"
        val_big = val_base + "font-size:clamp(1.2rem,4.4vw,1.6rem);"
        val_main = val_big + "color:var(--text);"
        val_sub = val_base + "font-size:clamp(1.05rem,3.8vw,1.4rem);color:var(--text-2);"
        val_profit = val_big + "color:" + (
            "var(--gain);" if total_profit_3m > 0 else "var(--loss);" if total_profit_3m < 0 else "var(--text);")
        summary_html = f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px 16px;margin:8px 0;">
          <div><div style="{label_st}">목표 투자금액</div><div style="{val_sub}">{total_invest:,.0f}원</div></div>
          <div><div style="{label_st}">연간 배당금</div><div style="{val_main}">{total_annual_div:,.0f}원</div></div>
          <div><div style="{label_st}">월 배당금</div><div style="{val_sub}">{monthly_div:,.0f}원</div></div>
          <div><div style="{label_st}">연간 주가차익</div><div style="{val_profit}">{total_profit_3m:,.0f}원</div></div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

        st.divider()

        st.caption("최근 3개월 / 1년 수익률이 향후 1년 동안 동일하게 반복된다고 가정한 결과입니다. 두 값이 예상 범위의 양 끝을 나타냅니다.")

        # 상세 테이블 (합계 행 포함) — HTML 렌더링
        sum_cols = ["예상 투자금", "주식수", "연배당금", "월배당금", "주가차익 (3M 추세 가정)", "총수익 (3M 추세 가정)", "주가차익 (1Y 추세 가정)", "총수익 (1Y 추세 가정)"]
        total_row = {"종목": "합계", "배당일": ""}
        for c in sum_cols:
            if c in df_result.columns:
                total_row[c] = int(df_result[c].sum())
        df_with_total = pd.concat([df_result, pd.DataFrame([total_row])], ignore_index=True)
        for c in sum_cols:
            if c in df_with_total.columns:
                df_with_total[c] = pd.to_numeric(df_with_total[c], errors="coerce").fillna(0).astype(int)

        disp_cols = list(df_with_total.columns)
        num_rows = len(df_with_total)
        # 모바일에서도 전체 컬럼 노출 (가로 스와이프). 종목만 sticky-left.
        sim_hide = set()

        def sim_class(i, col):
            parts = [f"sim-col-{i}"]
            if col in sim_hide:
                parts.append("mobile-hide-col")
            return " ".join(parts)

        # 긴 헤더는 2줄로 줄바꿈해 컬럼 폭을 줄임 (모바일 가독성)
        WRAP_HEADER = {
            "주가차익 (3M 추세 가정)": "주가차익<br/>(3M 추세 가정)",
            "총수익 (3M 추세 가정)": "총수익<br/>(3M 추세 가정)",
            "주가차익 (1Y 추세 가정)": "주가차익<br/>(1Y 추세 가정)",
            "총수익 (1Y 추세 가정)": "총수익<br/>(1Y 추세 가정)",
        }
        sim_headers = ""
        for i, col in enumerate(disp_cols):
            align = "left" if i == 0 else "right"
            pin = "left:0;z-index:3;" if i == 0 else "z-index:1;"
            head_html = WRAP_HEADER.get(col, col)
            inner = f'<div class="sim-name-text">{head_html}</div>' if i == 0 else head_html
            sim_headers += (f'<th class="{sim_class(i, col)}" style="position:sticky;top:0;{pin}'
                            f'background:{C["surface_2"]};color:{C["text_2"]};font-weight:600;padding:6px 12px;'
                            f'text-align:{align};white-space:nowrap;line-height:1.3;border-bottom:1px solid {C["border_strong"]};">'
                            f'{inner}</th>')
        sim_rows = ""
        for idx, row in df_with_total.iterrows():
            is_total = row["종목"] == "합계"
            bg = C["total"] if is_total else (C["surface"] if idx % 2 == 0 else C["row_alt"])
            weight = "font-weight:700;" if is_total else ""
            border_top = f'border-top:1px solid {C["border_strong"]};' if is_total else ""
            tds = ""
            for i, col in enumerate(disp_cols):
                val = row[col]
                if col in sum_cols:
                    val = f"{int(val):,}"
                cls = sim_class(i, col)
                if i == 0:
                    tds += (f'<td class="{cls}" style="position:sticky;left:0;background:{bg};'
                            f'padding:6px 12px;border-right:1px solid {C["border_strong"]};'
                            f'font-weight:{("700" if is_total else "500")};{border_top}'
                            f'color:{C["text"]};" title="{val}">'
                            f'<div class="sim-name-text">{val}</div></td>')
                else:
                    tds += (f'<td class="{cls}" style="padding:6px 12px;white-space:nowrap;background:{bg};'
                            f'text-align:right;font-variant-numeric:tabular-nums;'
                            f'color:{C["text"]};{weight}{border_top}">{val}</td>')
            sim_rows += f"<tr>{tds}</tr>"
        sim_height = min(60 + num_rows * 38, 500)
        sim_style = (
            '<style>'
            '.sim-name-text{width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}'
            '@media (max-width:600px){'
            '.sim-name-text{width:130px;}'
            '.mobile-hide-col{display:none;}'
            '.sim-table td,.sim-table th{padding:6px 8px !important;font-size:12.5px !important;}'
            '}'
            '</style>'
        )
        sim_html = (f'{FONT_LINK}{IFRAME_BASE}{sim_style}'
                    f'<div style="overflow-x:auto;overflow-y:auto;max-height:{sim_height}px;'
                    f'border:1px solid {C["border"]};border-radius:10px;">'
                    f'<table class="sim-table" style="border-collapse:collapse;font-size:13px;width:100%;font-family:{FONT_STACK};">'
                    f'<thead><tr>{sim_headers}</tr></thead>'
                    f'<tbody>{sim_rows}</tbody>'
                    f'</table></div>')
        components.html(sim_html, height=sim_height + 4, scrolling=False)

        # 참고: ETF 유형별 세제 비교 (시뮬 결과 해석용)
        st.markdown(
            """
<style>
.tax-tbl { border-collapse:collapse; width:100%; font-size:0.85rem; }
.tax-tbl th, .tax-tbl td { border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
.tax-tbl thead tr { background:var(--surface-2); }
@media (max-width: 640px) {
  .tax-tbl, .tax-tbl tbody, .tax-tbl tr, .tax-tbl td { display:block; width:100%; box-sizing:border-box; }
  .tax-tbl thead { display:none; }
  .tax-tbl tbody tr { margin-bottom:10px; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
  .tax-tbl tbody td { border:none; border-bottom:1px solid var(--border); padding:8px 10px; }
  .tax-tbl tbody td:last-child { border-bottom:none; }
  .tax-tbl tbody td:first-child { background:var(--surface-2); font-weight:600; }
  .tax-tbl tbody td[data-label]::before {
    content: attr(data-label) " · ";
    color:var(--text-muted); font-weight:600; margin-right:2px;
  }
}
</style>
<div style="margin-top:12px;font-size:0.88rem;color:var(--text-2);">
  <div style="font-weight:600;margin-bottom:6px;color:var(--text);">참고 — ETF 유형별 세제 비교</div>
  <table class="tax-tbl">
    <thead>
      <tr>
        <th>구분</th>
        <th>국내주식형 ETF</th>
        <th>기타 ETF (국내상장, 해외·CC·금·채권 등)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>매매차익 과세</td>
        <td data-label="국내주식형"><b>비과세</b></td>
        <td data-label="기타 ETF">배당소득세 15.4%</td>
      </tr>
      <tr>
        <td>매매차익 종합과세 합산</td>
        <td data-label="국내주식형"><b>합산 대상 아님</b></td>
        <td data-label="기타 ETF"><b>합산 대상</b> (금융소득 합계 2천만원 초과 시)</td>
      </tr>
      <tr>
        <td>분배금 과세</td>
        <td data-label="국내주식형">배당소득세 15.4%</td>
        <td data-label="기타 ETF">배당소득세 15.4%</td>
      </tr>
      <tr>
        <td>분배금 종합과세 합산</td>
        <td data-label="국내주식형"><b>합산 대상</b> (금융소득 합계 2천만원 초과 시)</td>
        <td data-label="기타 ETF"><b>합산 대상</b> (금융소득 합계 2천만원 초과 시)</td>
      </tr>
    </tbody>
  </table>
  <div style="margin-top:6px;color:var(--text-muted);font-size:0.78rem;">
    ※ 일반 위탁계좌 기준. 연금계좌(IRP·연금저축·DC)는 인출 시점까지 과세이연.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ── Tab 3: ETF 비교 차트 ─────────────────────

with tab3:
    st.subheader("ETF 주가 추이 비교")
    st.caption("시작값 100 기준 정규화 (1년)")

    compare_list = st.multiselect(
        "비교할 ETF 선택 (최대 8개)",
        options=sorted(final["Name"].tolist()),
        key="compare_list",
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
            series = []
            for etf_name in compare_list:
                row = final[final["Name"] == etf_name]
                if row.empty:
                    continue
                symbol = row["Symbol"].iloc[0]
                hist = _history_confirmed_close(symbol, period_map[period_label])
                if hist.empty:
                    st.warning(f"{etf_name}: 데이터 없음")
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0]) * 100
                series.append((etf_name, normalized, float(normalized.iloc[-1])))

            # 가장 최근일 정규화 값 내림차순 정렬
            series.sort(key=lambda s: s[2], reverse=True)

            for etf_name, normalized, _ in series:
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    mode="lines",
                    name=etf_name,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + etf_name + "</extra>",
                ))

        if fig.data:
            fig.add_hline(y=100, line_dash="dash", line_color=C["text_muted"], opacity=0.5)
            fig.update_layout(
                template="plotly_dark",   # 범례·호버 다크 기본값
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_STACK, color=C["text_2"]),
                title=f"ETF 주가 추이 비교 ({period_label}, 시작=100)",
                xaxis_title="날짜",
                yaxis_title="정규화 주가",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, xanchor="left"),
                margin=dict(b=160),
                height=580,
            )
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
            st.plotly_chart(fig, width="stretch")
        else:
            st.error("차트에 표시할 데이터가 없습니다.")

st.markdown("""
<div style="text-align:center; padding:20px 0; font-size:0.78rem; color:var(--text-muted);
            border-top:1px solid var(--border); margin-top:32px;">
  이 서비스는 참고용이며, 투자 권유가 아닙니다.<br>
  Built with Claude Code by 이종호 (<a href="https://jhawk.kr" style="color:var(--text-muted);">jhawk.kr</a>)
</div>
""", unsafe_allow_html=True)
