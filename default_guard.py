"""What-if 디폴트 설계 조건 점검(매일) + 비중 재최적화(월 1회).

디폴트의 설계 목표는 "분배금을 전액 인출해도 원금 유지 + 하락기 손실 상한 이내에서 세후 월분배 최대화"
(2026-10-28). select_default.py의 종목 교체는 이 조건을 다시 계산하지 않으므로 여기서 매일 확인한다.

- 매일: 현재 슬롯 금액으로 guard.start~오늘 가격 수익률(분배금은 인출 → 원주가 기준)을 돌려
  ① 기말 평가액 ≥ 원금 ② guard.stress_start 이후 손실 ≥ stress_floor_pct 를 판정한다.
  위반 시작·breach_alert_days 이상 지속·회복 때만 경고(메일)하고, 나머지 날은 실행 요약에만 남긴다.
- 월 1회(그 달 첫 실행) 또는 위반이 breach_alert_days 이상 지속되면: SLSQP로 같은 조건의 최적 비중을
  계산해, 월분배가 reopt_gain_pct 이상 많거나 위반이 지속 중이면 메일로 '제안'만 한다.
  default_portfolio.json은 바꾸지 않는다 — 반영은 사람이 확인 후 etf-default-apply 절차로.

상태(위반 시작일·마지막 재최적화 월)는 guard_state.json에 둔다.
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from build_snapshot import ROOT
from select_default import DIV_TAX

STATE_JSON = ROOT / "guard_state.json"
DIV_NET = 1 - DIV_TAX
UNIT = 10**7  # 재최적화 금액 단위(천만원)

# 재최적화 후보: 미국 지수형 + 커버드콜(테마 커버드콜·채권·레버리지 제외).
# 상위 폴더 tools/maxdiv_opt.py(2026-10-28 개편에 쓴 로컬 스크립트)와 같은 풀
BROAD = r"미국S&P500(?:\(H\))?$|미국나스닥100(?:\(H\))?$|미국배당다우존스(?:\(H\))?$|커버드콜"
POOL_EXCLUDE = "레버리지|국채|채권|하이일드|빅테크|반도체|AI|테크TOP|테크100|밸류체인"


def name_group(n: str) -> str:
    """재최적화 후보의 노출 그룹(이름 기준 근사). 슬롯 exposure 태그와 같은 분류."""
    if "배당다우존스" in n or "배당100" in n:
        return "SCHD형"
    if "500" in n or "배당커버드콜액티브" in n:
        return "S&P500형"
    return "나스닥형"


def fetch_returns(symbols: dict[str, str], start: str) -> tuple[pd.DataFrame, list[str]]:
    """원주가(auto_adjust=False) 일간 수익률. 분배락이 가격에 반영돼 '분배금 전액 인출'과 같다.
    start 이후 상장(첫 3거래일 결측)·조회 실패 종목은 빼고 이름을 돌려준다."""
    tickers = {f"{sym}.KS": name for name, sym in symbols.items()}
    raw = yf.download(list(tickers), start=start, auto_adjust=False, progress=False, timeout=30)["Close"]
    close = raw.rename(columns=tickers).reindex(columns=list(symbols)).sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    skipped = [n for n in close if close[n].iloc[:3].isna().any()]  # 조회 실패 포함
    return close.drop(columns=skipped).ffill().pct_change().iloc[1:], skipped


def path_stats(amounts: pd.Series, rets: pd.DataFrame, stress_start: str) -> dict:
    """amounts(이름→금액)를 매일 목표 비중으로 유지할 때의 원금 대비 수익률(%)."""
    r = rets[amounts.index] @ (amounts / amounts.sum())
    v = (1 + r).cumprod()
    return {"end": (v.iloc[-1] - 1) * 100,
            "low": (v.min() - 1) * 100,
            "stress": ((1 + r.loc[stress_start:]).prod() - 1) * 100,
            "mdd": (v / v.cummax() - 1).min() * 100}


def monthly_div(amounts: pd.Series, snap: pd.DataFrame) -> float:
    """app.py simul()과 같은 세후 월분배: 연간배당금 × 매수 가능 주식 수 × (1-15.4%) / 12."""
    s = snap.loc[amounts.index]
    return float((s["연간배당금"] * (amounts // s["주가"])).sum() * DIV_NET / 12)


def breaches(st: dict, g: dict) -> list[str]:
    out = []
    if st["end"] < 0:
        out.append(f"원금 미달 {st['end']:+.2f}%")
    if st["stress"] < g["stress_floor_pct"]:
        out.append(f"{g['stress_start']} 이후 손실 {st['stress']:+.2f}% < 상한 {g['stress_floor_pct']}%")
    return out


def fmt_stats(st: dict, g: dict) -> str:
    return (f"원금 대비 {st['end']:+.2f}% (최저 {st['low']:+.1f}%, MDD {st['mdd']:.1f}%) · "
            f"{g['stress_start']} 이후 {st['stress']:+.2f}% (상한 {g['stress_floor_pct']}%)")


def reoptimize(snap: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame] | None:
    """조건(원금 유지·하락기 손실 상한·그룹 한도·종목 상한·현금 고정)을 지키며 세후 분배율 가중합 최대화.
    천만원 단위로 내린 뒤 남는 돈은 조건을 지키는 범위에서 고분배 종목부터 채운다. 실패 시 None."""
    from scipy.optimize import minimize  # 대시보드 이미지엔 scipy가 없어 지연 import

    g, total = cfg["guard"], cfg["total"]
    cash_slot = next(s for s in cfg["slots"] if s["exposure"] == "현금")
    cash_name, cash = cash_slot["name"], cash_slot["amount"]
    cap, max_group = g["cap_per_etf"], cfg["rules"]["max_exposure_pct"] / 100
    log_floor = np.log1p(g["stress_floor_pct"] / 100)

    us = snap[snap.index.str.contains("미국") & (snap["시총"] >= cfg["rules"]["min_marcap"])]
    us = us[us.index.str.contains(BROAD) & ~us.index.str.contains(POOL_EXCLUDE)]
    pool = sorted(set(us.index) | {cash_name})
    rets, _ = fetch_returns(snap.loc[pool, "Symbol"].to_dict(), g["start"])
    names = list(rets.columns)
    if cash_name not in names:
        return None
    R, Rs = rets.values, rets.loc[g["stress_start"]:].values
    yld = (snap.loc[names, "연간배당금"] / snap.loc[names, "주가"] * DIV_NET).values
    k, ci = len(names), names.index(cash_name)
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        if i != ci:
            groups.setdefault(name_group(n), []).append(i)

    # 수익 조건(원금 유지·하락기 손실 상한)과 그룹 한도. SLSQP 제약과 반올림 후 검사가 같은 식을 쓴다
    returns_ok = [lambda w: np.log1p(R @ w).sum(), lambda w: np.log1p(Rs @ w).sum() - log_floor]
    group_ok = [lambda w, idx=idx: max_group - w[idx].sum() for idx in groups.values()]

    bnds = [(0, cap / total)] * k
    bnds[ci] = (cash / total, cash / total)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    cons += [{"type": "ineq", "fun": f} for f in returns_ok + group_ok]
    best = None
    for seed in range(8):
        w0 = np.random.default_rng(seed).random(k)
        w0[ci] = cash / total * w0.sum()
        res = minimize(lambda w: -yld @ w, w0 / w0.sum(), bounds=bnds, constraints=cons,
                       method="SLSQP", options={"maxiter": 500})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None

    def feasible(a: np.ndarray) -> bool:
        w = a / total
        return a.max() <= cap and all(f(w) >= -1e-12 for f in returns_ok + group_ok)

    amt = np.floor(best.x * total / UNIT + 1e-6) * UNIT
    amt[ci] = cash
    order = [i for i in np.argsort(-yld) if i != ci]  # 고분배 종목부터 천만원씩
    while amt.sum() < total:
        i = next((i for i in order if feasible(amt + UNIT * (np.arange(k) == i))), None)
        if i is None:
            return None  # 조건을 지키며 잔액을 채울 수 없음
        amt[i] += UNIT
    out = pd.Series(amt, index=names)
    return out[out > 0].astype(int).sort_values(ascending=False), rets


def run(cfg: dict, snap: pd.DataFrame, as_of: str) -> tuple[list[str], list[str]]:
    """(실행 요약 줄, 메일 경고 줄). guard_state.json을 갱신한다."""
    g = cfg["guard"]
    state = json.loads(STATE_JSON.read_text(encoding="utf-8")) if STATE_JSON.exists() else {}
    amounts = pd.Series({s["name"]: s["amount"] for s in cfg["slots"]}, dtype="int64")
    rets, skipped = fetch_returns(snap.loc[amounts.index, "Symbol"].to_dict(), g["start"])
    if skipped:
        return [], [f"설계 조건 점검 생략(시세 조회 실패·이력 부족): {', '.join(skipped)}"]

    st = path_stats(amounts, rets, g["stress_start"])
    cur_div = monthly_div(amounts, snap)
    bad = breaches(st, g)
    summary = [f"- 설계 조건: {fmt_stats(st, g)} · 세후 월분배 {cur_div / 1e4:,.0f}만 — "
               + ("⚠️ 위반" if bad else "충족")]
    alerts = []

    since = state.get("breach_since")
    long_breach = False
    if bad and not since:
        state["breach_since"] = as_of
        alerts.append(f"설계 조건 위반 시작: {', '.join(bad)} — {g['breach_alert_days']}일 넘게 이어지면 재최적화안을 보냅니다")
    elif bad:
        days = (date.fromisoformat(as_of) - date.fromisoformat(since)).days
        if days >= g["breach_alert_days"] and not state.get("long_alerted"):
            state["long_alerted"] = long_breach = True
            alerts.append(f"설계 조건 위반 {days}일째({since}~): {', '.join(bad)}")
    elif since:
        del state["breach_since"]
        state.pop("long_alerted", None)
        alerts.append(f"설계 조건 회복({since}부터 위반): {fmt_stats(st, g)}")

    month = as_of[:7]
    if long_breach or state.get("last_reopt") != month:
        state["last_reopt"] = month
        lines, proposal = _reopt(cfg, snap, amounts, cur_div, bool(bad))
        summary += lines
        alerts += proposal

    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary, alerts


def _reopt(cfg: dict, snap: pd.DataFrame, amounts: pd.Series, cur_div: float,
           breached: bool) -> tuple[list[str], list[str]]:
    """(실행 요약 줄, 메일 경고 줄)."""
    g = cfg["guard"]
    try:
        result = reoptimize(snap, cfg)
    except Exception as e:  # 재최적화 실패가 일일 점검을 막지 않도록
        return [], [f"월간 재최적화 실패: {type(e).__name__}: {e}"]
    if result is None:
        return [], ["월간 재최적화: 조건을 모두 지키는 비중을 찾지 못함 — 하락기 손실 상한·그룹 한도 재검토 필요"]
    amt, rets = result
    st, div = path_stats(amt, rets, g["stress_start"]), monthly_div(amt, snap)
    gain = (div / cur_div - 1) * 100
    head = f"재최적화안: 세후 월분배 {div / 1e4:,.0f}만 (현재 대비 {gain:+.1f}%) · {fmt_stats(st, g)}"
    if not (breached or gain >= g["reopt_gain_pct"]):
        return [f"- {head}"], []
    why = "현재 디폴트가 설계 조건 위반" if breached else f"월분배 {g['reopt_gain_pct']}% 이상 개선"
    lines = [f"디폴트 비중 재조정 제안({why}) — 자동 반영하지 않습니다. {head}"]
    current = amounts.to_dict()
    for n, a in amt.items():
        before = current.pop(n, 0)
        diff = "신규" if not before else ("유지" if before == a else f"현재 {before / 1e6:,.0f}백만")
        lines.append(f"    {a / 1e6:>5,.0f}백만  {n} ({diff})")
    lines += [f"    제외  {n} (현재 {a / 1e6:,.0f}백만)" for n, a in current.items()]
    return [f"- {head}"], ["\n".join(lines)]  # 메일·요약에서 경고 1건으로
