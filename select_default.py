"""What-if 디폴트 포트폴리오 자동 점검.

snapshot.yml이 `build_snapshot.py` 직후 실행한다. 구조(슬롯별 컨셉·금액)와 교체 기준은
`default_portfolio.json`에 고정돼 있고, 이 스크립트는 슬롯마다 보유 종목을
유지할지 같은 풀의 다른 종목으로 바꿀지만 판단한다.

교체 조건 (둘 중 하나)
1. 보유 종목이 필터 위반: 스냅샷에 없음 · 시총 미달 · 월배당 슬롯인데 월배당 아님
2. 같은 풀 후보가 1Y 총수익률(배당 포함) +swap_margin_1y 이상 앞서고, 3M도 뒤지지 않으며,
   월배당 슬롯이면 세후배당률이 div_yield_tolerance 넘게 낮지 않음

분산 점검
- 슬롯마다 실제 노출 그룹(exposure: SCHD형·S&P500형·나스닥형·현금)을 태그하고, 한 그룹 합계가
  max_exposure_pct를 넘으면 안전 점검 실패로 본다 (json 수동 편집 방지용)
- 다른 그룹 종목과 1년 주간 수익률 상관이 max_cross_corr 이상인 후보는 교체 대상에서 뺀다
  (같은 그룹끼리, 그리고 corr_exempt에 적은 그룹 쌍은 높은 상관이 의도된 구조라 보지 않음 —
  나스닥형↔S&P500형은 상관 0.93이지만 그룹 한도를 맞추려고 일부러 나눈 것)
- 매일 보유 종목끼리도 같은 기준으로 점검해 경고한다. 시세 조회 실패는 경고만 남긴다

데이터 점검: 스냅샷에서 빠진 시총 min_marcap 이상 종목(snapshot_report.json)과
ETF 목록 갱신 실패(LIST_UPDATE_OUTCOME)도 경고에 넣는다.

교체·경고가 있으면 GMAIL_* 환경변수로 메일을 보낸다. 대체 후보 없이 종목이 사라지는 등
안전 점검에 실패하면 json을 건드리지 않고 알림 메일을 보낸 뒤 비정상 종료한다.
"""
from __future__ import annotations

import json
import os
import smtplib
import sys
from collections import Counter
from email.message import EmailMessage
from functools import cache
from itertools import combinations
from pathlib import Path

import pandas as pd
import yfinance as yf

from build_snapshot import ETF_DATA_PARQUET, REPORT_JSON, ROOT

DEFAULT_JSON = ROOT / "default_portfolio.json"

DIV_TAX = 0.154
MONTHLY_LABELS = ["월초", "월중"]  # build_snapshot.py 배당일 분류
MIN_WEEKS = 26  # 상관 계산 최소 표본(신규 상장 종목은 점검 생략)
EXCLUDE = (
    "팔란티어|테슬라|엔비디아|애플|구글|아마존|메타|마이크로소프트|넷플릭스|퀄컴|AMD|"
    "코인베이스|마이크론|브로드컴|버크셔|TSMC|ASML|오라클|인텔|"
    "대만|중국|일본|유럽|인도|글로벌|레버리지|인버스|2X|3X|곱버스"
)


class FatalReview(Exception):
    """디폴트를 유지할 수 없어 사람이 개입해야 하는 상황."""


def violation(row: pd.Series | None, slot: dict, rules: dict) -> str | None:
    if row is None:
        return "스냅샷에 없음(상장폐지·데이터 누락)"
    if row["시총"] < rules["min_marcap"]:
        return f"시총 {row['시총']:,.0f}억 < {rules['min_marcap']:,}억"
    if slot["monthly"] and row["배당일"] not in MONTHLY_LABELS:
        return f"월배당 아님(배당일={row['배당일']})"
    return None


def beats(cand: pd.Series, inc: pd.Series, monthly: bool, rules: dict) -> bool:
    if cand["1Y총수익률"] - inc["1Y총수익률"] < rules["swap_margin_1y"]:
        return False
    if cand["3M총수익률"] < inc["3M총수익률"]:
        return False
    return not monthly or cand["세후배당률"] >= inc["세후배당률"] - rules["div_yield_tolerance"]


@cache
def weekly_returns(symbol: str) -> pd.Series | None:
    """1년 주간 총수익률. 일간은 한·미 장 시차로 상관이 낮게 왜곡돼 주간을 쓴다."""
    try:
        close = yf.Ticker(f"{symbol}.KS").history(period="1y", auto_adjust=True, timeout=30)["Close"]
    except Exception:
        return None
    ret = close.resample("W-FRI").last().pct_change().dropna()
    return ret if len(ret) >= MIN_WEEKS else None


def corr(df: pd.DataFrame, a: str, b: str) -> float | None:
    ra = weekly_returns(df.loc[a, "Symbol"]) if a in df.index else None
    rb = weekly_returns(df.loc[b, "Symbol"]) if b in df.index else None
    return None if ra is None or rb is None else ra.corr(rb)


def review(df: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[str]]:
    rules = cfg["rules"]
    df = df.assign(세후배당률=df["연간배당금"] / df["주가"] * 100 * (1 - DIV_TAX)).set_index("Name")

    exempt = {frozenset(p) for p in rules.get("corr_exempt", [])}

    def cross(a: str, b: str) -> bool:
        """상관 점검 대상인 그룹 쌍인지."""
        return a != b and frozenset((a, b)) not in exempt

    def similar(name: str, slot: dict) -> bool:
        """점검 대상 그룹 슬롯 종목과 상관이 max_cross_corr 이상인지."""
        others = (s["name"] for s in cfg["slots"] if cross(s["exposure"], slot["exposure"]))
        return any((c := corr(df, name, o)) is not None and c >= rules["max_cross_corr"] for o in others)

    eligible = df[~df.index.str.contains(EXCLUDE, regex=True)]
    taken = {s["name"] for s in cfg["slots"]}
    changes, warnings = [], []
    for slot in cfg["slots"]:
        pool = eligible[eligible.index.str.contains(slot["pool"], regex=True)]
        inc = df.loc[slot["name"]] if slot["name"] in df.index else None
        keep = [n for n, r in pool.iterrows() if n not in taken and violation(r, slot, rules) is None]
        candidates = pool.loc[keep].sort_values("1Y총수익률", ascending=False)

        reason = violation(inc, slot, rules)
        if reason:
            new = next((n for n in candidates.index if not similar(n, slot)), None)
            if new is None:
                if inc is None:
                    raise FatalReview(f"{slot['name']}: {reason} — 대체 후보 없음, 디폴트 수동 조정 필요")
                warnings.append(f"{slot['name']}: {reason} — 대체 후보가 없어 유지")
                continue
        else:
            new = next((n for n, r in candidates.iterrows()
                        if beats(r, inc, slot["monthly"], rules) and not similar(n, slot)), None)
            if new is None:
                continue
            cand = candidates.loc[new]
            reason = (f"1Y 총수익 {cand['1Y총수익률']:+.2f}% vs {inc['1Y총수익률']:+.2f}% · "
                      f"3M {cand['3M총수익률']:+.2f}% vs {inc['3M총수익률']:+.2f}%")
        changes.append({"slot": slot["id"], "group": slot["group"], "amount": slot["amount"],
                        "from": slot["name"], "to": new, "reason": reason})
        taken.discard(slot["name"])
        taken.add(new)
        slot["name"] = new

    for a, b in combinations(cfg["slots"], 2):
        if cross(a["exposure"], b["exposure"]) and (c := corr(df, a["name"], b["name"])) is not None \
                and c >= rules["max_cross_corr"]:
            warnings.append(f"분산 저하: {a['name']}({a['exposure']}) ↔ {b['name']}({b['exposure']}) 주간 상관 {c:.2f}")
    unchecked = [s["name"] for s in cfg["slots"]
                 if s["name"] in df.index and weekly_returns(df.loc[s["name"], "Symbol"]) is None]
    if unchecked:
        warnings.append(f"상관 점검 생략(시세 조회 실패·이력 부족): {', '.join(unchecked)}")
    return changes, warnings


def data_warnings(min_marcap: int) -> list[str]:
    """스냅샷 누락(대형 종목)·ETF 목록 갱신 실패."""
    out = []
    if os.environ.get("LIST_UPDATE_OUTCOME", "success") != "success":
        out.append("ETF 목록(etf_list.csv) 자동 갱신 실패 — 시총·신규 상장이 이전 값으로 유지됨")
    if REPORT_JSON.exists():
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        big = [m for m in report["누락"] if m["시총"] >= min_marcap]
        if big:
            names = ", ".join(f"{m['Name']}({m['시총']:,}억)" for m in big[:5])
            out.append(f"스냅샷 누락 {len(report['누락'])}종 중 시총 {min_marcap:,}억 이상 {len(big)}종: {names}")
    return out


def exposure_shares(cfg: dict, key: str = "exposure") -> Counter:
    """슬롯 key(exposure·group)별 비중(%). app.py 안내 박스도 같은 함수를 쓴다."""
    shares = Counter()
    for s in cfg["slots"]:
        shares[s[key]] += s["amount"] * 100 / cfg["total"]
    return shares


def validate(cfg: dict) -> None:
    names = [s["name"] for s in cfg["slots"]]
    if len(names) != len(set(names)):
        raise FatalReview(f"종목 중복: {names}")
    if sum(s["amount"] for s in cfg["slots"]) != cfg["total"]:
        raise FatalReview("슬롯 금액 합계가 total과 다름")
    limit = cfg["rules"]["max_exposure_pct"]
    if over := {k: f"{v:g}%" for k, v in exposure_shares(cfg).items() if v > limit}:
        raise FatalReview(f"노출 그룹 비중 상한 {limit}% 초과: {over}")


def send_mail(subject: str, lines: list[str]) -> None:
    user, pw, to = (os.environ.get(k) for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "MAIL_RECIPIENTS"))
    if not (user and pw and to):
        print("메일 설정 없음 — 발송 생략")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to.replace(",", " ").split())
    msg.set_content("\n".join(lines + ["", "https://etf.jhawk.kr"]))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, "".join(pw.split()))
            s.send_message(msg)
    except Exception as e:  # 메일 실패가 디폴트 반영을 막지 않도록
        print(f"메일 발송 실패: {type(e).__name__}")


def main() -> int:
    df = pd.read_parquet(ETF_DATA_PARQUET)
    cfg = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    as_of = str(df["기준일"].max())
    mail = "--mail" in sys.argv

    try:
        changes, warnings = review(df, cfg)
        validate(cfg)
        warnings += data_warnings(cfg["rules"]["min_marcap"])
    except FatalReview as e:
        print(f"디폴트 점검 중단: {e}")
        if mail:
            send_mail(f"[ETF] 디폴트 자동 점검 실패 ({as_of})", [f"기존 디폴트를 유지했습니다. 사유: {e}"])
        return 1

    if changes:
        cfg["history"].extend({"date": as_of, **c} for c in changes)
        DEFAULT_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = [f"### What-if 디폴트 자동 점검 ({as_of})", ""]
    summary += [f"- {c['from']} → **{c['to']}** ({c['reason']})" for c in changes]
    if not changes:
        summary.append(f"- 교체 없음 ({len(cfg['slots'])}개 슬롯 모두 유지)")
    summary.append("- 노출 그룹: " + " · ".join(f"{k} {v:g}%" for k, v in exposure_shares(cfg).most_common()))
    summary += [f"- ⚠️ {w}" for w in warnings]
    text = "\n".join(summary)
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(path).write_text(text + "\n", encoding="utf-8")

    if mail and (changes or warnings):
        lines = [f"ETF 대시보드 What-if 디폴트 자동 점검 결과입니다 (종가 기준일 {as_of}).", ""]
        for c in changes:
            lines += [f"[{c['group']} · {c['amount'] // 10**6:,}백만원]",
                      f"  {c['from']} → {c['to']}", f"  사유: {c['reason']}", ""]
        lines += [f"[경고] {w}" for w in warnings]
        send_mail(f"[ETF] 디폴트 교체 {len(changes)}건 · 경고 {len(warnings)}건 ({as_of})", lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
