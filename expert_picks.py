"""전문가 3인 합의 추천 생성 (주 1회, GitHub Actions).

흐름
  1. 후보 추리기(코드): 스냅샷에서 시총·상장기간·상품유형 필터 → 후보 표
  2. 1라운드: 전문가 3명이 웹 검색으로 시장 상황을 확인하고 묶음별 종목 제안 (병렬)
  3. 2라운드: 제안된 종목 전체에 3명이 각각 찬반 투표 (병렬, 구조화 출력)
  4. 합의 판정(코드): 찬성 2표 이상만 남김 — AI가 합의 여부를 쓰지 않는다
  5. 정리: 카드 문구(이유·반대 의견·시장 요약)만 AI가 작성
  6. 검증 통과 시 expert_picks.json 저장, 실패 시 기존 파일 유지 + 종료코드 1

카드의 수치(배당률·수익률·시총)는 json에 넣지 않고 app.py가 스냅샷에서 직접 읽는다.

사용: python expert_picks.py [--dry-run]   (ANTHROPIC_API_KEY 필요)
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import pandas as pd

BASE = Path(__file__).resolve().parent
PARQUET = BASE / "etf_data.parquet"
OUT = BASE / "expert_picks.json"

MODEL = "claude-opus-5-5"
KST = timezone(timedelta(hours=9))
MIN_CAP = 1000          # 시총(억원)
MIN_VOTES = 2           # 3명 중 찬성 수
MAX_PICKS = 3           # 묶음당 최대 종목 수
HISTORY_KEEP = 26       # 이력 보관 주 수
PRICE_IN, PRICE_OUT, PRICE_SEARCH = 4, 20, 0.01   # $/1M 토큰, $/검색 (비용 로그용)
# select_default.MONTHLY_LABELS와 같은 값. 그 모듈은 yfinance를 import해 생성 워크플로에 쓰지 않는다
MONTHLY_LABELS = ["월초", "월중"]

BUNDLES = [
    {"key": "income", "title": "월배당 현금흐름",
     "target": "매달 생활비·현금흐름이 필요한 분",
     "brief": "매달 분배금이 나오고 분배금이 지속 가능한 상품(배당주기 월초·월중만). 원금 훼손 위험도 함께 따진다.",
     "monthly": True},
    {"key": "growth", "title": "장기 성장",
     "target": "10년 이상 길게 모아갈 분",
     "brief": "분배보다 총수익(가격+배당)을 키우는 상품. 저비용 지수형 위주."},
    {"key": "defense", "title": "안정·방어",
     "target": "변동성을 줄이거나 대기 자금을 둘 분",
     "brief": "하락장 완충·현금성·채권 등 주식과 다르게 움직이는 상품."},
]
BUNDLE_KEYS = [b["key"] for b in BUNDLES]

EXPERTS = [
    {"key": "macro", "name": "거시·자산배분",
     "lens": "금리·환율·경기 사이클·자산 간 상대 매력도. 지금 어떤 자산군 비중이 유리한지 본다."},
    {"key": "income", "name": "배당·인컴",
     "lens": "분배금의 지속 가능성, 커버드콜 구조(상승 여력 제한·원금 잠식), 배당 성장성, 세후 분배율을 본다."},
    {"key": "risk", "name": "위험관리",
     "lens": "최대 하락폭, 쏠림·중복 노출, 환헤지 여부, 총보수·규모·유동성, 상품 구조의 숨은 위험을 본다."},
]

# 레버리지·인버스·단기 트레이딩용 상품은 일반 방문자 추천 대상에서 제외
EXCLUDE_PAT = r"레버리지|인버스|2X|3X|선물인버스|곱버스|울트라"

GUARDRAIL = (
    "이 결과는 공개 웹사이트에 '정보 제공용'으로 게시된다. "
    "'매수하세요', '지금 사야 한다', '수익 보장' 같은 권유·단정 표현을 쓰지 말고 "
    "'~에 적합합니다', '~점이 강점입니다', '~위험이 있습니다'처럼 서술한다. 개인 상황을 단정하지 않는다."
)


# ── 1. 후보 ────────────────────────────────────────────────
def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["시총"] >= MIN_CAP) & (~df["상장1Y미만"])]
    d = d[~d["Name"].str.contains(EXCLUDE_PAT, regex=True)]
    picks = pd.concat([
        d[d["배당일"].isin(MONTHLY_LABELS)].nlargest(30, "1Y배당률"),
        d.nlargest(30, "1Y총수익률"),
        d.nlargest(30, "시총"),
        d[d["Name"].str.contains("채권|국채|금리|머니마켓|단기|KOFR|CD|금현물|골드", regex=True)].nlargest(20, "시총"),
    ]).drop_duplicates("Symbol")
    return picks.sort_values("시총", ascending=False)


def candidates_table(c: pd.DataFrame) -> str:
    cols = ["Symbol", "Name", "시총", "1Y배당률", "1Y총수익률", "3M총수익률", "배당일"]
    lines = ["코드 | 상품명 | 시총(억원) | 1Y배당률(%) | 1Y총수익률(%) | 3M총수익률(%) | 배당주기"]
    for r in c[cols].itertuples(index=False):
        lines.append(" | ".join(str(v) for v in r))
    return "\n".join(lines)


# ── API 공통 ───────────────────────────────────────────────
client = None


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def _sources(resp) -> list:
    return [{"title": r.title, "url": r.url} for b in resp.content
            if b.type == "web_search_tool_result" and isinstance(b.content, list) for r in b.content]


def call(system, user, *, tools=None, schema=None, effort="medium"):
    """pause_turn(서버 도구 반복 한도)을 이어 받으며 최종 응답과 출처를 돌려준다."""
    messages = [{"role": "user", "content": user}]
    kw = {"model": MODEL, "max_tokens": 16000, "system": system,
          "output_config": {"effort": effort}}
    if tools:
        kw["tools"] = tools
    if schema:
        kw["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    sources = []
    for _ in range(5):
        resp = client.messages.create(messages=messages, **kw)
        usage.append(resp.usage)   # pause_turn 이어받기도 각각 과금된다
        sources += _sources(resp)
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"응답 비정상 종료: {resp.stop_reason}")
    return resp, list({s["url"]: s for s in sources}.values())   # URL 중복 제거


usage = []


# ── 2. 1라운드: 제안 ───────────────────────────────────────
def bundles_brief() -> str:
    return "\n".join(f"- {b['key']} ({b['title']}): {b['brief']}" for b in BUNDLES)


def round1(expert, table, today):
    system = (
        f"당신은 한국 상장 ETF를 다루는 {expert['name']} 전문가다. 관점: {expert['lens']}\n"
        "일반 개인 투자자를 위한 목적별 ETF 추천 패널에 참여한다. " + GUARDRAIL
    )
    user = (
        f"오늘은 {today}(KST)다. 웹 검색으로 이번 주 금리·환율·주식시장 상황과 당신 관점에서 중요한 뉴스를 확인하라(검색은 꼭 필요한 만큼만).\n"
        "그다음 아래 후보 표에서만 골라 묶음별로 2~4개 종목을 제안하라. 표에 없는 종목은 제안하지 않는다.\n\n"
        f"[묶음]\n{bundles_brief()}\n\n[후보 표 — 대시보드 스냅샷]\n{table}\n\n"
        "출력 형식(한국어, 간결하게):\n"
        "## 시장 판단\n2~3문장\n"
        "## 제안\n각 줄: 묶음키 | 코드 | 상품명 | 이유 1문장\n"
        "## 주의할 위험\n1~2문장"
    )
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}]
    resp, sources = call(system, user, tools=tools)
    return {"expert": expert, "text": _text(resp), "sources": sources[:5]}


def parse_proposals(text, eligible):
    """eligible: 묶음키 → 그 묶음에 넣을 수 있는 종목코드 집합(후보 표 + 묶음 조건)."""
    props = set()
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip("-* ").split("|")]
        if len(parts) >= 3 and parts[1] in eligible.get(parts[0], ()):
            props.add((parts[0], parts[1]))
    return props


# ── 3. 2라운드: 투표 ───────────────────────────────────────
VOTE_SCHEMA = {
    "type": "object",
    "properties": {"votes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "bundle": {"type": "string", "enum": BUNDLE_KEYS},
            "symbol": {"type": "string"},
            "agree": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["bundle", "symbol", "agree", "reason"],
        "additionalProperties": False}}},
    "required": ["votes"],
    "additionalProperties": False,
}


def round2(expert, r1_all, ballot, table):
    others = "\n\n".join(f"### {r['expert']['name']} 전문가 1라운드 의견\n{r['text']}" for r in r1_all)
    system = (
        f"당신은 {expert['name']} 전문가다. 관점: {expert['lens']}\n" + GUARDRAIL
    )
    user = (
        f"패널 3명의 1라운드 의견:\n{others}\n\n[후보 표]\n{table}\n\n"
        f"[투표 대상 — 묶음키 | 코드 | 상품명]\n{ballot}\n\n"
        "투표 대상의 모든 줄에 대해, 그 종목이 해당 묶음의 목적에 맞는 추천으로 적절한지 당신 관점에서 찬성/반대를 정하라. "
        "같은 묶음에 비슷한 상품이 겹치면 더 나은 쪽만 찬성한다. reason은 한국어 1문장(40자 안팎)."
    )
    resp, _ = call(system, user, schema=VOTE_SCHEMA)
    return {"expert": expert, "votes": json.loads(_text(resp))["votes"]}


# ── 4. 합의 판정 (코드) ────────────────────────────────────
def tally(r2_all, proposals, names):
    proposer_cnt = Counter(proposals)
    votes = {k: {"yes": [], "no": []} for k in proposer_cnt}
    for r in r2_all:
        for v in r["votes"]:
            k = (v["bundle"], v["symbol"])
            if k in votes:
                votes[k]["yes" if v["agree"] else "no"].append((r["expert"]["name"], v["reason"]))
    passed = sorted(
        ({"bundle": b, "symbol": s, "name": names[s], "votes": len(v["yes"]),
          "proposers": proposer_cnt[(b, s)], **v}
         for (b, s), v in votes.items() if len(v["yes"]) >= MIN_VOTES),
        key=lambda p: (-p["votes"], -p["proposers"]))
    # 같은 종목이 여러 묶음에서 통과하면 찬성이 많은 묶음 하나에만 남긴다
    final, seen = {b: [] for b in BUNDLE_KEYS}, set()
    for p in passed:
        if p["symbol"] not in seen and len(final[p["bundle"]]) < MAX_PICKS:
            seen.add(p["symbol"])
            final[p["bundle"]].append(p)
    return final


# ── 5. 정리 ────────────────────────────────────────────────
WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "market_summary": {"type": "string"},
        "expert_views": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": [e["key"] for e in EXPERTS]},
                           "view": {"type": "string"}},
            "required": ["key", "view"], "additionalProperties": False}},
        "picks": {"type": "array", "items": {
            "type": "object",
            "properties": {"bundle": {"type": "string", "enum": BUNDLE_KEYS},
                           "symbol": {"type": "string"},
                           "reason": {"type": "string"},
                           "caution": {"type": "string"},
                           "dissent": {"type": "string"}},
            "required": ["bundle", "symbol", "reason", "caution", "dissent"], "additionalProperties": False}},
    },
    "required": ["market_summary", "expert_views", "picks"],
    "additionalProperties": False,
}


def write_cards(r1_all, final):
    lines = []
    for b, lst in final.items():
        for p in lst:
            yes = "; ".join(f"{n}: {r}" for n, r in p["yes"])
            no = "; ".join(f"{n}: {r}" for n, r in p["no"]) or "없음"
            lines.append(f"{b} | {p['symbol']} | {p['name']} | 찬성 {p['votes']}/3\n  찬성: {yes}\n  반대: {no}")
    r1 = "\n\n".join(f"### {r['expert']['name']}\n{r['text']}" for r in r1_all)
    system = "당신은 ETF 추천 패널의 편집자다. 전문가 의견을 일반 투자자가 읽기 쉬운 한국어로 다듬는다. " + GUARDRAIL
    user = (
        f"[1라운드 의견]\n{r1}\n\n[합의 통과 종목과 투표 이유]\n" + "\n".join(lines) + "\n\n"
        "작성 규칙:\n"
        "- market_summary: 이번 주 시장 판단 1문장(60자 이내)\n"
        "- expert_views: 전문가별 핵심 의견 1문장(60자 이내), key는 macro/income/risk\n"
        "- picks: 위 통과 종목 전부. reason은 이 묶음에 맞는 강점 1문장(50자 이내, 위험은 넣지 않음), "
        "caution은 투자 전 유의할 위험 1문장(40자 이내, 없으면 빈 문자열), "
        "dissent는 반대 의견 1문장(반대가 없으면 빈 문자열, 화면에 '반대 의견' 라벨이 붙으므로 '~반대 의견이 있습니다'로 끝내지 않는다)\n"
        "- 수치는 쓰지 않는다(화면이 최신 수치를 따로 보여준다).\n"
        "- 문장 끝은 대시보드 문구와 같은 존칭으로 통일한다(예: '~에 적합합니다', '~위험이 있습니다').\n"
        "- 다른 종목을 가리킬 때는 6자리 코드가 아니라 상품명으로 쓴다."
    )
    resp, _ = call(system, user, schema=WRITE_SCHEMA, effort="low")
    return json.loads(_text(resp))


# ── 6. 검증·저장 ───────────────────────────────────────────
def validate(out, valid):
    errs = []
    for b in out["bundles"]:
        if not 1 <= len(b["picks"]) <= MAX_PICKS:
            errs.append(f"{b['key']}: 종목 수 {len(b['picks'])}")
        for p in b["picks"]:
            if p["symbol"] not in valid:
                errs.append(f"{p['symbol']}: 스냅샷에 없음")
            if not p["reason"].strip():
                errs.append(f"{p['symbol']}: 이유 없음")
    if not out["market_summary"].strip():
        errs.append("시장 요약 없음")
    return errs


def _dump(path, out):
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


def main(dry_run=False):
    global client
    client = anthropic.Anthropic(max_retries=3)
    df = pd.read_parquet(PARQUET)
    asof = str(df["기준일"].max())
    today = datetime.now(KST).strftime("%Y-%m-%d")
    cand = build_candidates(df)
    names = dict(zip(df["Symbol"], df["Name"]))
    valid = set(cand["Symbol"])
    monthly = set(cand.loc[cand["배당일"].isin(MONTHLY_LABELS), "Symbol"])
    eligible = {b["key"]: monthly if b.get("monthly") else valid for b in BUNDLES}
    table = candidates_table(cand)
    print(f"후보 {len(cand)}개 (스냅샷 기준일 {asof})")

    with ThreadPoolExecutor(3) as ex:
        r1_all = list(ex.map(lambda e: round1(e, table, today), EXPERTS))
    proposals = [p for r in r1_all for p in parse_proposals(r["text"], eligible)]
    keys = sorted(set(proposals))
    if not keys:
        raise RuntimeError("1라운드 제안을 읽지 못함")
    ballot = "\n".join(f"{b} | {s} | {names[s]}" for b, s in keys)
    # 2라운드는 투표 대상 종목의 수치만 있으면 된다
    ballot_table = candidates_table(cand[cand["Symbol"].isin({s for _, s in keys})])
    print(f"1라운드 제안 {len(keys)}건")

    with ThreadPoolExecutor(3) as ex:
        r2_all = list(ex.map(lambda e: round2(e, r1_all, ballot, ballot_table), EXPERTS))
    final = tally(r2_all, proposals, names)
    print("합의 통과:", {b: [p["name"] for p in v] for b, v in final.items()})

    text = write_cards(r1_all, final)
    reasons = {(p["bundle"], p["symbol"]): p for p in text["picks"]}
    views = {v["key"]: v["view"] for v in text["expert_views"]}
    src = {r["expert"]["key"]: r["sources"] for r in r1_all}

    out = {
        "asof": asof,
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "model": MODEL,
        "market_summary": text["market_summary"],
        "experts": [{"key": e["key"], "name": e["name"], "view": views.get(e["key"], ""),
                     "sources": src.get(e["key"], [])} for e in EXPERTS],
        "bundles": [{
            "key": b["key"], "title": b["title"], "target": b["target"],
            "picks": [{
                "symbol": p["symbol"], "name": p["name"], "votes": p["votes"],
                "reason": (r := reasons.get((b["key"], p["symbol"]), {})).get("reason", ""),
                "caution": r.get("caution", ""),
                "dissent": r.get("dissent", ""),
            } for p in final[b["key"]]],
        } for b in BUNDLES],
    }
    errs = validate(out, valid)

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = {p["symbol"] for b in prev.get("bundles", []) for p in b["picks"]}
    new = {p["symbol"] for b in out["bundles"] for p in b["picks"]}
    hist = prev.get("history", [])
    if hist and hist[0]["date"] == today:   # 같은 날 재실행은 그날 기록을 대체(변경 내역은 합침)
        same = hist.pop(0)
        old = (old - set(same.get("_added_syms", []))) | set(same.get("_removed_syms", []))
    out["history"] = ([{"date": today, "added": sorted(names[s] for s in new - old),
                        "removed": sorted(names.get(s, s) for s in old - new),
                        "_added_syms": sorted(new - old), "_removed_syms": sorted(old - new)}]
                      + hist)[:HISTORY_KEEP]

    cost = sum(u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT for u in usage) / 1e6
    searches = sum(getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0 for u in usage)
    print(f"토큰 비용 약 ${cost:.2f} + 웹 검색 {searches}회 (${searches * PRICE_SEARCH:.2f})")

    if errs:
        print("검증 실패 — 기존 추천 유지:\n  " + "\n  ".join(errs))
        _dump(BASE / "expert_picks_failed.json", out)
        return 1
    path = BASE / "expert_picks_preview.json" if dry_run else OUT
    _dump(path, out)
    print(f"저장: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
