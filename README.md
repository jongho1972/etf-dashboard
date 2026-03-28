# 한국 ETF 대시보드

한국 상장 ETF 전종목의 배당률·수익률 조회, What-if 투자 시뮬레이션, ETF 비교 차트를 제공하는 웹 앱입니다.

**배포 URL:** https://jhawk-etf-dashboard.streamlit.app

---

## 주요 기능

### 1. ETF 조회
- 한국 ETF 전종목 배당률·수익률 Top 20 랭킹
- 월배당금, 시총, 3M/1Y 수익률 필터링
- Name 컬럼 고정 테이블 (모바일 가로 스크롤 지원)

### 2. What-if 분석
- 종목 + 예상 투자금 입력 → 연/월 배당금, 주가차익, 총수익 자동 계산
- 배당소득세(15.4%) 및 양도세 적용
- 3M/1Y 두 기준으로 수익 범위 제시

### 3. ETF 비교 차트
- 복수 ETF 정규화(시작=100) 주가 추이 비교
- 기간 선택: 1개월 ~ 1년
- 기본 비교 종목: 나스닥100커버드콜OTM · S&P500커버드콜OTM · 배당다우존스타겟커버드콜

---

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 기술 스택

- [Streamlit](https://streamlit.io)
- [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) — ETF 종목 목록
- [yfinance](https://github.com/ranaroussi/yfinance) — 주가·배당 데이터

---

개발: 이종호 (jongho1972@gmail.com)
