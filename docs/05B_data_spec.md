# 05B. Data Specification

## 1. Purpose

이 문서는 프로젝트에서 사용할 데이터의

- 출처
- 원본 필드
- 주기
- 실제 사용 가능 시점
- 저장 위치
- 파생 Feature
- 사용 모델
- 우선순위
  를 정의한다.

모든 데이터는 가능한 한 Point-in-Time 기준으로 관리한다.

---

# 2. Data Source Overview

## 2.1 KRX Open API

Primary use:

- 한국 주식시장 일별 데이터
- 시장/지수 데이터
- 거래량 / 거래대금
- 종목 기본정보
- 선물 / 옵션 등 국내 파생시장 데이터

Current status:

- API Key 발급 완료
- API 활용 신청 완료
- HTTP 200 연결 확인 완료

---

## 2.2 pykrx + KRX Login

Primary use:

- KRX300 historical constituents
- Point-in-Time Index Membership

Confirmed dates:

- 2018-02-05
- 2018-06-15
- 2020-01-02
- 2023-01-02
- 2025-01-02
- 2026-09-15

Historical membership 변화 확인 완료.

---

## 2.3 ECOS

ECOS = Economic Statistics System

Primary use:

- USD/KRW
- 한국은행 기준금리
- 국고채 금리
- Macro / Monetary Context

Status:

- 추가 예정

---

## 2.4 OpenDART

Primary use:

- Financial Statements
- Corporate Disclosures
- Fundamental Features

Status:

- Priority 2~3
- 현재 Core Pipeline에서는 제외

---

## 2.5 KIS Open API

Primary use:

- Real-time market data
- WebSocket
- Order / Execution
- Paper Trading

Status:

- 서비스 / 실시간 단계에서 사용
- 현재 Research Dataset에는 사용하지 않음

---

# 3. Core Data Specification

| Group         | Raw Data              | Source       | Frequency         | Availability        | Derived Features           | Used By          | Priority |
| ------------- | --------------------- | ------------ | ----------------- | ------------------- | -------------------------- | ---------------- | -------- |
| Universe      | KRX300 constituents   | pykrx / KRX  | Event / Date      | 해당 시점 구성종목  | membership flag            | All              | P1       |
| Stock Price   | OHLC                  | KRX Open API | Daily             | 장 마감 후          | return, momentum, MA       | ML / RL / MVO    | P1       |
| Stock Volume  | Volume                | KRX Open API | Daily             | 장 마감 후          | volume change, liquidity   | ML / RL          | P1       |
| Trading Value | 거래대금              | KRX Open API | Daily             | 장 마감 후          | liquidity proxy            | Universe / ML    | P1       |
| Stock Info    | 종목기본정보          | KRX Open API | Daily / Reference | 조회 시점           | market / listing info      | Universe         | P1       |
| KRX300 Index  | Index OHLC            | KRX Open API | Daily             | 장 마감 후          | market return, momentum    | Context / Regime | P1       |
| Sector Index  | KRX300 sector indices | KRX Open API | Daily             | 장 마감 후          | sector leadership          | Context / Regime | P1       |
| Breadth       | constituent returns   | Derived      | Daily             | 장 마감 후          | advance ratio              | Regime           | P1       |
| Dispersion    | stock returns         | Derived      | Daily             | 장 마감 후          | cross-sectional std        | Regime           | P1       |
| Correlation   | stock returns         | Derived      | Daily             | rolling             | avg correlation            | Regime / Risk    | P1       |
| FX            | USD/KRW               | ECOS         | Daily             | publication 기준    | FX return / momentum       | Context          | P1       |
| Rate          | 기준금리              | ECOS         | Event             | 발표 시점           | policy rate level/change   | Context          | P2       |
| Bond Yield    | 국고채 금리           | ECOS         | Daily             | 해당 통계 공개 시점 | yield change               | Context          | P2       |
| Futures       | KOSPI200 futures      | KRX          | Daily             | 장 마감 후          | basis / return             | Regime           | P2       |
| Options       | KOSPI200 options      | KRX          | Daily             | 장 마감 후          | option activity / IV proxy | Regime           | P2       |
| Fundamentals  | 재무제표              | OpenDART     | Quarterly/Event   | 공시일 이후         | growth, margin             | Stock Selection  | P3       |
| Disclosure    | 기업공시              | OpenDART     | Event             | 공시 시점           | event features             | Event Model      | P3       |
| Order Book    | Bid/Ask/Size          | KIS          | Real-time         | Real-time           | spread, imbalance          | Execution        | P4       |

---

# 4. Point-in-Time Rules

## Universe

현재 시점의 KRX300 구성종목을 과거 전체 기간에 적용하지 않는다.
각 날짜에서 당시 실제 KRX300 구성종목만 사용한다.

---

## Market Data

날짜 t의 종가 기반 Feature는
t의 장 마감 이전 의사결정에 사용할 수 없다.

예:
t 종가를 이용해 계산한 Feature
→ earliest use = t 장 마감 이후 또는 t+1 거래

---

## Fundamental Data

재무제표 기준일이 아니라
실제 공시일을 Availability 기준으로 사용한다.

예:
2025-03-31 분기 데이터라도
2025-05-15에 공시되었다면

2025-03-31부터 사용하지 않고
2025-05-15 이후부터 사용한다.

---

# 5. Storage Design

초기 구조:
data/
├─ raw/
│ ├─ krx/
│ │ ├─ universe/
│ │ ├─ stocks/
│ │ ├─ index/
│ │ └─ derivatives/
│ │
│ ├─ ecos/
│ └─ dart/
│
├─ clean/
│
├─ aligned/
│
└─ features/

Raw 데이터는 가능한 한 수정하지 않는다.
정제 및 Feature 생성 결과는 별도 Layer에 저장한다.

---

# 6. Initial Feature Groups

## Price

- return_1d
- return_5d
- return_20d
- momentum_5d
- momentum_20d
- moving_average_distance
- drawdown

## Risk

- realized_volatility
- rolling_correlation
- cross_sectional_dispersion

## Liquidity

- volume_change
- trading_value
- rolling_trading_value

## Market Context

- KRX300 return
- KRX300 momentum
- breadth
- dispersion
- average correlation

## Sector Context

- sector return
- sector momentum
- sector relative strength

## Macro

- USD/KRW return
- rate level
- rate change

---

# 7. Model Information Set

## Traditional

EW / MVO / HRP

Main inputs:

- price
- return
- covariance
- liquidity filter

---

## Supervised

Ridge / RF / XGBoost / LightGBM

Main inputs:

- stock features
- market context
- liquidity
- FX
- optional macro

Target:
TBD after Horizon decision.

---

## Direct RL

EIIE / PPO / SAC

Main information set:

- Price
- Volume
- Momentum
- Volatility
- Market Context

각 모델 구조에 맞는 State 형태로 변환한다.
동일한 Information Set 사용을 원칙으로 한다.

---

# 8. Deferred Data

현재 Core Study에 바로 포함하지 않는다.

- PER
- PBR
- detailed fundamentals
- DART NLP
- order book
- intraday tick data
- real-time execution
- alternative data

필요한 연구 질문이 생길 때 추가한다.

---

# 9. Immediate Next Step

05C Common Feature Pipeline에서 다음 순서로 구현한다.

1. KRX300 Point-in-Time Membership 저장
2. KRX Stock Daily Data 수집
3. KRX Index / Sector Index 수집
4. ECOS Macro Data 수집
5. Date Alignment
6. Missing Value Policy
7. Feature Generation
8. Leakage Check
