# 06. Model Experiment Specification

## 1. 목적

본 실험의 핵심 질문은 다음과 같다.

> 동일한 Point-in-Time 정보와 동일한 운용 제약 아래에서 전통적 포트폴리오 최적화, 예측 후 최적화, 직접 강화학습 기반 자산배분 방식은 한국 주식시장에서 OOS 성과와 위험 특성이 어떻게 다른가?

본 프로젝트의 목적은 특정 AI 모델이 시장을 이긴다는 것을 보이는 것이 아니다.

비교의 핵심은 다음 세 가지 접근법의 차이를 공정하게 확인하는 것이다.

| 접근법                 | 핵심 아이디어                                                       |
| ---------------------- | ------------------------------------------------------------------- |
| Traditional            | 과거 수익률과 위험 구조를 이용해 직접 포트폴리오 구성               |
| Forecast-then-Optimize | 미래 수익률을 예측한 뒤 예측값을 포트폴리오 최적화에 사용           |
| Direct RL              | 미래 수익률을 따로 예측하지 않고 상태에서 직접 포트폴리오 비중 결정 |

---

## 2. 공통 실험 원칙

모든 모델은 가능한 한 동일한 정보와 동일한 운용 조건을 사용한다.

### Point-in-Time 원칙

날짜 `t`에서 사용할 수 있는 정보만 이용한다.

기본 의사결정 구조는 다음과 같다.

```text
t 거래일 장 마감
        ↓
t 시점 데이터 확정
        ↓
Feature(t) 계산
        ↓
Portfolio Decision
        ↓
t+1 이후 포트폴리오 적용
```

미래 데이터가 과거 Feature나 Universe 구성에 포함되는 것을 허용하지 않는다.

### Universe 원칙

현재 시점의 KRX300 구성종목을 과거 전체 기간에 소급 적용하지 않는다.

최종적으로 다음 세 Universe를 구분한다.

```text
Context Universe
= 당시 KRX300 전체 시장

Prediction Universe
= 예측 모델 학습에 사용하는 비교적 넓은 종목 집합

Investable Universe
= 실제 포트폴리오가 투자할 수 있는 종목 집합
```

Main Investable Universe는 약 50종목을 목표로 하며, 종목 선정은 해당 시점까지 알려진 유동성 및 규모 정보를 이용한다.

Universe 크기에 대한 robustness test에서는 다음 후보를 비교한다.

```text
N = 20 / 30 / 50 / 100
```

기존 5종목 실험은 Legacy Experiment로만 보존한다.

---

## 3. 공통 입력 정보

모든 모델이 사용할 수 있는 공통 시장 정보는 동일하게 유지한다.

### Market Context

```text
KRX300 Return
KRX300 Momentum
KRX300 Volatility
KRX300 Drawdown
Trading Volume Change
Trading Value Change
Trading Value Relative Level
```

추후 개별 종목 데이터 확보 후 다음 항목을 추가한다.

```text
Market Breadth
Cross-sectional Dispersion
Average Correlation
Sector Momentum
Sector Relative Strength
```

### Macro Context

```text
USD/KRW
USD/KRW Return
USD/KRW Momentum

Bank of Korea Base Rate
Base Rate Change

Korean Treasury 3Y Yield
Treasury Yield Change

3Y Yield - Base Rate Spread
```

### Asset-level Features

개별 종목 데이터 확보 후 추가한다.

예정 Feature는 다음과 같다.

```text
Asset Return
Momentum
Volatility
Drawdown
Trading Volume
Trading Value
Liquidity
Market Capitalization
```

최종 Asset Feature는 05D Common Dataset 구축 시 확정한다.

---

# 4. 비교 모델

## 4.1 Traditional Portfolio Models

### EW

EW = Equal Weight.

모든 종목에 동일한 비중을 배분한다.

```text
w_i = 1 / N
```

가장 단순한 Benchmark로 사용한다.

### MVO

MVO = Mean-Variance Optimization.

Mean은 기대수익률,
Variance는 수익률의 분산, 즉 위험을 의미한다.

예상 수익과 공분산 구조를 이용해 포트폴리오 비중을 결정한다.

MVO는 Traditional 모델뿐 아니라 Forecast-then-Optimize의 최종 Portfolio Construction 단계에서도 사용한다.

### HRP

HRP = Hierarchical Risk Parity.

Hierarchical은 계층적,
Risk Parity는 위험 균등 배분을 의미한다.

종목 간 상관관계 구조를 이용해 유사한 종목을 계층적으로 묶고 위험을 분산한다.

MVO처럼 기대수익률 추정에 크게 의존하지 않는 비교 모델로 사용한다.

---

# 5. Forecast-then-Optimize

전체 구조는 다음과 같다.

```text
Features(t)
        ↓
Return Prediction Model
        ↓
Expected Return
        ↓
MVO
        ↓
Target Portfolio Weights
```

즉 머신러닝 모델이 포트폴리오 비중을 직접 출력하지 않는다.

먼저 종목별 미래 기대수익률을 예측하고 그 값을 MVO의 Expected Return 입력으로 사용한다.

## 비교 모델

```text
Ridge Regression
Random Forest
XGBoost 또는 LightGBM
```

필요할 경우 이후 LSTM을 추가한다.

### Ridge

Ridge Regression은 선형 회귀에 L2 Regularization을 추가한 모델이다.

복잡한 모델과 비교하기 위한 단순하고 해석 가능한 ML Baseline으로 사용한다.

### Random Forest

여러 Decision Tree를 결합하는 Ensemble Model이다.

비선형 관계를 학습할 수 있는 비교 모델로 사용한다.

### XGBoost / LightGBM

Gradient Boosting 기반 Tree Ensemble이다.

Tabular Financial Data에서 강력한 비교 모델로 사용한다.

XGBoost와 LightGBM을 모두 사용하는 것이 목적은 아니며, Main Experiment에서는 하나를 선택할 수 있다.

---

# 6. Direct Reinforcement Learning

RL = Reinforcement Learning, 강화학습.

Forecast-then-Optimize와 달리 미래 수익률 예측값을 명시적으로 생성하지 않는다.

전체 구조는 다음과 같다.

```text
State(t)
        ↓
RL Policy
        ↓
Target Portfolio Weights
        ↓
Execution
```

Main RL 비교 후보는 다음과 같다.

```text
EIIE
PPO
SAC
```

TD3는 추가 실험이 필요한 경우에만 고려한다.

## EIIE

EIIE = Ensemble of Identical Independent Evaluators.

각 자산을 동일한 구조의 evaluator가 평가한 뒤 포트폴리오 비중을 생성하는 Portfolio RL 구조다.

기존 프로젝트에서 사용했던 EIIE 실험은 Legacy Result로 보존하고 새로운 공통 데이터셋으로 다시 평가한다.

## PPO

PPO = Proximal Policy Optimization.

정책이 한 번에 지나치게 크게 변하는 것을 제한하면서 학습하는 Policy Gradient 계열 알고리즘이다.

## SAC

SAC = Soft Actor-Critic.

Continuous Action Space를 다루는 Off-policy RL 알고리즘이다.

포트폴리오 비중처럼 연속적인 Action을 출력하는 문제의 비교 모델로 사용한다.

---

# 7. Portfolio Output

모든 전략의 최종 출력 형식을 통일한다.

```text
Date
Ticker
Target Weight
```

각 날짜에서 원칙적으로 다음 조건을 만족해야 한다.

```text
sum(weights) = 1
```

기본 연구는 Long-only Portfolio를 사용한다.

Short Selling 여부, 개별 종목 최대 비중 제한 등 세부 Constraint는 최종 Common Dataset과 투자 Universe 확정 후 별도 Configuration에서 고정한다.

---

# 8. Rebalancing

모든 모델의 Portfolio Weight가 매일 자동으로 전면 변경되는 것으로 가정하지 않는다.

다음 요소를 분리해서 관리한다.

```text
Model Output
        ↓
Target Weight
        ↓
Rebalance Rule
        ↓
Executable Weight
```

따라서 모델 성능과 실제 운용 정책을 구분한다.

Main Rebalance Frequency는 최종 데이터셋 완성 후 확정한다.

후보는 다음과 같다.

```text
Daily
Weekly
Monthly
```

모든 전략 비교에서는 동일한 Rebalance Rule을 적용하는 것을 원칙으로 한다.

---

# 9. Transaction Cost

Transaction Cost를 무시한 수익률만 비교하지 않는다.

비용은 Turnover와 연결한다.

```text
Turnover_t
=
Σ |w_t - w_(t-1)|
```

실제 Cost는 다음 구조로 계산한다.

```text
Trading Cost
=
Turnover × Cost Rate
```

Cost Rate는 최종 Backtest 설정에서 고정하고 모든 전략에 동일하게 적용한다.

결과에서는 반드시 Cost 적용 전과 적용 후 성과를 구분한다.

---

# 10. Risk Overlay

Regime은 전략 선택기가 아니다.

Main 구조는 다음과 같다.

```text
Market / Asset Features
        ↓
Base Portfolio Model
        ↓
Raw Target Weights
        ↓
Market / Risk Context
        ↓
Risk Overlay
        ↓
Executable Weights
```

따라서 다음과 같은 구조는 사용하지 않는다.

```text
Regime
↓
EIIE / MVO / EW 중 하나 선택
```

MVO와 EW는 RL이 선택하는 후보 전략이 아니라 Independent Baseline이다.

향후 실험에서는 다음 두 버전을 비교한다.

```text
No-Regime
vs
Regime-aware Risk Overlay
```

기존 Hard Regime Selector V1은 Failed OOS Experiment로 보존한다.

---

# 11. Walk-Forward Evaluation

Walk-Forward는 시간 순서를 유지하면서 학습 구간과 평가 구간을 앞으로 이동시키는 방식이다.

Random Train/Test Split을 사용하지 않는다.

구조는 다음과 같다.

```text
Past
│
├── Train
│
├── Validation
│
└── Test
                 ↓
           시간 앞으로 이동
                 ↓
       Train → Validation → Test
```

Financial Time Series에서는 미래 데이터가 과거 학습 과정에 들어가는 Data Leakage를 막기 위해 시간 순서를 반드시 보존한다.

2026년 데이터는 기존 실험에서 이미 관찰했으므로 완전히 새로운 Unseen Holdout이라고 표현하지 않는다.

---

# 12. 평가 지표

단순 누적 수익률만으로 모델을 평가하지 않는다.

## Return

```text
Cumulative Return
Annualized Return
Cost-adjusted Return
```

## Risk

```text
Annualized Volatility
Sharpe Ratio
Maximum Drawdown
```

MDD = Maximum Drawdown.

전체 기간에서 경험한 가장 큰 고점 대비 하락률이다.

## Portfolio Structure

```text
HHI
Maximum Weight
Effective Number of Assets
```

HHI = Herfindahl-Hirschman Index.

```text
HHI = Σ w_i²
```

특정 종목에 비중이 몰릴수록 값이 커진다.

Effective Number of Assets는 다음과 같이 계산한다.

```text
Effective N = 1 / HHI
```

## Trading / Execution

```text
Turnover
Rebalance Count
Transaction Cost
Weight Drift
```

---

# 13. Fair Comparison Rule

모델 간 비교에서 다음 조건은 가능한 한 동일하게 유지한다.

| 요소                   | 원칙                 |
| ---------------------- | -------------------- |
| Historical Period      | 동일                 |
| Point-in-Time Universe | 동일                 |
| Input Information      | 가능한 범위에서 동일 |
| Rebalance Rule         | 동일                 |
| Transaction Cost       | 동일                 |
| Portfolio Constraints  | 동일                 |
| Evaluation Period      | 동일                 |
| Performance Metrics    | 동일                 |

모델 구조상 반드시 달라야 하는 부분만 다르게 한다.

---

# 14. Legacy Experiments

01~04 Notebook 결과는 삭제하거나 새로운 실험에 맞춰 소급 수정하지 않는다.

기존 결과는 초기 연구 과정과 실패 사례를 보여주는 Legacy Experiment로 보존한다.

특히 기존 결과에서 확인된 사항은 다음과 같다.

```text
EIIE는 2025년 MVO/EW보다 우수하지 않았음.

기존 EIIE Portfolio는 NAVER 비중이 매우 높았음.

기존 Regime V1 Hard Selector는
2026년 OOS에서 Baseline을 이기지 못했음.
```

이 결과는 새로운 Common Dataset 실험을 설계하게 된 연구 동기로 사용한다.

---

# 15. Planned Experiment Pipeline

```text
05A
Point-in-Time Universe
        ↓
05B
Macro Data
        ↓
05C
Market / Macro Features
        ↓
05D
Common Dataset
        ↓
06
Supervised Return Models
        ↓
07
Forecast → Portfolio Optimization
        ↓
08
Direct RL Portfolio
        ↓
09
Risk / Regime Overlay
        ↓
10
Walk-Forward Evaluation
```

---

# 16. 아직 확정하지 않은 설정

다음 항목은 데이터 확보 후 Configuration 단계에서 확정한다.

```text
Prediction Horizon
Rebalance Frequency
Training Window Length
Validation Window Length
Investable Universe Size
Maximum Asset Weight
Minimum Asset Weight
Transaction Cost Rate
Slippage
MVO Objective / Risk Aversion
RL Reward Function
```

이 값들은 모델마다 임의로 다르게 선택하지 않고 공정 비교 원칙에 맞게 명시적으로 관리한다.

---

# 17. 핵심 연구 원칙

본 프로젝트의 핵심은 다음 구조를 지키는 것이다.

```text
같은 시점의 정보
+
같은 투자 가능 종목
+
같은 운용 제약
+
같은 거래비용
+
같은 평가 구간

↓

Traditional
vs
Forecast-then-Optimize
vs
Direct RL
```

따라서 최종 질문은 단순히

```text
어떤 모델의 수익률이 가장 높은가?
```

가 아니라,

```text
각 Portfolio Construction Paradigm이
수익률, 위험, 집중도, 회전율, 비용 측면에서
어떤 행동 차이를 보이는가?
```

이다.
