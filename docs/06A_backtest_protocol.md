# 06A. Backtest & Walk-Forward Protocol

## 1. 목적

모든 Portfolio Model을 동일한 시간축과 동일한 정보 조건에서 평가한다.

본 문서는 다음 항목을 고정한다.

- Signal 시점
- Execution 시점
- Prediction Horizon
- Rebalance Frequency
- Train / Validation / Test 구조
- Walk-Forward 방식

---

# 2. 기본 시간 구조

Main Experiment에서는 다음 순서를 사용한다.

```text
주 마지막 거래일 t
장 마감
    ↓
t까지의 정보 확정
    ↓
Feature(t) 계산
    ↓
Model Decision
    ↓
다음 거래일 t+1 시가에 실행
    ↓
다음 Weekly Rebalance까지 보유
```

따라서:

```text
Feature(t)
→ Decision(t)
→ Execution(t+1)
```

으로 정의한다.

`t`의 종가를 본 뒤 `t`의 종가에 체결했다고 가정하지 않는다.

이는 Look-ahead Bias를 피하기 위함이다.

---

# 3. Rebalance Frequency

## Main Setting

```text
Weekly Rebalancing
```

을 사용한다.

각 주의 마지막 실제 거래일 장 마감 후 포트폴리오를 계산하고, 다음 거래일에 새로운 비중을 적용한다.

예:

```text
금요일 장 마감
↓
Feature 계산
↓
Portfolio Weight 생성
↓
월요일 시가 실행
```

월요일이 휴장일이면 다음 실제 거래일에 실행한다.

## Weekly를 Main으로 사용하는 이유

Daily Rebalancing은 다음 문제가 있다.

```text
높은 Turnover
높은 Transaction Cost
단기 Noise에 대한 과민 반응
```

반대로 Monthly Rebalancing은 Market Context 변화에 너무 느리게 반응할 가능성이 있다.

따라서 Main Experiment에서는 Weekly를 중간적인 기본 설정으로 사용한다.

추후 Robustness Test에서는 다음을 비교할 수 있다.

```text
Daily
Weekly
Monthly
```

---

# 4. Prediction Horizon

Forecast-then-Optimize의 Main Target은 고정된 단순 `5D Return`이 아니라:

```text
Next Rebalance Period Return
```

으로 정의한다.

즉 현재 Portfolio Decision 이후 다음 Rebalance까지 발생하는 종목별 수익률을 예측한다.

개념적으로:

```text
현재 Rebalance 실행가격
        ↓
다음 Rebalance 실행가격
        ↓
Forward Return
```

이다.

### 예시

```text
이번 주 월요일 실행가격 = 100
다음 주 월요일 실행가격 = 105
```

이면:

```text
Forward Return = 105 / 100 - 1
               = 5%
```

이다.

휴일이 있는 경우에도 실제 Rebalance Date를 사용하므로 운용 주기와 Target이 일치한다.

---

# 5. Execution Price

Main Backtest에서는 다음을 우선 사용한다.

```text
Next Trading Day Open
```

즉 Signal은 이전 거래일 종가까지의 정보로 만들고 실제 Portfolio Weight 변경은 다음 거래일 시가부터 적용한다.

향후 종목별 Open Price가 확보되지 않는 경우에만 별도의 대체 규칙을 명시한다.

대체 규칙을 사용할 경우 Main Result와 섞지 않고 명확하게 기록한다.

---

# 6. Train / Validation / Test

Financial Time Series는 Random Split을 사용하지 않는다.

시간 순서를 유지한다.

Main 구조는:

```text
TRAIN
↓
VALIDATION
↓
TEST
```

이다.

## Minimum Training Period

최초 모델 학습에는 최소 약 3년의 데이터를 사용한다.

```text
Minimum Train Window
≈ 3 Years
```

프로젝트 데이터가 2018년부터 시작하기 때문에 지나치게 긴 고정 Training Window를 설정하면 OOS 평가 구간이 너무 짧아진다.

---

# 7. Expanding Window

Main Experiment에서는 Expanding Window를 사용한다.

`Expanding = 시간이 지나면서 학습 데이터가 계속 누적되는 방식`.

예:

```text
Fold 1

2018 ───────── 2020
       TRAIN

2021 H1
VALIDATION

2021 H2
TEST
```

다음 Fold:

```text
Fold 2

2018 ───────────── 2021 H1
          TRAIN

2021 H2
VALIDATION

2022 H1
TEST
```

다음 Fold:

```text
Fold 3

2018 ───────────────── 2021 H2
             TRAIN

2022 H1
VALIDATION

2022 H2
TEST
```

이런 방식으로 앞으로 이동한다.

---

# 8. Window Length

Main Setting:

```text
Training
= Minimum 3 Years
  이후 Expanding

Validation
= 6 Months

Test
= 6 Months

Walk-forward Step
= 6 Months
```

즉 6개월 단위로 새로운 OOS Test 결과를 생성한다.

---

# 9. Validation 역할

Validation 구간은 다음 용도로 사용한다.

```text
Hyperparameter Selection
Model Selection
Training Epoch Selection
Risk Parameter Selection
```

Test 결과를 보고 Hyperparameter를 수정하지 않는다.

기본 과정은:

```text
Train
↓
Validation으로 설정 선택
↓
설정 고정
↓
Train + Validation 정보로 필요 시 재학습
↓
다음 Test 구간 평가
```

로 한다.

---

# 10. Test의 의미

Test는 해당 Walk-Forward Fold에서 모델 선택에 사용하지 않은 미래 구간이다.

각 Test Fold의 결과를 연결하여 전체 OOS 성능을 계산한다.

```text
Test Fold 1
+
Test Fold 2
+
Test Fold 3
+
...
↓
Combined OOS Performance
```

---

# 11. 2026 데이터 처리

2026년 데이터는 기존 Legacy Experiment에서 이미 확인한 적이 있다.

따라서 프로젝트 설명에서:

```text
Completely Unseen 2026 Holdout
```

이라고 표현하지 않는다.

대신:

```text
Walk-Forward OOS Evaluation Period
```

의 일부로 취급한다.

---

# 12. Model별 시간축 통일

## Traditional

```text
Historical Information
↓
EW / MVO / HRP
↓
Target Weights
↓
Weekly Execution
```

## Forecast-then-Optimize

```text
Features
↓
Next Rebalance Return Prediction
↓
Expected Returns
↓
MVO
↓
Target Weights
↓
Weekly Execution
```

## Direct RL

```text
State
↓
Policy
↓
Target Weights
↓
Weekly Execution
```

세 Paradigm 모두 동일한 Rebalance Date와 Execution Rule을 사용한다.

---

# 13. RL과 Weekly Rebalance

RL만 매일 거래하도록 허용하지 않는다.

그렇게 하면:

```text
RL = Daily

MVO = Weekly

ML + MVO = Weekly
```

가 되어 모델 구조뿐 아니라 운용 빈도까지 달라진다.

따라서 Main Comparison에서는 RL도 동일한 Weekly Decision Frequency를 사용한다.

Daily RL은 별도의 추가 실험으로만 다룬다.

---

# 14. Transaction Cost 적용 시점

Portfolio Weight가 실제로 변경되는 Rebalance 시점에만 비용을 적용한다.

```text
Previous Weight
↓
New Target Weight
↓
Turnover 계산
↓
Transaction Cost 차감
↓
New Portfolio 시작
```

이를 통해 단순 수익률뿐 아니라 Frequent Trading의 비용까지 비교한다.

---

# 15. Warm-up Period

20일 Momentum / Volatility 등의 Feature를 계산하기 위해 초기 Warm-up 기간이 필요하다.

Warm-up 구간의 NaN을 0으로 대체하지 않는다.

```text
Raw Dataset Start
2018-02-05

↓ 약 20 거래일 Feature Warm-up

Model-usable Start
≈ 2018-03
```

정확한 시작 날짜는 05D Common Dataset 구축 후 자동 계산한다.

---

# 16. Main Experiment Configuration

현재 Main Configuration은 다음과 같이 고정한다.

| Setting             | Main Value                   |
| ------------------- | ---------------------------- |
| Signal Frequency    | Weekly                       |
| Signal Time         | Last trading day close       |
| Execution           | Next trading day open        |
| Prediction Target   | Next rebalance-period return |
| Minimum Train       | 3 years                      |
| Training Style      | Expanding                    |
| Validation          | 6 months                     |
| Test                | 6 months                     |
| Walk-forward Step   | 6 months                     |
| Portfolio Direction | Long-only                    |
| Transaction Cost    | 적용                         |
| Random Split        | 사용하지 않음                |

---

# 17. Robustness Tests

Main Experiment 완료 후 필요할 경우 다음 조건을 변경해 결과의 안정성을 확인한다.

```text
Rebalance
Weekly
vs Monthly

Universe Size
20
30
50
100

Prediction Horizon
Next Rebalance Period
vs Fixed 5D

Training Window
Expanding
vs Rolling

Transaction Cost
Low
Base
High
```

Main Setting을 본 뒤 유리한 설정만 골라 바꾸지 않는다.

Robustness Test는 사전에 정의한 범위 안에서 수행한다.

---

# 18. 핵심 원칙

최종 비교는 다음 조건을 유지한다.

```text
Same Dates
+
Same Information
+
Same Universe
+
Same Rebalance Dates
+
Same Execution Rule
+
Same Costs
+
Same Portfolio Constraints

↓

Traditional
vs
Forecast-then-Optimize
vs
Direct RL
```

이 원칙을 통해 모델 구조 자체의 차이를 가능한 한 분리해서 비교한다.
