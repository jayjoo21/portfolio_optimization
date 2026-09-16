\# 05A. Research Specification



\## 1. Project Objective



한국 주식시장을 대상으로 동일한 시점가용 정보(Point-in-Time Information)와

동일한 운용 제약 아래에서 서로 다른 포트폴리오 의사결정 방식을 비교한다.



비교할 세 가지 접근은 다음과 같다.



\### A. Traditional Portfolio

\- Equal Weight

\- MVO

\- HRP



\### B. Forecast-then-Optimize

\- Ridge → MVO

\- Random Forest → MVO

\- XGBoost / LightGBM → MVO

\- LSTM → MVO (Optional)



\### C. Direct Reinforcement Learning

\- EIIE

\- PPO

\- SAC



추가적으로 Market Context 및 Market Regime 정보를 활용한

Risk Control의 효과를 검증한다.





\---



\## 2. Main Research Question



동일한 시점가용 정보와 동일한 운용 제약 아래에서,



\- 전통적 포트폴리오 최적화

\- 수익률 예측 후 포트폴리오 최적화

\- 직접 강화학습 기반 Portfolio Allocation



은 한국 주식시장에서 Out-of-Sample 기준으로



\- 수익률

\- 위험

\- 집중도

\- Turnover

\- Transaction Cost



측면에서 어떤 차이를 보이는가?





\---



\## 3. Sub Research Questions



\### RQ2. Rich Market Context



가격 중심의 제한적인 State보다

시장 전체의 가격, 거래량, 변동성, Breadth, Correlation,

수급, 환율 등의 Market Context를 추가했을 때



예측 및 Portfolio Allocation의

성과와 안정성이 개선되는가?





\### RQ3. Regime-Aware Risk Control



Market Regime 정보를 이용한 Risk Control이



\- 단순 전략 선택 방식

\- Regime을 사용하지 않는 Risk Control



과 비교했을 때



거래비용을 반영한 이후에도

성과 또는 위험 특성을 개선하는가?





\### RQ4. Universe Robustness



모델 간 성과 차이가



\- Investable Universe 확대

\- Walk-Forward Evaluation



에서도 유지되는가?





\---



\# 4. Universe Design



Universe는 역할에 따라 세 종류로 분리한다.





\## 4.1 Legacy PoC Universe



기존 01\~04 실험에서 사용한 5종목.



\- 005930 삼성전자

\- 000660 SK하이닉스

\- 005380 현대차

\- 035420 NAVER

\- 105560 KB금융



Purpose:



\- 기존 FinRL/EIIE 실험 보존

\- 기존 MVO / EW 비교 보존

\- Regime V1 결과 보존

\- N=5 Scale Baseline



기존 결과는 수정하지 않는다.





\## 4.2 Context Universe



\### Primary Candidate

KRX300



Purpose:



\- Market Breadth

\- Market Dispersion

\- Cross-sectional Correlation

\- Sector Leadership

\- Market Momentum

\- Market Volatility

\- Market Regime

\- Risk Context



주의:



현재 KRX300 구성종목을

2018년 이후 전체 기간에 소급 적용하지 않는다.



가능한 경우 당시 실제 구성종목,

즉 Point-in-Time Constituents를 사용한다.





\## 4.3 Prediction Universe



초기 후보:



KRX300 Point-in-Time Constituent Universe



Purpose:



\- Cross-sectional Supervised Learning

\- Ridge

\- Random Forest

\- XGBoost / LightGBM

\- Optional LSTM



Prediction Universe와

실제 Investable Universe는 동일할 필요가 없다.





\## 4.4 Investable Universe



Main Experiment의 초기 목표:



약 50개 종목



선정 조건 후보:



\- Point-in-Time 데이터만 사용

\- 충분한 거래대금

\- 충분한 거래일

\- 거래정지 종목 처리

\- 신규상장 처리

\- 상장폐지 종목 처리

\- Size / Liquidity 기준



정확한 선정 규칙은

Data Specification 단계에서 확정한다.





\### Robustness Universes



\- N = 5 : 기존 PoC

\- N = 20 또는 30

\- N = 50 : Main Experiment

\- N = 100 : Scale Test

\- N = 200 : Optional Scale Test



500개 Direct Allocation은

현재 Main Scope에는 포함하지 않는다.



필요할 경우 별도의

Large-Universe Scalability 연구로 분리한다.





\---



\# 5. Time Design



\## Data Frequency



Daily





\## State Update Frequency



Daily





\## Forecast Horizon



TBD





\## Portfolio Decision Frequency



TBD





\## Rebalance Frequency



TBD





다음 네 개를 서로 다른 개념으로 관리한다.



1\. Data Frequency

2\. State / Signal Update Frequency

3\. Forecast Horizon

4\. Rebalance Frequency



Forecast Horizon과 Rebalance Frequency가

반드시 동일하다고 가정하지 않는다.





\---



\# 6. Portfolio Constraints



초기 후보:



\- Long-only

\- Short Selling 없음

\- Cash 허용 여부: TBD

\- Maximum Single-Stock Weight: TBD

\- Maximum Portfolio Exposure: TBD

\- Turnover Constraint: TBD

\- Rebalance Threshold: TBD

\- Transaction Cost Model: TBD



한국 시장에서 실제 적용 가능한

수수료 / 세금 / Slippage 구조를 별도로 정의한다.





\---



\# 7. Evaluation Metrics



\## Performance



\- Total Return

\- Annualized Return

\- Annualized Volatility

\- Sharpe Ratio

\- Maximum Drawdown





\## Portfolio Behavior



\- Turnover

\- Rebalance Count

\- Maximum Single-Asset Weight

\- HHI

\- Effective Number of Assets





\## Execution



\- Transaction Cost

\- Cost-Adjusted Return

\- Weight Drift





\## Robustness



\- Walk-Forward Performance

\- Universe Size Sensitivity

\- Regime Sensitivity

\- Concentration Stability





\---



\# 8. Validation Policy



2026 데이터는 기존 실험에서 이미 여러 차례 확인하였다.



따라서 새로운 모델의

완전히 untouched / pristine holdout으로 주장하지 않는다.



향후 주요 모델 비교는

Walk-Forward Evaluation을 중심으로 설계한다.





다음 Bias를 명시적으로 관리한다.



\- Look-Ahead Bias

\- Survivorship Bias

\- Selection Bias

\- Data Leakage





특히 다음을 Point-in-Time 기준으로 관리한다.



\- Universe membership

\- Financial data

\- Market data

\- Feature availability





\---



\# 9. Regime Experiments



\## Regime V1 - Existing



현재 04 notebook에서 완료한 실험.



Market State

→ Rule-Based Strategy Selector

→ Cash / Equal Weight / MVO / EIIE



결과는 그대로 보존한다.





\## Regime V2 - Future



Base Portfolio Model

→ Raw Target Weights

→ Market Regime / Risk Context

→ Risk Overlay

→ Executable Portfolio Weights



V1과 V2는 서로 다른 실험으로 취급한다.



V1을 삭제하거나 V2의 결과로 대체하지 않는다.





\---



\# 10. Modeling Scope



\## Traditional



\- Equal Weight

\- MVO

\- HRP





\## Supervised Forecasting



\- Ridge

\- Random Forest

\- XGBoost / LightGBM

\- LSTM Optional





\## Forecast-then-Optimize



Predicted Expected Return

→ Portfolio Optimization

→ Target Weights





\## Direct RL



\- EIIE

\- PPO

\- SAC



동일한 Market Information Set과

동일한 Portfolio Environment를 사용하는 것을 원칙으로 한다.





\---



\# 11. Data Priority



\## Priority 1



\- Stock OHLCV

\- Market Index

\- Market Breadth

\- Volatility

\- Drawdown

\- Cross-sectional Correlation

\- Trading Volume

\- USD/KRW

\- Investor Flow





\## Priority 2



\- KOSPI200 Futures

\- Options

\- VKOSPI / Volatility Information

\- Global Equity Index





\## Priority 3



\- PER

\- PBR

\- Financial Statements

\- DART Disclosures





\## Priority 4



\- Order Book

\- Bid / Ask

\- Spread

\- Intraday Data

\- Real-time Execution Data





\---



\# 12. Development Roadmap



01 EIIE Baseline                     DONE



02 MVO / Equal Weight Baseline       DONE



03 Strategy Comparison               DONE



04 Regime V1 Experiment              DONE



05A Research Specification           CURRENT



05B Data Specification



05C Common Feature Pipeline



06 Supervised Forecasting



07 Forecast → Portfolio



08 Direct RL Comparison



09 Regime Experiments / Risk Overlay



10 Walk-Forward / Robustness / Costs



11 Data Engineering



12 API / Model Serving



13 Frontend / Decision Replay



14 Real-Time Extension





\---



\# 13. Immediate Next Step



다음 단계는 모델 학습이 아니다.



먼저 다음을 검증한다.



1\. KRX300 historical Point-in-Time constituents 확보 가능 여부

2\. Constituents 데이터의 기간

3\. 신규 편입 / 제외 이력 처리 가능 여부

4\. 종목코드 변경 / 상장폐지 처리 가능 여부

5\. Daily OHLCV와 constituents 데이터 정렬 가능 여부



이 검증 후

05B Data Specification으로 이동한다.
