<script setup lang="ts">
import { ref } from 'vue'

import BacktestPerformanceChart from '../components/BacktestPerformanceChart.vue'

const selectedStrategy = ref('EIIE')

const strategies = [
  {
    name: 'EIIE',
    return: '+18.2%',
    sharpe: 1.24,
    mdd: -8.2,
    volatility: 13.1,
    turnover: 42.1,
  },
  {
    name: 'MVO',
    return: '+14.1%',
    sharpe: 1.01,
    mdd: -10.1,
    volatility: 14.2,
    turnover: 35.4,
  },
  {
    name: 'EW',
    return: '+11.4%',
    sharpe: 0.82,
    mdd: -12.4,
    volatility: 15.4,
    turnover: 8.2,
  },
]
</script>

<template>
  <main class="backtest-page">
    <!-- HEADER -->
    <header class="page-header">
      <div>
        <p class="eyebrow">STRATEGY ANALYSIS</p>

        <h1>Backtest</h1>

        <p class="subtitle">동일한 기간과 조건에서 EIIE를 MVO 및 EW baseline과 비교합니다.</p>
      </div>

      <span class="status"> Historical Simulation </span>
    </header>

    <!-- PERFORMANCE CHART -->
    <section class="panel performance-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">CUMULATIVE PERFORMANCE</p>

          <h2>Strategy Comparison</h2>
        </div>

        <span class="prototype-badge"> Prototype Data </span>
      </div>

      <BacktestPerformanceChart />
    </section>

    <!-- METRIC SUMMARY -->
    <section class="metric-grid">
      <button
        v-for="(strategy, index) in strategies"
        :key="strategy.name"
        class="strategy-card"
        :class="{
          selected: selectedStrategy === strategy.name,
        }"
        :style="{
          animationDelay: `${index * 70}ms`,
        }"
        @click="selectedStrategy = strategy.name"
      >
        <div class="strategy-heading">
          <div>
            <span class="strategy-label"> STRATEGY </span>

            <strong>
              {{ strategy.name }}
            </strong>
          </div>

          <span v-if="selectedStrategy === strategy.name" class="selected-indicator">
            Selected
          </span>
        </div>

        <div class="return-value">
          {{ strategy.return }}
        </div>

        <div class="mini-metrics">
          <div>
            <span>Sharpe</span>
            <strong>{{ strategy.sharpe }}</strong>
          </div>

          <div>
            <span>MDD</span>
            <strong>{{ strategy.mdd }}%</strong>
          </div>

          <div>
            <span>Volatility</span>
            <strong>{{ strategy.volatility }}%</strong>
          </div>
        </div>
      </button>
    </section>

    <!-- PERFORMANCE TABLE -->
    <section class="panel table-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">PERFORMANCE METRICS</p>

          <h2>Detailed Comparison</h2>
        </div>
      </div>

      <div class="table">
        <div class="table-row table-head">
          <span>Strategy</span>
          <span>Return</span>
          <span>Sharpe</span>
          <span>MDD</span>
          <span>Volatility</span>
          <span>Turnover</span>
        </div>

        <button
          v-for="strategy in strategies"
          :key="strategy.name"
          class="table-row data-row"
          :class="{
            selected: selectedStrategy === strategy.name,
          }"
          @click="selectedStrategy = strategy.name"
        >
          <strong>
            {{ strategy.name }}
          </strong>

          <span>
            {{ strategy.return }}
          </span>

          <span>
            {{ strategy.sharpe }}
          </span>

          <span> {{ strategy.mdd }}% </span>

          <span> {{ strategy.volatility }}% </span>

          <span> {{ strategy.turnover }}% </span>
        </button>
      </div>
    </section>

    <!-- LOWER ANALYSIS -->
    <section class="analysis-grid">
      <!-- DRAWDOWN -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">DRAWDOWN</p>

            <h2>Maximum Drawdown</h2>
          </div>
        </div>

        <div class="drawdown-list">
          <div v-for="strategy in strategies" :key="strategy.name" class="drawdown-row">
            <span>
              {{ strategy.name }}
            </span>

            <div class="drawdown-track">
              <div
                class="drawdown-fill"
                :style="{
                  width: `${Math.abs(strategy.mdd) * 6}%`,
                }"
              ></div>
            </div>

            <strong> {{ strategy.mdd }}% </strong>
          </div>
        </div>

        <p class="hint">MDD는 고점 대비 최대 하락폭을 의미합니다.</p>
      </div>

      <!-- SELECTED STRATEGY -->
      <Transition name="detail" mode="out-in">
        <div :key="selectedStrategy" class="selected-panel">
          <p class="panel-label dark-label">SELECTED</p>

          <h2>
            {{ selectedStrategy }}
          </h2>

          <p class="selected-description">
            차트와 표에서 전략을 선택해 주요 성과 지표를 빠르게 비교할 수 있습니다.
          </p>

          <div class="selected-stats">
            <template v-for="strategy in strategies" :key="strategy.name">
              <template v-if="strategy.name === selectedStrategy">
                <div>
                  <span>Total Return</span>
                  <strong>{{ strategy.return }}</strong>
                </div>

                <div>
                  <span>Sharpe Ratio</span>
                  <strong>{{ strategy.sharpe }}</strong>
                </div>

                <div>
                  <span>Maximum Drawdown</span>
                  <strong>{{ strategy.mdd }}%</strong>
                </div>

                <div>
                  <span>Turnover</span>
                  <strong>{{ strategy.turnover }}%</strong>
                </div>
              </template>
            </template>
          </div>
        </div>
      </Transition>
    </section>
  </main>
</template>

<style scoped>
.backtest-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 48px 56px 80px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 28px;
}

.page-header h1 {
  margin: 4px 0 8px;
  font-size: 38px;
  letter-spacing: -0.04em;
}

.subtitle {
  margin: 0;
  color: #71717a;
  font-size: 14px;
  line-height: 1.6;
}

.eyebrow,
.panel-label {
  margin: 0;
  color: #8b8b93;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.status,
.prototype-badge {
  padding: 8px 12px;
  border: 1px solid #e4e4e7;
  border-radius: 999px;
  background: #ffffff;
  color: #52525b;
  font-size: 11px;
  font-weight: 600;
}

.panel {
  padding: 26px;

  border: 1px solid var(--border);
  border-radius: 18px;

  background: var(--surface);

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.panel:hover {
  border-color: var(--primary-border);

  box-shadow: var(--shadow-soft);
}

.performance-panel,
.table-panel {
  margin-bottom: 18px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 22px;
}

.panel-header h2 {
  margin: 6px 0 0;
  font-size: 20px;
  letter-spacing: -0.025em;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 18px;
}

.strategy-card {
  padding: 22px;

  border: 1px solid #e4e4e7;
  border-radius: 16px;

  background: #ffffff;
  color: #18181b;

  text-align: left;
  font: inherit;
  cursor: pointer;

  opacity: 0;

  animation: card-enter 350ms ease forwards;

  transition:
    border-color 160ms ease,
    box-shadow 160ms ease,
    transform 160ms ease;
}

.strategy-card:hover {
  border-color: #cfcfd4;

  transform: translateY(-2px);

  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.035);
}

.strategy-card.selected {
  border-color: var(--primary);

  background: linear-gradient(180deg, #ffffff, #f8faff);

  box-shadow: 0 8px 24px rgba(54, 84, 255, 0.08);
}

.strategy-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.strategy-heading > div {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.strategy-heading strong {
  font-size: 18px;
}

.strategy-label {
  color: #a1a1aa;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.selected-indicator {
  padding: 5px 8px;

  border-radius: 999px;

  background: var(--primary-soft);

  color: var(--primary);

  font-size: 9px;
  font-weight: 700;
}

.return-value {
  margin: 28px 0;

  font-size: 29px;
  font-weight: 700;

  letter-spacing: -0.04em;
}

.mini-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.mini-metrics div {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.mini-metrics span {
  color: #8b8b93;
  font-size: 10px;
}

.mini-metrics strong {
  font-size: 12px;
}

@keyframes card-enter {
  from {
    opacity: 0;
    transform: translateY(8px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.table {
  overflow: hidden;
}

.table-row {
  display: grid;
  grid-template-columns: 1.3fr repeat(5, 1fr);
  gap: 18px;

  align-items: center;

  width: 100%;
  padding: 17px 12px;

  border: 0;
  border-top: 1px solid #eeeef0;

  background: transparent;
  color: #18181b;

  font: inherit;
  font-size: 13px;
  text-align: left;
}

.table-head {
  border-top: 0;

  color: #8b8b93;

  font-size: 10px;
  font-weight: 600;
}

.data-row {
  cursor: pointer;

  transition:
    background 150ms ease,
    transform 150ms ease;
}

.data-row:hover {
  background: #fafafa;
}

.data-row.selected {
  background: #f5f5f6;
}

.analysis-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 18px;
}

.drawdown-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.drawdown-row {
  display: grid;
  grid-template-columns: 70px 1fr 55px;
  gap: 12px;

  align-items: center;

  font-size: 12px;
}

.drawdown-track {
  height: 8px;

  overflow: hidden;

  border-radius: 999px;

  background: #f1f1f3;
}

.drawdown-fill {
  height: 100%;

  border-radius: inherit;

  background: linear-gradient(90deg, #697386, #9ba3b3);

  transform-origin: left;

  animation: bar-enter 650ms cubic-bezier(0.4, 0, 0.2, 1) both;
}

@keyframes bar-enter {
  from {
    transform: scaleX(0);
  }

  to {
    transform: scaleX(1);
  }
}

.hint {
  margin: 24px 0 0;

  color: #8b8b93;

  font-size: 11px;
}

.selected-panel {
  padding: 27px;

  border: 1px solid var(--primary-border);
  border-radius: 18px;

  background: linear-gradient(135deg, #eef2ff, #f8faff);

  color: var(--text);

  box-shadow: 0 12px 32px rgba(54, 84, 255, 0.08);
}

.dark-label {
  color: var(--primary);
}

.selected-panel h2 {
  margin: 7px 0 10px;

  font-size: 28px;
}

.selected-description {
  margin: 0;

  color: var(--text-secondary);

  font-size: 12px;
  line-height: 1.6;
}

.selected-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);

  gap: 12px;

  margin-top: 28px;
}

.selected-stats div {
  display: flex;

  flex-direction: column;

  gap: 5px;

  padding: 13px;

  border: 1px solid rgba(54, 84, 255, 0.11);
  border-radius: 10px;

  background: rgba(255, 255, 255, 0.72);
}

.selected-stats span {
  color: var(--text-muted);

  font-size: 10px;
}

.selected-stats strong {
  font-size: 14px;
}

.detail-enter-active,
.detail-leave-active {
  transition:
    opacity 180ms ease,
    transform 180ms ease;
}

.detail-enter-from {
  opacity: 0;
  transform: translateY(6px);
}

.detail-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

/* RESPONSIVE */

@media (max-width: 1000px) {
  .backtest-page {
    padding: 36px 28px 60px;
  }

  .metric-grid,
  .analysis-grid {
    grid-template-columns: 1fr;
  }

  .table {
    overflow-x: auto;
  }

  .table-row {
    min-width: 720px;
  }
}
</style>
