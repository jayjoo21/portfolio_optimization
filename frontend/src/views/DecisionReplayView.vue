<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'

import DecisionReplayChart from '@/components/DecisionReplayChart.vue'
import AllocationTransitionChart from '@/components/AllocationTransitionChart.vue'

const replayEvents = [
  {
    time: '13:00',

    status: 'Stable',
    action: 'Maintain Allocation',

    regime: 'Risk-On',

    riskOff: 24,
    volatility: 31,
    correlation: 38,
    momentum: 78,

    expectedRisk: 10.4,
    expectedSharpe: 1.17,
    transactionCost: 0.02,

    note: '시장 상태 변화가 크지 않아 EIIE 목표 비중을 유지합니다.',

    allocation: {
      samsung: 24,
      hynix: 21,
      naver: 15,
      defensive: 8,
      cash: 7,
      other: 25,
    },
  },

  {
    time: '13:10',

    status: 'Transition',
    action: 'Risk Signal Rising',

    regime: 'Transition',

    riskOff: 58,
    volatility: 56,
    correlation: 61,
    momentum: 49,

    expectedRisk: 10.8,
    expectedSharpe: 1.12,
    transactionCost: 0.04,

    note: 'Risk-Off 신호가 상승하고 있지만 목표 비중 변화는 아직 제한적입니다.',

    allocation: {
      samsung: 24,
      hynix: 21,
      naver: 15,
      defensive: 8,
      cash: 7,
      other: 25,
    },
  },

  {
    time: '13:15',

    status: 'Hold',
    action: 'No Rebalance',

    regime: 'Transition',

    riskOff: 64,
    volatility: 67,
    correlation: 69,
    momentum: 40,

    expectedRisk: 10.9,
    expectedSharpe: 1.1,
    transactionCost: 0.06,

    note: '시장 상태는 변화하고 있지만 현재 출력 비중과의 차이가 충분하지 않아 리밸런싱하지 않습니다.',

    allocation: {
      samsung: 24,
      hynix: 21,
      naver: 15,
      defensive: 8,
      cash: 7,
      other: 25,
    },
  },

  {
    time: '13:20',

    status: 'Rebalance',
    action: 'Portfolio Rebalanced',

    regime: 'Risk-Off',

    riskOff: 71,
    volatility: 78,
    correlation: 74,
    momentum: 31,

    expectedRisk: 8.1,
    expectedSharpe: 1.31,
    transactionCost: 0.08,

    note: 'EIIE가 새로운 목표 비중을 출력했고, 비중 변화가 리밸런싱 기준을 충족해 포트폴리오를 조정합니다.',

    allocation: {
      samsung: 18,
      hynix: 16,
      naver: 12,
      defensive: 26,
      cash: 15,
      other: 13,
    },
  },
]

const selectedIndex = ref(3)

const firstEvent = replayEvents[0]!

const currentEvent = computed(() => {
  return replayEvents[selectedIndex.value] ?? firstEvent
})

const previousEvent = computed(() => {
  if (selectedIndex.value === 0) {
    return firstEvent
  }

  return replayEvents[selectedIndex.value - 1] ?? firstEvent
})
</script>

<template>
  <main class="replay-page">
    <!-- PAGE HEADER -->
    <header class="page-header">
      <div>
        <p class="eyebrow">PORTFOLIO OPTIMIZATION</p>

        <h1>Decision Replay</h1>

        <p class="subtitle">
          시장 상태 변화와 EIIE 포트폴리오 의사결정 과정을 시간 순서대로 재현합니다.
        </p>
      </div>

      <div class="header-status">Historical Replay</div>
    </header>

    <!-- 1. MARKET TIMELINE -->
    <section class="panel market-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">PORTFOLIO / MARKET</p>

          <h2>Intraday Decision Timeline</h2>
        </div>

        <div class="current-time">
          {{ currentEvent.time }}
        </div>
      </div>

      <div class="chart-wrap">
        <DecisionReplayChart :selected-index="selectedIndex" @select="selectedIndex = $event" />
      </div>

      <div class="time-tabs">
        <button
          v-for="(event, index) in replayEvents"
          :key="event.time"
          :class="{ active: selectedIndex === index }"
          @click="selectedIndex = index"
        >
          {{ event.time }}
        </button>
      </div>
    </section>

    <!-- 2. TRACE + CONTEXT -->
    <section class="two-column">
      <!-- Decision Trace -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">DECISION TRACE</p>

            <h2>System Events</h2>
          </div>
        </div>

        <div class="trace-list">
          <button
            v-for="(event, index) in replayEvents"
            :key="event.time"
            class="trace-item"
            :class="{ selected: selectedIndex === index }"
            @click="selectedIndex = index"
          >
            <span class="trace-time">
              {{ event.time }}
            </span>

            <span class="trace-line">
              <span class="trace-dot"></span>
            </span>

            <span class="trace-content">
              <strong>
                {{ event.status }}
              </strong>

              <small>
                {{ event.action }}
              </small>
            </span>
          </button>
        </div>
      </div>

      <!-- Decision Context -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">DECISION CONTEXT</p>

            <h2>
              {{ currentEvent.regime }}
            </h2>
          </div>

          <span class="regime-badge"> Risk-Off Signal {{ currentEvent.riskOff }}% </span>
        </div>

        <div class="context-row">
          <div class="context-label">
            <span>Risk-Off Signal</span>

            <strong> {{ currentEvent.riskOff }}% </strong>
          </div>

          <div class="progress">
            <span
              :style="{
                width: `${currentEvent.riskOff}%`,
              }"
            ></span>
          </div>
        </div>

        <div class="context-row">
          <div class="context-label">
            <span>Volatility</span>

            <strong>
              {{ currentEvent.volatility }}
            </strong>
          </div>

          <div class="progress">
            <span
              :style="{
                width: `${currentEvent.volatility}%`,
              }"
            ></span>
          </div>
        </div>

        <div class="context-row">
          <div class="context-label">
            <span>Correlation</span>

            <strong>
              {{ currentEvent.correlation }}
            </strong>
          </div>

          <div class="progress">
            <span
              :style="{
                width: `${currentEvent.correlation}%`,
              }"
            ></span>
          </div>
        </div>

        <div class="context-row">
          <div class="context-label">
            <span>Momentum</span>

            <strong>
              {{ currentEvent.momentum }}
            </strong>
          </div>

          <div class="progress">
            <span
              :style="{
                width: `${currentEvent.momentum}%`,
              }"
            ></span>
          </div>
        </div>

        <div class="decision-note">
          {{ currentEvent.note }}
        </div>
      </div>
    </section>

    <!-- 3. MODEL OUTPUT + REBALANCING -->
    <section class="model-output-grid">
      <div class="panel model-output">
        <p class="panel-label">MODEL OUTPUT</p>

        <h2>EIIE Portfolio Weights</h2>

        <p class="model-description">
          현재 시장 입력에 대해 Deep RL · EIIE 모델이 출력한 목표 포트폴리오 비중입니다.
        </p>

        <div class="model-meta">
          <div>
            <span>Model</span>

            <strong> Deep RL · EIIE </strong>
          </div>

          <div>
            <span>Decision Time</span>

            <strong>
              {{ currentEvent.time }}
            </strong>
          </div>

          <div>
            <span>Market Regime</span>

            <strong>
              {{ currentEvent.regime }}
            </strong>
          </div>
        </div>
      </div>

      <div class="panel rebalance-output">
        <p class="panel-label">REBALANCING</p>

        <h2>Portfolio Adjustment</h2>

        <div class="adjustment">
          <span>Samsung</span>

          <strong>
            {{ previousEvent.allocation.samsung }}% → {{ currentEvent.allocation.samsung }}%
          </strong>
        </div>

        <div class="adjustment">
          <span>SK hynix</span>

          <strong>
            {{ previousEvent.allocation.hynix }}% → {{ currentEvent.allocation.hynix }}%
          </strong>
        </div>

        <div class="adjustment">
          <span>NAVER</span>

          <strong>
            {{ previousEvent.allocation.naver }}% → {{ currentEvent.allocation.naver }}%
          </strong>
        </div>

        <div class="adjustment">
          <span>Defensive</span>

          <strong>
            {{ previousEvent.allocation.defensive }}% → {{ currentEvent.allocation.defensive }}%
          </strong>
        </div>

        <div class="adjustment">
          <span>Cash</span>

          <strong>
            {{ previousEvent.allocation.cash }}% → {{ currentEvent.allocation.cash }}%
          </strong>
        </div>
      </div>
    </section>

    <!-- 4. CURRENT DECISION -->
    <Transition name="decision" mode="out-in">
      <section :key="currentEvent.time" class="decision-panel">
        <div>
          <p class="panel-label decision-label">CURRENT DECISION</p>

          <h2>
            {{ currentEvent.action }}
          </h2>

          <p>
            {{ currentEvent.note }}
          </p>
        </div>

        <div class="decision-metrics">
          <div>
            <span>Expected Risk</span>

            <strong> {{ currentEvent.expectedRisk }}% </strong>
          </div>

          <div>
            <span>Expected Sharpe</span>

            <strong>
              {{ currentEvent.expectedSharpe }}
            </strong>
          </div>

          <div>
            <span>Transaction Cost</span>

            <strong> {{ currentEvent.transactionCost }}% </strong>
          </div>
        </div>
      </section>
    </Transition>

    <!-- 5. ALLOCATION BEFORE / AFTER -->
    <section class="panel allocation-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">PORTFOLIO OUTPUT</p>

          <h2>Allocation Change</h2>
        </div>

        <div class="allocation-time">
          {{ previousEvent.time }}
          →
          {{ currentEvent.time }}
        </div>
      </div>

      <AllocationTransitionChart
        :before="previousEvent.allocation"
        :after="currentEvent.allocation"
      />
    </section>

    <section class="next-step">
      <div>
        <p class="panel-label">NEXT STEP</p>

        <h2>How was this rebalance executed?</h2>

        <p>
          목표 비중이 실제 주문으로 변환되고, KRX와 NXT 중 어떤 시장에서 주문이 실행되는지
          확인합니다.
        </p>
      </div>
      <RouterLink to="/execution" class="execution-link">
        View Execution

        <span> → </span>
      </RouterLink>
    </section>
  </main>
</template>

<style scoped>
.replay-page {
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
  max-width: 700px;
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

.header-status,
.regime-badge {
  padding: 8px 12px;
  border: 1px solid #e4e4e7;
  border-radius: 999px;
  background: #ffffff;
  color: #52525b;
  font-size: 12px;
  font-weight: 600;
}

.panel {
  margin-bottom: 18px;
  padding: 26px;

  border: 1px solid #e4e4e7;
  border-radius: 18px;

  background: #ffffff;

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.panel:hover {
  border-color: #d4d4d8;

  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.035);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
}

.panel-header h2,
.model-output h2,
.rebalance-output h2 {
  margin: 6px 0 0;
  font-size: 20px;
  letter-spacing: -0.025em;
}

.market-panel {
  overflow: hidden;
}

.current-time {
  font-size: 18px;
  font-weight: 700;
}

.chart-wrap {
  position: relative;
}

.time-tabs {
  display: flex;
  gap: 8px;
  margin-top: 20px;
}

.time-tabs button {
  padding: 8px 13px;

  border: 1px solid #e4e4e7;
  border-radius: 8px;

  background: #ffffff;
  color: #71717a;

  font: inherit;
  cursor: pointer;

  transition:
    background 150ms ease,
    color 150ms ease,
    border-color 150ms ease,
    transform 150ms ease;
}

.time-tabs button:hover {
  transform: translateY(-1px);
}

.time-tabs button:active {
  transform: translateY(0);
}

.time-tabs button.active {
  border-color: #18181b;
  background: #18181b;
  color: #ffffff;
}

.two-column {
  display: grid;
  grid-template-columns: 0.9fr 1.1fr;
  gap: 18px;
}

.trace-list {
  display: flex;
  flex-direction: column;
}

.trace-item {
  display: grid;
  grid-template-columns: 54px 30px 1fr;
  align-items: stretch;

  min-height: 62px;
  padding: 0;

  border: 0;

  background: transparent;
  color: inherit;

  text-align: left;
  cursor: pointer;

  transition:
    background 160ms ease,
    transform 160ms ease;
}

.trace-item:hover {
  transform: translateX(3px);
}

.trace-time {
  padding-top: 6px;

  color: #71717a;

  font-size: 12px;
}

.trace-line {
  position: relative;

  display: flex;
  justify-content: center;
}

.trace-line::after {
  position: absolute;

  top: 15px;
  bottom: -8px;

  width: 1px;

  background: #e4e4e7;

  content: '';
}

.trace-item:last-child .trace-line::after {
  display: none;
}

.trace-dot {
  z-index: 1;

  width: 9px;
  height: 9px;

  margin-top: 8px;

  border: 2px solid #a1a1aa;
  border-radius: 50%;

  background: #ffffff;

  transition:
    border-color 160ms ease,
    background 160ms ease,
    transform 160ms ease;
}

.trace-item.selected .trace-dot {
  border-color: #18181b;
  background: #18181b;

  transform: scale(1.18);
}

.trace-content {
  display: flex;
  flex-direction: column;

  padding: 2px 0 20px;

  transition: transform 160ms ease;
}

.trace-item.selected .trace-content {
  transform: translateX(3px);
}

.trace-content strong {
  font-size: 14px;
}

.trace-content small {
  margin-top: 4px;

  color: #71717a;

  font-size: 12px;
}

.context-row {
  margin-bottom: 18px;
}

.context-label {
  display: flex;
  justify-content: space-between;

  margin-bottom: 7px;

  font-size: 13px;
}

.progress {
  height: 7px;

  overflow: hidden;

  border-radius: 999px;

  background: #f1f1f3;
}

.progress span {
  display: block;

  height: 100%;

  border-radius: inherit;

  background: #313138;

  transition: width 350ms cubic-bezier(0.4, 0, 0.2, 1);
}

.decision-note {
  margin-top: 24px;
  padding: 16px;

  border-radius: 12px;

  background: #f7f7f8;
  color: #52525b;

  font-size: 13px;
  line-height: 1.6;
}

.model-output-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 18px;
}

.model-description {
  max-width: 650px;

  margin-bottom: 0;

  color: #71717a;

  font-size: 13px;
  line-height: 1.7;
}

.model-meta {
  display: grid;
  grid-template-columns: repeat(3, 1fr);

  gap: 12px;

  margin-top: 26px;
}

.model-meta div {
  display: flex;
  flex-direction: column;
  gap: 6px;

  padding: 14px;

  border-radius: 10px;

  background: #f7f7f8;
}

.model-meta span,
.adjustment span {
  color: #71717a;

  font-size: 11px;
}

.model-meta strong {
  font-size: 13px;
}

.adjustment {
  display: flex;
  justify-content: space-between;
  align-items: center;

  padding: 16px 0;

  border-bottom: 1px solid #eeeef0;
}

.adjustment:last-child {
  border-bottom: 0;
}

.adjustment strong {
  font-size: 13px;
}

.decision-panel {
  display: flex;
  justify-content: space-between;
  align-items: center;

  gap: 40px;

  margin-bottom: 18px;
  padding: 28px;

  border-radius: 18px;

  background: #18181b;
  color: #ffffff;

  box-shadow: 0 12px 35px rgba(0, 0, 0, 0.08);
}

.decision-label {
  color: #a1a1aa;
}

.decision-panel h2 {
  margin: 7px 0 9px;

  font-size: 25px;
}

.decision-panel p:not(.panel-label) {
  max-width: 650px;

  margin: 0;

  color: #c8c8ce;

  font-size: 13px;
  line-height: 1.6;
}

.decision-metrics {
  display: flex;
  gap: 28px;
  align-items: center;
}

.decision-metrics div {
  display: flex;
  min-width: 100px;
  flex-direction: column;

  gap: 5px;
}

.decision-metrics span {
  color: #a1a1aa;

  font-size: 11px;
}

.decision-metrics strong {
  font-size: 17px;
}

.decision-enter-active,
.decision-leave-active {
  transition:
    opacity 180ms ease,
    transform 180ms ease;
}

.decision-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

.decision-leave-to {
  opacity: 0;
  transform: translateY(-5px);
}

.allocation-time {
  color: #71717a;

  font-size: 12px;
  font-weight: 600;
}

.next-step {
  display: flex;
  justify-content: space-between;
  align-items: center;

  gap: 30px;

  margin-top: 18px;
  padding: 28px;

  border: 1px solid #e4e4e7;
  border-radius: 18px;

  background: #ffffff;

  flex-direction: column;
  align-items: flex-start;
}

.next-step h2 {
  margin: 6px 0 8px;

  font-size: 20px;
  letter-spacing: -0.025em;
}

.next-step p:not(.panel-label) {
  max-width: 650px;

  margin: 0;

  color: #71717a;

  font-size: 12px;
  line-height: 1.6;
}

.execution-link {
  display: flex;
  align-items: center;
  gap: 28px;

  flex-shrink: 0;

  padding: 13px 16px;

  border-radius: 10px;

  background: #18181b;
  color: #ffffff;

  font-size: 12px;
  font-weight: 650;

  transition:
    background 150ms ease,
    transform 150ms ease;
}

.execution-link:hover {
  background: #29292e;

  transform: translateY(-1px);
}

.execution-link span {
  transition: transform 150ms ease;
}

.execution-link:hover span {
  transform: translateX(3px);
}

@media (max-width: 1000px) {
  .replay-page {
    padding: 36px 28px 60px;
  }

  .two-column,
  .model-output-grid {
    grid-template-columns: 1fr;
  }

  .decision-panel {
    flex-direction: column;
    align-items: flex-start;
  }

  .decision-metrics {
    width: 100%;
    justify-content: space-between;
  }

  .model-meta {
    grid-template-columns: 1fr;
  }
}
</style>
