<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

import { RouterLink } from 'vue-router'

import DecisionReplayChart from '../components/DecisionReplayChart.vue'
import AllocationTransitionChart from '../components/AllocationTransitionChart.vue'

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

/* -------------------------
   AUTO REPLAY
------------------------- */

const isPlaying = ref(false)

const REPLAY_DELAY = 1400

let replayTimer: ReturnType<typeof setInterval> | null = null

const stopReplay = () => {
  if (replayTimer) {
    clearInterval(replayTimer)
  }

  replayTimer = null
  isPlaying.value = false
}

const toggleReplay = () => {
  /*
    이미 실행 중이면 Pause
  */
  if (isPlaying.value) {
    stopReplay()
    return
  }

  /*
    마지막 시점에서 다시 Play를 누르면
    처음부터 시작
  */
  if (selectedIndex.value >= replayEvents.length - 1) {
    selectedIndex.value = 0
  }

  isPlaying.value = true

  replayTimer = setInterval(() => {
    /*
      마지막 이벤트까지 도착했다면 종료
    */
    if (selectedIndex.value >= replayEvents.length - 1) {
      stopReplay()
      return
    }

    selectedIndex.value += 1
  }, REPLAY_DELAY)
}

const selectEvent = (index: number) => {
  /*
    사용자가 직접 클릭하면
    자동 재생 중지
  */
  stopReplay()

  selectedIndex.value = index
}

onBeforeUnmount(() => {
  stopReplay()
})
</script>

<template>
  <main class="replay-page">
    <!-- HEADER -->
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

    <!-- MARKET TIMELINE -->
    <section class="panel market-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">PORTFOLIO / MARKET</p>

          <h2>Intraday Decision Timeline</h2>
        </div>

        <div class="timeline-controls">
          <div class="time-status">
            <span> DECISION TIME </span>

            <strong>
              {{ currentEvent.time }}
            </strong>
          </div>

          <button
            class="replay-control"
            :class="{
              playing: isPlaying,
            }"
            @click="toggleReplay"
          >
            {{ isPlaying ? 'Pause' : 'Play replay' }}
          </button>
        </div>
      </div>

      <div class="chart-wrap">
        <DecisionReplayChart :selected-index="selectedIndex" @select="selectEvent" />
      </div>

      <div class="timeline-footer">
        <div class="time-tabs">
          <button
            v-for="(event, index) in replayEvents"
            :key="event.time"
            :class="{
              active: selectedIndex === index,
            }"
            @click="selectEvent(index)"
          >
            {{ event.time }}
          </button>
        </div>

        <div class="replay-counter">
          {{ selectedIndex + 1 }}
          /
          {{ replayEvents.length }}
        </div>
      </div>
    </section>

    <!-- TRACE + CONTEXT -->
    <section class="two-column">
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
            :class="{
              selected: selectedIndex === index,
            }"
            @click="selectEvent(index)"
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

      <!-- CONTEXT -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">DECISION CONTEXT</p>

            <h2>
              {{ currentEvent.regime }}
            </h2>
          </div>

          <span class="regime-badge">
            Risk-Off Signal
            {{ currentEvent.riskOff }}%
          </span>
        </div>

        <div class="context-row">
          <div class="context-label">
            <span> Risk-Off Signal </span>

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
            <span> Volatility </span>

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
            <span> Correlation </span>

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
            <span> Momentum </span>

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

        <Transition name="context" mode="out-in">
          <div :key="currentEvent.time" class="decision-note">
            {{ currentEvent.note }}
          </div>
        </Transition>
      </div>
    </section>

    <!-- MODEL OUTPUT -->
    <section class="model-output-grid">
      <div class="panel model-output">
        <p class="panel-label">MODEL OUTPUT</p>

        <h2>EIIE Portfolio Weights</h2>

        <p class="model-description">
          현재 시장 입력에 대해 Deep RL · EIIE 모델이 출력한 목표 포트폴리오 비중입니다.
        </p>

        <div class="model-meta">
          <div>
            <span> Model </span>

            <strong> Deep RL · EIIE </strong>
          </div>

          <div>
            <span> Decision Time </span>

            <strong>
              {{ currentEvent.time }}
            </strong>
          </div>

          <div>
            <span> Market Regime </span>

            <strong>
              {{ currentEvent.regime }}
            </strong>
          </div>
        </div>
      </div>

      <!-- REBALANCING -->
      <div class="panel rebalance-output">
        <p class="panel-label">REBALANCING</p>

        <h2>Portfolio Adjustment</h2>

        <div class="adjustment">
          <span> Samsung </span>

          <strong>
            {{ previousEvent.allocation.samsung }}% → {{ currentEvent.allocation.samsung }}%
          </strong>
        </div>

        <div class="adjustment">
          <span> SK hynix </span>

          <strong>
            {{ previousEvent.allocation.hynix }}% → {{ currentEvent.allocation.hynix }}%
          </strong>
        </div>

        <div class="adjustment">
          <span> NAVER </span>

          <strong>
            {{ previousEvent.allocation.naver }}% → {{ currentEvent.allocation.naver }}%
          </strong>
        </div>

        <div class="adjustment">
          <span> Defensive </span>

          <strong>
            {{ previousEvent.allocation.defensive }}% → {{ currentEvent.allocation.defensive }}%
          </strong>
        </div>

        <div class="adjustment">
          <span> Cash </span>

          <strong>
            {{ previousEvent.allocation.cash }}% → {{ currentEvent.allocation.cash }}%
          </strong>
        </div>
      </div>
    </section>

    <!-- CURRENT DECISION -->
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
            <span> Expected Risk </span>

            <strong> {{ currentEvent.expectedRisk }}% </strong>
          </div>

          <div>
            <span> Expected Sharpe </span>

            <strong>
              {{ currentEvent.expectedSharpe }}
            </strong>
          </div>

          <div>
            <span> Transaction Cost </span>

            <strong> {{ currentEvent.transactionCost }}% </strong>
          </div>
        </div>
      </section>
    </Transition>

    <!-- ALLOCATION -->
    <section class="panel">
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

    <!-- NEXT STEP -->
    <section class="next-step">
      <div>
        <p class="panel-label">NEXT STEP</p>

        <h2>How was this rebalance executed?</h2>

        <p>목표 비중이 실제 주문으로 변환되고 KRX와 NXT 중 어떤 시장에서 실행되는지 확인합니다.</p>
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

/* -------------------------
   HEADER
------------------------- */

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

  color: var(--text-secondary);

  font-size: 14px;

  line-height: 1.6;
}

.eyebrow,
.panel-label {
  margin: 0;

  color: var(--text-muted);

  font-size: 10px;
  font-weight: 700;

  letter-spacing: 0.11em;
}

.header-status,
.regime-badge {
  padding: 7px 11px;

  border: 1px solid var(--border);

  border-radius: 999px;

  background: var(--surface);

  color: var(--text-secondary);

  font-size: 11px;
  font-weight: 600;
}

/* -------------------------
   PANEL
------------------------- */

.panel {
  margin-bottom: 18px;

  padding: 26px;

  border: 1px solid var(--border);

  border-radius: 14px;

  background: var(--surface);

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.panel:hover {
  border-color: var(--border-strong);
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

/* -------------------------
   TIMELINE CONTROL
------------------------- */

.timeline-controls {
  display: flex;

  align-items: center;

  gap: 14px;
}

.time-status {
  display: flex;

  flex-direction: column;

  align-items: flex-end;

  gap: 2px;
}

.time-status span {
  color: var(--text-muted);

  font-size: 8px;
  font-weight: 700;

  letter-spacing: 0.09em;
}

.time-status strong {
  font-size: 16px;

  letter-spacing: -0.02em;
}

.replay-control {
  min-width: 88px;

  padding: 9px 12px;

  border: 1px solid var(--primary);

  border-radius: 8px;

  background: var(--primary);

  color: #ffffff;

  font-size: 11px;
  font-weight: 650;

  cursor: pointer;

  transition:
    background 150ms ease,
    transform 150ms ease;
}

.replay-control:hover {
  background: var(--primary-hover);

  transform: translateY(-1px);
}

.replay-control.playing {
  border-color: var(--primary-border);

  background: var(--primary-soft);

  color: var(--primary);
}

.chart-wrap {
  position: relative;
}

/* -------------------------
   TIME TABS
------------------------- */

.timeline-footer {
  display: flex;

  justify-content: space-between;

  align-items: center;

  margin-top: 18px;
}

.time-tabs {
  display: flex;

  gap: 7px;
}

.time-tabs button {
  padding: 7px 11px;

  border: 1px solid var(--border);

  border-radius: 7px;

  background: #ffffff;

  color: var(--text-muted);

  font-size: 11px;

  cursor: pointer;

  transition:
    background 150ms ease,
    color 150ms ease,
    border-color 150ms ease,
    transform 150ms ease;
}

.time-tabs button:hover {
  border-color: var(--primary-border);

  color: var(--primary);
}

.time-tabs button.active {
  border-color: var(--primary);

  background: var(--primary);

  color: #ffffff;

  box-shadow: 0 4px 12px rgba(54, 84, 255, 0.16);
}

.replay-counter {
  color: var(--text-muted);

  font-size: 10px;
  font-weight: 600;
}

/* -------------------------
   TRACE + CONTEXT
------------------------- */

.two-column {
  display: grid;

  grid-template-columns:
    0.9fr
    1.1fr;

  gap: 18px;
}

.trace-list {
  display: flex;

  flex-direction: column;
}

.trace-item {
  display: grid;

  grid-template-columns:
    54px
    30px
    1fr;

  align-items: stretch;

  min-height: 62px;

  padding: 0;

  border: 0;

  background: transparent;

  color: inherit;

  text-align: left;

  cursor: pointer;
}

.trace-time {
  padding-top: 6px;

  color: var(--text-muted);

  font-size: 11px;
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

  background: var(--border);

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

  border: 2px solid #aab1bd;

  border-radius: 50%;

  background: #ffffff;

  transition:
    border-color 160ms ease,
    background 160ms ease,
    transform 160ms ease;
}

.trace-item.selected .trace-dot {
  border-color: var(--primary);

  background: var(--primary);

  transform: scale(1.18);

  box-shadow: 0 0 0 4px rgba(54, 84, 255, 0.08);
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
  font-size: 13px;
}

.trace-content small {
  margin-top: 4px;

  color: var(--text-muted);

  font-size: 11px;
}

/* CONTEXT */

.context-row {
  margin-bottom: 18px;
}

.context-label {
  display: flex;

  justify-content: space-between;

  margin-bottom: 7px;

  font-size: 12px;
}

.progress {
  height: 6px;

  overflow: hidden;

  border-radius: 999px;

  background: #edf0f5;
}

.progress span {
  display: block;

  height: 100%;

  border-radius: inherit;

  background: linear-gradient(90deg, #3654ff, #7285ff);

  transition: width 500ms cubic-bezier(0.4, 0, 0.2, 1);
}

.decision-note {
  margin-top: 24px;

  padding: 15px;

  border-radius: 9px;

  background: var(--surface-soft);

  color: var(--text-secondary);

  font-size: 12px;

  line-height: 1.6;
}

/* CONTEXT TRANSITION */

.context-enter-active,
.context-leave-active {
  transition:
    opacity 180ms ease,
    transform 180ms ease;
}

.context-enter-from {
  opacity: 0;

  transform: translateY(5px);
}

.context-leave-to {
  opacity: 0;

  transform: translateY(-3px);
}

/* -------------------------
   MODEL OUTPUT
------------------------- */

.model-output-grid {
  display: grid;

  grid-template-columns:
    1.4fr
    1fr;

  gap: 18px;
}

.model-description {
  max-width: 650px;

  margin-bottom: 0;

  color: var(--text-secondary);

  font-size: 12px;

  line-height: 1.7;
}

.model-meta {
  display: grid;

  grid-template-columns: repeat(3, 1fr);

  gap: 10px;

  margin-top: 25px;
}

.model-meta div {
  display: flex;

  flex-direction: column;

  gap: 6px;

  padding: 13px;

  border-radius: 8px;

  background: var(--surface-soft);
}

.model-meta span,
.adjustment span {
  color: var(--text-muted);

  font-size: 10px;
}

.model-meta strong {
  font-size: 12px;
}

.adjustment {
  display: flex;

  justify-content: space-between;

  align-items: center;

  padding: 15px 0;

  border-bottom: 1px solid var(--border);
}

.adjustment:last-child {
  border-bottom: 0;
}

.adjustment strong {
  font-size: 12px;
}

/* -------------------------
   CURRENT DECISION
------------------------- */

.decision-panel {
  display: flex;

  justify-content: space-between;

  align-items: center;

  gap: 40px;

  margin-bottom: 18px;

  padding: 28px;

  border: 1px solid var(--primary-border);

  border-radius: 14px;

  background: linear-gradient(135deg, #eef2ff 0%, #f8faff 100%);

  color: var(--text);

  box-shadow: 0 10px 28px rgba(54, 84, 255, 0.07);
}

.decision-label {
  color: var(--primary);
}

.decision-panel h2 {
  margin: 7px 0 9px;

  font-size: 24px;
}

.decision-panel p:not(.panel-label) {
  max-width: 650px;

  margin: 0;

  color: var(--text-secondary);

  font-size: 12px;

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
  color: var(--text-muted);

  font-size: 10px;
}

.decision-metrics strong {
  font-size: 16px;
}

/* DECISION ANIMATION */

.decision-enter-active,
.decision-leave-active {
  transition:
    opacity 220ms ease,
    transform 220ms ease;
}

.decision-enter-from {
  opacity: 0;

  transform: translateY(7px);
}

.decision-leave-to {
  opacity: 0;

  transform: translateY(-4px);
}

/* -------------------------
   ALLOCATION
------------------------- */

.allocation-time {
  color: var(--text-muted);

  font-size: 11px;
  font-weight: 600;
}

/* -------------------------
   NEXT STEP
------------------------- */

.next-step {
  display: flex;

  justify-content: space-between;

  align-items: center;

  gap: 30px;

  margin-top: 18px;

  padding: 26px;

  border: 1px solid var(--border);

  border-radius: 14px;

  background: var(--surface);
}

.next-step h2 {
  margin: 6px 0 8px;

  font-size: 20px;

  letter-spacing: -0.025em;
}

.next-step p:not(.panel-label) {
  max-width: 650px;

  margin: 0;

  color: var(--text-secondary);

  font-size: 12px;

  line-height: 1.6;
}

.execution-link {
  display: flex;

  align-items: center;

  gap: 28px;

  flex-shrink: 0;

  padding: 12px 15px;

  border-radius: 8px;

  background: var(--primary);

  color: #ffffff;

  font-size: 11px;

  font-weight: 650;

  box-shadow: 0 6px 18px rgba(54, 84, 255, 0.17);

  transition:
    background 150ms ease,
    transform 150ms ease;
}

.execution-link:hover {
  background: var(--primary-hover);

  transform: translateY(-1px);
}

.execution-link span {
  transition: transform 150ms ease;
}

.execution-link:hover span {
  transform: translateX(3px);
}

/* -------------------------
   RESPONSIVE
------------------------- */

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

  .next-step {
    flex-direction: column;

    align-items: flex-start;
  }
}
</style>
