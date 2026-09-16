<script setup lang="ts">
const executionSteps = [
  {
    label: 'Target Generated',
    time: '13:20:00',
    description: 'EIIE 목표 비중 생성',
  },
  {
    label: 'Trade Required',
    time: '13:20:01',
    description: 'Samsung 비중 축소 필요',
  },
  {
    label: 'Venue Compared',
    time: '13:20:02',
    description: 'KRX / NXT 주문 조건 비교',
  },
  {
    label: 'Order Submitted',
    time: '13:20:04',
    description: 'NXT 지정가 주문 제출',
  },
  {
    label: 'Filled',
    time: '13:20:06',
    description: '주문 체결 완료',
  },
]

const venues = [
  {
    name: 'KRX',
    bestBid: 84100,
    bestAsk: 84200,
    spread: 100,
    estimatedCost: 2480,
    selected: false,
  },
  {
    name: 'NXT',
    bestBid: 84150,
    bestAsk: 84200,
    spread: 50,
    estimatedCost: 1920,
    selected: true,
  },
]
</script>

<template>
  <main class="execution-page">
    <!-- HEADER -->
    <header class="page-header">
      <div>
        <p class="eyebrow">ORDER EXECUTION</p>

        <h1>Execution</h1>

        <p class="subtitle">
          EIIE가 생성한 목표 포트폴리오를 실제 주문으로 변환하는 과정을 시뮬레이션합니다.
        </p>
      </div>

      <span class="status"> Paper Simulation </span>
    </header>

    <section class="source-context">
      <div>
        <span> SOURCE DECISION </span>

        <strong> 13:20 Rebalance </strong>
      </div>

      <div class="flow-arrow">→</div>

      <div>
        <span> MODEL OUTPUT </span>

        <strong> Samsung 24% → 18% </strong>
      </div>

      <div class="flow-arrow">→</div>

      <div>
        <span> REQUIRED ACTION </span>

        <strong> SELL 63 shares </strong>
      </div>
    </section>

    <!-- FLOW -->
    <section class="execution-flow">
      <div
        v-for="(step, index) in executionSteps"
        :key="step.label"
        class="flow-step"
        :style="{
          animationDelay: `${index * 90}ms`,
        }"
      >
        <div class="step-marker">
          <span></span>
        </div>

        <div class="step-body">
          <span class="step-time">
            {{ step.time }}
          </span>

          <strong>
            {{ step.label }}
          </strong>

          <p>
            {{ step.description }}
          </p>
        </div>
      </div>
    </section>

    <!-- TARGET → TRADE -->
    <section class="two-column">
      <div class="panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">TARGET PORTFOLIO</p>

            <h2>EIIE Output</h2>
          </div>

          <span class="small-badge"> 13:20 </span>
        </div>

        <div class="weight-change">
          <div>
            <span>Samsung</span>

            <strong> 24% </strong>
          </div>

          <span class="change-arrow"> → </span>

          <div class="after">
            <span>Target</span>

            <strong> 18% </strong>
          </div>
        </div>

        <div class="difference">
          <span>Weight Change</span>

          <strong>-6.0%p</strong>
        </div>
      </div>

      <div class="panel trade-panel">
        <div class="panel-header">
          <div>
            <p class="panel-label">REQUIRED TRADE</p>

            <h2>Samsung Electronics</h2>
          </div>

          <span class="sell-badge"> SELL </span>
        </div>

        <div class="trade-value">
          <strong> 63 shares </strong>

          <span> Estimated trade value </span>

          <h3>₩5,301,450</h3>
        </div>
      </div>
    </section>

    <!-- VENUE COMPARISON -->
    <section class="panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">MARKET VENUE</p>

          <h2>KRX / NXT Comparison</h2>
        </div>

        <span class="small-badge"> Rule-based </span>
      </div>

      <div class="venue-grid">
        <div
          v-for="(venue, index) in venues"
          :key="venue.name"
          class="venue-card"
          :class="{
            selected: venue.selected,
          }"
          :style="{
            animationDelay: `${index * 100}ms`,
          }"
        >
          <div class="venue-heading">
            <div>
              <span class="venue-label"> VENUE </span>

              <h3>
                {{ venue.name }}
              </h3>
            </div>

            <span v-if="venue.selected" class="selected-badge"> Selected </span>
          </div>

          <div class="orderbook">
            <div>
              <span>Best Bid</span>

              <strong> ₩{{ venue.bestBid.toLocaleString() }} </strong>
            </div>

            <div>
              <span>Best Ask</span>

              <strong> ₩{{ venue.bestAsk.toLocaleString() }} </strong>
            </div>
          </div>

          <div class="venue-metrics">
            <div>
              <span>Spread</span>

              <strong> ₩{{ venue.spread }} </strong>
            </div>

            <div>
              <span>Estimated Cost</span>

              <strong> ₩{{ venue.estimatedCost.toLocaleString() }} </strong>
            </div>
          </div>
        </div>
      </div>

      <div class="venue-reason">
        <div class="reason-icon">✓</div>

        <div>
          <strong> NXT Selected </strong>

          <p>
            더 높은 매수호가와 더 좁은 스프레드로 인해 예상 실행 비용이 낮은 것으로 계산되었습니다.
          </p>
        </div>
      </div>
    </section>

    <!-- ORDER DECISION -->
    <section class="decision-card">
      <div>
        <p class="decision-label">EXECUTION DECISION</p>

        <h2>NXT · Limit Order</h2>

        <p>목표 비중 조정을 위해 63주 매도 주문을 제출합니다.</p>
      </div>

      <div class="order-details">
        <div>
          <span>Side</span>
          <strong>SELL</strong>
        </div>

        <div>
          <span>Quantity</span>
          <strong>63</strong>
        </div>

        <div>
          <span>Limit Price</span>
          <strong>₩84,150</strong>
        </div>

        <div>
          <span>Venue</span>
          <strong>NXT</strong>
        </div>
      </div>
    </section>

    <!-- FILL -->
    <section class="panel fill-panel">
      <div class="panel-header">
        <div>
          <p class="panel-label">EXECUTION RESULT</p>

          <h2>Order Filled</h2>
        </div>

        <span class="filled-badge"> FILLED </span>
      </div>

      <div class="fill-grid">
        <div>
          <span>Submitted</span>
          <strong>13:20:04</strong>
        </div>

        <div>
          <span>Filled</span>
          <strong>13:20:06</strong>
        </div>

        <div>
          <span>Average Price</span>
          <strong>₩84,148</strong>
        </div>

        <div>
          <span>Slippage</span>
          <strong>0.02%</strong>
        </div>

        <div>
          <span>Execution Cost</span>
          <strong>₩1,920</strong>
        </div>
      </div>
    </section>

    <p class="prototype-note">
      현재 주문·호가·체결 수치는 UI 검증을 위한 mock simulation 데이터입니다.
    </p>
  </main>
</template>

<style scoped>
.execution-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 48px 56px 80px;
}

/* HEADER */

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
  max-width: 720px;
  margin: 0;
  color: #71717a;
  font-size: 14px;
  line-height: 1.6;
}

.eyebrow,
.panel-label,
.decision-label {
  margin: 0;
  color: #8b8b93;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.status,
.small-badge {
  padding: 8px 12px;
  border: 1px solid #e4e4e7;
  border-radius: 999px;
  background: #ffffff;
  color: #52525b;
  font-size: 11px;
  font-weight: 600;
}

/* COMMON PANEL */

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

.panel-header h2 {
  margin: 6px 0 0;
  font-size: 20px;
  letter-spacing: -0.025em;
}

/* EXECUTION FLOW */

.execution-flow {
  display: grid;
  grid-template-columns: repeat(5, 1fr);

  margin-bottom: 18px;
  padding: 22px 24px;

  border: 1px solid #e4e4e7;
  border-radius: 18px;

  background: #ffffff;
}

.flow-step {
  position: relative;

  display: flex;
  gap: 12px;

  opacity: 0;

  animation: flow-enter 400ms ease forwards;
}

.flow-step:not(:last-child)::after {
  position: absolute;

  top: 8px;
  left: 17px;
  right: -8px;

  height: 1px;

  background: #dedee2;

  content: '';

  transform: translateX(14px);
}

.step-marker {
  z-index: 1;

  display: flex;
  justify-content: center;

  width: 18px;

  flex-shrink: 0;
}

.step-marker span {
  width: 9px;
  height: 9px;

  margin-top: 4px;

  border: 2px solid #18181b;
  border-radius: 50%;

  background: #ffffff;

  animation: marker-enter 350ms ease both;
}

.flow-step:last-child .step-marker span {
  background: #18181b;
}

.step-body {
  display: flex;
  flex-direction: column;
}

.step-time {
  margin-bottom: 6px;

  color: #a1a1aa;

  font-size: 9px;
}

.step-body strong {
  font-size: 12px;
}

.step-body p {
  margin: 5px 0 0;

  color: #71717a;

  font-size: 10px;
  line-height: 1.5;
}

@keyframes flow-enter {
  from {
    opacity: 0;
    transform: translateY(6px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes marker-enter {
  from {
    transform: scale(0);
  }

  to {
    transform: scale(1);
  }
}

/* TARGET / TRADE */

.two-column {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}

.weight-change {
  display: grid;
  grid-template-columns: 1fr 50px 1fr;

  align-items: center;

  margin-top: 34px;
}

.weight-change > div {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.weight-change span {
  color: #71717a;
  font-size: 11px;
}

.weight-change strong {
  font-size: 30px;
  letter-spacing: -0.04em;
}

.change-arrow {
  text-align: center;
  font-size: 20px !important;
}

.after {
  text-align: right;
}

.difference {
  display: flex;
  justify-content: space-between;

  margin-top: 30px;
  padding: 14px;

  border-radius: 10px;

  background: #f7f7f8;

  font-size: 12px;
}

.sell-badge {
  padding: 6px 9px;

  border-radius: 7px;

  background: #f4eded;
  color: #8f3535;

  font-size: 10px;
  font-weight: 700;
}

.trade-value {
  display: flex;
  flex-direction: column;

  margin-top: 35px;
}

.trade-value > strong {
  font-size: 29px;
}

.trade-value span {
  margin-top: 25px;

  color: #71717a;

  font-size: 11px;
}

.trade-value h3 {
  margin: 5px 0 0;

  font-size: 20px;
}

/* VENUE */

.venue-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);

  gap: 16px;
}

.venue-card {
  padding: 22px;

  border: 1px solid #e4e4e7;
  border-radius: 14px;

  opacity: 0;

  animation: venue-enter 400ms ease forwards;

  transition:
    border-color 160ms ease,
    transform 160ms ease,
    box-shadow 160ms ease;
}

.venue-card:hover {
  transform: translateY(-2px);

  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.035);
}

.venue-card.selected {
  border-color: #18181b;
}

.venue-heading {
  display: flex;
  justify-content: space-between;
}

.venue-label {
  color: #a1a1aa;

  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.venue-heading h3 {
  margin: 5px 0 0;

  font-size: 22px;
}

.selected-badge {
  height: fit-content;

  padding: 5px 8px;

  border-radius: 999px;

  background: #18181b;
  color: #ffffff;

  font-size: 9px;
}

.orderbook {
  display: grid;
  grid-template-columns: 1fr 1fr;

  gap: 10px;

  margin-top: 26px;
}

.orderbook div,
.venue-metrics div {
  display: flex;
  flex-direction: column;
  gap: 6px;

  padding: 13px;

  border-radius: 9px;

  background: #f7f7f8;
}

.orderbook span,
.venue-metrics span {
  color: #71717a;

  font-size: 10px;
}

.orderbook strong {
  font-size: 15px;
}

.venue-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;

  gap: 10px;

  margin-top: 10px;
}

.venue-metrics strong {
  font-size: 12px;
}

.venue-reason {
  display: flex;

  gap: 12px;

  margin-top: 18px;
  padding: 16px;

  border-radius: 11px;

  background: #f7f7f8;
}

.reason-icon {
  display: grid;

  width: 28px;
  height: 28px;

  flex-shrink: 0;

  place-items: center;

  border-radius: 50%;

  background: #18181b;
  color: #ffffff;

  font-size: 11px;
}

.venue-reason strong {
  font-size: 13px;
}

.venue-reason p {
  margin: 5px 0 0;

  color: #71717a;

  font-size: 11px;
  line-height: 1.5;
}

@keyframes venue-enter {
  from {
    opacity: 0;
    transform: translateY(7px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* EXECUTION DECISION */

.decision-card {
  display: flex;
  justify-content: space-between;
  align-items: center;

  gap: 30px;

  margin-bottom: 18px;
  padding: 28px;

  border-radius: 18px;

  background: #18181b;
  color: #ffffff;

  box-shadow: 0 12px 35px rgba(0, 0, 0, 0.08);
}

.decision-card h2 {
  margin: 7px 0 8px;

  font-size: 25px;
}

.decision-card p:not(.decision-label) {
  margin: 0;

  color: #b8b8bf;

  font-size: 12px;
}

.decision-label {
  color: #a1a1aa;
}

.order-details {
  display: grid;
  grid-template-columns: repeat(4, 1fr);

  gap: 18px;
}

.order-details div {
  display: flex;
  min-width: 85px;
  flex-direction: column;

  gap: 5px;
}

.order-details span {
  color: #92929a;

  font-size: 9px;
}

.order-details strong {
  font-size: 13px;
}

/* FILL */

.filled-badge {
  padding: 6px 9px;

  border-radius: 999px;

  background: #edf5ef;
  color: #28633a;

  font-size: 9px;
  font-weight: 700;
}

.fill-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);

  gap: 12px;
}

.fill-grid div {
  display: flex;
  flex-direction: column;

  gap: 7px;

  padding: 15px;

  border-radius: 10px;

  background: #f7f7f8;
}

.fill-grid span {
  color: #71717a;

  font-size: 10px;
}

.fill-grid strong {
  font-size: 13px;
}

.prototype-note {
  margin-top: 12px;

  color: #a1a1aa;

  font-size: 10px;

  text-align: right;
}

/* RESPONSIVE */

@media (max-width: 1000px) {
  .execution-page {
    padding: 36px 28px 60px;
  }

  .execution-flow {
    grid-template-columns: 1fr;
    gap: 18px;
  }

  .flow-step:not(:last-child)::after {
    top: 14px;
    right: auto;
    bottom: -20px;
    left: 4px;

    width: 1px;
    height: auto;

    transform: none;
  }

  .two-column,
  .venue-grid {
    grid-template-columns: 1fr;
  }

  .decision-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .order-details,
  .fill-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.source-context {
  display: grid;

  grid-template-columns:
    1fr
    auto
    1fr
    auto
    1fr;

  align-items: center;

  gap: 18px;

  margin-bottom: 18px;
  padding: 18px 22px;

  border: 1px solid #e4e4e7;
  border-radius: 14px;

  background: #ffffff;
}

.source-context > div:not(.flow-arrow) {
  display: flex;
  flex-direction: column;

  gap: 5px;
}

.source-context span {
  color: #a1a1aa;

  font-size: 9px;
  font-weight: 700;

  letter-spacing: 0.08em;
}

.source-context strong {
  font-size: 12px;
}

.flow-arrow {
  color: #b4b4ba;

  font-size: 16px;
}
</style>
