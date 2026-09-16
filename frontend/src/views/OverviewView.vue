<script setup lang="ts">
import { RouterLink } from 'vue-router'

import PortfolioPerformanceChart from '@/components/PortfolioPerformanceChart.vue'

const allocation = [
  {
    name: 'Samsung',
    weight: 18,
  },
  {
    name: 'SK hynix',
    weight: 16,
  },
  {
    name: 'NAVER',
    weight: 12,
  },
  {
    name: 'Defensive',
    weight: 26,
  },
  {
    name: 'Cash',
    weight: 15,
  },
  {
    name: 'Other',
    weight: 13,
  },
]

const metrics = [
  {
    label: 'Total Return',
    value: '+4.82%',
  },
  {
    label: 'Sharpe Ratio',
    value: '1.21',
  },
  {
    label: 'MDD',
    value: '-3.20%',
  },
  {
    label: 'Turnover',
    value: '8.70%',
  },
]
</script>

<template>
  <main class="overview-page">
    <!-- HEADER -->
    <header class="page-header">
      <div>
        <p class="eyebrow">PORTFOLIO OPTIMIZATION</p>

        <h1>Overview</h1>

        <p class="subtitle">
          Deep RL · EIIE 기반 포트폴리오 운용 결과와 최근 리밸런싱 상태를 요약합니다.
        </p>
      </div>

      <span class="status"> Historical Replay </span>
    </header>

    <!-- HERO -->
    <section class="hero-grid">
      <div class="hero-card">
        <div class="portfolio-heading">
          <div>
            <p class="label">Portfolio Value</p>

            <h2>₩104,820,000</h2>

            <span class="positive"> +4.82% </span>
          </div>

          <div class="period">
            Simulation Period
            <strong>2026.09</strong>
          </div>
        </div>

        <PortfolioPerformanceChart />
      </div>

      <!-- CURRENT STATE -->
      <div class="state-card">
        <div>
          <p class="label">Portfolio Model</p>

          <h3>Deep RL · EIIE</h3>

          <span class="tag"> Active </span>
        </div>

        <div class="divider"></div>

        <div>
          <p class="label">Market Regime</p>

          <h3>Risk-Off</h3>

          <p class="muted">Risk-Off signal 71%</p>
        </div>

        <div class="regime-progress">
          <span></span>
        </div>
      </div>
    </section>

    <!-- METRICS -->
    <section class="metrics">
      <div
        v-for="(metric, index) in metrics"
        :key="metric.label"
        class="metric-card"
        :style="{
          animationDelay: `${index * 60}ms`,
        }"
      >
        <span>
          {{ metric.label }}
        </span>

        <strong>
          {{ metric.value }}
        </strong>
      </div>
    </section>

    <!-- LOWER -->
    <section class="bottom-grid">
      <!-- LATEST REBALANCE -->
      <div class="panel decision-card">
        <div class="panel-header">
          <div>
            <p class="eyebrow">LATEST REBALANCE</p>

            <h3>Portfolio Rebalanced</h3>
          </div>

          <span class="time"> 13:20 </span>
        </div>

        <div class="rebalance-summary">
          <div class="rebalance-icon">↗</div>

          <div>
            <strong> Risk exposure reduced </strong>

            <p>EIIE가 새로운 목표 비중을 출력하고 리밸런싱 조건을 충족했습니다.</p>
          </div>
        </div>

        <div class="decision-metrics">
          <div>
            <span> Expected Risk </span>

            <strong> 10.9% → 8.1% </strong>
          </div>

          <div>
            <span> Expected Sharpe </span>

            <strong> 1.10 → 1.31 </strong>
          </div>

          <div>
            <span> Transaction Cost </span>

            <strong> 0.08% </strong>
          </div>
        </div>

        <RouterLink to="/replay" class="detail-link">
          <span> View Decision Replay </span>

          <span class="arrow"> → </span>
        </RouterLink>
      </div>

      <!-- CURRENT ALLOCATION -->
      <div class="panel allocation-card">
        <div class="panel-header">
          <div>
            <p class="eyebrow">CURRENT ALLOCATION</p>

            <h3>Portfolio Weights</h3>
          </div>

          <span class="allocation-total"> 100% </span>
        </div>

        <div class="allocation">
          <div v-for="item in allocation" :key="item.name" class="allocation-row">
            <span>
              {{ item.name }}
            </span>

            <div class="track">
              <div
                class="fill"
                :style="{
                  width: `${item.weight * 3}%`,
                }"
              ></div>
            </div>

            <strong> {{ item.weight }}% </strong>
          </div>
        </div>
      </div>
    </section>
  </main>
</template>

<style scoped>
.overview-page {
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

.eyebrow,
.label {
  margin: 0;

  color: #8b8b93;

  font-size: 11px;

  font-weight: 700;

  letter-spacing: 0.09em;
}

.subtitle {
  max-width: 700px;

  margin: 0;

  color: #71717a;

  font-size: 14px;

  line-height: 1.6;
}

.status,
.tag {
  display: inline-flex;

  align-items: center;

  padding: 7px 11px;

  border: 1px solid #e4e4e7;

  border-radius: 999px;

  background: #ffffff;

  font-size: 12px;
}

.hero-grid {
  display: grid;

  grid-template-columns:
    2.3fr
    1fr;

  gap: 18px;
}

.hero-card,
.state-card,
.metric-card,
.panel {
  border: 1px solid var(--border);

  background: var(--surface);
}

.hero-card,
.state-card,
.panel {
  border-radius: 18px;
}

.hero-card {
  padding: 28px;

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.hero-card:hover {
  border-color: var(--primary-border);

  box-shadow: var(--shadow-soft);
}

.portfolio-heading {
  display: flex;

  justify-content: space-between;

  align-items: flex-start;
}

.hero-card h2 {
  margin: 8px 0 3px;

  font-size: 38px;

  letter-spacing: -0.04em;
}

.panel:hover {
  border-color: var(--primary-border);

  box-shadow: var(--shadow-soft);
}

.period {
  display: flex;

  flex-direction: column;

  gap: 4px;

  color: #8b8b93;

  font-size: 10px;

  text-align: right;
}

.period strong {
  color: #18181b;

  font-size: 12px;
}

.state-card {
  display: flex;

  flex-direction: column;

  justify-content: center;

  gap: 26px;

  padding: 28px;

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.state-card:hover {
  border-color: var(--primary-border);

  box-shadow: var(--shadow-soft);
}

.state-card h3 {
  margin: 7px 0 10px;

  font-size: 26px;

  letter-spacing: -0.03em;
}

.divider {
  height: 1px;

  background: #eeeef0;
}

.muted {
  margin: 8px 0 0;

  color: #71717a;

  font-size: 12px;
}

.regime-progress {
  height: 7px;

  overflow: hidden;

  border-radius: 999px;

  background: #f1f1f3;
}

.regime-progress span {
  display: block;

  width: 71%;
  height: 100%;

  border-radius: inherit;

  background: linear-gradient(90deg, #f0a24a, #dd6b45);

  animation: progress-enter 700ms cubic-bezier(0.4, 0, 0.2, 1) both;
}

@keyframes progress-enter {
  from {
    width: 0;
  }
}

.metrics {
  display: grid;

  grid-template-columns: repeat(4, 1fr);

  gap: 14px;

  margin-top: 14px;
}

.metric-card {
  display: flex;

  flex-direction: column;

  gap: 10px;

  padding: 20px;

  border-radius: 14px;

  opacity: 0;

  animation: metric-enter 350ms ease forwards;

  transition:
    border-color 150ms ease,
    transform 150ms ease;
}

.metric-card:hover {
  border-color: #d4d4d8;

  transform: translateY(-2px);
}

.metric-card span {
  color: #71717a;

  font-size: 12px;
}

.metric-card strong {
  font-size: 19px;

  letter-spacing: -0.02em;
}

@keyframes metric-enter {
  from {
    opacity: 0;

    transform: translateY(8px);
  }

  to {
    opacity: 1;

    transform: translateY(0);
  }
}

.bottom-grid {
  display: grid;

  grid-template-columns:
    1.2fr
    1fr;

  gap: 18px;

  margin-top: 18px;
}

.panel {
  padding: 26px;

  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.panel:hover {
  border-color: var(--primary-border);

  box-shadow: var(--shadow-soft);
}

.panel-header {
  display: flex;

  justify-content: space-between;

  align-items: flex-start;
}

.panel h3 {
  margin: 5px 0 0;

  font-size: 20px;

  letter-spacing: -0.025em;
}

.time,
.allocation-total {
  color: #71717a;

  font-size: 12px;

  font-weight: 600;
}

.rebalance-summary {
  display: flex;

  gap: 14px;

  align-items: flex-start;

  margin-top: 28px;

  padding: 18px 0;
}

.rebalance-icon {
  display: grid;

  width: 38px;
  height: 38px;

  flex-shrink: 0;

  place-items: center;

  border-radius: 11px;

  background: var(--primary-soft);

  color: var(--primary);

  font-size: 17px;
  font-weight: 700;

  transition:
    transform 180ms ease,
    background 180ms ease;
}

.decision-card:hover .rebalance-icon {
  transform: rotate(-4deg) scale(1.04);
}

.rebalance-summary strong {
  font-size: 14px;
}

.rebalance-summary p {
  max-width: 520px;

  margin: 5px 0 0;

  color: #71717a;

  font-size: 12px;

  line-height: 1.6;
}

.decision-metrics {
  display: grid;

  grid-template-columns: repeat(3, 1fr);

  gap: 12px;
}

.decision-metrics div {
  padding: 14px;

  border-radius: 10px;

  background: #f7f7f8;
}

.decision-metrics span {
  display: block;

  margin-bottom: 6px;

  color: #71717a;

  font-size: 10px;
}

.decision-metrics strong {
  font-size: 13px;
}

.detail-link {
  display: flex;

  justify-content: space-between;
  align-items: center;

  margin-top: 24px;

  padding: 13px 15px;

  border-radius: 11px;

  background: var(--primary);

  color: #ffffff;

  font-size: 12px;
  font-weight: 650;

  box-shadow: var(--shadow-primary);

  transition:
    transform 150ms ease,
    background 150ms ease,
    box-shadow 150ms ease;
}

.detail-link:hover {
  background: var(--primary-hover);

  transform: translateY(-1px);

  box-shadow: 0 10px 26px rgba(54, 84, 255, 0.23);
}

.detail-link .arrow {
  transition: transform 150ms ease;
}

.detail-link:hover .arrow {
  transform: translateX(3px);
}

.allocation {
  margin-top: 26px;
}

.allocation-row {
  display: grid;

  grid-template-columns:
    90px
    1fr
    42px;

  gap: 10px;

  align-items: center;

  margin-bottom: 17px;

  font-size: 12px;
}

.track {
  height: 7px;

  overflow: hidden;

  border-radius: 999px;

  background: #f0f0f1;
}

.fill {
  height: 100%;

  border-radius: inherit;

  transform-origin: left;

  animation: allocation-enter 650ms cubic-bezier(0.4, 0, 0.2, 1) both;
}

.allocation-row:nth-child(1) .fill {
  background: #4f6ff5;
}

.allocation-row:nth-child(2) .fill {
  background: #9ecb3b;
}

.allocation-row:nth-child(3) .fill {
  background: #626b89;
}

.allocation-row:nth-child(4) .fill {
  background: #eb985b;
}

.allocation-row:nth-child(5) .fill {
  background: #34a9c9;
}

.allocation-row:nth-child(6) .fill {
  background: #efbf32;
}

.positive {
  color: var(--success);
  font-size: 13px;
  font-weight: 700;
}

@keyframes allocation-enter {
  from {
    transform: scaleX(0);
  }

  to {
    transform: scaleX(1);
  }
}

@media (max-width: 1000px) {
  .overview-page {
    padding: 36px 28px 60px;
  }

  .hero-grid,
  .bottom-grid {
    grid-template-columns: 1fr;
  }

  .metrics {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
