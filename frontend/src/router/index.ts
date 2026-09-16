import { createRouter, createWebHistory } from 'vue-router'

import OverviewView from '../views/OverviewView.vue'
import DecisionReplayView from '../views/DecisionReplayView.vue'
import BacktestView from '../views/BacktestView.vue'
import ExecutionView from '../views/ExecutionView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),

  routes: [
    {
      path: '/',
      name: 'overview',
      component: OverviewView,
    },
    {
      path: '/replay',
      name: 'decision-replay',
      component: DecisionReplayView,
    },
    {
      path: '/backtest',
      name: 'backtest',
      component: BacktestView,
    },
    {
      path: '/execution',
      name: 'execution',
      component: ExecutionView,
    },
  ],
})

export default router
