<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'

import * as echarts from 'echarts'

type Allocation = {
  samsung: number
  hynix: number
  naver: number
  defensive: number
  cash: number
  other: number
}

const props = defineProps<{
  before: Allocation
  after: Allocation
}>()

const chartRef = ref<HTMLDivElement | null>(null)

let chart: echarts.ECharts | null = null

const assets = [
  {
    key: 'samsung',
    name: 'Samsung',
  },
  {
    key: 'hynix',
    name: 'SK hynix',
  },
  {
    key: 'naver',
    name: 'NAVER',
  },
  {
    key: 'defensive',
    name: 'Defensive',
  },
  {
    key: 'cash',
    name: 'Cash',
  },
  {
    key: 'other',
    name: 'Other',
  },
] as const

const getSeries = () => {
  return assets.map((asset) => ({
    name: asset.name,
    type: 'bar',
    stack: 'allocation',
    barWidth: 38,

    data: [props.before[asset.key], props.after[asset.key]],

    label: {
      show: true,

      formatter(params: any) {
        return params.value >= 10 ? `${params.value}%` : ''
      },

      fontSize: 10,
    },

    emphasis: {
      focus: 'series',
    },
  }))
}

const getOption = () => ({
  animation: true,
  animationDuration: 700,
  animationDurationUpdate: 650,
  animationEasing: 'cubicOut' as const,
  animationEasingUpdate: 'cubicOut' as const,

  tooltip: {
    trigger: 'axis',

    axisPointer: {
      type: 'shadow',
    },

    valueFormatter(value: any) {
      return `${value}%`
    },
  },

  legend: {
    bottom: 0,
    itemWidth: 10,
    itemHeight: 10,

    textStyle: {
      fontSize: 11,
      color: '#71717a',
    },
  },

  grid: {
    top: 10,
    left: 75,
    right: 20,
    bottom: 55,
  },

  xAxis: {
    type: 'value',
    max: 100,

    axisLabel: {
      formatter: '{value}%',
      color: '#8b8b93',
    },

    splitLine: {
      lineStyle: {
        color: '#f0f0f1',
      },
    },
  },

  yAxis: {
    type: 'category',

    data: ['Before', 'After'],

    axisTick: {
      show: false,
    },

    axisLine: {
      show: false,
    },

    axisLabel: {
      color: '#52525b',
      fontWeight: 600,
    },
  },

  series: getSeries(),
})

const renderChart = () => {
  if (!chartRef.value) return

  chart = echarts.init(chartRef.value)

  chart.setOption(getOption())
}

const updateChart = () => {
  if (!chart) return

  chart.setOption({
    series: getSeries(),
  })
}

watch(
  () => [props.before, props.after],

  () => {
    updateChart()
  },

  {
    deep: true,
  },
)

const resizeChart = () => {
  chart?.resize()
}

onMounted(() => {
  renderChart()

  window.addEventListener('resize', resizeChart)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeChart)

  chart?.dispose()
})
</script>

<template>
  <div ref="chartRef" class="allocation-chart"></div>
</template>

<style scoped>
.allocation-chart {
  width: 100%;
  height: 260px;
}
</style>
