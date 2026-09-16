<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'

const chartRef = ref<HTMLDivElement | null>(null)

let chart: echarts.ECharts | null = null

const dates = [
  '09:00',
  '09:30',
  '10:00',
  '10:30',
  '11:00',
  '11:30',
  '12:00',
  '12:30',
  '13:00',
  '13:30',
  '14:00',
  '14:30',
]

const portfolio = [100, 100.4, 100.8, 100.5, 101.2, 101.6, 101.3, 102.1, 101.8, 102.8, 103.4, 104.1]

const benchmark = [100, 100.2, 100.3, 100.1, 100.5, 100.7, 100.6, 100.9, 101.0, 101.2, 101.5, 101.7]

const renderChart = () => {
  if (!chartRef.value) return

  chart = echarts.init(chartRef.value)

  chart.setOption({
    animation: true,
    animationDuration: 900,
    animationEasing: 'cubicOut',

    tooltip: {
      trigger: 'axis',
    },

    legend: {
      top: 0,
      right: 0,
      data: ['Portfolio', 'Benchmark'],
    },

    grid: {
      top: 40,
      left: 15,
      right: 15,
      bottom: 25,
      containLabel: true,
    },

    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: dates,
      axisLine: {
        lineStyle: {
          color: '#e4e4e7',
        },
      },
      axisTick: {
        show: false,
      },
    },

    yAxis: {
      type: 'value',
      scale: true,
      axisLabel: {
        formatter: '{value}',
      },
      splitLine: {
        lineStyle: {
          color: '#f0f0f1',
        },
      },
    },

    series: [
      {
        name: 'Portfolio',
        type: 'line',
        data: portfolio,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 3,
        },
      },
      {
        name: 'Benchmark',
        type: 'line',
        data: benchmark,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          type: 'dashed',
        },
      },
    ],
  })
}

const handleResize = () => {
  chart?.resize()
}

onMounted(() => {
  renderChart()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  chart?.dispose()
})
</script>

<template>
  <div ref="chartRef" class="chart"></div>
</template>

<style scoped>
.chart {
  width: 100%;
  height: 280px;
}
</style>
