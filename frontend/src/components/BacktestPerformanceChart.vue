<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'

const chartRef = ref<HTMLDivElement | null>(null)

let chart: echarts.ECharts | null = null

const dates = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const eiie = [100, 101, 103, 105, 104, 108, 110, 112, 115, 116, 118, 120]

const mvo = [100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 113, 114]

const ew = [100, 100.5, 101, 102, 102.5, 103, 104, 105, 106, 107, 108, 109]

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
      data: ['EIIE', 'MVO', 'EW'],
    },

    grid: {
      top: 45,
      left: 15,
      right: 15,
      bottom: 25,
      containLabel: true,
    },

    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: dates,
      axisTick: {
        show: false,
      },
      axisLine: {
        lineStyle: {
          color: '#e4e4e7',
        },
      },
    },

    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {
        lineStyle: {
          color: '#f0f0f1',
        },
      },
    },

    series: [
      {
        name: 'EIIE',
        type: 'line',
        data: eiie,

        smooth: true,
        symbol: 'none',

        lineStyle: {
          width: 3,
          color: '#3654ff',
        },

        itemStyle: {
          color: '#3654ff',
        },

        emphasis: {
          focus: 'series',
        },
      },

      {
        name: 'MVO',
        type: 'line',
        data: mvo,

        smooth: true,
        symbol: 'none',

        lineStyle: {
          width: 2,
          color: '#9acb32',
        },

        itemStyle: {
          color: '#9acb32',
        },

        emphasis: {
          focus: 'series',
        },
      },

      {
        name: 'EW',
        type: 'line',
        data: ew,

        smooth: true,
        symbol: 'none',

        lineStyle: {
          width: 2,
          type: 'dashed',
          color: '#626b89',
        },

        itemStyle: {
          color: '#626b89',
        },

        emphasis: {
          focus: 'series',
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
  height: 320px;
}
</style>
