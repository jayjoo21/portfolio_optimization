<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'

import * as echarts from 'echarts'

const props = defineProps<{
  selectedIndex: number
}>()

const emit = defineEmits<{
  select: [index: number]
}>()

const chartRef = ref<HTMLDivElement | null>(null)

let chart: echarts.ECharts | null = null

const times = ['13:00', '13:10', '13:15', '13:20']

const portfolioValues = [101.8, 101.2, 101.4, 102.1]

const getSeriesData = () => {
  return portfolioValues.map((value, index) => ({
    value,

    symbolSize: index === props.selectedIndex ? 12 : 6,

    itemStyle: {
      opacity: index === props.selectedIndex ? 1 : 0.6,
    },
  }))
}

const getChartOption = () => ({
  tooltip: {
    trigger: 'axis',

    formatter(params: any) {
      const point = params[0]

      return `
        <strong>${point.axisValue}</strong><br/>
        Portfolio Index: ${point.value}
      `
    },
  },

  grid: {
    top: 40,
    left: 20,
    right: 25,
    bottom: 35,
    containLabel: true,
  },

  xAxis: {
    type: 'category',
    boundaryGap: false,

    data: times,

    axisTick: {
      show: false,
    },

    axisLine: {
      lineStyle: {
        color: '#e4e4e7',
      },
    },

    axisLabel: {
      color: '#71717a',
    },
  },

  yAxis: {
    type: 'value',
    scale: true,

    axisLabel: {
      color: '#71717a',
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

      data: getSeriesData(),

      smooth: true,

      symbol: 'circle',

      lineStyle: {
        width: 3,
      },

      emphasis: {
        focus: 'series',
      },

      /*
        시장 국면 배경
      */
      markArea: {
        silent: true,

        label: {
          position: 'insideTop',
          fontSize: 11,
          color: '#71717a',
        },

        data: [
          [
            {
              name: 'RISK-ON',
              xAxis: '13:00',

              itemStyle: {
                color: 'rgba(22, 163, 74, 0.06)',
              },
            },
            {
              xAxis: '13:10',
            },
          ],

          [
            {
              name: 'TRANSITION',
              xAxis: '13:10',

              itemStyle: {
                color: 'rgba(113, 113, 122, 0.06)',
              },
            },
            {
              xAxis: '13:15',
            },
          ],

          [
            {
              name: 'RISK-OFF',
              xAxis: '13:15',

              itemStyle: {
                color: 'rgba(220, 38, 38, 0.05)',
              },
            },
            {
              xAxis: '13:20',
            },
          ],
        ],
      },

      /*
        중요 이벤트
      */
      markPoint: {
        symbol: 'pin',
        symbolSize: 42,

        data: [
          {
            name: 'Strategy Switch',

            coord: ['13:20', 102.1],

            value: 'SWITCH',

            label: {
              show: true,
              formatter: 'SWITCH',
              fontSize: 9,
            },
          },
        ],
      },

      /*
        현재 선택 시점 세로선
      */
      markLine: {
        silent: true,
        symbol: 'none',

        lineStyle: {
          type: 'dashed',
          width: 1,
        },

        label: {
          show: false,
        },

        data: [
          {
            xAxis: times[props.selectedIndex],
          },
        ],
      },
    },
  ],
})

const renderChart = () => {
  if (!chartRef.value) return

  chart = echarts.init(chartRef.value)

  chart.setOption(getChartOption())

  chart.on('click', (params) => {
    if (params.componentType === 'series' && params.dataIndex !== undefined) {
      emit('select', params.dataIndex)
    }
  })
}

const updateChart = () => {
  if (!chart) return

  chart.setOption({
    series: [
      {
        data: getSeriesData(),

        markLine: {
          data: [
            {
              xAxis: times[props.selectedIndex],
            },
          ],
        },
      },
    ],
  })
}

watch(
  () => props.selectedIndex,

  () => {
    updateChart()
  },
)

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
  height: 290px;
}
</style>
