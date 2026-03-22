<template>
  <div class="page">
    <div class="header">
      <span class="header-icon">📡</span>
      <span class="header-title">信号接收端</span>
      <span class="header-sub">Signal Receiver</span>
    </div>

    <div class="card" style="max-width:960px">
      <el-button
        type="primary"
        :loading="loading"
        size="large"
        style="width:100%"
        @click="receive"
      >
        {{ loading ? '识别中...' : '接收并识别解调' }}
      </el-button>
    </div>

    <transition name="fade">
      <div v-if="result">
        <el-alert
          v-if="result.status !== 'success'"
          :title="result.message"
          type="error"
          show-icon
          style="max-width:720px;margin:0 auto 16px"
        />

        <template v-if="result.status === 'success'">
          <!-- 顶部状态卡 -->
          <div class="card">
            <div class="card-title">
              解调结果
              <el-tag type="success" size="small">成功</el-tag>
            </div>
            <div class="demod-text">{{ result.demodulated_text }}</div>

            <!-- 实际 vs 识别 对比 -->
            <div class="mod-compare">
              <div class="mod-box actual">
                <div class="mod-box-label">实际调制类型</div>
                <div class="mod-box-value">{{ formatMod(result.actual_modulation_type) }}</div>
              </div>
              <div class="mod-arrow">
                <span :class="isCorrect ? 'arrow-ok' : 'arrow-err'">
                  {{ isCorrect ? '✓ 识别正确' : '✗ 识别偏差' }}
                </span>
              </div>
              <div class="mod-box" :class="isCorrect ? 'recognized-ok' : 'recognized-err'">
                <div class="mod-box-label">MLP 识别类型</div>
                <div class="mod-box-value">{{ formatMod(result.recognized_modulation_type) }}</div>
              </div>
            </div>
          </div>

          <!-- 识别概率柱状图 -->
          <div class="card">
            <div class="card-title">MLP 识别概率分布</div>
            <div ref="chartRef" style="width:100%;height:320px" />
          </div>

          <!-- 二进制流 -->
          <div class="card">
            <div class="card-title">二进制流</div>
            <div class="bin-display">{{ result.binary_string }}</div>
          </div>
        </template>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'

const loading = ref(false)
const result = ref(null)
const chartRef = ref(null)
let chartInstance = null

const modMap = {
  AM: 'AM 调幅',
  '2FSK': '2FSK 频移键控',
  BPSK: 'BPSK 相移键控',
  QPSK: 'QPSK 四相键控',
  '16QAM': '16QAM 正交调幅'
}
const formatMod = key => modMap[key] ?? key

const isCorrect = computed(() =>
  result.value?.actual_modulation_type === result.value?.recognized_modulation_type
)

async function receive() {
  loading.value = true
  result.value = null
  try {
    const { data } = await axios.get('/api/receive_and_demodulate')
    result.value = data
    ElMessage.success('接收识别成功')
    await nextTick()
    renderChart(data.recognition_probability, data.recognized_modulation_type)
  } catch (e) {
    result.value = {
      status: 'error',
      message: e.response?.data?.detail ?? '网络异常：' + e.message
    }
    ElMessage.error('接收失败')
  } finally {
    loading.value = false
  }
}

function renderChart(probDict, recognized) {
  if (!chartRef.value) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(chartRef.value)

  const keys = Object.keys(probDict)
  const vals = keys.map(k => +(probDict[k] * 100).toFixed(2))
  const colors = keys.map(k => k === recognized ? '#409eff' : '#c0d8f5')

  chartInstance.setOption({
    animation: true,
    grid: { top: 16, bottom: 48, left: 48, right: 16 },
    xAxis: {
      type: 'category',
      data: keys,
      axisLabel: { fontSize: 11, interval: 0 }
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLabel: { formatter: '{value}%', fontSize: 11 }
    },
    series: [{
      type: 'bar',
      data: vals.map((v, i) => ({ value: v, itemStyle: { color: colors[i], borderRadius: [6,6,0,0] } })),
      label: {
        show: true,
        position: 'top',
        formatter: '{c}%',
        fontSize: 11,
        color: '#555'
      },
      barMaxWidth: 56
    }],
    tooltip: {
      trigger: 'axis',
      formatter: params => `${params[0].name}<br/>概率：${params[0].value}%`
    }
  })
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  background: linear-gradient(135deg, #f0f4ff 0%, #f5f7fa 100%);
  padding: 40px 40px 64px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.header {
  width: 100%;
  max-width: 960px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
}
.header-icon { font-size: 28px; }
.header-title {
  font-size: 26px;
  font-weight: 700;
  color: #1a1a2e;
}
.header-sub {
  font-size: 14px;
  color: #aaa;
  margin-top: 2px;
}

.card {
  width: 100%;
  max-width: 960px;
  background: #fff;
  border-radius: 16px;
  padding: 28px 36px;
  margin-bottom: 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.07);
}
.card-title {
  font-size: 15px;
  font-weight: 600;
  color: #333;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.demod-text {
  font-size: 36px;
  font-weight: 700;
  color: #1a1a2e;
  letter-spacing: 4px;
  margin-bottom: 24px;
  padding: 20px;
  background: #f8faff;
  border-radius: 10px;
  text-align: center;
}

.mod-compare {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-top: 8px;
}
.mod-box {
  flex: 1;
  border-radius: 12px;
  padding: 18px 24px;
  background: #f0f6ff;
  border: 1.5px solid #d0e4ff;
}
.mod-box.actual {
  background: #f0f6ff;
  border-color: #b0ccf0;
}
.mod-box.recognized-ok {
  background: #f0fff4;
  border-color: #67c23a;
}
.mod-box.recognized-err {
  background: #fff5f5;
  border-color: #f56c6c;
}
.mod-box-label {
  font-size: 12px;
  color: #999;
  margin-bottom: 6px;
}
.mod-box-value {
  font-size: 18px;
  font-weight: 700;
  color: #333;
}
.mod-arrow {
  text-align: center;
  flex-shrink: 0;
  font-size: 14px;
}
.arrow-ok {
  color: #67c23a;
  font-size: 12px;
  font-weight: 600;
}
.arrow-err {
  color: #f56c6c;
  font-size: 12px;
  font-weight: 600;
}

.bin-display {
  font-family: monospace;
  font-size: 12px;
  color: #555;
  background: #f5f5f5;
  border-radius: 8px;
  padding: 10px 12px;
  word-break: break-all;
  line-height: 1.6;
  max-height: 80px;
  overflow-y: auto;
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
