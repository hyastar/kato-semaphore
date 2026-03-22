<template>
  <div class="page">
    <div class="header">
      <span class="header-icon">📡</span>
      <span class="header-title">信号发送端</span>
      <span class="header-sub">Signal Transmitter</span>
    </div>

    <div class="card">
      <div class="card-title">发送配置</div>
      <el-form label-width="100px">
        <el-form-item label="输入文本">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="4"
            placeholder="请输入要发送的文本内容"
          />
        </el-form-item>
        <el-form-item label="调制类型">
          <el-select v-model="modType" style="width:100%">
            <el-option label="AM — 调幅" value="AM" />
            <el-option label="2FSK — 频移键控" value="2FSK" />
            <el-option label="BPSK — 二进制相移键控" value="BPSK" />
            <el-option label="QPSK — 四进制相移键控" value="QPSK" />
            <el-option label="16QAM — 正交振幅调制" value="16QAM" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-alert type="info" :closable="false" show-icon>
            <template #title>
              QPSK 需二进制长度为 2 的倍数；16QAM 需为 4 的倍数
            </template>
          </el-alert>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            size="large"
            style="width:200px"
            @click="send"
          >
            {{ loading ? '发送中...' : '发送信号' }}
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <transition name="fade">
      <div v-if="result" class="card">
        <div class="card-title">
          发送结果
          <el-tag :type="result.status === 'success' ? 'success' : 'danger'" size="small">
            {{ result.status === 'success' ? '成功' : '失败' }}
          </el-tag>
        </div>

        <el-alert
          :title="result.message"
          :type="result.status === 'success' ? 'success' : 'error'"
          show-icon
          :closable="false"
          style="margin-bottom:16px"
        />

        <template v-if="result.status === 'success'">
          <div class="info-grid">
            <div class="info-item">
              <div class="info-label">调制类型</div>
              <div class="info-value accent">{{ result.modulation_type }}</div>
            </div>
            <div class="info-item">
              <div class="info-label">信号长度</div>
              <div class="info-value">{{ result.signal_length.toLocaleString() }} 采样点</div>
            </div>
          </div>

          <div class="info-item" style="margin-top:12px">
            <div class="info-label">二进制流</div>
            <div class="bin-display">{{ result.binary_string }}</div>
          </div>

          <!-- 波形图 -->
          <div class="chart-wrap">
            <div class="info-label" style="margin-bottom:8px">时域信号波形（前 400 采样点）</div>
            <div ref="chartRef" style="width:100%;height:280px" />
          </div>
        </template>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'

const inputText = ref('')
const modType = ref('2FSK')
const loading = ref(false)
const result = ref(null)
const chartRef = ref(null)
let chartInstance = null

function validate() {
  if (!inputText.value) {
    ElMessage.warning('请输入要发送的文本')
    return false
  }
  const bin = inputText.value.split('').map(c =>
    c.charCodeAt(0).toString(2).padStart(8, '0')
  ).join('')
  if (modType.value === 'QPSK' && bin.length % 2 !== 0) {
    ElMessage.warning('QPSK 要求二进制长度为 2 的倍数')
    return false
  }
  if (modType.value === '16QAM' && bin.length % 4 !== 0) {
    ElMessage.warning('16QAM 要求二进制长度为 4 的倍数')
    return false
  }
  return true
}

async function send() {
  if (!validate()) return
  loading.value = true
  result.value = null
  try {
    const { data } = await axios.post('/api/modulate_and_send', {
      text: inputText.value,
      mod_type: modType.value
    })
    result.value = data
    ElMessage.success('发送成功')
    await nextTick()
    renderChart(data.binary_string, data.modulation_type)
  } catch (e) {
    result.value = {
      status: 'error',
      message: e.response?.data?.detail ?? '网络异常：' + e.message
    }
    ElMessage.error('发送失败')
  } finally {
    loading.value = false
  }
}

function renderChart(binStr, modType) {
  if (!chartRef.value) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(chartRef.value)

  // 用二进制流模拟简单波形（仅供可视化）
  const points = binStr.slice(0, 400).split('').map((b, i) => [i, parseInt(b)])

  chartInstance.setOption({
    animation: false,
    grid: { top: 10, bottom: 30, left: 40, right: 10 },
    xAxis: {
      type: 'value',
      name: '采样点',
      nameTextStyle: { fontSize: 11 },
      axisLabel: { fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      min: -0.2,
      max: 1.2,
      axisLabel: { fontSize: 10 }
    },
    series: [{
      type: 'line',
      data: points,
      symbol: 'none',
      lineStyle: { color: '#409eff', width: 1.5 },
      areaStyle: { color: 'rgba(64,158,255,0.08)' },
      step: 'end'
    }],
    tooltip: { trigger: 'axis', axisPointer: { type: 'line' } }
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
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 16px;
}
.info-item {
  background: #f8faff;
  border-radius: 10px;
  padding: 16px 20px;
}
.info-label {
  font-size: 12px;
  color: #999;
  margin-bottom: 6px;
}
.info-value {
  font-size: 16px;
  font-weight: 700;
  color: #333;
}
.info-value.accent { color: #409eff; }

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

.chart-wrap {
  margin-top: 16px;
  background: #f8faff;
  border-radius: 10px;
  padding: 12px;
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
