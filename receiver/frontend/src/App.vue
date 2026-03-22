<template>
  <div class="page">
    <el-card class="box">
      <template #header>
        <span class="title">📥 信号接收端</span>
      </template>

      <el-button type="primary" :loading="loading" @click="receive">
        接收并识别解调
      </el-button>

      <template v-if="result">
        <el-alert
          :title="result.message"
          :type="result.status === 'success' ? 'success' : 'error'"
          show-icon
          style="margin-top: 20px"
        />

        <el-descriptions
          v-if="result.status === 'success'"
          :column="1"
          border
          style="margin-top: 16px"
        >
          <el-descriptions-item label="解调文本">
            {{ result.demodulated_text }}
          </el-descriptions-item>
          <el-descriptions-item label="实际调制类型">
            {{ formatMod(result.actual_modulation_type) }}
          </el-descriptions-item>
          <el-descriptions-item label="MLP 识别类型">
            {{ formatMod(result.recognized_modulation_type) }}
          </el-descriptions-item>
          <el-descriptions-item label="识别概率">
            <div v-for="(prob, key) in result.recognition_probability" :key="key">
              <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
                <span style="width:200px">{{ formatMod(key) }}</span>
                <el-progress
                  :percentage="+(prob * 100).toFixed(2)"
                  :stroke-width="10"
                  style="flex:1"
                />
              </div>
            </div>
          </el-descriptions-item>
          <el-descriptions-item label="二进制流">
            <el-text style="word-break:break-all">{{ result.binary_string }}</el-text>
          </el-descriptions-item>
        </el-descriptions>
      </template>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const loading = ref(false)
const result = ref(null)

const modMap = {
  AM: 'AM 调幅',
  '2FSK': '2FSK 频移键控',
  BPSK: 'BPSK 二进制相移键控',
  QPSK: 'QPSK 四进制相移键控',
  '16QAM': '16QAM 正交振幅调制'
}
const formatMod = key => modMap[key] ?? key

async function receive() {
  loading.value = true
  result.value = null
  try {
    const { data } = await axios.get('/api/receive_and_demodulate')
    result.value = data
    ElMessage.success('接收识别成功')
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
</script>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 48px 16px;
  background: #f5f7fa;
}
.box {
  width: 720px;
}
.title {
  font-size: 18px;
  font-weight: 600;
}
</style>
