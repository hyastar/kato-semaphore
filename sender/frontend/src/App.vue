<template>
  <div class="page">
    <el-card class="box">
      <template #header>
        <span class="title">📤 信号发送端</span>
      </template>

      <el-form label-width="90px">
        <el-form-item label="输入文本">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="请输入要发送的文本"
          />
        </el-form-item>

        <el-form-item label="调制类型">
          <el-select v-model="modType" style="width: 240px">
            <el-option label="AM 调幅" value="AM" />
            <el-option label="2FSK 频移键控" value="2FSK" />
            <el-option label="BPSK 二进制相移键控" value="BPSK" />
            <el-option label="QPSK 四进制相移键控" value="QPSK" />
            <el-option label="16QAM 正交振幅调制" value="16QAM" />
          </el-select>
        </el-form-item>

        <el-form-item label="说明">
          <el-text type="info" size="small">
            QPSK：二进制长度需为 2 的倍数；16QAM：需为 4 的倍数
          </el-text>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" :loading="loading" @click="send">
            发送信号
          </el-button>
        </el-form-item>
      </el-form>

      <el-alert
        v-if="result"
        :title="result.message"
        :type="result.status === 'success' ? 'success' : 'error'"
        show-icon
        style="margin-top: 16px"
      />

      <el-descriptions
        v-if="result?.status === 'success'"
        :column="1"
        border
        style="margin-top: 16px"
      >
        <el-descriptions-item label="调制类型">{{ result.modulation_type }}</el-descriptions-item>
        <el-descriptions-item label="二进制流">
          <el-text style="word-break: break-all">{{ result.binary_string }}</el-text>
        </el-descriptions-item>
        <el-descriptions-item label="信号长度">{{ result.signal_length }} 采样点</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const inputText = ref('')
const modType = ref('2FSK')
const loading = ref(false)
const result = ref(null)

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
  width: 640px;
}
.title {
  font-size: 18px;
  font-weight: 600;
}
</style>
