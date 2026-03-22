# Kato-Semaphore 信号调制与识别系统

基于 FastAPI + Vue3 的无线信号调制发送与接收识别系统，支持 AM、2FSK、BPSK、QPSK、16QAM 五种调制类型，集成 MLP 神经网络自动识别调制类型。

## 项目结构

```
kato-semaphore/
├── mlp/
│   ├── train/
│   │   └── train_mlp.py              # 训练调制识别 MLP 模型
│   └── model/
│       ├── modulation_classifier.pkl  # 训练好的 MLP 分类器
│       ├── feature_scaler.pkl         # 特征标准化器
│       ├── mod_types.pkl              # 调制类型标签
│       └── confusion_matrix.png       # 训练混淆矩阵
├── sender/
│   ├── backend/
│   │   ├── main.py                   # 发送端 FastAPI 服务 (:11001)
│   │   └── send/
│   │       ├── __init__.py
│   │       └── modem_utils.py        # 调制核心函数
│   └── frontend/
│       ├── src/
│       │   ├── main.js
│       │   └── App.vue               # 发送端 Vue3 页面
│       ├── index.html
│       ├── vite.config.js            # 代理到 :11001
│       └── package.json
├── receiver/
│   ├── backend/
│   │   ├── main.py                   # 接收端 FastAPI 服务 (:11245)
│   │   └── receive/
│   │       ├── __init__.py
│   │       ├── modem_utils.py        # 解调核心函数
│   │       ├── network_receiver.py   # Socket 接收
│   │       └── recognition_utils.py   # MLP 特征提取与识别
│   └── frontend/
│       ├── src/
│       │   ├── main.js
│       │   └── App.vue               # 接收端 Vue3 页面
│       ├── index.html
│       ├── vite.config.js            # 代理到 :11245
│       └── package.json
├── requirements.txt                  # Python 依赖
└── README.md
```

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| 发送端后端 | 11001 | FastAPI HTTP 接口 |
| 接收端后端 | 11245 | FastAPI HTTP 接口 |
| Socket 通信 | 11244 | 发送端→接收端信号传输 |
| 发送端前端 | 12001 | Vue3 开发服务器 |
| 接收端前端 | 12002 | Vue3 开发服务器 |

## 快速开始

### 环境准备（仅首次）

```bash
conda create -n kato-semaphore python=3.12 -y
conda activate kato-semaphore
pip install -r requirements.txt
```

### 1. 训练模型（仅首次，或重新训练时）

```bash
conda activate kato-semaphore
cd mlp/train
python train_mlp.py
```

模型自动保存到 `mlp/model/`，训练完成后可直接启动服务。

### 2. 安装前端依赖（仅首次）

```bash
cd sender/frontend
pnpm install

cd receiver/frontend
pnpm install
```

### 3. 启动服务（每次使用）

**必须按顺序启动，接收端后端最先启动**（需要提前监听 Socket 11244 端口）

终端1 — 接收端后端：
```bash
conda activate kato-semaphore
cd receiver/backend
uvicorn main:app --host 127.0.0.1 --port 11245
```

终端2 — 发送端后端：
```bash
conda activate kato-semaphore
cd sender/backend
uvicorn main:app --host 127.0.0.1 --port 11001
```

终端3 — 接收端前端：
```bash
cd receiver/frontend
pnpm dev
```

终端4 — 发送端前端：
```bash
cd sender/frontend
pnpm dev
```

### 4. 访问页面

- 发送端：http://localhost:12001
- 接收端：http://localhost:12002

## 使用流程

1. 先打开接收端页面 http://localhost:12002
   点击**接收并识别解调**，使接收端进入等待状态
2. 再打开发送端页面 http://localhost:12001 
   输入文本，选择调制类型，点击**发送信号**
3. 返回接收端页面查看识别结果、解调文本和各调制类型的概率分布

## 调制类型说明

| 类型 | 全称 | 输入限制 |
|------|------|----------|
| AM | 调幅 | 无 |
| 2FSK | 二进制频移键控 | 无 |
| BPSK | 二进制相移键控 | 无 |
| QPSK | 四进制相移键控 | 文本转二进制后长度须为 2 的倍数 |
| 16QAM | 正交振幅调制 | 文本转二进制后长度须为 4 的倍数 |

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + uvicorn |
| 前端框架 | Vue 3 + Vite + Element Plus |
| HTTP 客户端 | Axios |
| 信号处理 | NumPy + SciPy |
| 调制识别 | scikit-learn MLPClassifier |
| 模型持久化 | joblib |
| 包管理 | conda (Python) + pnpm (Node) |
