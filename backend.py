from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import threading
import sys
import os
# 新增：跨域支持
from fastapi.middleware.cors import CORSMiddleware

# 添加send和receive目录到Python路径，确保能导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), "send"))
sys.path.append(os.path.join(os.path.dirname(__file__), "receive"))

# 导入发送端/接收端核心模块
from send.modem_utils import text_to_bin, modulate_signal
from send.network_sender import send_modulated_signal
from receive.network_receiver import receive_modulated_signal
from receive.modem_utils import demodulate_signal, bin_to_text
from receive.recognition_utils import recognize_modulation, load_trained_model

# 初始化FastAPI应用
app = FastAPI(title="通信信号调制识别系统", version="1.0")

# 新增：配置跨域（允许前端网页调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法
    allow_headers=["*"],  # 允许所有请求头
)

# 加载训练好的MLP模型和scaler（优先）
clf, mod_types, scaler = load_trained_model()

# 全局变量：存储接收的信号和调制类型（多线程共享）
received_signal = None
received_mod_type = None

# 定义请求体模型（FastAPI接口参数）
class ModulateRequest(BaseModel):
    text: str
    mod_type: str = "2FSK"

# ===================== 接收端线程（后台运行） =====================
def start_receiver_thread():
    """启动接收端线程，持续监听信号"""
    global received_signal, received_mod_type
    while True:
        try:
            # 接收信号
            mod_type, signal = receive_modulated_signal()
            # 更新全局变量
            received_signal = signal
            received_mod_type = mod_type
            print(f"后台接收线程：已接收{mod_type}调制信号")
        except Exception as e:
            print(f"接收线程异常：{e}")
            continue

# 启动接收端后台线程（守护线程，关闭后端时自动退出）
receiver_thread = threading.Thread(target=start_receiver_thread, daemon=True)
receiver_thread.start()

# ===================== API接口 =====================
@app.post("/modulate_and_send")
async def modulate_and_send(req: ModulateRequest):
    """接口1：调制文本并发送信号"""
    try:
        # 1. 文本转二进制
        bin_str = text_to_bin(req.text)
        # 2. 调制信号
        modulated_signal, fs = modulate_signal(bin_str, req.mod_type)
        # 3. 发送信号
        send_modulated_signal(signal_data=modulated_signal, mod_type=req.mod_type)
        # 返回结果
        return {
            "status": "success",
            "message": "信号调制并发送成功",
            "input_text": req.text,
            "binary_string": bin_str,
            "modulation_type": req.mod_type,
            "signal_length": len(modulated_signal)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送失败：{str(e)}")

@app.get("/receive_and_demodulate")
async def receive_and_demodulate():
    """接口2：接收信号、识别调制类型、解调并还原文本"""
    try:
        global received_signal, received_mod_type
        # 检查是否有接收的信号
        if received_signal is None:
            raise HTTPException(status_code=400, detail="暂无接收的信号，请先发送信号")
        
        # 1. 识别调制类型 + 概率
        recognized_mod_type, prob_dict = recognize_modulation(received_signal, clf, mod_types, scaler)
        # 2. 解调信号
        bin_str = demodulate_signal(received_signal, recognized_mod_type)
        # 3. 二进制转文本
        demodulated_text = bin_to_text(bin_str)
        
        # 返回结果
        return {
            "status": "success",
            "message": "信号接收、识别、解调成功",
            "recognized_modulation_type": recognized_mod_type,
            "recognition_probability": prob_dict,  # 新增：概率字典
            "actual_modulation_type": received_mod_type,
            "binary_string": bin_str,
            "demodulated_text": demodulated_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解调失败：{str(e)}")

# 启动命令（在PowerShell中执行）：uvicorn backend:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
