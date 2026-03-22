from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import socket
import pickle

# 添加send目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "send"))
from send.modem_utils import text_to_bin, modulate_signal

# 初始化FastAPI（发送端）
app = FastAPI(title="信号发送端服务", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origors=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 接收端Socket配置
RECEIVER_HOST = "127.0.0.1"
RECEIVER_SOCKET_PORT = 5244

# 请求体模型
class ModulateRequest(BaseModel):
    text: str
    mod_type: str = "2FSK"

# 发送信号到接收端Socket
def send_to_receiver(signal_data, mod_type):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RECEIVER_HOST, RECEIVER_SOCKET_PORT))
            # 序列化信号数据（numpy数组转列表）
            data = {
                "mod_type": mod_type,
                "signal": signal_data.tolist()
            }
            pickle_data = pickle.dumps(data)
            # 先发送数据长度，再发送数据（避免粘包）
            s.sendall(len(pickle_data).to_bytes(4, byteorder='big'))
            s.sendall(pickle_data)
        return True
    except Exception as e:
        raise Exception(f"发送到接收端失败：{str(e)}")

# 调制+发送接口
@app.post("/modulate_and_send")
async def modulate_and_send(req: ModulateRequest):
    try:
        # 1. 文本转二进制
        bin_str = text_to_bin(req.text)
        # 2. 调制信号
        modulated_signal, fs = modulate_signal(bin_str, req.mod_type)
        # 3. 发送到接收端
        send_to_receiver(modulated_signal, req.mod_type)
        return {
            "status": "success",
            "message": "信号调制并发送到接收端成功",
            "input_text": req.text,
            "binary_string": bin_str,
            "modulation_type": req.mod_type,
            "signal_length": len(modulated_signal)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送失败：{str(e)}")

# 启动发送端（API监听8001端口）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
