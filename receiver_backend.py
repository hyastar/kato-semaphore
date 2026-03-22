from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import sys
import os
import socket
import pickle
import threading
import time

# 添加receive目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "receive"))
from receive.modem_utils import demodulate_signal, bin_to_text
from receive.recognition_utils import recognize_modulation, load_trained_model

# 初始化FastAPI（接收端）
app = FastAPI(title="信号接收端服务", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量：存储接收的信号和调制类型
received_signal = None
received_mod_type = None

# 加载MLP识别模型
clf, mod_types, scaler = load_trained_model()

# Socket监听线程（接收发送端信号）
def start_socket_listener():
    global received_signal, received_mod_type
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 5244))
        s.listen()
        print(f"接收端Socket监听启动：127.0.0.1:5244")
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    # 接收数据长度
                    len_data = conn.recv(4)
                    if not len_data:
                        continue
                    data_len = int.from_bytes(len_data, byteorder='big')
                    # 接收完整数据
                    pickle_data = b""
                    while len(pickle_data) < data_len:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        pickle_data += chunk
                    # 反序列化
                    data = pickle.loads(pickle_data)
                    mod_type = data["mod_type"]
                    signal = np.array(data["signal"])
                    # 更新全局变量
                    received_signal = signal
                    received_mod_type = mod_type
                    print(f"接收成功：{mod_type}调制信号（长度{len(signal)}）")
            except Exception as e:
                print(f"Socket监听异常：{str(e)}")
                time.sleep(1)
                continue

# 启动Socket监听线程（守护线程）
listener_thread = threading.Thread(target=start_socket_listener, daemon=True)
listener_thread.start()

# 解调+识别接口
@app.get("/receive_and_demodulate")
async def receive_and_demodulate():
    try:
        global received_signal, received_mod_type
        if received_signal is None:
            raise HTTPException(status_code=400, detail="暂无接收的信号，请先发送信号")
        
        # 1. 识别调制类型+概率
        recognized_mod_type, prob_dict = recognize_modulation(received_signal, clf, mod_types, scaler)
        # 2. 解调信号
        bin_str = demodulate_signal(received_signal, recognized_mod_type)
        # 3. 二进制转文本
        demodulated_text = bin_to_text(bin_str)
        
        return {
            "status": "success",
            "message": "信号接收、识别、解调成功",
            "recognized_modulation_type": recognized_mod_type,
            "recognition_probability": prob_dict,
            "actual_modulation_type": received_mod_type,
            "binary_string": bin_str,
            "demodulated_text": demodulated_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解调失败：{str(e)}")

# 启动接收端（API监听5245端口，避免和Socket的5244冲突）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5245, reload=True)
