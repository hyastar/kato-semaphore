import sys
import os
import socket
import pickle
import struct
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

sys.path.append(__file__.rsplit('\\', 1)[0])

from receive.modem_utils import demodulate_signal, bin_to_text
from receive.recognition_utils import load_trained_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "mlp", "model")

received_signal = None
received_mod_type = None
clf, mod_types, scaler = None, None, None

def start_socket_listener():
    """Socket 监听线程（daemon=True），在 while True 循环外只 bind/listen 一次"""
    global received_signal, received_mod_type
    host = '127.0.0.1'
    port = 11244

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Socket 监听线程已启动，监听 {host}:{port}")

    while True:
        try:
            conn, addr = server_socket.accept()
            with conn:
                print(f"连接来自：{addr}")
                raw_len = conn.recv(4)
                if not raw_len:
                    continue
                payload_len = struct.unpack('>I', raw_len)[0]
                payload = b''
                while len(payload) < payload_len:
                    chunk = conn.recv(payload_len - len(payload))
                    if not chunk:
                        break
                    payload += chunk
                data = pickle.loads(payload)
                received_mod_type = data["mod_type"]
                received_signal = np.array(data["signal"])
                print(f"已接收信号：调制类型={received_mod_type}, 信号长度={len(received_signal)}")
        except Exception as e:
            print(f"接收错误：{e}")
            continue

@app.on_event("startup")
async def startup_event():
    global clf, mod_types, scaler
    try:
        clf, mod_types, scaler = load_trained_model(MODEL_DIR)
        listener_thread = threading.Thread(target=start_socket_listener, daemon=True)
        listener_thread.start()
    except Exception as e:
        print(f"启动警告：{e}")

@app.get("/receive_and_demodulate")
async def receive_and_demodulate():
    global received_signal, received_mod_type

    if received_signal is None:
        raise HTTPException(status_code=400, detail="暂无接收信号，请先发送")

    try:
        recognized_type, recognition_prob = None, None
        if clf is not None and mod_types is not None and scaler is not None:
            from receive.recognition_utils import recognize_modulation
            recognized_type, recognition_prob = recognize_modulation(
                received_signal, clf, mod_types, scaler
            )

        bin_str = demodulate_signal(received_signal, received_mod_type)
        text = bin_to_text(bin_str)

        return {
            "status": "success",
            "message": "接收并解调成功",
            "recognized_modulation_type": recognized_type,
            "recognition_probability": recognition_prob,
            "actual_modulation_type": received_mod_type,
            "binary_string": bin_str,
            "demodulated_text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11245)
