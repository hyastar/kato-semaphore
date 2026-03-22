import socket
import numpy as np
import pickle
import struct

def receive_modulated_signal(host='127.0.0.1', port=5244):
    """接收调制后的信号数据和调制类型（使用pickle序列化）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"接收端已启动，监听{host}:{port}...")
        conn, addr = s.accept()
        with conn:
            print(f"连接来自：{addr}")
            raw_len = conn.recv(4)
            if not raw_len:
                return None, None
            payload_len = struct.unpack('>I', raw_len)[0]
            payload = b''
            while len(payload) < payload_len:
                chunk = conn.recv(payload_len - len(payload))
                if not chunk:
                    break
                payload += chunk
            data = pickle.loads(payload)
            mod_type = data["mod_type"]
            signal_data = np.array(data["signal"])
    return mod_type, signal_data
