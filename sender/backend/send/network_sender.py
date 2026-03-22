import socket
import numpy as np

def send_modulated_signal(host='127.0.0.1', port=5244, signal_data=None, mod_type=None):
    """发送调制后的信号数据和调制类型（发送端核心函数）"""
    if signal_data is None:
        raise ValueError("信号数据不能为空")
    signal_bytes = signal_data.tobytes()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(mod_type.ljust(32).encode('utf-8'))
        s.sendall(len(signal_bytes).to_bytes(8, byteorder='big'))
        s.sendall(signal_bytes)
    print(f"已发送{mod_type}调制信号，长度：{len(signal_bytes)}字节")
