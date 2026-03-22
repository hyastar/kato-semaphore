import socket
import numpy as np

def send_modulated_signal(host='127.0.0.1', port=5244, signal_data=None, mod_type=None):
    """发送调制后的信号数据和调制类型（发送端核心函数）"""
    if signal_data is None:
        raise ValueError("信号数据不能为空")
    # 将numpy数组转为字节流
    signal_bytes = signal_data.tobytes()
    # 先发送调制类型（定长32字节），再发送信号长度，最后发送信号数据
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # 发送调制类型
        s.sendall(mod_type.ljust(32).encode('utf-8'))
        # 发送信号长度
        s.sendall(len(signal_bytes).to_bytes(8, byteorder='big'))
        # 发送信号数据
        s.sendall(signal_bytes)
    print(f"已发送{mod_type}调制信号，长度：{len(signal_bytes)}字节")

# 测试发送端（可选）
if __name__ == "__main__":
    from modem_utils import text_to_bin, modulate_signal
    # 测试文本
    text = "我爱你"
    # 文本转二进制
    bin_str = text_to_bin(text)
    # 调制信号（FSK）
    signal, fs = modulate_signal(bin_str, mod_type="FSK")
    # 发送信号
    send_modulated_signal(signal_data=signal, mod_type="FSK")
