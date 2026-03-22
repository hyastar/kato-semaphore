import socket
import numpy as np

def receive_modulated_signal(host='127.0.0.1', port=5244):
    """接收调制后的信号数据和调制类型（接收端核心函数）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"接收端已启动，监听{host}:{port}...")
        conn, addr = s.accept()
        with conn:
            print(f"连接来自：{addr}")
            # 接收调制类型（32字节）
            mod_type = conn.recv(32).decode('utf-8').strip()
            # 接收信号长度
            len_bytes = conn.recv(8)
            signal_len = int.from_bytes(len_bytes, byteorder='big')
            # 接收信号数据（分块接收，避免数据丢失）
            signal_bytes = b''
            while len(signal_bytes) < signal_len:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                signal_bytes += chunk
            # 转回numpy数组
            signal_data = np.frombuffer(signal_bytes, dtype=np.float64)
    return mod_type, signal_data

# 测试接收端（可选）
if __name__ == "__main__":
    from modem_utils import demodulate_signal, bin_to_text
    # 接收信号
    mod_type, signal = receive_modulated_signal()
    # 解调信号
    bin_str = demodulate_signal(signal, mod_type)
    # 二进制转文本
    text = bin_to_text(bin_str)
    print(f"解调后的文本：{text}")
