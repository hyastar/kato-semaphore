import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft, ifft

def text_to_bin(text, encoding='utf-8'):
    """将文本转为二进制字符串（如"我"→01100010...）"""
    bin_str = ''.join(format(byte, '08b') for byte in text.encode(encoding))
    return bin_str

def bin_to_symbols(bin_str, bits_per_symbol):
    """
    bin_str: 二进制字符串
    bits_per_symbol: 每个符号的bit数（2=QPSK，4=16QAM，1=AM/2FSK/BPSK）
    return: 符号列表（如QPSK返回['00','01','10','11']，16QAM返回4bit组）
    """
    pad_len = (bits_per_symbol - len(bin_str) % bits_per_symbol) % bits_per_symbol
    bin_str = bin_str.ljust(len(bin_str) + pad_len, '0')
    symbols = [bin_str[i:i+bits_per_symbol] for i in range(0, len(bin_str), bits_per_symbol)]
    return symbols

def generate_carrier(fc=1000, fs=8000, duration=0.1):
    """生成载波信号：fc=载波频率，fs=采样率，duration=单比特持续时间"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * fc * t)
    return t, carrier

def am_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    """AM调制（标准调幅，非通断键控）"""
    t, carrier = generate_carrier(fc, fs, duration)
    mod_coeff = 0.8
    if bin_bit == '1':
        amp = 1 + mod_coeff
    else:
        amp = 1 - mod_coeff
    modulated = amp * carrier
    return t, modulated

def fsk_modulate(bin_bit, f0=400, f1=1600, fs=8000, duration=0.1):
    """2FSK调制：二进制频移键控（1bit/符号）0→f0，1→f1"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if bin_bit == '1':
        modulated = np.cos(2 * np.pi * f1 * t)
    else:
        modulated = np.cos(2 * np.pi * f0 * t)
    return t, modulated

def psk_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    """BPSK调制：二进制相移键控（1bit/符号）0→0相位，1→π相位"""
    t, carrier = generate_carrier(fc, fs, duration)
    phase = np.pi if bin_bit == '1' else 0
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return t, modulated

def modulate_qpsk(symbol, fc=1000, fs=8000, duration=0.1):
    """QPSK调制：2bit符号→4个相位（格雷码映射，减少误码）"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    phase_map = {'00': 0, '01': np.pi/2, '11': np.pi, '10': 3*np.pi/2}
    phase = phase_map.get(symbol, 0)
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return t, modulated

def modulate_16qam(symbol, fc=1000, fs=8000, duration=0.1):
    """16QAM调制：4bit符号→16个星座点（标准矩形星座图）"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier_i = np.cos(2 * np.pi * fc * t)
    carrier_q = np.sin(2 * np.pi * fc * t)
    qam16_map = {
        '0000': (1, 1), '0001': (1, 3), '0011': (3, 3), '0010': (3, 1),
        '0100': (1, -1), '0101': (1, -3), '0111': (3, -3), '0110': (3, -1),
        '1100': (-1, -1), '1101': (-1, -3), '1111': (-3, -3), '1110': (-3, -1),
        '1000': (-1, 1), '1001': (-1, 3), '1011': (-3, 3), '1010': (-3, 1)
    }
    i_amp, q_amp = qam16_map.get(symbol, (1, 1))
    modulated = i_amp * carrier_i - q_amp * carrier_q
    return t, modulated

def modulate_signal(bin_str, mod_type="2FSK", fc=1000, fs=8000, duration=0.1):
    """统一调制接口：输入二进制字符串，输出调制后的完整信号"""
    if mod_type in ["AM", "2FSK", "BPSK", "FSK"]:
        bits_per_symbol = 1
    elif mod_type == "QPSK":
        bits_per_symbol = 2
    elif mod_type == "16QAM":
        bits_per_symbol = 4
    else:
        raise ValueError(f"不支持的调制类型：{mod_type}")
    
    symbols = bin_to_symbols(bin_str, bits_per_symbol)
    
    modulated_signal = []
    for symbol in symbols:
        if mod_type == "AM":
            t, sig = am_modulate(symbol, fc, fs, duration)
        elif mod_type in ["2FSK", "FSK"]:
            t, sig = fsk_modulate(symbol, 400, 1600, fs, duration)
        elif mod_type in ["BPSK", "PSK"]:
            t, sig = psk_modulate(symbol, fc, fs, duration)
        elif mod_type == "QPSK":
            t, sig = modulate_qpsk(symbol, fc, fs, duration)
        elif mod_type == "16QAM":
            t, sig = modulate_16qam(symbol, fc, fs, duration)
        else:
            raise ValueError(f"暂支持AM/2FSK/BPSK/QPSK/16QAM，不支持{mod_type}")
        modulated_signal.extend(sig)
    return np.array(modulated_signal), fs
