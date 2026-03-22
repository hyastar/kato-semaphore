import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft, ifft

def bin_to_text(bin_str, encoding='utf-8'):
    """二进制字符串转回文本"""
    bin_str = bin_str.ljust(len(bin_str) + (8 - len(bin_str) % 8) % 8, '0')
    bytes_list = [int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8)]
    try:
        return bytes(bytes_list).decode(encoding, errors='ignore')
    except:
        return "解码失败"

def symbols_to_bin(symbols):
    """
    symbols: 符号列表（如QPSK的['00','01'], 16QAM的['0000','0001']）
    return: 拼接后的二进制字符串
    """
    return ''.join(symbols)

# 优化AM解调：适配标准调幅（bit=0时幅值0.2，bit=1时幅值1.8）
def am_demodulate(modulated_signal, fs=8000, duration=0.1):
    """AM解调：检测幅值（适配标准调幅）"""
    bit_duration_samples = int(fs * duration)
    bin_str = ""
    # 计算信号整体幅值（用于动态阈值）
    global_amp = np.mean(abs(modulated_signal))
    # 标准调幅：bit=0时幅值≈0.2，bit=1时幅值≈1.8
    # 阈值设置为中间值（约1.0），区分0和1
    threshold = 1.0  # 标准调幅的中间阈值
    
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        # 计算当前比特的平均幅值
        amp = np.mean(abs(bit_signal))
        # 判断逻辑：幅值>1.0为1，否则为0
        bin_str += '1' if amp > threshold else '0'
    return bin_str

# FSK解调（微调频率判断）
def fsk_demodulate(modulated_signal, f0=800, f1=1200, fs=8000, duration=0.1):
    """2FSK解调：二进制频移键控（1bit/符号）"""
    bit_duration_samples = int(fs * duration)
    bin_str = ""
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        fft_vals = fft(bit_signal)
        freqs = np.fft.fftfreq(len(bit_signal), 1/fs)
        peak_freq = abs(freqs[np.argmax(abs(fft_vals))])
        # 增加频率容差（±50Hz）
        if (f1 - 50) <= peak_freq <= (f1 + 50):
            bin_str += '1'
        elif (f0 - 50) <= peak_freq <= (f0 + 50):
            bin_str += '0'
        else:
            bin_str += '0'  # 容错：默认0
    return bin_str

# PSK解调（不变）
def psk_demodulate(modulated_signal, fc=1000, fs=8000, duration=0.1):
    """BPSK解调：二进制相移键控（1bit/符号）"""
    bit_duration_samples = int(fs * duration)
    bin_str = ""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * fc * t)
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        corr = np.sum(bit_signal * carrier)
        bin_str += '1' if corr < 0 else '0'
    return bin_str

def qpsk_demodulate(modulated_signal, fc=1000, fs=8000, duration=0.1):
    """
    QPSK解调：相位判决→2bit符号
    判决逻辑：
    相位∈[-45°,45°] → 00；45°-135°→01；135°-225°→11；225°-315°→10
    """
    bit_duration_samples = int(fs * duration)
    symbols = []
    t = np.linspace(0, duration, bit_duration_samples, endpoint=False)
    carrier_i = np.cos(2 * np.pi * fc * t)  # I支路载波
    carrier_q = np.sin(2 * np.pi * fc * t)  # Q支路载波
    
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        # 相干解调：I/Q支路积分
        i_val = np.sum(bit_signal * carrier_i)
        q_val = -np.sum(bit_signal * carrier_q)
        # 相位计算（atan2(Q,I)）
        phase = np.arctan2(q_val, i_val)
        phase = phase if phase >= 0 else phase + 2*np.pi  # 转0-2π
        # 相位判决（格雷码映射）
        if (phase >= 7*np.pi/4) or (phase < np.pi/4):
            symbol = '00'
        elif phase < 3*np.pi/4:
            symbol = '01'
        elif phase < 5*np.pi/4:
            symbol = '11'
        else:
            symbol = '10'
        symbols.append(symbol)
    return symbols

def qam16_demodulate(modulated_signal, fc=1000, fs=8000, duration=0.1):
    """
    16QAM解调：I/Q幅值判决→4bit符号
    判决逻辑：
    I幅值∈(-∞,-2)→-3, (-2,0)→-1, (0,2)→1, (2,+∞)→3；
    Q幅值判决规则同I；
    再根据(I,Q)映射回4bit符号
    """
    bit_duration_samples = int(fs * duration)
    symbols = []
    t = np.linspace(0, duration, bit_duration_samples, endpoint=False)
    carrier_i = np.cos(2 * np.pi * fc * t)
    carrier_q = np.sin(2 * np.pi * fc * t)
    # 16QAM逆映射表（I/Q→4bit符号）
    qam16_inv_map = {
        (1, 1): '0000', (1, 3): '0001', (3, 3): '0011', (3, 1): '0010',
        (1, -1): '0100', (1, -3): '0101', (3, -3): '0111', (3, -1): '0110',
        (-1, -1): '1100', (-1, -3): '1101', (-3, -3): '1111', (-3, -1): '1110',
        (-1, 1): '1000', (-1, 3): '1001', (-3, 3): '1011', (-3, 1): '1010'
    }
    
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        # 相干解调：计算I/Q幅值
        i_val = np.sum(bit_signal * carrier_i)
        q_val = -np.sum(bit_signal * carrier_q)
        # 幅值判决（量化为±1/±3）
        i_quant = 3 if i_val > 2 else 1 if i_val > 0 else -1 if i_val > -2 else -3
        q_quant = 3 if q_val > 2 else 1 if q_val > 0 else -1 if q_val > -2 else -3
        # 映射为4bit符号
        symbol = qam16_inv_map.get((i_quant, q_quant), '0000')
        symbols.append(symbol)
    return symbols

def demodulate_signal(modulated_signal, mod_type, fs=8000, duration=0.1):
    """统一解调接口"""
    if mod_type in ["2FSK", "FSK"]:
        return fsk_demodulate(modulated_signal, 800, 1200, fs, duration)
    elif mod_type == "AM":
        return am_demodulate(modulated_signal, fs, duration)
    elif mod_type in ["BPSK", "PSK"]:
        return psk_demodulate(modulated_signal, 1000, fs, duration)
    elif mod_type == "QPSK":
        symbols = qpsk_demodulate(modulated_signal, 1000, fs, duration)
        return symbols_to_bin(symbols)
    elif mod_type == "16QAM":
        symbols = qam16_demodulate(modulated_signal, 1000, fs, duration)
        return symbols_to_bin(symbols)
    else:
        raise ValueError(f"暂支持AM/2FSK/BPSK/QPSK/16QAM，不支持{mod_type}")
