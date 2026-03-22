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
    """symbols: 符号列表 → 拼接后的二进制字符串"""
    return ''.join(symbols)

def am_demodulate(modulated_signal, fs=8000, duration=0.1):
    """AM解调：检测幅值"""
    bit_duration_samples = int(fs * duration)
    bin_str = ""
    global_amp = np.mean(abs(modulated_signal))
    threshold = 1.0
    
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        amp = np.mean(abs(bit_signal))
        bin_str += '1' if amp > threshold else '0'
    return bin_str

def fsk_demodulate(modulated_signal, f0=400, f1=1600, fs=8000, duration=0.1):
    """2FSK解调：二进制频移键控"""
    bit_duration_samples = int(fs * duration)
    bin_str = ""
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        fft_vals = fft(bit_signal)
        freqs = np.fft.fftfreq(len(bit_signal), 1/fs)
        peak_freq = abs(freqs[np.argmax(abs(fft_vals))])
        if (f1 - 100) <= peak_freq <= (f1 + 100):
            bin_str += '1'
        elif (f0 - 100) <= peak_freq <= (f0 + 100):
            bin_str += '0'
        else:
            bin_str += '0'
    return bin_str

def psk_demodulate(modulated_signal, fc=1000, fs=8000, duration=0.1):
    """BPSK解调：二进制相移键控"""
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
    """QPSK解调：相位判决→2bit符号"""
    bit_duration_samples = int(fs * duration)
    symbols = []
    t = np.linspace(0, duration, bit_duration_samples, endpoint=False)
    carrier_i = np.cos(2 * np.pi * fc * t)
    carrier_q = np.sin(2 * np.pi * fc * t)
    
    for i in range(0, len(modulated_signal), bit_duration_samples):
        bit_signal = modulated_signal[i:i+bit_duration_samples]
        if len(bit_signal) < bit_duration_samples:
            break
        i_val = np.sum(bit_signal * carrier_i)
        q_val = -np.sum(bit_signal * carrier_q)
        phase = np.arctan2(q_val, i_val)
        phase = phase if phase >= 0 else phase + 2*np.pi
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
    """16QAM解调：I/Q幅值判决→4bit符号"""
    bit_duration_samples = int(fs * duration)
    symbols = []
    t = np.linspace(0, duration, bit_duration_samples, endpoint=False)
    carrier_i = np.cos(2 * np.pi * fc * t)
    carrier_q = np.sin(2 * np.pi * fc * t)
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
        i_val = np.sum(bit_signal * carrier_i)
        q_val = -np.sum(bit_signal * carrier_q)
        i_quant = 3 if i_val > 2 else 1 if i_val > 0 else -1 if i_val > -2 else -3
        q_quant = 3 if q_val > 2 else 1 if q_val > 0 else -1 if q_val > -2 else -3
        symbol = qam16_inv_map.get((i_quant, q_quant), '0000')
        symbols.append(symbol)
    return symbols

def demodulate_signal(modulated_signal, mod_type, fs=8000, duration=0.1):
    """统一解调接口"""
    if mod_type in ["2FSK", "FSK"]:
        return fsk_demodulate(modulated_signal, 400, 1600, fs, duration)
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
