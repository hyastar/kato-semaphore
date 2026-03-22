import numpy as np
import joblib  # 新增：加载模型
from scipy.stats import moment
from scipy.fftpack import fft

# ===================== 高阶统计量+频谱特征提取（与train_mlp.py一致） =====================
def extract_hos_features(signal):
    """优化版：增强AM/PSK区分特征"""
    signal_centered = signal - np.mean(signal)
    var = np.var(signal_centered)
    if var == 0:
        var = 1e-8
    m4 = moment(signal_centered, moment=4)
    norm_m4 = m4 / (var **2)
    c4 = m4 - 3 * (var** 2)
    skewness = moment(signal_centered, moment=3) / (var ** 1.5)
    kurtosis = norm_m4 - 3

    # 原有频谱特征
    fft_vals = fft(signal_centered)
    freqs = np.fft.fftfreq(len(signal_centered), 1/8000)
    peak_freq = abs(freqs[np.argmax(abs(fft_vals))])
    
    # 频谱熵（保留）
    fft_abs = abs(fft_vals)
    fft_sum = np.sum(fft_abs)
    if fft_sum > 1e-8:
        fft_abs_norm = fft_abs / fft_sum
        fft_abs_norm = fft_abs_norm[fft_abs_norm > 1e-8]
        spectral_entropy = -np.sum(fft_abs_norm * np.log2(fft_abs_norm))
    else:
        spectral_entropy = 0.0

    # ========== 优化/新增特征 ==========
    # 1. 幅值均值（AM特有：bit=1时均值大，bit=0时均值小；PSK均值稳定）
    amp_mean = np.mean(np.abs(signal_centered))
    
    # 2. 优化相位标准差（去除噪声干扰，仅计算载波频率附近相位）
    analytic_signal = np.fft.fft(signal_centered)
    # 仅保留载波频率±100Hz范围的相位（聚焦有效信号）
    carrier_band = (abs(freqs) > (peak_freq - 100)) & (abs(freqs) < (peak_freq + 100))
    analytic_signal[~carrier_band] = 0  
    analytic_signal = np.fft.ifft(analytic_signal)
    phase_std = np.std(np.angle(analytic_signal))
    
    # 3. 载波幅值占比（AM的载波幅值随bit变化，PSK载波幅值稳定）
    carrier_idx = np.argmin(abs(freqs - peak_freq))  # 载波频率索引
    carrier_amp = abs(fft_vals[carrier_idx])
    carrier_amp_ratio = carrier_amp / (np.sum(fft_abs) + 1e-8)  # 归一化
    
    # 4. 原有幅值标准差（保留）
    amp_std = np.std(np.abs(signal_centered))
    
    # 5. 原有双频峰差值（保留，FSK特有）
    top2_idx = np.argsort(abs(fft_vals))[::-1][:2]
    top2_freq = abs(freqs[top2_idx])
    freq_diff = np.abs(top2_freq[0] - top2_freq[1])

    # ========== 新增：相位差分特征（针对BPSK/QPSK优化） ==========
    # 使用Hilbert变换获取更稳定的解析信号和相位
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal_centered)
    phase = np.unwrap(np.angle(analytic_signal))  # 相位解缠绕（避免π跳变被截断）
    phase_diff = np.diff(phase)
    
    # 1. 相位跳变特征（区分BPSK/QPSK）
    # BPSK：跳变≈π；QPSK：跳变≈π/2；其他调制：无固定跳变
    phase_jump_bpsk = np.sum(np.abs(phase_diff - np.pi) < 0.1)  # π跳变次数
    phase_jump_qpsk = np.sum(np.abs(phase_diff - np.pi/2) < 0.1) + np.sum(np.abs(phase_diff - 3*np.pi/2) < 0.1)  # π/2/3π/2跳变次数
    phase_jump_ratio = phase_jump_bpsk / (phase_jump_qpsk + 1e-8)  # 跳变比（BPSK>>1，QPSK<<1）
    
    # 2. 相位统计特征（增强鲁棒性）
    phase_mean = np.mean(phase)
    phase_median = np.median(phase)
    phase_rms = np.sqrt(np.mean(phase**2))  # 相位均方根
    
    # 3. 原有相位差分特征（保留）
    phase_jump_count = np.sum(np.abs(phase_diff) > np.pi/2)
    phase_diff_mean = np.mean(np.abs(phase_diff))
    phase_diff_std = np.std(np.abs(phase_diff))

    # 返回特征：原有12个 + 新增7个 = 19个特征
    return [var, norm_m4, c4, skewness, kurtosis, peak_freq, spectral_entropy,
            amp_mean, amp_std, phase_std, carrier_amp_ratio, freq_diff,
            phase_jump_count, phase_diff_mean, phase_diff_std,
            phase_jump_bpsk, phase_jump_qpsk, phase_jump_ratio, phase_rms]

# ===================== 加载训练好的模型（替换模拟训练） =====================
def load_trained_model():
    """加载本地训练好的MLP模型和scaler"""
    try:
        clf = joblib.load("modulation_classifier.pkl")
        mod_types = joblib.load("mod_types.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        print("成功加载训练好的MLP模型和scaler！")
        return clf, mod_types, scaler
    except FileNotFoundError as e:
        print(f"未找到训练好的模型或scaler: {e}")
        print("请先运行train_mlp.py训练模型！")
        raise

# ===================== 识别函数（新增scaler参数，使用标准化） =====================
def recognize_modulation(signal, clf, mod_types, scaler):
    """识别信号的调制类型 + 返回识别概率"""
    # 提取特征
    features = extract_hos_features(signal)
    # 特征标准化
    features_scaled = scaler.transform([features])
    # 预测调制类型标签
    label = clf.predict(features_scaled)[0]
    mod_name = mod_types[label]
    # 预测概率（获取每个调制类型的置信度）
    prob = clf.predict_proba(features_scaled)[0]
    # 构造概率字典：{调制类型: 概率值}
    prob_dict = {mod_types[i]: round(prob[i], 4) for i in range(len(mod_types))}
    # 返回调制类型 + 概率字典
    return mod_name, prob_dict

# 测试识别功能（可选）
if __name__ == "__main__":
    try:
        # 加载训练好的模型
        clf, mod_types, scaler = load_trained_model()
        # 模拟一个信号（这里仅做示例）
        test_signal = np.random.randn(800)  # 800个采样点，对应0.1秒@8000Hz
        # 识别
        mod_name, prob = recognize_modulation(test_signal, clf, mod_types, scaler)
        print(f"识别结果：{mod_name}")
        print(f"概率：{prob}")
    except Exception as e:
        print(f"测试失败：{e}")
