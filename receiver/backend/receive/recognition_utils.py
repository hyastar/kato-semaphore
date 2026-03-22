import numpy as np
import joblib
import os
from scipy.stats import moment
from scipy.fftpack import fft

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

    fft_vals = fft(signal_centered)
    freqs = np.fft.fftfreq(len(signal_centered), 1/8000)
    peak_freq = abs(freqs[np.argmax(abs(fft_vals))])

    fft_abs = abs(fft_vals)
    fft_sum = np.sum(fft_abs)
    if fft_sum > 1e-8:
        fft_abs_norm = fft_abs / fft_sum
        fft_abs_norm = fft_abs_norm[fft_abs_norm > 1e-8]
        spectral_entropy = -np.sum(fft_abs_norm * np.log2(fft_abs_norm))
    else:
        spectral_entropy = 0.0

    amp_mean = np.mean(np.abs(signal_centered))

    analytic_signal = np.fft.fft(signal_centered)
    carrier_band = (abs(freqs) > (peak_freq - 100)) & (abs(freqs) < (peak_freq + 100))
    analytic_signal[~carrier_band] = 0
    analytic_signal = np.fft.ifft(analytic_signal)
    phase_std = np.std(np.angle(analytic_signal))

    carrier_idx = np.argmin(abs(freqs - peak_freq))
    carrier_amp = abs(fft_vals[carrier_idx])
    carrier_amp_ratio = carrier_amp / (np.sum(fft_abs) + 1e-8)

    amp_std = np.std(np.abs(signal_centered))

    top2_idx = np.argsort(abs(fft_vals))[::-1][:2]
    top2_freq = abs(freqs[top2_idx])
    freq_diff = np.abs(top2_freq[0] - top2_freq[1])

    from scipy.signal import hilbert
    analytic_signal = hilbert(signal_centered)
    phase = np.unwrap(np.angle(analytic_signal))
    phase_diff = np.diff(phase)

    phase_jump_bpsk = np.sum(np.abs(phase_diff - np.pi) < 0.3)
    phase_jump_qpsk = np.sum(np.abs(phase_diff - np.pi/2) < 0.3) + np.sum(np.abs(phase_diff - 3*np.pi/2) < 0.3)
    phase_jump_ratio = phase_jump_bpsk / (phase_jump_qpsk + 1e-8)

    phase_mean = np.mean(phase)
    phase_median = np.median(phase)
    phase_rms = np.sqrt(np.mean(phase**2))

    phase_jump_count = np.sum(np.abs(phase_diff) > np.pi/2)
    phase_diff_mean = np.mean(np.abs(phase_diff))
    phase_diff_std = np.std(np.abs(phase_diff))

    # ===== 新增：基于解调星座点的特征（最能区分BPSK/QPSK）=====
    # 对信号做I/Q解调，统计星座点的分布
    t_sig = np.arange(len(signal_centered)) / 8000
    # 用peak_freq作为载波估计频率
    i_demod = signal_centered * np.cos(2 * np.pi * peak_freq * t_sig)
    q_demod = signal_centered * (-np.sin(2 * np.pi * peak_freq * t_sig))

    # 低通滤波（简单滑动平均，窗口=单符号长度800点）
    win = 800
    i_lp = np.convolve(i_demod, np.ones(win)/win, mode='same')
    q_lp = np.convolve(q_demod, np.ones(win)/win, mode='same')

    # 取中间稳定段（去掉首尾各25%的边缘效应）
    n = len(i_lp)
    i_stable = i_lp[n//4 : 3*n//4]
    q_stable = q_lp[n//4 : 3*n//4]

    # 特征1：Q支路功率占比（BPSK的Q≈0，QPSK的Q有实质功率）
    q_power_ratio = np.var(q_stable) / (np.var(i_stable) + np.var(q_stable) + 1e-8)

    # 特征2：I/Q联合分布的聚类数估计（BPSK=2簇，QPSK=4簇）
    # 用I和Q各自的过零率来估计
    i_signs = np.sign(i_stable)
    q_signs = np.sign(q_stable)
    i_zero_cross = np.sum(np.diff(i_signs) != 0)
    q_zero_cross = np.sum(np.diff(q_signs) != 0)
    iq_cross_ratio = q_zero_cross / (i_zero_cross + 1e-8)

    # 特征3：Q支路的绝对值均值（BPSK的Q接近0，QPSK的Q有明显均值）
    q_abs_mean = np.mean(np.abs(q_stable))
    i_abs_mean = np.mean(np.abs(i_stable))
    qi_abs_ratio = q_abs_mean / (i_abs_mean + 1e-8)

    # ===== 新增：星座点聚类数估计（BPSK=2簇，QPSK=4簇，核心区分特征）=====
    # 对I/Q稳定段做归一化，然后统计落在4个象限的点数
    i_norm = i_stable / (np.std(i_stable) + 1e-8)
    q_norm = q_stable / (np.std(q_stable) + 1e-8)

    # 四象限点数（BPSK只用I/III象限，QPSK四象限均匀）
    q1 = np.sum((i_norm > 0) & (q_norm > 0))   # 第一象限
    q2 = np.sum((i_norm < 0) & (q_norm > 0))   # 第二象限
    q3 = np.sum((i_norm < 0) & (q_norm < 0))  # 第三象限
    q4 = np.sum((i_norm > 0) & (q_norm < 0))  # 第四象限
    total = q1 + q2 + q3 + q4 + 1e-8

    # BPSK: q1+q4≈1, q2+q3≈0；QPSK: 四象限均匀≈0.25各
    q_diag_ratio = (q1 + q3) / total        # 对角象限比（BPSK的I/III象限多）
    q_anti_ratio = (q2 + q4) / total        # 反对角象限比（QPSK的II/IV象限也多）
    q_balance = min(q1,q2,q3,q4) / (max(q1,q2,q3,q4) + 1e-8)  # 均衡度（QPSK接近1）

    # Q支路方差与I支路方差的比（BPSK的Q支路几乎无变化，QPSK的Q和I方差相近）
    q_var = np.var(q_norm)
    i_var = np.var(i_norm)
    var_ratio = q_var / (i_var + 1e-8)      # BPSK≈0，QPSK≈1

    # I/Q相关系数（BPSK的I和Q不相关但Q近似0常数，QPSK的I和Q正交）
    iq_corr = np.corrcoef(i_norm, q_norm)[0, 1] if len(i_norm) > 1 else 0.0

    # 星座点到原点距离的标准差（BPSK距离集中在两个值，QPSK距离更均匀）
    dist = np.sqrt(i_norm**2 + q_norm**2)
    dist_std = np.std(dist)
    dist_mean = np.mean(dist)
    dist_cv = dist_std / (dist_mean + 1e-8)  # 变异系数

    # 相位跳变幅度直方图（保留mid和high，删除low）
    abs_phase_diff = np.abs(phase_diff)
    hist_bins = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
    hist, _ = np.histogram(abs_phase_diff, bins=hist_bins, density=True)
    phase_hist_mid = float(hist[2])
    phase_hist_high = float(hist[3])

    return [var, norm_m4, c4, skewness, kurtosis, peak_freq, spectral_entropy,
            amp_mean, amp_std, phase_std, carrier_amp_ratio, freq_diff,
            phase_jump_count, phase_diff_mean, phase_diff_std,
            phase_jump_bpsk, phase_jump_qpsk, phase_jump_ratio, phase_rms,
            phase_hist_mid, phase_hist_high,
            q_power_ratio, iq_cross_ratio, qi_abs_ratio,
            q_diag_ratio, q_anti_ratio, q_balance, var_ratio,
            float(iq_corr), dist_cv, dist_mean]

def load_trained_model(model_dir="."):
    """加载本地训练好的MLP模型和scaler"""
    try:
        clf = joblib.load(os.path.join(model_dir, "modulation_classifier.pkl"))
        mod_types = joblib.load(os.path.join(model_dir, "mod_types.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
        print("成功加载训练好的MLP模型和scaler！")
        return clf, mod_types, scaler
    except FileNotFoundError as e:
        print(f"未找到训练好的模型或scaler: {e}")
        print("请先运行 mlp/train/train_mlp.py 训练模型！")
        raise

def recognize_modulation(signal, clf, mod_types, scaler):
    """识别信号的调制类型 + 返回识别概率"""
    # 统一取信号中间12800点，与训练数据长度一致
    target_len = 12800
    sig_len = len(signal)
    if sig_len >= target_len:
        start = (sig_len - target_len) // 2
        signal_crop = signal[start: start + target_len]
    else:
        # 信号太短则重复填充
        repeats = target_len // sig_len + 1
        signal_crop = np.tile(signal, repeats)[:target_len]

    features = extract_hos_features(signal_crop)
    features_scaled = scaler.transform([features])
    label = clf.predict(features_scaled)[0]
    mod_name = mod_types[label]
    prob = clf.predict_proba(features_scaled)[0]
    prob_dict = {mod_types[i]: round(prob[i], 4) for i in range(len(mod_types))}
    return mod_name, prob_dict
