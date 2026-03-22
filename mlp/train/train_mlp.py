import numpy as np
import joblib
from scipy.stats import moment
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
import os

def generate_carrier(fc=1000, fs=8000, duration=0.1):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * fc * t)
    return t, carrier

def am_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    t, carrier = generate_carrier(fc, fs, duration)
    mod_coeff = 0.8
    if bin_bit == '1':
        amp = 1 + mod_coeff
    else:
        amp = 1 - mod_coeff
    modulated = amp * carrier
    return modulated

def fsk_modulate(bin_bit, f0=400, f1=1600, fs=8000, duration=0.1):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if bin_bit == '1':
        modulated = 1.2 * np.cos(2 * np.pi * f1 * t)
    else:
        modulated = 0.8 * np.cos(2 * np.pi * f0 * t)
    return modulated

def psk_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    t, carrier = generate_carrier(fc, fs, duration)
    phase = np.pi if bin_bit == '1' else 0
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return modulated

def qpsk_modulate(symbol, fc=1000, fs=8000, duration=0.1):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    phase_map = {'00': 0, '01': np.pi/2, '11': np.pi, '10': 3*np.pi/2}
    phase = phase_map.get(symbol, 0)
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return modulated

def qam16_modulate(symbol, fc=1000, fs=8000, duration=0.1):
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
    return modulated

def extract_hos_features(signal):
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

def generate_training_data(n_samples_per_mod=2000, fs=8000, duration=0.1, snr=10):
    mod_functions = {
        "AM": am_modulate,
        "2FSK": fsk_modulate,
        "BPSK": psk_modulate,
        "QPSK": qpsk_modulate,
        "16QAM": qam16_modulate
    }
    mod_types = ["AM", "2FSK", "BPSK", "QPSK", "16QAM"]
    features = []
    labels = []

    for label, mod_name in enumerate(mod_types):
        print(f"正在生成 {mod_name} 训练数据...")
        mod_func = mod_functions[mod_name]
        for _ in range(n_samples_per_mod):
            # 每个样本固定一个载波频率，保证符号间相位连续
            if mod_name in ["AM", "BPSK", "QPSK", "16QAM"]:
                fc = np.random.uniform(800, 1200)
            elif mod_name == "2FSK":
                f0 = np.random.uniform(300, 600)
                f1 = np.random.uniform(1400, 1800)

            segments = []
            for _ in range(16):
                if mod_name in ["AM", "2FSK", "BPSK"]:
                    sym = '1' if np.random.rand() > 0.5 else '0'
                elif mod_name == "QPSK":
                    sym = ['00','01','10','11'][np.random.randint(4)]
                elif mod_name == "16QAM":
                    sym = format(np.random.randint(16), '04b')

                if mod_name in ["AM", "BPSK", "QPSK", "16QAM"]:
                    seg = mod_func(sym, fc=fc, fs=fs, duration=duration)
                elif mod_name == "2FSK":
                    seg = mod_func(sym, f0=f0, f1=f1, fs=fs, duration=duration)
                segments.append(seg)

            signal = np.concatenate(segments)

            snr_dynamic = np.random.uniform(5, 18)
            gauss_noise = np.random.normal(0, np.std(signal)/snr_dynamic, len(signal))
            if mod_name in ["BPSK", "QPSK"]:
                pulse_noise = np.random.choice([0, 2*np.std(signal), -2*np.std(signal)],
                                              size=len(signal), p=[0.998, 0.001, 0.001])
            else:
                pulse_noise = np.random.choice([0, 2*np.std(signal), -2*np.std(signal)],
                                              size=len(signal), p=[0.99, 0.005, 0.005])
            signal = signal + gauss_noise + pulse_noise
            if mod_name in ["BPSK", "QPSK"]:
                signal = signal * np.random.uniform(0.95, 1.05)
            else:
                signal = signal * np.random.uniform(0.9, 1.1)
            feat = extract_hos_features(signal)
            features.append(feat)
            labels.append(label)

    return np.array(features), np.array(labels), mod_types

def train_and_save_model():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, mod_types = generate_training_data(n_samples_per_mod=10000, snr=6)
    class_weights = {0:1.0, 1:1.0, 2:2.0, 3:2.0, 4:1.0}

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        max_iter=8000,
        learning_rate_init=0.001,
        learning_rate='adaptive',
        alpha=0.00005,
        tol=1e-7,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=60,
        random_state=42,
        verbose=True
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n模型测试集精度：{acc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=mod_types))

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=mod_types, yticklabels=mod_types,
                annot_kws={'size': 12}, cbar_kws={'label': '样本数'})

    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('调制类型识别混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    print("\n混淆矩阵已保存")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "modulation_classifier.pkl"))
    joblib.dump(mod_types, os.path.join(MODEL_DIR, "mod_types.pkl"))
    print("\n模型已保存到 mlp/model/")
    return clf, mod_types

if __name__ == "__main__":
    clf, mod_types = train_and_save_model()
