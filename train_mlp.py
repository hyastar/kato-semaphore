import numpy as np
import joblib  # 保存/加载模型
from scipy.stats import moment
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler  # 新增导入

# ===================== 1. 复用调制函数（生成真实信号） =====================
def generate_carrier(fc=1000, fs=8000, duration=0.1):
    """生成载波"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * fc * t)
    return t, carrier

def am_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    """生成真实AM信号（标准调幅，非通断键控）"""
    t, carrier = generate_carrier(fc, fs, duration)
    # 标准AM调制：A_c*(1 + k*Am)，k=0.8（调制系数），避免过调制
    mod_coeff = 0.8  
    if bin_bit == '1':
        amp = 1 + mod_coeff  # 调幅后幅值更大
    else:
        amp = 1 - mod_coeff  # 调幅后幅值更小（非0）
    modulated = amp * carrier
    return modulated

def fsk_modulate(bin_bit, f0=600, f1=1400, fs=8000, duration=0.1):
    """生成真实FSK信号（提升特征区分度）"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if bin_bit == '1':
        modulated = 1.2 * np.cos(2 * np.pi * f1 * t)  # 幅值1.2
    else:
        modulated = 0.8 * np.cos(2 * np.pi * f0 * t)  # 幅值0.8
    return modulated

def psk_modulate(bin_bit, fc=1000, fs=8000, duration=0.1):
    """BPSK调制：二进制相移键控（1bit/符号）"""
    t, carrier = generate_carrier(fc, fs, duration)
    phase = np.pi if bin_bit == '1' else 0
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return modulated

def qpsk_modulate(symbol, fc=1000, fs=8000, duration=0.1):
    """
    QPSK调制：2bit符号→4个相位（格雷码映射）
    symbol: 2bit字符串（如'00','01','10','11'）
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    phase_map = {'00': 0, '01': np.pi/2, '11': np.pi, '10': 3*np.pi/2}
    phase = phase_map.get(symbol, 0)
    modulated = np.cos(2 * np.pi * fc * t + phase)
    return modulated

def qam16_modulate(symbol, fc=1000, fs=8000, duration=0.1):
    """
    16QAM调制：4bit符号→16个星座点
    symbol: 4bit字符串（如'0000'-'1111'）
    """
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

# ===================== 2. 高阶统计量+频谱特征提取 =====================
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

# ===================== 3. 生成训练数据（真实调制信号） =====================
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
        mod_func = mod_functions[mod_name]
        for _ in range(n_samples_per_mod):
            # 根据调制类型生成符号
            if mod_name in ["AM", "2FSK", "BPSK"]:
                symbol = '1' if np.random.rand() > 0.5 else '0'
            elif mod_name == "QPSK":
                qpsk_symbols = ['00', '01', '10', '11']
                symbol = qpsk_symbols[np.random.randint(0, 4)]
            elif mod_name == "16QAM":
                qam16_symbols = [format(i, '04b') for i in range(16)]
                symbol = qam16_symbols[np.random.randint(0, 16)]
            
            # 随机化载波频率（更大范围）
            if mod_name in ["AM", "BPSK", "QPSK", "16QAM"]:
                fc = np.random.uniform(800, 1200)  # 扩大载波范围
                signal = mod_func(symbol, fc=fc, fs=fs, duration=duration)
            elif mod_name == "2FSK":
                f0 = np.random.uniform(400, 800)
                f1 = np.random.uniform(1200, 1600)
                signal = mod_func(symbol, f0=f0, f1=f1, fs=fs, duration=duration)
            
            # ========== 噪声优化：对BPSK/QPSK降低相位噪声 ==========
            snr_dynamic = np.random.uniform(5, 18)
            gauss_noise = np.random.normal(0, np.std(signal)/snr_dynamic, len(signal))
            # 对BPSK/QPSK，脉冲噪声概率降低（避免相位被脉冲干扰）
            if mod_name in ["BPSK", "QPSK"]:
                pulse_noise = np.random.choice([0, 2*np.std(signal), -2*np.std(signal)], 
                                              size=len(signal), p=[0.998, 0.001, 0.001])
            else:
                pulse_noise = np.random.choice([0, 2*np.std(signal), -2*np.std(signal)], 
                                              size=len(signal), p=[0.99, 0.005, 0.005])
            signal = signal + gauss_noise + pulse_noise
            # BPSK/QPSK幅值偏移降低（相位是核心，幅值偏移无意义）
            if mod_name in ["BPSK", "QPSK"]:
                signal = signal * np.random.uniform(0.95, 1.05)
            else:
                signal = signal * np.random.uniform(0.9, 1.1)
            # 提取特征
            feat = extract_hos_features(signal)
            features.append(feat)
            labels.append(label)
    
    return np.array(features), np.array(labels), mod_types

# ===================== 4. 训练并保存MLP模型 =====================
def train_and_save_model():
    # 1. 生成训练数据（BPSK/QPSK样本数翻倍）
    X, y, mod_types = generate_training_data(n_samples_per_mod=8000, snr=6)
    # 对BPSK/QPSK样本加权（提升学习优先级）
    class_weights = {0:1.0, 1:1.0, 2:2.0, 3:2.0, 4:1.0}  # BPSK(2)/QPSK(3)权重翻倍
    
    # 2. 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. 优化MLP参数（针对相位特征）
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),  # 更大网络容量，学习细粒度相位特征
        max_iter=8000,                          # 延长训练轮数
        learning_rate_init=0.0001,              # 更低学习率，精细拟合相位特征
        learning_rate='adaptive',
        alpha=0.0001,                           # 降低正则化，避免相位特征被压制
        tol=1e-7,                               # 更严格的早停阈值
        early_stopping=True,
        validation_fraction=0.25,
        n_iter_no_change=80,                    # 延长早停等待
        random_state=42,
        verbose=True
    )
    clf.fit(X_train, y_train)
    
    # 5. 验证模型
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n模型测试集精度：{acc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=mod_types))
    
    # 6. 混淆矩阵可视化
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制更美观的混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=mod_types, yticklabels=mod_types,
                annot_kws={'size': 12}, cbar_kws={'label': '样本数'})
    
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('调制类型识别混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n混淆矩阵已保存为 confusion_matrix.png")
    
    # 7. 保存模型
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(clf, "modulation_classifier.pkl")
    joblib.dump(mod_types, "mod_types.pkl")
    print("\n模型已保存！")
    return clf, mod_types

# 执行训练
if __name__ == "__main__":
    clf, mod_types = train_and_save_model()
