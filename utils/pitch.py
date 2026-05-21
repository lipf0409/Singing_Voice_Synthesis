"""音高处理工具"""
import numpy as np
import parselmouth
from config import config

def get_f0(wav, method="parselmouth"):
    """提取基频F0"""
    if method == "parselmouth":
        snd = parselmouth.Sound(wav, sampling_frequency=config.SAMPLE_RATE)
        pitch = snd.to_pitch(time_step=config.HOP_LENGTH / config.SAMPLE_RATE)
        f0 = pitch.selected_array["frequency"]
        # 将f0填充到与mel帧数一致
        return f0
    else:
        # 使用pyworld作为备选
        import pyworld as pw
        f0, t = pw.dio(wav.astype(np.float64), config.SAMPLE_RATE,
                       frame_period=config.HOP_LENGTH / config.SAMPLE_RATE * 1000)
        f0 = pw.stonemask(wav.astype(np.float64), f0, t, config.SAMPLE_RATE)
        return f0

def norm_pitch(f0):
    """归一化音高"""
    # 将f0映射到对数空间并归一化
    nonzero = f0 > 0
    f0_norm = np.zeros_like(f0)
    if np.any(nonzero):
        f0_norm[nonzero] = np.log(f0[nonzero])
        mean = np.mean(f0_norm[nonzero])
        std = np.std(f0_norm[nonzero])
        if std > 0:
            f0_norm[nonzero] = (f0_norm[nonzero] - mean) / std
    return f0_norm, mean, std

def denorm_pitch(f0_norm, mean, std):
    """反归一化音高"""
    f0 = np.exp(f0_norm * std + mean)
    return f0
