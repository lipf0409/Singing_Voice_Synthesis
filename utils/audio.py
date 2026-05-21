"""音频处理工具"""
import numpy as np
import librosa
import soundfile as sf
import torch
from config import config

def load_wav(path):
    """加载音频并归一化"""
    wav, sr = librosa.load(path, sr=config.SAMPLE_RATE)
    return wav

def save_wav(wav, path):
    """保存音频"""
    sf.write(path, wav, config.SAMPLE_RATE)

def mel_spectrogram(y):
    """计算Mel谱图"""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.MEL_FMIN,
        fmax=config.MEL_FMAX
    )
    mel = np.clip(mel, a_min=1e-5, a_max=None)
    mel = np.log(mel)
    return mel

def inv_mel_spectrogram(mel):
    """Griffin-Lim反变换（备用）"""
    mel = np.exp(mel)
    S = librosa.feature.inverse.mel_to_stft(
        mel,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        power=1.0
    )
    wav = librosa.griffinlim(S, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH)
    return wav

def get_mel_from_wav(wav_path):
    """从音频文件获取Mel谱"""
    wav = load_wav(wav_path)
    mel = mel_spectrogram(wav)
    return mel, wav

def pad_1D(inputs, PAD=0):
    """填充1D序列"""
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]),
            mode="constant",
            constant_values=PAD
        )
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded

def pad_2D(inputs, maxlen=None):
    """填充2D序列"""
    def pad(x, max_len):
        s = np.shape(x)
        if s[1] == max_len:
            return x
        return np.pad(
            x, ((0, 0), (0, max_len - s[1])),
            mode="constant",
            constant_values=0
        )
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output
