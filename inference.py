"""推理脚本：歌词+音高 -> 音频"""
import os
import argparse
import torch
import numpy as np
import soundfile as sf

from config import config, PHONEME_TO_ID
from models.fastspeech2 import FastSpeech2SVS
from utils.audio import inv_mel_spectrogram

# HiFi-GAN 声码器加载
try:
    from hifi_gan import load_hifigan
    HIFIGAN_AVAILABLE = True
except ImportError:
    HIFIGAN_AVAILABLE = False
    print("Warning: HiFi-GAN not found. Will use Griffin-Lim as fallback.")


def load_stats(processed_dir="data/processed"):
    """加载归一化统计量"""
    stats_path = os.path.join(processed_dir, "stats_train.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return {
            "mel_mean": float(stats["mel_mean"]),
            "mel_std": float(stats["mel_std"]),
            "f0_mean": float(stats["f0_mean"]),
            "f0_std": float(stats["f0_std"]),
        }
    else:
        print("Warning: stats_train.npz not found, using default values")
        return {"mel_mean": 0.0, "mel_std": 1.0, "f0_mean": 0.0, "f0_std": 1.0}


def smooth_mel(mel, window=3):
    """Mel谱帧间平滑，减少抖动"""
    kernel = np.ones(window) / window
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=1, arr=mel
    )


def denormalize_mel(mel, mel_mean, mel_std):
    """反归一化 Mel 谱"""
    return mel * mel_std + mel_mean


def text_to_phoneme_ids(text):
    """将歌词文本转为音素ID序列（声母+韵母格式）"""
    from pypinyin import lazy_pinyin, Style

    pys = lazy_pinyin(text, style=Style.TONE3)

    phonemes = []
    for py in pys:
        if not py:
            continue
        initial, final = split_pinyin(py)
        if initial:
            phonemes.append(initial)
        if final:
            phonemes.append(final)

    phonemes.append("~")

    print(f"Phonemes: {phonemes}")
    ids = [PHONEME_TO_ID.get(p, PHONEME_TO_ID["sp"]) for p in phonemes]
    print(f"Mapped IDs: {ids}")
    return torch.LongTensor(ids).unsqueeze(0)


def split_pinyin(py):
    """将拼音拆分为声母和韵母"""
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']

    py_clean = ''.join(c for c in py if not c.isdigit())

    for init in sorted(initials, key=len, reverse=True):
        if py_clean.startswith(init):
            final = py_clean[len(init):]
            return init, final if final else None

    return None, py_clean


def normalize_audio(audio, target_db=-20.0):
    """
    音频音量归一化
    解决不同合成结果音量不一致的问题
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    audio = audio * gain
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def enhance_audio(audio, sample_rate=22050):
    """音频后处理增强 - 让声音更自然"""
    # 动态压缩
    audio = compressor(audio, threshold=0.3, ratio=3.0)
    # 添加自然噪声
    audio = add_natural_noise(audio, noise_level=0.001)
    # 轻微混响
    audio = add_reverb(audio, sample_rate, decay=0.2, delay=0.015)
    # 归一化
    audio = normalize_audio(audio, target_db=-18.0)
    return audio


def compressor(audio, threshold=0.3, ratio=3.0):
    """动态范围压缩"""
    envelope = np.abs(audio)
    kernel_size = max(1, int(22050 * 0.005))
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(envelope, kernel, mode='same')

    gain = np.ones_like(audio)
    mask = envelope > threshold
    gain[mask] = threshold + (envelope[mask] - threshold) / ratio
    gain[mask] = gain[mask] / envelope[mask]

    return audio * gain


def add_natural_noise(audio, noise_level=0.001):
    """添加自然噪声"""
    noise = np.random.randn(len(audio)) * noise_level
    envelope = np.abs(audio)
    kernel_size = max(1, int(22050 * 0.01))
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(envelope, kernel, mode='same')
    envelope = envelope / (envelope.max() + 1e-8)
    return audio + noise * envelope


def add_reverb(audio, sample_rate, decay=0.2, delay=0.015):
    """添加轻微混响"""
    delay_samples = int(delay * sample_rate)
    if delay_samples >= len(audio):
        return audio

    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = audio[:-delay_samples] * decay

    delay2 = int(delay_samples * 1.5)
    if delay2 < len(audio):
        reverb[delay2:] += audio[:-delay2] * decay * 0.3

    return audio + reverb


def synthesize(model, text, hifigan=None, output_path="output.wav", max_len=1000, stats=None, pitch_shift=0.0, speed=1.0):
    """合成歌声"""
    model.eval()

    phoneme_ids = text_to_phoneme_ids(text).to(config.DEVICE)
    print(f"Phoneme IDs shape: {phoneme_ids.shape}")

    with torch.no_grad():
        mel_pred, dur_pred, pitch_pred = model(
            phoneme_ids,
            max_mel_len=max_len,
            pitch_shift=pitch_shift,
            speed=speed
        )
        print(f"Duration prediction: {dur_pred[0].tolist()}")
        print(f"Duration sum: {dur_pred[0].sum().item():.1f}")
        print(f"Mel shape: {mel_pred.shape}")

        # 只取有效长度的Mel谱（根据时长预测）
        expected_len = int(dur_pred[0].sum().item())
        print(f"Expected valid frames: {expected_len} ({expected_len * 256 / 22050:.2f}s)")

        mel = mel_pred[0, :, :expected_len].cpu().numpy()

    # 反归一化 Mel 谱
    if stats is not None:
        mel = denormalize_mel(mel, stats["mel_mean"], stats["mel_std"])
        print(f"Denormalized mel range: [{mel.min():.2f}, {mel.max():.2f}]")

    # Mel谱平滑，减少帧间抖动
    mel = smooth_mel(mel, window=3)
    print("Applied mel smoothing (window=3)")

    # 声码器合成音频
    if hifigan is not None:
        with torch.no_grad():
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(config.DEVICE)
            audio = hifigan(mel_tensor).squeeze()
            audio = audio.cpu().numpy()
        print("Generated with HiFi-GAN vocoder")
    else:
        audio = inv_mel_spectrogram(mel)
        print("Generated with Griffin-Lim vocoder")

    audio = np.clip(audio, -1.0, 1.0)

    # 音频增强
    audio = enhance_audio(audio)
    print("Applied audio enhancement")

    sf.write(output_path, audio, config.SAMPLE_RATE)
    print(f"Saved: {output_path}")
    return audio


def main(args):
    # 加载归一化统计量
    stats = load_stats(args.processed_dir)

    # 加载模型
    model = FastSpeech2SVS().to(config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    # 加载声码器
    hifigan = None
    if HIFIGAN_AVAILABLE and os.path.exists(args.hifigan_config) and os.path.exists(args.hifigan_ckpt):
        try:
            hifigan = load_hifigan(args.hifigan_config, args.hifigan_ckpt, config.DEVICE)
            print("Loaded HiFi-GAN vocoder")
        except Exception as e:
            print(f"Failed to load HiFi-GAN: {e}")
            print("Using Griffin-Lim as fallback")

    # 合成
    synthesize(model, args.text, hifigan, args.output, args.max_len, stats, args.pitch_shift, args.speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_epoch_960.pt", help="模型检查点路径")
    parser.add_argument("--text", default="我爱你中国", help="要合成的歌词")
    parser.add_argument("--output", default="output.wav", help="输出音频路径")
    parser.add_argument("--hifigan_config", default="checkpoints/hifigan/config.json")
    parser.add_argument("--hifigan_ckpt", default="checkpoints/hifigan/generator_v1")
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--processed_dir", default="data/processed", help="预处理数据目录")
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="音高偏移（半音）")
    parser.add_argument("--speed", type=float, default=1.0, help="语速系数（>1加快，<1减慢）")
    args = parser.parse_args()
    main(args)
