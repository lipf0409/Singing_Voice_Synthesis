"""简化版FastSpeech2用于歌声合成"""
import torch
import torch.nn as nn
from config import config, N_PHONEMES
from models.modules import PositionalEncoding, FFTBlock, VariancePredictor, PitchPredictor, LengthRegulator, init_weights

class FastSpeech2SVS(nn.Module):
    """FastSpeech2 歌声合成模型"""
    def __init__(self):
        super().__init__()

        # 音素嵌入
        self.phoneme_emb = nn.Embedding(N_PHONEMES, config.PHONEME_EMB_DIM, padding_idx=0)

        # 位置编码
        self.pos_enc = PositionalEncoding(config.PHONEME_EMB_DIM)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            FFTBlock(
                config.PHONEME_EMB_DIM,
                config.ENCODER_HEAD,
                config.CONV_FILTER_SIZE,
                config.CONV_KERNEL_SIZE,
                config.DROPOUT
            ) for _ in range(config.ENCODER_N_LAYER)
        ])

        # Variance Predictors
        self.duration_predictor = VariancePredictor(
            config.PHONEME_EMB_DIM,
            config.VARIANCE_PREDICTOR_FILTER_SIZE,
            config.VARIANCE_PREDICTOR_KERNEL_SIZE,
            config.VARIANCE_PREDICTOR_DROPOUT
        )
        self.pitch_predictor = PitchPredictor(
            config.PHONEME_EMB_DIM,
            config.VARIANCE_PREDICTOR_FILTER_SIZE,
            config.VARIANCE_PREDICTOR_KERNEL_SIZE,
            config.VARIANCE_PREDICTOR_DROPOUT
        )

        # 音高嵌入
        self.pitch_emb = nn.Linear(1, config.PHONEME_EMB_DIM)

        # Length Regulator
        self.length_regulator = LengthRegulator()

        # Decoder
        self.decoder_layers = nn.ModuleList([
            FFTBlock(
                config.PHONEME_EMB_DIM,
                config.DECODER_HEAD,
                config.CONV_FILTER_SIZE,
                config.CONV_KERNEL_SIZE,
                config.DROPOUT
            ) for _ in range(config.DECODER_N_LAYER)
        ])

        # Mel谱输出
        self.mel_linear = nn.Linear(config.PHONEME_EMB_DIM, config.N_MELS)

        # 权重初始化
        self.apply(init_weights)

    def forward(self, phoneme_ids, duration_target=None, pitch_target=None, mel_target=None, max_mel_len=None, pitch_shift=0.0, speed=1.0):
        """
        phoneme_ids: (B, T_p)
        duration_target: (B, T_p) - 训练时使用真实时长
        pitch_target: (B, T_mel) - 训练时使用真实音高
        mel_target: (B, N_MELS, T_mel) - 用于获取长度
        """
        # 音素嵌入
        x = self.phoneme_emb(phoneme_ids)
        x = self.pos_enc(x)

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # 预测时长
        duration_pred = self.duration_predictor(x)

        # 预测音高（在音素级别）
        pitch_pred_phoneme = self.pitch_predictor(x)

        # 长度调节
        if duration_target is not None:
            # 训练时使用真实时长
            x_expanded = self.length_regulator(x, duration_target, max_mel_len)
            # 扩展音素级音高到帧级别
            pitch_expanded = self.expand_pitch_by_duration(pitch_pred_phoneme, duration_target, max_mel_len)
        else:
            # 推理时使用预测时长（添加温度采样增加自然度）
            temperature = 0.3  # 温度参数，值越大随机性越强
            noise = torch.randn_like(duration_pred) * temperature
            # 语速控制：speed > 1 加快，speed < 1 减慢
            duration_adjusted = (duration_pred + noise) / speed
            duration_pred_rounded = torch.clamp(torch.round(duration_adjusted), min=1)
            x_expanded = self.length_regulator(x, duration_pred_rounded, max_mel_len)
            # 扩展音素级音高到帧级别
            pitch_expanded = self.expand_pitch_by_duration(pitch_pred_phoneme, duration_pred_rounded, max_mel_len)

        # 添加音高信息
        pitch_emb = self.pitch_emb(pitch_expanded.unsqueeze(-1))

        # 对齐长度
        if pitch_emb.size(1) != x_expanded.size(1):
            min_len = min(pitch_emb.size(1), x_expanded.size(1))
            pitch_emb = pitch_emb[:, :min_len, :]
            x_expanded = x_expanded[:, :min_len, :]

        # 音高曲线平滑（减少帧间抖动）
        pitch_expanded = self.smooth_pitch(pitch_expanded, window=5)
        # 添加音高偏移（半音为单位）
        pitch_expanded = pitch_expanded + pitch_shift
        pitch_emb = self.pitch_emb(pitch_expanded.unsqueeze(-1))

        x = x_expanded + pitch_emb

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        # Mel谱预测
        mel_out = self.mel_linear(x)
        mel_out = mel_out.transpose(1, 2)  # (B, T_mel, N_MELS) -> (B, N_MELS, T_mel)

        # 返回实际使用的时长（推理时返回调整后的时长）
        if duration_target is None:
            return mel_out, duration_pred_rounded, pitch_pred_phoneme
        else:
            return mel_out, duration_pred, pitch_pred_phoneme

    def smooth_pitch(self, pitch, window=5):
        """音高曲线平滑，减少帧间抖动"""
        if pitch.size(1) < window:
            return pitch
        kernel = torch.ones(window, device=pitch.device, dtype=pitch.dtype) / window
        smoothed = torch.nn.functional.conv1d(
            pitch.unsqueeze(1),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window // 2
        ).squeeze(1)
        return smoothed

    def expand_pitch_by_duration(self, pitch_phoneme, duration, max_len=None):
        """将音素级音高按时长扩展到帧级别（优化版）"""
        B = pitch_phoneme.size(0)
        duration_int = torch.clamp(torch.round(duration).long(), min=1)

        # 使用 repeat_interleave 批量扩展，效率更高
        expanded = []
        for i in range(B):
            exp = torch.repeat_interleave(pitch_phoneme[i], duration_int[i], dim=0)
            expanded.append(exp)

        # 填充到相同长度
        if max_len is None:
            max_len = max(e.size(0) for e in expanded) if expanded else 0

        max_len = max(max_len, 1)

        padded = []
        for e in expanded:
            if e.size(0) < max_len:
                pad = torch.zeros(max_len - e.size(0), device=e.device, dtype=e.dtype)
                e = torch.cat([e, pad], dim=0)
            else:
                e = e[:max_len]
            padded.append(e)

        return torch.stack(padded, dim=0)

    def inference(self, phoneme_ids, max_mel_len=1000, pitch_shift=0.0, speed=1.0):
        """推理接口"""
        self.eval()
        with torch.no_grad():
            mel_out, _, _ = self.forward(phoneme_ids, max_mel_len=max_mel_len, pitch_shift=pitch_shift, speed=speed)
        return mel_out
