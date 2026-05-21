"""损失函数"""
import torch
import torch.nn as nn


def get_pitch_target_phoneme_level(pitch_target, duration_target):
    """将帧级音高转换为音素级音高（按时长平均）"""
    B = pitch_target.size(0)
    pitch_phoneme = []

    for i in range(B):
        durations = duration_target[i].cpu().numpy()
        pitches = pitch_target[i].cpu().numpy()

        phoneme_pitches = []
        frame_idx = 0
        for d in durations:
            d_int = int(d)
            if d_int > 0 and frame_idx < len(pitches):
                # 取该音素对应帧的平均音高
                segment = pitches[frame_idx:frame_idx + d_int]
                # 只取非零值（有声段）
                nonzero = segment[segment != 0]
                if len(nonzero) > 0:
                    phoneme_pitches.append(nonzero.mean())
                else:
                    phoneme_pitches.append(0.0)
                frame_idx += d_int
            else:
                phoneme_pitches.append(0.0)

        pitch_phoneme.append(phoneme_pitches)

    # 转换为tensor
    max_len = max(len(p) for p in pitch_phoneme)
    pitch_tensor = torch.zeros(B, max_len, device=pitch_target.device)
    for i, p in enumerate(pitch_phoneme):
        pitch_tensor[i, :len(p)] = torch.tensor(p, device=pitch_target.device)

    return pitch_tensor


class FastSpeech2Loss(nn.Module):
    def __init__(self, mel_weight=1.0, dur_weight=0.1, pitch_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mel_weight = mel_weight
        self.dur_weight = dur_weight
        self.pitch_weight = pitch_weight

    def forward(self, mel_pred, mel_target, duration_pred, duration_target, pitch_pred, pitch_target, mel_mask):
        """
        mel_pred: (B, N_MELS, T_mel)
        mel_target: (B, N_MELS, T_mel)
        duration_pred: (B, T_p)
        duration_target: (B, T_p)
        pitch_pred: (B, T_p)
        pitch_target: (B, T_mel)
        mel_mask: (B, T_mel)
        """
        # Mel谱损失 - 只计算有效区域（主任务）
        mel_mask_expanded = mel_mask.unsqueeze(1).expand_as(mel_pred).float()
        mel_diff = (mel_pred - mel_target).abs() * mel_mask_expanded
        mel_loss = mel_diff.sum() / (mel_mask_expanded.sum() + 1e-6)

        # 时长损失 - 使用log尺度（辅助任务）
        duration_target_float = duration_target.float().clamp(min=1.0)
        duration_pred_clamped = duration_pred.clamp(min=1.0)
        duration_target_log = torch.log(duration_target_float)
        duration_pred_log = torch.log(duration_pred_clamped)
        duration_loss = self.mse(duration_pred_log, duration_target_log)

        # 音高损失 - 将帧级音高转换为音素级（辅助任务）
        if pitch_target is not None:
            pitch_target_phoneme = get_pitch_target_phoneme_level(pitch_target, duration_target)
            # 只计算有效长度的损失
            max_len = min(pitch_pred.size(1), pitch_target_phoneme.size(1))
            pitch_loss = self.mse(pitch_pred[:, :max_len], pitch_target_phoneme[:, :max_len])
        else:
            pitch_loss = torch.tensor(0.0, device=mel_pred.device)

        # 加权总损失
        total_loss = (self.mel_weight * mel_loss +
                      self.dur_weight * duration_loss +
                      self.pitch_weight * pitch_loss)
        return total_loss, mel_loss, duration_loss, pitch_loss
