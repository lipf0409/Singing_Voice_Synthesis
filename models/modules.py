"""模型组件"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m):
    """权重初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FFTBlock(nn.Module):
    """FastSpeech中的FFT块 - 使用Pre-LN（更稳定）"""
    def __init__(self, d_model, n_head, d_inner, kernel_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.slf_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.pos_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, padding=(kernel_size - 1) // 2),
            nn.GELU(),
            nn.Conv1d(d_inner, d_model, kernel_size, padding=(kernel_size - 1) // 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.slf_attn(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(attn_out)

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        ff_out = self.pos_ffn(x.transpose(1, 2)).transpose(1, 2)
        x = residual + self.dropout(ff_out)
        return x


class VariancePredictor(nn.Module):
    """时长预测器 - 输出非负值（>= 1）"""
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.conv1(x.transpose(1, 2))  # (B, filter_size, T)
        x = self.norm1(x.transpose(1, 2))  # (B, T, filter_size)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x.transpose(1, 2))  # (B, filter_size, T)
        x = self.norm2(x.transpose(1, 2))  # (B, T, filter_size)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.linear(x)  # (B, T, 1)
        # 使用 ReLU + 1 确保输出 >= 1（时长至少为1帧）
        return F.relu(out.squeeze(-1)) + 1.0  # (B, T)


class PitchPredictor(nn.Module):
    """音高预测器 - 可输出任意实数值"""
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.conv1(x.transpose(1, 2))  # (B, filter_size, T)
        x = self.norm1(x.transpose(1, 2))  # (B, T, filter_size)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x.transpose(1, 2))  # (B, filter_size, T)
        x = self.norm2(x.transpose(1, 2))  # (B, T, filter_size)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.linear(x)  # (B, T, 1)
        # 直接输出，不限制范围（归一化后的pitch可以是负值）
        return out.squeeze(-1)  # (B, T)


class LengthRegulator(nn.Module):
    """长度调节器：根据时长扩展音素序列（优化版）"""
    def __init__(self):
        super().__init__()

    def forward(self, x, duration, max_len=None):
        # x: (B, T_phoneme, D)
        # duration: (B, T_phoneme)
        B = x.size(0)
        duration_int = torch.clamp(torch.round(duration).long(), min=1)

        # 使用 repeat_interleave 批量扩展，效率更高
        expanded = []
        for i in range(B):
            exp = torch.repeat_interleave(x[i], duration_int[i], dim=0)
            expanded.append(exp)

        # 填充到相同长度
        if max_len is None:
            max_len = max(e.size(0) for e in expanded) if expanded else 0

        max_len = max(max_len, 1)

        padded = []
        for e in expanded:
            if e.size(0) < max_len:
                pad = torch.zeros(max_len - e.size(0), e.size(1), device=e.device, dtype=e.dtype)
                e = torch.cat([e, pad], dim=0)
            else:
                e = e[:max_len]
            padded.append(e)

        return torch.stack(padded, dim=0)
