"""训练脚本"""
import os
import argparse
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import config
from data.dataset import OpenCPopDataset, OpenCPopDatasetFast, load_opencpop_data, collate_fn
from models.fastspeech2 import FastSpeech2SVS
from models.loss import FastSpeech2Loss


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not checkpoints:
        return None
    # 按epoch数排序
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, global_step, loss, max_retries=3):
    """保存checkpoint（带重试机制）"""
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "loss": loss,
    }

    # 先保存到临时文件，成功后再重命名
    temp_path = path + ".tmp"

    for attempt in range(max_retries):
        try:
            torch.save(checkpoint_data, temp_path)
            # 验证文件是否完整
            torch.load(temp_path, map_location='cpu', weights_only=False)
            # 成功后重命名
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
            return True
        except Exception as e:
            print(f"Save attempt {attempt + 1} failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if attempt < max_retries - 1:
                print("Retrying...")
                import time
                time.sleep(2)
            else:
                print(f"Failed to save checkpoint after {max_retries} attempts")
                # 至少保存模型权重
                try:
                    backup_path = path.replace(".pt", "_backup.pt")
                    torch.save({"model_state_dict": model.state_dict()}, backup_path)
                    print(f"Saved backup model weights to {backup_path}")
                except:
                    pass
                return False
    return True


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """加载checkpoint"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载scheduler状态（如果存在）
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 加载scaler状态（如果存在）
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    loss = checkpoint.get("loss", 0)

    return epoch, global_step, loss


def train(args):
    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # 加载数据
    train_data = load_opencpop_data(config.TRANS_FILE, config.TRAIN_FILE)
    test_data = load_opencpop_data(config.TRANS_FILE, config.TEST_FILE)

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # 检查是否使用预处理数据
    use_preprocessed = getattr(config, 'USE_PREPROCESSED', False)
    mel_dir = os.path.join(config.PROCESSED_DIR, "mel")

    if use_preprocessed and os.path.exists(mel_dir):
        print(f"Using preprocessed data from {config.PROCESSED_DIR}")
        train_dataset = OpenCPopDatasetFast(train_data, config.PROCESSED_DIR)
        test_dataset = OpenCPopDatasetFast(test_data, config.PROCESSED_DIR)
    else:
        print("Using raw audio (slower). Run preprocess.py first for faster training.")
        train_dataset = OpenCPopDataset(train_data, config.WAV_DIR)
        test_dataset = OpenCPopDataset(test_data, config.WAV_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True  # 16GB显存可以开启pin_memory加速数据传输
    )

    # 模型
    model = FastSpeech2SVS().to(config.DEVICE)
    criterion = FastSpeech2Loss(mel_weight=1.0, dur_weight=0.1, pitch_weight=0.1)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 混合精度训练
    use_amp = getattr(config, 'USE_AMP', False) and config.DEVICE.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {use_amp}")

    # 梯度累积步数
    grad_accum_steps = getattr(config, 'GRAD_ACCUM_STEPS', 1)
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {config.BATCH_SIZE * grad_accum_steps}")

    # 学习率调度（方案1：余弦退火重启）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=getattr(config, 'COSINE_T0', 50),       # 第一次重启周期
        T_mult=getattr(config, 'COSINE_T_MULT', 2), # 每次重启后周期翻倍
        eta_min=getattr(config, 'COSINE_ETA_MIN', 1e-6)  # 最小学习率
    )
    print(f"Using CosineAnnealingWarmRestarts: T_0={config.COSINE_T0}, T_mult={config.COSINE_T_MULT}")

    writer = SummaryWriter(config.LOG_DIR)
    global_step = 0
    start_epoch = 0

    # 断点续训
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(config.CHECKPOINT_DIR)

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        start_epoch, global_step, prev_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, config.DEVICE
        )
        print(f"Resumed from epoch {start_epoch}, global_step {global_step}, loss {prev_loss:.4f}")
        start_epoch = start_epoch  # 从下一个epoch开始

    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_losses = {"total": [], "mel": [], "dur": [], "pitch": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        optimizer.zero_grad()

        for step_idx, batch in enumerate(pbar):
            phoneme_ids = batch["phoneme_ids"].to(config.DEVICE)
            mels = batch["mels"].to(config.DEVICE)
            f0s = batch["f0s"].to(config.DEVICE)
            durations = batch["durations"].to(config.DEVICE)
            mel_lens = batch["mel_lens"].to(config.DEVICE)
            max_mel_len = batch["max_mel_len"]

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=use_amp):
                mel_pred, dur_pred, pitch_pred = model(
                    phoneme_ids,
                    duration_target=durations,
                    pitch_target=f0s,
                    mel_target=mels,
                    max_mel_len=max_mel_len
                )

                # 计算损失
                mel_mask = torch.arange(max_mel_len, device=config.DEVICE).unsqueeze(0) < mel_lens.unsqueeze(1)
                loss, mel_loss, dur_loss, pitch_loss = criterion(
                    mel_pred, mels, dur_pred, durations, pitch_pred, f0s, mel_mask
                )

            # 检测NaN
            if torch.isnan(loss):
                print("NaN detected, skipping batch...")
                optimizer.zero_grad()
                continue

            # 梯度累积：损失除以累积步数
            loss = loss / grad_accum_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 每隔grad_accum_steps步更新一次参数
            if (step_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_THRESH)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # 注意：余弦退火重启按epoch调度，不在step内调用

            # 记录（使用原始损失值）
            loss_val = loss.item() * grad_accum_steps
            epoch_losses["total"].append(loss_val)
            epoch_losses["mel"].append(mel_loss.item())
            epoch_losses["dur"].append(dur_loss.item())
            epoch_losses["pitch"].append(pitch_loss.item())

            writer.add_scalar("Loss/total", loss_val, global_step)
            writer.add_scalar("Loss/mel", mel_loss.item(), global_step)
            writer.add_scalar("Loss/duration", dur_loss.item(), global_step)
            writer.add_scalar("Loss/pitch", pitch_loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "mel": f"{mel_loss.item():.4f}",
                "dur": f"{dur_loss.item():.4f}"
            })
            global_step += 1

        # Epoch 日志
        avg_loss = np.mean(epoch_losses["total"])
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 余弦退火重启：每个epoch结束后更新学习率
        scheduler.step()

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

        # 保存检查点（每5个epoch保存一次，减少IO压力）
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            if save_checkpoint(checkpoint_path, epoch + 1, model, optimizer, scheduler, scaler, global_step, avg_loss):
                print(f"Saved checkpoint: {checkpoint_path}")
                # 删除旧的checkpoint，只保留最新的2个
                checkpoints = sorted([f for f in os.listdir(config.CHECKPOINT_DIR)
                                     if f.startswith("checkpoint_epoch_") and f.endswith(".pt")])
                for old_ckpt in checkpoints[:-2]:
                    os.remove(os.path.join(config.CHECKPOINT_DIR, old_ckpt))

    # 保存最终模型
    final_path = os.path.join(config.CHECKPOINT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training completed! Saved final model to {final_path}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from specific checkpoint path")
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from latest checkpoint")
    args = parser.parse_args()
    train(args)
