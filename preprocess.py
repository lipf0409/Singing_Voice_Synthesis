"""数据预处理脚本"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from config import config
from data.dataset import load_opencpop_data
from utils.audio import get_mel_from_wav
from utils.pitch import get_f0, norm_pitch

def preprocess_opencpop(data_dir, output_dir):
    """预处理OpenCPop数据集"""
    wav_dir = os.path.join(data_dir, "wavs")
    trans_file = os.path.join(data_dir, "transcriptions.txt")
    train_file = os.path.join(data_dir, "train.txt")
    test_file = os.path.join(data_dir, "test.txt")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "f0"), exist_ok=True)

    # 加载数据列表
    train_data = load_opencpop_data(trans_file, train_file)
    test_data = load_opencpop_data(trans_file, test_file)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    def process_split(data, split_name):
        stats = {"mel": [], "f0": []}
        for item in tqdm(data, desc=f"Processing {split_name}"):
            wav_name = item["wav_name"]
            wav_path = os.path.join(wav_dir, f"{wav_name}.wav")

            if not os.path.exists(wav_path):
                continue

            try:
                mel, wav = get_mel_from_wav(wav_path)
                f0 = get_f0(wav)

                # 对齐长度
                n_frames = mel.shape[1]
                if len(f0) < n_frames:
                    f0 = np.pad(f0, (0, n_frames - len(f0)))
                else:
                    f0 = f0[:n_frames]

                # 保存
                np.save(os.path.join(output_dir, "mel", f"{wav_name}.npy"), mel)
                np.save(os.path.join(output_dir, "f0", f"{wav_name}.npy"), f0)

                stats["mel"].append(mel)
                stats["f0"].append(f0)
            except Exception as e:
                print(f"Error processing {wav_name}: {e}")

        # 计算统计信息用于归一化
        if stats["mel"]:
            all_mel = np.concatenate([m.flatten() for m in stats["mel"]])
            all_f0 = np.concatenate([f[f > 0] for f in stats["f0"]])

            mel_mean = np.mean(all_mel)
            mel_std = np.std(all_mel)
            # F0 使用 log 域统计量（更适合 F0 的偏态分布）
            f0_mean = np.mean(np.log(all_f0)) if len(all_f0) > 0 else 0
            f0_std = np.std(np.log(all_f0)) if len(all_f0) > 0 else 1

            np.savez(
                os.path.join(output_dir, f"stats_{split_name}.npz"),
                mel_mean=mel_mean,
                mel_std=mel_std,
                f0_mean=f0_mean,
                f0_std=f0_std
            )
            print(f"{split_name} stats: mel_mean={mel_mean:.4f}, mel_std={mel_std:.4f}, f0_mean={f0_mean:.4f}, f0_std={f0_std:.4f}")

    process_split(train_data, "train")
    process_split(test_data, "test")
    print("Preprocessing done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="segments")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    preprocess_opencpop(args.data_dir, args.output_dir)
