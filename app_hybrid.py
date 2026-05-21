"""
混合版桌面应用 - FastSpeech2 (PyTorch) + HiFi-GAN (ONNX)
体积减少约 50%，推理更快
"""
import os
import sys
import threading
import time
import subprocess
import webbrowser

# 设置路径（支持 PyInstaller 打包）
def get_base_path():
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，使用 exe 所在目录
        return os.path.dirname(sys.executable)
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

# 添加路径到 sys.path（确保能找到模块）
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

# 对于 PyInstaller，还需要检查 _MEIPASS
if hasattr(sys, '_MEIPASS'):
    meipass = sys._MEIPASS
    if meipass not in sys.path:
        sys.path.insert(0, meipass)

os.chdir(BASE_PATH)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import numpy as np
import soundfile as sf

from config import config, PHONEME_TO_ID
from models.fastspeech2 import FastSpeech2SVS

# 全局变量
model = None
hifigan_onnx = None
hifigan_pytorch = None
stats = None


def load_stats():
    """加载统计量"""
    global stats
    stats_path = os.path.join(BASE_PATH, "data/processed/stats_train.npz")
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        stats = {"mel_mean": float(s["mel_mean"]), "mel_std": float(s["mel_std"])}
    else:
        stats = {"mel_mean": 0.0, "mel_std": 1.0}


def load_hifigan_onnx():
    """加载 HiFi-GAN ONNX"""
    global hifigan_onnx
    try:
        import onnxruntime as ort
        onnx_path = os.path.join(BASE_PATH, "onnx_models/hifigan.onnx")
        if os.path.exists(onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            hifigan_onnx = ort.InferenceSession(onnx_path, providers=providers)
            return True
    except Exception as e:
        print(f"ONNX load failed: {e}")
    return False


def load_hifigan_pytorch():
    """加载 HiFi-GAN PyTorch"""
    global hifigan_pytorch
    try:
        from hifi_gan import load_hifigan
        hifigan_path = os.path.join(BASE_PATH, "checkpoints/hifigan")
        hifigan_pytorch = load_hifigan(
            os.path.join(hifigan_path, "config.json"),
            os.path.join(hifigan_path, "generator_v1"),
            config.DEVICE
        )
        hifigan_pytorch.eval()
        return True
    except Exception as e:
        print(f"PyTorch HiFi-GAN load failed: {e}")
    return False


def load_model(checkpoint_path):
    """加载 FastSpeech2 模型"""
    global model
    model = FastSpeech2SVS().to(config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()


def smooth_mel(mel, window=3):
    """Mel谱平滑"""
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=mel)


def normalize_audio(audio, target_db=-20.0):
    """音频音量归一化"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    audio = audio * gain
    return np.clip(audio, -1.0, 1.0)


def enhance_audio(audio, sample_rate=22050):
    """音频后处理增强 - 让声音更自然"""
    audio = compressor(audio, threshold=0.3, ratio=3.0)
    audio = add_natural_noise(audio, noise_level=0.001)
    audio = add_reverb(audio, sample_rate, decay=0.2, delay=0.015)
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


def text_to_phoneme_ids(text):
    """文本转音素ID"""
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
    return torch.LongTensor([PHONEME_TO_ID.get(p, PHONEME_TO_ID["sp"]) for p in phonemes]).to(config.DEVICE)


def split_pinyin(py):
    """拆分拼音"""
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
    py_clean = ''.join(c for c in py if not c.isdigit())
    for init in sorted(initials, key=len, reverse=True):
        if py_clean.startswith(init):
            final = py_clean[len(init):]
            return init, final if final else None
    return None, py_clean


def synthesize(text, pitch_shift, speed, progress_callback=None):
    """合成歌声（混合推理）"""
    global model, hifigan_onnx, hifigan_pytorch, stats

    if model is None:
        return None, "Model not loaded"

    if hifigan_onnx is None and hifigan_pytorch is None:
        return None, "Vocoder not loaded"

    if not text.strip():
        return None, "Please enter lyrics"

    try:
        if progress_callback:
            progress_callback("Converting phonemes...")
        phoneme_ids = text_to_phoneme_ids(text)

        if progress_callback:
            progress_callback("FastSpeech2 inference...")
        with torch.no_grad():
            mel_pred, dur_pred, _ = model(phoneme_ids, max_mel_len=1000, pitch_shift=pitch_shift, speed=speed)
            expected_len = int(dur_pred[0].sum().item())
            mel = mel_pred[0, :, :expected_len].cpu().numpy()

        if progress_callback:
            progress_callback("HiFi-GAN inference...")
        mel = mel * stats["mel_std"] + stats["mel_mean"]
        mel = smooth_mel(mel, window=3)

        # ONNX 或 PyTorch 推理
        if hifigan_onnx:
            mel_input = mel[np.newaxis, :, :].astype(np.float32)
            audio = hifigan_onnx.run(None, {"mel_input": mel_input})[0].squeeze()
        else:
            with torch.no_grad():
                mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(config.DEVICE)
                audio = hifigan_pytorch(mel_tensor).squeeze().cpu().numpy()

        audio = np.clip(audio, -1.0, 1.0)

        # 音频增强
        if progress_callback:
            progress_callback("Enhancing audio...")
        audio = enhance_audio(audio, config.SAMPLE_RATE)

        # 保存
        output_dir = os.path.join(BASE_PATH, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.wav")
        sf.write(output_path, audio, config.SAMPLE_RATE)

        duration = expected_len * 256 / 22050
        vocoder_type = "ONNX" if hifigan_onnx else "PyTorch"
        return output_path, f"Done! Duration: {duration:.2f}s (Vocoder: {vocoder_type})"

    except Exception as e:
        return None, f"Error: {str(e)}"


class ModernStyle:
    """现代风格配置"""
    BG_COLOR = "#f0f4f8"
    CARD_BG = "#ffffff"
    PRIMARY = "#3b82f6"
    PRIMARY_HOVER = "#2563eb"
    SUCCESS = "#22c55e"
    TEXT_PRIMARY = "#1e293b"
    TEXT_SECONDARY = "#64748b"
    BORDER = "#e2e8f0"
    ACCENT = "#8b5cf6"


class HybridApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Singing Voice Synthesis (Hybrid)")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg=ModernStyle.BG_COLOR)
        self.center_window(900, 700)

        self.is_ready = False
        self.is_synthetizing = False
        self.current_audio = None

        self.setup_styles()
        self.create_ui()
        threading.Thread(target=self.load_models, daemon=True).start()

    def center_window(self, width, height):
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

    def create_ui(self):
        main_frame = ttk.Frame(self.root, style='Card.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # 标题
        title_frame = tk.Frame(main_frame, bg=ModernStyle.BG_COLOR)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(title_frame, text="Singing Voice Synthesis",
                              font=('Segoe UI', 28, 'bold'),
                              fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.BG_COLOR)
        title_label.pack()

        subtitle_label = tk.Label(title_frame, text="Hybrid Mode: PyTorch + ONNX | Faster & Smaller",
                                  font=('Segoe UI', 12),
                                  fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        subtitle_label.pack(pady=(5, 10))

        # 状态栏
        self.status_frame = tk.Frame(title_frame, bg=ModernStyle.BG_COLOR)
        self.status_frame.pack(pady=5)

        self.status_icon = tk.Label(self.status_frame, text="...", font=('Segoe UI Emoji', 16),
                                    bg=ModernStyle.BG_COLOR)
        self.status_icon.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.status_frame, text="Loading models...",
                                     font=('Segoe UI', 11),
                                     fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        self.status_label.pack(side=tk.LEFT)

        # 歌词输入
        self.create_lyrics_card(main_frame)

        # 参数控制
        self.create_params_card(main_frame)

        # 操作区域
        self.create_action_area(main_frame)

        # 结果区域
        self.create_result_area(main_frame)

    def create_lyrics_card(self, parent):
        card = tk.Frame(parent, bg=ModernStyle.CARD_BG, relief=tk.FLAT)
        card.pack(fill=tk.X, pady=10)
        card.configure(highlightbackground=ModernStyle.BORDER, highlightthickness=1)

        title_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        title_frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        title_label = tk.Label(title_frame, text="Lyrics Input",
                              font=('Segoe UI', 14, 'bold'),
                              fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG)
        title_label.pack(anchor=tk.W)

        input_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.lyrics_text = tk.Text(input_frame, height=4, font=('Segoe UI', 12),
                                   wrap=tk.WORD, relief=tk.FLAT, bg="#f8fafc",
                                   highlightthickness=1, highlightcolor=ModernStyle.PRIMARY,
                                   highlightbackground=ModernStyle.BORDER)
        self.lyrics_text.pack(fill=tk.X, pady=(0, 10))
        self.lyrics_text.insert("1.0", "好想能这样就白头到老")

        # 示例
        example_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        example_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        tk.Label(example_frame, text="Quick select:", font=('Segoe UI', 10),
                fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        examples = ["好想能这样就白头到老", "时间静止的美好", "为梦想狂都是戏里编的谎话"]
        for ex in examples:
            btn = tk.Button(example_frame, text=ex[:8]+"..." if len(ex) > 8 else ex,
                           font=('Segoe UI', 9), relief=tk.FLAT,
                           bg="#f1f5f9", fg=ModernStyle.TEXT_SECONDARY,
                           cursor="hand2", padx=8, pady=2,
                           command=lambda t=ex: self.set_lyrics(t))
            btn.pack(side=tk.LEFT, padx=5)

    def create_params_card(self, parent):
        card = tk.Frame(parent, bg=ModernStyle.CARD_BG, relief=tk.FLAT)
        card.pack(fill=tk.X, pady=10)
        card.configure(highlightbackground=ModernStyle.BORDER, highlightthickness=1)

        title_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        title_frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        title_label = tk.Label(title_frame, text="Parameters",
                              font=('Segoe UI', 14, 'bold'),
                              fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG)
        title_label.pack(anchor=tk.W)

        params_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        params_frame.pack(fill=tk.X, padx=20, pady=15)

        # 音高
        pitch_frame = tk.Frame(params_frame, bg=ModernStyle.CARD_BG)
        pitch_frame.pack(fill=tk.X, pady=5)

        tk.Label(pitch_frame, text="Pitch Shift", font=('Segoe UI', 11),
                fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        self.pitch_value_label = tk.Label(pitch_frame, text="0 semitones",
                                          font=('Segoe UI', 11, 'bold'),
                                          fg=ModernStyle.PRIMARY, bg=ModernStyle.CARD_BG)
        self.pitch_value_label.pack(side=tk.RIGHT)

        self.pitch_var = tk.DoubleVar(value=0)
        self.pitch_slider = ttk.Scale(params_frame, from_=-12, to=12, variable=self.pitch_var,
                                      orient=tk.HORIZONTAL, command=self.on_pitch_change)
        self.pitch_slider.pack(fill=tk.X, pady=5)

        # 语速
        speed_frame = tk.Frame(params_frame, bg=ModernStyle.CARD_BG)
        speed_frame.pack(fill=tk.X, pady=(15, 5))

        tk.Label(speed_frame, text="Speed", font=('Segoe UI', 11),
                fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        self.speed_value_label = tk.Label(speed_frame, text="1.0x",
                                          font=('Segoe UI', 11, 'bold'),
                                          fg=ModernStyle.SUCCESS, bg=ModernStyle.CARD_BG)
        self.speed_value_label.pack(side=tk.RIGHT)

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = ttk.Scale(params_frame, from_=0.5, to=2.0, variable=self.speed_var,
                                      orient=tk.HORIZONTAL, command=self.on_speed_change)
        self.speed_slider.pack(fill=tk.X, pady=5)

    def create_action_area(self, parent):
        action_frame = tk.Frame(parent, bg=ModernStyle.BG_COLOR)
        action_frame.pack(fill=tk.X, pady=20)

        self.synth_button = tk.Button(action_frame, text="Start Synthesis",
                                      font=('Segoe UI', 14, 'bold'),
                                      bg=ModernStyle.PRIMARY, fg="white",
                                      relief=tk.FLAT, padx=40, pady=15,
                                      cursor="hand2", state=tk.DISABLED,
                                      command=self.on_synthesize)
        self.synth_button.pack()

        self.progress_label = tk.Label(action_frame, text="",
                                       font=('Segoe UI', 11),
                                       fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        self.progress_label.pack(pady=10)

    def create_result_area(self, parent):
        self.result_frame = tk.Frame(parent, bg=ModernStyle.BG_COLOR)
        self.result_frame.pack(fill=tk.X, pady=10)

        self.result_label = tk.Label(self.result_frame, text="",
                                     font=('Segoe UI', 12),
                                     fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.BG_COLOR)
        self.result_label.pack()

        self.button_frame = tk.Frame(self.result_frame, bg=ModernStyle.BG_COLOR)
        self.button_frame.pack(pady=15)

        self.play_button = tk.Button(self.button_frame, text="Play",
                                     font=('Segoe UI', 11),
                                     bg=ModernStyle.SUCCESS, fg="white",
                                     relief=tk.FLAT, padx=25, pady=10,
                                     cursor="hand2", state=tk.DISABLED,
                                     command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=10)

        self.open_folder_button = tk.Button(self.button_frame, text="Open Folder",
                                            font=('Segoe UI', 11),
                                            bg="#64748b", fg="white",
                                            relief=tk.FLAT, padx=25, pady=10,
                                            cursor="hand2", state=tk.DISABLED,
                                            command=self.open_output_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=10)

    def load_models(self):
        """加载模型"""
        try:
            self.update_status("...", "Loading statistics...")
            load_stats()
            time.sleep(0.2)

            self.update_status("...", "Loading FastSpeech2...")
            ckpt_dir = os.path.join(BASE_PATH, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
                if ckpts:
                    load_model(os.path.join(ckpt_dir, ckpts[-1]))
                    self.update_status("...", f"FastSpeech2: {ckpts[-1]}")
            time.sleep(0.2)

            self.update_status("...", "Loading HiFi-GAN...")
            if load_hifigan_onnx():
                self.update_status("[OK]", "Ready (ONNX vocoder)")
            elif load_hifigan_pytorch():
                self.update_status("[OK]", "Ready (PyTorch vocoder)")
            else:
                self.update_status("[!]", "Vocoder not loaded")
                return

            self.is_ready = True
            self.synth_button.configure(state=tk.NORMAL)

        except Exception as e:
            self.update_status("[X]", f"Error: {str(e)[:20]}")

    def update_status(self, icon, text):
        self.status_icon.configure(text=icon)
        self.status_label.configure(text=text)

    def on_pitch_change(self, value):
        v = int(float(value))
        self.pitch_value_label.configure(text=f"{v:+d} semitones" if v != 0 else "0 semitones")

    def on_speed_change(self, value):
        v = float(value)
        self.speed_value_label.configure(text=f"{v:.1f}x")

    def set_lyrics(self, text):
        self.lyrics_text.delete("1.0", tk.END)
        self.lyrics_text.insert("1.0", text)

    def on_synthesize(self):
        if self.is_synthetizing or not self.is_ready:
            return

        self.is_synthetizing = True
        self.synth_button.configure(state=tk.DISABLED, text="Synthesizing...")
        self.progress_label.configure(text="Processing...")

        def do_synth():
            def progress_cb(msg):
                self.progress_label.configure(text=msg)

            output_path, message = synthesize(
                self.lyrics_text.get("1.0", tk.END).strip(),
                self.pitch_var.get(),
                self.speed_var.get(),
                progress_cb
            )

            if output_path:
                self.current_audio = output_path
                self.result_label.configure(text=message)
                self.play_button.configure(state=tk.NORMAL)
                self.open_folder_button.configure(state=tk.NORMAL)
            else:
                self.result_label.configure(text=f"[Error] {message}")

            self.is_synthetizing = False
            self.synth_button.configure(state=tk.NORMAL, text="Start Synthesis")
            self.progress_label.configure(text="")

        threading.Thread(target=do_synth, daemon=True).start()

    def play_audio(self):
        if self.current_audio and os.path.exists(self.current_audio):
            os.startfile(self.current_audio)

    def open_output_folder(self):
        folder = os.path.join(BASE_PATH, "outputs")
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.startfile(folder)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = HybridApp()
    app.run()
