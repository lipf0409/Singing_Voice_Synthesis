"""
歌声合成桌面应用 - 独立可执行版本
"""
import os
import sys
import threading
import time
import subprocess
import webbrowser

# 设置路径
def get_base_path():
    """获取基础路径"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
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
hifigan = None
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


def load_vocoder():
    """加载声码器"""
    global hifigan
    try:
        from hifi_gan import load_hifigan
        hifigan_path = os.path.join(BASE_PATH, "checkpoints/hifigan")
        config_path = os.path.join(hifigan_path, "config.json")
        ckpt_path = os.path.join(hifigan_path, "generator_v1")
        if os.path.exists(config_path) and os.path.exists(ckpt_path):
            hifigan = load_hifigan(config_path, ckpt_path, config.DEVICE)
            return True
    except Exception as e:
        print(f"声码器加载失败: {e}")
    return False


def load_model(checkpoint_path):
    """加载模型"""
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
    return torch.LongTensor([PHONEME_TO_ID.get(p, PHONEME_TO_ID["sp"]) for p in phonemes]).unsqueeze(0)


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
    """合成歌声"""
    global model, hifigan, stats

    if model is None or hifigan is None:
        return None, "模型未加载"

    if not text.strip():
        return None, "请输入歌词"

    try:
        if progress_callback:
            progress_callback("转换音素...")
        phoneme_ids = text_to_phoneme_ids(text).to(config.DEVICE)

        if progress_callback:
            progress_callback("模型推理...")
        with torch.no_grad():
            mel_pred, dur_pred, _ = model(phoneme_ids, max_mel_len=1000, pitch_shift=pitch_shift, speed=speed)
            expected_len = int(dur_pred[0].sum().item())
            mel = mel_pred[0, :, :expected_len].cpu().numpy()

        if progress_callback:
            progress_callback("声码器合成...")
        mel = mel * stats["mel_std"] + stats["mel_mean"]
        mel = smooth_mel(mel, window=3)

        with torch.no_grad():
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(config.DEVICE)
            audio = hifigan(mel_tensor).squeeze().cpu().numpy()

        audio = np.clip(audio, -1.0, 1.0)

        # 保存
        output_dir = os.path.join(BASE_PATH, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.wav")
        sf.write(output_path, audio, config.SAMPLE_RATE)

        duration = expected_len * 256 / 22050
        return output_path, f"合成完成！时长: {duration:.2f}s"

    except Exception as e:
        return None, f"合成出错: {str(e)}"


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


class SingingVoiceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 歌声合成系统")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg=ModernStyle.BG_COLOR)

        # 居中窗口
        self.center_window(900, 700)

        # 状态变量
        self.is_ready = False
        self.is_synthetizing = False
        self.current_audio = None

        # 设置样式
        self.setup_styles()

        # 创建UI
        self.create_ui()

        # 加载模型
        threading.Thread(target=self.load_models, daemon=True).start()

    def center_window(self, width, height):
        """窗口居中"""
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 配置样式
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 28, 'bold'),
                       foreground=ModernStyle.TEXT_PRIMARY, background=ModernStyle.BG_COLOR)
        style.configure('Subtitle.TLabel', font=('Microsoft YaHei UI', 12),
                       foreground=ModernStyle.TEXT_SECONDARY, background=ModernStyle.BG_COLOR)
        style.configure('Status.TLabel', font=('Microsoft YaHei UI', 11),
                       foreground=ModernStyle.TEXT_SECONDARY, background=ModernStyle.BG_COLOR)
        style.configure('Card.TFrame', background=ModernStyle.CARD_BG)
        style.configure('CardTitle.TLabel', font=('Microsoft YaHei UI', 14, 'bold'),
                       foreground=ModernStyle.TEXT_PRIMARY, background=ModernStyle.CARD_BG)
        style.configure('Result.TLabel', font=('Microsoft YaHei UI', 12),
                       foreground=ModernStyle.TEXT_PRIMARY, background=ModernStyle.BG_COLOR)

        # 按钮样式
        style.configure('Primary.TButton', font=('Microsoft YaHei UI', 12, 'bold'),
                       padding=(30, 15))
        style.configure('Success.TButton', font=('Microsoft YaHei UI', 11),
                       padding=(20, 10))

    def create_ui(self):
        """创建UI"""
        # 主容器
        main_frame = ttk.Frame(self.root, style='Card.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # 标题区域
        title_frame = tk.Frame(main_frame, bg=ModernStyle.BG_COLOR)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(title_frame, text="🎵 歌声合成系统",
                              font=('Microsoft YaHei UI', 28, 'bold'),
                              fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.BG_COLOR)
        title_label.pack()

        subtitle_label = tk.Label(title_frame, text="FastSpeech2 + HiFi-GAN | 中文歌声合成",
                                  font=('Microsoft YaHei UI', 12),
                                  fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        subtitle_label.pack(pady=(5, 10))

        # 状态栏
        self.status_frame = tk.Frame(title_frame, bg=ModernStyle.BG_COLOR)
        self.status_frame.pack(pady=5)

        self.status_icon = tk.Label(self.status_frame, text="⏳", font=('Segoe UI Emoji', 16),
                                    bg=ModernStyle.BG_COLOR)
        self.status_icon.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.status_frame, text="正在加载模型...",
                                     font=('Microsoft YaHei UI', 11),
                                     fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        self.status_label.pack(side=tk.LEFT)

        # 歌词输入卡片
        self.create_lyrics_card(main_frame)

        # 参数控制卡片
        self.create_params_card(main_frame)

        # 操作区域
        self.create_action_area(main_frame)

        # 结果区域
        self.create_result_area(main_frame)

    def create_card(self, parent, title, icon):
        """创建卡片"""
        card = tk.Frame(parent, bg=ModernStyle.CARD_BG, relief=tk.FLAT)
        card.pack(fill=tk.X, pady=10)

        # 添加阴影效果
        card.configure(highlightbackground=ModernStyle.BORDER, highlightthickness=1)

        # 标题
        title_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        title_frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        title_label = tk.Label(title_frame, text=f"{icon} {title}",
                              font=('Microsoft YaHei UI', 14, 'bold'),
                              fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG)
        title_label.pack(anchor=tk.W)

        return card

    def create_lyrics_card(self, parent):
        """歌词输入卡片"""
        card = self.create_card(parent, "歌词输入", "📝")

        # 输入框
        input_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.lyrics_text = tk.Text(input_frame, height=4, font=('Microsoft YaHei UI', 12),
                                   wrap=tk.WORD, relief=tk.FLAT, bg="#f8fafc",
                                   highlightthickness=1, highlightcolor=ModernStyle.PRIMARY,
                                   highlightbackground=ModernStyle.BORDER)
        self.lyrics_text.pack(fill=tk.X, pady=(0, 10))
        self.lyrics_text.insert("1.0", "好想能这样就白头到老")

        # 示例歌词
        example_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        example_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        tk.Label(example_frame, text="💡 快速选择：", font=('Microsoft YaHei UI', 10),
                fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        examples = ["好想能这样就白头到老", "时间静止的美好", "为梦想狂都是戏里编的谎话"]
        for ex in examples:
            btn = tk.Button(example_frame, text=ex[:8]+"..." if len(ex) > 8 else ex,
                           font=('Microsoft YaHei UI', 9), relief=tk.FLAT,
                           bg="#f1f5f9", fg=ModernStyle.TEXT_SECONDARY,
                           cursor="hand2", padx=8, pady=2,
                           command=lambda t=ex: self.set_lyrics(t))
            btn.pack(side=tk.LEFT, padx=5)

    def create_params_card(self, parent):
        """参数控制卡片"""
        card = self.create_card(parent, "参数调节", "🎛️")

        params_frame = tk.Frame(card, bg=ModernStyle.CARD_BG)
        params_frame.pack(fill=tk.X, padx=20, pady=15)

        # 音高控制
        pitch_frame = tk.Frame(params_frame, bg=ModernStyle.CARD_BG)
        pitch_frame.pack(fill=tk.X, pady=5)

        tk.Label(pitch_frame, text="🎵 音高偏移", font=('Microsoft YaHei UI', 11),
                fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        self.pitch_value_label = tk.Label(pitch_frame, text="0 半音",
                                          font=('Microsoft YaHei UI', 11, 'bold'),
                                          fg=ModernStyle.PRIMARY, bg=ModernStyle.CARD_BG)
        self.pitch_value_label.pack(side=tk.RIGHT)

        self.pitch_var = tk.DoubleVar(value=0)
        self.pitch_slider = ttk.Scale(params_frame, from_=-12, to=12, variable=self.pitch_var,
                                      orient=tk.HORIZONTAL, command=self.on_pitch_change)
        self.pitch_slider.pack(fill=tk.X, pady=5)

        # 语速控制
        speed_frame = tk.Frame(params_frame, bg=ModernStyle.CARD_BG)
        speed_frame.pack(fill=tk.X, pady=(15, 5))

        tk.Label(speed_frame, text="⚡ 播放速度", font=('Microsoft YaHei UI', 11),
                fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.CARD_BG).pack(side=tk.LEFT)

        self.speed_value_label = tk.Label(speed_frame, text="1.0x",
                                          font=('Microsoft YaHei UI', 11, 'bold'),
                                          fg=ModernStyle.SUCCESS, bg=ModernStyle.CARD_BG)
        self.speed_value_label.pack(side=tk.RIGHT)

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = ttk.Scale(params_frame, from_=0.5, to=2.0, variable=self.speed_var,
                                      orient=tk.HORIZONTAL, command=self.on_speed_change)
        self.speed_slider.pack(fill=tk.X, pady=5)

    def create_action_area(self, parent):
        """操作区域"""
        action_frame = tk.Frame(parent, bg=ModernStyle.BG_COLOR)
        action_frame.pack(fill=tk.X, pady=20)

        # 合成按钮
        self.synth_button = tk.Button(action_frame, text="🎤  开始合成",
                                      font=('Microsoft YaHei UI', 14, 'bold'),
                                      bg=ModernStyle.PRIMARY, fg="white",
                                      relief=tk.FLAT, padx=40, pady=15,
                                      cursor="hand2", state=tk.DISABLED,
                                      command=self.on_synthesize)
        self.synth_button.pack()

        # 进度标签
        self.progress_label = tk.Label(action_frame, text="",
                                       font=('Microsoft YaHei UI', 11),
                                       fg=ModernStyle.TEXT_SECONDARY, bg=ModernStyle.BG_COLOR)
        self.progress_label.pack(pady=10)

    def create_result_area(self, parent):
        """结果区域"""
        self.result_frame = tk.Frame(parent, bg=ModernStyle.BG_COLOR)
        self.result_frame.pack(fill=tk.X, pady=10)

        self.result_label = tk.Label(self.result_frame, text="",
                                     font=('Microsoft YaHei UI', 12),
                                     fg=ModernStyle.TEXT_PRIMARY, bg=ModernStyle.BG_COLOR)
        self.result_label.pack()

        # 按钮区域
        self.button_frame = tk.Frame(self.result_frame, bg=ModernStyle.BG_COLOR)
        self.button_frame.pack(pady=15)

        self.play_button = tk.Button(self.button_frame, text="▶️  播放",
                                     font=('Microsoft YaHei UI', 11),
                                     bg=ModernStyle.SUCCESS, fg="white",
                                     relief=tk.FLAT, padx=25, pady=10,
                                     cursor="hand2", state=tk.DISABLED,
                                     command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=10)

        self.open_folder_button = tk.Button(self.button_frame, text="📁  打开目录",
                                            font=('Microsoft YaHei UI', 11),
                                            bg="#64748b", fg="white",
                                            relief=tk.FLAT, padx=25, pady=10,
                                            cursor="hand2", state=tk.DISABLED,
                                            command=self.open_output_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=10)

    def load_models(self):
        """加载模型"""
        try:
            self.update_status("⏳", "加载配置...")
            load_stats()
            time.sleep(0.3)

            self.update_status("⏳", "加载声码器...")
            if not load_vocoder():
                self.update_status("❌", "声码器加载失败")
                return
            time.sleep(0.3)

            self.update_status("⏳", "加载声学模型...")
            ckpt_dir = os.path.join(BASE_PATH, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
                if ckpts:
                    load_model(os.path.join(ckpt_dir, ckpts[-1]))
                    self.update_status("✅", f"就绪 | {ckpts[-1]}")
                    self.is_ready = True
                    self.synth_button.configure(state=tk.NORMAL)
                    return

            self.update_status("❌", "未找到模型文件")

        except Exception as e:
            self.update_status("❌", f"加载失败: {str(e)[:20]}")

    def update_status(self, icon, text):
        """更新状态"""
        self.status_icon.configure(text=icon)
        self.status_label.configure(text=text)

    def on_pitch_change(self, value):
        """音高变化"""
        v = int(float(value))
        self.pitch_value_label.configure(text=f"{v:+d} 半音" if v != 0 else "0 半音")

    def on_speed_change(self, value):
        """语速变化"""
        v = float(value)
        self.speed_value_label.configure(text=f"{v:.1f}x")

    def set_lyrics(self, text):
        """设置歌词"""
        self.lyrics_text.delete("1.0", tk.END)
        self.lyrics_text.insert("1.0", text)

    def on_synthesize(self):
        """合成"""
        if self.is_synthetizing or not self.is_ready:
            return

        self.is_synthetizing = True
        self.synth_button.configure(state=tk.DISABLED, text="⏳ 合成中...")
        self.progress_label.configure(text="准备合成...")

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
                self.result_label.configure(text=f"✨ {message}")
                self.play_button.configure(state=tk.NORMAL)
                self.open_folder_button.configure(state=tk.NORMAL)
            else:
                self.result_label.configure(text=f"❌ {message}")

            self.is_synthetizing = False
            self.synth_button.configure(state=tk.NORMAL, text="🎤  开始合成")
            self.progress_label.configure(text="")

        threading.Thread(target=do_synth, daemon=True).start()

    def play_audio(self):
        """播放音频"""
        if self.current_audio and os.path.exists(self.current_audio):
            os.startfile(self.current_audio)

    def open_output_folder(self):
        """打开输出目录"""
        folder = os.path.join(BASE_PATH, "outputs")
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.startfile(folder)

    def run(self):
        """运行应用"""
        self.root.mainloop()


if __name__ == "__main__":
    app = SingingVoiceApp()
    app.run()
