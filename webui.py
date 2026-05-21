"""
歌声合成 Web 应用 - 美观现代界面
使用 Gradio 构建
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
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
is_loaded = False


def load_models():
    """加载模型"""
    global model, hifigan_onnx, hifigan_pytorch, stats, is_loaded

    if is_loaded:
        return True

    try:
        # 加载统计量
        stats_path = "data/processed/stats_train.npz"
        if os.path.exists(stats_path):
            s = np.load(stats_path)
            stats = {"mel_mean": float(s["mel_mean"]), "mel_std": float(s["mel_std"])}
        else:
            stats = {"mel_mean": 0.0, "mel_std": 1.0}

        # 加载 FastSpeech2
        ckpt_dir = "checkpoints"
        if os.path.exists(ckpt_dir):
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
            if ckpts:
                model = FastSpeech2SVS().to(config.DEVICE)
                checkpoint = torch.load(
                    os.path.join(ckpt_dir, ckpts[-1]),
                    map_location=config.DEVICE,
                    weights_only=False
                )
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()

        # 加载 HiFi-GAN ONNX
        try:
            import onnxruntime as ort
            onnx_path = "onnx_models/hifigan.onnx"
            if os.path.exists(onnx_path):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                hifigan_onnx = ort.InferenceSession(onnx_path, providers=providers)
        except:
            pass

        # 加载 HiFi-GAN PyTorch (备用)
        if hifigan_onnx is None:
            try:
                from hifi_gan import load_hifigan
                hifigan_pytorch = load_hifigan(
                    "checkpoints/hifigan/config.json",
                    "checkpoints/hifigan/generator_v1",
                    config.DEVICE
                )
                hifigan_pytorch.eval()
            except:
                pass

        is_loaded = True
        return True

    except Exception as e:
        print(f"Model load error: {e}")
        return False


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
    ids = [PHONEME_TO_ID.get(p, PHONEME_TO_ID["sp"]) for p in phonemes]
    return torch.LongTensor([ids]).to(config.DEVICE)


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


def smooth_mel(mel, window=3):
    """Mel谱平滑"""
    kernel = np.ones(window) / window
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=1, arr=mel
    )


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


def synthesize(text, pitch_shift, speed):
    """合成歌声"""
    global model, hifigan_onnx, hifigan_pytorch, stats

    if model is None:
        load_models()

    if model is None:
        return None, "❌ 模型加载失败，请检查 checkpoints 目录"

    if not text.strip():
        return None, "❌ 请输入歌词"

    try:
        # 转换音素
        phoneme_ids = text_to_phoneme_ids(text)

        # FastSpeech2 推理
        with torch.no_grad():
            mel_pred, dur_pred, _ = model(
                phoneme_ids,
                max_mel_len=1000,
                pitch_shift=pitch_shift,
                speed=speed
            )

        expected_len = int(dur_pred[0].sum().item())
        mel = mel_pred[0, :, :expected_len].cpu().numpy()

        # 反归一化
        mel = mel * stats["mel_std"] + stats["mel_mean"]
        mel = smooth_mel(mel, window=3)

        # HiFi-GAN 推理
        if hifigan_onnx:
            mel_input = mel[np.newaxis, :, :].astype(np.float32)
            audio = hifigan_onnx.run(None, {"mel_input": mel_input})[0].squeeze()
        elif hifigan_pytorch:
            with torch.no_grad():
                mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(config.DEVICE)
                audio = hifigan_pytorch(mel_tensor).squeeze().cpu().numpy()
        else:
            return None, "❌ 声码器未加载"

        audio = np.clip(audio, -1.0, 1.0)

        # 音频增强
        audio = enhance_audio(audio, config.SAMPLE_RATE)

        duration = len(audio) / config.SAMPLE_RATE
        info = f"✅ 合成成功！时长: {duration:.2f}s"

        return (config.SAMPLE_RATE, audio), info

    except Exception as e:
        return None, f"❌ 合成失败: {str(e)}"


# 自定义 CSS
custom_css = """
/* 全局样式 */
.gradio-container {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* 主容器 */
.main-container {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    padding: 40px !important;
    margin: 20px auto !important;
    max-width: 900px !important;
}

/* 标题 */
.title-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
}

.subtitle-text {
    text-align: center;
    color: #666;
    font-size: 1.1em !important;
    margin-bottom: 30px !important;
}

/* 输入框 */
.input-box textarea {
    border: 2px solid #e0e0e0 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    padding: 15px !important;
    transition: all 0.3s ease !important;
}

.input-box textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
}

/* 滑块 */
.slider-container {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

.slider-container label {
    font-weight: 600 !important;
    color: #333 !important;
}

/* 按钮 */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 40px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* 音频播放器 */
.audio-player {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* 示例按钮 */
.example-btn {
    background: #f0f4f8 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    margin: 5px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.example-btn:hover {
    background: #667eea !important;
    color: white !important;
    border-color: #667eea !important;
}

/* 状态信息 */
.status-info {
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}

/* 卡片效果 */
.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

/* 页脚 */
.footer {
    text-align: center;
    color: #888;
    font-size: 0.9em;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #eee;
}
"""


def create_ui():
    """创建界面"""

    with gr.Blocks(title="歌声合成系统") as demo:

        # 主容器
        with gr.Column(elem_classes=["main-container"]):

            # 标题
            gr.HTML("""
                <div style="text-align: center;">
                    <h1 class="title-text">🎵 歌声合成系统</h1>
                    <p class="subtitle-text">FastSpeech2 + HiFi-GAN | 中文歌声合成</p>
                </div>
            """)

            # 歌词输入卡片
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<h3 style="margin: 0 0 15px 0; color: #333;">📝 歌词输入</h3>')

                text_input = gr.Textbox(
                    label="",
                    placeholder="请输入中文歌词，如：好想能这样就白头到老",
                    value="好想能这样就白头到老",
                    lines=4,
                    elem_classes=["input-box"]
                )

                # 示例歌词
                gr.HTML('<p style="color: #666; margin: 10px 0;">💡 快速选择示例：</p>')
                with gr.Row():
                    examples = ["好想能这样就白头到老", "时间静止的美好", "为梦想狂都是戏里编的谎话", "说干就干聪明的人又怎能弄明白"]
                    for ex in examples:
                        gr.Button(ex[:8]+"..." if len(ex) > 8 else ex, elem_classes=["example-btn"]).click(
                            lambda t=ex: t, outputs=[text_input]
                        )

            # 参数调节卡片
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<h3 style="margin: 0 0 15px 0; color: #333;">🎛️ 参数调节</h3>')

                with gr.Row():
                    with gr.Column(scale=1):
                        pitch_slider = gr.Slider(
                            minimum=-12,
                            maximum=12,
                            value=0,
                            step=1,
                            label="🎵 音高偏移（半音）",
                            info="正值升高，负值降低"
                        )

                    with gr.Column(scale=1):
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="⚡ 播放速度",
                            info=">1 加快，<1 减慢"
                        )

            # 合成按钮
            with gr.Row():
                synth_btn = gr.Button(
                    "🎤 开始合成",
                    elem_classes=["primary-btn"]
                )

            # 结果显示
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<h3 style="margin: 0 0 15px 0; color: #333;">🔊 合成结果</h3>')

                status_output = gr.Textbox(
                    label="",
                    value="等待合成...",
                    interactive=False,
                    elem_classes=["status-info"]
                )

                audio_output = gr.Audio(
                    label="",
                    type="numpy",
                    interactive=False,
                    elem_classes=["audio-player"]
                )

            # 页脚
            gr.HTML("""
                <div class="footer">
                    <p>🚀 技术栈：FastSpeech2 (声学模型) + HiFi-GAN (声码器)</p>
                    <p>💡 提示：调整音高和语速参数可以获得不同的演唱效果</p>
                </div>
            """)

        # 事件绑定
        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, pitch_slider, speed_slider],
            outputs=[audio_output, status_output]
        )

    return demo


if __name__ == "__main__":
    # 预加载模型
    print("预加载模型...")
    threading.Thread(target=load_models, daemon=True).start()

    # 启动界面
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        css=custom_css
    )
