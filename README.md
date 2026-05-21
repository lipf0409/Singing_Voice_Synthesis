# 歌声合成系统

基于 FastSpeech2 + HiFi-GAN 的中文歌声合成系统

## 项目简介

本项目是一个端到端的歌声合成（Singing Voice Synthesis, SVS）系统，使用 Python + PyTorch 实现。系统基于 OpenCPop 数据集训练，采用 FastSpeech2 作为声学模型预测 Mel 谱，HiFi-GAN 作为神经声码器生成高质量音频。

### 主要特性

- 中文歌声合成：支持任意中文歌词输入
- 音高控制：支持 -12 ~ +12 半音偏移
- 语速控制：支持 0.5x ~ 2.0x 速度调节
- 桌面应用：现代化 GUI 界面，开箱即用
- 混合推理：PyTorch + ONNX，体积减少 50%

## 快速开始

### 方式一：Web 界面（推荐）

```bash
# 双击运行
run_webui.bat

# 或命令行
python webui.py
```

浏览器访问：http://127.0.0.1:7860

### 方式二：桌面应用

```bash
# 双击运行
run.bat

# 或命令行
python app_hybrid.py
```

### 方式三：命令行推理

```bash
# 混合模式（推荐，更快更小）
python inference_hybrid.py --text "好想能这样就白头到老"

# PyTorch 模式
python inference.py --text "好想能这样就白头到老"
```

## 安装部署

### 1. 安装依赖

```bash
# 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 安装 ONNX Runtime（混合模式需要）
pip install onnxruntime-gpu
```

### 2. 准备数据

1. 下载 OpenCPop 数据集
2. 解压 segments.zip 到 segments/ 目录

### 3. 预处理数据

```bash
python preprocess.py --data_dir segments --output_dir data/processed
```

### 4. 训练模型

```bash
python train.py
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --text | 好想能这样就白头到老 | 要合成的歌词文本 |
| --output | output.wav | 输出音频路径 |
| --pitch_shift | 0.0 | 音高偏移（半音），正值升高，负值降低 |
| --speed | 1.0 | 语速系数，>1 加快，<1 减慢 |

### 使用示例

```bash
# 升高 2 个半音
python inference_hybrid.py --text "我爱你中国" --pitch_shift 2.0

# 加快语速 1.5 倍
python inference_hybrid.py --text "我爱你中国" --speed 1.5

# 降低 3 个半音并减慢语速
python inference_hybrid.py --text "我爱你中国" --pitch_shift -3.0 --speed 0.8
```

## 项目结构

```
Singing_Voice_Synthesis/
├── config.py              # 全局配置
├── app.py                 # 桌面应用 (PyTorch)
├── app_hybrid.py          # 桌面应用 (混合模式)
├── webui.py               # Web 界面 (Gradio)
├── inference.py           # 推理脚本 (PyTorch)
├── inference_hybrid.py    # 推理脚本 (混合模式)
├── train.py               # 训练脚本
├── preprocess.py          # 预处理脚本
│
├── run_webui.bat          # 启动 Web 界面
├── run.bat                # 启动桌面应用
├── build.bat              # 打包 PyTorch 版本
├── build_hybrid.bat       # 打包混合版本
│
├── models/
│   ├── fastspeech2.py     # FastSpeech2 模型
│   ├── modules.py         # 网络组件
│   └── loss.py            # 损失函数
│
├── hifi_gan/
│   └── models.py          # HiFi-GAN 声码器
│
├── data/
│   └── dataset.py         # 数据集加载
│
├── utils/
│   ├── audio.py           # 音频处理
│   └── pitch.py           # 音高提取
│
├── onnx_models/           # ONNX 模型
│   └── hifigan.onnx
│
├── checkpoints/           # 模型权重
└── outputs/               # 输出音频
```

## 模型架构

### FastSpeech2 声学模型

```
歌词 -> 拼音 -> 音素嵌入 -> Encoder -> Duration/Pitch Predictor -> Length Regulator -> Decoder -> Mel谱
```

- Encoder: 6 层 Transformer FFT Block (384 dim, 4 heads)
- Decoder: 6 层 Transformer FFT Block (384 dim, 4 heads)
- Duration Predictor: 预测音素时长
- Pitch Predictor: 预测音高曲线

### 推理优化

| 优化项 | 说明 |
|--------|------|
| 温度采样 | 时长预测添加随机扰动，增加自然度 |
| 音高平滑 | 帧间音高曲线平滑，减少抖动 |
| Mel谱平滑 | 频谱帧间平滑，减少噪声 |
| 效率优化 | torch.repeat_interleave 批量处理 |

## 打包发布

### 混合版（推荐）

```bash
build_hybrid.bat
```

输出：dist_hybrid/SingingVoiceSynthesis_Hybrid.exe

特点：
- 体积：约2.0 GB（包含 PyTorch）
- HiFi-GAN 使用 ONNX 加速
- 推理速度提升 20-30%

### 发布包结构

```
歌声合成系统/
├── SingingVoiceSynthesis_Hybrid.exe    # 主程序
├── checkpoints/                         # 模型权重
│   ├── checkpoint_epoch_xxx.pt
│   └── hifigan/
├── onnx_models/                         # ONNX 模型
│   └── hifigan.onnx
└── data/processed/                      # 配置文件
    └── stats_train.npz
```

## 部署方式对比

| 方式 | 体积 | 推理速度 | 说明 |
|------|------|----------|------|
| Web 界面 | 无需打包 | 基准 | run_webui.bat，浏览器访问 |
| 桌面应用 | 约2.0 GB | 基准 | run.bat / app_hybrid.py |
| 混合版 EXE | 约2.0 GB | +20-30% | build_hybrid.bat |

## HiFi-GAN 声码器

### 下载预训练模型

1. 访问 HiFi-GAN 官方仓库
2. 下载预训练模型（推荐 V1 版本）
3. 放入 checkpoints/hifigan/

## 注意事项

1. 数据格式：transcriptions.txt 格式为 wav_name|lyrics|phonemes|notes|...

2. 音高提取：推荐安装 praat-parselmouth：
   ```bash
   pip install praat-parselmouth
   ```

3. 训练时间：RTX 5060 Ti 16GB 约 400 epoch 收敛

4. 显存需求：
   - 训练：>= 8GB
   - 推理：>= 2GB

## 许可证

本项目仅用于学习交流。OpenCPop 数据集使用需遵守其原始许可协议。

## 致谢

- FastSpeech2 - 声学模型
- HiFi-GAN - 神经声码器
- OpenCPop - 中文歌声数据集