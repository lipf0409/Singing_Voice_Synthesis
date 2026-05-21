"""项目配置文件"""
import torch

class Config:
    # 数据路径
    DATA_DIR = "segments"
    WAV_DIR = f"{DATA_DIR}/wavs"
    TRAIN_FILE = f"{DATA_DIR}/train.txt"
    TEST_FILE = f"{DATA_DIR}/test.txt"
    TRANS_FILE = f"{DATA_DIR}/transcriptions.txt"
    PROCESSED_DIR = "data/processed"
    USE_PREPROCESSED = True

    # 音频参数
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 80
    MEL_FMIN = 0
    MEL_FMAX = 8000
    MAX_WAV_VALUE = 32768.0

    # 模型参数（增强版）
    MAX_SEQ_LEN = 1000
    PHONEME_EMB_DIM = 384  # 增大嵌入维度
    ENCODER_HIDDEN = 384
    ENCODER_N_LAYER = 6    # 增加层数
    ENCODER_HEAD = 4       # 增加注意力头
    DECODER_HIDDEN = 384
    DECODER_N_LAYER = 6
    DECODER_HEAD = 4
    CONV_FILTER_SIZE = 1536  # 增大FFN
    CONV_KERNEL_SIZE = 9
    DROPOUT = 0.1

    # Variance Predictor
    VARIANCE_PREDICTOR_FILTER_SIZE = 384
    VARIANCE_PREDICTOR_KERNEL_SIZE = 3
    VARIANCE_PREDICTOR_DROPOUT = 0.3

    # 训练参数 (RTX 5060 Ti 16GB)
    BATCH_SIZE = 64  # 模型变大，适当减小batch
    LEARNING_RATE = 5e-5  # 方案3：降低学习率
    WEIGHT_DECAY = 1e-6
    EPOCHS = 400  # 增加训练轮数
    WARMUP_STEPS = 2000
    GRAD_CLIP_THRESH = 1.0
    GRAD_ACCUM_STEPS = 1

    # 余弦退火重启参数（方案1）
    COSINE_T0 = 50       # 第一次重启周期（epoch数）
    COSINE_T_MULT = 2    # 每次重启后周期翻倍
    COSINE_ETA_MIN = 1e-6  # 最小学习率

    # Dropout 调整
    DROPOUT = 0.1
    VARIANCE_PREDICTOR_DROPOUT = 0.3

    # 混合精度训练
    USE_AMP = True

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查点
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"

    # HiFi-GAN 声码器路径
    HIFIGAN_CONFIG = "checkpoints/hifigan/config.json"
    HIFIGAN_CHECKPOINT = "checkpoints/hifigan/generator_v1"

    # 音素表（从数据动态生成，确保覆盖完整）
    PAD = "_"
    EOS = "~"
    PUNCTUATIONS = [" ", ",", ".", "!", "?", ":", ";", "(", ")", "\"", "'"]

    # OpenCPop 使用的中文歌声合成音素集
    # 声母
    INITIALS = [
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"
    ]
    # 韵母
    FINALS = [
        "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu",
        "ie", "ve", "ue", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong",
        "ian", "iao", "iang", "iong", "ua", "uo", "uai", "uan", "uang", "uei", "uen", "ueng",
        "ia", "io", "er"
    ]
    # 特殊标记（停顿）
    SPECIAL_TOKENS = ["sp", "SP", "AP"]

    @classmethod
    def get_phoneme_set(cls):
        phonemes = [cls.PAD, cls.EOS]  # 填充和结束
        phonemes.extend(cls.SPECIAL_TOKENS)  # sp, SP, AP 停顿标记
        phonemes.extend(cls.INITIALS)
        phonemes.extend(cls.FINALS)
        for p in cls.PUNCTUATIONS:
            phonemes.append(p)
        seen = set()
        unique_phonemes = []
        for p in phonemes:
            if p not in seen:
                seen.add(p)
                unique_phonemes.append(p)
        return unique_phonemes

config = Config()
PHONEME_TO_ID = {p: i for i, p in enumerate(Config.get_phoneme_set())}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}
N_PHONEMES = len(PHONEME_TO_ID)