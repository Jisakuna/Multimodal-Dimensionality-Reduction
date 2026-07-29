import os
import librosa
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from project_paths import AUDIO_OUTPUT_DIR, DATASET_DIR

# 数据集路径
dataset_root = DATASET_DIR
output_base = AUDIO_OUTPUT_DIR
os.makedirs(output_base, exist_ok=True)

# 标签关键词映射
keyword_map = {
    "burnthrough": "burn_through",
    "burn_through": "burn_through",
    "crater": "crater_cracks",
    "excessive_convexity": "excessive_convexity",
    "convexity": "excessive_convexity",
    "excessive_penetration": "excessive_penetration",
    "penetration": "excessive_penetration",
    "good": "good",
    "lack_of_fusion": "lack_of_fusion",
    "fusion": "lack_of_fusion",
    "overlap": "overlap",
    "porosity": "porosity",
    "spatter": "spatter",
    "undercut": "undercut",
    "warping": "warping"
}

def get_standard_label(path):
    path_lower = path.lower()
    for keyword, std_label in keyword_map.items():
        if keyword in path_lower:
            return std_label
    return None

# 搜索所有 .flac 文件
audio_files = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith(".flac"):
            audio_files.append(os.path.join(root, file))

print(f"🔊 共找到 {len(audio_files)} 个音频文件")

# 存储特征
label_features = defaultdict(list)

for path in tqdm(audio_files, desc="提取音频特征"):
    label = get_standard_label(path)
    if label is None:
        print(f"⚠️ 未识别标签，跳过: {path}")
        continue

    try:
        # 加载音频（16kHz 单声道）
        y, sr = librosa.load(path, sr=16000)

        # 提取 MFCC 特征（20维）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)

        # 对所有帧取平均 → 得到 20维向量
        mfcc_mean = np.mean(mfcc, axis=1)

        label_features[label].append(mfcc_mean)

    except Exception as e:
        print(f"❌ 错误处理 {path}: {e}")

# 保存为 .npy
for label, vectors in label_features.items():
    output_dir = os.path.join(output_base, f"Audio_process_{label}")
    os.makedirs(output_dir, exist_ok=True)
    arr = np.stack(vectors)
    save_path = os.path.join(output_dir, f"{label}.npy")
    np.save(save_path, arr)
    print(f"✅ {label}: 保存 {len(vectors)} 条音频特征 → {save_path}")
