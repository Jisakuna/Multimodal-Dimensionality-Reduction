import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from project_paths import DATASET_DIR, SENSOR_OUTPUT_DIR

# 设置根目录
dataset_root = DATASET_DIR
output_base = SENSOR_OUTPUT_DIR
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

# 1D-CNN 自编码器模型定义
class Sensor1DCNNEncoder(nn.Module):
    def __init__(self, input_channels=6):
        super(Sensor1DCNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.squeeze(-1)

model = Sensor1DCNNEncoder()
model.eval()

# 查找所有 CSV 文件
csv_files = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print(f"📈 共发现 {len(csv_files)} 个 CSV 文件")

# 选择的传感器字段
sensor_cols = ['Pressure', 'CO2 Weld Flow', 'Feed',
               'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
fixed_length = 500  # 固定序列长度

label_features = defaultdict(list)

for path in tqdm(csv_files, desc="提取传感器特征"):
    label = get_standard_label(path)
    if label is None:
        print(f"⚠️ 无标签匹配，跳过: {path}")
        continue

    try:
        df = pd.read_csv(path)
        if not all(col in df.columns for col in sensor_cols):
            print(f"⚠️ 缺少字段，跳过: {path}")
            continue

        data = df[sensor_cols].to_numpy(dtype=np.float32).T  # shape: [6, T]
        T = data.shape[1]

        if T < fixed_length:
            # 右侧补0
            pad = np.zeros((6, fixed_length - T), dtype=np.float32)
            data_padded = np.concatenate([data, pad], axis=1)
        else:
            data_padded = data[:, :fixed_length]

        tensor_input = torch.from_numpy(data_padded).unsqueeze(0)  # [1, 6, fixed_length]
        with torch.no_grad():
            encoded = model(tensor_input).squeeze(0).numpy()  # [64]
        label_features[label].append(encoded)

    except Exception as e:
        print(f"❌ 处理错误: {path} -- {e}")

# 保存为 .npy
for label, vectors in label_features.items():
    output_dir = os.path.join(output_base, f"Sensor_process_{label}")
    os.makedirs(output_dir, exist_ok=True)
    arr = np.stack(vectors)
    save_path = os.path.join(output_dir, f"{label}.npy")
    np.save(save_path, arr)
    print(f"✅ {label}: 保存 {len(vectors)} 条传感器特征 → {save_path}")
