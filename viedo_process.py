import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from project_paths import DATASET_DIR, VIDEO_OUTPUT_DIR

# 1. 数据路径与保存路径
dataset_root = DATASET_DIR
output_base = VIDEO_OUTPUT_DIR
os.makedirs(output_base, exist_ok=True)

# 2. 定义标准标签（12类）和映射关键词
standard_labels = [
    "burn_through", "crater_cracks", "excessive_convexity", "excessive_penetration",
    "good", "lack_of_fusion", "overlap", "porosity",
    "spatter", "undercut", "warping"
]

# 映射规则：路径中出现下列关键字，归入对应标签（全部小写处理）
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

# 3. ResNet18 设置
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Identity()
resnet.eval()

# 4. 搜索所有视频
video_files = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith(".avi"):
            video_files.append(os.path.join(root, file))

print(f"🔍 共找到 {len(video_files)} 个视频")

# 5. 根据路径映射为标准标签
def get_standard_label(path):
    path_lower = path.lower()
    for keyword, std_label in keyword_map.items():
        if keyword in path_lower:
            return std_label
    return None  # 未匹配到标准标签

# 6. 提取并保存特征
label_features = defaultdict(list)

for video_path in tqdm(video_files, desc="处理视频"):
    label = get_standard_label(video_path)
    if label is None:
        print(f" 跳过未匹配的文件: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames, frame_id = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % fps == 0:
            frames.append(frame)
        frame_id += 1
    cap.release()

    if not frames:
        continue

    try:
        feats = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(img_tensor)
            feats.append(feat.squeeze(0))

        avg_feat = torch.stack(feats).mean(dim=0)  # [512]
        label_features[label].append(avg_feat.numpy())

    except Exception as e:
        print(f" 错误处理: {video_path} -- {e}")

# 7. 将特征保存到对应标签子目录
for label, vectors in label_features.items():
    output_dir = os.path.join(output_base, f"Video_process_{label}")
    os.makedirs(output_dir, exist_ok=True)

    arr = np.stack(vectors)
    np.save(os.path.join(output_dir, f"{label}.npy"), arr)
    print(f" 已保存 {len(vectors)} 条特征至 {output_dir}/{label}.npy")
