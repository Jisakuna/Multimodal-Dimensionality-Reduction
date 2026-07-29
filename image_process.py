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

from project_paths import DATASET_DIR, IMAGE_OUTPUT_DIR

# 数据集根目录
dataset_root = DATASET_DIR
output_base = IMAGE_OUTPUT_DIR
os.makedirs(output_base, exist_ok=True)

# 标签映射表
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

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载 ResNet18 模型
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Identity()
resnet.eval()

# 收集所有含 images 文件夹的路径
image_folders = []
for root, dirs, files in os.walk(dataset_root):
    for d in dirs:
        if d.lower() == "images":
            full_path = os.path.join(root, d)
            if os.listdir(full_path):  # 有图像才处理
                image_folders.append(full_path)

print(f"📂 共发现 {len(image_folders)} 个图像目录")

# 存储每类特征
label_features = defaultdict(list)

# 遍历每个图像目录
for folder in tqdm(image_folders, desc="处理图像组"):
    label = get_standard_label(folder)
    if label is None:
        print(f"⚠️ 未识别标签，跳过: {folder}")
        continue

    image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(image_paths) == 0:
        continue

    try:
        feats = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(img_tensor)
            feats.append(feat.squeeze(0))

        if feats:
            avg_feat = torch.stack(feats).mean(dim=0)  # 聚合为512维向量
            label_features[label].append(avg_feat.numpy())

    except Exception as e:
        print(f"❌ 处理出错: {folder} -- {e}")

# 保存到对应目录
for label, vectors in label_features.items():
    output_dir = os.path.join(output_base, f"Image_process_{label}")
    os.makedirs(output_dir, exist_ok=True)
    arr = np.stack(vectors)
    save_path = os.path.join(output_dir, f"{label}.npy")
    np.save(save_path, arr)
    print(f"✅ {label}: 保存 {len(vectors)} 条图像特征 → {save_path}")
