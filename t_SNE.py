import os
import numpy as np
from sklearn.manifold import TSNE

from project_paths import FUSED_VECTORS_DIR, TSNE_OUTPUT_DIR

# 原始特征向量存放目录（注意 Windows 路径分隔符）
input_dir = FUSED_VECTORS_DIR
# 新建输出目录（与原目录平级）
output_dir = TSNE_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# 遍历每个 .npy 文件，分别读取并降维
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_dir, filename)
        # 读取原始 1101 维多模态特征向量
        data = np.load(file_path)  # 假设 data.shape
        
        # 使用 t-SNE 降维至 32 维
        tsne = TSNE(n_components=32, method='exact', random_state=42)
        data_reduced = tsne.fit_transform(data)  # 结果形状为 (样本数, 32)
        
        # 保存降维后的特征向量到新目录
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, data_reduced)
        
        # 输出该文件的降维后形状和保存路径
        print(f"{filename} -> reduced to shape {data_reduced.shape}, saved at {save_path}")
