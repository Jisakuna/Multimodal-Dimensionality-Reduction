import os
import numpy as np
from sklearn.decomposition import PCA

from project_paths import FUSED_VECTORS_DIR, PCA_OUTPUT_DIR

# 📁 输入与输出路径
input_dir = FUSED_VECTORS_DIR
output_dir = PCA_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# ⚙️ PCA 参数
n_components = 32

# 📦 遍历并处理每个文件
for filename in os.listdir(input_dir):
    if not filename.endswith(".npy"):
        continue

    file_path = os.path.join(input_dir, filename)
    data = np.load(file_path)

    # 降维处理
    pca = PCA(n_components=n_components, random_state=42)
    data_reduced = pca.fit_transform(data)

    # 保存降维结果
    save_path = os.path.join(output_dir, filename)
    np.save(save_path, data_reduced)

    print(f"{filename} -> reduced to shape {data_reduced.shape}, saved at {save_path}")
