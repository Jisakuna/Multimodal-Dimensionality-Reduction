import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from project_paths import FUSED_VECTORS_DIR, UMAP_OUTPUT_DIR

# 输入路径
fused_path = FUSED_VECTORS_DIR
output_path = UMAP_OUTPUT_DIR
os.makedirs(output_path, exist_ok=True)

# 读取多类拼接向量
all_data, all_labels = [], []
for file in os.listdir(fused_path):
    if file.endswith(".npy"):
        label = file.replace(".npy", "")
        data = np.load(os.path.join(fused_path, file))
        all_data.append(data)
        all_labels.extend([label] * data.shape[0])

X = np.concatenate(all_data, axis=0)
labels = np.array(all_labels)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维至 32维
umap_model = umap.UMAP(n_components=32, random_state=42)
X_umap_32 = umap_model.fit_transform(X_scaled)

# 保存 32维向量
np.save(os.path.join(output_path, "fused_umap_32d.npy"), X_umap_32)
np.save(os.path.join(output_path, "labels.npy"), labels)
print(f"✅ 降维完成: shape={X_umap_32.shape} → 保存至 {output_path}")

# 可视化（UMAP 再降到 2维）
umap_vis = umap.UMAP(n_components=2, random_state=42)
X_vis = umap_vis.fit_transform(X_scaled)

# 绘图
plt.figure(figsize=(10, 8))
unique_labels = sorted(set(labels))
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for i, label in enumerate(unique_labels):
    idx = labels == label
    plt.scatter(X_vis[idx, 0], X_vis[idx, 1], label=label, s=20, alpha=0.7)

plt.legend()
plt.title("UMAP 2D Projection of Fused Welding Features")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "umap_2d_visualization.png"))
plt.show()
