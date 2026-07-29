import os
import numpy as np

from project_paths import (
    AUDIO_OUTPUT_DIR,
    FUSED_VECTORS_DIR,
    IMAGE_OUTPUT_DIR,
    SENSOR_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
)

modal_dirs = {
    "video": VIDEO_OUTPUT_DIR,
    "image": IMAGE_OUTPUT_DIR,
    "audio": AUDIO_OUTPUT_DIR,
    "sensor": SENSOR_OUTPUT_DIR,
}

# 所有模态的特征维度（如结构改变可调整）
modal_dims = {
    "video": 512,
    "image": 512,
    "audio": 32,
    "sensor": 64
}

# 获取可用标签（以 video 为基准）
labels = []
for folder in os.listdir(modal_dirs["video"]):
    if folder.startswith("Video_process_"):
        labels.append(folder.replace("Video_process_", ""))

print(f"🧩 可用焊接类型标签：{labels}")

# 创建输出目录
output_dir = FUSED_VECTORS_DIR
os.makedirs(output_dir, exist_ok=True)

# 对每个标签进行拼接
for label in labels:
    try:
        vecs = []
        for modal, modal_path in modal_dirs.items():
            npy_path = os.path.join(modal_path, f"{modal.capitalize()}_process_{label}", f"{label}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"❌ 缺失: {modal} 模态 {npy_path}")
            vecs.append(np.load(npy_path))

        # 检查样本数一致性
        shapes = [v.shape[0] for v in vecs]
        min_len = min(shapes)
        if min_len == 0:
            print(f"⚠️ 标签 {label} 的某模态无数据，跳过")
            continue
        # 对齐最短长度
        vecs = [v[:min_len] for v in vecs]

        fused = np.concatenate(vecs, axis=1)  # 拼接成每行 1120维
        np.save(os.path.join(output_dir, f"{label}.npy"), fused)
        print(f"✅ {label}: 拼接成功 → {fused.shape[0]} 条样本，每条 {fused.shape[1]} 维")
    except Exception as e:
        print(f"❌ {label} 拼接失败: {e}")
