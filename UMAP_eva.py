import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import matplotlib

from project_paths import UMAP_OUTPUT_DIR

# 📁 设置输入目录
input_dir = UMAP_OUTPUT_DIR
X = np.load(os.path.join(input_dir, "fused_umap_32d.npy"))
y = np.load(os.path.join(input_dir, "labels.npy"))

# 🧪 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🌲 随机森林训练
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 🎯 准确率输出
train_acc = clf.score(X_train, y_train)
val_acc = clf.score(X_val, y_val)
print(f"✅ 训练准确率: {train_acc:.4f}")
print(f"✅ 验证准确率: {val_acc:.4f}")

# 📋 分类报告
y_pred = clf.predict(X_val)
report_dict = classification_report(y_val, y_pred, output_dict=True)
print("\n📋 分类报告:\n", classification_report(y_val, y_pred))

# ✅ 设置中文字体以显示标题
matplotlib.rcParams['font.family'] = 'SimHei'  # 可替换为 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 📊 混淆矩阵图
cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45, values_format='d')
plt.title("混淆矩阵 - 验证集", fontsize=14)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(input_dir, "confusion_matrix.png"))
plt.show()

# 🔥 分类报告热力图
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.iloc[:-3, :]  # 去掉 accuracy, macro avg, weighted avg

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:, :3], annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("分类报告（精度 / 召回率 / F1 分数）", fontsize=14)
plt.xlabel("指标")
plt.ylabel("缺陷类别")
plt.tight_layout()
plt.savefig(os.path.join(input_dir, "classification_report.png"))
plt.show()
