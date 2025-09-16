import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# 1. 数据准备（精简样本加速计算）
data = fetch_california_housing()
X, y = data.data[:1000], data.target[:1000]  # 取1000个样本
feature_names = data.feature_names  # 特征名称
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. 训练轻量模型
model = RandomForestRegressor(
    n_estimators=20,  # 少量树加速训练
    max_depth=5,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)


# 3. 计算SHAP值并包装为Explanation对象（关键：传递特征名称）
explainer = shap.TreeExplainer(model)
sample_size = 80  # 用于多样本图的样本量
shap_values_array = explainer.shap_values(
    X_test[:sample_size],
    check_additivity=False  # 关闭可加性检查避免报错
)

# 包装为Explanation对象（统一传递特征名称、基线值等）
shap_exp = shap.Explanation(
    values=shap_values_array,
    base_values=explainer.expected_value,  # 模型基线值（平均预测）
    data=X_test[:sample_size],  # 对应的样本数据
    feature_names=feature_names  # 特征名称（关键：在这里传递）
)


# ---------------------- 多样本可视化 ----------------------
print("生成多样本可视化图...")

# 1. 蜂窝图（beeswarm plot）- 直接使用Explanation对象
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_exp)  # 无需单独传feature_names
plt.title('多样本特征影响蜂窝图')
plt.tight_layout()
plt.show()

# 2. 平均SHAP值条形图（全局特征重要性）
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_exp)  # 无需单独传feature_names
plt.title('特征平均SHAP值条形图（全局重要性）')
plt.tight_layout()
plt.show()


# ---------------------- 单样本可视化 ----------------------
# 选择样本（0到sample_size-1）
try:
    sample_idx = int(input(f"请输入样本编号（0-{sample_size-1}）："))
    sample_idx = np.clip(sample_idx, 0, sample_size-1)  # 限制范围
except:
    sample_idx = 0
    print(f"输入无效，默认展示样本 {sample_idx}")

print(f"生成样本 {sample_idx} 的可视化图...")

# 提取单个样本的Explanation对象
single_exp = shap_exp[sample_idx]  # 从整体中取单个样本

# 1. 单样本瀑布图（waterfall plot）
plt.figure(figsize=(10, 6))
shap.plots.waterfall(single_exp)  # 直接使用单个样本的Explanation对象
plt.title(f'样本 {sample_idx} 的SHAP瀑布图')
plt.tight_layout()
plt.show()

# 2. 单样本力图（force plot）
shap.plots.force(
    single_exp,  # 使用单个样本的Explanation对象
    matplotlib=True,
    figsize=(12, 4)
)
plt.title(f'样本 {sample_idx} 的SHAP力图')
plt.tight_layout()
plt.show()

print("所有图生成完成！")