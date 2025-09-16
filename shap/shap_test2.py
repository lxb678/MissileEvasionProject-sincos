import shap
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib

# -------------------------
# 1. 设置全局图形样式（支持中文）
# -------------------------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体，或使用微软雅黑/Microsoft YaHei
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['figure.figsize'] = (10, 6)

# -------------------------
# 2. 载入数据并预处理
# -------------------------
data = fetch_openml("adult", version=2, as_frame=True)
X = data.data
y = data.target

# 中文特征名映射示例（需根据你的模型实际特征做调整）
feature_name_map = {
    "age": "年龄",
    "education-num": "受教育年限",
    "hours-per-week": "每周工作时长",
    "capital-gain": "资本收益",
    "capital-loss": "资本损失",
    "fnlwgt": "人口权重",
    # 可继续添加...
}

# 替换特征名为中文
X.rename(columns=feature_name_map, inplace=True)

# 编码分类变量
for col in X.select_dtypes(include="category").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. 训练模型
# -------------------------
model = xgboost.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# -------------------------
# 4. SHAP 分析器
# -------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# -------------------------
# 5. 可视化 - 战术层示意（单样本）
# -------------------------
idx = 0
sample = X_test.iloc[idx:idx+1]
sample_shap = shap_values[idx]

# 瀑布图
plt.title("战术层智能决策模型 - 单样本特征影响（瀑布图）", fontsize=16)
shap.plots.waterfall(sample_shap, show=False)
plt.tight_layout()
plt.savefig("tactical_waterfall.png", dpi=300)
plt.close()

# 力图（保存为 HTML）
shap.initjs()
force_html = shap.plots.force(sample_shap)
shap.save_html("tactical_force_plot.html", force_html)

# -------------------------
# 6. 可视化 - 执行层示意（多样本）
# -------------------------

# 蜂窝图
plt.title("执行层策略模型 - 特征重要性密度分布（蜂窝图）", fontsize=16)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("execution_beeswarm.png", dpi=300)
plt.close()

# 平均 SHAP 图
plt.title("执行层策略模型 - 平均特征影响（条形图）", fontsize=16)
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("execution_bar.png", dpi=300)
plt.close()
