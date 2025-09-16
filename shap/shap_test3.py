import shap
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

# -------------------------
# 1. 设置图形风格（中文 + 大小）
# -------------------------
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus'] = False         # 正确显示负号 "-"
plt.rcParams['figure.figsize'] = (15, 10)

# -------------------------
# 2. 加载加州住房数据集
# -------------------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 特征名中文映射（可根据任务调整）
feature_name_map = {
    # 'MedInc': '家庭收入（中位数）',
    # 'HouseAge': '房龄',
    # 'AveRooms': '平均房间数',
    # 'AveBedrms': '平均卧室数',
    # 'Population': '人口数量',
    # 'AveOccup': '平均居住人数',
    # 'Latitude': '纬度',
    # 'Longitude': '经度'
    'MedInc': '状态信息1',
    'HouseAge': '状态信息2',
    'AveRooms': '状态信息3',
    'AveBedrms': '状态信息4',
    'Population': '状态信息5',
    'AveOccup': '状态信息6',
    'Latitude': '状态信息7',
    'Longitude': '状态信息8'
}
X.rename(columns=feature_name_map, inplace=True)

# 标准化（可选）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -------------------------
# 3. 训练模型
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = xgboost.XGBRegressor(n_estimators=100, max_depth=4)
model.fit(X_train, y_train)

# -------------------------
# 4. 初始化 SHAP 解释器
# -------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# -------------------------
# 5. 可视化：单样本分析（战术层）
# -------------------------
idx = 2
sample = X_test.iloc[idx:idx+1]
sample_shap = shap_values[idx]



# 瀑布图
# plt.title("战术层模型输出解释（瀑布图）", fontsize=16)
shap.plots.waterfall(sample_shap, show=False)
plt.tight_layout()  # 默认是1.08，你可以尝试更大一些
plt.show()

# # 力图（HTML 交互形式）
shap.initjs()
shap.save_html("force_plot2.html", shap.plots.force(sample_shap))

# -------------------------
# 6. 可视化：多样本分析（执行层）
# -------------------------
# 蜂窝图
# plt.title("执行层模型特征贡献分布（蜂窝图）", fontsize=16)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.show()

# 平均 SHAP 值柱状图
# plt.title("执行层模型特征平均影响力（条形图）", fontsize=16)
shap.plots.bar(shap_values, show=True)
