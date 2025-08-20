import shap
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

# -------------------------
# 1. 设置图形风格（中文 + 大小）
# -------------------------
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)

# -------------------------
# 2. 加载糖尿病数据集
# -------------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 特征名中文映射（可以自行调整）
feature_name_map = {
    'age': '状态信息1',
    'sex': '状态信息2',
    'bmi': '状态信息3',
    'bp': '状态信息4',
    's1': '状态信息5',
    's2': '状态信息6',
    's3': '状态信息7',
    's4': '状态信息8',
    's5': '状态信息9',
    's6': '状态信息10'
}
X.rename(columns=feature_name_map, inplace=True)

# 标准化
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
idx = 0
sample = X_test.iloc[idx:idx+1]
sample_shap = shap_values[idx]
shap.plots.waterfall(sample_shap, show=False)
plt.tight_layout(pad=1.1)  # 默认是1.08，你可以尝试更大一些
plt.show()

shap.initjs()
shap.save_html("force_plot2.html", shap.plots.force(sample_shap))

# -------------------------
# 6. 可视化：多样本分析（执行层）
# -------------------------
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout(pad=2.0)  # 默认是1.08，你可以尝试更大一些

plt.show()

shap.plots.bar(shap_values, show=False)
plt.tight_layout()  # 默认是1.08，你可以尝试更大一些
plt.show()
