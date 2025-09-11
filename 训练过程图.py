import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter # 引入格式化工具
import os

# sns.set_style("darkgrid")
sns.set_style("whitegrid")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# === A. 设置字体为 Times New Roman (关键步骤！) ===
# 很多学术期刊都要求使用 Times New Roman 字体
plt.rcParams['font.family'] = 'serif'  # 设置字体家族为衬线字体
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置衬线字体为 Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 可选：设置数学公式字体，stix与Times New Roman很接近
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# === 1. 读取数据 ===
# csv_file_path = "C:\\Users\\LXB\\Desktop\\桌面资料\\ICACR\\训练数据\\Trans_seed1_time_08_21_17_46_56_loadable_False.csv"
# csv_file_path = "C:\\Users\\LXB\\Desktop\\桌面资料\\ICACR\\训练数据\\Trans_seed1_time_08_21_17_46_56_loadable_False (1).csv"
# csv_file_path = "C:\\Users\LXB\Desktop\桌面资料\ICACR\训练数据\Trans_seed1_time_08_25_11_51_39_loadable_False (1).csv"
csv_file_path = "C:\\Users\LXB\Desktop\桌面资料\ICACR\训练数据\Trans_seed1_time_08_25_11_51_39_loadable_False.csv"
df = pd.read_csv(csv_file_path)

# === 2. 配置 ===
episode_col = "Step"  # 横轴列名
reward_cols = [
    "Value",
]
colors = sns.color_palette("tab10", len(reward_cols))  # 自动配色
smooth_type = "EMA"   # "EMA" 或 "SMA"
smooth_strength = 0.1  # 0~1，越大越平滑

# === 3. 将 0~1 的平滑强度映射到合适窗口/跨度 ===
min_window, max_window = 2, 200  # 窗口范围
smooth_window = int(min_window + smooth_strength * (max_window - min_window))

plt.figure(figsize=(10, 6))
ax = plt.gca() # 获取当前坐标轴，方便后续操作

for i, col in enumerate(reward_cols):
    x = df[episode_col]
    y = df[col]

    # 原始曲线（透明）
    # sns.lineplot(x=x, y=y, color=colors[i], alpha=0.1,label=f"{col} (Original)")
    sns.lineplot(x=x, y=y, color=colors[i], alpha=0.1, label=f"Survival Rate (Original)")
    # 平滑曲线
    if smooth_type.upper() == "EMA":
        smoothed_y = y.ewm(span=smooth_window, adjust=False).mean()
    elif smooth_type.upper() == "SMA":
        smoothed_y = y.rolling(window=smooth_window, min_periods=1).mean()
    else:
        smoothed_y = y

    sns.lineplot(x=x, y=smoothed_y, color=colors[i], label=f"Survival Rate (Smooth)")
    # sns.regplot(x=x, y=smoothed_y, ci=95)  # ci是置信区间，此处设置为95%

    # # ... 在你的绘图循环中 ...
    #
    # # --- 采用 滚动平均+标准差 的平滑方式 ---
    # window_size = 200  # 这是一个关键参数，需要根据你数据的噪声程度来调整
    #
    # # 1. 计算滚动均值和标准差
    # rolling_mean = y.rolling(window=window_size, min_periods=1).mean()
    # rolling_std = y.rolling(window=window_size, min_periods=1).std()
    #
    # # 2. 绘制原始曲线 (可以省略，因为阴影已经能反映波动)
    # # sns.lineplot(x=x, y=y, color=colors[i], alpha=0.1)
    #
    # # 3. 绘制均值曲线 (作为平滑结果)
    # sns.lineplot(x=x, y=rolling_mean, color=colors[i], linewidth=2, label="Reward (Smoothed Mean)")
    #
    # # 4. 填充标准差阴影区域
    # ax.fill_between(
    #     x,
    #     rolling_mean - rolling_std,
    #     rolling_mean + rolling_std,
    #     color=colors[i],
    #     alpha=0.2,  # 使用更淡的透明度
    #     label="Standard Deviation"
    # )
    #
    # # 在图例中，你可能需要手动调整一下，只显示均值曲线和置信区间
    # handles, labels = ax.get_legend_handles_labels()
    # # 你可以根据需要筛选 handles 和 labels
    # ax.legend(handles=handles[:2], labels=labels[:2])  # 举例：只显示前两个图例项

# === 设置图表属性 (重点在这里！) ===

# 1. 调整标题字号
# plt.title("Training Reward over Steps", fontsize=18, fontweight='bold') # 加粗更醒目

# 2. 调整坐标轴标签字号
plt.ylabel("Survival Rate", fontsize=14)
# 3. 调整坐标轴刻度字号
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 4. 调整图例字号
plt.legend(
    loc="upper left",
    frameon=True,
    edgecolor='black',
    fontsize=12  # 控制图例文字大小
)
# === 5. 格式化X轴 (重点！) ===
# 定义一个将数值转换为百万单位的函数
def millions_formatter(x, pos):
    return f'{x / 1e6:.1f}' # f-string格式化：除以一百万，保留一位小数

# 应用这个格式化函数到X轴的主刻度上
ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

# 更新X轴标签，注明单位是百万
plt.xlabel("Step (×10⁶)", fontsize=14)
# plt.xlim(0, 12000)      # 设置X轴范围
# plt.ylim(-300, 300)    # 设置Y轴范围

# c. 设置刻度位置和标签
# tick_locations = np.arange(0, 12001, 2000) # 每隔 2000 设置一个主刻度
# plt.xticks(ticks=tick_locations)

# d. (可选) 禁用科学计数法 (如果你的数值非常大)
# ax.ticklabel_format(style='plain', axis='x')

# plt.legend()
plt.legend(loc="upper right")   # 与plt.legend(loc=1)等价

# === 5. 设置图表边框、网格和图例 (重点在这里！) ===

# 方案一：加深所有边框
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0) # 1.0 或 1.2 是一个不错的选择

# 加深网格线
plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

# 加深图例边框
plt.legend(
    loc="upper left", # 根据你的图，左上角位置更好
    frameon=True,
    edgecolor='black'
)


# === 5. 保存图片 (最佳实践) ===

# 定义保存图片的文件夹和文件名
output_dir = "figures"  # 在当前脚本目录下创建一个名为 'figures' 的文件夹
output_filename = "Survival Rate.png"

# 检查文件夹是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用 os.path.join 来构建跨平台的、安全的文件路径
full_path = os.path.join(output_dir, output_filename)

# # 执行保存操作
# plt.savefig(
#     full_path,
#     dpi=300,              # 高分辨率
#     bbox_inches='tight'   # 裁剪白边
# )
#
# # 打印保存信息
# print(f"图片已保存至: {full_path}")

# (可选) 同时保存一份矢量图 (PDF) 用于论文
pdf_path = os.path.join(output_dir, "Survival Rate.svg")
plt.savefig(pdf_path, bbox_inches='tight')
print(f"SVG矢量图已保存至: {pdf_path}")

plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# def tensorboard_smooth(y, smooth_factor):
#     """
#     TensorBoard 风格平滑
#     smooth_factor: 0~1, TensorBoard UI 的 smooth 滑块值
#     """
#     if smooth_factor <= 0:  # 不平滑
#         return y
#
#     # TensorBoard 源码中的 α 计算方式
#     weight = np.exp(np.log(0.01) * (1 - smooth_factor))
#
#     smoothed = np.zeros_like(y, dtype=np.float64)
#     smoothed[0] = y[0]
#     for i in range(1, len(y)):
#         smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * y[i]
#     return smoothed
#
#
# sns.set_style("darkgrid")
#
# # === 1. 读取数据 ===
# csv_file_path = "wandb_export_2025-08-11T16_44_45.043+08_00.csv"
# df = pd.read_csv(csv_file_path)
#
# # === 2. 配置 ===
# episode_col = "train_Episode: "
# reward_cols = [
#     "ICM-PER-SAC - train_reward",
#     "PER-SAC - train_reward",
# ]
# colors = sns.color_palette("tab10", len(reward_cols))
#
# smooth_type = "TensorBoard"  # 可选: "EMA" / "SMA" / "TensorBoard"
# smooth_strength = 0.6        # 0~1, 对应不同平滑方式的强度
#
# # === 3. 映射窗口大小（仅 EMA / SMA 用）
# min_window, max_window = 2, 100
# smooth_window = int(min_window + smooth_strength * (max_window - min_window))
#
# plt.figure(figsize=(10, 6))
#
# for i, col in enumerate(reward_cols):
#     x = df[episode_col]
#     y = df[col].values
#
#     #原始曲线（透明）
#     sns.lineplot(x=x, y=y, color=colors[i], alpha=0.2)
#
#     if smooth_type.upper() == "EMA":
#         smoothed_y = pd.Series(y).ewm(span=smooth_window, adjust=False).mean()
#
#     elif smooth_type.upper() == "SMA":
#         smoothed_y = pd.Series(y).rolling(window=smooth_window, min_periods=1).mean()
#
#     elif smooth_type.lower() == "tensorboard":
#         smoothed_y = tensorboard_smooth(y, smooth_strength)
#
#     else:
#         smoothed_y = y  # 不平滑
#
#     sns.lineplot(x=x, y=smoothed_y, color=colors[i], label=f"{col} ({smooth_type})")
#
# plt.title("LunarLander-v2")
# plt.xlabel("steps")
# plt.ylabel("reward")
# plt.legend()
# plt.show()





