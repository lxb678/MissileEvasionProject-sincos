import numpy as np
import matplotlib.pyplot as plt
import math


def calculate_penalty_simulation(angle_deg, distance):
    """
    模拟 _compute_missile_head_on_penalty 函数的核心逻辑
    """
    # 1. 将角度转换为弧度，并计算 cos 值
    # 0度 = 拖尾 (cos=1), 90度 = 三九 (cos=0), 180度 = 迎头 (cos=-1)
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)

    # 2. 模拟函数的参数
    HEAD_ON_PENALTY_WIDTH_DEG = 60.0  # 惩罚宽度
    THRESHOLD_COS = -0.2  # 触发惩罚的阈值

    # 3. 阈值检查 (if cos_angle > -0.2: return 0.0)
    if cos_angle > THRESHOLD_COS:
        return 0.0

    # 4. 计算高斯惩罚
    angle_error_deg = abs(angle_deg - 180.0)
    penalty_factor = math.exp(-(angle_error_deg ** 2) / (2 * HEAD_ON_PENALTY_WIDTH_DEG ** 2))

    # 5. 距离权重逻辑
    dist_weight = 1.0
    if distance > 8000:
        dist_weight = 0.2
    elif distance > 5000:
        dist_weight = 0.5

    return -1.0 * penalty_factor * dist_weight


# --- 准备绘图数据 ---
# 生成从 0度 (背离) 到 180度 (迎头) 的角度数据
angles = np.linspace(0, 180, 500)

# 定义三种距离情况
distances = [
    2000,  # 近距 (权重 1.0)
    6000,  # 中距 (权重 0.5)
    9000  # 远距 (权重 0.2)
]

# 绘图设置
plt.figure(figsize=(10, 6), dpi=100)

# 循环绘制三条线
colors = ['red', 'orange', 'green']
labels = ['Close Range (<5km)', 'Medium Range (5-8km)', 'Long Range (>8km)']

for dist, color, label in zip(distances, colors, labels):
    penalties = [calculate_penalty_simulation(a, dist) for a in angles]
    plt.plot(angles, penalties, label=label, color=color, linewidth=2.5)

# --- 添加辅助线和说明 ---

# 1. 标出阈值线 (-0.2 对应的角度)
threshold_angle = np.rad2deg(np.arccos(-0.2))  # 约 101.5度
plt.axvline(x=threshold_angle, color='gray', linestyle='--', alpha=0.7)
plt.text(threshold_angle + 2, -0.1, f'Threshold ~{threshold_angle:.1f}°\n(cos=-0.2)', color='gray')

# 2. 标出迎头线 (180度)
plt.axvline(x=180, color='black', linestyle=':', alpha=0.5)
plt.text(170, -0.05, 'Head-On\n(180°)', color='black', ha='center')

# 3. 区域标注
plt.fill_between([0, threshold_angle], 0, -1.1, color='green', alpha=0.05)
plt.text(50, -0.5, 'Safe Zone\n(No Penalty)', fontsize=12, color='green', ha='center')

plt.fill_between([threshold_angle, 180], 0, -1.1, color='red', alpha=0.05)
plt.text(140, -0.5, 'Penalty Zone', fontsize=12, color='red', ha='center')

# 图表装饰
plt.title('Visualization of Head-on Penalty Function', fontsize=14)
plt.xlabel('Angle between Velocity and Missile LOS (Degrees)', fontsize=12)
plt.ylabel('Penalty Value (Negative Reward)', fontsize=12)
plt.ylim(-1.1, 0.1)
plt.xlim(0, 185)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()