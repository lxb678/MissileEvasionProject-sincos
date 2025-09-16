import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# 四元数旋转函数（罗德里格斯公式）
def roll(alpha_hat, alpha, vector):
    """列向量旋转"""
    if np.linalg.norm(alpha_hat) < 1e-10:
        return vector

    alpha_hat = alpha_hat / np.linalg.norm(alpha_hat)
    dot_product = np.dot(alpha_hat, vector)
    cross_product = np.cross(alpha_hat, vector)

    new_vector = (1 - np.cos(alpha)) * dot_product * alpha_hat + \
                 np.cos(alpha) * vector + \
                 np.sin(alpha) * cross_product
    return new_vector


# 坐标转换函数（北天东到东北天）
def show_loc(P):
    """将北天东坐标系转换为东北天坐标系"""
    P_show = np.zeros_like(P)
    P_show[0, :] = P[2, :]  # E -> x
    P_show[1, :] = P[0, :]  # N -> y
    P_show[2, :] = P[1, :]  # U -> z
    return P_show


# 初始化参数
dt = 0.02
t_max = 10

# 初始位置和速度
p_me = np.array([0.0, 0.0, 0.0])  # 北天东坐标系

psi = 0 * np.pi / 180
theta = 0 * np.pi / 180
phi = 0 * 90 * np.pi / 180

v_me = 100.0  # 速度大小
m = 2.0  # 质量
g_earth = 9.8  # 重力加速度
G = np.array([0.0, -1.0, 0.0]) * m * g_earth  # 重力矢量

# 计算初始旋转矩阵（北天东到前上右）
R_iyzx = np.array([
    [np.cos(-psi), 0, np.sin(-psi)],
    [0, 1, 0],
    [-np.sin(-psi), 0, np.cos(-psi)]
]) @ np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
]) @ np.array([
    [1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi), np.cos(phi)]
])

# 初始化速度轴（前上右坐标系）
Axv = R_iyzx @ np.array([1.0, 0.0, 0.0])
Ayv = R_iyzx @ np.array([0.0, 1.0, 0.0])
Azv = R_iyzx @ np.array([0.0, 0.0, 1.0])

v_me_vec = v_me * Axv  # 速度矢量

# 初始化角速度
xv_omega = 0.0
yv_omega = 0.0
zv_omega = 0.0

# 数据记录列表
P_me_list = []
V_me_list = []
Axv_list = []
Ayv_list = []
Azv_list = []
T_list = []
N_list = []
B_list = []
Omega_list = []
Theta_list = []
Psi_list = []
Phi_list = []

# 主循环
for t in np.arange(0, t_max + dt, dt):
    # 引导规则（简化）
    omega_B_required = np.array([0.0, 0.0, 0.0])
    v_required = 10.0

    # 控制规则（简化）
    F_thrust = 0.0

    # 简单的滚转控制
    if phi > np.pi / 2:
        omega_xv_me = -1 * np.pi / 180 / dt
    elif phi < np.pi / 2:
        omega_xv_me = 1 * np.pi / 180 / dt
    else:
        omega_xv_me = 0.0

    omega_xv_me = 0.0  # 禁用滚转控制

    # 升力计算
    if v_me > 0:
        F_wing = 2 * g_earth * m
    else:
        F_wing = 0.0

    # 运动学与动力学
    # 切向加速度
    v_me_vec = Axv * v_me
    T = Axv.copy()
    G_paral = np.dot(G, T)
    G_paral_vec = G_paral * T
    F_thrust_vec = F_thrust * Axv

    # 阻力（简化）
    F_x = 0.0
    F_x_vec = F_x * Axv

    aT_vec = (F_thrust_vec + G_paral_vec + F_x_vec) / m
    aT = (F_thrust + G_paral - F_x) / m

    # 法向加速度
    F_wing_vec = F_wing * Ayv
    G_perp_vec = G - G_paral_vec

    # 检查垂直分量是否正确
    if np.abs(np.dot(G_perp_vec, G_paral_vec)) > 0.001:
        print('error: G_perp and G_paral are not perpendicular')
        break

    aN_vec = (F_wing_vec + G_perp_vec) / m
    aN = np.linalg.norm(aN_vec)

    # 计算角速度
    omega_B = aN / v_me if v_me > 1e-10 else 0.0

    # 计算法向和副法向向量
    if aN < 1e-1:
        N = Ayv.copy()
        B = Azv.copy()
    else:
        N = aN_vec / aN
        B = np.cross(T, N)
        B = B / np.linalg.norm(B)

    # 根据速度大小选择不同的更新策略
    if v_me > 10:
        omega_B_vec = omega_B * B
        omega_T_vec = omega_xv_me * T
        omega_N_vec = np.zeros(3)  # 通常为零
        omega_vec = omega_T_vec + omega_N_vec + omega_B_vec
        omega = np.linalg.norm(omega_vec)

        # 使用四元数旋转更新速度轴
        if omega > 1e-10:
            omega_hat = omega_vec / omega
            Axv = roll(omega_hat, omega * dt, Axv)
            Ayv = roll(omega_hat, omega * dt, Ayv)
            Azv = roll(omega_hat, omega * dt, Azv)

        # 更新速度大小和矢量
        v_me_new = v_me + aT * dt
        v_me_new_vec = v_me_new * Axv
    else:
        # 低速时的特殊处理
        v_me_new_vec = v_me_vec + (aT_vec + aN_vec) * dt
        v_me_new = np.linalg.norm(v_me_new_vec)
        Axv = G / (m * g_earth)  # 重力方向
        Ayv = np.cross(Azv, Axv)
        Ayv = Ayv / np.linalg.norm(Ayv)
        omega_vec = np.zeros(3)

    # 更新位置（梯形积分）
    p_me = p_me + (v_me_new_vec + v_me_vec) / 2 * dt
    v_me_vec = v_me_new_vec
    v_me = v_me_new

    # 记录数据
    P_me_list.append(p_me.copy())
    V_me_list.append(v_me_vec.copy())
    Axv_list.append(Axv.copy())
    Ayv_list.append(Ayv.copy())
    Azv_list.append(Azv.copy())
    T_list.append(Axv.copy())  # T轴与Axv相同
    N_list.append(N.copy())
    B_list.append(B.copy())
    Omega_list.append(omega_vec.copy())

    # 计算欧拉角
    v_me_h = np.linalg.norm(np.array([1, 0, 1]) * v_me_vec)
    theta = np.arctan2(v_me_vec[1], v_me_h)  # 俯仰角
    psi = np.arctan2(v_me_vec[2], v_me_vec[0])  # 偏航角

    # 计算滚转角
    y_refer = -G_perp_vec / (np.linalg.norm(G_perp_vec) + 1e-10)
    cosphi = np.dot(y_refer, Ayv)
    temp = np.cross(y_refer, Ayv)
    sinphi = np.dot(temp, Axv)
    phi = np.arctan2(sinphi, cosphi)

    Theta_list.append(theta)
    Psi_list.append(psi)
    Phi_list.append(phi)

# 转换为numpy数组以便绘图
P_me_list = np.array(P_me_list).T
V_me_list = np.array(V_me_list).T
Axv_list = np.array(Axv_list).T
Ayv_list = np.array(Ayv_list).T
Azv_list = np.array(Azv_list).T
T_list = np.array(T_list).T
N_list = np.array(N_list).T
B_list = np.array(B_list).T
Omega_list = np.array(Omega_list).T
Theta_list = np.array(Theta_list)
Psi_list = np.array(Psi_list)
Phi_list = np.array(Phi_list)

# 绘图
# 1. 轨迹图
P_show = show_loc(P_me_list)

fig = plt.figure(figsize=(15, 10))

# 轨迹
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(P_show[0, :], P_show[1, :], P_show[2, :])
ax1.scatter(P_show[0, 0], P_show[1, 0], P_show[2, 0], c='r', marker='o', s=50)
ax1.set_xlabel("E")
ax1.set_ylabel("N")
ax1.set_zlabel("U")
ax1.set_title("Trajectory")
ax1.grid(True)

# 欧拉角变化
ax2 = fig.add_subplot(2, 3, 2)
t_list = np.arange(0, t_max + dt, dt)
ax2.plot(t_list, Theta_list * 180 / np.pi, label='θ')
ax2.plot(t_list, Psi_list * 180 / np.pi, label='ψ')
ax2.plot(t_list, Phi_list * 180 / np.pi, label='φ')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Angle (deg)")
ax2.set_title("Euler Angles")
ax2.legend()
ax2.grid(True)

# TNB坐标系可视化（只显示最后一帧）
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
i = -1  # 最后一帧
T = T_list[:, i]
N = N_list[:, i]
B = B_list[:, i]

ax3.quiver(0, 0, 0, T[2], T[0], T[1], color='r', label='T', length=1.0)
ax3.quiver(0, 0, 0, N[2], N[0], N[1], color='g', label='N', length=1.0)
ax3.quiver(0, 0, 0, B[2], B[0], B[1], color='b', label='B', length=1.0)
ax3.quiver(-B[2], -B[0], -B[1], B[2], B[0], B[1], color='b', linestyle='--')
ax3.quiver(T[2], T[0], T[1], T[2], T[0], T[1], color='black')

ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])
ax3.set_zlim([-2, 2])
ax3.set_xlabel("E")
ax3.set_ylabel("N")
ax3.set_zlabel("U")
ax3.set_title("TNB Frame (Last Frame)")
ax3.legend()
ax3.grid(True)

# 速度坐标系可视化（只显示最后一帧）
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
Axv = Axv_list[:, i]
Ayv = Ayv_list[:, i]
Azv = Azv_list[:, i]

ax4.quiver(0, 0, 0, Axv[2], Axv[0], Axv[1], color='r', label='Axv', length=1.0)
ax4.quiver(0, 0, 0, Ayv[2], Ayv[0], Ayv[1], color='g', label='Ayv', length=1.0)
ax4.quiver(0, 0, 0, Azv[2], Azv[0], Azv[1], color='b', label='Azv', length=1.0)
ax4.quiver(-Azv[2], -Azv[0], -Azv[1], Azv[2], Azv[0], Azv[1], color='b', linestyle='--')
ax4.quiver(Axv[2], Axv[0], Axv[1], Axv[2], Axv[0], Axv[1], color='black')

ax4.set_xlim([-2, 2])
ax4.set_ylim([-2, 2])
ax4.set_zlim([-2, 2])
ax4.set_xlabel("E")
ax4.set_ylabel("N")
ax4.set_zlabel("U")
ax4.set_title("Velocity Frame (Last Frame)")
ax4.legend()
ax4.grid(True)

# 速度大小变化
ax5 = fig.add_subplot(2, 3, 5)
v_magnitude = np.linalg.norm(V_me_list, axis=0)
ax5.plot(t_list, v_magnitude)
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Speed (m/s)")
ax5.set_title("Speed vs Time")
ax5.grid(True)

# 角速度变化
ax6 = fig.add_subplot(2, 3, 6)
omega_magnitude = np.linalg.norm(Omega_list, axis=0)
ax6.plot(t_list, omega_magnitude)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Angular Velocity (rad/s)")
ax6.set_title("Angular Velocity vs Time")
ax6.grid(True)

plt.tight_layout()
plt.show()

# 如果需要动画效果，可以取消注释下面的代码
"""
# 创建TNB坐标系动画
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

for i in range(0, len(t_list), 5):  # 每5帧显示一次
    ax_anim.cla()

    T = T_list[:, i]
    N = N_list[:, i]
    B = B_list[:, i]

    ax_anim.quiver(0, 0, 0, T[2], T[0], T[1], color='r', label='T', length=1.0)
    ax_anim.quiver(0, 0, 0, N[2], N[0], N[1], color='g', label='N', length=1.0)
    ax_anim.quiver(0, 0, 0, B[2], B[0], B[1], color='b', label='B', length=1.0)
    ax_anim.quiver(-B[2], -B[0], -B[1], B[2], B[0], B[1], color='b', linestyle='--')
    ax_anim.quiver(T[2], T[0], T[1], T[2], T[0], T[1], color='black')

    ax_anim.set_xlim([-2, 2])
    ax_anim.set_ylim([-2, 2])
    ax_anim.set_zlim([-2, 2])
    ax_anim.set_xlabel("E")
    ax_anim.set_ylabel("N")
    ax_anim.set_zlabel("U")
    ax_anim.set_title(f"TNB Frame (t = {t_list[i]:.2f}s)")
    ax_anim.legend()
    ax_anim.grid(True)

    plt.pause(0.01)

plt.show()
"""