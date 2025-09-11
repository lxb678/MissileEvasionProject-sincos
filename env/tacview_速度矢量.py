import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import socket
import time
import keyboard  # pip install keyboard


# ================== Tacview 接口 ==================
class Tacview(object):
    def __init__(self):
        host = "localhost"
        port = 42674
        print("请打开 Tacview 高级版 → 记录 → 实时遥测，设置如下：")
        print(f"IP地址: {host}, 端口: {port}")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        client_socket, address = server_socket.accept()
        print(f"连接成功: {address}")

        self.client_socket = client_socket

        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        client_socket.send(handshake_data.encode())

        data = client_socket.recv(1024)
        print(f"握手应答: {data.decode()}")

        header = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                  "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n")
        client_socket.send(header.encode())

    def send(self, data: str):
        self.client_socket.send(data.encode())


# ================== 四元数旋转函数 ==================
def roll(alpha_hat, alpha, vector):
    """列向量旋转（罗德里格斯公式）"""
    if np.linalg.norm(alpha_hat) < 1e-10:
        return vector

    alpha_hat = alpha_hat / np.linalg.norm(alpha_hat)
    dot_product = np.dot(alpha_hat, vector)
    cross_product = np.cross(alpha_hat, vector)

    new_vector = (1 - np.cos(alpha)) * dot_product * alpha_hat + \
                 np.cos(alpha) * vector + \
                 np.sin(alpha) * cross_product
    return new_vector


# ================== 坐标转换函数 ==================
def show_loc(P):
    """将北天东坐标系转换为东北天坐标系"""
    P_show = np.zeros_like(P)
    P_show[0, :] = P[2, :]  # E -> x
    P_show[1, :] = P[0, :]  # N -> y
    P_show[2, :] = P[1, :]  # U -> z
    return P_show


# ================== 参数设置 ==================
tacview_show = True
if tacview_show:
    tacview = Tacview()

dt = 0.02
t_total = 600.0

# 物理参数
m = 2.0  # 质量
g_earth = 9.8  # 重力加速度
G = np.array([0.0, -1.0, 0.0]) * m * g_earth  # 重力矢量

# 初始状态
p_me = np.array([0.0, 5000.0, 0.0])  # 北天东坐标系 (N, U, E)
v_me = 300.0  # 速度大小

# 初始欧拉角
psi = np.deg2rad(0.0)  # 偏航角
theta = np.deg2rad(0.0)  # 俯仰角
phi = np.deg2rad(0.0)  # 滚转角

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

# 控制参数
n_tx, n_ty, mu_t = 0.0, 1.0, phi
tau_phi = 0.5  # 滚转角响应时间常数

# 数据记录
time_steps = []
positions = []
velocities = []
thetas, psis, phis = [], [], []

# ================== 主循环 ==================
for step in range(int(t_total / dt)):
    t = step * dt
    time_steps.append(t)

    # n_tx, n_ty, mu_t = 0.0, 1.0, phi
    # ======== 键盘控制 ========
    if keyboard.is_pressed('w'):  # 抬头
        n_ty = min(9.0, n_ty + 3)  # 最大 9g
    if keyboard.is_pressed('s'):  # 俯冲
        n_ty = max(-5.0, n_ty - 3)  # 最小 -5g
    if keyboard.is_pressed('a'):  # 左滚
        mu_t = max(-np.pi, mu_t - np.deg2rad(30) )  # 限幅 -180°
    if keyboard.is_pressed('d'):  # 右滚
        mu_t = min(np.pi, mu_t + np.deg2rad(30) )  # 限幅 +180°
    if keyboard.is_pressed('shift'):  # 加速
        n_tx = min(2.0, n_tx + 1 )  # 最大推力系数 2
    if keyboard.is_pressed('ctrl'):  # 减速
        n_tx = max(-2.0, n_tx - 1 )  # 最小推力系数 -2

    # ======== 动力学方程（基于Matlab模型） ========
    # 控制规则
    F_thrust = n_tx * m * g_earth

    # 升力计算
    if v_me > 0:
        F_wing = n_ty * m * g_earth
    else:
        F_wing = 0.0

    # 滚转角一阶响应
    dphi = (mu_t - phi) / tau_phi
    phi = phi + dphi * dt

    # 切向加速度
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
        omega_T_vec = 0.0 * T  # 简化处理
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

    # 更新位置（梯形积分）
    p_me = p_me + (v_me_new_vec + v_me_vec) / 2 * dt
    v_me_vec = v_me_new_vec
    v_me = v_me_new

    # 计算欧拉角（用于记录和Tacview输出）
    v_me_h = np.linalg.norm(np.array([1, 0, 1]) * v_me_vec)
    theta = np.arctan2(v_me_vec[1], v_me_h)  # 俯仰角
    psi = np.arctan2(v_me_vec[2], v_me_vec[0])  # 偏航角

    # # 计算滚转角
    # y_refer = -G_perp_vec / (np.linalg.norm(G_perp_vec) + 1e-10)
    # cosphi = np.dot(y_refer, Ayv)
    # temp = np.cross(y_refer, Ayv)
    # sinphi = np.dot(temp, Axv)
    # phi = np.arctan2(sinphi, cosphi)

    # # 滚转角一阶响应
    # dphi = (mu_t - phi) / tau_phi
    # phi = phi + dphi * dt

    # ======== 保存数据 ========
    positions.append(p_me.copy())
    velocities.append(v_me)
    thetas.append(np.rad2deg(theta))
    psis.append(np.rad2deg(psi))
    phis.append(np.rad2deg(phi))

    # ======== Tacview 输出 ========
    if tacview_show:
        # 转换为东北天坐标系（Tacview使用）
        lon = 116.0 + p_me[2] / 111320.0  # 经度 (E)
        lat = 39.0 + p_me[0] / 110540.0  # 纬度 (N)
        alt = p_me[1]  # 高度 (U)

        data_to_send = "#%.2f\n001,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            t, lon, lat, alt,
            np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi)
        )
        tacview.send(data_to_send)
        time.sleep(0.01)

# ================== 绘图 ==================
positions = np.array(positions).T
x_vals, y_vals, z_vals = positions[0], positions[1], positions[2]  # N, U, E

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 位置随时间变化
axes[0, 0].plot(time_steps, x_vals, label='X-North')
axes[0, 0].plot(time_steps, y_vals, label='Y-Up')
axes[0, 0].plot(time_steps, z_vals, label='Z-East')
axes[0, 0].set_title("Position vs Time")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Position (m)")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 速度随时间变化
axes[0, 1].plot(time_steps, velocities, color='tab:blue')
axes[0, 1].set_title("Speed vs Time")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Speed (m/s)")
axes[0, 1].grid(True)

# 欧拉角随时间变化
axes[1, 0].plot(time_steps, thetas, label='Pitch θ')
axes[1, 0].plot(time_steps, psis, label='Heading ψ')
axes[1, 0].plot(time_steps, phis, label='Roll φ')
axes[1, 0].set_title("Attitude vs Time")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Angle (deg)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 轨迹投影（东-北平面）
axes[1, 1].plot(z_vals, x_vals, label='Z-X plane')
axes[1, 1].set_title("Trajectory (East-North)")
axes[1, 1].set_xlabel("East (m)")
axes[1, 1].set_ylabel("North (m)")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ================== 3D轨迹图 ==================
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 转换为东北天坐标系用于显示
P_show = show_loc(positions)
ax_3d.plot(P_show[0, :], P_show[1, :], P_show[2, :])
ax_3d.scatter(P_show[0, 0], P_show[1, 0], P_show[2, 0], c='r', marker='o', s=50)
ax_3d.set_xlabel("East (m)")
ax_3d.set_ylabel("North (m)")
ax_3d.set_zlabel("Up (m)")
ax_3d.set_title("3D Trajectory")
plt.show()