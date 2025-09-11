import numpy as np
import matplotlib.pyplot as plt
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


# ================== 参数设置 ==================
tacview_show = True
if tacview_show:
    tacview = Tacview()

g = 9.81
dt = 0.02
t_total = 60.0

# 状态变量 [x, y, z, Vt, theta, psi, phi]
state = np.array([0.0, 5000.0, 0.0, 300.0,
                  0.0, np.deg2rad(0.0), 0.0])
# x,y,z(m), Vt(m/s), θ(rad), ψ(rad), φ(rad)

tau_phi = 0.5   # 滚转角响应时间常数 (s)

# 数据记录
time_steps = []
positions = []
velocities = []
thetas, psis, phis = [], [], []
n_tx, n_ty, mu_t = 0.0, 1.0, 0


# ================== 主循环 ==================
for step in range(int(t_total / dt)):
    x, y, z, Vt, theta, psi, phi = state
    t = step * dt
    time_steps.append(t)

    # ======== 键盘控制：n_tx, n_ty, mu_t ========
    n_tx, n_ty, mu_t = 0.0, 1.0, phi
    if keyboard.is_pressed('w'):  # 抬头
        n_ty = min(9.0, n_ty + 6)  # 最大 9g
    if keyboard.is_pressed('s'):  # 俯冲
        n_ty = max(-5.0, n_ty - 6)  # 最小 -5g
    if keyboard.is_pressed('a'):  # 左滚
        mu_t = max(-np.pi, mu_t - np.deg2rad(30))  # 每次按键 -5°，限幅 -180°
    if keyboard.is_pressed('d'):  # 右滚
        mu_t = min(np.pi, mu_t + np.deg2rad(30))  # 每次按键 +5°，限幅 +180°
    if keyboard.is_pressed('shift'):  # 加速
        n_tx = min(2.0, n_tx + 1)  # 最大推力系数 2
    if keyboard.is_pressed('ctrl'):  # 减速
        n_tx = max(-2.0, n_tx - 1)  # 最小推力系数 -2

    # ======== 动力学方程 ========
    dxt    = Vt * np.cos(theta) * np.cos(psi)
    dyt    = Vt * np.sin(theta)
    dzt    = Vt * np.cos(theta) * np.sin(psi)
    dVt    = g * (n_tx - np.sin(theta))
    dtheta = (g / Vt) * (n_ty * np.cos(phi) - np.cos(theta))
    dpsi   = (n_ty * g * np.sin(phi)) / (Vt * np.cos(theta))
    dphi   = (mu_t - phi) / tau_phi   # 滚转角一阶动力学

    dstate = np.array([dxt, dyt, dzt, dVt, dtheta, dpsi, dphi])
    state = state + dt * dstate  # 欧拉积分

    # # === 加上俯仰角限制 ===
    # if state[4] > np.pi / 2:  # theta 最大 90°
    #     # state[4] = np.pi - state[4]
    #     state[6] +=  np.pi
    # elif state[4] < -np.pi / 2:  # theta 最小 -90°
    #     state[4] = -np.pi / 2

    # ======== 保存数据 ========
    positions.append((x, y, z))
    velocities.append(Vt)
    thetas.append(np.rad2deg(theta))
    psis.append(np.rad2deg(psi))
    phis.append(np.rad2deg(phi))

    # ======== Tacview 输出 ========
    if tacview_show:
        lon = 116.0 + z / 111320.0  # 经度 (E)
        lat = 39.0 + x / 110540.0  # 纬度 (N)
        alt = y  # 天 = 高度Up

        data_to_send = "#%.2f\n001,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            t, lon, lat, alt,
            np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi)
        )
        tacview.send(data_to_send)
        time.sleep(0.01)

# ================== 绘图 ==================
x_vals, y_vals, z_vals = zip(*positions)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(time_steps, x_vals, label='X-East')
axes[0, 0].plot(time_steps, y_vals, label='Y-North')
axes[0, 0].plot(time_steps, z_vals, label='Z-Height')
axes[0, 0].set_title("Position vs Time")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Position (m)")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time_steps, velocities, color='tab:blue')
axes[0, 1].set_title("Speed vs Time")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Speed (m/s)")
axes[0, 1].grid(True)

axes[1, 0].plot(time_steps, thetas, label='Pitch θ')
axes[1, 0].plot(time_steps, psis, label='Heading ψ')
axes[1, 0].plot(time_steps, phis, label='Roll φ')
axes[1, 0].set_title("Attitude vs Time")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Angle (deg)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 轨迹投影
axes[1, 1].plot(x_vals, z_vals, label='X-Z plane')
axes[1, 1].set_title("Trajectory (X-Z)")
axes[1, 1].set_xlabel("X-East (m)")
axes[1, 1].set_ylabel("Height (m)")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
