import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import keyboard  # pip install keyboard


# ================== 坐标系定义 ==================
# 惯性坐标系 (Inertial Frame):
#   - 外部状态表示: 北-天-东 (North-Up-East, NUE) -> 用于 state 向量和绘图
#   - 内部物理计算: 北-东-地 (North-East-Down, NED) -> 用于核心物理引擎
# 机体坐标系 (Body Frame): 前-右-下 (Forward-Right-Down, FRD)
# =========================================================


# ================== 四元数与旋转矩阵函数 (保持不变) ==================
# 这些函数依然基于 NED/FRD 的欧拉角定义 (phi-roll, theta-pitch, psi-yaw)
def normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-9: return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def euler_to_quaternion(phi, theta, psi):
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return normalize(np.array([q0, q1, q2, q3]))


def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    # 从机体系(FRD)到惯性系(NED)的旋转矩阵 R_frd_to_ned
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)]
    ])


def quaternion_to_continuous_euler(q):
    R = quaternion_to_rotation_matrix(q)
    sin_theta = -R[2, 0]
    theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    cos_theta = np.cos(theta)
    if np.abs(cos_theta) < 1e-6:
        phi = 0.0
        psi = np.arctan2(-R[0, 1], R[1, 1])
        print("np.abs(cos_theta) < 1e-6")
        # psi = 0.0
    else:
        psi = np.arctan2(R[1, 0], R[0, 0])
        phi = np.arctan2(R[2, 1], R[2, 2])
    return phi, theta, psi


# ================== Tacview 接口 (无变化) ==================
class Tacview(object):
    def __init__(self):
        try:
            host = "localhost"
            port = 42674
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {host}, 端口: {port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(5)
            self.client_socket, address = server_socket.accept()
            print(f"连接成功: {address}")
            handshake = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake.encode())
            self.client_socket.recv(1024)
            header = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
            self.client_socket.send(header.encode())
        except Exception as e:
            print(f"Tacview连接失败: {e}")
            self.client_socket = None

    def send(self, data: str):
        if self.client_socket:
            try:
                self.client_socket.send(data.encode())
            except:
                self.client_socket = None


# ================== 参数设置 ==================
tacview_show = True
tacview = Tacview() if tacview_show else None
g = 9.81
dt = 0.02
t_total = 60.0
m = 1.0

# MODIFIED: 状态向量定义更新为 北-天-东 (NUE)
# 状态向量: [x(北), y(上), z(东), Vt, theta(俯仰), psi(偏航), phi(滚转)]
# 初始高度 5000 米，在 NUE 中 y 坐标为 5000
state = np.array([0.0, 5000.0, 0.0, 300.0, 0.0, 0.0, 0.0])
time_steps, positions, velocities, thetas, psis, phis = [], [], [], [], [], []

# ================== 主循环 (状态NUE, 物理NED/FRD) ==================
for step in range(int(t_total / dt)):
    t = step * dt
    time_steps.append(t)

    # === 解包状态向量 (NUE 格式) ===
    x_nue, y_nue, z_nue, Vt, theta, psi, phi = state
    if psi == -0.00:
        psi = 0.00
    # print(f"t={t:.2f}, x={x_nue:.2f}, y={y_nue:.2f}, z={z_nue:.2f}, Vt={Vt:.2f}, theta={np.rad2deg(theta):.2f}, psi={np.rad2deg(psi):.2f}, phi={np.rad2deg(phi):.2f}")

    if tacview:
        lon = 116.0 + state[2] / (111320.0 * np.cos(np.deg2rad(39.0)))  # state[2] is East
        lat = 39.0 + state[0] / 110574.0  # state[0] is North
        alt = state[1]  # state[1] is Up (Altitude)
        data = f"#{t:.2f}\n001,T={lon:.6f}|{lat:.6f}|{alt:.1f}|{np.rad2deg(phi):.2f}|{np.rad2deg(theta):.2f}|{np.rad2deg(psi):.2f},Name=F16_NUE_State,Color=Blue\n"
        print(data)
        tacview.send(data)
        time.sleep(0.01)

    # === 坐标转换: 从 NUE (状态) 到 NED (物理) ===
    # pos_ned = [North, East, Down]
    # x_ned = x_nue
    # y_ned = z_nue
    # z_ned = -y_nue
    pos_ned = np.array([x_nue, z_nue, -y_nue])

    # === 转换 (欧拉角到旋转矩阵) ===
    q_current = euler_to_quaternion(phi, theta, psi)
    R_frd_to_ned = quaternion_to_rotation_matrix(q_current)
    R_ned_to_frd = R_frd_to_ned.T

    # --- 核心物理计算 ---
    # 1. 获取控制指令
    # --- 滚转控制 (保持不变) ---
    phi_cmd = phi  # 默认保持当前滚转角
    if keyboard.is_pressed('a'):     phi_cmd -= np.deg2rad(60)  # 左滚
    if keyboard.is_pressed('d'):     phi_cmd += np.deg2rad(60)  # 右滚

    # --- 过载指令 ---
    # 默认值: 维持速度(nx=0), 维持1g平飞(nz=1)
    nx_cmd = 0.0  # 切向过载指令
    nz_cmd = 1.0  # 法向过载指令

    # 'shift' / 'ctrl' 控制切向过载 (推背感)
    if keyboard.is_pressed('shift'): nx_cmd = 2.0  # 2g 加速
    if keyboard.is_pressed('ctrl'):  nx_cmd = -2.0  # 2g 减速

    # 'w' / 's' 控制法向过载 (拉杆/推杆)
    if keyboard.is_pressed('w'):     nz_cmd = 9.0  # 拉 9g
    if keyboard.is_pressed('s'):     nz_cmd = -2.0  # 推 -2g

    # --- 将过载指令转换为力 ---
    thrust_minus_drag = nx_cmd * m * g
    lift = nz_cmd * m * g

    # 对升力进行限制，模拟飞机结构极限
    lift = np.clip(lift, -3.0 * m * g, 9.0 * m * g)

    # 2. MODIFIED: 在机体系(FRD: 前-右-下)中定义气动力
    # 升力 (lift) 产生向上的力, 在 FRD 的 Z 轴 (向下) 是负方向
    F_aero_frd = np.array([thrust_minus_drag, 0, -lift])
    V_vec_frd = np.array([Vt, 0, 0])  # 速度矢量在机体系中总是指向前方

    # 3. 转换到惯性系 (NED)
    F_aero_ned = R_frd_to_ned @ F_aero_frd
    V_vec_ned = R_frd_to_ned @ V_vec_frd

    # 4. MODIFIED: 在 NED 中加入重力
    # 重力指向地心, 即 NED 的 Z 轴正方向
    F_gravity_ned = np.array([0, 0, m * g])
    F_total_ned = F_aero_ned + F_gravity_ned

    # 5. 计算航迹角速度
    if Vt < 1e-3:
        omega_ned = np.array([0.0, 0.0, 0.0])
    else:
        # F_total_ned 中垂直于 V_vec_ned 的分力导致了速度方向的改变
        F_perp_ned = F_total_ned - np.dot(F_total_ned, V_vec_ned / Vt) * (V_vec_ned / Vt)
        # a_perp = F_perp / m, R = V^2 / a_perp, omega = V / R = a_perp / V
        omega_ned = np.cross(V_vec_ned, F_total_ned) / (m * Vt * Vt)

    # 6. 转换回机体系 (FRD)
    omega_frd = R_ned_to_frd @ omega_ned

    # 7. 混合控制与角速度最终确定
    # 滚转率 p (绕 X 轴) 由直接控制决定
    tau_p = 0.1
    p_body_cmd = (1 / tau_p) * (phi_cmd - phi)
    max_roll_rate = np.deg2rad(240)
    p_body = np.clip(p_body_cmd, -max_roll_rate, max_roll_rate)

    # MODIFIED: 俯仰率 q (绕 Y 轴) 和 偏航率 r (绕 Z 轴) 由动力学决定
    q_body = omega_frd[1]  # Pitch rate is omega_y
    r_body = omega_frd[2]  # Yaw rate is omega_z

    # --- 动力学与运动学积分 (仍然在 NED/FRD 框架下) ---
    p, q, r = p_body, q_body, r_body
    dq_dt = 0.5 * np.array([
        -q_current[1] * p - q_current[2] * q - q_current[3] * r,
        q_current[0] * p + q_current[2] * r - q_current[3] * q,
        q_current[0] * q - q_current[1] * r + q_current[3] * p,
        q_current[0] * r + q_current[1] * q - q_current[2] * p
    ])
    q_new = normalize(q_current + dt * dq_dt)
    phi_new, theta_new, psi_new = quaternion_to_continuous_euler(q_new)

    acceleration_ned = F_total_ned / m
    V_unit_vec_ned = V_vec_ned / Vt if Vt > 1e-3 else np.array([1., 0., 0.])
    Vt_new = Vt + dt * np.dot(acceleration_ned, V_unit_vec_ned)

    # 计算 NED 坐标下的位移
    d_pos_ned = (R_frd_to_ned @ np.array([Vt, 0, 0])) * dt
    pos_ned_new = pos_ned + d_pos_ned

    # === 坐标转换: 从 NED (物理) 回到 NUE (状态) ===
    # x_nue = x_ned
    # y_nue = -z_ned
    # z_nue = y_ned
    x_nue_new = pos_ned_new[0]
    y_nue_new = -pos_ned_new[2]
    z_nue_new = pos_ned_new[1]

    # === 更新整个 state 向量 (NUE 格式) ===
    state[0] = x_nue_new
    state[1] = y_nue_new
    state[2] = z_nue_new
    state[3] = Vt_new
    state[4:] = [theta_new, psi_new, phi_new]

    # === 数据记录与显示 (现在直接使用 NUE 状态) ===
    phi_disp, theta_disp, psi_disp = np.rad2deg(state[6]), np.rad2deg(state[4]), np.rad2deg(state[5])
    positions.append(state[:3].copy())  # 存储 [North, Up, East]
    velocities.append(state[3].copy())
    thetas.append(theta_disp.copy())
    psis.append(psi_disp.copy())
    phis.append(phi_disp.copy())

    # if tacview:
    #     lon = 116.0 + state[2] / (111320.0 * np.cos(np.deg2rad(39.0)))  # state[2] is East
    #     lat = 39.0 + state[0] / 110574.0  # state[0] is North
    #     alt = state[1]  # state[1] is Up (Altitude)
    #     data = f"#{t:.2f}\n001,T={lon:.6f}|{lat:.6f}|{alt:.1f}|{phi_disp:.2f}|{theta_disp:.2f}|{psi_disp:.2f},Name=F16_NUE_State,Color=Blue\n"
    #     print(data)
    #     tacview.send(data)
    #     time.sleep(0.01)

# ================== 绘图 (已修改为 NUE 格式) ==================
x_vals, y_vals, z_vals = zip(*positions)  # North, Up, East

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(time_steps, x_vals, label='X-North')
axes[0, 0].plot(time_steps, y_vals, label='Y-Up (Altitude)')  # y_vals is now altitude
axes[0, 0].plot(time_steps, z_vals, label='Z-East')
axes[0, 0].set_title("Position vs Time")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time_steps, velocities, color='tab:blue')
axes[0, 1].set_title("Speed vs Time")
axes[0, 1].grid(True)

axes[1, 0].plot(time_steps, thetas, label='Pitch θ (Nose Up +)')
axes[1, 0].plot(time_steps, psis, label='Heading ψ (Turn Right +)')
axes[1, 0].plot(time_steps, phis, label='Roll φ (Right Wing Down +)')
axes[1, 0].set_title("Attitude vs Time")
axes[1, 0].set_ylabel("Angle (deg)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 绘制 东-高度 轨迹图
axes[1, 1].plot(z_vals, y_vals, label='East-Altitude plane')  # z_vals (East) vs y_vals (Altitude)
axes[1, 1].set_title("Trajectory (Side View from South)")
axes[1, 1].set_xlabel("Z-East (m)")
axes[1, 1].set_ylabel("Y-Up (Altitude) (m)")
axes[1, 1].axis('equal')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()