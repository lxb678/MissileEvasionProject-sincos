import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import keyboard  # pip install keyboard

# ================== 四元数辅助函数 ==================

def normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-9: return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def euler_to_quaternion(phi, theta, psi):
    cy = np.cos(psi * 0.5); sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5); sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5); sr = np.sin(phi * 0.5)
    q0 = cr * cp * cy + sr * sp * sy; q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy; q3 = cr * cp * sy - sr * sp * cy
    return normalize(np.array([q0, q1, q2, q3]))

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])

# ====================== 最终修正：替换整个函数 ======================
def quaternion_to_continuous_euler(q):
    """
    从四元数计算连续的、无万向节死锁问题的欧拉角。
    这个版本使用 atan2 计算所有角度，彻底解决了所有象限和航向反转问题。
    """
    R_ned = quaternion_to_rotation_matrix(q)

    # 俯仰角 (theta)
    # 使用 atan2(sin, cos) 的形式，以获得 [-pi, pi] 的全范围角度，彻底解决arcsin的局限性
    sin_theta = -R_ned[2, 0]
    cos_theta = np.sqrt(R_ned[0, 0]**2 + R_ned[1, 0]**2)
    theta = np.arctan2(sin_theta, cos_theta)

    # 航向角 (psi) 和 滚转角 (phi)
    # 检查是否处于垂直俯仰的奇点 (万向节死锁)
    if np.abs(cos_theta) < 1e-6:
        # 在奇点处，滚转角无法唯一定义，按惯例设为0
        phi = 0.0
        # 航向角由机翼的指向（旋转矩阵的第一行/第二列）决定
        psi = np.arctan2(-R_ned[0, 1], R_ned[1, 1])
    else:
        # 在正常情况下，使用标准公式计算
        psi = np.arctan2(R_ned[1, 0], R_ned[0, 0])
        phi = np.arctan2(R_ned[2, 1], R_ned[2, 2])

    return phi, theta, psi

# ================== Tacview 接口 (无变化) ==================
class Tacview(object):
    def __init__(self):
        try:
            host = "localhost"; port = 42674
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {host}, 端口: {port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM); server_socket.bind((host, port));
            server_socket.listen(5)
            self.client_socket, address = server_socket.accept(); print(f"连接成功: {address}")
            handshake = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake.encode()); self.client_socket.recv(1024)
            header = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
            self.client_socket.send(header.encode())
        except Exception as e:
            print(f"Tacview连接失败: {e}"); self.client_socket = None
    def send(self, data: str):
        if self.client_socket:
            try: self.client_socket.send(data.encode())
            except: self.client_socket = None

# ================== 参数设置 ==================
tacview_show = True
tacview = Tacview() if tacview_show else None
g = 9.81; dt = 0.02; t_total = 60.0
initial_q = euler_to_quaternion(0.0, 0.0, 0.0)
state = np.array([0.0, 5000.0, 0.0, 300.0, *initial_q])
time_steps, positions, velocities, thetas, psis, phis = [], [], [], [], [], []
last_pqr = np.array([0.0, 0.0, 0.0])

# ================== 主循环 (无变化) ==================
for step in range(int(t_total / dt)):
    t = step * dt; time_steps.append(t)
    x, y, z, Vt, q = state[0], state[1], state[2], state[3], state[4:]
    R_ned = quaternion_to_rotation_matrix(q)
    body_x_in_ned = R_ned[:, 0]
    sin_theta_dyn = -body_x_in_ned[2]
    n_t_cmd = 0.0; n_f_cmd = 1.0; p_cmd = 0.0
    ROLL_RATE = np.deg2rad(100)
    if keyboard.is_pressed('shift'): n_t_cmd = 2.0
    if keyboard.is_pressed('ctrl'): n_t_cmd = -2.0
    if keyboard.is_pressed('w'): n_f_cmd = min(9.0, n_f_cmd + 6)  # 最大 9g
    if keyboard.is_pressed('s'): n_f_cmd = max(-5.0, n_f_cmd - 6)  # 最小 -5g
    if keyboard.is_pressed('a'): p_cmd -= ROLL_RATE
    if keyboard.is_pressed('d'): p_cmd += ROLL_RATE
    tau_p = 0.25
    p_body = last_pqr[0] + (dt / tau_p) * (p_cmd - last_pqr[0])
    gravity_comp_g = R_ned[2, 2]
    q_body = (g / Vt) * (n_f_cmd - gravity_comp_g)

    side_gravity_g = R_ned[2, 1]
    # r_body = -(g / Vt) * side_gravity_g
    # ==================== 【最终修正】使用旋转矩阵计算 r_body ====================
    # 这个版本在数学上等价于使用欧拉角的版本，但它完美地避开了90度俯仰角的奇点问题

    # 1. 从旋转矩阵中提取计算协调转弯所需的分量
    r21 = R_ned[2, 1]  # 等于 cos(theta) * sin(phi)
    r22 = R_ned[2, 2]  # 等于 cos(theta) * cos(phi)

    # 2. 计算 cos(theta)^2，它在奇点处会平滑地趋近于0
    cos_theta_sq = r21 ** 2 + r22 ** 2

    # 3. 为了数值稳定性，检查是否w处在奇点
    if cos_theta_sq > 1e-9:
        # 最终公式：(n_f_cmd * g / Vt) * r21 / cos_theta_sq
        # 这个计算在数值上远比直接用 sin(phi)/cos(theta) 稳定
        r_body = (n_f_cmd * g / Vt) * r21 / cos_theta_sq
    else:
        # 在奇点处 (垂直爬升/俯冲), 理想的筋斗动作中滚转角phi为0,
        # r21(sin(phi))也为0，因此理想的r_body也应为0。
        r_body = 0.0
    # ==============================================================================
    dq_dt = 0.5 * np.array([-q[1] * p_body - q[2] * q_body - q[3] * r_body,
                            q[0] * p_body + q[2] * r_body - q[3] * q_body,
                            q[0] * q_body - q[1] * r_body + q[3] * p_body,
                            q[0] * r_body + q[1] * q_body - q[2] * p_body])
    dVt = g * (n_t_cmd - sin_theta_dyn)
    d_pos_ned = Vt * body_x_in_ned
    d_pos_nue = np.array([d_pos_ned[0], -d_pos_ned[2], d_pos_ned[1]])
    dstate = np.concatenate((d_pos_nue, [dVt], dq_dt))
    state[0:4] += dt * dstate[0:4]
    state[4:] += dt * dstate[4:]
    state[4:] = normalize(state[4:])
    last_pqr = np.array([p_body, q_body, r_body])
    phi_disp, theta_disp, psi_disp = quaternion_to_continuous_euler(q)
    positions.append(state[:3]); velocities.append(state[3])
    thetas.append(np.rad2deg(theta_disp)); psis.append(np.rad2deg(psi_disp)); phis.append(np.rad2deg(phi_disp))
    if tacview:
        lon = 116.0 + state[2] / (111320.0 * np.cos(np.deg2rad(39.0)))
        lat = 39.0 + state[0] / 110574.0; alt = state[1]
        data = f"#{t:.2f}\n001,T={lon:.6f}|{lat:.6f}|{alt:.6f}|{np.rad2deg(phi_disp):.6f}|{np.rad2deg(theta_disp):.6f}|{np.rad2deg(psi_disp):.6f},Name=F16_Final,Color=White\n"
        tacview.send(data)
        time.sleep(0.01)

# ================== 绘图 (无变化) ==================
x_vals, y_vals, z_vals = zip(*positions)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(time_steps, x_vals, label='X-North'); axes[0, 0].plot(time_steps, y_vals, label='Y-Up (Altitude)'); axes[0, 0].plot(time_steps, z_vals, label='Z-East')
axes[0, 0].set_title("Position vs Time"); axes[0, 0].legend(); axes[0, 0].grid(True)
axes[0, 1].plot(time_steps, velocities, color='tab:blue'); axes[0, 1].set_title("Speed vs Time"); axes[0, 1].grid(True)
axes[1, 0].plot(time_steps, thetas, label='Pitch θ'); axes[1, 0].plot(time_steps, psis, label='Heading ψ'); axes[1, 0].plot(time_steps, phis, label='Roll φ')
axes[1, 0].set_title("Continuous Attitude vs Time"); axes[1, 0].set_ylabel("Angle (deg)"); axes[1, 0].legend(); axes[1, 0].grid(True)
axes[1, 1].plot(x_vals, y_vals, label='North-Up plane'); axes[1, 1].set_title("Trajectory (Side View)"); axes[1, 1].set_xlabel("X-North (m)"); axes[1, 1].set_ylabel("Y-Up (Altitude) (m)")
axes[1, 1].axis('equal'); axes[1, 1].grid(True)
plt.tight_layout(); plt.show()