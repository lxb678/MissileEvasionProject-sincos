import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import keyboard  # pip install keyboard


# ================== Tacview 接口 (No changes needed) ==================
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
t_total = 600.0

# ========== 飞机物理和气动参数 ==========
m = 15000.0
Ix, Iy, Iz = 60000.0, 100000.0, 120000.0
rho = 1.225
S = 50.0
b = 12.0
c = 4.5
max_thrust = 120000.0

# NEW: 定义力作用点相对于质心的位置向量 [x, y, z] (体轴系)
# 气动中心 (通常在质心前方和上方一点)
ac_pos = np.array([0.1 * c, 0.0, -0.1 * b])
# 推力作用点 (通常在质心后方和下方一点)
thrust_pos = np.array([-0.5 * c, 0.0, 0.05 * b])

# 气动导数
C_L_0 = 0.2
C_L_alpha = 4.5
C_L_delta_e = 0.4
C_D_0 = 0.02
k = 0.04
C_Y_beta = -0.8
C_Y_delta_r = 0.2
C_l_p = -0.5
C_l_delta_a = 0.15
C_m_0 = 0.0  # 保持为0，因为其他力矩会由配平处理
C_m_alpha = -1.5
C_m_q = -8.0
C_m_delta_e = -1.8
C_n_beta = 0.15
C_n_r = -0.2
C_n_delta_r = -0.1

# ========== 6-DOF 状态变量 ==========
state = np.array([0.0, 5000.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ========== 控制面状态和执行机构 ==========
throttle = 0.5
delta_e, delta_a, delta_r = 0.0, 0.0, 0.0
actuator_tau = 0.1

# ================== 初始配平计算 ==================
# 目标：计算一个 delta_e_trim 使得在初始飞行状态下，总俯仰力矩 M 为 0
# 初始状态: u=300, v=0, w=0, theta=0, q=0, alpha=0
initial_u = state[3]
initial_Vt = initial_u
initial_Q_bar = 0.5 * rho * initial_Vt**2

# 1. 初始升力和阻力
initial_C_L = C_L_0 # alpha=0, delta_e=0
initial_Lift = initial_Q_bar * S * initial_C_L
initial_Drag = initial_Q_bar * S * (C_D_0 + k * initial_C_L**2)
initial_aero_force_body = np.array([-initial_Drag, 0, -initial_Lift])

# 2. 初始推力
# 为了平飞，推力需要抵消阻力和重力分量。简化为抵消阻力。
initial_thrust = initial_Drag
# 也可以根据初始油门计算
# initial_thrust = throttle * max_thrust
initial_thrust_force = np.array([initial_thrust, 0, 0])

# 3. 计算由气动力和推力产生的力矩
M_aero_trim = np.cross(ac_pos, initial_aero_force_body)[1] # 取y分量
M_thrust_trim = np.cross(thrust_pos, initial_thrust_force)[1] # 取y分量
M_unbalanced = M_aero_trim + M_thrust_trim + (initial_Q_bar * S * c * C_m_0)

# 4. 计算抵消该不平衡力矩所需的升降舵偏角
# M_delta_e = Q_bar * S * c * C_m_delta_e * delta_e_trim
# 所以 delta_e_trim = -M_unbalanced / (Q_bar * S * c * C_m_delta_e)
delta_e_trim = -M_unbalanced / (initial_Q_bar * S * c * C_m_delta_e)

print(f"初始配平计算完成。")
print(f" - 不平衡力矩: {M_unbalanced:.2f} Nm")
print(f" - 所需配平舵偏: {np.rad2deg(delta_e_trim):.4f} deg")

# 数据记录
time_steps, positions, velocities = [], [], []
thetas, psis, phis = [], [], []

# ================== 主循环 ==================
for step in range(int(t_total / dt)):
    x, y, z, u, v, w, phi, theta, psi, p, q, r = state
    t = step * dt
    time_steps.append(t)

    # ======== 键盘控制 -> 目标舵偏/油门 ========
    throttle_cmd = throttle
    # 目标舵偏现在是配平值 + 手动输入
    delta_e_cmd, delta_a_cmd, delta_r_cmd = delta_e_trim, 0.0, 0.0

    if keyboard.is_pressed('w'): delta_e_cmd += np.deg2rad(-20)
    if keyboard.is_pressed('s'): delta_e_cmd += np.deg2rad(20)
    if keyboard.is_pressed('a'): delta_a_cmd = np.deg2rad(-20)
    if keyboard.is_pressed('d'): delta_a_cmd = np.deg2rad(20)
    if keyboard.is_pressed('q'): delta_r_cmd = np.deg2rad(-20)
    if keyboard.is_pressed('e'): delta_r_cmd = np.deg2rad(20)
    if keyboard.is_pressed('shift'): throttle_cmd = min(1.0, throttle + 0.5 * dt)
    if keyboard.is_pressed('ctrl'): throttle_cmd = max(0.0, throttle - 0.5 * dt)

    # ======== 舵面和油门执行机构模型 ========
    throttle += (throttle_cmd - throttle) / actuator_tau * dt
    delta_e  += (delta_e_cmd - delta_e) / actuator_tau * dt
    delta_a  += (delta_a_cmd - delta_a) / actuator_tau * dt
    delta_r  += (delta_a_cmd - delta_r) / actuator_tau * dt # Bug fix: was delta_a_cmd

    # ======== 6-DOF 动力学方程 ========
    Vt = np.sqrt(u**2 + v**2 + w**2)
    if Vt < 1.0: Vt = 1.0
    alpha = np.arctan2(w, u) if u != 0 else np.pi/2 * np.sign(w)
    beta = np.arcsin(v / Vt) if Vt != 0 else 0.0
    Q_bar = 0.5 * rho * Vt**2

    # 气动力/力矩系数
    C_L = C_L_0 + C_L_alpha * alpha + C_L_delta_e * delta_e
    C_D = C_D_0 + k * C_L**2
    C_Y = C_Y_beta * beta + C_Y_delta_r * delta_r
    p_hat, q_hat, r_hat = p*b/(2*Vt), q*c/(2*Vt), r*b/(2*Vt)
    C_l = C_l_p * p_hat + C_l_delta_a * delta_a
    C_m = C_m_0 + C_m_alpha * alpha + C_m_q * q_hat + C_m_delta_e * delta_e
    C_n = C_n_beta * beta + C_n_r * r_hat + C_n_delta_r * delta_r

    # 计算风轴系气动力并旋转到体轴系
    Lift = Q_bar * S * C_L
    Drag = Q_bar * S * C_D
    SideForce = Q_bar * S * C_Y
    ca, sa = np.cos(alpha), np.sin(alpha)
    aero_force_body = np.array([-Drag*ca + Lift*sa, SideForce, -Drag*sa - Lift*ca])

    # 总的力和力矩
    thrust_force = np.array([throttle * max_thrust, 0, 0])
    total_force = aero_force_body + thrust_force

    # MODIFIED: 力矩计算现在包括了力和力矩臂的叉积
    M_aero_pure = Q_bar * S * np.array([b*C_l, c*C_m, b*C_n])
    M_aero_pos = np.cross(ac_pos, aero_force_body)
    M_thrust_pos = np.cross(thrust_pos, thrust_force)
    total_moment = M_aero_pure + M_aero_pos + M_thrust_pos

    Fx, Fy, Fz = total_force
    L, M, N = total_moment

    # 刚体动力学/运动学方程 (这部分不变)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    dx = u*(c_theta*np.cos(psi)) + v*(s_phi*s_theta*np.cos(psi) - c_phi*np.sin(psi)) + w*(c_phi*s_theta*np.cos(psi) + s_phi*np.sin(psi))
    dy = u*s_theta - v*s_phi*c_theta - w*c_phi*c_theta
    dz = u*(c_theta*np.sin(psi)) + v*(s_phi*s_theta*np.sin(psi) + c_phi*np.cos(psi)) + w*(c_phi*s_theta*np.sin(psi) - s_phi*np.cos(psi))
    du = Fx/m + g*s_theta + r*v - q*w
    dv = Fy/m - g*c_theta*s_phi + p*w - r*u
    dw = Fz/m - g*c_theta*c_phi + q*u - p*v
    tan_theta = np.tan(theta)
    dphi = p + q*s_phi*tan_theta + r*c_phi*tan_theta
    dtheta = q*c_phi - r*s_phi
    dpsi = (q*s_phi + r*c_phi)/c_theta if abs(c_theta)>0.01 else 0
    dp = (L + (Iy - Iz)*q*r)/Ix
    dq = (M + (Iz - Ix)*r*p)/Iy
    dr = (N + (Ix - Iy)*p*q)/Iz

    dstate = np.array([dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi, dp, dq, dr])
    state = state + dt * dstate

    # ======== 数据保存 & Tacview 输出 (No changes needed) ========
    positions.append((x, y, z)); velocities.append(Vt)
    thetas.append(np.rad2deg(theta)); psis.append(np.rad2deg(psi)); phis.append(np.rad2deg(phi))
    if tacview_show:
        lon = 116.0 + z / 111320.0
        lat = 39.0 + x / 110540.0
        alt = y
        data_to_send = "#%.2f\n001,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            t, lon, lat, alt, np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi)
        )
        tacview.send(data_to_send)
        time.sleep(0.01)

# ================== 绘图 (No changes needed) ==================
x_vals, y_vals, z_vals = zip(*positions)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].plot(time_steps, y_vals); axes[0, 0].set_title("Altitude vs Time"); axes[0, 0].set_ylabel("Altitude (m)"); axes[0, 0].grid(True)
axes[0, 1].plot(time_steps, velocities, color='tab:blue'); axes[0, 1].set_title("Speed vs Time"); axes[0, 1].set_ylabel("Speed (m/s)"); axes[0, 1].grid(True)
axes[1, 0].plot(time_steps, thetas, label='Pitch θ'); axes[1, 0].plot(time_steps, psis, label='Heading ψ'); axes[1, 0].plot(time_steps, phis, label='Roll φ'); axes[1, 0].set_title("Attitude Angles vs Time"); axes[1, 0].set_ylabel("Angle (deg)"); axes[1, 0].legend(); axes[1, 0].grid(True)
axes[1, 1].plot(z_vals, x_vals); axes[1, 1].set_title("Trajectory (Top-Down View)"); axes[1, 1].set_xlabel("East (m)"); axes[1, 1].set_ylabel("North (m)"); axes[1, 1].grid(True); axes[1, 1].axis('equal')
plt.tight_layout(); plt.show()