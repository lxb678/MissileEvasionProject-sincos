import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========================= 飞机动力学模型 =========================
def aircraft_dynamics(t, x, control_input, g):
    xt, yt, zt, Vt, theta, psi = x
    n_tx, n_ty, mu_t = control_input(x)
    dxt = Vt * np.cos(theta) * np.cos(psi)
    dyt = Vt * np.sin(theta)
    dzt = -Vt * np.cos(theta) * np.sin(psi)
    dVt = g * (n_tx - np.sin(theta))
    dtheta = (g / Vt) * (n_ty * np.cos(mu_t) - np.cos(theta))
    dpsi = -(n_ty * g * np.sin(mu_t)) / (Vt * np.cos(theta))
    return np.array([dxt, dyt, dzt, dVt, dtheta, dpsi])


# ========================= 飞机红外辐射强度模型 =========================
# 关键角度点（单位：度），0°表示导弹正对飞机尾部，180°为正前方
beta_deg = np.array([0, 40, 90, 140, 180])
intensity_vals = np.array([3800, 5000, 2500, 2000, 800])
beta_rad = np.deg2rad(beta_deg)

# 三次样条插值模型
spline_model = CubicSpline(beta_rad, intensity_vals, bc_type='natural')

def infrared_intensity_model(beta):
    """
    输入：beta（弧度），导弹相对飞机机体的水平角
    输出：对应红外辐射强度
    """
    return np.maximum(spline_model(beta), 0.0)

# 相对角度计算（
def compute_relative_beta(x_target, x_missile):
    """
    输入：
        x_target: 飞机状态向量 [x, y, z, V, theta, psi]
        x_missile: 导弹状态向量 [V, theta, psi, x, y, z]
    输出：
        beta: 目标视线相对于飞机机头水平方向的夹角（弧度）
    """
    # 弹目相对矢量（R）并投影到水平面（xz）
    R_vec = x_target[0:3] - x_missile[3:6]
    R_proj = np.array([R_vec[0], 0.0, R_vec[2]])

    # 飞机水平方向单位向量（注意坐标系方向 x前 y上 z右）
    psi_t = x_target[5]  # 飞机偏航角
    V_body = np.array([np.cos(psi_t), 0.0, -np.sin(psi_t)])  # 机头朝向单位向量

    # 计算夹角 β（视线与机头水平夹角）
    cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    return beta


# ========================= 导弹动力学模型 =========================
def missile_dynamics_given_rate( y, theta_L_dot, phi_L_dot, N, ny_max, nz_max):
    V, theta, psi_c, x, y_pos, z = y
    g = 9.81

    # === 导引律过载控制 ===
    ny_cmd = N * V * theta_L_dot / g
    nz_cmd = N * V * phi_L_dot / g
    ny = np.clip(ny_cmd, -ny_max, ny_max)
    nz = np.clip(nz_cmd, -nz_max, nz_max)

    # === 飞行高度 ===
    H = y_pos  # y坐标为高度（天方向）

    # === 空气温度、密度、音速 ===
    Temper = 15.0  # 海平面温度（单位°C）
    T_H = 273 + Temper - 0.6 * H / 100  # 高度相关温度 (K)
    P_H = (1 - H / 44300) ** 5.256  # 气压随高度变化（相对比值）
    rho = 1.293 * P_H * (273 / T_H)  # 空气密度 ρ

    # === 马赫数 Ma、音速 ===
    Ma = V / 340

    # === 阻力系数（分段） ===
    def get_Cx(Ma):
        if Ma <= 1.0:
            return 0.45
        elif Ma <= 1.3:
            return 0.45 + (0.65 - 0.45) * (Ma - 1.0) / (1.3 - 1.0)
        elif Ma <= 4.0:
            return 0.65 * np.exp(-0.5 * (Ma - 1.3))
        else:
            return 0.3

    Cx = get_Cx(Ma)

    # === 动压、空气阻力 ===
    S = np.pi * (0.127) ** 2 / 4  # 假设导弹直径 0.15 m，对应迎风面积
    q = 0.5 * rho * V ** 2
    X = Cx * q * S  # 空气阻力

    # === 导弹动力学微分方程 ===
    m = 86  # 导弹质量（单位 kg，可根据实际更改）
    dV = (-X - m * g * np.sin(theta)) / m
    dtheta = (ny - np.cos(theta)) * g / V
    dpsi_c = -nz * g / (V * np.cos(theta))
    dx = V * np.cos(theta) * np.cos(psi_c)
    dy = V * np.sin(theta)
    dz = -V * np.cos(theta) * np.sin(psi_c)

    return np.array([dV, dtheta, dpsi_c, dx, dy, dz])

# ========================= 视线角 =========================
def compute_los_angles(x_target, x_missile, prev_theta_L, prev_phi_L, dt):
    Rx = x_target[0] - x_missile[3]
    Ry = x_target[1] - x_missile[4]
    Rz = x_target[2] - x_missile[5]
    R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    theta_L = np.arcsin(Ry / R)
    phi_L = np.arctan2(Rz, Rx)
    if prev_theta_L is None:
        theta_L_dot = 0.0
    else:
        theta_L_dot = (theta_L - prev_theta_L) / dt
    if prev_phi_L is None:
        phi_L_dot = 0.0
    else:
        dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
        phi_L_dot = dphi / dt
    return theta_L, phi_L, theta_L_dot, phi_L_dot

# ========================= 飞机速度向量 =========================
def compute_velocity_vector_from_target(x):
    V, theta, psi = x[3], x[4], x[5]
    vx = V * np.cos(theta) * np.cos(psi)
    vy = V * np.sin(theta)
    vz = -V * np.cos(theta) * np.sin(psi)
    return np.array([vx, vy, vz])

def compute_velocity_vector(x):
    V, theta, psi = x[0], x[1], x[2]
    vx = V * np.cos(theta) * np.cos(psi)
    vy = V * np.sin(theta)
    vz = -V * np.cos(theta) * np.sin(psi)
    return np.array([vx, vy, vz])

# ========================= 锁定判据 =========================
def check_seeker_lock(x_missile, x_target, V_missile_vec, V_target_vec, t_now,
                      D_max, Angle_IR, omega_max, T_max):
    R_vec = np.array(x_target[0:3]) - np.array(x_missile[3:6])
    R_mag = np.linalg.norm(R_vec)
    Flag_D = R_mag <= D_max
    cos_Angle = np.dot(R_vec, V_missile_vec) / (np.linalg.norm(R_vec) * np.linalg.norm(V_missile_vec))
    cos_Angle = np.clip(cos_Angle, -1.0, 1.0)
    Angle = np.arccos(cos_Angle)
    Flag_A = Angle <= Angle_IR
    delta_V = V_missile_vec - V_target_vec
    omega_R = np.linalg.norm(delta_V) * np.sin(Angle) / np.linalg.norm(R_vec)
    Flag_omega = omega_R <= omega_max
    Flag_T = t_now <= T_max
    IR_seeker_lock = Flag_D and Flag_A and Flag_omega and Flag_T
    return IR_seeker_lock, Flag_D, Flag_A, Flag_omega, Flag_T

# ========================= 飞机控制输入 =========================
def generate_control_input(x0, n_ty_max, mode):
    if mode == 1:
        return lambda x: (np.sin(x[4]), n_ty_max, -np.arccos(np.cos(x[4])/n_ty_max))
    elif mode == 2:
        return lambda x: (np.sin(x[4]), n_ty_max, np.arccos(np.cos(x[4])/n_ty_max))
    elif mode == 3:
        return lambda x: (np.sin(x[4]), (np.cos(x[4]) + n_ty_max)/np.cos(0), 0)
    elif mode == 4:
        return lambda x: (np.sin(x[4]), (np.cos(x[4]) - n_ty_max)/np.cos(0), 0)
    else:
        raise ValueError('Invalid maneuver mode')

# ========================= 脱靶评估 =========================
def evaluate_miss(t, Y, Xt, R_kill):
    R_all = []
    for i in range(len(t)):
        xt, yt, zt = Xt[i][:3]
        xm, ym, zm = Y[i][3:6]
        R = np.linalg.norm([xt - xm, yt - ym, zt - zm])
        R_all.append(R)
    R_all = np.array(R_all)
    miss_distance = np.min(R_all)
    idx_min = np.argmin(R_all)
    t_minR = t[idx_min]
    is_hit = (miss_distance <= R_kill)
    print(f">>> {'命中' if is_hit else '未命中'}，脱靶量为：{miss_distance:.2f} m")
    return miss_distance, is_hit, R_all, t_minR, idx_min

# ================= 引信起爆判断 =================
def check_fuze_trigger(x_missile, x_target, R_cons, V_fuse):
    pos_m = x_missile[3:6]
    pos_t = x_target[:3]
    R_mt = np.linalg.norm(pos_m - pos_t)
    Cons_D = R_mt <= R_cons

    V_m = compute_velocity_vector(x_missile)
    V_t = compute_velocity_vector_from_target(x_target)
    V_rel = np.linalg.norm(V_m - V_t)
    Cons_V = V_rel >= V_fuse

    ACT_F = Cons_D and Cons_V
    return ACT_F, R_mt, V_rel

# ========================= 红外诱饵弹类 =========================
class Flare:
    def __init__(self, position, velocity, release_time, m0=0.5, m_dot=0.01,
                 rho=1.225, c=0.5, s=0.01, I_max=5000, t1=1.5, t2=3.5, t3=5.0):
        self.pos = np.array(position, dtype=float)   #释放位置（飞机当前位置）
        self.vel = np.array(velocity, dtype=float)   #初速度（飞机速度 + 后下方扰动）
        self.release_time = release_time   #释放时刻
        self.m0 = m0   #初始质量
        self.m_dot = m_dot   #单位时间内质量损失速率（质量随时间线性下降）
        self.rho = rho   #空气密度
        self.c = c   #阻力系数
        self.s = s   #迎风面积
        self.I_max = I_max   #最大红外辐射强度
        self.t1 = t1   #
        self.t2 = t2   #
        self.t3 = t3   #
        self.history = [self.pos.copy()]   #存储轨迹

    def update(self, t, dt):          #更新位置与速度
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return
        v_mag = np.linalg.norm(self.vel)
        m_t = max(self.m0 - self.m_dot * t_rel, 0.01)
        f = 0.5 * self.rho * v_mag**2 * self.c * self.s
        a = f / m_t
        a_vec = -a * (self.vel / v_mag)
        self.pos += self.vel * dt + 0.5 * a_vec * dt**2
        self.vel += a_vec * dt
        self.history.append(self.pos.copy())

    def get_intensity(self, t): #根据当前时间 t 和释放时刻 t_release，计算诱饵弹的红外辐射强度
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return 0.0
        if t_rel <= self.t1:
            return self.I_max * t_rel / self.t1
        elif t_rel <= self.t2:
            return self.I_max
        elif t_rel <= self.t3:
            return (self.t3 - t_rel) * self.I_max / (self.t3 - self.t2)
        else:
            return 0.0

class FlareManager:
    def __init__(self, flare_per_group=6, interval=0.1, release_speed=50):
        self.flare_per_group = flare_per_group   #每一组包含6
        self.interval = interval   #每隔0.1秒释放一次
        self.release_speed = release_speed   # 相对释放速度
        self.flares = []   #  当前存在的所有诱饵弹
        self.schedule = []   #计划投放清单（等待释放）

    def release_flare_group(self, t_start):
        for i in range(self.flare_per_group):
            t_i = t_start + i * self.interval
            self.schedule.append((t_i, None, None))

    def update(self, t, dt, aircraft_state):
        # === 判断当前有哪些诱饵弹需要释放 ===
        newly_released = [(ti, i) for i, (ti, _, _) in enumerate(self.schedule) if abs(t - ti) < dt / 2]

        for _, idx in newly_released:
            ti, _, _ = self.schedule[idx]

            # --- 获取飞机当前位置与姿态 ---
            pos_now = aircraft_state[:3].copy()
            vel_now = compute_velocity_vector_from_target(aircraft_state)
            theta = aircraft_state[4]  # 俯仰角
            psi = aircraft_state[5]  # 偏航角

            # --- 机体坐标系下的“后下方”方向向量 ---
            v_b = np.array([-1.0, -1.0, 0.0])  # 后+下，在你定义的机体坐标：x前，y上，z右
            v_b /= np.linalg.norm(v_b)

            # --- 构造方向余弦矩阵，将v_b从机体系变换到惯性系（北-天-东） ---
            R_pitch = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            R_yaw = np.array([
                [np.cos(psi), 0, np.sin(psi)],
                [0, 1, 0],
                [-np.sin(psi), 0, np.cos(psi)]
            ])
            R_bn = R_yaw @ R_pitch  # 总旋转矩阵
            back_down = R_bn @ v_b  # 释放方向在惯性系下的表达

            # --- 合成诱饵弹初始速度 ---
            v_rel = self.release_speed * back_down
            v0 = vel_now + v_rel

            # --- 创建诱饵弹对象并加入当前诱饵弹列表 ---
            self.flares.append(Flare(pos_now, v0, ti))

        # === 清理已释放完的计划条目 ===
        self.schedule = [(ti, pos, vel) for (ti, pos, vel) in self.schedule if ti > t]

        # === 更新所有在空中的诱饵弹轨迹 ===
        for flare in self.flares:
            flare.update(t, dt)

# ========================= 激光定向干扰 =========================
def laser_interference(R_vec, R_rel, V_missile_vec):
    """
    激光定向干扰计算
    :return: 饱和光斑的半径 (mm), 光斑面积 (mm^2), 目标在探测器上的面积 (mm^2), 是否成功干扰导弹 (bool)
    """
    # 参数设定
    H = 10  # 目标高度 (m)
    W = 5   # 目标宽度 (m)
    R = R_rel / 1000   # 目标与热像仪的距离 (km)
    d0 = 12   # 探测器像元尺寸 (mm)
    d = 27  # mm
    f = 57  # 焦距 (mm)
    P0 = 0.1  # 激光辐照能量 (W)
    theta = 1  # 束散角 (mrad)
    D0 = 100  # 光学镜头通光口径 (mm)
    a = 43.6  # 衍射光阑半径 (mm)
    tau0 = 0.8  #大气透过率
    tau1 = 0.8  #光学镜头透过率
    Ith = 10**-6  # 刚饱和时的激光能量 (W/m^2)
    lambda_ = 10.6  # 激光波长 (um)

    #计算激光干扰入射角
    alpha = np.arccos(np.dot(R_vec, V_missile_vec) / (np.linalg.norm(R_vec) * np.linalg.norm(V_missile_vec)))  # 激光干扰入射角 (rad)
    if alpha > np.pi / 2:
        print("激光入射角度过大")
        return False
    else:
        # 计算目标成像尺寸
        nH = (H * f) / (R * 1000 * d0)  #目标像在探测器纵向上所占像素数
        nW = (W * f) / (R * 1000 * d0)  #目标像在探测器横向上所占像素数

        # 计算激光打在探测器表面焦面功率密度
        I0 = (4 * tau0 * tau1 * P0 * (D0 / d0)**2 * np.cos(alpha)) / (np.pi * R**2 * theta**2)
        #print(f"激光打在探测器表面焦面功率密度: {I0} W/m^2")

        # 计算饱和光斑的半径
        if I0 < Ith:
            r_laser = 0  # 光斑较小，对干扰效果影响不大
        else:
            r_laser = (lambda_ * 0.001 * d) / (2 * np.pi * a) * np.cbrt(4 * I0 / Ith)
        #print(f"饱和光斑的半径: {r_laser} mm")

        # 计算光斑面积和目标在探测器上的面积
        spot_area = np.pi * r_laser**2  #光斑面积
        target_area = np.pi * d0**2 * nH * nW  #目标在探测器上的面积

        # print(f"光斑面积: {spot_area} mm^2")
        # print(f"目标在探测器上的面积: {target_area} mm^2")

        # 判断是否成功干扰导弹
        if spot_area >= target_area:
            print("成功干扰")
            return True  # 成功干扰导弹导引头
        else:
            print("失败")
            return False  # 没有成功干扰导弹导引头

def run_one_step(dt):
    global t_now, x_target_now, x_missile_now, prev_theta_L, prev_phi_L, prev_R_rel , lost_and_separating_duration, t, Xt, Y
    interference_success = False

    def rk4_aircraft(x):
        k1 = aircraft_dynamics(t_now, x, control_input, g)
        k2 = aircraft_dynamics(t_now + dt/2, x + dt/2*k1, control_input, g)
        k3 = aircraft_dynamics(t_now + dt/2, x + dt/2*k2, control_input, g)
        k4 = aircraft_dynamics(t_now + dt, x + dt*k3, control_input, g)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    x_target_next = rk4_aircraft(x_target_now)

    if abs(t_now - 50.0) < dt / 2 or abs(t_now - 100.0) < dt / 2 or abs(t_now - 150.0) < dt / 2:
        flare_manager.release_flare_group(t_now)
    flare_manager.update(t_now, dt, x_target_now)

    R_vec = x_target_now[0:3] - x_missile_now[3:6]
    R_rel = np.linalg.norm(R_vec)

    if 0 < t_now < 30:
        interference_success = laser_interference(R_vec, R_rel, compute_velocity_vector(x_missile_now))

    lock_aircraft, *_ = check_seeker_lock(x_missile_now, x_target_now,
                                          compute_velocity_vector(x_missile_now),
                                          compute_velocity_vector_from_target(x_target_now),
                                          t_now, D_max, Angle_IR, omega_max, T_max)

    if lock_aircraft and not interference_success:
        beta_p = compute_relative_beta(x_target_now, x_missile_now)
        I_p = infrared_intensity_model(beta_p)
        pos_p = x_target_now[:3]
    else:
        I_p = 0.0
        pos_p = np.zeros(3)

    visible_flares = []
    for flare in flare_manager.flares:
        if flare.get_intensity(t_now) <= 1e-3:
            continue
        lock_flare, *_ = check_seeker_lock(x_missile_now,
                                           np.concatenate((flare.pos, [0, 0, 0])),
                                           compute_velocity_vector(x_missile_now),
                                           flare.vel,
                                           t_now, D_max, Angle_IR, omega_max, T_max)
        if lock_flare:
            visible_flares.append(flare)

    I_total = I_p
    numerator = I_p * pos_p
    for flare in visible_flares:
        I_k = flare.get_intensity(t_now)
        numerator += I_k * flare.pos
        I_total += I_k

    if I_total > 0:
        target_pos_equiv = numerator / I_total
        Rx = target_pos_equiv[0] - x_missile_now[3]
        Ry = target_pos_equiv[1] - x_missile_now[4]
        Rz = target_pos_equiv[2] - x_missile_now[5]
        R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
        phi_L = np.arctan2(Rz, Rx)
        theta_L_dot = 0.0 if prev_theta_L is None else (theta_L - prev_theta_L) / dt
        dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L)) if prev_phi_L else 0.0
        phi_L_dot = dphi / dt
    else:
        theta_L = prev_theta_L if prev_theta_L is not None else 0.0
        phi_L = prev_phi_L if prev_phi_L is not None else 0.0
        theta_L_dot = last_valid_theta_dot
        phi_L_dot = last_valid_phi_dot

    def rk4_missile(x):
        k1 = missile_dynamics_given_rate(x, theta_L_dot, phi_L_dot, N, ny_max, nz_max)
        k2 = missile_dynamics_given_rate(x + dt/2*k1, theta_L_dot, phi_L_dot, N, ny_max, nz_max)
        k3 = missile_dynamics_given_rate(x + dt/2*k2, theta_L_dot, phi_L_dot, N, ny_max, nz_max)
        k4 = missile_dynamics_given_rate(x + dt*k3, theta_L_dot, phi_L_dot, N, ny_max, nz_max)
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    x_missile_next = rk4_missile(x_missile_now)

    # 状态更新
    t_now += dt
    t.append(t_now)
    Xt.append(x_target_next)
    Y.append(x_missile_next)
    x_target_now = x_target_next
    x_missile_now = x_missile_next
    prev_theta_L = theta_L
    prev_phi_L = phi_L

    ACT_F, R_mt_now, V_rel_now = check_fuze_trigger(x_missile_now, x_target_now, R_kill, 150)
    if ACT_F:
        print(f">>> 引信引爆！R = {R_mt_now:.2f} m, V_rel = {V_rel_now:.2f} m/s")
        return True  # 停止仿真
    else:
        # 飞机逃脱判据******************************************************************
        R_vec = x_target_now[0:3] - x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        lock_aircraft, *_ = check_seeker_lock(
            x_missile_now, x_target_now,
            compute_velocity_vector(x_missile_now),
            compute_velocity_vector_from_target(x_target_now),
            t_now, D_max, Angle_IR, omega_max, T_max
        )
        if (not lock_aircraft or interference_success) and \
                (prev_R_rel is not None and R_rel > prev_R_rel):
            lost_and_separating_duration += dt
        else:
            lost_and_separating_duration = 0.0

        prev_R_rel = R_rel

        if lost_and_separating_duration >= 1.0:
            print(f">>> 持续丢失目标 + 距离增大超过1秒，仿真提前终止！m")
            done = True
            Xt = np.array(Xt)
            Y = np.array(Y)
            t = np.array(t)
            miss_distance, is_hit, R_all, t_minR, idx_min = evaluate_miss(t, Y, Xt,
                                                                                         R_kill)
            return True

# ========================= 初始化与主仿真循环 =========================
g = 9.81
y0 = np.array([800, 0, 0, 0, 5000, 0])   # 导弹状态 [V, theta, psi_c, x, y, z]
x0_target = np.array([9000, 1000, 1000, 300, 0, 0])   # 飞机状态[x, y, z, V, theta, psi]
N = 3       # 导引律参数
ny_max = 30     # 导弹最大过载
nz_max = 30     # 导弹最大过载
R_kill = 12     # 导弹杀伤半径
dt_outer = 0.05  # 大步长
dt_inner = 0.01  # 小步长
N_inner = int(dt_outer / dt_inner)
R_switch = 500        # 步长切换的相对距离阈值
delta = 50            # 滞回带，避免频繁切换
t_end = 60            # 仿真结束时间
target_mode = 1       # 1=左转，2=右转，3=爬升，4=俯冲
n_tz_max = 3          # 飞机过载
control_input = generate_control_input(x0_target, n_tz_max, target_mode)

D_max = 13000         # 导引头最大距离
Angle_IR = np.deg2rad(90)     # 导引头角度
omega_max = 12               #0.2秒内完成180度偏转，太夸张了吧！！！！！# 导引头角速度
T_max = 30                   # 导引头工作时间

t_now = 0
x_target_now = x0_target.copy()
x_missile_now = y0.copy()
t = [t_now]
Xt = [x_target_now]
Y = [x_missile_now]
seeker_log = []
prev_theta_L = None
prev_phi_L = None
last_valid_theta_dot = 0
last_valid_phi_dot = 0
prev_R_rel = None
lost_and_separating_duration = 0.0

flare_manager = FlareManager()

# print(N_inner)
while t_now <= t_end:
    # print(t_now)
    R_vec = x_target_now[0:3] - x_missile_now[3:6]
    R_rel = np.linalg.norm(R_vec)

    if R_rel < R_switch:
        for _ in range(N_inner):
            done = run_one_step(dt_inner)
            if done:
                break
        if done:
            break
    else:
        done = run_one_step(dt_outer)
        if done:
            break





Xt = np.array(Xt)
Y = np.array(Y)
t = np.array(t)
miss_distance, is_hit, R_all, t_minR, idx_min = evaluate_miss(t, Y, Xt, R_kill)

# ========================= 可视化 =========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Y[:,3], Y[:,5], Y[:,4], 'b-', label='导弹轨迹')
ax.plot(Xt[:,0], Xt[:,2], Xt[:,1], 'r--', label='目标轨迹')
ax.scatter(Y[0,3], Y[0,5], Y[0,4], color='g', label='导弹起点')
ax.scatter(Y[idx_min,3], Y[idx_min,5], Y[idx_min,4], color='m', label='最近点')

for i, flare in enumerate(flare_manager.flares):
    traj = np.array(flare.history)
    if traj.shape[0] > 0:
        ax.plot(traj[:,0], traj[:,2], traj[:,1], color='orange', linewidth=1,
                label='红外诱饵弹' if i == 0 else "")

ax.set_xlabel('X / m')
ax.set_ylabel('Z / m')
ax.set_zlabel('Y / m')
ax.invert_yaxis()
ax.legend()
ax.set_title('导引头模型 + 红外诱饵弹建模仿真')
ax.grid()
ax.view_init(elev=20, azim=-150)
plt.show()
