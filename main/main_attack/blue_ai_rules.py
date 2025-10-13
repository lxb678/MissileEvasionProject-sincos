import numpy as np
from Interference_code.main.main_attack.fire_control_rules_all import can_launch_missile_with_pk

# ==============================================================================
# --- 蓝方 AI 逻辑：追踪法  ---
# ==============================================================================
# def get_blue_ai_action(blue_obs: dict, env) -> list:
#     """
#     一个基于规则的AI，使用 [nx, nz, phi_cmd] 来控制蓝方飞机
#     追踪并攻击红方飞机。 phi_cmd 是目标滚转角 (rad)。
#     """
#     blue_ac = env.blue_aircraft
#     red_ac = env.red_aircraft
#
#     los_nue = red_ac.pos - blue_ac.pos
#     distance = np.linalg.norm(los_nue)
#     los_ned = np.array([los_nue[0], los_nue[2], -los_nue[1]])
#
#     def euler_to_quaternion(phi, theta, psi):
#         cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
#         cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
#         cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
#         q0 = cr * cp * cy + sr * sp * sy
#         q1 = sr * cp * cy - cr * sp * sy
#         q2 = cr * sp * cy + sr * cp * sy
#         q3 = cr * cp * sy - sr * sp * cy
#         norm = np.linalg.norm([q0, q1, q2, q3])
#         return np.array([q0, q1, q2, q3]) / norm if norm > 1e-9 else np.array([1, 0, 0, 0])
#
#     def quaternion_to_rotation_matrix(q):
#         q0, q1, q2, q3 = q
#         return np.array([
#             [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
#             [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
#             [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])
#
#     theta, psi, phi = blue_ac.attitude_rad
#     q_blue = euler_to_quaternion(phi, theta, psi)
#     R_frd_to_ned = quaternion_to_rotation_matrix(q_blue)
#     R_ned_to_frd = R_frd_to_ned.T
#     los_body = R_ned_to_frd @ los_ned
#     pitch_error_rad = np.arctan2(-los_body[2], los_body[0])
#     yaw_error_rad = np.arctan2(los_body[1], los_body[0])
#     total_error_rad = np.sqrt(pitch_error_rad ** 2 + yaw_error_rad ** 2)
#
#     Kp_bank_angle = 100.0
#     MAX_BANK_ANGLE_RAD = np.deg2rad(180.0)
#     phi_cmd_rad = Kp_bank_angle * yaw_error_rad
#     phi_cmd = np.clip(phi_cmd_rad, -MAX_BANK_ANGLE_RAD, MAX_BANK_ANGLE_RAD)
#
#     NZ_COMMAND_HIGH = 8.0
#     NZ_COMMAND_LOW = 1.0
#     nz_cmd = NZ_COMMAND_HIGH if total_error_rad > np.deg2rad(2.0) else NZ_COMMAND_LOW
#     nz_cmd = np.clip(nz_cmd, -5.0, 9.0)
#
#     if distance > 4000:
#         nx_cmd = 1.0
#     elif distance < 1500:
#         nx_cmd = 0.4
#     else:
#         nx_cmd = 0.85
#     if distance < 500 and np.dot(blue_ac.get_velocity_vector(), los_nue) > 0: nx_cmd = -0.5
#
#     fire_missile = 0.0
#     in_range = 800 < distance < 6000
#     is_aligned = total_error_rad < np.deg2rad(4.0)
#     if in_range and is_aligned: fire_missile = 0.0  # 根据您的代码，这里暂时不开火
#
#     release_flare = 0.0
#
#     return [nx_cmd, nz_cmd, phi_cmd, 0.0, release_flare, fire_missile]


# AI 状态存储，现在包含积分项
blue_ai_state = {
    'last_yaw_error_rad': 0.0,
    'integral_yaw_error_rad': 0.0  # 新增：用于累积偏航误差
}


def get_blue_ai_action(blue_obs: dict, env) -> list:
    """
    一个基于规则的AI，使用 [nx, nz, phi_cmd] 来控制蓝方飞机。
    追踪并攻击红方飞机。
    使用 PID 控制器来计算目标滚转角 (phi_cmd)，以实现更精确、无稳态误差的追踪。
    """
    blue_ac = env.blue_aircraft
    red_ac = env.red_aircraft
    dt = 0.2  # 假设每步时间间隔为0.02秒 (50Hz)

    # --- 1, 2, 3: 几何计算与误差获取 (不变) ---
    los_nue = red_ac.pos - blue_ac.pos
    distance = np.linalg.norm(los_nue)
    los_ned = np.array([los_nue[0], los_nue[2], -los_nue[1]])

    def euler_to_quaternion(phi, theta, psi):  # ... (内容不变)
        cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5);
        cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5);
        cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
        q0 = cr * cp * cy + sr * sp * sy;
        q1 = sr * cp * cy - cr * sp * sy;
        q2 = cr * sp * cy + sr * cp * sy;
        q3 = cr * cp * sy - sr * sp * cy
        norm = np.linalg.norm([q0, q1, q2, q3]);
        return np.array([q0, q1, q2, q3]) / norm if norm > 1e-9 else np.array([1, 0, 0, 0])

    def quaternion_to_rotation_matrix(q):  # ... (内容不变)
        q0, q1, q2, q3 = q
        return np.array([[1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                         [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
                         [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])

    theta, psi, phi = blue_ac.attitude_rad
    q_blue = euler_to_quaternion(phi, theta, psi)
    R_frd_to_ned = quaternion_to_rotation_matrix(q_blue)
    R_ned_to_frd = R_frd_to_ned.T
    los_body = R_ned_to_frd @ los_ned
    pitch_error_rad = np.arctan2(-los_body[2], los_body[0])
    yaw_error_rad = np.arctan2(los_body[1], los_body[0])
    total_error_rad = np.sqrt(pitch_error_rad ** 2 + yaw_error_rad ** 2)

    # --- 4. 生成 G-Command 指令 [nx, nz, phi_cmd] (核心修改部分) ---

    # ------------------- START OF MODIFICATION -------------------

    # a) 目标滚转角指令 (phi_cmd) - 从 PD 控制器升级为 PID 控制器
    # -----------------------------------------------------------
    # PID 控制器参数 (需要仔细调整)
    Kp_roll = 5.0  # 比例增益 (Proportional)
    Ki_roll = 0.5  # 积分增益 (Integral) - 消除稳态误差
    Kd_roll = 2.5  # 微分增益 (Derivative) - 抑制震荡

    MAX_BANK_ANGLE_RAD = np.deg2rad(120.0)

    # --- 比例项 (P) ---
    p_term = Kp_roll * yaw_error_rad

    # --- 积分项 (I) with Anti-Windup ---
    # 仅在误差较小（接近目标）时才激活积分，防止积分饱和
    INTEGRAL_ACTIVE_ZONE_RAD = np.deg2rad(20.0)
    # 设定一个积分累积值的上限，进一步防止饱和
    MAX_INTEGRAL = 2.0

    if abs(yaw_error_rad) < INTEGRAL_ACTIVE_ZONE_RAD:
        blue_ai_state['integral_yaw_error_rad'] += yaw_error_rad * dt
    else:
        # 当误差较大时，重置积分项，这是关键的 Anti-Windup 步骤
        blue_ai_state['integral_yaw_error_rad'] = 0.0

    # 对积分累积值进行钳位
    blue_ai_state['integral_yaw_error_rad'] = np.clip(
        blue_ai_state['integral_yaw_error_rad'], -MAX_INTEGRAL, MAX_INTEGRAL
    )

    i_term = Ki_roll * blue_ai_state['integral_yaw_error_rad']

    # --- 微分项 (D) ---
    yaw_error_rate = (yaw_error_rad - blue_ai_state['last_yaw_error_rad']) / dt
    d_term = Kd_roll * yaw_error_rate

    # 更新状态，为下一帧做准备
    blue_ai_state['last_yaw_error_rad'] = yaw_error_rad

    # --- PID 输出 ---
    phi_cmd_rad = p_term + i_term + d_term
    phi_cmd = np.clip(phi_cmd_rad, -MAX_BANK_ANGLE_RAD, MAX_BANK_ANGLE_RAD)

    # b) 法向过载指令 (nz_cmd) (与PD版本相同，保持平滑控制)
    # Kp_pitch = 8.0
    # nz_cmd = 1.0 + Kp_pitch * abs(pitch_error_rad)
    # nz_cmd = np.clip(nz_cmd, 1.0, 9.0)
    # b) 法向过载指令 (nz_cmd) - 基于总误差，与滚转协同
    # ----------------------------------------------------------
    # 这是关键的改进：拉杆的强度不应只看垂直误差，
    # 而应看机头需要转过的总角度(total_error_rad)。
    # 这样，当飞机大角度滚转以进行水平转弯时，它也会大力拉杆来收紧转弯半径。
    Kp_total_g = 9.0  # 将总误差(rad)映射到G值的增益
    # 基础G值为1，然后根据总误差增加G值
    nz_cmd = 1.0 + Kp_total_g * total_error_rad
    nz_cmd = np.clip(nz_cmd, 1.0, 9.0)

    # c) 切向过载指令 (nx_cmd) (与PD版本相同，保持平滑控制)
    dist_points = [500, 1500, 4000, 8000]
    # nx_points = [-0.5, 0.4, 0.85, 1.0]
    nx_points = [0.0, 0.4, 0.85, 1.0]
    nx_cmd = np.interp(distance, dist_points, nx_points)

    # -------------------- END OF MODIFICATION --------------------

    # --- 5, 6, 7: 武器、防御和返回 (不变) ---
    fire_missile = 0.0
    # in_range = 800 < distance < 6000
    # is_aligned = total_error_rad < np.deg2rad(4.0)
    # if in_range and is_aligned:
    #     fire_missile = 0.0

    # # 调用新的火控决策逻辑
    # fire_missile_decision = should_fire_missile(env)
    # fire_missile = 1.0 if fire_missile_decision else 0.0
    # 调用通用函数
    if can_launch_missile_with_pk(launcher_ac=env.blue_aircraft,
                                  target_ac=env.red_aircraft,
                                  current_sim_time=env.t_now):
        fire_missile = 1.0
        # fire_missile = 0.0
    else:
        fire_missile = 0.0

    release_flare = 0.0

    return [nx_cmd, nz_cmd, phi_cmd, 0.0, release_flare, fire_missile]