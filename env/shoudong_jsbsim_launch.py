# 文件名: shoudong_vs_rule_ai.py
# 描述: 人机对抗脚本。红方由人类键盘操作，蓝方由规则AI控制。

from Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim_pointmass import AirCombatEnv
from Interference_code.main_attack.fire_control_rules import should_fire_missile
import keyboard
import time
import numpy as np


# ========================= 蓝方规则AI函数 =========================
# def get_blue_ai_action(blue_obs: np.ndarray, env: AirCombatEnv) -> list:
#     """
#     一个简单的规则AI，用于控制蓝方飞机。
#     :param blue_obs: 蓝方智能体的观测向量 (10维)
#     :param env: 环境实例，用于获取一些额外信息
#     :return: 蓝方的动作向量 (6维)
#     """
#     # --- 解包观测向量 (obs已经归一化了) ---
#     # o_vel, o_alt, o_pitch, o_roll, o_missiles, o_flares,
#     # o_target_dist, o_target_bearing, o_target_elev, o_threat_dist
#
#     target_bearing = blue_obs[7]  # 目标相对方位角 (归一化, -1 到 1, 0表示正前方)
#     target_elev = blue_obs[8]  # 目标相对俯仰角 (归一化, -1 到 1, 0表示水平)
#     target_dist_norm = blue_obs[6]  # 目标距离 (归一化)
#
#     # --- 1. 机动控制 ---
#     # 油门保持高速
#     throttle = 0.9
#
#     # 俯仰控制：尝试让目标保持在水平线上
#     # 如果目标在上方 (target_elev > 0)，需要拉杆 (elevator < 0)
#     elevator = -target_elev * 2.0  # 乘以一个增益系数
#
#     # 滚转/偏航控制：尝试让目标保持在正前方
#     # 如果目标在右侧 (target_bearing > 0)，需要向右滚转 (aileron > 0)
#     aileron = target_bearing * 2.0
#
#     # 简单起见，我们主要用副翼转弯，方向舵辅助或不用
#     rudder = 0.0
#
#     # 限制控制输入在 [-1, 1] 范围内
#     elevator = np.clip(elevator, -1.0, 1.0)
#     aileron = np.clip(aileron, -1.0, 1.0)
#
#     # --- 2. 火控逻辑 ---
#     fire_missile = 0.0
#     # 将归一化的距离和角度转回物理值
#     target_dist_physical = target_dist_norm * env.max_disengagement_range
#     target_bearing_deg = target_bearing * 180
#
#     # 开火条件：
#     # 1. 目标在射程内 (例如: 小于 20 公里)
#     # 2. 目标在机头前方的一个小角度范围内 (例如: ±15度)
#     # 3. 还有剩余导弹
#     can_fire = (target_dist_physical < 20000) and \
#                (abs(target_bearing_deg) < 15) and \
#                (env.blue_aircraft.missile_ammo > 0)
#
#     if can_fire:
#         fire_missile = 1.0
#
#     # --- 3. 防御逻辑 (当前版本不使用) ---
#     release_flare = 0.0
#
#     # --- 4. 打包动作 ---
#     blue_action = [
#         throttle, elevator, aileron, rudder,
#         release_flare, fire_missile
#     ]
#     return blue_action


# # ========================= 蓝方规则AI函数 (仅平飞) =========================
# def get_blue_ai_action(blue_obs: np.ndarray, env: AirCombatEnv) -> list:
#     """
#     一个极简的规则AI，只让蓝方飞机保持平飞。
#     它不进行任何机动，也不发射武器。
#     :param blue_obs: 蓝方智能体的观测向量 (为了保持接口一致性，但未使用)
#     :param env: 环境实例 (未使用)
#     :return: 蓝方的动作向量 (6维)
#     """
#     # 油门保持在80%
#     throttle = 0.8
#
#     # 所有控制舵面都保持在中立位置
#     elevator = 0.0
#     aileron = 0.0
#     rudder = 0.0
#
#     # 不执行任何离散动作
#     release_flare = 0.0
#     fire_missile = 0.0
#
#     # 打包动作
#     blue_action = [
#         throttle, elevator, aileron, rudder,
#         release_flare, fire_missile
#     ]
#
#     return blue_action

# ==============================================================================
# --- 蓝方 AI 逻辑：追踪法 (G-Command Model) ---
# ==============================================================================

# def get_blue_ai_action(blue_obs: dict, env) -> list:
#     """
#     一个基于规则的AI，使用G-Command模型 [nx, nz, p_cmd] 来控制蓝方飞机
#     追踪并攻击红方飞机。
#
#     Args:
#         blue_obs (dict): 蓝方智能体的观测值 (本次AI中未使用)。
#         env: 主环境对象，用于访问飞机状态。
#
#     Returns:
#         list: 蓝方飞机的动作指令。根据您的主循环，这可能是一个6元素的向量，
#               我们将填充前三个元素为 [nx, nz, p_cmd]。
#     """
#     blue_ac = env.blue_aircraft
#     red_ac = env.red_aircraft
#
#     # --- 1. 几何计算：获取从蓝方指向红方的视线(LOS)向量 (这部分不变) ---
#     los_nue = red_ac.pos - blue_ac.pos
#     distance = np.linalg.norm(los_nue)
#     los_ned = np.array([los_nue[0], los_nue[2], -los_nue[1]])  # NUE -> NED
#
#     # --- 2. 坐标系转换：将LOS向量从世界系转换到蓝方机体系(FRD) (这部分不变) ---
#     # (为了简洁，这里省略了内部辅助函数 euler_to_quaternion 等，假设它们可用)
#     # (在实际使用时，您可以将这些函数放在文件顶部或作为辅助类)
#     def euler_to_quaternion(phi, theta, psi):
#         cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
#         cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
#         cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
#         q0 = cr * cp * cy + sr * sp * sy
#         q1 = sr * cp * cy - cr * sp * sy
#         q2 = cr * sp * cy + sr * cp * sy
#         q3 = cr * cp * sy - sr * sp * cy
#         norm = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
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
#
#     # --- 3. 计算角度误差 (这部分不变) ---
#     # 目标在机体垂直平面上的角度 (决定了需要"拉杆"的程度)
#     pitch_error_rad = np.arctan2(-los_body[2], los_body[0])
#     # 目标在机体水平平面上的角度 (决定了需要"滚转"的方向和程度)
#     yaw_error_rad = np.arctan2(los_body[1], los_body[0])
#
#     # --- 4. 生成 G-Command 指令 [nx, nz, p_cmd] ---
#
#     # a) 滚转速率指令 (p_cmd)
#     # 目标：通过滚转消除水平误差 (yaw_error)，将目标置于机体 x-z 平面内（正上方或正下方）。
#     # 如果目标在右边 (yaw_error > 0)，需要向右滚转 (p_cmd > 0)。
#     # 使用P控制器，指令与误差成正比。
#     Kp_roll = 100.0  # 增益系数，将弧度误差映射到度/秒的滚转速率
#     p_cmd_deg_s = Kp_roll * yaw_error_rad
#     # 限制滚转速率在飞机能力范围内
#     p_cmd = np.clip(p_cmd_deg_s, -240.0, 240.0)
#
#     # b) 法向过载指令 (nz_cmd)
#     # 目标：一旦滚转到位，就拉杆（施加正G）将机头抬向目标。
#     # 只要存在指向误差，就应该施加G力进行机动。
#     total_error_rad = np.sqrt(pitch_error_rad ** 2 + yaw_error_rad ** 2)
#     NZ_COMMAND_HIGH = 7.0  # 高G机动
#     NZ_COMMAND_LOW = 1.0  # 保持平飞
#
#     # 如果机头没有对准目标 (误差大于2度)，则执行高G机动
#     if total_error_rad > np.deg2rad(2.0):
#         nz_cmd = NZ_COMMAND_HIGH
#     else:
#         # 如果已经对准，保持1G平飞
#         nz_cmd = NZ_COMMAND_LOW
#     # 限制法向过载在飞机能力范围内
#     nz_cmd = np.clip(nz_cmd, -5.0, 9.0)
#
#     # c) 切向过载指令 (nx_cmd)
#     # 目标：控制能量/速度。远了加速追，近了减速防冲过。
#     # nx_cmd 在 [0, 1] 范围内映射到油门，在 [-1, 0) 映射到减速板。
#     if distance > 4000:
#         nx_cmd = 1.0  # 全力追击 (100% 可用推力)
#     elif distance < 1500:
#         nx_cmd = 0.4  # 减小推力，准备近距离缠斗
#     else:
#         nx_cmd = 0.85  # 保持能量接敌
#     # 如果非常近，有冲过头的风险，可以开启减速板
#     if distance < 500 and np.dot(blue_ac.get_velocity_vector(), los_nue) > 0:
#         nx_cmd = -0.5  # 开启减速板
#
#     # --- 5. 武器发射逻辑 (这部分不变) ---
#     fire_missile = 0.0
#     in_range = 800 < distance < 6000
#     is_aligned = total_error_rad < np.deg2rad(4.0)  # 4度以内算对准
#
#     if in_range and is_aligned:
#         fire_missile = 1.0
#
#     # --- 6. 防御措施 (Flares) - 待实现 ---
#     release_flare = 0.0
#
#     # --- 7. 组装并返回最终动作 ---
#     # 您的 update 函数需要 [nx, nz, p_cmd]。
#     # 您的主循环可能需要一个不同长度的动作向量。
#     # 我们假设环境的 step 函数会智能地处理，AI应该输出它最自然的控制指令。
#     # 这里我们返回一个6元素的列表，以匹配您主循环中的 red_action 格式。
#     # 第4个元素(rudder)在这种控制模型下通常为0。
#     return [
#         nx_cmd,
#         nz_cmd,
#         p_cmd,
#         0.0,  # Rudder/Unused
#         release_flare,
#         fire_missile
#     ]


# # ==============================================================================
# # --- 蓝方 AI 逻辑：追踪法 (目标滚转角模型) ---
# # ==============================================================================
#
# def get_blue_ai_action(blue_obs: dict, env) -> list:
#     """
#     一个基于规则的AI，使用 [nx, nz, phi_cmd] 来控制蓝方飞机
#     追踪并攻击红方飞机。
#     phi_cmd 是目标滚转角 (rad)。
#     """
#     blue_ac = env.blue_aircraft
#     red_ac = env.red_aircraft
#
#     # --- 1, 2, 3: 几何计算与误差获取 (这部分完全不变) ---
#     los_nue = red_ac.pos - blue_ac.pos
#     distance = np.linalg.norm(los_nue)
#     los_ned = np.array([los_nue[0], los_nue[2], -los_nue[1]])
#
#     def euler_to_quaternion(phi, theta, psi):  # ... (内容不变)
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
#     def quaternion_to_rotation_matrix(q):  # ... (内容不变)
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
#     # --- 4. 生成 G-Command 指令 [nx, nz, phi_cmd] (核心修改部分) ---
#
#     # ------------------- START OF MODIFICATION -------------------
#     # a) 目标滚转角指令 (phi_cmd)
#     # 目标：计算一个理想的倾斜角 (bank angle) 来进行 "Bank-to-Turn" 机动。
#     # 倾斜角的大小与水平误差 (yaw_error) 成正比。
#
#     Kp_bank_angle = 10.0  # 增益：将水平误差(rad)映射到目标倾斜角(rad)   #100.0对应0.02太大了，让蓝方很猛，建议小一点
#     # 值越大，AI转弯越“猛”，倾角越大。
#     MAX_BANK_ANGLE_RAD = np.deg2rad(180.0)  # 限制最大倾斜角为180度
#
#     # 计算目标滚转角
#     phi_cmd_rad = Kp_bank_angle * yaw_error_rad
#
#     # 限制目标滚转角
#     phi_cmd = np.clip(phi_cmd_rad, -MAX_BANK_ANGLE_RAD, MAX_BANK_ANGLE_RAD)
#
#     # -------------------- END OF MODIFICATION --------------------
#
#     # b) 法向过载指令 (nz_cmd) (逻辑不变)
#     NZ_COMMAND_HIGH = 9.0
#     NZ_COMMAND_LOW = 1.0
#     nz_cmd = NZ_COMMAND_HIGH if total_error_rad > np.deg2rad(2.0) else NZ_COMMAND_LOW
#     nz_cmd = np.clip(nz_cmd, -5.0, 9.0)
#
#     # c) 切向过载指令 (nx_cmd) (逻辑不变)
#     if distance > 4000:
#         nx_cmd = 1.0
#     elif distance < 1500:
#         nx_cmd = 0.4
#     else:
#         nx_cmd = 0.85
#     if distance < 500 and np.dot(blue_ac.get_velocity_vector(), los_nue) > 0:
#         nx_cmd = -0.5
#
#     # --- 5, 6, 7: 武器、防御和返回 (逻辑不变) ---
#     fire_missile = 0.0
#     in_range = 800 < distance < 6000
#     is_aligned = total_error_rad < np.deg2rad(4.0)
#     if in_range and is_aligned:
#         fire_missile = 0.0
#
#     release_flare = 0.0
#
#     return [
#         nx_cmd,
#         nz_cmd,
#         phi_cmd,  # 现在返回的是目标滚转角 (rad)
#         0.0,
#         release_flare,
#         fire_missile
#     ]

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
    nx_points = [-0.5, 0.4, 0.85, 1.0]
    nx_cmd = np.interp(distance, dist_points, nx_points)

    # -------------------- END OF MODIFICATION --------------------

    # --- 5, 6, 7: 武器、防御和返回 (不变) ---
    # fire_missile = 0.0
    # in_range = 800 < distance < 6000
    # is_aligned = total_error_rad < np.deg2rad(60.0)
    # if in_range and is_aligned:
    #     fire_missile = 1.0

    release_flare = 0.0

    # 调用新的火控决策逻辑
    fire_missile_decision = should_fire_missile(env)
    fire_missile = 1.0 if fire_missile_decision else 0.0

    return [nx_cmd, nz_cmd, phi_cmd, 0.0, release_flare, fire_missile]


# ========================= 主执行循环 (人 vs AI) =========================
if __name__ == '__main__':
    # --- 1. 初始化环境 ---
    env = AirCombatEnv(tacview_enabled=True)
    observations = env.reset()
    # if hasattr(get_blue_ai_action, "prev_action"):
    #     del get_blue_ai_action.prev_action
    # if hasattr(get_blue_ai_action, "prev_los"):
    #     del get_blue_ai_action.prev_los
    dones = {"__all__": False}

    print("\n" + "=" * 50)
    print("--- 人机对抗模式 ---")
    print("=" * 50)
    print("你将操作 [红方 F-16]，对抗一台规则AI [蓝方 F-16]")
    print("\n--- 你的控制 (Red F-16) ---")
    print("  油门: Left Shift (加) / Left Ctrl (减)")
    print("  俯仰: S (拉杆) / W (推杆)")
    print("  滚转: A (左) / D (右)")
    print("  偏航: Z (左) / C (右)")
    print("  发射导弹: Space (空格键)")
    print("  投放诱饵: F")
    print("\n--- 通用控制 ---")
    print("  Q: 退出仿真")
    print("-" * 50 + "\n")

    # --- 2. 初始化红方动作变量 ---
    red_throttle = 0.0

    # --- 3. 主循环 ---
    while not dones["__all__"]:
        # --- 3.1. 获取人类玩家（红方）的动作 ---
        if keyboard.is_pressed('left shift'): red_throttle = min(1.0, red_throttle + 0.05)
        if keyboard.is_pressed('left ctrl'): red_throttle = max(0.0, red_throttle - 0.05)
        red_elevator = 0.0
        if keyboard.is_pressed('s'): red_elevator = -0.5
        if keyboard.is_pressed('w'): red_elevator = 0.5
        red_aileron = 0.0
        if keyboard.is_pressed('a'): red_aileron = -0.5
        if keyboard.is_pressed('d'): red_aileron = 0.5
        red_rudder = 0.0
        if keyboard.is_pressed('z'): red_rudder = -0.5
        if keyboard.is_pressed('c'): red_rudder = 0.5
        red_fire_missile = 1.0 if keyboard.is_pressed('space') else 0.0
        red_release_flare = 1.0 if keyboard.is_pressed('f') else 0.0

        red_action = [
            red_throttle, red_elevator, red_aileron, red_rudder,
            red_release_flare, red_fire_missile
        ]

        # --- 3.2. 获取AI（蓝方）的动作 ---
        blue_obs = observations['blue_agent']
        blue_action = get_blue_ai_action(blue_obs, env)
        # blue_action = get_blue_ai_action_random(blue_obs, env)

        # --- 3.3. 检查通用控制 ---
        if keyboard.is_pressed('q'):
            print("用户请求退出仿真。")
            break

        # --- 3.4. 将指令打包为 action 字典 ---
        actions = {
            'red_agent': red_action,
            'blue_agent': blue_action
        }

        # --- 3.5. 执行仿真步 ---
        observations, rewards, dones, info = env.step(actions)

        # 打印简要状态信息
        print(
            f"\rTime: {env.t_now:6.2f}s | Red Vel: {env.red_aircraft.velocity:6.1f} m/s | Blue Vel: {env.blue_aircraft.velocity:6.1f} m/s",
            end="")

        # 短暂休眠
        time.sleep(env.dt_dec)

    # --- 4. 仿真结束处理 ---
    print("\n\n" + "=" * 50)
    print("仿真结束!")
    if env.red_alive and not env.blue_alive:
        print(">>> 恭喜！你胜利了！ <<<")
    elif not env.red_alive and env.blue_alive:
        print(">>> 你被AI击落了！ <<<")
    elif not env.red_alive and not env.blue_alive:
        print(">>> 双方都被击落！(平局) <<<")
    else:
        print(">>> 仿真超时或双方脱离接触！(平局) <<<")
    print("=" * 50)

    # 显示matplotlib轨迹图
    env.render()