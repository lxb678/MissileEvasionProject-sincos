# 文件: fire_control_rules.py

import numpy as np
import time

# --- 导入模型类 (路径保持不变) ---
from Interference_code.env.Missilelaunch_environment_jsbsim.aircraft import AircraftPointMass
from Interference_code.env.Missilelaunch_environment_jsbsim.missile import Missile


# ==============================================================================
# --- 模拟参数与配置 (不变) ---
# ==============================================================================
class SimConfig:
    RED_AC_MAX_G = 9.0
    RED_AC_MIN_G = -5.0
    MISSILE_KILL_RADIUS_M = 12.0
    MAX_SIM_TIME_S = 45.0
    SIM_DT = 0.1
    SIM_DT_big = 0.3
    # ... (其他参数不变)
    MISS_DURATION_THRESHOLD_S = 2.0
    MISS_CHECK_START_TIME_S = 3.0
    MAX_OFF_BORESIGHT_ANGLE_DEG = 90.0


# ==============================================================================
# --- <<< 核心修改：机动策略生成器现在直接返回控制量 >>> ---
# ==============================================================================

def generate_maneuver_controls(mode, n_ty_max, current_state):
    """
    根据您提供的特定公式，直接计算并返回一个控制指令列表。

    Args:
        mode (int): 1-左转, 2-右转, 3-爬升, 4-俯冲, 5-平飞
        n_ty_max (float): 飞机的最大可用G值。
        current_state (np.ndarray): 飞机【当前】的状态向量，用于计算。

    Returns:
        list: 一个控制指令列表 [nx, nz, phi_cmd]
    """
    pitch_rad = current_state[4]

    if mode == 1:  # 左转
        nz_cmd = n_ty_max
        cos_pitch_over_g = np.clip(np.cos(pitch_rad) / nz_cmd, -1.0, 1.0)
        phi_cmd = -np.arccos(cos_pitch_over_g)
        # nx_cmd = np.sin(pitch_rad)
        nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]

    elif mode == 2:  # 右转
        nz_cmd = n_ty_max
        cos_pitch_over_g = np.clip(np.cos(pitch_rad) / nz_cmd, -1.0, 1.0)
        phi_cmd = np.arccos(cos_pitch_over_g)
        # nx_cmd = np.sin(pitch_rad)
        nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]

    elif mode == 3:  # 爬升
        # nz_cmd = np.cos(pitch_rad) + n_ty_max
        nz_cmd = n_ty_max
        phi_cmd = 0.0
        # nx_cmd = np.sin(pitch_rad)
        nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]

    elif mode == 4:  # 俯冲
        # nz_cmd = np.cos(pitch_rad) - n_ty_max
        nz_cmd = -5.0
        phi_cmd = 0.0
        # nx_cmd = np.sin(pitch_rad)
        nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]

    elif mode == 5:  # 平飞
        return [1.0, 1.0, 0.0]

    else:
        raise ValueError(f'无效的机动模式: {mode}')


# ==============================================================================
# --- <<< 新增：高精度命中检查辅助函数 (已修正) >>> ---
# ==============================================================================

def check_interpolated_hit(T1, T2, M1, M2, kill_radius, dt) -> bool:
    """
    一个独立的、从您的环境中改编的函数，用于检查两个时间步之间的插值脱靶量。
    (V2 - 已根据用户反馈修正了核心数学逻辑)

    Args:
        T1 (np.ndarray): 上一步的目标位置
        T2 (np.ndarray): 当前的目标位置
        M1 (np.ndarray): 上一步的导弹位置
        M2 (np.ndarray): 当前的导弹位置
        kill_radius (float): 引信触发半径 (R_kill)
        dt (float): 时间步长

    Returns:
        bool: 如果在步长内触发了引信，则返回 True。
    """
    # 只有当当前距离足够近时才进行昂贵的插值计算
    if np.linalg.norm(T2 - M2) > 500:
        return False

    M1_M2 = M2 - M1
    T1_T2 = T2 - T1
    M1_T1 = T1 - M1

    # V_rel_sq = || V_T - V_M ||^2 * dt^2
    relative_velocity_sq = np.linalg.norm(T1_T2 - M1_M2) ** 2

    if relative_velocity_sq < 1e-6:
        # 相对速度接近零，无法进行稳定插值，直接使用起始点距离
        interpolated_miss_distance = np.linalg.norm(M1_T1)
    else:
        # --- <<< 核心修正部分 >>> ---

        # 1. 计算归一化的最接近时刻 tau (一个在 [0, 1] 范围内的比例)
        # tau = - (R1 · V_rel) / ||V_rel||^2
        tau = -np.dot(M1_T1, T1_T2 - M1_M2) / relative_velocity_sq

        # 2. 检查时刻是否在有效范围内 [0, 1]
        #    这对应于绝对时间在 [t1, t1 + dt] 的区间内
        if 0 <= tau <= 1:
            # 3. 计算该时刻的相对位置矢量并求其模长
            # R(tau) = R1 + tau * (V_T - V_M) * dt
            relative_pos_at_closest = M1_T1 + tau * (T1_T2 - M1_M2)
            interpolated_miss_distance = np.linalg.norm(relative_pos_at_closest)
        else:
            # 最近点不在步长区间内，不触发
            return False
        # --- <<< 修正结束 >>> ---

    # 检查引信是否触发
    if interpolated_miss_distance < kill_radius:
        return True

    return False



# ==============================================================================
# --- 核心模拟函数 (已修改，接收通用参数) ---
# ==============================================================================

def simulate_intercept(launcher_ac, target_ac, target_maneuver_controls) -> str:
    """
    (通用版) 运行一次交战模拟。
    Args:
        launcher_ac: 发射方飞机对象。
        target_ac: 目标方飞机对象。
        target_maneuver_controls: 目标在此次模拟中采取的固定控制指令。
    """
    # --- 1. 初始化模拟状态 ---
    target_sim_ac = AircraftPointMass(initial_state=np.copy(target_ac.state_vector))

    launcher_pos = launcher_ac.pos
    launcher_vel_vec = launcher_ac.get_velocity_vector()
    missile_initial_vel_vec = launcher_vel_vec
    V_init = np.linalg.norm(missile_initial_vel_vec)
    if V_init < 1e-6:
        _, theta_init, psi_init, _, _ = launcher_ac.attitude_rad
        V_init = 0.0
    else:
        vx, vy, vz = missile_initial_vel_vec
        theta_init = np.arcsin(np.clip(vy / V_init, -1, 1))
        psi_init = np.arctan2(vz, vx)
    missile_initial_state = np.array([V_init, theta_init, psi_init, *launcher_pos])
    missile_sim = Missile(initial_state=missile_initial_state)

    # --- 2. 模拟循环 ---
    sim_time = 0.0
    target_pos_prev = np.copy(target_sim_ac.pos)
    missile_pos_prev = np.copy(missile_sim.pos)
    last_distance = np.inf
    time_distance_increasing = 0.0

    while sim_time < SimConfig.MAX_SIM_TIME_S:
        if not missile_sim.is_active: return 'TIMEOUT'

        # --- <<< 核心修正：动态步长逻辑 >>> ---
        # 步长的选择应该基于【当前】的距离，而不是上一步的
        current_distance = np.linalg.norm(missile_sim.pos - target_sim_ac.pos)
        dt = SimConfig.SIM_DT_big if current_distance > 500 else SimConfig.SIM_DT
        # --- <<< 修正结束 >>> ---

        # 更新状态
        target_sim_ac.update(dt, target_maneuver_controls)
        # 导弹追踪的是【当前】的目标位置
        # missile_sim.update(dt, target_pos_equiv=target_sim_ac.pos)
        missile_sim.update(dt, target_pos_equiv=target_pos_prev)

        # 检查命中/坠毁/失速
        altitude = target_sim_ac.pos[1];
        speed = target_sim_ac.velocity
        if altitude <= 100.0: return 'HIT'  # (坠机)
        if speed < target_sim_ac.min_velocity_ms: return 'HIT'  # (失速)

        is_hit = check_interpolated_hit(
            T1=target_pos_prev, T2=target_sim_ac.pos,
            M1=missile_pos_prev, M2=missile_sim.pos,
            kill_radius=SimConfig.MISSILE_KILL_RADIUS_M, dt=dt
        )
        if is_hit: return 'HIT'

        # 检查错过
        if sim_time > SimConfig.MISS_CHECK_START_TIME_S:
            if current_distance > last_distance:
                time_distance_increasing += dt
            else:
                time_distance_increasing = 0.0
            if time_distance_increasing >= SimConfig.MISS_DURATION_THRESHOLD_S:
                return 'MISS'

        # 更新历史状态
        target_pos_prev = np.copy(target_sim_ac.pos)
        missile_pos_prev = np.copy(missile_sim.pos)
        last_distance = current_distance
        sim_time += dt

    return 'TIMEOUT'


# ==============================================================================
# --- <<< 核心修改：通用化的主决策函数 >>> ---
# ==============================================================================

# 我们需要一个地方来存储双方的开火冷却时间
fire_control_timers = {
    'red': -10.0,
    'blue': -10.0
}
FIRE_COOLDOWN_S = 10.0


def can_launch_missile_with_pk(launcher_ac, target_ac, current_sim_time: float) -> bool:
    """
    (通用版) 根据前向模拟结果，决定是否可以发射高命中率导弹。

    Args:
        launcher_ac: 攻击方飞机对象。
        target_ac: 目标方飞机对象。
        current_sim_time (float): 当前主环境的仿真时间，用于冷却计算。

    Returns:
        bool: 如果满足所有条件并预测命中，则返回 True。
    """
    global fire_control_timers
    launcher_side = 'red' if 'Red' in launcher_ac.color else 'blue'

    # --- 0. 预检查 ---
    MIN_FIRE_RANGE = 1000.0
    MAX_FIRE_RANGE = 18000.0

    # a) 弹药和冷却检查
    if not hasattr(launcher_ac, 'missile_ammo') or launcher_ac.missile_ammo <= 0: return False
    if current_sim_time - fire_control_timers[launcher_side] < FIRE_COOLDOWN_S: return False

    # b) 射程检查
    distance = np.linalg.norm(target_ac.pos - launcher_ac.pos)
    if not (MIN_FIRE_RANGE < distance < MAX_FIRE_RANGE): return False

    # c) 离轴角检查
    theta, psi, _ = launcher_ac.attitude_rad
    forward_vector = np.array([np.cos(theta) * np.cos(psi), np.sin(theta), np.cos(theta) * np.sin(psi)])
    los_vector = target_ac.pos - launcher_ac.pos
    cos_angle = np.dot(forward_vector, los_vector) / (np.linalg.norm(los_vector) + 1e-6)
    off_boresight_angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if np.rad2deg(off_boresight_angle_rad) > SimConfig.MAX_OFF_BORESIGHT_ANGLE_DEG:
        return False

    # --- 1. 定义敌机可能的规避机动 ---
    max_g = SimConfig.RED_AC_MAX_G  # 假设双方最大G相同
    evasive_maneuvers = {'STRAIGHT': 5, 'LEFT_BREAK': 1, 'RIGHT_BREAK': 2, 'PULL_UP': 3, 'PUSH_DOWN': 4}

    # --- 2. 运行所有模拟场景 ---
    all_scenarios_hit = True
    current_target_state = target_ac.state_vector
    for maneuver_name, mode in evasive_maneuvers.items():
        fixed_controls = generate_maneuver_controls(mode, max_g, current_target_state)
        result = simulate_intercept(launcher_ac, target_ac, fixed_controls)
        if result != 'HIT':
            all_scenarios_hit = False
            break

    # --- 3. 做出最终决策 ---
    if all_scenarios_hit:
        # print(f">>> {launcher_side.upper()} 方决策: 发射导弹！(所有模拟场景均命中)")
        fire_control_timers[launcher_side] = current_sim_time  # 更新该方的开火计时器
        return True
    else:
        return False