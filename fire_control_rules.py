# 文件: fire_control_rules.py

import numpy as np
import time

# --- 导入您提供的飞机和导弹模型 ---
from Interference_code.env.Missilelaunch_environment_jsbsim.aircraft import AircraftPointMass
from Interference_code.env.Missilelaunch_environment_jsbsim.missile import Missile


# ==============================================================================
# --- 模拟参数与配置 (可调整) ---
# ==============================================================================

class SimConfig:
    """存放所有前向模拟所需的物理和性能参数"""
    # --- 敌机模型参数 ---
    RED_AC_MAX_G = 9.0  # 敌机最大可用过载
    RED_AC_MIN_G = -5.0  # 敌机最小可用过载 (推杆)

    # --- 引信与超时 ---
    MISSILE_KILL_RADIUS_M = 12.0  # (米) 导弹引信触发半径
    MAX_SIM_TIME_S = 45.0  # (秒) 单次模拟的最长时间

    # --- 模拟器参数 ---
    # SIM_DT = 0.05  # (秒) 前向模拟的时间步长，可以比主环境粗糙以提高速度
    # 仿真步长增加到0.1秒，因为插值命中检查可以弥补离散步长带来的误差

    SIM_DT = 0.1 #距离小于500米，小步长
    SIM_DT_big = 0.3 #距离大于500米，大步长


    # <<< 核心修改 >>>
    # 新增：判断为“错过”所需的时间阈值
    MISS_DURATION_THRESHOLD_S = 2.0  # (秒) 距离必须持续增加这么长时间才算错过
    MISS_CHECK_START_TIME_S = 3.0  # (秒) 在模拟开始后这么长时间才开始检查“错过”条件，给导弹足够时间制导
    # <<< 新增：战术发射参数 >>>
    MAX_OFF_BORESIGHT_ANGLE_DEG = 90.0  # (度) 最大离轴发射角


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
# --- 核心模拟函数 ---
# ==============================================================================

def simulate_intercept(env, red_maneuver_controls) -> str:
    """
    运行一次交战模拟，预测导弹是否能命中采取特定机动的敌机。现在使用高精度的插值命中检查。

    Args:
        env: 主环境对象，用于获取初始状态。
        red_maneuver_controls: 敌机在此次模拟中采取的固定控制指令 [nx, nz, phi_cmd]。

    Returns:
        'HIT', 'MISS', or 'TIMEOUT'
    """
    blue_ac = env.blue_aircraft
    red_ac = env.red_aircraft
    dt = SimConfig.SIM_DT

    # --- 1. 初始化模拟状态 (创建飞机和导弹的副本) ---

    # a) 敌机副本
    # 我们创建一个新的飞机实例，其初始状态与当前环境中的红方完全相同
    red_sim_ac = AircraftPointMass(initial_state=np.copy(red_ac.state_vector))

    # b) 导弹副本
    # 导弹的初始状态由发射它的蓝方飞机决定
    launcher_pos = blue_ac.pos
    launcher_vel_vec = blue_ac.get_velocity_vector()

    # ### <<< 核心修正: 正确初始化导弹速度 >>> ###
    # 导弹的初始速度矢量就是发射载机的速度矢量。
    # 导弹自身的动力学模型 (missile_sim.update) 会在模拟的前几秒内
    # 自动施加其内部定义的助推器推力 (self.thrust)。
    # 我们不再需要在这里手动添加一个瞬时的速度增益。
    missile_initial_vel_vec = launcher_vel_vec

    # 将初始速度矢量转换回导弹状态所需的 [V, theta, psi]
    V_init = np.linalg.norm(missile_initial_vel_vec)

    # 处理载机速度极低或为零的边缘情况，防止除零错误
    if V_init < 1e-6:
        # 如果载机静止，导弹初始方向沿机头方向，速度为0
        _, theta_init, psi_init, _, _ = blue_ac.attitude_rad
        V_init = 0.0
    else:
        vx, vy, vz = missile_initial_vel_vec
        theta_init = np.arcsin(np.clip(vy / V_init, -1, 1))
        psi_init = np.arctan2(vz, vx)

    missile_initial_state = np.array([V_init, theta_init, psi_init, *launcher_pos])
    missile_sim = Missile(initial_state=missile_initial_state)
    # ### <<< 修正结束 >>> ###

    # --- 2. 模拟循环 ---
    sim_time = 0.0
    # <<< 核心修改: 增加历史状态用于插值 >>>
    red_pos_prev = np.copy(red_sim_ac.pos)
    missile_pos_prev = np.copy(missile_sim.pos)

    # <<< 核心修改: 增加状态变量用于“错过”判断 >>>
    last_distance = np.inf
    time_distance_increasing = 0.0  # 计时器，用于记录距离持续增加的时长

    while sim_time < SimConfig.MAX_SIM_TIME_S:
        # a) 检查导弹自身状态
        if not missile_sim.is_active:
            return 'TIMEOUT'  # 导弹自毁或燃料耗尽
        if last_distance > 500:
            dt = SimConfig.SIM_DT_big
        else:
            dt = SimConfig.SIM_DT

        # b) 更新敌机状态
        # 敌机在整个模拟过程中执行固定的规避机动
        red_sim_ac.update(dt, red_maneuver_controls)

        # c) 更新导弹状态
        # 导弹的 update 方法需要目标的位置
        missile_sim.update(dt, target_pos_equiv=red_pos_prev)

        # --- d) 检查所有终止条件 ---

        # ------------------- START OF MODIFICATION -------------------
        # 1. 检查敌机是否因自身机动而自毁 (坠机或失速)
        #    这被视为一次成功的 "战术击杀"
        altitude = red_sim_ac.pos[1]  # y 是 'Up' 坐标
        speed = red_sim_ac.velocity

        if altitude <= 100.0:
            return 'HIT'  # 判定为命中 (坠机)

        if speed < red_sim_ac.min_velocity_ms:
            return 'HIT'  # 判定为命中 (失速)
        # -------------------- END OF MODIFICATION --------------------

        # 2. 检查导弹是否命中目标
        is_hit = check_interpolated_hit(
            T1=red_pos_prev,
            T2=red_sim_ac.pos,
            M1=missile_pos_prev,
            M2=missile_sim.pos,
            kill_radius=SimConfig.MISSILE_KILL_RADIUS_M,
            dt=dt
        )
        if is_hit:
            return 'HIT'

        # <<< 核心修改: 实现新的“错过”判断逻辑 >>>
        # 3. 检查导弹是否错过目标
        current_distance = np.linalg.norm(missile_sim.pos - red_sim_ac.pos)

        # 仅在模拟进行一段时间后才开始检查
        if sim_time > SimConfig.MISS_CHECK_START_TIME_S:
            if current_distance > last_distance:
                # 如果距离在增加，累加计时器
                time_distance_increasing += dt
            else:
                # 如果距离在减小或不变，清零计时器
                time_distance_increasing = 0.0

            # 如果计时器超过阈值，则判定为“错过”
            if time_distance_increasing >= SimConfig.MISS_DURATION_THRESHOLD_S:
                return 'MISS'

            # --- d) 更新历史状态，为下一次循环做准备 ---
        red_pos_prev = np.copy(red_sim_ac.pos)
        missile_pos_prev = np.copy(missile_sim.pos)
        last_distance = current_distance
        sim_time += dt

    return 'TIMEOUT'


# ==============================================================================
# --- 主决策函数 ---
# ==============================================================================

def should_fire_missile(env) -> bool:
    """
    根据前向模拟结果，决定是否发射导弹。
    只有在所有模拟的敌机规避场景中都能命中时，才决定发射。
    """
    # --- 0. 定义战术参数和预检查 ---
    MIN_FIRE_RANGE = 1000.0
    MAX_FIRE_RANGE = 18000.0  # 这个最大距离是启动昂贵计算的“门槛”
    FIRE_COOLDOWN_S = 10.0 #10.0  #5.0

    blue_ac = env.blue_aircraft
    red_ac = env.red_aircraft

    # a) 弹药和冷却检查
    if not hasattr(blue_ac, 'missile_ammo') or blue_ac.missile_ammo <= 0: return False
    current_time = env.t_now
    if not hasattr(should_fire_missile, "last_fire_time"):
        should_fire_missile.last_fire_time = -FIRE_COOLDOWN_S
    if current_time - should_fire_missile.last_fire_time < FIRE_COOLDOWN_S: return False

    # b) 粗略的射程检查，避免不必要的计算
    distance = np.linalg.norm(red_ac.pos - blue_ac.pos)
    if not (MIN_FIRE_RANGE < distance < MAX_FIRE_RANGE): return False

    # ------------------- START OF MODIFICATION -------------------
    # c) <<< 新增：离轴角 (Off-Boresight Angle) 发射检查 >>>
    # 这个检查确保目标在导弹导引头的视场范围内，才能进行发射前锁定。

    # 1. 获取蓝方机头的指向矢量 (Forward Vector)
    #    state_vector: [x,y,z, Vt, theta, psi, phi, p_real]
    theta, psi, phi = blue_ac.attitude_rad
    forward_vector = np.array([
        np.cos(theta) * np.cos(psi),
        np.sin(theta),
        np.cos(theta) * np.sin(psi)
    ])

    # 2. 获取从蓝方指向红方的视线矢量 (Line-of-Sight Vector)
    los_vector = red_ac.pos - blue_ac.pos
    los_vector_unit = los_vector / (np.linalg.norm(los_vector) + 1e-6)

    # 3. 计算两个矢量之间的夹角 (离轴角)
    #    点积公式: a · b = |a| |b| cos(angle)
    cos_angle = np.dot(forward_vector, los_vector_unit)
    # 限制在[-1, 1]以避免数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    off_boresight_angle_rad = np.arccos(cos_angle)
    off_boresight_angle_deg = np.rad2deg(off_boresight_angle_rad)

    # 4. 判断是否在允许的最大离轴角内
    if off_boresight_angle_deg > SimConfig.MAX_OFF_BORESIGHT_ANGLE_DEG:
        # (可选) 打印调试信息，了解为什么不开火
        # print(f"--- 暂不发射: 离轴角过大 ({off_boresight_angle_deg:.1f}° > {SimConfig.MAX_OFF_BORESIGHT_ANGLE_DEG}°)")
        return False
    # -------------------- END OF MODIFICATION --------------------

    # --- 1. 定义敌机可能的规避机动 ---
    # 格式: [nx (油门/减速板), nz (法向过载), phi_cmd (目标滚转角)]
    max_g = SimConfig.RED_AC_MAX_G
    min_g = SimConfig.RED_AC_MIN_G

    # 我们不再定义固定的控制量列表，而是定义机动模式的字典
    # 键是描述性名称，值是 generate_maneuver_function 所需的模式编号
    evasive_maneuvers = {
        'STRAIGHT': 5,  # 平飞
        'LEFT_BREAK': 1,  # 左转
        'RIGHT_BREAK': 2,  # 右转
        'PULL_UP': 3,  # 爬升
        'PUSH_DOWN': 4,  # 俯冲
    }

    # --- 2. 运行所有模拟场景 ---
    # print(f"\n--- {env.t_now:.1f}s: 蓝方AI正在进行开火决策模拟 (距离: {distance:.0f}m) ---")
    all_scenarios_hit = True
    # 获取敌机【当前】的真实状态，我们将基于这个状态来计算规避指令
    current_red_state = red_ac.state_vector
    for maneuver_name, mode in evasive_maneuvers.items():
        # <<< 核心修改：在这里预先计算出固定的控制指令 >>>
        # 敌机将会在整个模拟过程中，一直使用这个基于初始状态计算出的指令
        fixed_controls = generate_maneuver_controls(mode, max_g, current_red_state)
        result = simulate_intercept(env, fixed_controls)
        # print(f"    - 模拟敌机机动 '{maneuver_name}': 结果 -> {result}")
        if result != 'HIT':
            all_scenarios_hit = False
            break  # 只要有一个场景打不中，就没必要继续模拟了

    # --- 3. 做出最终决策 ---
    if all_scenarios_hit:
        print(f">>> 决策: 发射导弹！所有模拟场景均命中。")
        should_fire_missile.last_fire_time = current_time
        return True
    else:
        # print(f">>> 决策: 暂不发射。存在导弹可能脱靶的场景。")
        return False