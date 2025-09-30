import numpy as np
import random

# --- 导入您的机动生成函数 ---
# 假设 generate_maneuver_controls 也在这个文件中，或者可以从 fire_control_rules.py 导入
# from fire_control_rules import generate_maneuver_controls

# --- 蓝方AI的状态存储 ---
# 我们需要在这里“记住”蓝方当前正在执行的机动
blue_ai_maneuver_state = {
    'current_mode': 5,  # 初始机动：平飞
    'time_in_mode': 0.0,  # 当前机动已持续的时间
    'duration_of_mode': 5.0  # 初始机动的持续时间
}


# (这里是您的 generate_maneuver_controls 函数，从 fire_control_rules.py 复制过来)
def generate_maneuver_controls(mode, n_ty_max, current_state):
    """
    根据您提供的特定公式，直接计算并返回一个控制指令列表。
    """
    pitch_rad = current_state[4]
    if mode == 1:  # 左转
        nz_cmd = n_ty_max
        cos_pitch_over_g = np.clip(np.cos(pitch_rad) / nz_cmd, -1.0, 1.0)
        phi_cmd = -np.arccos(cos_pitch_over_g)
        # phi_cmd = np.deg2rad(-90)
        nx_cmd = np.sin(pitch_rad)
        # nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]
    elif mode == 2:  # 右转
        nz_cmd = n_ty_max
        cos_pitch_over_g = np.clip(np.cos(pitch_rad) / nz_cmd, -1.0, 1.0)
        phi_cmd = np.arccos(cos_pitch_over_g)
        # phi_cmd = np.deg2rad(90)
        nx_cmd = np.sin(pitch_rad)
        # nx_cmd = 1.0
        return [nx_cmd, nz_cmd, phi_cmd]
    elif mode == 3:  # 爬升
        # nz_cmd = np.cos(pitch_rad) + n_ty_max
        nz_cmd = 9.0
        phi_cmd = 0.0
        nx_cmd = np.sin(pitch_rad)
        # nx_cmd = 1.0
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


# --- 蓝方AI主逻辑函数 ---
def get_blue_ai_action_random(blue_obs: np.ndarray, env) -> list:
    """
   一个基于规则的AI，从一个机动库中随机选择机动来飞行。
   增加了低空安全爬升和高空控制俯冲的逻辑。
    """
    global blue_ai_maneuver_state  # 使用全局状态来保持记忆

    blue_ac = env.blue_aircraft
    current_blue_state = blue_ac.state_vector
    current_altitude = current_blue_state[1]
    current_pitch_rad = current_blue_state[4]

    # ------------------- START OF MODIFICATION -------------------
    # --- 0. 高优先级高度控制逻辑 ---

    LOW_SAFETY_ALTITUDE_M = 3000.0
    HIGH_CEILING_ALTITUDE_M = 10000.0

    altitude_control_triggered = False

    if current_altitude < LOW_SAFETY_ALTITUDE_M:
        # --- 低空保护：强制爬升 ---
        nx_cmd, nz_cmd, phi_cmd = generate_maneuver_controls(3, 9.0, current_blue_state)  # 模式3: 爬升
        # print(f"--- 蓝方AI: 低空警告! (高度 {current_altitude:.0f}m), 强制爬升 ---")
        altitude_control_triggered = True

    elif current_altitude > HIGH_CEILING_ALTITUDE_M:
        # --- 高空控制：强制俯冲 ---
        nx_cmd, nz_cmd, phi_cmd = generate_maneuver_controls(4, 9.0, current_blue_state)  # 模式4: 俯冲
        # print(f"--- 蓝方AI: 高空警告! (高度 {current_altitude:.0f}m), 强制俯冲 ---")
        altitude_control_triggered = True

    if altitude_control_triggered:
        # 如果触发了高度控制，就重置随机机动的计时器
        blue_ai_maneuver_state['time_in_mode'] = blue_ai_maneuver_state['duration_of_mode']
    else:
        # --- 1. 常规随机机动逻辑 (仅在安全高度范围内执行) ---
        # --- 1. 更新并决策机动模式 ---
        # 累加当前机动已执行的时间
        blue_ai_maneuver_state['time_in_mode'] += env.dt_dec

        # 检查当前机动是否已达到预设时长
        if blue_ai_maneuver_state['time_in_mode'] >= blue_ai_maneuver_state['duration_of_mode']:
            # 如果是，就随机选择一个新的机动
            # 基础机动库
            maneuver_library = [1, 2, 3, 4, 5]  # 包含所有可能性# 1:左转, 2:右转, 3:爬升, 4:俯冲, 5:平飞

            # 规则1: 高度限制
            if current_altitude < LOW_SAFETY_ALTITUDE_M + 2000:  # 5000米以下
                if 4 in maneuver_library: maneuver_library.remove(4)  # 禁止俯冲
            elif current_altitude > HIGH_CEILING_ALTITUDE_M - 2000:  # 8000米以上
                if 3 in maneuver_library: maneuver_library.remove(3)  # 禁止爬升

            # 规则2: 大俯仰角限制
            MAX_PITCH_DEG = 60.0
            if abs(np.rad2deg(current_pitch_rad)) > MAX_PITCH_DEG:
                # print(f"--- 蓝方AI: 大俯仰角警告! ({np.rad2deg(current_pitch_rad):.1f}°), 限制滚转机动 ---")
                if 1 in maneuver_library: maneuver_library.remove(1)  # 禁止左转
                if 2 in maneuver_library: maneuver_library.remove(2)  # 禁止右转

            # 安全备用：如果所有机动都被排除了，至少保留平飞
            if not maneuver_library:
                maneuver_library = [5]  # 保底策略：平飞

            new_mode = random.choice(maneuver_library)

            # 为新机动设置一个随机的持续时间 (例如，3到8秒之间)
            new_duration = random.uniform(3.0, 5.0)

            # 更新状态
            blue_ai_maneuver_state['current_mode'] = new_mode
            blue_ai_maneuver_state['time_in_mode'] = 0.0  # 重置计时器
            blue_ai_maneuver_state['duration_of_mode'] = new_duration

            # print(f"--- 蓝方AI: 切换机动 -> 模式 {new_mode}, 持续 {new_duration:.1f} 秒 ---")

        # --- 2. 计算当前帧的控制指令 ---

        blue_ac = env.blue_aircraft
        # 获取蓝方飞机的当前状态，用于计算
        current_blue_state = blue_ac.state_vector

        # 获取当前正在执行的机动模式
        current_mode = blue_ai_maneuver_state['current_mode']

        # 从您的机动库中为当前模式计算出控制指令
        # 假设蓝方最大G值为9.0
        nx_cmd, nz_cmd, phi_cmd = generate_maneuver_controls(current_mode, 9.0, current_blue_state)

    # --- 3. 武器与防御 (可以添加简单的逻辑) ---

    # 简单的开火逻辑：如果敌机在前方一定角度和距离内，就开火
    fire_missile = 0.0
    red_ac = env.red_aircraft
    distance = np.linalg.norm(red_ac.pos - blue_ac.pos)

    # 计算离轴角 (与 fire_control_rules.py 中逻辑相同)
    theta, psi, _ = blue_ac.attitude_rad
    forward_vector = np.array([np.cos(theta) * np.cos(psi), np.sin(theta), np.cos(theta) * np.sin(psi)])
    los_vector = red_ac.pos - blue_ac.pos
    los_norm = np.linalg.norm(los_vector) + 1e-9
    cos_angle = np.dot(forward_vector, los_vector) / los_norm
    off_boresight_angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if 1000 < distance < 6000 and off_boresight_angle_rad < np.deg2rad(30):
        fire_missile = 0.0

    release_flare = 0.0

    # --- 4. 返回最终的控制动作 ---
    # 注意: AircraftPointMass 的 update 方法期望 [nx, nz, p_cmd]
    # 而您的 generate_maneuver_controls 返回 [nx, nz, phi_cmd]
    # 这里我们假设蓝方是质点模型，并且它能处理 phi_cmd
    # 如果它只能处理滚转速率 p_cmd，则需要一个简单的 P 控制器来转换
    # p_cmd = K * (phi_cmd - current_phi)

    # 假设蓝方update能处理phi_cmd，我们把p_cmd设为0
    # (如果蓝方是JSBSim模型，则返回的动作格式需要调整)
    return [
        nx_cmd,
        nz_cmd,
        phi_cmd,  # 目标滚转角
        0.0,  # 滚转角速率 (如果模型需要)
        release_flare,
        fire_missile
    ]