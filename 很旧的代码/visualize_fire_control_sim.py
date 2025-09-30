# 文件: visualize_fire_control_sim.py

import numpy as np
import time
import os

# --- 导入您项目中的核心类和函数 ---
# 请确保这些文件的路径相对于此脚本是正确的
from env.Missilelaunch_environment_jsbsim.aircraft import AircraftPointMass
from env.Missilelaunch_environment_jsbsim.missile import Missile
from env.Missilelaunch_environment_jsbsim.tacview_interface import TacviewInterface as Tacview
from fire_control_rules import *


# ==============================================================================
# --- 这是一个修改版的模拟函数，专门用于可视化 ---
# ==============================================================================
def visualize_simulated_intercept(tacview, blue_ac_initial, red_ac_initial, red_maneuver_controls) -> str:
    """
    运行一次交战模拟，并将整个过程流式传输到Tacview。
    """
    dt = SimConfig.SIM_DT

    # --- 1. 初始化模拟状态 ---
    red_sim_ac = AircraftPointMass(initial_state=np.copy(red_ac_initial.state_vector))
    # 在可视化中，我们也需要一个蓝方飞机对象用于显示
    blue_sim_ac = AircraftPointMass(initial_state=np.copy(blue_ac_initial.state_vector))

    # --- 导弹初始化 (与原始代码相同) ---
    launcher_pos = blue_sim_ac.pos
    launcher_vel_vec = blue_sim_ac.get_velocity_vector()
    missile_initial_vel_vec = launcher_vel_vec
    V_init = np.linalg.norm(missile_initial_vel_vec)
    if V_init < 1e-6:
        _, theta_init, psi_init, _, _ = blue_sim_ac.attitude_rad;
        V_init = 0.0
    else:
        vx, vy, vz = missile_initial_vel_vec
        theta_init = np.arcsin(np.clip(vy / V_init, -1, 1));
        psi_init = np.arctan2(vz, vx)
    missile_initial_state = np.array([V_init, theta_init, psi_init, *launcher_pos])
    missile_sim = Missile(initial_state=missile_initial_state)

    # --- 2. 模拟循环与可视化 ---
    sim_time = 0.0
    red_pos_prev = np.copy(red_sim_ac.pos);
    missile_pos_prev = np.copy(missile_sim.pos)
    last_distance = np.inf;
    time_distance_increasing = 0.0

    # 分配唯一的Tacview ID
    blue_id = 1
    red_id = 2
    missile_id = 100

    print("\n--- 开始模拟与可视化 ---")
    while sim_time < SimConfig.MAX_SIM_TIME_S:
        if not missile_sim.is_active:
            print(f"--- 模拟结束于 {sim_time:.2f}s: 导弹失效 ---")
            return 'TIMEOUT'

        # 更新状态
        red_sim_ac.update(dt, red_maneuver_controls)
        missile_sim.update(dt, target_pos_equiv=red_sim_ac.pos)

        # --- 流式传输到 Tacview ---
        tacview.stream_aircraft(sim_time, blue_id, blue_sim_ac, color='Blue')
        tacview.stream_aircraft(sim_time, red_id, red_sim_ac, color='Red')
        tacview.stream_missile(sim_time, missile_id, missile_sim, parent_id=blue_id)

        # 检查命中/错过等条件
        if check_interpolated_hit(T1=red_pos_prev, T2=red_sim_ac.pos, M1=missile_pos_prev, M2=missile_sim.pos,
                                  kill_radius=SimConfig.MISSILE_KILL_RADIUS_M, dt=dt):
            print(f"--- 模拟结束于 {sim_time:.2f}s: 命中! ---")
            tacview.stream_explosion(sim_time, red_sim_ac.pos)  # 假设有这个方法
            return 'HIT'

        altitude = red_sim_ac.pos[1];
        speed = red_sim_ac.velocity
        if altitude <= 100.0:
            print(f"--- 模拟结束于 {sim_time:.2f}s: 敌机坠机 ---")
            return '(坠机)'
        if speed < red_sim_ac.min_velocity_ms:
            print(f"--- 模拟结束于 {sim_time:.2f}s: 敌机失速 ---")
            return '(失速)'

        current_distance = np.linalg.norm(missile_sim.pos - red_sim_ac.pos)
        if sim_time > SimConfig.MISS_CHECK_START_TIME_S:
            if current_distance > last_distance:
                time_distance_increasing += dt
            else:
                time_distance_increasing = 0.0
            if time_distance_increasing >= SimConfig.MISS_DURATION_THRESHOLD_S:
                print(f"--- 模拟结束于 {sim_time:.2f}s: 错过 ---")
                return 'MISS'

        red_pos_prev = np.copy(red_sim_ac.pos);
        missile_pos_prev = np.copy(missile_sim.pos)
        last_distance = current_distance;
        sim_time += dt

    print(f"--- 模拟结束于 {sim_time:.2f}s: 超时 ---")
    return 'TIMEOUT'


# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 选择你想测试的规避机动 ---
    # 可选项: 'STRAIGHT', 'LEFT_BREAK', 'RIGHT_BREAK', 'PULL_UP', 'PUSH_DOWN'
    MANEUVER_TO_TEST = 'LEFT_BREAK'

    print(f"--- 准备可视化场景: 敌机执行 '{MANEUVER_TO_TEST}' 机动 ---")

    # --- 2. 设置一个初始对抗场景 ---
    # 蓝方在西边，向东飞；红方在东边，向西飞（迎头）
    blue_initial_state = np.array([
        -2500.0, 5000.0, 0.0,  # pos (x, y, z)
        350.0,  # Vel (m/s)
        0.0,  # theta (pitch)
        np.deg2rad(90),  # psi (heading) -> East
        0.0,  # phi (roll)
        0.0  # roll rate
    ])

    red_initial_state = np.array([
        2500.0, 5000.0, 0.0,  # pos (x, y, z)
        350.0,  # Vel (m/s)
        0.0,  # theta (pitch)
        np.deg2rad(-90),  # psi (heading) -> West
        0.0,  # phi (roll)
        0.0  # roll rate
    ])

    # 创建飞机对象
    blue_ac = AircraftPointMass(initial_state=blue_initial_state)
    red_ac = AircraftPointMass(initial_state=red_initial_state)

    # --- 3. 初始化 Tacview ---
    output_filename = f"sim_visualization_{MANEUVER_TO_TEST}.acmi"
    tacview = Tacview(output_filename, "Fire Control Simulation")
    tacview.connect()  # 准备写入文件
    print(f"Tacview 文件将被保存在: {os.path.abspath(output_filename)}")

    # --- 4. 计算敌机将要执行的固定机动指令 ---
    maneuver_modes = {
        'STRAIGHT': 5, 'LEFT_BREAK': 1, 'RIGHT_BREAK': 2,
        'PULL_UP': 3, 'PUSH_DOWN': 4
    }
    mode = maneuver_modes[MANEUVER_TO_TEST]
    # 我们基于红方的初始状态来计算它将要执行的整个机动过程中的固定指令
    fixed_controls = generate_maneuver_controls(mode, SimConfig.RED_AC_MAX_G, red_ac.state_vector)

    # --- 5. 运行可视化模拟 ---
    result = visualize_simulated_intercept(tacview, blue_ac, red_ac, fixed_controls)

    # --- 6. 关闭 Tacview 并保存文件 ---
    tacview.close()

    print(f"\n可视化完成！最终结果: {result}")
    print(f"请用 Tacview 打开文件 '{output_filename}' 来分析模拟过程。")