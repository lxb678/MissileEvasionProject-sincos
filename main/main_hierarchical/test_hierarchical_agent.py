# 文件: test_hierarchical_agent.py

import torch
import numpy as np
import random

# --- 导入您的环境和新的分层AI ---
from Interference_code.env.Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim_pointmass import AirCombatEnv
from Interference_code.env.missile_evasion_environment_jsbsim.Vec_missile_evasion_environment_jsbsim import AirCombatEnv as EvadeEnv  # 导入规避环境以获取其观测函数
from Interference_code.main.main_hierarchical.hierarchical_agent2 import HierarchicalAgent
from Interference_code.main.main_attack.blue_ai_rules import get_blue_ai_action

# --- 全局测试设置 ---
TACVIEW_ENABLED_DURING_TESTING = True
TOTAL_TEST_EPISODES = 100
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ========================= 主测试函数 =========================
if __name__ == '__main__':
    # 1. 初始化主空战环境
    env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TESTING)

    # (重要) 我们需要一个临时的规避环境实例，仅仅是为了调用它的观测函数
    # 这确保了我们能为规避Agent生成完全正确的观测数据
    temp_evade_env = EvadeEnv()

    set_seed()

    # 2. 初始化并加载分层智能体 (红方)
    # !!! 请将这里的路径替换为您真实模型文件的路径 !!!
    attack_model_folder= "../../test/test_hierarchical_model/attack_model"
    evade_model_folder = "../../test/test_hierarchical_model/evade_model"

    red_agent = HierarchicalAgent(
        attack_model_dir=attack_model_folder,  # <<< 使用新的参数名
        evade_model_dir=evade_model_folder,  # <<< 使用新的参数名
        load_models=True
    )

    # 3. 初始化统计变量
    red_win_count = 0
    blue_win_count = 0
    draw_count = 0

    print(f"\n--- 开始分层智能体评估（共 {TOTAL_TEST_EPISODES} 回合） ---")

    with torch.no_grad():
        for i_episode in range(TOTAL_TEST_EPISODES):
            # 重置主环境，获取初始的14维观测
            observations = env.reset()
            done = False
            # --- <<< 核心修改 1: 新增状态变量，用于跟踪威胁导弹 >>> ---
            # 初始化为 None，表示回合开始时没有威胁
            previous_threat_missile_id = None

            for t in range(5000):
                # --- 为红方准备两种不同的观测数据 ---
                # a) 主环境直接提供了攻击Agent所需的14维观测
                combat_obs_red = observations['red_agent']

                # --- <<< 核心修正：仅当导弹锁定时才视为威胁 >>> ---
                evade_obs_red = np.zeros(9)  # 默认无威胁
                most_dangerous_missile = None

                # 1. 找出所有激活的来袭导弹
                active_blue_missiles = [m for m in env.blue_missiles if m.is_active]

                if active_blue_missiles:
                    # 2. 筛选出所有【正在锁定】我机的导弹
                    locked_missiles = []
                    for m in active_blue_missiles:
                        # (调用 check_seeker_lock 的逻辑保持不变)
                        is_locked, _, _, _, _ = temp_evade_env._check_seeker_lock(
                            x_missile=m.state_vector,
                            x_target=env.red_aircraft.state_vector,
                            V_missile_vec=m.get_velocity_vector(),
                            V_target_vec=env.red_aircraft.get_velocity_vector(),
                            t_now=env.t_now  # (或 m.flight_time, 取决于您的函数签名)
                        )

                        if is_locked:
                            locked_missiles.append(m)

                    # 3. 如果存在锁定的导弹，则在它们之中找最近的
                    if locked_missiles:
                        most_dangerous_missile = min(
                            locked_missiles,
                            key=lambda m: np.linalg.norm(m.pos - env.red_aircraft.pos)
                        )
                        # print(f"  [威胁评估]: 锁定 {most_dangerous_missile.name}") # 调试信息

                        # --- <<< 核心修改 2: 比较并打印威胁切换信息 >>> ---
                        current_threat_missile_id = most_dangerous_missile.id if most_dangerous_missile else None

                        if current_threat_missile_id != previous_threat_missile_id:
                            if current_threat_missile_id is not None:
                                # 从无威胁变为有威胁，或切换到新的威胁
                                dist = np.linalg.norm(most_dangerous_missile.pos - env.red_aircraft.pos)
                                print(f"\n>>> [威胁切换] {env.t_now:.2f}s: "
                                      f"红方AI锁定新威胁 -> {most_dangerous_missile.name} (ID: ...{current_threat_missile_id[-6:]}), "
                                      f"距离: {dist:.0f}m <<<\n")
                            else:
                                # 从有威胁变为无威胁
                                print(f"\n>>> [威胁解除] {env.t_now:.2f}s: "
                                      f"红方AI认为所有导弹威胁已解除。 <<<\n")

                        # 更新状态，为下一帧做准备
                        previous_threat_missile_id = current_threat_missile_id
                        # --- <<< 修改结束 >>> ---

                # 4. 只有在找到了最危险的（已锁定的）导弹时，才生成规避观测
                if most_dangerous_missile:
                    temp_evade_env.aircraft = env.red_aircraft
                    temp_evade_env.missile = most_dangerous_missile
                    evade_obs_red = temp_evade_env._get_observation()
                # --- <<< 修改结束 >>> ---

                # --- 动作选择 ---
                # 红方: 分层AI决策
                # 调用 get_action 时，将 env 对象传递进去
                red_action = red_agent.get_action(
                    combat_obs=combat_obs_red,
                    evade_obs=evade_obs_red,
                    env=env  # <--- 新增的参数
                )
                # red_action 现在直接就是我们需要的6维向量了

                # 蓝方: 使用我们之前编写的随机机动规则AI
                blue_obs = observations['blue_agent']
                # blue_action = get_blue_ai_action_random(blue_obs, env)
                blue_action = get_blue_ai_action(blue_obs, env)

                # --- 环境交互 ---
                actions = {'red_agent': red_action, 'blue_agent': blue_action}
                observations, rewards, dones, info = env.step(actions)

                done = dones['__all__']
                if done:
                    break

            # --- 回合结束，记录统计数据 ---
            if env.red_alive and not env.blue_alive:
                red_win_count += 1
                result = "红方胜利 (HRL Agent)"
            elif not env.red_alive and env.blue_alive:
                blue_win_count += 1
                result = "蓝方胜利 (Random AI)"
            else:
                draw_count += 1
                result = "平局"

            print(f"回合 {i_episode + 1}/{TOTAL_TEST_EPISODES} | 结果: {result} | 仿真时长: {env.t_now:.2f}s")

            # (可选) 如果你想看每一局的轨迹，可以取消这行注释
            env.render()

    # --- 最终总结 ---
    total_games = red_win_count + blue_win_count + draw_count
    print("\n" + "=" * 50 + "\n--- 最终评估结果 ---\n" + "=" * 50)
    print(f"红方 (HRL Agent) 胜率: {red_win_count / total_games:.2%} ({red_win_count} 场)")
    print(f"蓝方 (Random AI) 胜率: {blue_win_count / total_games:.2%} ({blue_win_count} 场)")
    print(f"平局率:               {draw_count / total_games:.2%} ({draw_count} 场)")