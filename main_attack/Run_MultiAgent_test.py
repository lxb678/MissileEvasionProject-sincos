# 文件名: Run_MultiAgent_test.py
# 描述: 用于测试和评估已训练好的红方智能体，对抗一个基于规则的蓝方AI。

import torch
import numpy as np
import random
import time

# --- 导入您的环境和PPO算法 ---
# (确保这里的路径和文件名是正确的)
from Interference_code.env.Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim_pointmass import AirCombatEnv
from Interference_code.PPO_model.Config_launch import *
from Interference_code.PPO_model.Hybrid_PPO_jsbsim_launch import *
from blue_ai_rules import get_blue_ai_action  # <<< 导入蓝方AI

# --- 全局测试设置 ---
LOAD_MODELS = True  # 测试时必须加载模型
TACVIEW_ENABLED_DURING_TESTING = True
TOTAL_TEST_EPISODES = 100  # 您希望运行的总测试回合数
RANDOM_SEED = AGENTPARA.RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ========================= 主测试函数 =========================
if __name__ == '__main__':
    # 1. 初始化环境
    env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TESTING)
    set_seed()

    # 2. 初始化并加载红方智能体
    # 确保您的 PPO 类在 load_able=True 时会从正确的路径加载模型
    red_agent = PPO_continuous(load_able=LOAD_MODELS)

    # 3. 设置智能体为评估模式
    red_agent.prep_eval_rl()

    # 4. 初始化统计变量
    red_win_count = 0
    blue_win_count = 0
    draw_count = 0
    total_red_reward = 0.0
    episode_times = []

    print(f"--- 开始模型评估（共 {TOTAL_TEST_EPISODES} 回合） ---")

    # 使用 torch.no_grad() 可以提高推理速度，因为不需要计算梯度
    with torch.no_grad():
        for i_episode in range(TOTAL_TEST_EPISODES):
            episode_start_time = time.time()
            observations = env.reset()
            done = False
            episode_red_reward = 0.0

            for t in range(5000):  # 设置最大步数
                # --- 动作选择 ---
                # 红方智能体使用确定性动作 (deterministic=True)
                red_obs = observations['red_agent']
                red_action, _, _ = red_agent.choose_action(red_obs, deterministic=True)

                # 蓝方AI保持不变
                blue_obs = observations['blue_agent']
                blue_action = get_blue_ai_action(blue_obs, env)
                # blue_action = get_blue_ai_action_random(blue_obs, env) # 使用规则AI
                # blue_action = [0, 1, 0, 0.0, 0, 0]
                # --- 环境交互 ---
                actions = {'red_agent': red_action, 'blue_agent': blue_action}
                observations, rewards, dones, info = env.step(actions)

                episode_red_reward += rewards['red_agent']
                done = dones['__all__']

                if done:
                    break

            # --- 回合结束，记录统计数据 ---
            if env.red_alive and not env.blue_alive:
                red_win_count += 1
                result = "红方胜利"
            elif not env.red_alive and env.blue_alive:
                blue_win_count += 1
                result = "蓝方胜利"
            else:
                draw_count += 1
                result = "平局"

            total_red_reward += episode_red_reward
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)

            print(f"回合 {i_episode + 1}/{TOTAL_TEST_EPISODES} | 结果: {result} | "
                  f"红方奖励: {episode_red_reward:.2f} | 仿真时长: {env.t_now:.2f}s | "
                  f"实际用时: {episode_time:.2f}s")

            env.render()

    # --- 所有测试回合结束，打印最终总结 ---
    total_games = red_win_count + blue_win_count + draw_count

    print("\n" + "=" * 50)
    print("--- 最终评估结果 ---")
    print(f"总测试回合数: {total_games}")
    print(f"  - 红方 (PPO) 胜率: {red_win_count / total_games:.2%} ({red_win_count} 场)")
    print(f"  - 蓝方 (Rule) 胜率: {blue_win_count / total_games:.2%} ({blue_win_count} 场)")
    print(f"  - 平局率:          {draw_count / total_games:.2%} ({draw_count} 场)")
    print("-" * 20)
    print(f"红方平均回合奖励: {total_red_reward / total_games:.2f}")
    print(f"平均每回合实际用时: {np.mean(episode_times):.2f} 秒")
    print("=" * 50)