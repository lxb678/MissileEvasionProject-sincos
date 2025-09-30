# 文件名: Run_MultiAgent_train.py
# 描述: 用于训练红蓝双方智能体的多智能体强化学习主脚本。

import torch
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter

# --- 导入您的环境和PPO算法 ---
# (确保这里的路径和文件名是正确的)
from env.Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim_pointmass import AirCombatEnv
from PPO_model.Config_launch import *
from PPO_model.Hybrid_PPO_jsbsim_launch import *  # 导入我们为进攻任务简化的PPO
from blue_ai_rules import get_blue_ai_action # <<< 导入蓝方AI
from Interference_code.blue_random_rules import get_blue_ai_action_random

# --- 全局设置 ---
# 如果要加载预训练模型，请将此设为 True
LOAD_MODELS = False
# 训练时是否开启 Tacview 可视化
TACVIEW_ENABLED_DURING_TRAINING = False
# 训练多少个回合
MAX_EPISODES = 100000
# 每隔多少个回合训练一次网络
TRAIN_INTERVAL_EPISODES = 5
# 随机种子
RANDOM_SEED = AGENTPARA.RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """设置所有相关的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir=C:\Users\LXB\Desktop\桌面资料\规避导弹项目\Interference_code\logs 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
#writer = SummaryWriter(log_dir='log/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))

# --- Tensorboard 日志记录 ---
log_time = time.strftime("%m-%d_%H-%M-%S", time.localtime())
writer = SummaryWriter(log_dir=f'logs/MARL_seed{RANDOM_SEED}_{log_time}_load_{LOAD_MODELS}')

# ========================= 主执行函数 =========================
if __name__ == '__main__':
    # 1. 初始化环境
    env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)
    set_seed()

    # 2. <<< 核心修改：为红蓝双方创建独立的智能体 >>>
    red_agent = PPO_continuous(load_able=LOAD_MODELS)
    # blue_agent = PPO_continuous(load_able=LOAD_MODELS)  # 蓝方也可以加载模型，实现自我对抗

    global_step = 0
    red_win_count = 0
    blue_win_count = 0
    draw_count = 0

    print("--- 开始训练红方智能体 (对抗规则AI) ---")
    for i_episode in range(MAX_EPISODES):
        # --- 回合开始 ---
        observations = env.reset()
        done = False

        # 回合奖励记录
        episode_rewards = {'red_agent': 0.0, 'blue_agent': 0.0}

        for t in range(5000):  # 设置一个最大步数，防止无限循环
            global_step += 1

            # --- 3. 双方智能体根据各自观测选择动作 ---
            red_agent.prep_eval_rl()  # 交互经验的时候不会传梯度
            with torch.no_grad():  # 交互时梯度不回传
                # 红方
                red_obs = observations['red_agent']
                red_env_action, red_action_to_store, red_prob = red_agent.choose_action(red_obs)
                red_value = red_agent.get_value(red_obs).cpu().detach().numpy()

                # 蓝方
                blue_obs = observations['blue_agent']
                # blue_env_action, blue_action_to_store, blue_prob = blue_agent.choose_action(blue_obs)
                # blue_value = blue_agent.get_value(blue_obs).cpu().detach().numpy()
                # blue_env_action = get_blue_ai_action(blue_obs, env)
                blue_env_action = get_blue_ai_action_random(blue_obs, env)

            # --- 4. 将动作打包并与环境交互 ---
            actions = {
                'red_agent': red_env_action,
                'blue_agent': blue_env_action
            }
            next_observations, rewards, dones, info = env.step(actions)
            done = dones['__all__']

            # --- 5. 将各自的经验存入各自的 Buffer ---
            red_agent.store_experience(red_obs, red_action_to_store, red_prob, red_value, rewards['red_agent'], done)
            # blue_agent.store_experience(blue_obs, blue_action_to_store, blue_prob, blue_value, rewards['blue_agent'], done)

            observations = next_observations
            episode_rewards['red_agent'] += rewards['red_agent']
            episode_rewards['blue_agent'] += rewards['blue_agent']

            if done:
                break

        # --- 回合结束 ---
        # 记录胜负情况
        if env.red_alive and not env.blue_alive:
            red_win_count += 1
        elif not env.red_alive and env.blue_alive:
            blue_win_count += 1
        else:
            draw_count += 1

        print(
            f"Episode {i_episode + 1} | T: {env.t_now:.2f}s | R_Red: {episode_rewards['red_agent']:.2f} | R_Blue: {episode_rewards['blue_agent']:.2f}")

        # --- 6. 定期训练和记录 ---
        if (i_episode + 1) % TRAIN_INTERVAL_EPISODES == 0:
            print(f"\n--- Episode {i_episode + 1}: 开始训练 ---")

            # 分别训练红蓝双方的智能体
            print("  - 训练红方智能体...")
            red_agent.prep_training_rl()
            red_train_info = red_agent.learn()
            # print("  - 训练蓝方智能体...")
            # blue_train_info = blue_agent.learn()

            print("--- 训练完成 ---\n")

            # 将训练信息写入 Tensorboard
            for key, value in red_train_info.items():
                writer.add_scalar(f'Train/red_agent/{key}', value, global_step)
            # for key, value in blue_train_info.items():
            #     writer.add_scalar(f'Train/blue_agent/{key}', value, global_step)

            # 训练完之后，需要验证模型，绘制奖励曲线(这个测试环境的奖励曲线使用幕奖励总和，在项目中可以考虑使用幕平均奖励)
            print("\n--------------------------")
            print("eval, global_step:{}".format(global_step))
            eval_episodes = 1
            red_agent.prep_eval_rl()
            with torch.no_grad():
                for i in range(eval_episodes):
                    obs = env.reset()
                    done = False
                    episode_red_reward = 0.0
                    for t in range(5000):  # 与训练循环相同的最大步数
                        # 关键：红方智能体使用确定性动作 (deterministic=True)
                        red_action, _, _ = red_agent.choose_action(obs['red_agent'], deterministic=True)

                        # 蓝方AI保持不变
                        # blue_action = get_blue_ai_action(obs['blue_agent'], env)
                        blue_action = get_blue_ai_action_random(obs['blue_agent'], env)

                        actions = {'red_agent': red_action, 'blue_agent': blue_action}
                        obs, rewards, dones, info = env.step(actions)
                        episode_red_reward += rewards['red_agent']
                        done = dones['__all__']
                        if done:
                            break
                    writer.add_scalar('Eval/episode_red_reward',
                                      episode_red_reward,
                                      global_step=global_step
                                      )
            print(f"--- 评估完成 ---")
            print("--------------------------\n")

        # 定期记录胜率 (例如每100回合)
        if (i_episode + 1) % 100 == 0:
            total_games = red_win_count + blue_win_count + draw_count
            if total_games > 0:
                red_win_rate = red_win_count / total_games
                blue_win_rate = blue_win_count / total_games
                draw_rate = draw_count / total_games

                print(f"\n--- 统计 (最近100回合) ---")
                print(f"  红方胜率: {red_win_rate:.2%}")
                print(f"  蓝方胜率: {blue_win_rate:.2%}")
                print(f"  平局率:   {draw_rate:.2%}")
                print("--------------------------\n")

                writer.add_scalar('Eval/red_win_rate', red_win_rate, i_episode + 1)
                writer.add_scalar('Eval/blue_win_rate', blue_win_rate, i_episode + 1)

                # 如果红方胜率很高，可以保存模型
                if red_win_rate > 0.9:
                    print(f"红方胜率达到 {red_win_rate:.2%}，保存模型...")
                    red_agent.save(prefix=f"red_win_rate_{int(red_win_rate * 100)}_ep{i_episode + 1}")
                    # blue_agent.save(prefix=f"blue_opponent_ep{i_episode + 1}")  # 同时保存对手的模型

                # 重置计数器
                red_win_count, blue_win_count, draw_count = 0, 0, 0

    # 训练结束后关闭 writer
    writer.close()
    print("训练完成！")