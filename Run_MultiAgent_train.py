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


# ==============================================================================
# --- 蓝方 AI 逻辑：追踪法 (从您的问题中复制) ---
# ==============================================================================
def get_blue_ai_action(blue_obs: dict, env) -> list:
    """
    一个基于规则的AI，使用 [nx, nz, phi_cmd] 来控制蓝方飞机
    追踪并攻击红方飞机。 phi_cmd 是目标滚转角 (rad)。
    """
    blue_ac = env.blue_aircraft
    red_ac = env.red_aircraft

    los_nue = red_ac.pos - blue_ac.pos
    distance = np.linalg.norm(los_nue)
    los_ned = np.array([los_nue[0], los_nue[2], -los_nue[1]])

    def euler_to_quaternion(phi, theta, psi):
        cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
        cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
        cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy
        norm = np.linalg.norm([q0, q1, q2, q3])
        return np.array([q0, q1, q2, q3]) / norm if norm > 1e-9 else np.array([1, 0, 0, 0])

    def quaternion_to_rotation_matrix(q):
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
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

    Kp_bank_angle = 10.0
    MAX_BANK_ANGLE_RAD = np.deg2rad(180.0)
    phi_cmd_rad = Kp_bank_angle * yaw_error_rad
    phi_cmd = np.clip(phi_cmd_rad, -MAX_BANK_ANGLE_RAD, MAX_BANK_ANGLE_RAD)

    NZ_COMMAND_HIGH = 7.0
    NZ_COMMAND_LOW = 1.0
    nz_cmd = NZ_COMMAND_HIGH if total_error_rad > np.deg2rad(2.0) else NZ_COMMAND_LOW
    nz_cmd = np.clip(nz_cmd, -5.0, 9.0)

    if distance > 4000:
        nx_cmd = 1.0
    elif distance < 1500:
        nx_cmd = 0.4
    else:
        nx_cmd = 0.85
    if distance < 500 and np.dot(blue_ac.get_velocity_vector(), los_nue) > 0: nx_cmd = -0.5

    fire_missile = 0.0
    in_range = 800 < distance < 6000
    is_aligned = total_error_rad < np.deg2rad(4.0)
    if in_range and is_aligned: fire_missile = 0.0  # 根据您的代码，这里暂时不开火

    release_flare = 0.0

    return [nx_cmd, nz_cmd, phi_cmd, 0.0, release_flare, fire_missile]


# --- 全局设置 ---
# 如果要加载预训练模型，请将此设为 True
LOAD_MODELS = False
# 训练时是否开启 Tacview 可视化
TACVIEW_ENABLED_DURING_TRAINING = True
# 训练多少个回合
MAX_EPISODES = 100000
# 每隔多少个回合训练一次网络
TRAIN_INTERVAL_EPISODES = 10
# 随机种子
RANDOM_SEED = AGENTPARA.RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """设置所有相关的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            # 红方
            red_obs = observations['red_agent']
            red_env_action, red_action_to_store, red_prob = red_agent.choose_action(red_obs)
            red_value = red_agent.get_value(red_obs).cpu().detach().numpy()

            # 蓝方
            blue_obs = observations['blue_agent']
            # blue_env_action, blue_action_to_store, blue_prob = blue_agent.choose_action(blue_obs)
            # blue_value = blue_agent.get_value(blue_obs).cpu().detach().numpy()
            blue_env_action = get_blue_ai_action(blue_obs, env)

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
            red_train_info = red_agent.learn()
            # print("  - 训练蓝方智能体...")
            # blue_train_info = blue_agent.learn()

            print("--- 训练完成 ---\n")

            # 将训练信息写入 Tensorboard
            for key, value in red_train_info.items():
                writer.add_scalar(f'red_agent/{key}', value, global_step)
            # for key, value in blue_train_info.items():
            #     writer.add_scalar(f'blue_agent/{key}', value, global_step)

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

                writer.add_scalar('win_rate/red', red_win_rate, i_episode + 1)
                writer.add_scalar('win_rate/blue', blue_win_rate, i_episode + 1)

                # 如果红方胜率很高，可以保存模型
                if red_win_rate > 0.8:
                    print(f"红方胜率达到 {red_win_rate:.2%}，保存模型...")
                    red_agent.save(prefix=f"red_win_rate_{int(red_win_rate * 100)}_ep{i_episode + 1}")
                    # blue_agent.save(prefix=f"blue_opponent_ep{i_episode + 1}")  # 同时保存对手的模型

                # 重置计数器
                red_win_count, blue_win_count, draw_count = 0, 0, 0

    # 训练结束后关闭 writer
    writer.close()
    print("训练完成！")