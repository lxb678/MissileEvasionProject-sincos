# 文件: Run_AirCombatEnv_train.py

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time

# --- 导入配置和新的离散 PPO 类 ---
# 请根据实际路径调整 import
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.PPOMLP混合架构.Hybrid_PPO_混合架构雅可比修正优势归一化 import \
    PPO_discrete, DISCRETE_DIMS
from Interference_code.env.missile_evasion_environment_jsbsim_fuza_单一干扰.Vec_missile_evasion_environment_jsbsim2 import \
    AirCombatEnv

LOAD_ABLE = False  # 是否使用save文件夹中的模型
TACVIEW_ENABLED_DURING_TRAINING = False  # 训练时通常关闭以提高速度


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置全栈随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)
    print(f"[Info] Seed set to {seed}")


def pack_action_into_dict(discrete_indices: np.ndarray) -> dict:
    """
    将 PPO 输出的离散索引数组包装成 Environment 需要的字典格式
    Args:
        discrete_indices: [trigger_idx, salvo_idx, groups_idx, interval_idx]
    """
    # 确保是整数类型
    discrete_part = discrete_indices.astype(int)

    # 只需要 discrete_actions 键 (连续动作由PID接管)
    action_dict = {
        "discrete_actions": discrete_part
    }
    return action_dict


# --- 初始化 ---
# Tensorboard 设置
writer_log_dir = f'../../log/log_evade_fuza_单一干扰/PPO_Discrete_{time.strftime("%Y-%m-%d_%H-%M-%S")}_seed{AGENTPARA.RANDOM_SEED}'
writer = SummaryWriter(log_dir=writer_log_dir)
print(f"Tensorboard 日志将保存在: {writer_log_dir}")

# 环境初始化
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING, dt=0.02)
set_seed(env)

# Agent 初始化 (使用 PPO_discrete)
model_load_path = r"你的模型路径" if LOAD_ABLE else None
if LOAD_ABLE:
    print(f"--- 正在加载预训练模型: {model_load_path} ---")

agent = PPO_discrete(load_able=LOAD_ABLE, model_dir_path=model_load_path)

# --- 训练循环变量 ---
global_step = 0
success_num = 0
MAX_EXE_NUM = 20000  # 最大训练回合数
MAX_STEP = 10000  # 每回合最大步数
UPDATE_CYCLE = 5 #10  # 每多少回合训练一次

eval_reward_buffer = []

for i_episode in range(MAX_EXE_NUM):
    # ========================== 1. 训练数据收集阶段 ==========================
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    if np.isnan(observation).any():
        print("!!! 严重错误: reset() 返回了 NaN! 退出训练。")
        break

    episode_reward = 0
    done = False
    step = 0

    # 经验收集循环
    for t in range(MAX_STEP):
        agent.prep_eval_rl()  # 采样时使用评估模式 (但不确定性采样)

        with torch.no_grad():
            # a. 获取动作索引 (随机采样)
            action_indices, action_to_store, log_prob = agent.choose_action(observation, deterministic=False)

            # b. 获取价值
            value = agent.get_value(observation).cpu().detach().numpy()
            state_to_store = observation

        # c. 打包动作并执行
        action_dict = pack_action_into_dict(action_indices)
        next_obs, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        # d. 存储经验 (注意：存入的是纯离散索引)
        agent.store_experience(state_to_store, action_to_store, log_prob, value, reward, done)

        episode_reward += reward
        observation = next_obs
        global_step += 1
        step += 1

        if done: break

    # --- 训练回合日志 ---
    print(f"Episode {i_episode + 1} | Steps: {step} | SimTime: {env.t_now:.2f}s | Reward: {episode_reward:.2f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)

    # 统计成功率
    if info.get('success', False):
        success_num += 1

    # 每100回合打印并记录成功率
    if (i_episode + 1) % 100 == 0:
        rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合 (Ep {i_episode - 98}-{i_episode + 1}) 统计:")
        print(f"  - 成功率: {rate * 100:.1f}%")
        print("-" * 50)
        writer.add_scalar('Episode/Success_Rate', rate, i_episode)

        # 保存高成功率模型
        if rate >= 0.90:
            agent.save(prefix=f"Succ_{int(rate * 100)}")

        success_num = 0

    # ========================== 2. 模型更新与评估阶段 ==========================
    if (i_episode + 1) % UPDATE_CYCLE == 0:
        print(f"\n--- [Episode {i_episode + 1}] 开始训练 | Global Steps: {global_step} ---")

        # --- A. 训练 ---
        agent.prep_training_rl()
        train_info = agent.learn()
        for k, v in train_info.items():
            writer.add_scalar(f"Train/{k}", v, global_step)
        print("--- 训练结束 ---")

        # --- B. 评估 (使用独立种子) ---
        print(f"--- [Episode {i_episode + 1}] 开始评估 ---")
        agent.prep_eval_rl()

        # 使用独立的测试种子，确保评估环境与训练环境隔离
        test_seed = AGENTPARA.RANDOM_SEED + 100000 + i_episode
        eval_obs, _ = env.reset(seed=test_seed)
        print(f"    (使用测试种子: {test_seed})")

        eval_reward_sum = 0
        eval_step = 0  # 显式计数
        eval_done = False

        for _ in range(MAX_STEP):
            with torch.no_grad():
                # 评估时使用确定性策略 (deterministic=True)
                act_idx, _, _ = agent.choose_action(eval_obs, deterministic=True)

            # 执行动作
            eval_obs, r, term, trunc, _ = env.step(pack_action_into_dict(act_idx))

            eval_reward_sum += r
            eval_step += 1

            if term or trunc:
                eval_done = True
                break

        # <<< 关键修复：立即打印本次评估的详细结果 >>>
        # 这样您就能看到每回合的步数、时间和奖励了
        print(f"    -> 评估结果 | Steps: {eval_step} | SimTime: {env.t_now:.2f}s | Reward: {eval_reward_sum:.2f}")

        # 记录单次评估奖励
        writer.add_scalar('Eval/Reward_Single', eval_reward_sum, global_step)

        # 存入 buffer 计算平均值 (用于平滑曲线)
        eval_reward_buffer.append(eval_reward_sum)
        if len(eval_reward_buffer) >= 10:
            mean_reward = np.mean(eval_reward_buffer)
            print(f"\n{'#' * 40}")
            print(f"### 过去10次评估平均奖励: {mean_reward:.2f} ###")
            print(f"{'#' * 40}\n")
            writer.add_scalar('Eval/Mean_10_Reward', mean_reward, global_step)
            eval_reward_buffer = []  # 清空 buffer

        print("--- 评估结束 ---\n")

writer.close()