# --- START OF FILE Run_AirCombatEnv_trainATTMLP实体.py (仅Attention+MLP版) ---

import random
from typing import Optional

# <<< 核心修改 >>>: 导入仅包含 Attention+MLP 的 PPO 模型
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.旧代码.Hybrid_PPO_ATTMLP交叉注意力2_1120 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from torch.utils.tensorboard import SummaryWriter
# 导入环境 (保持不变)
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体 import *
import time

LOAD_ABLE = False  # 是否加载预训练模型
# <<< 核心修改 >>>: 关闭 RNN 模式
USE_RNN_MODEL = False

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TRAINING = False


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)

# pack_action_into_dict 函数保持不变，因为它已经能够处理 attn_weights
def pack_action_into_dict(flat_action_np: np.ndarray, attn_weights: Optional[np.ndarray] = None) -> dict:
    continuous_part = flat_action_np[:CONTINUOUS_DIM]
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)
    action_dict = {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part
    }
    if attn_weights is not None:
        action_dict["attention_weights"] = attn_weights
    return action_dict


# ------------------- Tensorboard 设置 -------------------
model_type_str = "交叉注意力ATT_GRU" if USE_RNN_MODEL else "交叉注意力ATT_MLP" # 修改日志标识
writer_log_dir = f'../../log/log_evade_fuza两个导弹/PPO_{model_type_str}_{time.strftime("%Y-%m-%d_%H-%M-%S")}_seed{AGENTPARA.RANDOM_SEED}_load{LOAD_ABLE}'
writer = SummaryWriter(log_dir=writer_log_dir)
print(f"Tensorboard 日志将保存在: {writer_log_dir}")

# ------------------- 环境和智能体初始化 -------------------
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)
set_seed(env)

model_load_path = None
if LOAD_ABLE:
    model_load_path = r"path/to/your/pretrained/model"
    print(f"--- 正在加载预训练模型: {model_load_path} ---")

# <<< 核心修改 >>>: 初始化 Agent 时传入 use_rnn=False
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path, use_rnn=USE_RNN_MODEL)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 100000
MAX_STEP = 10000
UPDATE_CYCLE = 10 # 建议使用稍大的更新周期

# alpha 更新参数 (保持不变)
TOTAL_TRAINING_STEPS_FOR_ALPHA = 500 * UPDATE_CYCLE * 20

for i_episode in range(MAX_EXE_NUM):
    # --- 1. 经验收集阶段 ---
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    if np.isnan(observation).any():
        print(f"!!! 严重错误: 第 {i_episode + 1} 回合 reset() 返回了 NaN! 退出训练。")
        break

    # <<< 核心修改 >>>: 移除所有与 hidden_state 相关的代码
    # if USE_RNN_MODEL:
    #     actor_hidden, critic_hidden = agent.get_initial_hidden_states()
    # else:
    #     actor_hidden, critic_hidden = None, None

    done = False
    step = 0
    episode_reward = 0

    for t in range(MAX_STEP):
        agent.prep_eval_rl()
        with torch.no_grad():
            # <<< 核心修改 >>>: choose_action 调用简化，不再处理 hidden_state
            # a. 获取动作、价值和注意力权重
            env_action_flat, action_to_store, prob, value, attn_weights = agent.choose_action(observation)

            # b. 记录当前状态
            state_to_store = observation

        # c. 将动作和权重打包成字典
        action_dict = pack_action_into_dict(env_action_flat, attn_weights)

        # d. 动态更新奖励函数的 alpha 值
        # current_alpha = min(1.0, global_step / TOTAL_TRAINING_STEPS_FOR_ALPHA)
        current_alpha = 1.0
        env.reward_calculator.set_attention_blending_alpha(current_alpha)

        # e. 与环境交互
        observation, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        episode_reward += reward

        # f. <<< 核心修改 >>>: store_experience 调用简化
        agent.store_experience(state_to_store, action_to_store, prob, value, reward, done, attn_weights=attn_weights)

        # <<< 核心修改 >>>: 移除 hidden_state 更新
        # if USE_RNN_MODEL:
        #     actor_hidden = new_actor_hidden
        #     critic_hidden = new_critic_hidden

        global_step += 1
        step += 1

        if done:
            break

    # --- 回合结束后的日志记录 (保持不变) ---
    print(
        f"Episode {i_episode + 1} | Steps: {step} | SimTime: {env.t_now:.2f}s | Reward: {episode_reward:.2f} | Alpha: {current_alpha:.3f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)
    # writer.add_scalar('Metrics/Attention_Alpha', current_alpha, global_step)

    if "success" in info and info['success']:
        success_num += 1

    if (i_episode + 1) % 100 == 0:
        success_rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合 (Ep {i_episode - 99 + 1}-{i_episode + 1}) 统计:")
        print(f"  - 成功率: {success_rate * 100:.2f}% ({success_num}/100)")
        print("-" * 50)
        writer.add_scalar('Metrics/Success_Rate_per_100_ep', success_rate, i_episode)

        if success_rate >= 0.95:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")
        success_num = 0

    # --- 2. 训练阶段 ---
    if (i_episode + 1) % UPDATE_CYCLE == 0:
        if agent.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            print(f"\n--- [Episode {i_episode + 1}] 数据不足，跳过本次训练 ---")
            continue

        print(f"\n--- [Episode {i_episode + 1}] 开始训练 | Global Steps: {global_step} ---")
        agent.prep_training_rl()
        train_info = agent.learn()

        if train_info:
            for key, value in train_info.items():
                writer.add_scalar(f"Train/{key}", value, global_step)
            print("--- 训练结束 ---\n")
        else:
            print("--- 训练因数据不足而跳过 ---\n")

        # --- 3. 评估阶段 ---
        print(f"--- [Episode {i_episode + 1}] 开始评估 ---")
        agent.prep_eval_rl()

        with torch.no_grad():
            eval_obs, _ = env.reset(seed=AGENTPARA.RANDOM_SEED)

            # <<< 核心修改 >>>: 移除评估循环中的 hidden_state 管理
            # if USE_RNN_MODEL:
            #     eval_actor_hidden, eval_critic_hidden = agent.get_initial_hidden_states()

            eval_reward_sum = 0
            for _ in range(MAX_STEP):
                # a. 获取动作和注意力权重
                eval_action_flat, _, _, _, eval_attn_weights = agent.choose_action(eval_obs, deterministic=True)

                # b. 打包动作
                eval_action_dict = pack_action_into_dict(eval_action_flat, eval_attn_weights)

                # c. 与环境交互
                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = env.step(eval_action_dict)
                eval_done = eval_terminated or eval_truncated

                eval_reward_sum += eval_reward

                # <<< 核心修改 >>>: 移除 hidden_state 更新
                # if USE_RNN_MODEL:
                #     eval_actor_hidden = new_eval_actor_hidden
                #     eval_critic_hidden = new_eval_critic_hidden

                if eval_done:
                    break

            print(f"评估回合奖励: {eval_reward_sum:.2f}")
            writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)
        print("--- 评估结束 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)