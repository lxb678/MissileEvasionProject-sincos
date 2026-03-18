# --- START OF FILE Run_AirCombatEnv_trainATTMLP实体.py (仅Attention+MLP版) ---

import random
from typing import Optional

# <<< 核心修改 >>>: 导入仅包含 Attention+MLP 的 PPO 模型
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_ATTMLP注意力雅可比修正优势归一化 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from torch.utils.tensorboard import SummaryWriter
# 导入环境 (保持不变)
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体2 import *
import time

LOAD_ABLE = False  # 是否加载预训练模型
# <<< 核心修改 >>>: 关闭 RNN 模式
USE_RNN_MODEL = False

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TRAINING = False


# def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
#     ''' 设置随机种子 '''
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if hasattr(env, 'reset'):
#         env.reset(seed=seed)

def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置全栈随机种子 '''
    # 1. Python 原生 & Numpy (环境的核心)
    random.seed(seed)
    np.random.seed(seed)

    # 2. PyTorch (智能体的核心)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多张卡，全部设置

        # (可选) 强制 CuDNN 使用确定性算法
        # 这会让训练变慢一点点，但能保证完全复现
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 3. Gym 环境 (初始化的核心)
    if hasattr(env, 'action_space'):
        env.action_space.seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)

    print(f"[Info] Random seed set to {seed}")


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
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING,dt = 0.05)
set_seed(env)
eval_env = AirCombatEnv(tacview_enabled=False, dt = 0.05) # <<< 新增：专属的评估环境，步长设为0.02
set_seed(eval_env) # <<< 新增：为评估环境也设置初始种子

model_load_path = None
if LOAD_ABLE:
    model_load_path = r"path/to/your/pretrained/model"
    print(f"--- 正在加载预训练模型: {model_load_path} ---")

# <<< 核心修改 >>>: 初始化 Agent 时传入 use_rnn=False
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path, use_rnn=USE_RNN_MODEL)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 20000 #20000
MAX_STEP = 10000
UPDATE_CYCLE = 25 # 建议使用稍大的更新周期

eval_reward_buffer = []
eval_success_buffer =[]  # <<< 新增：用于存储评估回合的胜负记录
# <<< 新增：用于存储过去 100 次评估结果的 Buffer
eval_reward_buffer_50 = []
eval_success_buffer_50 = []
eval_counter = 0  # 计数器，专门用来控制测试的种子

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
        writer.add_scalar('Episode/Success_Rate_per_100_ep', success_rate, i_episode)

        if success_rate >= 0.90:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")
        success_num = 0

    # --- 2. 训练阶段 ---
    if (i_episode + 1) % UPDATE_CYCLE == 0:
        # if agent.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
        #     print(f"\n--- [Episode {i_episode + 1}] 数据不足，跳过本次训练 ---")
        #     continue

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

        NUM_EVAL_EPISODES = 1 #2 #5  # <<< 修改这里：每次测试跑 5 个回合（或者你想设置的数字）

        with torch.no_grad():
            # eval_obs, _ = env.reset(seed=AGENTPARA.RANDOM_SEED)
            # =========================================================
            # <<< 核心修改开始 >>> 设置互不重复的测试种子
            # =========================================================

            eval_rewards_this_test = []
            eval_successes_this_test = []

            # 定义一个巨大的偏移量，比如 100,000 (确保大于你的 MAX_EXE_NUM)
            TEST_SEED_OFFSET = 100000 #MAX_EXE_NUM

            # -----------------------------------------------------
            # 1. 执行一次包含 N 个回合的测试
            # -----------------------------------------------------
            for eval_idx in range(NUM_EVAL_EPISODES):
                current_test_seed = AGENTPARA.RANDOM_SEED + TEST_SEED_OFFSET + i_episode * NUM_EVAL_EPISODES + eval_idx

                # # [测试种子] = 基础种子 + 偏移量 + 当前回合数
                # # 例如：1000 + 100000 + 10 = 101010
                # # 下一次测试是：1000 + 100000 + 20 = 101020
                # # 结果：永远不重复，且跟训练集完全隔离
                # current_test_seed = AGENTPARA.RANDOM_SEED + TEST_SEED_OFFSET + i_episode

                # 显式传入计算好的种子
                eval_obs, _ = eval_env.reset(seed=current_test_seed)

                print(f"    (使用测试种子: {current_test_seed})")  # 打印出来方便检查

                # =========================================================
                # <<< 核心修改结束 >>>
                # =========================================================
                # <<< 核心修改 >>>: 移除评估循环中的 hidden_state 管理
                # if USE_RNN_MODEL:
                #     eval_actor_hidden, eval_critic_hidden = agent.get_initial_hidden_states()

                # eval_reward_sum = 0
                # eval_success = False  # <<< 新增：初始化当前评估回合的胜负状态为 False
                eval_reward_sum_single = 0
                eval_success_single = False

                for _ in range(MAX_STEP):
                    # a. 获取动作和注意力权重
                    eval_action_flat, _, _, _, eval_attn_weights = agent.choose_action(eval_obs, deterministic=True)

                    # b. 打包动作
                    eval_action_dict = pack_action_into_dict(eval_action_flat, eval_attn_weights)

                    # c. 与环境交互
                    eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action_dict)
                    eval_done = eval_terminated or eval_truncated

                    # eval_reward_sum += eval_reward
                    eval_reward_sum_single += eval_reward

                    # <<< 核心修改 >>>: 移除 hidden_state 更新
                    # if USE_RNN_MODEL:
                    #     eval_actor_hidden = new_eval_actor_hidden
                    #     eval_critic_hidden = new_eval_critic_hidden

                    # <<< 新增：如果在评估步中出现了成功标志，则记录为成功
                    if "success" in eval_info and eval_info['success']:
                        # eval_success = True
                        eval_success_single = True

                    if eval_done:
                        break
                eval_rewards_this_test.append(eval_reward_sum_single)
                eval_successes_this_test.append(1.0 if eval_success_single else 0.0)

            # 计算【这一次测试（包含 N 个回合）】的平均结果
            mean_reward_this_test = np.mean(eval_rewards_this_test)
            mean_success_this_test = np.mean(eval_successes_this_test)

            print(
                f"    评估结束 | 本次测试({NUM_EVAL_EPISODES}局) 平均奖励: {mean_reward_this_test:.2f} | 平均胜率: {mean_success_this_test * 100:.2f}%")

            # 记录【这一次测试】的平均结果 (保留你原代码的命名)
            writer.add_scalar('Eval/Reward_Sum', mean_reward_this_test, global_step)
            writer.add_scalar('Eval/Success_Single', mean_success_this_test, global_step)
            # # -----------------------------------------------------
            # # 2. 将本次测试结果存入 Buffer，用于计算 10 次测试的长线平均
            # # -----------------------------------------------------
            # eval_reward_buffer.append(mean_reward_this_test)
            # eval_success_buffer.append(mean_success_this_test)
            # # # <<< 新增：同时将数据存入 100 次的缓存中
            # # eval_reward_buffer_50.append(mean_reward_this_test)
            # # eval_success_buffer_50.append(mean_success_this_test)
            #     # # 记录单次评估结果
            # # print(f"评估回合奖励: {eval_reward_sum:.2f} | 是否胜利: {'是' if eval_success else '否'}")
            # # writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)
            # # writer.add_scalar('Eval/Success_Single', int(eval_success), global_step)  # 可选：记录单次胜负(0或1)
            # # # 将奖励和胜负情况存入 buffer
            # # eval_reward_buffer.append(eval_reward_sum)
            # # eval_success_buffer.append(1.0 if eval_success else 0.0)  # <<< 新增：胜利存1.0，失败存0.0
            # # --- 核心修改：判断是否攒够了 10 个 ---
            # if len(eval_reward_buffer) >= 5:
            #     mean_reward = np.mean(eval_reward_buffer)
            #     mean_success_rate = np.mean(eval_success_buffer)  # <<< 新增：计算过去10次测试的胜率
            #     print(f"\n{'#' * 40}")
            #     print(f"### 统计报告: 过去5次测试平均奖励: {mean_reward:.2f} ###")
            #     print(f"### 统计报告: 过去5次测试平均胜率: {mean_success_rate * 100:.2f}% ###")
            #     print(f"{'#' * 40}\n")
            #
            #     # 记录到 Tensorboard，使用的是'Eval/Mean_10_Buffer'
            #     writer.add_scalar('Eval/Mean_5_Buffer', mean_reward, global_step)
            #     writer.add_scalar('Eval/Mean_5_Success_Rate', mean_success_rate,
            #                       global_step)  # <<< 新增：将测试胜率写入Tensorboard
            #
            #     # 清空缓存，准备下一轮积累
            #     eval_reward_buffer = []
            #     eval_success_buffer = []  # <<< 新增：清空胜负记录缓存

            # # =========================================================
            # # <<< 新增：100 次测试的统计逻辑
            # # =========================================================
            # if len(eval_reward_buffer_50) >= 20:
            #     mean_50_reward = np.mean(eval_reward_buffer_50)
            #     mean_50_success_rate = np.mean(eval_success_buffer_50)
            #
            #     print(f"\n{'=' * 40}")
            #     print(f"=== 统计报告: 过去50次测试阶段 平均奖励: {mean_50_reward:.2f} ===")
            #     print(f"=== 统计报告: 过去50次测试阶段 平均胜率: {mean_50_success_rate * 100:.2f}% ===")
            #     print(f"{'=' * 40}\n")
            #
            #     # 记录到 Tensorboard，方便观察大趋势
            #     writer.add_scalar('Eval/Mean_20_Buffer', mean_50_reward, global_step)
            #     writer.add_scalar('Eval/Mean_20_Success_Rate', mean_50_success_rate, global_step)
            #
            #     # 清空缓存
            #     eval_reward_buffer_50 = []
            #     eval_success_buffer_50 = []
        print("--- 评估结束 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)