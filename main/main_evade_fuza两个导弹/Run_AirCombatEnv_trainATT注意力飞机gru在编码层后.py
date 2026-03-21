import random
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

# ------------------- 导入模型和配置 -------------------
# 保持你原有的导入路径不变
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_ATTMLP注意力GRU注意力后yakebi修正优势归一化2 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttngru import *
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体2 import *

# ------------------- 全局配置 -------------------
LOAD_ABLE = False
USE_RNN_MODEL = True  # 保持开启 RNN
TACVIEW_ENABLED_DURING_TRAINING = False

# <<< 新增配置: 每隔多少个回合更新一次 >>>
UPDATE_CYCLE = 50 #25


# def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
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
model_type_str = "交叉注意力ATT_GRU" if USE_RNN_MODEL else "交叉注意力ATT_MLP"
writer_log_dir = f'../../log/log_evade_fuza两个导弹/PPO_{model_type_str}_EP_UPDATE_{time.strftime("%Y-%m-%d_%H-%M-%S")}_seed{AGENTPARA.RANDOM_SEED}'
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

agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path, use_rnn=USE_RNN_MODEL)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 20000 #20000
MAX_STEP = 10000

eval_reward_buffer = []
eval_success_buffer =[]  # <<< 新增：用于存储评估回合的胜负记录
# <<< 新增：用于存储过去 100 次评估结果的 Buffer
eval_reward_buffer_50 = []
eval_success_buffer_50 = []
eval_counter = 0  # 计数器，专门用来控制测试的种子

for i_episode in range(MAX_EXE_NUM):
    # --- 1. 重置环境 ---
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    # [RNN关键点] 每个 Episode 开始必须重置 RNN 状态
    if USE_RNN_MODEL:
        agent.reset_rnn_state()

    if np.isnan(observation).any():
        print(f"!!! NaN Error in reset at Ep {i_episode} !!!")
        break

    done = False
    step = 0
    episode_reward = 0

    # =================================================================
    #                   Step 循环 (只负责收集数据)
    # =================================================================
    for t in range(MAX_STEP):
        agent.prep_eval_rl()

        # 1. 获取动作
        with torch.no_grad():
            env_action_flat, action_to_store, prob, value, attn_weights = agent.choose_action(observation)
            state_to_store = observation

        # 2. 执行动作
        action_dict = pack_action_into_dict(env_action_flat, attn_weights)

        current_alpha = 1.0  # 或者根据 global_step 动态调整
        env.reward_calculator.set_attention_blending_alpha(current_alpha)

        observation, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated
        episode_reward += reward

        # 3. 存储经验
        agent.store_experience(state_to_store, action_to_store, prob, value, reward, done, attn_weights=attn_weights)

        global_step += 1
        step += 1

        if done:
            break

    # =================================================================
    #                   Episode 结束处理
    # =================================================================

    # --- 回合日志 ---
    print(f"Episode {i_episode + 1} Finished | Total Steps: {step} | Reward: {episode_reward:.2f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)

    if "success" in info and info['success']:
        success_num += 1

    # --- 成功率统计 ---
    if (i_episode + 1) % 100 == 0:
        success_rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合成功率: {success_rate * 100:.2f}%")
        writer.add_scalar('Episode/Success_Rate_per_100_ep', success_rate, i_episode)
        if success_rate >= 0.90:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")
        success_num = 0

    # =================================================================
    #             <<< 核心修改：每隔 UPDATE_CYCLE 回合进行训练 >>>
    # =================================================================
    if (i_episode + 1) % UPDATE_CYCLE == 0:

        # --- 1. 准备训练 ---
        print(f"\n--- [Episode {i_episode + 1}] 达到更新周期，开始训练... ---")
        agent.prep_training_rl()

        # --- 2. 计算 Next Value (Bootstrapping) ---
        # 我们需要这批数据最后一个状态的价值，用于计算优势函数（GAE）
        # 如果最后一个回合是因为 done (撞击/被击落/成功) 结束，其实这个值会被 mask 掉，不影响
        # 如果是因为 truncated (超时) 结束，这个值很重要

        last_observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=CRITIC_PARA.device)

        with torch.no_grad():
            if last_observation_tensor.dim() == 1:
                last_observation_tensor = last_observation_tensor.unsqueeze(0)

            if USE_RNN_MODEL:
                # [RNN关键点] 使用当前的 Critic RNN 状态 (即本回合最后一刻的状态)
                estimated_next_value, _ = agent.Critic(last_observation_tensor, agent.critic_rnn_state)
                estimated_next_value = estimated_next_value.cpu().numpy().item()
            else:
                estimated_next_value = agent.Critic(last_observation_tensor).cpu().numpy().item()

        # --- 3. 执行更新 ---
        train_info = agent.learn(next_visual_value=estimated_next_value)

        if train_info:
            for key, value in train_info.items():
                writer.add_scalar(f"Train/{key}", value, global_step)
            print(f"--- 训练完成 (Steps: {global_step}) ---")
        else:
            print("--- 警告: 数据不足或训练未执行 ---")

        # --- 4. 训练后评估 (Evaluation) ---
        print(f">>> 开始评估 (Episode {i_episode + 1}) <<<")
        agent.prep_eval_rl()

        NUM_EVAL_EPISODES = 1 #2 #5  # <<< 修改这里：每次测试跑 5 个回合（或者你想设置的数字）

        with torch.no_grad():

            eval_rewards_this_test = []
            eval_successes_this_test = []
            # eval_obs, _ = env.reset(seed=AGENTPARA.RANDOM_SEED)  # 固定种子评估

            # =========================================================
            # <<< 核心修改开始 >>> 设置互不重复的测试种子
            # =========================================================

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
                # [RNN关键点] 评估时必须重置 RNN
                if USE_RNN_MODEL:
                    agent.reset_rnn_state()

                # eval_reward_sum = 0
                # eval_success = False  # <<< 新增：初始化当前评估回合的胜负状态为 False
                eval_reward_sum_single = 0
                eval_success_single = False

                for _ in range(MAX_STEP):
                    # 确定性策略
                    eval_action_flat, _, _, _, eval_attn_weights = agent.choose_action(eval_obs, deterministic=True)

                    eval_action_dict = pack_action_into_dict(eval_action_flat, eval_attn_weights)

                    eval_obs, eval_reward, eval_terminated, eval_truncated,  eval_info = eval_env.step(eval_action_dict)
                    eval_done = eval_terminated or eval_truncated

                    # eval_reward_sum += eval_reward
                    eval_reward_sum_single += eval_reward

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
            #
            # # # 记录单次评估结果
            # # print(f"评估回合奖励: {eval_reward_sum:.2f} | 是否胜利: {'是' if eval_success else '否'}")
            # # writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)
            # # writer.add_scalar('Eval/Success_Single', int(eval_success), global_step)  # 可选：记录单次胜负(0或1)
            # #
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

        print("--- 准备开始下一轮采集 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)