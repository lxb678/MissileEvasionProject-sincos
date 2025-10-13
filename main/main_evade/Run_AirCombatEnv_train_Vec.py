# 文件名: Run_AirCombatEnv_train_vectorized.py

import torch
import numpy as np
import random
import time
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

# --- 导入您的自定义模块 ---
# 确保这里的路径是正确的
from Interference_code.PPO_model.旧文件.Hybrid_PPO_jsbsim_Vec import PPO_continuous
from Interference_code.PPO_model.旧文件.Config_Vec import AGENTPARA
from Interference_code.env.missile_evasion_environment_jsbsim.Vec_missile_evasion_environment_jsbsim import AirCombatEnv

# ========================= 配置区 =========================
LOAD_ABLE = False  # 是否从 'save/' 文件夹加载预训练模型
TACVIEW_ENABLED_DURING_TRAINING = False  # 是否为第一个并行环境开启Tacview
# 只有在评估时才可能开启Tacview，训练时关闭以提高效率
TACVIEW_ENABLED_DURING_EVAL = False
NUM_ENVS = 4  # <<<--- 并行环境的数量 ---<<<
RANDOM_SEED = AGENTPARA.RANDOM_SEED
COLLECTION_EPISODES = 10  # 每收集10个回合的数据后，进行一次学习
EVALUATION_EPISODES = 1  # 每次评估时，运行5个回合来计算平均奖励
UPDATE_INTERVAL = 2048   # 每收集 2048 个 step 后更新一次 (可调)

# ========================= 辅助函数 =========================
def set_seed(seed=RANDOM_SEED):
    """ 设置所有相关库的随机种子 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用CUDA，也设置CUDA的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_env(rank: int, seed: int = 0, tacview_enabled: bool = False):
    """
    创建单个环境实例的辅助函数，用于矢量化。
    """

    def _init():
        # 只为第一个环境 (rank=0) 启用 Tacview 以便观察  # 训练时通常关闭Tacview以获得最大速度
        is_tacview_on = tacview_enabled and (rank == 0)
        env = AirCombatEnv(tacview_enabled=is_tacview_on)
        # 为每个环境设置不同的随机种子
        # 注意: gym.Env 的 reset 现在接受 seed 参数
        env.reset(seed=seed + rank)
        return env

    return _init


# ========================= 主执行逻辑 =========================
if __name__ == "__main__":
    # --- 1. 初始化 ---
    set_seed(RANDOM_SEED)

    # 初始化 TensorBoard writer
    log_dir = f'log/Vec_seed{RANDOM_SEED}_time_{time.strftime("%m_%d_%H_%M_%S")}_load_{LOAD_ABLE}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # --- 2. 创建矢量化环境 ---
    print(f"正在创建 {NUM_ENVS} 个并行环境...")
    env_fns = [make_env(i, seed=RANDOM_SEED, tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING) for i in range(NUM_ENVS)]
    vec_env = AsyncVectorEnv(env_fns)
    # vec_env = SyncVectorEnv(env_fns,autoreset_mode = AutoresetMode.DISABLED)

    print("矢量化环境创建成功！")

    # --- 3. 初始化智能体 ---
    agent = PPO_continuous(LOAD_ABLE)

    # --- 4. 主训练循环 ---
    global_step = 0
    episodes_collected = 0
    steps_collected = 0  # ✅ 用步数来控制更新
    total_episodes_trained = 0

    # 用于统计成功率
    success_num = 0
    total_completed_episodes = 0

    # 初始化观测
    observations, infos = vec_env.reset()

    while total_episodes_trained < 100000:

        # --- 4.1 数据收集阶段 ---
        agent.prep_eval_rl()  # 设置为评估模式进行数据收集

        # 收集直到 Buffer 满或者达到指定的回合数
        # 注意：在矢量化环境中，我们通常按步数收集，而不是回合数
        # 但为了保留您的逻辑，我们仍然按回合数来触发学习

        episode_rewards = [0] * NUM_ENVS  # 记录每个并行环境的当前回合奖励

        # while episodes_collected < COLLECTION_EPISODES:
        # ✅ 持续收集，直到累计的环境总步数达到 UPDATE_INTERVAL
        while steps_collected < UPDATE_INTERVAL:
            with torch.no_grad():
                # Agent 根据批量观测选择动作
                env_actions, actions_to_store, log_probs = agent.choose_action(observations, deterministic=False)
                # 获取批量价值估计
                values = agent.get_value(observations).cpu().detach().numpy().flatten()

            # 在矢量化环境中执行动作
            next_observations, rewards, terminateds, truncateds, infos = vec_env.step(env_actions)
            # print("terminateds:", terminateds)
            # print("truncateds:", truncateds)
            # print("infos:", infos)

            # 存储经验
            for i in range(NUM_ENVS):
                agent.buffer.store_transition(
                    state=observations[i],
                    value=values[i],
                    action=actions_to_store[i],
                    probs=log_probs[i],
                    reward=rewards[i],
                    done=(terminateds[i] or truncateds[i])
                )
                episode_rewards[i] += rewards[i]

            global_step += NUM_ENVS  # 每次 step，总步数增加 NUM_ENVS
            steps_collected += NUM_ENVS  # 本轮累计步数
            observations = next_observations

            # 检查是否有环境结束
            # 检查哪些环境结束
            dones = terminateds | truncateds
            # print('dones:', dones)
            if np.any(dones):
                # # 找到结束的环境索引
                # dones_idx = np.where(dones)[0]
                # # 假设 dones_idx 是需要重置的环境索引
                # reset_mask = np.zeros(vec_env.num_envs, dtype=bool)
                # reset_mask[dones_idx] = True

                for i in np.where(dones)[0]:
                    # 默认 success=False
                    success_flag = False

                    # 安全读取 success
                    if 'success' in infos:
                        success_flag = infos['success'][i]

                    # 累计成功次数
                    if success_flag:
                        success_num += 1

                    episodes_collected += 1
                    total_episodes_trained += 1
                    total_completed_episodes += 1

                    print(f"Episode {total_episodes_trained} (Env {i}) finished. "
                          f"Reward: {episode_rewards[i]:.2f}  Success: {success_flag}")

                    # 重置当前环境奖励
                    episode_rewards[i] = 0
#                 # 🔑 手动 reset 已结束的环境，并更新 observations
#                 reset_obs, reset_infos = vec_env.reset(options={"reset_mask": reset_mask}
# )
#                 # reset_obs shape: (len(dones_idx), obs_dim)
#                 # 更新对应位置的观测
#                 observations[dones_idx] = reset_obs

        # --- 4.2 学习阶段 ---
        # print(f"\n--- 收集了 {episodes_collected} 个回合, 开始学习. Global Step: {global_step} ---")
        # ✅ 固定步数收集完成 -> 训练一次
        print(f"\n--- 收集 {steps_collected} 步, 开始学习. Global Step: {global_step} ---")
        agent.prep_training_rl()
        train_info = agent.learn()

        # 记录训练日志
        for key, value in train_info.items():
            writer.add_scalar(f"train/{key}", value, global_step=global_step)

        episodes_collected = 0  # 重置收集计数器
        steps_collected = 0  # 重置累计步数计数器

        # --- 4.3 评估阶段 (使用单个环境) ---
        print(f"--- 开始评估 (单个回合) ---")

        # 创建一个专用于评估的单环境实例
        eval_env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_EVAL)

        agent.prep_eval_rl()
        with torch.no_grad():
            eval_obs, _ = eval_env.reset(seed=RANDOM_SEED + 1000)  # 使用不同的种子进行评估
            eval_done = False
            eval_reward_sum = 0.0

            # 单环境评估循环
            while not eval_done:
                # choose_action 现在接收单个观测 (需要 unsqueeze 添加 batch 维度)
                # action 返回的是批量动作，需要取第一个 [0]
                eval_action, _, _ = agent.choose_action(eval_obs, deterministic=True)

                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)

                eval_reward_sum += eval_reward
                eval_done = eval_terminated or eval_truncated

        writer.add_scalar('reward_sum', eval_reward_sum, global_step=global_step)
        print(f"--- 评估完成. 单回合奖励: {eval_reward_sum:.2f} ---")

        # 关闭评估环境，释放资源
        eval_env.close()


        # --- 4.4 检查成功率和保存模型 ---
        if total_completed_episodes >= 100:
            success_rate = (success_num / total_completed_episodes) * 100
            print(f"--- 过去 {total_completed_episodes} 回合的成功率: {success_rate:.2f}% (成功 {success_num} 次) ---")
            writer.add_scalar('success_num', success_rate, global_step=global_step)

            if success_rate >= 90:
                print(f"*** 成功率达到 {success_rate:.2f}%, 保存模型! ***")
                agent.save(f"success_{int(success_rate)}_ep{total_episodes_trained}")

            # 重置成功率计数器
            success_num = 0
            total_completed_episodes = 0

    # --- 5. 结束训练 ---
    vec_env.close()
    writer.close()
    print("训练结束。")





# # --- 4.3 评估阶段 (修正版) ---
#         print(f"--- 开始评估 (目标: {EVALUATION_EPISODES * NUM_ENVS} 个完整回合) ---")
#         agent.prep_eval_rl()
#
#         # 这个列表将存储所有完成的回合的总奖励
#         completed_episode_rewards = []
#
#         # 目标是收集到足够多的回合数据
#         target_episodes = EVALUATION_EPISODES
#
#         with torch.no_grad():
#             eval_obs, _ = vec_env.reset()
#             # 记录每个并行环境当前正在进行的回合的奖励
#             current_episode_rewards = np.zeros(NUM_ENVS)
#
#             # 循环直到收集到足够的回合
#             while len(completed_episode_rewards) < target_episodes:
#                 # 1. 选择动作
#                 eval_actions, _, _ = agent.choose_action(eval_obs, deterministic=True)
#
#                 # 2. 与环境交互
#                 next_eval_obs, eval_rewards_step, eval_terminateds, eval_truncateds, infos = vec_env.step(eval_actions)
#
#                 # 3. 累加当前回合的奖励
#                 current_episode_rewards += eval_rewards_step
#
#                 # 4. 更新观测
#                 eval_obs = next_eval_obs
#
#                 # 5. 检查是否有环境结束
#                 dones = eval_terminateds | eval_truncateds
#                 if np.any(dones):
#                     for i in range(NUM_ENVS):
#                         if dones[i]:
#                             # 如果一个环境结束了，将其总奖励存入列表
#                             completed_episode_rewards.append(current_episode_rewards[i])
#                             # 并且重置这个环境的当前奖励计数器
#                             current_episode_rewards[i] = 0
#
#         # 计算平均奖励
#         if len(completed_episode_rewards) > 0:
#             avg_eval_reward = np.mean(completed_episode_rewards)
#         else:
#             avg_eval_reward = 0.0  # 如果一个回合都没完成（不太可能）
#
#         writer.add_scalar('eval/reward_sum', avg_eval_reward, global_step=global_step)
#         print(f"--- 评估完成. 在 {len(completed_episode_rewards)} 个回合中, 平均奖励: {avg_eval_reward:.2f} ---")