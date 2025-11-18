# --- START OF FILE Run_AirCombatEnv_train.py ---

import random
# from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.Hybrid_PPO_jsbsim import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.旧代码.Hybrid_PPO_ATTGRUMLP混合架构 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from torch.utils.tensorboard import SummaryWriter
# from env.AirCombatEnv import *
from Interference_code.env.missile_evasion_environment_jsbsim_fuza.Vec_missile_evasion_environment_jsbsim import *
import time

LOAD_ABLE = False  # 是否使用save文件夹中的模型
# <<< GRU/RNN 修改 >>>: 新增一个开关来决定是否使用RNN模型
USE_RNN_MODEL = True  # <--- 在这里控制是否启用 GRU

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TRAINING = False


# ---
def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# def set_seed(env, seed=AGENTPARA.RANDOM_SEED, seed_env=True):
#     """
#     设置所有相关库的随机种子以保证实验的可复现性。
#
#     Args:
#         env: 强化学习环境对象。
#         seed (int): 要设置的随机种子。
#         seed_env (bool): 是否为环境本身设置种子。
#     """
#     # 1. 设置 Python, NumPy, PyTorch 的种子
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     # # 2. 设置 PyTorch 在 CUDA (GPU) 上的种子
#     # if torch.cuda.is_available():
#     #     torch.cuda.manual_seed(seed)
#     #     torch.cuda.manual_seed_all(seed)  # 适用于多GPU环境
#     #
#     # # 3. 设置 cuDNN 的行为，以实现确定性 (可能会牺牲性能)
#     # #    在追求严格复现时开启，在常规训练中可以注释掉以获得更好的性能
#     # # torch.backends.cudnn.deterministic = True
#     # # torch.backends.cudnn.benchmark = False
#     #
#     # # 4. 为强化学习环境设置种子
#     # if seed_env and hasattr(env, 'seed'):
#     #     try:
#     #         env.seed(seed)
#     #     except Exception as e:
#     #         print(f"Warning: Failed to seed environment. Error: {e}")
#     #
#     # # （可选）某些较新的 Gym 环境使用 action_space.seed()
#     # if seed_env and hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
#     #     try:
#     #         env.action_space.seed(seed)
#     #     except Exception as e:
#     #         print(f"Warning: Failed to seed environment action_space. Error: {e}")


def pack_action_into_dict(flat_action_np: np.ndarray) -> dict:
    """
    将PPO agent.choose_action返回的扁平化NumPy数组转换为
    环境env.step()所期望的字典格式。
    """
    continuous_part = flat_action_np[:CONTINUOUS_DIM]
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)

    action_dict = {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part
    }
    return action_dict


# ------------------- Tensorboard 设置 -------------------
# <<< GRU/RNN 修改 >>>: 在日志文件名中加入 RNN 标识
model_type_str = "ATTGRU" if USE_RNN_MODEL else "MLP"
writer_log_dir = f'../../log/log_evade_fuza/PPO_{model_type_str}_{time.strftime("%Y-%m-%d_%H-%M-%S")}_seed{AGENTPARA.RANDOM_SEED}_load{LOAD_ABLE}'
writer = SummaryWriter(log_dir=writer_log_dir)
print(f"Tensorboard 日志将保存在: {writer_log_dir}")

# ------------------- 环境和智能体初始化 -------------------
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)
set_seed(env)

model_load_path = None
if LOAD_ABLE:
    model_load_path = r"D:\DESKTOP\毕设\JSBSim_py\JSBSim-master - a\save\save_evade\2024-05-18_09-17-06"
    print(f"--- 正在加载预训练模型: {model_load_path} ---")

# <<< GRU/RNN 修改 >>>: 初始化 Agent 时传入 use_rnn 参数
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path, use_rnn=USE_RNN_MODEL)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 100000
MAX_STEP = 10000
UPDATE_CYCLE = 10

for i_episode in range(MAX_EXE_NUM):
    # --- 1. 经验收集阶段 ---
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    if np.isnan(observation).any():
        print(f"!!! 严重错误: 第 {i_episode + 1} 回合 reset() 返回了 NaN! 退出训练。")
        break

    # <<< GRU/RNN 修改 >>>: 在每个 episode 开始时，获取初始隐藏状态
    if USE_RNN_MODEL:
        actor_hidden, critic_hidden = agent.get_initial_hidden_states()
    else:
        actor_hidden, critic_hidden = None, None  # 对于 MLP 模型，这些是 None

    done = False
    step = 0
    episode_reward = 0

    for t in range(MAX_STEP):
        agent.prep_eval_rl()
        with torch.no_grad():
            # <<< GRU/RNN 修改 >>>: 将隐藏状态传入 choose_action，并接收新的隐藏状态
            # a. 获取动作、价值和新的隐藏状态
            env_action_flat, action_to_store, prob, value, \
                new_actor_hidden, new_critic_hidden = agent.choose_action(observation, actor_hidden, critic_hidden)

            # b. 记录当前状态
            state_to_store = observation

        # c. 将扁平动作打包成字典
        action_dict = pack_action_into_dict(env_action_flat)

        # d. 与环境交互
        observation, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        episode_reward += reward

        # e. 将经验存入Buffer，注意传入的是【做出动作前】的隐藏状态
        agent.store_experience(state_to_store, action_to_store, prob, value, reward, done, actor_hidden, critic_hidden)

        # <<< GRU/RNN 修改 >>>: 更新隐藏状态以备下一个时间步使用
        if USE_RNN_MODEL:
            actor_hidden = new_actor_hidden
            critic_hidden = new_critic_hidden

        global_step += 1
        step += 1

        if done:
            break

    # --- 回合结束后的日志记录 ---
    print(f"Episode {i_episode + 1} | Steps: {step} | SimTime: {env.t_now:.2f}s | Reward: {episode_reward:.2f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)

    if "success" in info and info['success']:
        success_num += 1

    if (i_episode + 1) % 100 == 0:
        success_rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合 (Ep {i_episode - 99}-{i_episode}) 统计:")  # 修正了索引
        print(f"  - 成功率: {success_rate * 100:.2f}% ({success_num}/100)")
        print("-" * 50)
        writer.add_scalar('Metrics/Success_Rate_per_100_ep', success_rate, i_episode)

        if success_rate >= 0.95:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")
        success_num = 0

    # --- 2. 训练阶段 ---
    if (i_episode + 1) % UPDATE_CYCLE == 0:
        # <<< GRU/RNN 修改 >>>: 确保 buffer 中有足够的数据才开始训练
        # 对于 RNN，最好数据量能覆盖几个 batch_size * sequence_length
        # 这里用一个简单的检查
        if agent.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            print(f"\n--- [Episode {i_episode + 1}] 数据不足，跳过本次训练 ---")
            continue

        print(f"\n--- [Episode {i_episode + 1}] 开始训练 | Global Steps: {global_step} ---")
        agent.prep_training_rl()
        train_info = agent.learn()

        if train_info:  # learn() 在数据不足时可能返回 None
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

            # <<< GRU/RNN 修改 >>>: 评估循环也需要管理隐藏状态
            if USE_RNN_MODEL:
                eval_actor_hidden, eval_critic_hidden = agent.get_initial_hidden_states()
            else:
                eval_actor_hidden, eval_critic_hidden = None, None

            eval_reward_sum = 0
            for _ in range(MAX_STEP):
                # a. 获取动作和新的隐藏状态
                eval_action_flat, _, _, _, \
                    new_eval_actor_hidden, new_eval_critic_hidden = agent.choose_action(eval_obs, eval_actor_hidden,
                                                                                        eval_critic_hidden,
                                                                                        deterministic=True)

                # b. 打包动作
                eval_action_dict = pack_action_into_dict(eval_action_flat)

                # c. 与环境交互
                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = env.step(eval_action_dict)
                eval_done = eval_terminated or eval_truncated

                eval_reward_sum += eval_reward

                # <<< GRU/RNN 修改 >>>: 更新评估循环的隐藏状态
                if USE_RNN_MODEL:
                    eval_actor_hidden = new_eval_actor_hidden
                    eval_critic_hidden = new_eval_critic_hidden

                if eval_done:
                    break

            print(f"评估回合奖励: {eval_reward_sum:.2f}")
            writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)
        print("--- 评估结束 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)