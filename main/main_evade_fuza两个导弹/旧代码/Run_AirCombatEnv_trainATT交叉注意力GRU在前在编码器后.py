import random
from typing import Optional

# 导入模型和配置
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.旧代码.Hybrid_PPO_ATTMLP交叉注意力GRU在前在编码器后 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from torch.utils.tensorboard import SummaryWriter
# 导入环境
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体 import *
import time

LOAD_ABLE = False
# <<< 修改 1: 启用 RNN >>>
USE_RNN_MODEL = True
TACVIEW_ENABLED_DURING_TRAINING = False


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)


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

agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path, use_rnn=USE_RNN_MODEL)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 100000
MAX_STEP = 10000

# 触发训练的经验阈值
UPDATE_STEP_THRESHOLD = 1024
TOTAL_TRAINING_STEPS_FOR_ALPHA = 100000

for i_episode in range(MAX_EXE_NUM):
    # --- 1. 重置环境 ---
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    # <<< 修改 2: 每个 Episode 开始必须重置 RNN 状态 >>>
    if USE_RNN_MODEL:
        agent.reset_rnn_state()

    if np.isnan(observation).any():
        print(f"!!! NaN Error in reset at Ep {i_episode} !!!")
        break

    done = False
    step = 0
    episode_reward = 0

    has_trained_in_this_episode = False

    # =================================================================
    #                   Step 循环
    # =================================================================
    for t in range(MAX_STEP):
        agent.prep_eval_rl()

        # 1. 获取动作
        with torch.no_grad():
            env_action_flat, action_to_store, prob, value, attn_weights = agent.choose_action(observation)
            state_to_store = observation

        # 2. 执行动作
        action_dict = pack_action_into_dict(env_action_flat, attn_weights)

        current_alpha = 1.0
        env.reward_calculator.set_attention_blending_alpha(current_alpha)

        observation, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated
        episode_reward += reward

        # 3. 存储经验
        agent.store_experience(state_to_store, action_to_store, prob, value, reward, done, attn_weights=attn_weights)

        global_step += 1
        step += 1

        # =================================================================
        # <<< 核心修改：满了就更新，更新完标记一下，继续跑 >>>
        # =================================================================
        if agent.buffer.get_buffer_size() >= UPDATE_STEP_THRESHOLD:
            print(f"\n--- [Episode {i_episode + 1} Step {step}] Buffer满，暂停训练... ---")

            agent.prep_training_rl()

            # 计算截断价值 (Bootstrapping)
            last_observation = observation
            last_observation_tensor = torch.as_tensor(last_observation, dtype=torch.float32,
                                                      device=CRITIC_PARA.device)

            with torch.no_grad():
                if last_observation_tensor.dim() == 1:
                    last_observation_tensor = last_observation_tensor.unsqueeze(0)

                # <<< 修改 3: Critic 调用适配 RNN >>>
                if USE_RNN_MODEL:
                    # 使用当前的 Critic 状态进行估值
                    estimated_next_value, _ = agent.Critic(last_observation_tensor, agent.critic_rnn_state)
                    estimated_next_value = estimated_next_value.cpu().numpy().item()
                else:
                    estimated_next_value = agent.Critic(last_observation_tensor).cpu().numpy().item()

            # 执行训练
            train_info = agent.learn(next_visual_value=estimated_next_value)

            if train_info:
                for key, value in train_info.items():
                    writer.add_scalar(f"Train/{key}", value, global_step)

            print(f"--- 训练完成，继续完成当前 Episode {i_episode + 1} ---")

            # 训练后 RNN 状态可能会断裂（因为 Buffer 不连续），但我们选择保留当前状态继续跑完这局
            # 如果你想要极度精确，这里其实很难处理，但保留状态继续跑是业界通用做法。

            has_trained_in_this_episode = True
            agent.prep_eval_rl()

        # =================================================================

        if done:
            break

    # --- 回合结束日志 ---
    print(f"Episode {i_episode + 1} Finished | Total Steps: {step} | Reward: {episode_reward:.2f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)

    if "success" in info and info['success']:
        success_num += 1

    if (i_episode + 1) % 100 == 0:
        success_rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合成功率: {success_rate * 100:.2f}%")
        writer.add_scalar('Metrics/Success_Rate_per_100_ep', success_rate, i_episode)
        if success_rate >= 0.95:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")
        success_num = 0

    # =================================================================
    # <<< 核心逻辑：若本回合发生过训练，则在回合彻底结束后进行一次评估 >>>
    # =================================================================

    if has_trained_in_this_episode:

        print(f"\n>>> [Post-Episode Eval] 检测到本回合发生了模型更新，开始评估... <<<")
        agent.prep_eval_rl()

        with torch.no_grad():
            eval_obs, _ = env.reset(seed=AGENTPARA.RANDOM_SEED)

            # <<< 修改 2 (复用): 评估必须重置 RNN >>>
            if USE_RNN_MODEL:
                agent.reset_rnn_state()

            eval_reward_sum = 0
            for _ in range(MAX_STEP):
                # a. 确定性策略
                eval_action_flat, _, _, _, eval_attn_weights = agent.choose_action(eval_obs, deterministic=True)

                # b. 打包
                eval_action_dict = pack_action_into_dict(eval_action_flat, eval_attn_weights)

                # c. 交互
                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = env.step(eval_action_dict)
                eval_done = eval_terminated or eval_truncated

                eval_reward_sum += eval_reward

                if eval_done:
                    break

            print(f">>> 评估结束 | Reward: {eval_reward_sum:.2f}")
            writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)

        print("--- 评估完成，准备开始下一回合 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)