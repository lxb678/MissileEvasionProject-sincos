import random
from typing import Optional

# ------------------- 导入部分 -------------------
# <<< 核心修改 >>>: 导入 交叉注意力 + GRU 的模型定义
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.旧代码.Hybrid_PPO_ATTMLP交叉注意力GRU在前在编码器后 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
# 导入环境
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体 import *

# ------------------- 配置部分 -------------------
LOAD_ABLE = True  # 测试模式必须为 True
USE_RNN_MODEL = True  # <<< 核心修改 >>>: GRU模式开启
TACVIEW_ENABLED_DURING_TESTING = True  # Tacview 可视化开关


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)


def pack_action_into_dict(flat_action_np: np.ndarray, attn_weights: Optional[np.ndarray] = None) -> dict:
    ''' 将扁平动作解包为环境所需的字典格式 '''
    continuous_part = flat_action_np[:CONTINUOUS_DIM]
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)
    action_dict = {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part
    }
    if attn_weights is not None:
        action_dict["attention_weights"] = attn_weights
    return action_dict


def plot_attention_weights(weights_history, episode_num):
    """
    绘制单个回合的注意力权重变化曲线。
    """
    if not weights_history:
        print("警告: 注意力权重历史为空，无法绘图。")
        return

    # 将列表转换为Numpy数组，方便操作
    weights_np = np.array(weights_history)

    # 防止空数组报错
    if weights_np.ndim < 2:
        return

    # 获取时间步数量
    timesteps = np.arange(weights_np.shape[0])

    plt.figure(figsize=(10, 5))

    # 绘制导弹1的注意力权重曲线 (索引为0)
    plt.plot(timesteps, weights_np[:, 0], label='Attention to Missile 1', color='blue', marker='.', markersize=2,
             linewidth=1)

    # 绘制导弹2的注意力权重曲线 (索引为1)
    plt.plot(timesteps, weights_np[:, 1], label='Attention to Missile 2', color='red', marker='.', markersize=2,
             linewidth=1)

    plt.title(f'Episode {episode_num}: Attention Weights vs Time (GRU Model)')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.ylim(-0.05, 1.05)  # 将Y轴范围固定在0到1之间
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show() # 如果不需要每次阻塞弹窗，可以注释掉这行，或者改为 plt.savefig(...)
    plt.show()


# ------------------- 主程序 -------------------

# 1. 初始化环境
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TESTING)
set_seed(env)

# 2. 指定模型路径
# <<< 请修改这里的路径为你训练好的 GRU 模型文件夹路径 >>>
# model_path = r"D:\code\save\save_evade_fuza两个导弹\PPO_EntityCrossATT_GRU_2025-11-25_16-25-36"
model_path = r"D:\code\save\save_evade_fuza两个导弹\PPO_EntityCrossATT_GRU_2025-11-26_11-31-11"
print(f"--- 正在加载 交叉注意力+GRU 模型: {model_path} ---")

# 3. 初始化 Agent
# 注意: use_rnn=True 会让 Agent 内部初始化 GRU 结构
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_path, use_rnn=USE_RNN_MODEL)

# 切换到评估模式 (关闭 Dropout 等)
agent.prep_eval_rl()
print("模型加载完成，开始测试...")

success_num = 0
episode_times = []
episodes = 100  # 测试回合数

for i_episode in range(episodes):
    episode_start_time = time.time()

    with torch.no_grad():
        observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

        # =================================================================
        # <<< 核心修改 >>>: GRU 模式下，每个回合开始必须重置内部隐状态
        # =================================================================
        if USE_RNN_MODEL:
            agent.reset_rnn_state()
            # print("  [Info] RNN State Reset")

        done_eval = False
        reward_sum = 0
        step = 0

        # 初始化用于存储当前回合权重的列表
        weights_history = []

        while not done_eval:
            # 1. 获取动作
            # GRU 的 hidden state 在 agent 内部维护 (self.actor_rnn_state)，
            # choose_action 会自动读取并更新它，所以这里不需要手动传递 hidden state。
            action_eval_flat, _, _, _, attn_weights = agent.choose_action(observation_eval, deterministic=True)

            # 存储权重用于绘图
            if attn_weights is not None:
                weights_history.append(attn_weights.copy())

            # 2. 打包动作
            action_dict_eval = pack_action_into_dict(action_eval_flat, attn_weights)

            # (可选) 打印调试信息
            # if attn_weights is not None and step % 50 == 0:
            #     print(f"Step {step}, Attn: {attn_weights}")

            # 3. 环境步进
            observation_eval, reward_eval, terminated, truncated, info = env.step(action_dict_eval)

            done_eval = terminated or truncated
            reward_sum += reward_eval
            step += 1

            if done_eval:
                episode_time = time.time() - episode_start_time
                episode_times.append(episode_time)

                print(
                    f"Episode {i_episode + 1} Finished | Steps: {step} | Reward: {reward_sum:.2f} | Time: {episode_time:.2f}s")

                if "success" in info:
                    result_str = 'Success' if info['success'] else 'Fail'
                    print(f"Result: {result_str}")
                    if info['success']:
                        success_num += 1

                # 绘制本回合的注意力曲线
                plot_attention_weights(weights_history, i_episode + 1)

    # 每10回合打印统计
    if (i_episode + 1) % 10 == 0:
        current_acc = success_num / (i_episode + 1) * 100
        print("-" * 50)
        print(f"Running Stats (Ep {i_episode + 1}):")
        print(f"  - Success Rate: {current_acc:.2f}%")
        print(f"  - Avg Duration: {np.mean(episode_times):.2f}s")
        print("-" * 50)

# ------------------- 测试结束总结 -------------------
print("\n" + "=" * 20 + " GRU Model Test Finished " + "=" * 20)
print(f"Total Episodes: {episodes}")
print(f"Total Success: {success_num}")
print(f"Final Success Rate: {success_num / episodes * 100:.2f}%")
print(f"Avg Time per Episode: {np.mean(episode_times):.2f}s")