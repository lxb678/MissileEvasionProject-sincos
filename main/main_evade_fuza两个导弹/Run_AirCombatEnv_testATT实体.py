# --- START OF FILE Run_AirCombatEnv_testATTMLP实体.py (仅Attention+MLP版) ---

import random
from typing import Optional

# <<< 核心修改 >>>: 导入仅包含 Attention+MLP 的 PPO 模型
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_ATTMLP交叉注意力2 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
# 导入环境 (保持不变)
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体 import *
import time

LOAD_ABLE = True  # 测试时必须为 True
# <<< 核心修改 >>>: 关闭 RNN 模式
USE_RNN_MODEL = False


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)


# pack_action_into_dict 函数保持不变
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


# <<< 新增 >>> 绘图函数
def plot_attention_weights(weights_history, episode_num):
    """
    绘制单个回合的注意力权重变化曲线。
    """
    if not weights_history:
        print("警告: 注意力权重历史为空，无法绘图。")
        return

    # 将列表转换为Numpy数组，方便操作
    weights_np = np.array(weights_history)

    # 获取时间步数量
    timesteps = np.arange(weights_np.shape[0])

    plt.figure(figsize=(10, 5))

    # 绘制导弹1的注意力权重曲线 (索引为0)
    plt.plot(timesteps, weights_np[:, 0], label='对导弹1的注意力', color='blue', marker='.')

    # 绘制导弹2的注意力权重曲线 (索引为1)
    plt.plot(timesteps, weights_np[:, 1], label='对导弹2的注意力', color='red', marker='.')

    plt.title(f'第 {episode_num} 回合: 注意力权重随时间步的变化')
    plt.xlabel('时间步 (Step)')
    plt.ylabel('注意力权重 (Attention Weight)')
    plt.ylim(0, 1.05)  # 将Y轴范围固定在0到1之间
    plt.grid(True)
    plt.legend()
    plt.show()


# ------------------- 主程序 -------------------

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TESTING = True
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TESTING)
set_seed(env)

# <<< 核心修改 >>>: 确保此路径指向您训练好的【Attention+MLP】模型
# model_path = r"D:\code\规避导弹项目\Interference_code\test\test_evade_fuza两个导弹"  # 请替换为您的实际模型路径
# model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza两个导弹\PPO_EntityCrossATT_MLP_2025-11-21_16-13-43"  # 请替换为您的实际模型路径
model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza两个导弹\PPO_EntityCrossATT_MLP_2025-11-21_16-24-41"
print(f"正在加载实体注意力MLP模型: {model_path}")

# <<< 核心修改 >>>: 初始化Agent时，传入 use_rnn=False
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_path, use_rnn=USE_RNN_MODEL)

success_num = 0

# 切换到评估模式
agent.prep_eval_rl()
print("开始验证实体注意力MLP模型")
episode_times = []

episodes = 100
for i_episode in range(episodes):
    episode_start_time = time.time()

    with torch.no_grad():
        done_eval = False
        observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

        # <<< 核心修改 >>>: 移除所有与 hidden_state 相关的代码
        # actor_hidden, critic_hidden = agent.get_initial_hidden_states()

        reward_sum = 0
        step = 0

        # <<< 新增 >>> 初始化用于存储当前回合权重的列表
        weights_history = []

        while not done_eval:
            # <<< 核心修改 >>>: choose_action 调用简化，不再处理 hidden_state
            # 1. 获取动作和注意力权重
            action_eval_flat, _, _, _, attn_weights = agent.choose_action(observation_eval,
                                                                          deterministic=True)

            # print("action_eval_flat", action_eval_flat)

            # <<< 新增 >>> 存储当前步的注意力权重
            if attn_weights is not None:
                weights_history.append(attn_weights.copy())

            # 2. 将动作和权重打包成字典
            action_dict_eval = pack_action_into_dict(action_eval_flat, attn_weights)

            # (可选) 打印注意力权重
            if attn_weights is not None and step % 10 == 0:
                print(f"Step {step + 1}, Attention Weights: [{attn_weights[0]:.2f}, {attn_weights[1]:.2f}]")

            # 3. 与环境交互
            observation_eval, reward_eval, terminated, truncated, info = env.step(action_dict_eval)

            # <<< 核心修改 >>>: 移除 hidden_state 更新
            # actor_hidden = new_actor_hidden
            # critic_hidden = new_critic_hidden

            done_eval = terminated or truncated
            reward_sum += reward_eval
            step += 1

            if done_eval:
                episode_time = time.time() - episode_start_time
                episode_times.append(episode_time)

                print(f"Episode {i_episode + 1} finished after {step} timesteps, "
                      f"仿真时间 t = {env.t_now:.2f}s, "
                      f"用时 {episode_time:.2f}s")
                print(f"奖励: {reward_sum:.2f}")

                if "success" in info:
                    print(f"结果: {'成功' if info['success'] else '失败'}")
                    if info['success']:
                        success_num += 1

                # env.render()

                # <<< 新增 >>> 调用绘图函数来显示注意力权重曲线
                plot_attention_weights(weights_history, i_episode + 1)

    if (i_episode + 1) % 10 == 0:
        print("-" * 50)
        print(f"前 {i_episode + 1} 回合统计:")
        print(f"  - 成功率: {success_num / (i_episode + 1) * 100:.2f}% ({success_num}/{i_episode + 1})")
        print(f"  - 平均耗时: {np.mean(episode_times):.2f}s")
        print("-" * 50)

# 最终测试结束
print("\n" + "=" * 20 + " 实体注意力MLP模型测试结束 " + "=" * 20)
print(f"总测试回合数: {episodes}")
print(f"总成功次数: {success_num}")
print(f"最终成功率: {success_num / episodes * 100:.2f}%")
print(f"所有回合平均耗时: {np.mean(episode_times):.2f}s")

# env.render()