# 文件: Run_AirCombatEnv_testATTGRU.py (实体注意力适配版)

import random
# <<< 核心修改 >>>: 导入实体注意力版本的PPO模型
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.旧代码.Hybrid_PPO_ATTGRUMLP混合架构实体 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
# <<< 核心修改 >>>: 确保导入的是与训练时完全一致的环境
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体 import *
import time
from typing import Optional  # <<< 新增 >>> 导入 Optional 类型提示

LOAD_ABLE = True  # 测试时必须为 True
# 明确表示正在使用实体注意力的RNN模型
USE_RNN_MODEL = True


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, 'reset'):
        env.reset(seed=seed)


# <<< 核心修改 1/3 >>>: 扩展 pack_action_into_dict 函数以处理注意力权重
def pack_action_into_dict(flat_action_np: np.ndarray, attn_weights: Optional[np.ndarray] = None) -> dict:
    """
    将PPO agent.choose_action返回的扁平化NumPy数组和注意力权重
    转换为环境env.step()所期望的字典格式。
    """
    continuous_part = flat_action_np[:CONTINUOUS_DIM]
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)

    action_dict = {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part
    }

    # 如果提供了注意力权重，也将其添加到字典中
    if attn_weights is not None:
        action_dict["attention_weights"] = attn_weights

    return action_dict


# ------------------- 主程序 -------------------

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TESTING = True  # 测试时通常建议开启
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TESTING)
set_seed(env)

# <<< 核心修改 2/3 >>>: 确保此路径指向您训练好的【实体注意力】模型
model_path = r"D:\code\规避导弹项目\save\save_evade_fuza两个导弹\PPO_EntityATT_GRU_2025-11-11_22-35-38"  # 示例路径，请替换为您的实际模型路径
print(f"正在加载实体注意力GRU模型: {model_path}")

# 初始化Agent时，必须传入 use_rnn=True
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_path, use_rnn=USE_RNN_MODEL)

success_num = 0

# 切换到评估模式
agent.prep_eval_rl()
print("开始验证实体注意力GRU模型")
episode_times = []

episodes = 100  # 总测试回合数
for i_episode in range(episodes):
    episode_start_time = time.time()

    with torch.no_grad():
        done_eval = False
        observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

        # 在每个测试回合开始时，重置隐藏状态
        actor_hidden, critic_hidden = agent.get_initial_hidden_states()

        reward_sum = 0
        step = 0

        while not done_eval:
            # <<< 核心修改 3/3 >>>: choose_action 现在会返回注意力权重
            # 1. 获取动作、新的隐藏状态和【注意力权重】
            action_eval_flat, _, _, _, \
                new_actor_hidden, new_critic_hidden, attn_weights = agent.choose_action(observation_eval,
                                                                                        actor_hidden,
                                                                                        critic_hidden,
                                                                                        deterministic=True)

            # 2. 将扁平数组和注意力权重打包成字典
            action_dict_eval = pack_action_into_dict(action_eval_flat, attn_weights)

            # (可选) 打印注意力权重以供调试和分析
            if attn_weights is not None and step % 10 == 0:  # 每10步打印一次
                print(f"Step {step + 1}, Attention Weights: [{attn_weights[0]:.2f}, {attn_weights[1]:.2f}]")

            # 3. 将打包好的字典传递给环境
            observation_eval, reward_eval, terminated, truncated, info = env.step(action_dict_eval)

            # 4. 更新隐藏状态以备下一个时间步使用
            actor_hidden = new_actor_hidden
            critic_hidden = new_critic_hidden

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
                # 假设环境在info中返回最终脱靶量
                # if "miss_distance" in info:
                #     print(f"脱靶量: {info['miss_distance']:.2f} m")

                env.render() # 如果您想在每回合结束时看图，可以取消注释

    # 在一定回合后打印成功率
    if (i_episode + 1) % 10 == 0:
        print("-" * 50)
        print(f"前 {i_episode + 1} 回合统计:")
        print(f"  - 成功率: {success_num / (i_episode + 1) * 100:.2f}% ({success_num}/{i_episode + 1})")
        print(f"  - 平均耗时: {np.mean(episode_times):.2f}s")
        print("-" * 50)

# 最终测试结束
print("\n" + "=" * 20 + " 实体注意力GRU模型测试结束 " + "=" * 20)
print(f"总测试回合数: {episodes}")
print(f"总成功次数: {success_num}")
print(f"最终成功率: {success_num / episodes * 100:.2f}%")
print(f"所有回合平均耗时: {np.mean(episode_times):.2f}s")

# 渲染最后一个回合的轨迹
# env.render()