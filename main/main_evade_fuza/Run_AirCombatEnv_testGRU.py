# ==============================================================================
# 这是针对PPO-GRU模型的测试脚本。
# 它基于标准的PPO测试脚本，并加入了在每个回合中管理和传递
# GRU隐藏状态（hidden states）的必要逻辑。
# ==============================================================================

import random
# <<< GRU 修改 >>>: 导入包含GRU模型的PPO实现
from Interference_code.PPO_model.PPO_evasion_fuza.PPOMLP混合架构.Hybrid_PPOMLP_GRU残差拼接雅可比修正 import *
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigGRU import *
# <<< 更改 >>>: 确保您正在使用的环境与训练时一致
from Interference_code.env.missile_evasion_environment_jsbsim_fuza.Vec_missile_evasion_environment_jsbsim import *
import time

LOAD_ABLE = True  # 是否使用save文件夹中的模型
# <<< GRU 修改 >>>: 添加开关以明确表示正在使用RNN模型
USE_RNN_MODEL = True  # 测试时必须为True以启用GRU逻辑


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


# ------------------- 主程序 -------------------

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TRAINING = True
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)
set_seed(env)

# <<< GRU 修改 >>>: 确保此路径指向您训练好的GRU模型
# model_path = r"D:\code\规避导弹项目\Interference_code\test\test_evade_fuza"
model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza\PPOGRU_2025-12-05_21-12-20"  # 示例路径
print(f"正在加载GRU模型: {model_path}")

# <<< GRU 修改 >>>: 初始化Agent时，必须传入 use_rnn=True
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_path, use_rnn=USE_RNN_MODEL)

success_num = 0

# 训练完之后，需要验证模型
agent.prep_eval_rl()
print("开始验证GRU模型")
episode_times = []

episodes = 100  # 总测试回合数
for i_episode in range(episodes):
    episode_start_time = time.time()  # 记录当前 episode 开始时间

    with torch.no_grad():
        done_eval = False
        observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

        # <<< GRU 修改 >>>: 在每个测试回合开始时，重置隐藏状态
        # 这是GRU模型与MLP模型在测试时的关键区别
        actor_hidden, critic_hidden = agent.get_initial_hidden_states()

        reward_sum = 0
        step = 0

        while not done_eval:
            # 1. <<< GRU 修改 >>>: 将隐藏状态传入 choose_action，并接收返回的新隐藏状态
            #    测试时我们不需要prob, value等，所以用 `_` 忽略它们
            action_eval_flat, _, _, _, \
                new_actor_hidden, new_critic_hidden = agent.choose_action(observation_eval,
                                                                          actor_hidden,
                                                                          critic_hidden,
                                                                          deterministic=True)

            # 2. 将扁平数组打包成字典 (这部分不变)
            action_dict_eval = pack_action_into_dict(action_eval_flat)
            # print(f"Step {step + 1}, Action: {action_dict_eval}")

            # 3. 将打包好的字典传递给环境
            observation_eval, reward_eval, terminated, truncated, info = env.step(action_dict_eval)

            # <<< GRU 修改 >>>: 更新隐藏状态以备下一个时间步使用
            actor_hidden = new_actor_hidden
            critic_hidden = new_critic_hidden

            # done_eval 现在是 terminated 或 truncated 的组合
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

                # 从info字典中获取最终结果
                if "success" in info:
                    print(f"结果: {'成功' if info['success'] else '失败'}")
                    if info['success']:
                        success_num += 1
                if "miss_distance" in info:
                    print(f"脱靶量: {info['miss_distance']:.2f} m")

                env.render() # 如果您想在每回合结束时看图，可以取消注释

    # 在一定回合后打印成功率
    if (i_episode + 1) % 10 == 0:
        print("-" * 50)
        print(f"前 {i_episode + 1} 回合统计:")
        print(f"  - 成功率: {success_num / (i_episode + 1) * 100:.2f}% ({success_num}/{i_episode + 1})")
        print(f"  - 平均耗时: {np.mean(episode_times):.2f}s")
        print("-" * 50)

# 最终测试结束
print("\n" + "=" * 20 + " GRU模型测试结束 " + "=" * 20)
print(f"总测试回合数: {episodes}")
print(f"总成功次数: {success_num}")
print(f"最终成功率: {success_num / episodes * 100:.2f}%")
print(f"所有回合平均耗时: {np.mean(episode_times):.2f}s")

# 可以在所有回合结束后，选择一个典型的失败或成功案例进行渲染
# env.render()