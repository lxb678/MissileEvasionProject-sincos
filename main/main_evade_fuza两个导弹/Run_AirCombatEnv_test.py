import numpy as np
import random
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_混合架构 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.Config import *
#from env.AirCombatEnv import *
# from env.AirCombatEnv6_maneuver_flare import *
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim import *
import time

LOAD_ABLE = True  #是否使用save文件夹中的模型


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    # env.action_space.seed(seed)  # Gymnasium中不推荐直接对action_space seed
    # env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# <<< 新增 >>> 辅助函数，用于将PPO输出的扁平NumPy数组打包成环境期望的字典
def pack_action_into_dict(flat_action_np: np.ndarray) -> dict:
    """
    将PPO agent.choose_action返回的扁平化NumPy数组转换为
    环境env.step()所期望的字典格式。

    Args:
        flat_action_np (np.ndarray): 来自PPO的动作数组, 结构为
            [c1, c2, c3, c4, d1_idx, d2_idx, d3_idx, d4_idx, d5_idx]

    Returns:
        dict: 格式化的动作字典, 例如:
            {
                "continuous_actions": np.array([...]),
                "discrete_actions": np.array([...])
            }
    """
    # 1. 提取连续动作部分
    continuous_part = flat_action_np[:CONTINUOUS_DIM]

    # 2. 提取离散动作部分
    # PPO输出的已经是离散索引了，但它们是浮点数，需要转换为整数
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)



    # 3. 组装成字典
    action_dict = {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part
    }

    return action_dict


# ------------------- 主程序 -------------------

# <<<--- Tacview 可视化开关 ---<<<
TACVIEW_ENABLED_DURING_TRAINING = True
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING,dt = 0.02)
set_seed(env)

# <<< 更改 >>> 确保这里的模型路径与您训练时保存的路径一致
# 假设您的模型保存在 "../../test/test_evade"
# model_path = "../../test/test_evade" # 或者您训练模型时使用的其他路径
# model_path = r"D:\code\规避导弹项目\Interference_code\test\test_evade_fuza"
model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza两个导弹\PPO_2025-11-20_18-26-31"
agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_path)

success_num = 0

# 训练完之后，需要验证模型
agent.prep_eval_rl()
print("开始验证模型")
episode_times = []

episodes = 100  # 总测试回合数
for i_episode in range(episodes):
    episode_start_time = time.time()  # 记录当前 episode 开始时间

    with torch.no_grad():
        done_eval = False
        # <<< 更改 >>> Gymnasium的reset返回一个元组 (obs, info)
        observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)  # 建议每回合用不同种子
        reward_sum = 0
        step = 0

        while not done_eval:
            # 1. 从PPO获取扁平化的动作数组 (这部分不变)
            action_eval_flat, action_to_store, prob = agent.choose_action(observation_eval, deterministic=True)

            # print("action_eval_flat", action_eval_flat)

            # 2. <<< 核心更改 >>> 将扁平数组打包成字典
            action_dict_eval = pack_action_into_dict(action_eval_flat)
            # print(f"动作: {action_dict_eval}")

            # print(f"第 {i_episode + 1} 回合, 第 {step + 1} 步, 动作: {action_dict_eval}")

            # 3. 将打包好的字典传递给环境 (这部分改变)
            # <<< 更改 >>> Gymnasium的step返回一个五元组
            observation_eval, reward_eval, terminated, truncated, info = env.step(action_dict_eval)

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
print("\n" + "=" * 20 + " 测试结束 " + "=" * 20)
print(f"总测试回合数: {episodes}")
print(f"总成功次数: {success_num}")
print(f"最终成功率: {success_num / episodes * 100:.2f}%")
print(f"所有回合平均耗时: {np.mean(episode_times):.2f}s")

# 可以在所有回合结束后，选择一个典型的失败或成功案例进行渲染
# env.render()