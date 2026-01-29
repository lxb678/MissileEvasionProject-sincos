# 文件: Run_AirCombatEnv_test.py

import numpy as np
import random
import torch
import time
import os

# --- 导入配置和新的离散 PPO 类 ---
# 请根据您的实际目录结构确认路径是否正确
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.PPOMLP混合架构.Hybrid_PPO_混合架构雅可比修正优势归一化 import \
    PPO_discrete, DISCRETE_DIMS

# --- 导入修改后的混合控制环境 ---
from Interference_code.env.missile_evasion_environment_jsbsim_fuza_单一干扰.Vec_missile_evasion_environment_jsbsim2 import \
    AirCombatEnv

LOAD_ABLE = True  # 必须为True才能加载模型进行测试

# <<<--- Tacview 可视化开关 ---<<<
# 在测试时开启，可以直观看到飞机规避和干扰弹投放效果
TACVIEW_ENABLED_DURING_TEST = True


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置全栈随机种子，保证结果可复现 '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Gymnasium 环境通常在 reset 时接受种子，不需要显式 seed()
    if hasattr(env, 'reset'):
        # 这一步其实已经在 main loop 的 reset 中做了，这里主要是保险
        pass
    print(f"[Info] Global random seed set to {seed}")


def pack_action_into_dict(discrete_indices: np.ndarray) -> dict:
    """
    将 PPO_discrete 输出的离散动作索引数组 (numpy)
    打包成 Environment step() 所需的字典格式。

    Args:
        discrete_indices (np.ndarray): 形状为 (4,) 的数组，
                                       包含 [trigger, salvo, groups, interval] 的索引。

    Returns:
        dict: {"discrete_actions": np.array([...])}
    """
    # 确保转换为整数类型
    discrete_part = discrete_indices.astype(int)

    # 组装字典，键名必须与环境 action_space 定义一致
    action_dict = {
        "discrete_actions": discrete_part
    }
    return action_dict


# ------------------- 主程序 -------------------

if __name__ == "__main__":
    # 1. 初始化环境
    # dt=0.02 保证物理仿真精度，与训练时一致
    env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TEST, dt=0.02)
    set_seed(env)

    # 2. 指定模型路径
    # 请修改为您实际保存模型的文件夹路径
    # 例如: r"D:\code\规避导弹项目\save\save_evade_fuza\PPO_2025-xx-xx_xx-xx-xx\胜率模型"
    # 或者如果不使用胜率模型，则指向上一级目录
    model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza_单一干扰\PPO_2026-01-10_11-17-28"

    if not os.path.exists(model_path):
        print(f"[警告] 模型路径不存在: {model_path}")
        # 如果路径不对，您可以手动指定一个存在的路径，或者程序会尝试加载默认路径但可能失败

    print(f"正在加载模型: {model_path}")

    # 3. 初始化 Agent
    # 注意：这里使用的是 PPO_discrete 类
    agent = PPO_discrete(load_able=LOAD_ABLE, model_dir_path=model_path)

    # 4. 切换到评估模式 (Eval Mode)
    # 这会设置 PyTorch 模块为 eval()，关闭 Dropout/BatchNorm 等训练特有行为
    agent.prep_eval_rl()
    print("模型加载完成，开始测试...")

    # --- 统计变量 ---
    success_num = 0
    episodes = 100  # 总测试回合数
    episode_times = []  # 记录每回合耗时

    # --- 测试循环 ---
    for i_episode in range(episodes):
        episode_start_time = time.time()

        # 推荐：测试时使用不同的种子，覆盖更多随机情况
        # 训练时用的种子通常较小，测试种子加上一个大偏移量以避开训练集
        test_seed = AGENTPARA.RANDOM_SEED + 100000 + i_episode

        observation, info = env.reset(seed=test_seed)

        if np.isnan(observation).any():
            print(f"Episode {i_episode + 1}: NaN detected in observation, skipping...")
            continue

        done = False
        reward_sum = 0
        step = 0

        while not done:
            with torch.no_grad():
                # 1. 智能体决策
                # choose_action 返回: (action_indices, action_to_store, log_prob)
                # 我们只需要第一个返回值：动作索引
                action_indices, _, _ = agent.choose_action(observation, deterministic=True)

            # 2. 格式化动作
            action_dict = pack_action_into_dict(action_indices)

            # 3. 环境步进
            # PID 控制器会在 env.step 内部自动接管飞行控制
            observation, reward, terminated, truncated, info = env.step(action_dict)

            done = terminated or truncated
            reward_sum += reward
            step += 1

            # (可选) 实时打印某些关键动作，例如干扰弹投放
            if action_indices[0] == 1:  # trigger == 1
                # print(f"  [Step {step}] 投放干扰弹! 策略: {action_indices[1:]}")
                pass

        # --- 回合结束处理 ---
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)

        # 获取最终结果
        is_success = info.get('success', False)
        miss_dist = env.miss_distance if env.miss_distance is not None else 0.0

        if is_success:
            success_num += 1
            result_str = "成功 (Success)"
        else:
            result_str = "失败 (Fail)"

        print(f"Ep {i_episode + 1:03d} | "
              f"Result: {result_str} | "
              f"Miss Dist: {miss_dist:.2f}m | "
              f"Reward: {reward_sum:.2f} | "
              f"Steps: {step} | "
              f"SimTime: {env.t_now:.2f}s")

        # (可选) 渲染 3D 轨迹图
        # 如果您想每回合都看图，取消注释下面这行
        env.render()

    # --- 最终统计 ---
    print("\n" + "=" * 30)
    print("       测试统计报告        ")
    print("=" * 30)
    print(f"测试模型路径: {model_path}")
    print(f"总测试回合: {episodes}")
    print(f"成功回合数: {success_num}")
    print(f"平均成功率: {success_num / episodes * 100:.2f}%")
    print(f"平均每回合耗时: {np.mean(episode_times):.2f}s")
    print("=" * 30)

    # 在所有测试结束后，展示最后一次的轨迹图作为示例
    print("展示最后一回合的轨迹...")
    env.render()