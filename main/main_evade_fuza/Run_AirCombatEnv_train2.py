import random
# from Interference_code.PPO_model.PPO_evasion_fuza.Hybrid_PPO_jsbsim import *
from Interference_code.PPO_model.PPO_evasion_fuza.PPOMLP混合架构.Hybrid_PPO_混合架构雅可比修正优势归一化 import *
from Interference_code.PPO_model.PPO_evasion_fuza.Config import *
from torch.utils.tensorboard import SummaryWriter
#from env.AirCombatEnv import *
from Interference_code.env.missile_evasion_environment_jsbsim_fuza.Vec_missile_evasion_environment_jsbsim import *
import time

LOAD_ABLE = False  #是否使用save文件夹中的模型

# <<<--- Tacview 可视化开关 ---<<<
# 将此项设为 True 即可在训练时开启 Tacview
TACVIEW_ENABLED_DURING_TRAINING = False
# ---
# def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
#     ''' 设置随机种子 '''
#     # env.action_space.seed(seed)   #可注释
#     # env.reset(seed=seed)          #可注释
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

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


# <<< 新增 >>> 复用与测试脚本相同的辅助函数，用于打包动作
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


#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir= 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
# writer = SummaryWriter(log_dir='../../log/log_evade/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))

# ------------------- Tensorboard 设置 -------------------
writer_log_dir = f'../../log/log_evade_fuza/PPO_{time.strftime("%Y-%m-%d_%H-%M-%S")}_seed{AGENTPARA.RANDOM_SEED}_load{LOAD_ABLE}'
writer = SummaryWriter(log_dir=writer_log_dir)
print(f"Tensorboard 日志将保存在: {writer_log_dir}")

# ------------------- 环境和智能体初始化 -------------------
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING,dt = 0.02)
set_seed(env)

# <<< 更改 >>> 初始化PPO时，如果LOAD_ABLE=True，需要指定模型路径
# 如果是False，则model_dir_path可以为None
model_load_path = None  # 从头训练
if LOAD_ABLE:
    # 在这里指定你要加载的预训练模型的文件夹路径
    model_load_path = r"D:\DESKTOP\毕设\JSBSim_py\JSBSim-master - a\save\save_evade\2024-05-18_09-17-06"
    print(f"--- 正在加载预训练模型: {model_load_path} ---")

agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=model_load_path)

# ------------------- 训练主循环 -------------------
global_step = 0
success_num = 0
MAX_EXE_NUM = 15000 #20000  # 最大训练回合数
MAX_STEP = 10000  # 每回合最大步数
UPDATE_CYCLE = 10  # 每多少回合训练一次

eval_reward_buffer = []
eval_counter = 0  # 计数器，专门用来控制测试的种子

for i_episode in range(MAX_EXE_NUM):
    # --- 1. 经验收集阶段 ---
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)

    if np.isnan(observation).any():
        print(f"!!! 严重错误: 第 {i_episode + 1} 回合 reset() 返回了 NaN! 退出训练。")
        break

    done = False
    step = 0
    episode_reward = 0

    # 每个 episode 最多跑 AGENTPARA.MAX_STEP 步
    for t in range(MAX_STEP):
        agent.prep_eval_rl()  # 切换到评估模式以进行交互
        with torch.no_grad():
            # a. 从PPO获取扁平化的动作数组
            env_action_flat, action_to_store_in_buffer, prob = agent.choose_action(observation)

            # b. 获取当前状态的价值
            value = agent.get_value(observation).cpu().detach().numpy()

            # c. 记录当前状态
            state_to_store = observation

        # d. <<< 核心更改 >>> 将扁平动作打包成字典
        action_dict = pack_action_into_dict(env_action_flat)

        # e. 与环境交互
        observation, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        episode_reward += reward

        # f. 将经验存入Buffer
        # 注意：存入buffer的是 action_to_store_in_buffer (包含原始u和离散索引)
        agent.store_experience(state_to_store, action_to_store_in_buffer, prob, value, reward, done)

        global_step += 1
        step += 1

        if done:
            break

    # --- 回合结束后的日志记录 ---
    print(f"Episode {i_episode + 1} | Steps: {step} | SimTime: {env.t_now:.2f}s | Reward: {episode_reward:.2f}")
    writer.add_scalar('Episode/Reward', episode_reward, global_step)
    # writer.add_scalar('Episode/Steps', step, i_episode)

    if "success" in info:
        if info['success']:
            success_num += 1

    # 每100回合记录一次成功率
    if (i_episode + 1) % 100 == 0:
        success_rate = success_num / 100.0
        print("-" * 50)
        print(f"最近100回合 (Ep {i_episode - 98}-{i_episode + 1}) 统计:")
        print(f"  - 成功率: {success_rate * 100:.2f}% ({success_num}/100)")
        print("-" * 50)
        writer.add_scalar('Metrics/Success_Rate_per_100_ep', success_rate, i_episode)

        # 如果成功率很高，保存一个带标记的模型
        if success_rate >= 0.95:
            agent.save(prefix=f"success_{int(success_rate * 100)}_ep{i_episode + 1}")

        success_num = 0  # 重置计数器

    # --- 2. 训练阶段 ---
    # 每 AGENTPARA.UPDATE_CYCLE 个 episode 训练一次
    if (i_episode + 1) % UPDATE_CYCLE == 0 :
        print(f"\n--- [Episode {i_episode + 1}] 开始训练 | Global Steps: {global_step} ---")
        agent.prep_training_rl()  # 切换到训练模式
        train_info = agent.learn()

        # 记录训练相关的统计数据到Tensorboard
        for key, value in train_info.items():
            writer.add_scalar(f"Train/{key}", value, global_step)
        print("--- 训练结束 ---\n")

        # --- 3. 评估阶段 ---
        # 训练后立即进行一次评估
        print(f"--- [Episode {i_episode + 1}] 开始评估 ---")
        agent.prep_eval_rl()  # 切换回评估模式

        with torch.no_grad():
            eval_done = False
            # eval_obs, _ = env.reset(seed=AGENTPARA.RANDOM_SEED)  # 使用固定的种子进行评估

            # =========================================================
            # <<< 核心修改开始 >>> 设置互不重复的测试种子
            # =========================================================

            # 定义一个巨大的偏移量，比如 100,000 (确保大于你的 MAX_EXE_NUM)
            TEST_SEED_OFFSET = 100000 #MAX_EXE_NUM

            # [测试种子] = 基础种子 + 偏移量 + 当前回合数
            # 例如：1000 + 100000 + 10 = 101010
            # 下一次测试是：1000 + 100000 + 20 = 101020
            # 结果：永远不重复，且跟训练集完全隔离
            current_test_seed = AGENTPARA.RANDOM_SEED + TEST_SEED_OFFSET + i_episode

            # 显式传入计算好的种子
            eval_obs, _ = env.reset(seed=current_test_seed)

            print(f"    (使用测试种子: {current_test_seed})")  # 打印出来方便检查

            # =========================================================
            # <<< 核心修改结束 >>>
            # =========================================================

            eval_reward_sum = 0

            for _ in range(MAX_STEP):
                # a. 获取扁平动作
                eval_action_flat, _, _ = agent.choose_action(eval_obs, deterministic=True)

                # b. <<< 核心更改 >>> 打包成字典
                eval_action_dict = pack_action_into_dict(eval_action_flat)

                # c. 与环境交互
                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = env.step(eval_action_dict)
                eval_done = eval_terminated or eval_truncated

                eval_reward_sum += eval_reward
                if eval_done:
                    break

            # 记录评估结果
            print(f"评估回合奖励: {eval_reward_sum:.2f}")
            writer.add_scalar('Eval/Reward_Sum', eval_reward_sum, global_step)
            eval_reward_buffer.append(eval_reward_sum)  # 存入列表
            # --- 核心修改：判断是否攒够了 10 个 ---
            if len(eval_reward_buffer) >= 10:
                mean_reward = np.mean(eval_reward_buffer)
                print(f"\n{'#' * 40}")
                print(f"### 统计报告: 过去10次测试平均奖励: {mean_reward:.2f} ###")
                print(f"{'#' * 40}\n")

                # 记录到 Tensorboard，使用的是'Eval/Mean_10_Buffer'
                writer.add_scalar('Eval/Mean_10_Buffer', mean_reward, global_step)

                # 清空缓存，准备下一轮积累
                eval_reward_buffer = []

        print("--- 评估结束 ---\n")

# 训练结束
writer.close()
print("=" * 20 + " 训练完成 " + "=" * 20)