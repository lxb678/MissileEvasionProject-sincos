# -*- coding: utf-8 -*-
import numpy as np
import random
import time
from Interference_code.env.oldenv.AirCombatEnv6_onlyflare import *
from PPO_model.Config import *

LOAD_ABLE = False  # 不使用 PPO 模型

def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rule_based_action(env, flare_count, flare_max_groups, laser_on, laser_start_time):
    action = np.zeros(2, dtype=float)  # [红外诱饵弹, 激光干扰]

    # 参数设定
    FLARE_TRIGGER_DISTANCE = 2000   # 米
    LASER_LEAD_TIME = 0.5         # 秒
    LASER_DURATION = 1            # 秒

    R_vec = env.x_target_now[0:3] - env.x_missile_now[3:6]
    R_rel = np.linalg.norm(R_vec)
    V_missile = env.x_missile_now[0]
    t_now = env.t_now
    o_dis = env.observation[0]

    # 激光开启逻辑：提前 LASER_LEAD_TIME 秒照射
    if (not laser_on) and (R_rel < FLARE_TRIGGER_DISTANCE + LASER_LEAD_TIME * V_missile):
        action[1] = 1
        laser_on = True
        laser_start_time = t_now

    # 激光关闭逻辑
    if laser_on and (t_now - laser_start_time > LASER_DURATION):
        action[1] = 0
        laser_on = False

    # 红外诱饵弹释放逻辑（可投多组）
    if (R_rel < FLARE_TRIGGER_DISTANCE) and (flare_count < flare_max_groups):
        action[0] = 1
        flare_count += 1

    return action, flare_count, laser_on, laser_start_time


def rule_based_action_improved_flares(env, flare_count, flare_max_groups, laser_on, laser_start_time):
    """
    改进了红外诱饵弹投放逻辑的规则动作函数。
    - 将连续投放修改为按距离阈值分批次投放。
    - 激光逻辑保持不变。
    """
    action = np.zeros(2, dtype=float)  # [红外诱饵弹, 激光干扰]

    # --- 参数设定 ---
    # 红外诱饵弹：定义多个距离阈值，用于分批次投放
    # 列表长度应与 flare_max_groups 匹配
    FLARE_THRESHOLDS = [2500, 1800, 1000,500]  # 假设 flare_max_groups = 3

    # 激光参数（保持不变）
    LASER_LEAD_TIME = 0.5  # 秒
    LASER_DURATION = 1  # 秒
    LASER_TRIGGER_DISTANCE = 2000  # 这里用一个单独的距离，与诱饵弹解耦

    # --- 状态获取 ---
    R_vec = env.x_target_now[0:3] - env.x_missile_now[3:6]
    R_rel = np.linalg.norm(R_vec)
    V_missile = env.x_missile_now[0]  # 注意：x_missile_now[0] 可能是速度标量，也可能是Vx，取决于你的状态定义
    t_now = env.t_now

    # --- 决策逻辑 ---

    # 1. 改进的红外诱饵弹释放逻辑 (分批次)
    # 检查是否还有可用的诱饵弹组
    if flare_count < flare_max_groups:
        # 获取当前待投放批次对应的距离阈值
        current_flare_threshold = FLARE_THRESHOLDS[flare_count]

        # 如果导弹距离小于当前批次的触发阈值，则投放
        if R_rel < current_flare_threshold:
            action[0] = 1
            flare_count += 1
            print(f"t={t_now:.2f}s, R={R_rel:.0f}m: 距离小于 {current_flare_threshold}m，投放第 {flare_count} 组诱饵弹。")

    # 2. 激光干扰逻辑 (保持不变)
    # 激光开启逻辑：提前 LASER_LEAD_TIME 秒照射
    if (not laser_on) and (R_rel < LASER_TRIGGER_DISTANCE + LASER_LEAD_TIME * V_missile):
        action[1] = 1
        laser_on = True
        laser_start_time = t_now

    # 激光关闭逻辑
    if laser_on and (t_now - laser_start_time > LASER_DURATION):
        action[1] = 0
        laser_on = False
    elif laser_on:  # 如果激光正处于开启状态，则保持开启
        action[1] = 1

    return action, flare_count, laser_on, laser_start_time


def spherical_to_cartesian_velocity(V, theta, psi):
    """
    将球坐标系下的速度 (V, theta, psi) 转换为笛卡尔坐标系下的速度 (vx, vy, vz)。

    Args:
        V (float): 速度大小 (Speed)
        theta (float): 俯仰角 (Pitch/Elevation angle) in radians
        psi (float): 偏航角 (Yaw/Azimuth angle) in radians

    Returns:
        np.ndarray: [vx, vy, vz]
    """
    vx = V * np.cos(theta) * np.cos(psi)
    vy = V * np.cos(theta) * np.sin(psi)
    vz = V * np.sin(theta)
    return np.array([vx, vy, vz])


def rule_based_action_by_velocity_corrected(env, flare_count, flare_max_groups, laser_on, laser_start_time):
    """
    针对您特定环境状态定义的、基于接近速度和TTI的修正版规则函数。
    """
    action = np.zeros(2, dtype=float)

    # --- 1. 参数设定 (保持不变) ---
    FLARE_TTI_THRESHOLDS = [5.0, 3.0, 1.5]
    LASER_LEAD_TIME = 0.5
    LASER_DURATION = 1.0
    LASER_TRIGGER_DISTANCE = 2000

    # --- 2. 状态获取与关键指标计算 (已修正) ---
    t_now = env.t_now

    # 提取飞机状态
    pos_target = env.x_target_now[0:3]
    V_target, theta_target, psi_target = env.x_target_now[3], env.x_target_now[4], env.x_target_now[5]

    # 提取导弹状态
    # ★★★ 修正点 1: 正确提取导弹位置 ★★★
    pos_missile = env.x_missile_now[3:6]
    V_missile, theta_missile, psi_missile = env.x_missile_now[0], env.x_missile_now[1], env.x_missile_now[2]

    # ★★★ 修正点 2: 将速度从球坐标转换为笛卡尔坐标 ★★★
    vel_target_cartesian = spherical_to_cartesian_velocity(V_target, theta_target, psi_target)
    vel_missile_cartesian = spherical_to_cartesian_velocity(V_missile, theta_missile, psi_missile)

    # --- 使用修正后的笛卡尔坐标进行后续计算 ---
    R_vec = pos_target - pos_missile
    # ★★★ 修正点 3: 使用转换后的速度计算相对速度 ★★★
    V_vec_rel = vel_target_cartesian - vel_missile_cartesian
    R_rel = np.linalg.norm(R_vec)

    if R_rel < 10:
        return action, flare_count, laser_on, laser_start_time

    # 计算接近速度 (Vc)
    Vc = -np.dot(V_vec_rel, R_vec) / R_rel

    # 计算预估碰撞时间 (TTI)
    TTI = R_rel / Vc if Vc > 0 else float('inf')

    # --- 3. 决策逻辑 (逻辑本身不变，但输入数据已正确) ---

    # 3.1 基于TTI的红外诱饵弹释放逻辑
    if flare_count < flare_max_groups:
        current_tti_threshold = FLARE_TTI_THRESHOLDS[flare_count]
        if TTI < current_tti_threshold:
            action[0] = 1
            flare_count += 1
            print(
                f"★★★ ACTION: t={t_now:.2f}s, TTI={TTI:.2f}s. Breached TTI threshold {current_tti_threshold}s. Deploying flare group #{flare_count}. ★★★")

    # 3.2 激光干扰逻辑
    # 注意: 这里的 V_missile 是速度大小，可以直接用
    if (not laser_on) and (R_rel < LASER_TRIGGER_DISTANCE + LASER_LEAD_TIME * V_missile):
        action[1] = 1
        laser_on = True
        laser_start_time = t_now

    if laser_on and (t_now - laser_start_time > LASER_DURATION):
        action[1] = 0
        laser_on = False
    elif laser_on:
        action[1] = 1

    return action, flare_count, laser_on, laser_start_time


def rule_based_action_dynamic_distance(env, flare_count, flare_max_groups, laser_on, laser_start_time):
    """
    基于动态距离和高接近速度覆盖规则的动作函数。
    规则 1: 动态触发距离 = 基础距离 + Vc * 反应时间
    规则 2: 如果 Vc > 1000 m/s (且在一定距离内)，则提前投放
    """
    action = np.zeros(2, dtype=float)

    # --- 1. 参数设定 ---
    # 红外诱饵弹参数
    # 为每个投放阶段设置一个“基础”距离
    BASE_THRESHOLDS = [4000, 3500, 3000]  # 米, 对应第1, 2, 3组诱饵弹
    # BASE_THRESHOLDS = [3000, 2000, 1000]  # 米, 对应第1, 2, 3组诱饵弹
    # BASE_THRESHOLDS = [2500, 2000, 1500,1000]  # 米, 对应第1, 2, 3组诱饵弹
    REACTION_TIME = 0.8  # 秒, Vc的乘数，决定了距离阈值的动态调整幅度

    # 高威胁覆盖规则的参数
    HIGH_VC_THRESHOLD = 1000.0  # 米/秒, 触发紧急投放的接近速度阈值
    HIGH_VC_DISTANCE_THRESHOLD = 4000.0  # 米, 紧急规则生效的最大距离，防止太远就投放

    # 激光参数 (保持不变)
    LASER_LEAD_TIME = 0.5
    LASER_DURATION = 1.0
    LASER_TRIGGER_DISTANCE = 2000

    # --- 2. 状态获取与关键指标计算 ---
    t_now = env.t_now
    pos_target = env.x_target_now[0:3]
    V_target, theta_target, psi_target = env.x_target_now[3:6]

    pos_missile = env.x_missile_now[3:6]
    V_missile, theta_missile, psi_missile = env.x_missile_now[0:3]

    vel_target_cartesian = spherical_to_cartesian_velocity(V_target, theta_target, psi_target)
    vel_missile_cartesian = spherical_to_cartesian_velocity(V_missile, theta_missile, psi_missile)

    R_vec = pos_target - pos_missile
    V_vec_rel = vel_target_cartesian - vel_missile_cartesian
    R_rel = np.linalg.norm(R_vec)

    if R_rel < 10:
        return action, flare_count, laser_on, laser_start_time

    Vc = -np.dot(V_vec_rel, R_vec) / R_rel

    # --- 3. 决策逻辑 ---

    # 3.1 红外诱饵弹释放逻辑
    # 首先检查是否还有可用的诱饵弹
    if flare_count < flare_max_groups:
        should_deploy = False
        deploy_reason = ""

        # --- 规则 2: 高威胁覆盖规则 (最高优先级) ---
        # 如果接近速度非常快，并且已经进入了威胁范围
        if Vc > HIGH_VC_THRESHOLD and R_rel < HIGH_VC_DISTANCE_THRESHOLD:
            should_deploy = True
            deploy_reason = f"High Vc override! (Vc={Vc:.0f} m/s)"

        # --- 规则 1: 常规动态距离规则 ---
        # 如果高威胁规则未触发，则检查常规规则
        else:
            # 获取当前阶段的基础距离
            base_threshold = BASE_THRESHOLDS[flare_count]
            # 计算动态触发距离
            dynamic_threshold = base_threshold + Vc * REACTION_TIME

            if R_rel < dynamic_threshold:
                should_deploy = True
                deploy_reason = f"Dynamic distance threshold breached (R={R_rel:.0f}m < Dyn_T={dynamic_threshold:.0f}m)"

        # 如果任一规则决定投放，则执行动作
        if should_deploy:
            action[0] = 1
            flare_count += 1
            print(f"★★★ ACTION: t={t_now:.2f}s, Deploying flare group #{flare_count}. Reason: {deploy_reason} ★★★")

    # 3.2 激光干扰逻辑 (保持不变)
    if (not laser_on) and (R_rel < LASER_TRIGGER_DISTANCE + LASER_LEAD_TIME * V_missile):
        action[1] = 1
        laser_on = True
        laser_start_time = t_now

    if laser_on and (t_now - laser_start_time > LASER_DURATION):
        action[1] = 0
        laser_on = False
    elif laser_on:
        action[1] = 1

    return action, flare_count, laser_on, laser_start_time

# 主测试脚本
env = AirCombatEnv()
set_seed(env)

success_num = 0
episode_times = []

print("开始验证人工经验策略")
miss = []

flare_max_groups = 3   # 设置最多投几组诱饵弹

for i_episode in range(100):
    episode_start_time = time.time()

    done_eval = False
    observation_eval = np.array(env.reset())
    reward_sum = 0
    t = 0
    step = 0
    reward_eval = 0
    reward_4 = 0

    # 初始化干扰状态变量
    flare_count = 0
    laser_on = False
    laser_start_time = 0

    while not done_eval:
        if t % (round(env.dt_dec / env.dt_normal)) == 0:
            action_eval, flare_count, laser_on, laser_start_time = rule_based_action_dynamic_distance(
                env, flare_count, flare_max_groups, laser_on, laser_start_time)
            action_eval = np.array([action_eval[0], 0])

            observation_eval, reward_eval, done_eval, reward_4, _ = env.step(action_eval)
            reward_sum += reward_eval + reward_4
            t += 1
            step += 1
        else:
            action_eval1 = np.array([0, action_eval[1]])
            observation_eval, reward_eval, done_eval, info, _ = env.step(action_eval1)
            t += 1

        if done_eval:
            miss.append(env.miss_distance)  # 保存脱靶量
            print("Episode {} finished after {} steps, 仿真时间 t = {}s, 脱靶量 {:.2f}m".format(
                    i_episode + 1, step + 1, round(env.t_now, 2), env.miss_distance))

            # env.render()
            if env.success:
                success_num += 1
            break

print("人工经验策略飞机存活率：{:.2f}%".format(success_num / 100 * 100))


# 绘制散点图
KILL_RADIUS = 12  # 单位：m

# 数据准备
episode_times_ms = np.array(miss)  # miss 是脱靶量列表
x = np.arange(1, len(episode_times_ms) + 1)
y = episode_times_ms
x = x[1:]
y = y[1:]

# 判断哪些点需要标红（脱靶量小于杀伤半径）
hit_mask = y < KILL_RADIUS
miss_mask = ~hit_mask

# 绘图
plt.figure(figsize=(16, 6))
plt.style.use('seaborn-v0_8-muted')

# 先画蓝色（未命中点）
plt.scatter(x[miss_mask], y[miss_mask], c='dodgerblue', edgecolors='k', s=60, alpha=0.8, label='未命中')

# 再画红色（命中点）
plt.scatter(x[hit_mask], y[hit_mask], c='red', edgecolors='k', s=60, alpha=0.8, label='命中（脱靶量 < 12m）')

# 添加平均线
mean_y = np.mean(y)
plt.axhline(mean_y, color='orange', linestyle='--', linewidth=2, label=f'平均脱靶量: {mean_y:.1f} m')

# 标签与设置
plt.xlabel('仿真次数', fontsize=16)
plt.ylabel('基于人工经验方法的导弹脱靶量 (m)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()