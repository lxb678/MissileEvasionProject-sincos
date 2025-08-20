# -*- coding: utf-8 -*-
import gym
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from env.AirCombatEnv6 import *
from PPO_model.Config import *

LOAD_ABLE = False  # 不使用 PPO 模型

def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rule_based_action(env, flare_released, laser_on, laser_start_time):
    action = np.zeros(2, dtype=float)  # [红外诱饵弹, 激光干扰]

    # 参数设定
    FLARE_TRIGGER_DISTANCE = 2000   # 米
    # FLARE_TRIGGER_DISTANCE = 0.1  # 模糊值
    LASER_LEAD_TIME = 0.5         # 秒
    LASER_DURATION = 1          # 秒

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

    # 红外诱饵弹释放逻辑
    if (not flare_released) and (R_rel < FLARE_TRIGGER_DISTANCE):
        action[0] = 1
        flare_released = True

    return action, flare_released, laser_on, laser_start_time


# 主测试脚本
env = AirCombatEnv()
set_seed(env)

success_num = 0
episode_times = []

print("开始验证人工经验策略")
miss = []
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
    flare_released = False
    laser_on = False
    laser_start_time = 0

    while not done_eval:
        if t % (round(env.dt_dec / env.dt_normal)) == 0:
            action_eval, flare_released, laser_on, laser_start_time = rule_based_action(
                env, flare_released, laser_on, laser_start_time)

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

            # env.render()        #这个是可视化的  自己写的环境可以注释掉

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