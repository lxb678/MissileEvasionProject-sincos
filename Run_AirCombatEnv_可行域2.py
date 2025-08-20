# -*- coding: utf-8 -*-
import os

import csv

import gym
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from env.AirCombatEnv可行域 import *
# from env.AirCombatEnv可行域_pianyou import *
from PPO_model.Config import *

LOAD_ABLE = False  # 不使用 PPO 模型

def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rule_based_action(env,LASER_START_TIME, LASER_DURATION, FLARE_RELEASE_TIMES):
    # 参数设定
    action = np.zeros(2, dtype=float)  # [红外诱饵弹, 激光干扰]
    FLARE_TIME_WINDOW = 0.2            # 容差，允许误差 ±0.1 秒
    t_now = env.t_now

    # 激光开启逻辑
    if LASER_START_TIME <= round(t_now, 2) < LASER_START_TIME + LASER_DURATION:
        action[1] = 1

    # === 红外诱饵弹释放逻辑（只看当前时间是否接近任何一个时间点） ===
    for release_time in FLARE_RELEASE_TIMES:
        if abs(t_now - release_time) < FLARE_TIME_WINDOW:
            action[0] = 1
            break  # 找到一个就退出循环

    return action
def generate_uniform_laser_start_times(laser_start_min, laser_start_max, step=0.6):
    """
    生成在 [laser_start_min, laser_start_max] 范围内，所有与 step 对齐的激光开启时间。

    :param laser_start_min: float，最早开始时间（秒）
    :param laser_start_max: float，最晚开始时间（秒）
    :param step: float，步长（秒）
    :return: list[float]，对齐后的激光开启时间列表
    """
    import math

    # 向上取整最小起点，向下取整最大终点
    start = math.ceil(laser_start_min / step)
    end = math.floor(laser_start_max / step)

    times = [round(i * step, 2) for i in range(start, end + 1)]
    return times

def generate_uniform_laser_durations(duration_min, duration_max, step=0.6):
    """
    生成在 [duration_min, duration_max] 范围内，所有与 step 对齐的激光干扰持续时间。

    :param duration_min: float，最短持续时间（秒）
    :param duration_max: float，最长持续时间（秒）
    :param step: float，决策步长（秒）
    :return: list[float]，所有合法的持续时间（与step对齐）
    """
    import math

    # 向上取整起点，向下取整终点（以避免超出边界）
    start = math.ceil(duration_min / step)
    end = math.floor(duration_max / step)

    durations = [round(i * step, 2) for i in range(start, end + 1)]
    return durations

def generate_interval_list(min_interval, max_interval, step):
    """
    在 [min_interval, max_interval] 区间内，生成所有与 step 对齐的间隔值。

    :param min_interval: float，最小间隔值
    :param max_interval: float，最大间隔值
    :param step: float，对齐步长
    :return: list[float]，所有合法的对齐间隔值
    """
    import math

    start = math.ceil(min_interval / step)
    end = math.floor(max_interval / step)

    intervals = [round(i * step, 2) for i in range(start, end + 1)]
    return intervals

def generate_aligned_start_times(start, end, interval):
    """
    生成与决策步长对齐的诱饵弹开始释放时间
    """
    times = []
    t = start
    epsilon = 1e-8  # 微小容差，用于防止浮点误差跳过end
    while t <= end + epsilon:
        times.append(round(t, 2))
        t += interval
    return times

def generate_flare_release_times(start_time, group_num, group_interval):
    """
    根据开始时间、释放组数、组间隔生成释放时间组
    所有时间都自动对齐到决策步长（0.6s）
    """
    step = 0.6  # 决策步长
    group_interval = round(group_interval / step) * step  # 对齐
    return [round(start_time + i * group_interval, 2) for i in range(group_num)]


# 主测试脚本
env = AirCombatEnv()
set_seed(env)

success_num = 0
records = []  # 用于存储所有的仿真结果
episode = 0
FLARE_RELEASE_TIMES = []

# 可配置参数
step = 0.6   #环境决策步长
#设置激光干扰开启时间
# LASER_LEAD_TIME = [0.0]
LASER_LEAD_TIME = generate_uniform_laser_start_times(0.0, 27.0,  step=0.6)
print("对齐后的激光开始时间列表：", LASER_LEAD_TIME)
#设置激光干扰持续时间
# LASER_DURATION = [27.0]
LASER_DURATION = generate_uniform_laser_durations(0.0, 27.0, step=0.6)
print("对齐后的激光持续时间列表：", LASER_DURATION)
#设置诱饵弹释放组数
group_nums = [2]
#设置诱饵弹释放间隔
group_intervals = [0.6]
# group_intervals = generate_interval_list(min_interval=0.6, max_interval=3.0, step=0.6)
print("对齐后的组间隔列表：", group_intervals)
#设置诱饵弹释放时间
start_times = [24.6]
# start_times = generate_aligned_start_times(start=0.0, end=30.0, interval=0.6)

# print("诱饵弹释放时间：", start_times)

for start in start_times:
    for group_num in group_nums:
        for group_interval in group_intervals:
            flare_times = generate_flare_release_times(start, group_num, group_interval)
            FLARE_RELEASE_TIMES.append(flare_times)

print(FLARE_RELEASE_TIMES)

#计算一共会有多少个仿真回合
flare_combo_num = len(start_times) * len(group_nums) * len(group_intervals)
laser_combo_num = len(LASER_LEAD_TIME) * len(LASER_DURATION)
total_episodes = flare_combo_num * laser_combo_num

print(f"总共有 {total_episodes} 个仿真回合")
for laser_start_time in LASER_LEAD_TIME:
    for laser_duration in LASER_DURATION:
        for start in start_times:
            for group_num in group_nums:
                for group_interval in group_intervals:
                    flare_times = generate_flare_release_times(start, group_num, group_interval)
                    miss = []
                    for i_episode in range(1):
                        episode_start_time = time.time()
                        action_list = []
                        distance_to_flare = []  # 初始化

                        done_eval = False
                        observation_eval = np.array(env.reset())
                        reward_sum = 0
                        t = 0
                        step = 0
                        reward_eval = 0
                        reward_4 = 0

                        while not done_eval:
                            if t % (round(env.dt_dec / env.dt_normal)) == 0:
                                action_eval = rule_based_action(env, laser_start_time, laser_duration, flare_times)

                                if action_eval[0] == 1:
                                    R_vec = env.x_target_now[0:3] - env.x_missile_now[3:6]
                                    R_rel = np.linalg.norm(R_vec)
                                    distance_to_flare.append(R_rel)


                                action_list.append(np.array([round(env.t_now, 2), action_eval[0], action_eval[1]]))

                                observation_eval, reward_eval, done_eval, reward_4, _ = env.step(action_eval)
                                reward_sum += reward_eval + reward_4
                                t += 1
                                step += 1
                            else:
                                action_eval1 = np.array([0, action_eval[1]])
                                observation_eval, reward_eval, done_eval, info, _ = env.step(action_eval1)
                                t += 1

                            if done_eval:
                                print(np.array(action_list))
                                miss.append(env.miss_distance)  # 保存脱靶量
                                episode += 1

                                distance_to_flare_rounded = [round(d, 2) for d in distance_to_flare]
                                # 记录本轮的参数与结果
                                records.append({
                                    "仿真回合数episode": episode,
                                    "诱饵弹释放组数group_num": len(flare_times),
                                    "诱饵弹释放时间flare_times": flare_times,
                                    "诱饵弹开始释放时机flare_begin_release_time": flare_times[0],
                                    "诱饵弹释放组间隔group_interval": group_interval,
                                    "激光定向干扰开始时机laser_start": laser_start_time,
                                    "激光定向干扰持续时间laser_duration": laser_duration,
                                    "脱靶量miss_distance": round(env.miss_distance, 2),
                                    "回合时长episode_duration": round(env.t_now, 2),
                                    "干扰是否成功success": int(env.success),
                                    "诱饵弹释放时机弹距离missile_to_flare_distance": distance_to_flare_rounded if distance_to_flare is not None else None
                                })

                                print("Episode {} finished after {} steps, 仿真时间 t = {}s, 脱靶量 {:.2f}m".format(
                                        episode, step + 1, round(env.t_now, 2), env.miss_distance))

                                # env.render()        #这个是可视化的  自己写的环境可以注释掉

                                if env.success:
                                    success_num += 1
                                break

# from datetime import datetime
# now_str = datetime.now().strftime("%Y%m%d_%H_%M_%S")
# # csv_filename = f"simulation_results_{now_str}.csv"
# csv_filename = os.path.join("CSV", f"simulation_results_{now_str}.csv")
# # csv_filename = "simulation_results.csv"
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=["episode", "laser_start","laser_duration",
#                                               "group_num", "flare_times","flare_begin_release_time", "group_interval", "miss_distance", "success",
#                                               "episode_duration", "missile_to_flare_distance"])
#     writer.writeheader()
#     for record in records:
#         writer.writerow(record)
#
# print(f"干扰仿真结果已保存至: {csv_filename}")

import pandas as pd
import os
from datetime import datetime

# 创建文件夹
os.makedirs("Excel", exist_ok=True)

# 当前时间作为文件名
now_str = datetime.now().strftime("%Y%m%d_%H_%M_%S")
excel_filename = os.path.join("Excel", f"simulation_results_{now_str}.xlsx")

# 用 pandas 保存 Excel
df = pd.DataFrame(records)  # 你的 records 列表是字典组成的，适合直接转换为 DataFrame
df.to_excel(excel_filename, index=False)

print(f"干扰仿真结果已保存为 Excel 文件：{excel_filename}")
