# -*- coding: utf-8 -*-
import csv

import gym
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from env.AirCombatEnv可行域 import *
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
    if LASER_START_TIME <= round(t_now, 1) <= LASER_START_TIME + LASER_DURATION:
        action[1] = 1

    # === 红外诱饵弹释放逻辑（只看当前时间是否接近任何一个时间点） ===
    for release_time in FLARE_RELEASE_TIMES:
        if abs(t_now - release_time) < FLARE_TIME_WINDOW:
            action[0] = 1
            break  # 找到一个就退出循环

    return action

def generate_flare_times(start_time, group_num, interval=0.6):
    return [round(start_time + i * interval, 2) for i in range(group_num)]

def generate_flare_release_times_by_interval(start_time, group_num, interval_list):
    """
    根据起始时间、释放组数和不同组间隔，生成多个 FLARE_RELEASE_TIMES 列表。

    :param start_time: float，诱饵弹首次释放时间（秒）
    :param group_num: int，总共释放几组
    :param interval_list: list[float]，不同组间隔（秒）列表，用于批量生成不同策略
    :return: list[list[float]]，每个子列表为一个 FLARE_RELEASE_TIMES 序列
    """
    results = []
    for interval in interval_list:
        times = [round(start_time + i * interval, 2) for i in range(group_num)]
        results.append(times)
    return results

# start_time = 6.0
# group_num = 3
# interval_list = [0.2, 0.3, 0.5, 0.8]  # 生成多个间隔策略
#
# flare_release_times_list = generate_flare_release_times_by_interval(start_time, group_num, interval_list)

def generate_flare_release_times_by_groupnum(start_time, interval, max_group_num):
    """
    生成多个 FLARE_RELEASE_TIMES，组数从1到max_group_num，时间间隔固定。

    :param start_time: float，诱饵弹开始释放时间
    :param interval: float，组间时间间隔
    :param max_group_num: int，最大释放组数
    :return: list of list，包含不同组数对应的释放时间列表
    """
    results = []
    for group_num in range(1, max_group_num + 1):
        times = [round(start_time + i * interval, 2) for i in range(group_num)]
        results.append(times)
    return results

# start_time = 6.0
# interval = 0.3
# max_group_num = 4
#
# flare_release_times_list = generate_flare_release_times_by_groupnum(start_time, interval, max_group_num)

def generate_flare_release_times_by_start_times(group_num, interval, start_times):
    """
    给定释放组数、组间隔，以及多个起始释放时间，生成多个FLARE_RELEASE_TIMES。

    :param group_num: int，释放组数
    :param interval: float，组间时间间隔
    :param start_times: list of float，多个起始释放时间
    :return: list of list，每个内层list是一个释放时间列表
    """
    results = []
    for start_time in start_times:
        times = [round(start_time + i * interval, 2) for i in range(group_num)]
        results.append(times)
    return results

# group_num = 3
# interval = 0.3
# start_times = [5.5, 6.0, 6.5, 7.0]
#
# flare_release_times_list = generate_flare_release_times_by_start_times(group_num, interval, start_times)


def generate_uniform_laser_start_times(laser_start_min, laser_start_max, num_groups):
    """
    在区间[laser_start_min, laser_start_max]内均匀生成num_groups个激光开启时间

    :param laser_start_min: float，起始时间
    :param laser_start_max: float，结束时间
    :param num_groups: int，生成组数
    :return: list，激光开启时间列表
    """
    if num_groups == 1:
        return [(laser_start_min + laser_start_max) / 2]
    else:
        interval = (laser_start_max - laser_start_min) / (num_groups - 1)
        return [round(laser_start_min + i * interval, 2) for i in range(num_groups)]

# laser_start_min = 0.0
# laser_start_max = 2.0
# num_groups = 5
#
# LASER_LEAD_TIME = generate_uniform_laser_start_times(laser_start_min, laser_start_max, num_groups)
# print(LASER_LEAD_TIME)

def generate_uniform_laser_durations(duration_min, duration_max, num_groups):
    """
    在区间 [duration_min, duration_max] 内均匀生成 num_groups 个激光干扰持续时间（单位：秒）

    :param duration_min: float，最短持续时间
    :param duration_max: float，最长持续时间
    :param num_groups: int，生成的数量
    :return: list[float]，持续时间列表
    """
    if num_groups == 1:
        return [(duration_min + duration_max) / 2]
    else:
        interval = (duration_max - duration_min) / (num_groups - 1)
        return [round(duration_min + i * interval, 2) for i in range(num_groups)]

# LASER_DURATION_LIST = generate_uniform_laser_durations(1.0, 5.0, 5)

def generate_interval_list(min_interval=0.1, max_interval=1.0, num_intervals=10):
    """
    生成均匀分布的释放组间隔列表

    :param min_interval: float，最小组间隔（单位：秒）
    :param max_interval: float，最大组间隔（单位：秒）
    :param num_intervals: int，生成的间隔数量
    :return: list[float]，均匀间隔值
    """
    return np.round(np.linspace(min_interval, max_interval, num_intervals), 2).tolist()
# interval_list = generate_interval_list(min_interval=0.2, max_interval=1.0, num_intervals=5)

def generate_aligned_start_times(start=3.0, end=10.0, interval=0.6):
    """
    生成与决策步长对齐的诱饵弹开始释放时间
    """
    times = []
    t = start
    while t <= end:
        times.append(round(t, 2))
        t += interval
    return times
# start_times = generate_aligned_start_times(start=3.0, end=10.0, interval=0.6)

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

# 可配置参数.
group_nums = [2, 3, 4]
group_intervals = [0.6, 1.2]
start_times = generate_aligned_start_times(3.0, 8.0)
FLARE_RELEASE_TIMES = []

for group_num in group_nums:
    for group_interval in group_intervals:
        for start in start_times:
            flare_times = generate_flare_release_times(start, group_num, group_interval)
            FLARE_RELEASE_TIMES.append(flare_times)

print(FLARE_RELEASE_TIMES)

LASER_LEAD_TIME = [0.0]
LASER_DURATION = [0.0]

for flare_times in FLARE_RELEASE_TIMES:
    for laser_start_time in LASER_LEAD_TIME:
        for laser_duration in LASER_DURATION:
                miss = []
                for i_episode in range(1):
                    episode_start_time = time.time()
                    action_list = []

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
                            # print(np.array(action_list))
                            miss.append(env.miss_distance)  # 保存脱靶量
                            episode += 1

                            # 记录本轮的参数与结果
                            records.append({
                                "episode": episode,
                                "group_num": len(flare_times),
                                "flare_times": flare_times,
                                "laser_start": laser_start_time,
                                "laser_duration": laser_duration,
                                "miss_distance": round(env.miss_distance, 2)
                            })

                            print("Episode {} finished after {} steps, 仿真时间 t = {}s, 脱靶量 {:.2f}m".format(
                                    i_episode + 1, step + 1, round(env.t_now, 2), env.miss_distance))

                            # env.render()        #这个是可视化的  自己写的环境可以注释掉

                            if env.success:
                                success_num += 1
                            break

    csv_filename = "simulation_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "group_num", "flare_times", "laser_start",
                                                  "laser_duration", "miss_distance"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"干扰仿真结果已保存至: {csv_filename}")
