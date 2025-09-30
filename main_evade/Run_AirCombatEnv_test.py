
import gym
import torch
import numpy as np
import random
from Interference_code.PPO_model.Hybrid_PPO_不修正 import *
from Interference_code.PPO_model.Hybrid_PPO_jsbsim import *
from Interference_code.PPO_model.Config import *
from torch.utils.tensorboard import SummaryWriter
#from env.AirCombatEnv import *
# from env.AirCombatEnv6_maneuver_flare import *
from Interference_code.env.missile_evasion_environment.missile_evasion_environment import *
from Interference_code.env.missile_evasion_environment_jsbsim.Vec_missile_evasion_environment_jsbsim import *
import time
import matplotlib.pyplot as plt

LOAD_ABLE = True  #是否使用save文件夹中的模型


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    #env.action_space.seed(seed)   #可注释
    #env.reset(seed=seed)          #可注释
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir= 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
#writer = SummaryWriter(log_dir='log/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))

# <<<--- Tacview 可视化开关 ---<<<
# 将此项设为 True 即可在训练时开启 Tacview
TACVIEW_ENABLED_DURING_TRAINING = True
env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)   #环境 后续用自己写的环境替换  不使用gym 这里只是作为验证
set_seed(env)   #设置环境的随机种子，如果这里报错  注释掉函数中标记的部分
agent = PPO_continuous(LOAD_ABLE)
success_num = 0

#训练完之后，需要验证模型，绘制奖励曲线(这个测试环境的奖励曲线使用幕奖励总和，在项目中可以考虑使用幕平均奖励)
agent.prep_eval_rl()
print("开始验证模型")
episode_times = []

episodes = 100  #总测试回合数
for i_episode in range(episodes):
    episode_start_time = time.time()  # 记录当前 episode 开始时间

    with torch.no_grad():
        done_eval = False
        observation_eval,_ = env.reset(seed=AGENTPARA.RANDOM_SEED)  # 自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态
        reward_sum = 0
        t = 0
        step = 0
        reward_eval = 0
        action_list = []
        reward_4 = 0

        while (not done_eval):
            action_eval,action_tanh, prob = agent.choose_action(observation_eval, deterministic=True)
            observation_eval, reward_eval, done_eval, _, info = env.step(action_eval)
            reward_sum += reward_eval
            t += 1
            step += 1
            if done_eval:
                episode_time = time.time() - episode_start_time  # 计算单个 episode 用时
                episode_times.append(episode_time)  # 保存用时
                print("Episode {} finished after {} timesteps, 仿真时间 t = {}s, 用时 {:.2f}s".format(
                    i_episode + 1, step, round(env.t_now, 2), episode_time))
                print("奖励：", reward_sum)
                if env.success:
                    success_num += 1
                # if (i_episode + 1) % 100 == 0:
                #     print("每一百回合飞机存活次数{} ".format(success_num))
                #     success_num = 0