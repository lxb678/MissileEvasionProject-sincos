# -*- coding: utf-8 -*-
# @Time:    29/4/2025 下午9:19
# @Author:  Zeming M
# @File:    Run
# @IDE:     PyCharm
# -*- coding: utf-8 -*-
# @Time:    1/5/2025 下午5:33
# @Author:  Zeming M
# @File:    env
# @IDE:     PyCharm
import gym
import torch
import numpy as np
import random
from PPO_model.PPO_algorithm import *
from torch.utils.tensorboard import SummaryWriter
import time

LOAD_ABLE = False  #是否使用save文件夹中的模型


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    env.action_space.seed(seed)   #可注释
    env.reset(seed=seed)          #可注释
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir= 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
writer = SummaryWriter(log_dir='log/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))


env = gym.make('CartPole-v0')   #环境 后续用自己写的环境替换  不使用gym 这里只是作为验证
set_seed(env)   #设置环境的随机种子，如果这里报错  注释掉函数中标记的部分
agent = PPO_discrete()
global_step = 0               #总交互步数
for i_episode in range(100000):
    observation = np.array(env.reset()[0])    #自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态
    for t in range(200):
        env.render()        #这个是可视化的  自己写的环境可以注释掉
        # print(observation)
        agent.prep_eval_rl()    #交互经验的时候不会传梯度
        with torch.no_grad():   #交互时梯度不回传
            action,prob = agent.choose_action(observation)
            action = action.cpu().detach().numpy()
            prob =  prob.cpu().detach().numpy()
            value = agent.get_value(observation).cpu().detach().numpy()
            state = observation
        observation, reward, done, info,_ = env.step(int(action[0]))
        agent.buffer.store_transition(state, value, action, prob, reward, done)     #收集经验
        global_step += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    if i_episode % 10 == 0 and i_episode != 0:  # 每交互10局训练一次
        print("train, global_step:{}".format(global_step))
        agent.prep_training_rl()
        train_info = agent.learn()  # 获得训练中的信息，用于绘制图表
        for item in list(train_info.keys()):
            writer.add_scalar(item,
                              train_info[item],
                              global_step=global_step
                              )
        #训练完之后，需要验证模型，绘制奖励曲线(这个测试环境的奖励曲线使用幕奖励总和，在项目中可以考虑使用幕平均奖励)
        agent.prep_eval_rl()
        print("eval, global_step:{}".format(global_step))
        with torch.no_grad():
            done_eval = False
            observation_eval = np.array(env.reset()[0])  # 自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态
            reward_sum = 0
            while (not done_eval):
                dist = agent.Actor(observation_eval)     #在验证时采用确定的策略，而不是采样
                action_eval  = (dist.mean >= 0.5).float()
                action_eval = action_eval.cpu().detach().numpy()
                observation_eval, reward_eval, done_eval, info, _ = env.step(int(action_eval[0]))
                reward_sum+=reward_eval
            writer.add_scalar('reward_sum',
                              reward_sum,
                              global_step=global_step
                              )






env.close()