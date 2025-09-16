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
from PPO_model.PPO_Continuous import *
from PPO_model.Config import *
from torch.utils.tensorboard import SummaryWriter
#from env.AirCombatEnv import *
from env.AirCombatEnv6_maneuver_flare import *
import time

LOAD_ABLE = False  #是否使用save文件夹中的模型

# <<<--- Tacview 可视化开关 ---<<<
# 将此项设为 True 即可在训练时开启 Tacview
TACVIEW_ENABLED_DURING_TRAINING = False
# ---
def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    #env.action_space.seed(seed)   #可注释
    #env.reset(seed=seed)          #可注释
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir= 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
writer = SummaryWriter(log_dir='log/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))


env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)   #环境 后续用自己写的环境替换  不使用gym 这里只是作为验证
set_seed(env)   #设置环境的随机种子，如果这里报错  注释掉函数中标记的部分
agent = PPO_continuous(LOAD_ABLE)
global_step = 0               #总交互步数
success_num = 0
for i_episode in range(100000):
    observation = np.array(env.reset())    #自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态

    # <<<--- 在这里检查 ---<<<
    if np.isnan(observation).any():
        print("!!! FATAL: env.reset() returned NaN!")
        exit()  # 或者 raise an error

    # rewards = 0
    done = False
    step = 0
    action_list = []
    for t in range(10000):

        #env.render()        #这个是可视化的  自己写的环境可以注释掉
        #print(round(env.t_now, 2))
        if t % (round(env.dt_dec/env.dt_normal)) == 0:
            if t != 0:
                # reward = (reward + 10.0) / 20.0
                agent.buffer.store_transition(state, value, action_tanh, prob, reward, done)     #收集经验
                # rewards = 0
                global_step += 1
                step += 1
            agent.prep_eval_rl()    #交互经验的时候不会传梯度
            with torch.no_grad():   #交互时梯度不回传
                env_action, action_tanh, prob = agent.choose_action(observation)
                # env_action = env_action[0]

                # 2. 打印出即将送入环境的动作的所有信息
                # print("\n" + "=" * 50)
                # print(f"DEBUG: 即将执行 env.step(env_action)")
                # print(f"  - env_action 的类型: {type(env_action)}")
                # print(f"  - env_action 的形状: {env_action.shape}")
                # print(f"  - env_action 的内容: {env_action}")
                # print("=" * 50 + "\n")
                # --- 调试代码结束 ---
                # action = action.cpu().detach().numpy()
                # action_list.append(np.array([round(env.t_now, 2), action[0], action[1]]))
                # prob = prob.cpu().detach().numpy()
                value = agent.get_value(observation).cpu().detach().numpy()
                state = observation

            observation, reward, done, reward_4,_ = env.step(env_action)
            # rewards = reward
            # agent.buffer.store_transition(state, value, action, prob, rewards, done)  # 收集经验
            # if done:
            #     agent.buffer.store_transition(state, value, action, prob, reward, done)  # 收集经验
            #     # rewards = 0
            #     global_step += 1
            #     step += 1
            #     print("Episode {} finished after {} timesteps,仿真时间t = {}".format(i_episode+1, step+1,round(env.t_now, 2)))
            #     if env.success:
            #         success_num += 1
            #     if (i_episode + 1) % 100 == 0:
            #         print("每一百回合飞机存活次数{} ".format(success_num))
            #         success_num = 0
            #     break
        else:
            action1 = np.array([env_action[0], env_action[1], env_action[2], 0])
            #rewards += reward
            observation, reward, done, info, _ = env.step(action1)
            # rewards = reward
        reward += reward_4
        if done:
            agent.buffer.store_transition(state, value, action_tanh, prob, reward, done)  # 收集经验
            # rewards = 0
            global_step += 1
            step += 1
            print("Episode {} finished after {} timesteps,仿真时间t = {}s".format(i_episode+1, step+1,round(env.t_now, 2)))
            if env.success:
                success_num += 1
            if (i_episode + 1) % 100 == 0:
                print("每一百回合飞机存活次数{} ".format(success_num))
                if success_num >= 80:
                    # agent.save()
                    agent.save(f"success_{success_num}_ep{i_episode + 1}")
                writer.add_scalar('success_num',
                                  success_num,
                                  global_step=global_step
                                  )
                success_num = 0
            break
    # print(np.array(action_list))
    #env.render()        #这个是可视化的  自己写的环境可以注释掉
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
            observation_eval = np.array(env.reset())  # 自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态
            reward_sum = 0
            step = 0
            reward_eval = 0
            reward_4 = 0
            for t in range(10000):
                if t % (round(env.dt_dec/env.dt_normal)) == 0:
                    # dist = agent.Actor(observation_eval)  # 在验证时采用确定的策略，而不是采样
                    # action_eval_tanh = dist.mean
                    # action_eval = agent.scale_action(action_eval_tanh)
                    # action_eval = action_eval.cpu().detach().numpy()
                    # action_eval = action_eval[0]
                    action_eval, _, _ = agent.choose_action(observation_eval, deterministic=True)

                    # reward = (reward_eval + reward_4 + 10.0) / 20.0
                    # reward_sum += reward
                    reward_sum += reward_eval + reward_4

                    observation_eval, reward_eval, done_eval, reward_4, _ = env.step(action_eval)
                    # reward_sum += reward_eva
                    step += 1
                else:
                    # action_eval1 = np.array([0, action_eval[1]])
                    action_eval1 = np.array([action_eval[0], action_eval[1], action_eval[2], 0])
                    observation_eval, reward_eval, done_eval, info, _ = env.step(action_eval1)
                    # reward_sum += reward_eval
                    # step += 1
                if done_eval:
                    reward_sum += reward_eval
                    break

            writer.add_scalar('reward_sum',
                              reward_sum,
                              global_step=global_step
                              )

#env.close()