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
import random
from PPO_model.PPO_algorithm import *
from PPO_model.Config import *
#from env.AirCombatEnv import *
from Interference_code.env.oldenv.AirCombatEnv4 import *

LOAD_ABLE = False  #是否使用save文件夹中的模型


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 设置随机种子 '''
    #env.action_space.seed(seed)   #可注释
    #env.reset(seed=seed)          #可注释
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#记录训练的损失等数值，用于绘制图表  使用tensorboard --logdir= 路径 的命令绘制 文件名是随机种子-训练日期-是否使用储存的模型
#writer = SummaryWriter(log_dir='log/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))


env = AirCombatEnv()   #环境 后续用自己写的环境替换  不使用gym 这里只是作为验证
set_seed(env)   #设置环境的随机种子，如果这里报错  注释掉函数中标记的部分
agent = PPO_discrete(LOAD_ABLE)
success_num = 0

#训练完之后，需要验证模型，绘制奖励曲线(这个测试环境的奖励曲线使用幕奖励总和，在项目中可以考虑使用幕平均奖励)
agent.prep_eval_rl()
print("开始验证模型")
for i_episode in range(100):
    with torch.no_grad():
        done_eval = False
        observation_eval = np.array(env.reset())  # 自己写的环境可以改为 env.reset()   这个函数返回初始化时的观测状态
        reward_sum = 0
        t = 0
        step = 0
        reward_eval = 0
        for t in range(10000):
            if t % (round(env.dt_dec/env.dt_normal)) == 0:
                dist = agent.Actor(observation_eval)     #在验证时采用确定的策略，而不是采样
                # action_eval = (dist.mean >= 0.5).float()
                # action_eval = action_eval.cpu().detach().numpy()
                action_eval = np.array([0, 0])
                reward_sum += reward_eval
                observation_eval, reward_eval, done_eval, info, _ = env.step(action_eval)
                # reward_sum += reward_eval
                step += 1
            else:
                action_eval1 = np.array([0, action_eval[1]])
                observation_eval, reward_eval, done_eval, info, _ = env.step(action_eval1)
                # reward_sum += reward_eval
            if done_eval:
                print("Episode {} finished after {} timesteps,仿真时间t = {}s".format(i_episode + 1, step + 1,
                                                                                      round(env.t_now, 2)))
                if env.success:
                    success_num += 1
                if (i_episode + 1) % 100 == 0:
                    print("每一百回合飞机存活次数{} ".format(success_num))
                    success_num = 0
                break
        env.render()
        # writer.add_scalar('reward_sum',
        #                   reward_sum,
        #                   global_step=global_step
        #                   )

#env.close()