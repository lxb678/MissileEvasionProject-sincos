# -*- coding: utf-8 -*-
# @Time:    29/4/2025 下午9:19
# @Author:  Zeming M
# @File:    Actor
# @IDE:     PyCharm
import torch
from torch.nn import *
from torch.distributions import Bernoulli
from PPO_model.Config import *
from torch.optim import lr_scheduler
class Actor(Module):
    '''
    Actor网络
    '''
    def __init__(self):
        super(Actor,self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        self.output_dim = ACTOR_PARA.output_dim
        self.init_model()
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)

        # 初始化 LinearLR 学习率调度器
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )

        self.to(ACTOR_PARA.device)


    def init_model(self):
        self.network = Sequential()
        for i in range(len(ACTOR_PARA.model_layer_dim) + 1):  # 模型的每层维度在config里边
            if i == 0:  # 第一层输入
                self.network.add_module('fc_{}'.format(i), Linear(self.input_dim, ACTOR_PARA.model_layer_dim[0]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())


            elif i < len(ACTOR_PARA.model_layer_dim):
                self.network.add_module('fc_{}'.format(i),
                                        Linear(ACTOR_PARA.model_layer_dim[i - 1], ACTOR_PARA.model_layer_dim[i]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())


            else:  # 最后一层输出
                self.network.add_module('fc_{}'.format(i),
                                        Linear(ACTOR_PARA.model_layer_dim[-1], self.output_dim))
                # self.network.add_module('Sigmoid_{}'.format(i), Sigmoid())#输出每一种动作的概率（0~1）


    def forward(self,obs):
        obs = check(obs).to(**ACTOR_PARA.tpdv)   #统一数据格式
        #discrete_prob = self.network(obs)        #输出动作的概率
        output = self.network(obs)
        # if obs[7] == 0:
        #     available = torch.finfo(torch.float32).min
        #     output[0] += available
        #mask 红外诱饵弹
        if len(obs[7].shape) == 0:  # 处理单个元素的情况
            if obs[7] == 0:
                available = torch.finfo(torch.float32).min
                output[0] += available
        else:
            for i, o_ir in enumerate(obs[7]):
                if o_ir == 0:
                    available = torch.finfo(torch.float32).min
                    output[i][0] += available
        #mask 激光定向干扰
        # if len(obs[0].shape) == 0:  # 处理单个元素的情况
        #     if obs[0] < 0.5:
        #         available = torch.finfo(torch.float32).min
        #         output[1] += available
        #     else:
        #         available = torch.finfo(torch.float32).max
        #         output[1] += available
        # else:
        #     for i, dis in enumerate(obs[0]):
        #         if dis < 0.5:
        #             available = torch.finfo(torch.float32).min
        #             output[i][1] += available
        #         else:
        #             available = torch.finfo(torch.float32).max
        #             output[i][1] += available

        discrete_prob = torch.nn.Sigmoid()(output)    #若当前不能投、不能开启输出是0，available=torch.finfo(torch.float32).min
        discrete_action_dist = Bernoulli(probs=discrete_prob)  #动作的分布  二元动作则为伯努利分布
        return discrete_action_dist
