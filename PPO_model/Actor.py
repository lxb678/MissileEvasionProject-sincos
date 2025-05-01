# -*- coding: utf-8 -*-
# @Time:    29/4/2025 下午9:19
# @Author:  Zeming M
# @File:    Actor
# @IDE:     PyCharm
import torch
from torch.nn import *
from torch.distributions import Bernoulli
from PPO_model.Config import *
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
                self.network.add_module('Sigmoid_{}'.format(i), Sigmoid())#输出每一种动作的概率（0~1）


    def forward(self,obs):
        obs = check(obs).to(**ACTOR_PARA.tpdv)   #统一数据格式
        discrete_prob = self.network(obs)        #输出动作的概率
        discrete_action_dist = Bernoulli(probs=discrete_prob)  #动作的分布  二元动作则为伯努利分布
        return discrete_action_dist
