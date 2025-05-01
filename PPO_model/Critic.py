# -*- coding: utf-8 -*-
# @Time:    29/4/2025 下午9:19
# @Author:  Zeming M
# @File:    Critic
# @IDE:     PyCharm
import torch
from torch.nn import *
from torch.distributions import Categorical
from PPO_model.Config import *


class Critic(Module):
    '''
    Critic网络
    '''

    def __init__(self):
        super(Critic, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.init_model()
        self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        self.to(CRITIC_PARA.device)

    def init_model(self):
        self.network = Sequential()
        for i in range(len(CRITIC_PARA.model_layer_dim) + 1):  # 模型的每层维度在config里边
            if i == 0:  # 第一层输入
                self.network.add_module('fc_{}'.format(i), Linear(self.input_dim, CRITIC_PARA.model_layer_dim[0]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())


            elif i < len(CRITIC_PARA.model_layer_dim):
                self.network.add_module('fc_{}'.format(i),
                                        Linear(CRITIC_PARA.model_layer_dim[i - 1], CRITIC_PARA.model_layer_dim[i]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())


            else:  # 最后一层输出  维度是1
                self.network.add_module('fc_{}'.format(i),
                                        Linear(CRITIC_PARA.model_layer_dim[-1], self.output_dim))

    def forward(self, obs):
        obs = check(obs).to(**CRITIC_PARA.tpdv)  # 统一数据格式
        value = self.network(obs)
        return value