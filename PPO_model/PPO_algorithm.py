# -*- coding: utf-8 -*-
# @Time:    29/4/2025 下午9:20
# @Author:  Zeming M
# @File:    PPO_algorithm
# @IDE:     PyCharm
import numpy as np
import torch

from PPO_model.Actor import *
from PPO_model.Critic import *
from PPO_model.Buffer import *

class PPO_discrete(object):
    def __init__(self,load_able = False):
        '''

        :param load_able: 是否加载已经储存的模型
        '''
        super(PPO_discrete,self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()
        self.buffer =Buffer()
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        if load_able:
            for net in ['Actor', 'Critic']:
                try:
                    getattr(self, net).load_state_dict(
                        torch.load("save/" + net + '.pkl', weights_only=True))
                except:
                    print("load error")
            pass
    def store_experience(self, state, action, probs, value, reward, done):
        '''
        储存经验到经验池
        :param state:
        :param action:
        :param probs:
        :param value:
        :param reward:
        :param done:
        :return:
        '''
        self.buffer.store_transition(state, value, action, probs, reward, done)

    def choose_action(self,state):
        '''
        返回当前状态的动作和策略分布
        :param state:
        :return:
        '''
        dist = self.Actor(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action, prob

    def get_value(self,state):
        '''
        获得当前的状态价值
        :param state:
        :return:
        '''
        value = self.Critic(state)
        return  value

    def cal_gae(self,states, values, actions, probs, rewards, dones):
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0  # 初始化 GAE

        # 从后往前计算 GAE
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            advantage[t] = gae

        return advantage


    def learn(self):
        '''
        模型训练阶段
        :return:
        '''
        states, values, actions, probs, rewards, dones = (self.buffer.sample())  #获得经验池中全部的经验
        advantages = self.cal_gae(states, values, actions, probs, rewards, dones)
        train_info = {}   #记录训练相关信息，后边使用tensorboard可以查看训练曲线
        train_info['critic_loss'] = []
        train_info['actor_loss'] = []
        train_info['dist_entropy'] = []
        train_info['adv_targ'] = []
        train_info['ratio'] = []
        for _ in range(self.ppo_epoch):
            batches = self.buffer.generate_batches()   #获得随机采样batch的索引值
            for batch in batches:
                state = states[batch]
                action = actions[batch]
                old_value = values[batch]
                old_prob = probs[batch]
                advantage = advantages[batch]
                state = check(state).to(**ACTOR_PARA.tpdv)
                action = check(action).to(**ACTOR_PARA.tpdv)
                old_value = check(old_value).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)
                advantage = check(advantage).to(**ACTOR_PARA.tpdv)

                #########################################Actor训练############################################
                dist = self.Actor(state)               #获得策略分布
                new_prob = dist.log_prob(action)          #原动作在新的策略分布下的log_pi
                entropy = dist.entropy().mean()            #策略熵
                ratio = torch.exp(new_prob-old_prob)

                surr1 = ratio * advantage
                surr2 = torch.clip(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage

                actor_loss=  -torch.min(surr1, surr2) - AGENTPARA.entropy * entropy#最小化损失，加符号后为最大化目标函数

                self.Actor.optim.zero_grad()
                actor_loss.mean().backward()
                self.Actor.optim.step()
                #######################################Critic训练##############################################
                return_ = advantage.unsqueeze(-1) + old_value   #目标值
                new_value = self.Critic(state)


                critic_loss = torch.nn.functional.mse_loss(new_value,return_)


                self.Critic.optim.zero_grad()
                critic_loss.backward()
                self.Critic.optim.step()
                ##############################################################################################
                train_info['critic_loss'].append(critic_loss.mean().cpu().detach().numpy())
                train_info['actor_loss'] = [actor_loss.mean().cpu().detach().numpy()]
                train_info['dist_entropy'] = [entropy.cpu().detach().numpy()]
                train_info['adv_targ'] = [advantage.mean().cpu().detach().numpy()]
                train_info['ratio'] = [ratio.mean().cpu().detach().numpy()]
        self.buffer.clear_memory()
        train_info['critic_loss'] = np.array(train_info['critic_loss']).mean()
        train_info['actor_loss'] = np.array(train_info['actor_loss']).mean()
        train_info['dist_entropy'] = np.array(train_info['dist_entropy']).mean()
        train_info['adv_targ'] = np.array(train_info['adv_targ']).mean()
        train_info['ratio'] = np.array(train_info['ratio']).mean()
        self.save()
        return  train_info

    def prep_training_rl(self):
        '''
        训练阶段  梯度可回传
        :return:
        '''
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        '''
        验证阶段 梯度不回传
        :return:
        '''
        self.Actor.eval()
        self.Critic.eval()

    def save(self):
        for net in ['Actor', 'Critic']:
            try:
                torch.save(getattr(self, net).state_dict(), "save/" + net  + ".pkl")
            except:
                print("write_error")

        pass





