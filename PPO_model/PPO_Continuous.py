import torch
from torch.nn import *
from torch.distributions import Bernoulli, TransformedDistribution
from torch.distributions import Normal, TanhTransform
from PPO_model.Config import *
from PPO_model.Buffer import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
# 假设的配置参数
# --- 动作范围定义 ---
# 根据你的环境来定义每个动作的最小值和最大值
ACTION_RANGES = {
    'nx':       {'low': -1.0, 'high': 2.0},
    'nz':       {'low': -5.0, 'high': 9.0},
    'phi_cmd':  {'low': -np.pi, 'high': np.pi},   # 假设滚转指令是-90到+90度
    'flare':    {'low': 0.0,  'high': 1.0}        # 输出一个[0,1]的倾向值
}

# 这是一个辅助类，封装了 Tanh 变换的正态分布
class SquashedNormal(TransformedDistribution):
    """
    一个经过 Tanh 变换的正态分布。
    它的样本在 (-1, 1) 范围内，并且 log_prob() 会自动包含修正项。
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        # 基础分布是正态分布
        self.base_dist = Normal(loc, scale)
        # 变换列表，这里只有一个 TanhTransform
        transforms = [TanhTransform(cache_size=1)] # cache_size=1 提升性能
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        # 注意：变换后的均值不是 self.loc，而是经过变换后的期望值
        # 这里返回 tanh(self.loc) 作为一个近似
        return torch.tanh(self.loc)

class Actor(Module):
    '''
    Actor网络
    1. 动态生成均值和标准差。
    2. 输出一个能处理概率修正的分布对象。
    '''
    def __init__(self):
        super(Actor,self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        self.output_dim = ACTOR_PARA.output_dim
        # 构建网络主体
        self.init_model()

        # 输出层将输出 2 * output_dim 个值：均值(mu)和对数标准差(log_std)
        # 让标准差也依赖于状态，可以提供更灵活的探索策略
        # 网络将输出 output_dim * 2 的向量：前一半是均值，后一半是对数标准差
        # self.output_layer = torch.nn.Linear(ACTOR_PARA.model_layer_dim[-1], self.output_dim * 2)

        # 定义log_std的范围，防止其过大或过小导致数值不稳定
        self.log_std_min = -20
        self.log_std_max = 2

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
                                        Linear(ACTOR_PARA.model_layer_dim[-1], self.output_dim * 2))
                # self.network.add_module('Sigmoid_{}'.format(i), Sigmoid())#输出每一种动作的概率（0~1）

    def forward(self, obs):
        obs = check(obs).to(**ACTOR_PARA.tpdv)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 1. 获取网络原始输出
        output = self.network(obs)
        mu, log_std = output.chunk(2, dim=-1)

        # 2. 不做任何掩码，直接创建和返回分布
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return SquashedNormal(mu, std)


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

        # 初始化 LinearLR 学习率调度器
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )

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

class PPO_continuous(object):
    def __init__(self, load_able):
        '''

        :param load_able: 是否加载已经储存的模型
        '''
        super(PPO_continuous,self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()
        self.buffer =Buffer()
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        if load_able:
            for net in ['Actor', 'Critic']:
                try:
                    getattr(self, net).load_state_dict(
                        torch.load("save/" + net + '.pkl', weights_only=True))
                except:
                    print("load error")
            pass

        # if load_able:
        #     self.load_model_from_save_folder()

    def load_model_from_save_folder(self):
        save_dir = "test"
        files = os.listdir(save_dir)

        # 找唯一的 *_Actor.pkl 文件
        # ------- 情况 1：查找 *_Actor.pkl 文件 -------
        actor_files = [f for f in files if f.endswith("_Actor.pkl")]
        if len(actor_files) == 1:
            match = re.match(r"(.+)_Actor\.pkl", actor_files[0])
            if not match:
                print("文件名格式错误")
                return
            prefix = match.group(1)
            for net in ['Actor', 'Critic']:
                try:
                    filename = os.path.join(save_dir, f"{prefix}_{net}.pkl")
                    getattr(self, net).load_state_dict(torch.load(filename, weights_only=True))
                    print(f"成功加载模型: {filename}")
                except Exception as e:
                    print(f"加载失败: {filename}，原因: {e}")
            return  # 成功加载后退出

        # ------- 情况 2：查找老格式 Actor.pkl 和 Critic.pkl -------
        elif "Actor.pkl" in files and "Critic.pkl" in files:
            for net in ['Actor', 'Critic']:
                try:
                    filename = os.path.join(save_dir, f"{net}.pkl")
                    getattr(self, net).load_state_dict(torch.load(filename, weights_only=True))
                    print(f"成功加载旧格式模型: {filename}")
                except Exception as e:
                    print(f"加载失败: {filename}，原因: {e}")
            return  # 成功加载后退出

        # ------- 都不符合，报错 -------
        else:
            print("模型加载错误：未找到符合要求的模型文件，请确保 save/ 中存在一对 Actor/Critic 模型")
            return
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

    def scale_action(self, action_tanh):
        """
        将 tanh 压缩后的动作 [-1, 1] 缩放到环境的实际范围
        """
        # 将字典中的范围转换为tensor
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in ['nx', 'nz', 'phi_cmd', 'flare']], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in ['nx', 'nz', 'phi_cmd', 'flare']], **ACTOR_PARA.tpdv)

        # 线性缩放公式: a_scaled = low + (a_tanh + 1) * 0.5 * (high - low)
        scaled_action = lows + (action_tanh + 1.0) * 0.5 * (highs - lows)
        return scaled_action
    def choose_action(self,state,deterministic=False):
        '''
        返回当前状态的动作和策略分布
        :param state:
        :return:
        '''
        '''
        返回当前状态的动作（已缩放并处理），以及用于训练的对数概率
        '''
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # 1. 获取无条件的分布
            unconditional_dist = self.Actor(state_tensor)

            # 2. 采样或取均值
            if deterministic:
                action_tanh = unconditional_dist.mean
            else:
                action_tanh = unconditional_dist.rsample()

            # 3. 在采样出的 tensor 上应用掩码
            has_flares_info = state_tensor[:, -1]
            mask = (has_flares_info == 0)  # shape: [batch_size]

            # 使用索引来修改，这比 where 更直接
            if torch.any(mask):
                action_tanh[mask, 3] = -1.0

            # 4. 用被掩码后的动作，计算其在原始分布下的 log_prob
            # 这是完全合法的，我们只是在评估一个特定动作的概率
            log_prob_all_dims = unconditional_dist.log_prob(action_tanh)  # shape 应该是 [batch_size, 4]

            # 5. 将被掩码维度的 log_prob 贡献设为 0
            if torch.any(mask):
                log_prob_all_dims[mask, 3] = 0.0

            log_prob = log_prob_all_dims.sum(dim=-1)

            # 4. 将动作缩放到环境的实际范围
            env_action_tensor = self.scale_action(action_tanh)  # 使用之前定义的 scale_action 函数
            # 由于 Actor 内部的掩码，当无弹时，env_action 的第4维会自动是 0.0


        # 将 tensor 转为 numpy，用于存储和与环境交互
        action_to_store = action_tanh.cpu().numpy()
        log_prob_to_store = log_prob.cpu().numpy()

        # 准备最终用于环境的动作（将 flare 部分二值化）
        final_env_action = np.copy(env_action_tensor.cpu().numpy())
        if final_env_action.ndim == 1:
            final_env_action[3] = 1 if final_env_action[3] > 0.5 else 0
        else:
            final_env_action[:, 3] = (final_env_action[:, 3] > 0.5).astype(int)

        if not is_batch:
            final_env_action = final_env_action[0]
            action_to_store = action_to_store[0]
            # log_prob_to_store 已经是标量了

        # 返回给主循环的是二值化后的动作
        # 存入buffer的是连续值动作和对应的log_prob
        return final_env_action, action_to_store, log_prob_to_store


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
        # <<<--- 在 learn 的最开始启用异常检测 ---<<<
        torch.autograd.set_detect_anomaly(True)

        states, values, actions_tanh, old_probs, rewards, dones = (self.buffer.sample())  #获得经验池中全部的经验
        advantages = self.cal_gae(states, values, actions_tanh, old_probs, rewards, dones)

        # <<<--- 在这里加入优势标准化 ---<<<
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        # 1e-8 是为了防止除以零

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
                action_tanh = actions_tanh[batch]
                old_value = values[batch]
                old_prob = old_probs[batch]
                advantage = advantages[batch]
                state = check(state).to(**ACTOR_PARA.tpdv)
                action_tanh = check(action_tanh).to(**ACTOR_PARA.tpdv)
                old_value = check(old_value).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)
                advantage = check(advantage).to(**ACTOR_PARA.tpdv).unsqueeze(-1)

                #########################################Actor训练############################################
                dist = self.Actor(state)               #获得策略分布
                # <<<--- 核心简化：直接计算 new_prob，无需任何逆运算！---<<<
                # 因为 dist.log_prob() 的输入就是 tanh 后的动作
                # new_prob = dist.log_prob(action_tanh).sum(dim=-1)          #原动作在新的策略分布下的log_pi

                # <<<--- 在这里加入对 action_tanh 的裁剪 ---<<<
                # 为了数值稳定性，将 action_tanh 的值稍微向内挤压一下
                # 防止其值精确地等于 -1.0 或 1.0，导致 atanh(±1) = ±inf
                action_tanh_clipped = torch.clamp(action_tanh, -0.99999, 0.99999)
                # --- 修改结束 ---

                # 使用裁剪后的 action_tanh 来计算 log_prob
                log_prob_all_dims = dist.log_prob(action_tanh_clipped)

                # <<<--- 核心修改：同样地，正确计算 new_prob ---<<<
                # log_prob_all_dims = dist.log_prob(action_tanh)
                mask_indices = (state[:, -1] == 0)
                if torch.any(mask_indices):
                    log_prob_all_dims[mask_indices, 3] = 0.0
                new_prob = log_prob_all_dims.sum(dim=-1)
                # --- 修改结束 ---

                # entropy = dist.entropy().mean()            #策略熵
                # 正确的代码：我们计算其基础正态分布的熵
                # SquashedNormal 实例有一个 .base_dist 属性，指向它内部的 Normal 分布
                entropy = dist.base_dist.entropy().mean()

                # ratio = torch.exp(new_prob-old_prob)
                # <<<--- 在这里加入对 log_prob 差值的裁剪 ---<<<
                log_ratio = new_prob - old_prob
                # 将 log_ratio 裁剪到一个合理的范围，例如 [-20, 20]
                # exp(20) 已经是一个非常大的数了，但不会是 inf
                log_ratio_clipped = torch.clamp(log_ratio, -20.0, 20.0)
                ratio = torch.exp(log_ratio_clipped)


                surr1 = ratio * advantage
                surr2 = torch.clip(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage

                actor_loss=  -torch.min(surr1, surr2) - AGENTPARA.entropy * entropy#最小化损失，加符号后为最大化目标函数

                self.Actor.optim.zero_grad()
                actor_loss.mean().backward()

                # <<<--- 在这里加入梯度裁剪 ---<<<
                # torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=0.5)

                self.Actor.optim.step()
                #######################################Critic训练##############################################
                return_ = advantage + old_value   #目标值
                new_value = self.Critic(state)


                critic_loss = torch.nn.functional.mse_loss(new_value,return_)


                self.Critic.optim.zero_grad()
                critic_loss.backward()

                # <<<--- Critic 也需要梯度裁剪 ---<<<
                # torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)

                self.Critic.optim.step()

                # 学习率调度
                # self.Actor.actor_scheduler.step()
                # self.Critic.critic_scheduler.step()
                # self.total_steps += 1

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

        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
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
    # def save(self, prefix=""):
    #     for net in ['Actor', 'Critic']:
    #         try:
    #             filename = f"save/{prefix}_{net}.pkl" if prefix else f"save/{net}.pkl"
    #             torch.save(getattr(self, net).state_dict(), filename)
    #         except:
    #             print("write_error")
