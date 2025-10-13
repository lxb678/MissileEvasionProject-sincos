import torch
from torch.nn import *
# <<< 更改 >>> 不再需要伯努利分布
from torch.distributions import Categorical, TransformedDistribution
from torch.distributions import Normal, TanhTransform
from Interference_code.PPO_model.PPO_evasion_onlymaneuver.Config import *
from Interference_code.PPO_model.PPO_evasion_onlymaneuver.Buffer import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time

##PPO算法：解决梯度爆炸   动作采用tanh缩放的话，经验池存储原始动作u，可以不用雅可比修正动作对数概率log. （用了雅可比修正也不会爆炸，暂时不清楚为什么），如果用反tanh算原始动作u就会有误差，就会梯度爆炸
## 如果动作为加速度，观测状态中有速度，可以对速度进行裁剪，不会爆炸；如果观测状态没有速度，对环境的速度进行裁剪会爆炸，因为同一观测状态导致不同结果
# 优势标准化不用了、Minibatch可以留

# --- 动作空间配置 ---
# <<< 更改 >>> 定义为纯连续动作空间
CONTINUOUS_DIM = 4  # 保持不变: 油门, 升降舵, 副翼, 方向舵
DISCRETE_DIM = 0  # 离散动作维度设为 0
# 定义连续动作的键名，用于后续的动作缩放
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']

# --- 动作范围定义 (已更新) ---
# <<< 更改 >>> 移除离散动作 'flare' 的定义
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},  # 油门指令范围 [0, 1]
    'elevator': {'low': -1.0, 'high': 1.0},  # 升降舵指令范围 [-1, 1]
    'aileron': {'low': -1.0, 'high': 1.0},  # 副翼指令范围 [-1, 1]
    'rudder': {'low': -1.0, 'high': 1.0},  # 方向舵指令范围 [-1, 1]
}


class Actor(Module):
    """
   Actor 网络 (策略网络) - 纯连续动作空间版本
   """

    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # <<< 更改 >>> output_dim 现在只包含连续动作的参数
        # 输出维度 = 连续动作数量 * 2 (均值mu, 标准差log_std)
        self.output_dim = CONTINUOUS_DIM * 2
        # 定义标准差的对数值 log_std 的范围，防止其过大或过小导致数值不稳定
        self.log_std_min = -20.0
        self.log_std_max = 2.0  # 2.0
        # 初始化网络模型、优化器和学习率调度器
        self.init_model()
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def init_model(self):
        """初始化神经网络结构"""
        self.network = Sequential()
        layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim + [self.output_dim]
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            # 最后一层是输出层，不需要激活函数
            if i < len(layers_dims) - 2:
                self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())

    def forward(self, obs):
        """
       前向传播方法 (纯连续版本)

       Args:
           obs (torch.Tensor): 输入的状态观测值

       Returns:
           torch.distributions.Normal: 用于连续动作的【基础】正态分布对象 (pre-tanh)
       """
        # 注意：这里的 obs 是已经转换为 tensor 的
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        # 如果输入是单个样本，增加一个批次维度
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 获取神经网络的原始输出 (现在只包含连续动作参数)
        output = self.network(obs_tensor)

        # <<< 删除 >>> 移除所有与离散动作和动作掩码相关的代码

        # 2. 创建连续动作的【基础】正态分布
        mu, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # 限制 log_std 范围
        std = torch.exp(log_std)
        continuous_base_dist = Normal(mu, std)

        # <<< 更改 >>> 只返回连续动作的分布
        return continuous_base_dist


# Critic 类的定义完全保持不变
class Critic(Module):
    """
   Critic 网络 (价值网络)

   该网络负责评估当前状态(observation)的价值 V(s)，即从当前状态开始，
   遵循当前策略所能获得的期望回报。它的输出是一个标量。
   """

    def __init__(self):
        super(Critic, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.init_model()
        self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)

    def init_model(self):
        """初始化神经网络结构"""
        self.network = Sequential()
        for i in range(len(CRITIC_PARA.model_layer_dim) + 1):
            if i == 0:
                self.network.add_module('fc_{}'.format(i), Linear(self.input_dim, CRITIC_PARA.model_layer_dim[0]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())
            elif i < len(CRITIC_PARA.model_layer_dim):
                self.network.add_module('fc_{}'.format(i),
                                        Linear(CRITIC_PARA.model_layer_dim[i - 1], CRITIC_PARA.model_layer_dim[i]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())
            else:
                self.network.add_module('fc_{}'.format(i), Linear(CRITIC_PARA.model_layer_dim[-1], self.output_dim))

    def forward(self, obs):
        """前向传播，计算状态价值"""
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


class PPO_continuous(object):
    """
   PPO 智能体主类

   该类整合了 Actor 和 Critic 网络，并实现了 PPO 算法的核心逻辑，
   包括动作选择、经验存储、优势计算和网络更新。
   """

    def __init__(self, load_able: bool, model_dir_path: str = None):
        super(PPO_continuous, self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()
        self.buffer = Buffer()
        # PPO 算法超参数
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0

        self.training_start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../save/save_evade_onlymaneuver"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)

        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../../test/test_evade_onlymaneuver")

    def load_models_from_directory(self, directory_path: str):
        """
        从指定的文件夹路径加载模型，能自动识别多种命名格式。
        - 格式1 (带前缀): "prefix_Actor.pkl", "prefix_Critic.pkl"
        - 格式2 (无前缀): "Actor.pkl", "Critic.pkl"
        """
        if not os.path.isdir(directory_path):
            print(f"[错误] 模型加载失败：提供的路径 '{directory_path}' 不是一个有效的文件夹。")
            return

        files = os.listdir(directory_path)
        actor_files_with_prefix = [f for f in files if f.endswith("_Actor.pkl")]
        if len(actor_files_with_prefix) > 0:
            actor_filename = actor_files_with_prefix[0]
            prefix = actor_filename.replace("_Actor.pkl", "")
            critic_filename = f"{prefix}_Critic.pkl"
            print(f"  - 检测到前缀 '{prefix}'，准备加载模型...")
            if critic_filename in files:
                actor_full_path = os.path.join(directory_path, actor_filename)
                critic_full_path = os.path.join(directory_path, critic_filename)
                try:
                    self.Actor.load_state_dict(torch.load(actor_full_path, map_location=ACTOR_PARA.device))
                    print(f"    - 成功加载 Actor: {actor_full_path}")
                    self.Critic.load_state_dict(torch.load(critic_full_path, map_location=CRITIC_PARA.device))
                    print(f"    - 成功加载 Critic: {critic_full_path}")
                    return
                except Exception as e:
                    print(f"    - [错误] 加载带前缀的模型时失败: {e}")
            else:
                print(f"    - [警告] 找到了 '{actor_filename}' 但未找到对应的 '{critic_filename}'。")

        if "Actor.pkl" in files and "Critic.pkl" in files:
            print("  - 检测到无前缀格式，准备加载 'Actor.pkl' 和 'Critic.pkl'...")
            actor_full_path = os.path.join(directory_path, "Actor.pkl")
            critic_full_path = os.path.join(directory_path, "Critic.pkl")
            try:
                self.Actor.load_state_dict(torch.load(actor_full_path, map_location=ACTOR_PARA.device))
                print(f"    - 成功加载 Actor: {actor_full_path}")
                self.Critic.load_state_dict(torch.load(critic_full_path, map_location=CRITIC_PARA.device))
                print(f"    - 成功加载 Critic: {critic_full_path}")
                return
            except Exception as e:
                print(f"    - [错误] 加载无前缀模型时失败: {e}")

        print(f"[错误] 模型加载失败：在文件夹 '{directory_path}' 中未找到任何有效的 Actor/Critic 模型对。")

    def store_experience(self, state, action, probs, value, reward, done):
        """
        存储经验到 Buffer，并在存储前进行数值检查。
        """
        if not np.all(np.isfinite(probs)):
            print("=" * 50)
            print(f"!!! 严重错误: 在 log_prob 中检测到非有限值 (NaN/Inf) !!!")
            print(f"Log_prob 值: {probs}")
            print(f"导致错误的状态: {state}")
            print(f"导致错误的动作: {action}")
            print("=" * 50)
            raise ValueError("在 log_prob 中检测到 NaN/Inf！")

        self.buffer.store_transition(state, value, action, probs, reward, done)

    def scale_action(self, action_cont_tanh):
        """
        将连续动作从 [-1, 1] 映射到环境的实际范围。
        """
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)

        return scaled_action

    def check_numerics(self, name, tensor, state=None, action=None, threshold=1e4):
        """检查 tensor 是否存在 NaN、Inf 或异常大值"""
        arr = tensor.detach().cpu().numpy()
        if not np.all(np.isfinite(arr)):
            print("=" * 50)
            print(f"[数值错误] {name} 出现 NaN/Inf")
            print(f"值: {arr}")
            if state is not None: print(f"对应的 state: {state}")
            if action is not None: print(f"对应的 action: {action}")
            print("=" * 50)
            raise ValueError(f"NaN/Inf detected in {name}")

        if np.any(np.abs(arr) > threshold):
            print("=" * 50)
            print(f"[警告] {name} 数值过大 (> {threshold})")
            print(f"最大值: {arr.max()}, 最小值: {arr.min()}")
            if state is not None: print(f"对应的 state: {state}")
            if action is not None: print(f"对应的 action: {action}")
            print("=" * 50)

    def choose_action(self, state, deterministic=False):
        """
        根据当前状态选择动作。(纯连续版本)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # <<< 更改 >>> Actor 现在只返回一个分布
            continuous_base_dist = self.Actor(state_tensor)

            # --- 连续动作处理 (逻辑不变) ---
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            action_cont_tanh = torch.tanh(u)
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # <<< 删除 >>> 移除所有离散动作处理逻辑

            # --- 组合与输出 ---
            # <<< 更改 >>> 总的 log_prob 就是连续动作的 log_prob
            total_log_prob = log_prob_cont
            self.check_numerics("total_log_prob", total_log_prob, state_tensor)

            # <<< 更改 >>> 存储到 Buffer 的动作就是原始的 u
            action_to_store = u

            # <<< 更改 >>> 发送到环境的最终动作就是缩放后的连续动作
            env_action_cont = self.scale_action(action_cont_tanh)
            final_env_action_tensor = env_action_cont

        # 将 Tensors 转换为 Numpy 数组
        action_to_store_np = action_to_store.cpu().numpy()
        log_prob_to_store_np = total_log_prob.cpu().numpy()
        final_env_action_np = final_env_action_tensor.cpu().numpy()

        # 如果输入不是批处理，则移除批次维度
        if not is_batch:
            final_env_action_np = final_env_action_np[0]
            action_to_store_np = action_to_store_np[0]
            log_prob_to_store_np = log_prob_to_store_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np

    def get_value(self, state):
        """使用 Critic 网络获取给定状态的价值。"""
        with torch.no_grad():
            value = self.Critic(state)
        return value

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        """
        计算广义优势估计 (Generalized Advantage Estimation, GAE)。
        """
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            done_mask = 1.0 - int(dones[t])
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantage[t] = gae
        return advantage

    def learn(self):
        """
        执行 PPO 的学习和更新步骤。(纯连续版本)
        """
        torch.autograd.set_detect_anomaly(True)
        states, values, actions, old_probs, rewards, dones = self.buffer.sample()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'adv_targ': [], 'ratio': []}

        for _ in range(self.ppo_epoch):
            for batch in self.buffer.generate_batches():
                state = check(states[batch]).to(**ACTOR_PARA.tpdv)
                # <<< 更改 >>> action_batch 现在只包含 u_cont
                action_batch = check(actions[batch]).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_probs[batch]).to(**ACTOR_PARA.tpdv).view(-1)
                advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv).view(-1, 1)

                # <<< 更改 >>> 从 Buffer 中获取的动作就是 u
                u_from_buffer = action_batch
                # <<< 删除 >>> 不再需要分离离散动作

                ######################### Actor 训练 #########################
                # <<< 更改 >>> Actor 只返回一个分布
                new_cont_base_dist = self.Actor(state)

                # <<< 更改 >>> 重新计算 log_prob
                new_log_prob_cont = new_cont_base_dist.log_prob(u_from_buffer).sum(dim=-1)
                new_prob = new_log_prob_cont

                # <<< 更改 >>> 计算策略熵
                entropy_cont = new_cont_base_dist.entropy().sum(dim=-1)
                total_entropy = entropy_cont.mean()

                # 计算重要性采样比率 ratio (逻辑不变)
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                self.check_numerics("new_prob", new_prob, state, action_batch)
                self.check_numerics("ratio", ratio, state, action_batch)

                # 计算 PPO 的 clip 代理目标函数 (逻辑不变)
                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # Actor 梯度更新 (逻辑不变)
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                ######################### Critic 训练 #########################
                # (Critic 训练部分完全不需要修改)
                old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                return_ = advantage + old_value_from_buffer
                new_value = self.Critic(state)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # 记录训练信息
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())

        self.buffer.clear_memory()
        for key in train_info:
            train_info[key] = np.mean(train_info[key])
        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        self.save()
        return train_info

    # --- 实用方法 ---
    def prep_training_rl(self):
        """将网络设置为训练模式"""
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        """将网络设置为评估模式"""
        self.Actor.eval()
        self.Critic.eval()

    def save(self, prefix=""):
        """
        将模型保存到以训练开始时间命名的专属文件夹中。
        """
        try:
            os.makedirs(self.run_save_dir, exist_ok=True)
            print(f"模型将被保存至: {self.run_save_dir}")
        except Exception as e:
            print(f"创建模型文件夹 {self.run_save_dir} 失败: {e}")
            return

        for net in ['Actor', 'Critic']:
            try:
                filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"
                full_path = os.path.join(self.run_save_dir, filename)
                torch.save(getattr(self, net).state_dict(), full_path)
                print(f"  - {filename} 保存成功。")
            except Exception as e:
                print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")