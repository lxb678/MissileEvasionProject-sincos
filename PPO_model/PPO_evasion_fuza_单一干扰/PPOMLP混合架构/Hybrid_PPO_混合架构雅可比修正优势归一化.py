import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
# Normal 分布已不再需要
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.Buffer import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time
from torch.cuda.amp import GradScaler, autocast

# --- 动作空间配置 ---
# <<< 修改 >>> 连续动作维度设为 0，因为飞行控制已移交给 PID
CONTINUOUS_DIM = 0
# CONTINUOUS_ACTION_KEYS = [] # 不再需要

# <<< 保持 >>> 离散动作空间定义
DISCRETE_DIMS = {
    'flare_trigger': 1,  # 是否投放: 1个logit -> Bernoulli
    'salvo_size': 3,  # 数量
    'num_groups': 3,  # 组数
    'inter_interval': 3,  # 组间隔
}

# 计算离散部分总共需要的网络输出数量
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())  # 1 + 3 + 3 + 3 = 10

# <<< 修改 >>> Buffer 中存储的动作维度仅为离散动作的数量
TOTAL_ACTION_DIM_BUFFER = len(DISCRETE_DIMS)  # 4

# <<< 保持 >>> 离散动作映射表
DISCRETE_ACTION_MAP = {
    'salvo_size': [2, 3, 4],
    'num_groups': [2, 3, 4],
    'inter_interval': [0.2, 0.4, 0.6]
}


def init_weights(m):
    """通用权重初始化函数"""
    if isinstance(m, Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Actor(Module):
    """
    Actor 网络 (策略网络) - 纯离散动作版
    """

    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim

        # --- 网络架构定义 ---

        # 1. 共享基座网络 (Shared Base)
        # 假设 model_layer_dim = [256, 256, 256]，我们在第2层后拆分
        split_point = 2
        base_dims = ACTOR_PARA.model_layer_dim[:split_point]

        # 离散塔楼使用剩余的层配置
        discrete_tower_dims = ACTOR_PARA.model_layer_dim[split_point:]

        # 构建共享基座
        self.shared_base = Sequential()
        base_input_dim = self.input_dim
        for i, dim in enumerate(base_dims):
            self.shared_base.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
            self.shared_base.add_module(f'base_leakyrelu_{i}', LeakyReLU())
            base_input_dim = dim

        base_output_dim = base_dims[-1]

        # 2. 构建离散动作塔楼 (Discrete Tower)
        # <<< 修改 >>> 移除了 Continuous Tower
        self.discrete_tower = Sequential()
        tower_input_dim = base_output_dim
        for i, dim in enumerate(discrete_tower_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim

        # 3. 定义输出头 (Heads)
        # <<< 修改 >>> 移除了 Continuous Head
        discrete_tower_output_dim = discrete_tower_dims[-1] if discrete_tower_dims else base_output_dim
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)

        # 权重初始化
        self._init_weights()

        # 优化器
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # 特殊初始化：离散动作头
        nn.init.orthogonal_(self.discrete_head.weight, gain=0.01)
        nn.init.constant_(self.discrete_head.bias, 0)

    def forward(self, obs):
        """
        前向传播
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 共享特征
        base_features = self.shared_base(obs_tensor)

        # 2. 离散特征
        discrete_features = self.discrete_tower(base_features)

        # 3. 计算 logits
        all_disc_logits = self.discrete_head(discrete_features)

        # 4. 切分 logits
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        # 5. 动作掩码逻辑 (如果观测显示没有诱饵弹，禁止投放)
        # 假设 obs_tensor 的第 9 个特征 (索引为9) 是归一化后的诱饵弹数量
        # 注意：这里的索引 9 需要与 Environment 中的 _get_observation 对应
        has_flares_info = obs_tensor[:, 10]
        mask = (has_flares_info <= -0.99)  # 归一化后 -1 表示 0 发，给一点容差

        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            # 将对应样本的 logit 设为极小值，使 sigmoid 后概率接近 0
            trigger_logits_masked[mask] = -1e4

        # 6. 层级控制逻辑 (如果决定不投放，屏蔽后续参数的梯度)
        trigger_probs = torch.sigmoid(trigger_logits_masked)
        no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)

        salvo_size_logits_masked = salvo_size_logits.clone()
        num_groups_logits_masked = num_groups_logits.clone()
        inter_interval_logits_masked = inter_interval_logits.clone()

        if torch.any(no_trigger_mask):
            INF = 1e6
            NEG_INF = -1e6
            for logits_tensor in [salvo_size_logits_masked, num_groups_logits_masked, inter_interval_logits_masked]:
                logits_sub = logits_tensor[no_trigger_mask]
                if logits_sub.numel() > 0:
                    logits_sub[:] = NEG_INF  # 全部置为极小
                    logits_sub[:, 0] = INF  # 仅 index=0 置为极大 (默认值)
                    logits_tensor[no_trigger_mask] = logits_sub

        # 7. 创建分布对象
        distributions = {
            'trigger': Bernoulli(logits=trigger_logits_masked.squeeze(-1)),
            'salvo_size': Categorical(logits=salvo_size_logits_masked),
            'num_groups': Categorical(logits=num_groups_logits_masked),
            'inter_interval': Categorical(logits=inter_interval_logits_masked)
        }
        return distributions


class Critic(Module):
    """
    Critic 网络 (保持不变)
    """

    def __init__(self):
        super(Critic, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.network = Sequential()
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

        self._init_weights()

        self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        # Critic 输出层初始化
        output_layer = self.network[-1]
        if isinstance(output_layer, nn.Linear):
            nn.init.orthogonal_(output_layer.weight, gain=1.0)
            nn.init.constant_(output_layer.bias, 0)

    def forward(self, obs):
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


class PPO_discrete(object):
    """
    PPO 智能体 - 纯离散动作版
    """

    def __init__(self, load_able: bool, model_dir_path: str = None):
        super(PPO_discrete, self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()

        self.actor_scaler = GradScaler()
        self.critic_scaler = GradScaler()

        self.buffer = Buffer()
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        self.training_start_time = time.strftime("PPO_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../save/save_evade_fuza_单一干扰"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)

        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹 ---")

    def load_models_from_directory(self, directory_path: str):
        # (加载逻辑保持不变)
        if not os.path.isdir(directory_path):
            return
        files = os.listdir(directory_path)
        if "Actor.pkl" in files and "Critic.pkl" in files:
            try:
                self.Actor.load_state_dict(
                    torch.load(os.path.join(directory_path, "Actor.pkl"), map_location=ACTOR_PARA.device))
                self.Critic.load_state_dict(
                    torch.load(os.path.join(directory_path, "Critic.pkl"), map_location=CRITIC_PARA.device))
                print("    - 模型加载成功")
            except Exception as e:
                print(f"    - 模型加载失败: {e}")

    def store_experience(self, state, action, probs, value, reward, done):
        self.buffer.store_transition(state, value, action, probs, reward, done)

    def choose_action(self, state, deterministic=False):
        """
        选择动作
        注意：仅涉及离散动作采样，移除了所有连续动作相关代码
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # 1. 获取分布
            dists = self.Actor(state_tensor)

            # 2. 采样动作
            sampled_actions_dict = {}
            for key, dist in dists.items():
                if deterministic:
                    if isinstance(dist, Categorical):
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                    elif isinstance(dist, Bernoulli):
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                else:
                    sampled_actions_dict[key] = dist.sample()

            trigger_action = sampled_actions_dict['trigger']
            salvo_size_action = sampled_actions_dict['salvo_size']
            num_groups_action = sampled_actions_dict['num_groups']
            inter_interval_action = sampled_actions_dict['inter_interval']

            # 3. 计算离散动作总对数概率
            log_prob_disc = (dists['trigger'].log_prob(trigger_action) +
                             dists['salvo_size'].log_prob(salvo_size_action) +
                             dists['num_groups'].log_prob(num_groups_action) +
                             dists['inter_interval'].log_prob(inter_interval_action))

            total_log_prob = log_prob_disc

            # 4. 准备存入 Buffer 的动作向量 (仅离散索引)
            action_to_store = torch.stack([
                trigger_action, salvo_size_action,
                num_groups_action, inter_interval_action
            ], dim=-1).float()

            # 5. 准备环境动作 (应用置零逻辑，不投放时参数为0)
            zero_mask = (trigger_action == 0)

            env_salvo_size_action = salvo_size_action.clone()
            env_num_groups_action = num_groups_action.clone()
            env_inter_interval_action = inter_interval_action.clone()

            env_salvo_size_action[zero_mask] = 0
            env_num_groups_action[zero_mask] = 0
            env_inter_interval_action[zero_mask] = 0

            # 最终发送给环境的向量
            final_env_action_tensor = torch.stack([
                trigger_action, env_salvo_size_action,
                env_num_groups_action, env_inter_interval_action
            ], dim=-1).float()

        action_to_store_np = action_to_store.cpu().numpy()
        log_prob_to_store_np = total_log_prob.cpu().numpy()
        final_env_action_np = final_env_action_tensor.cpu().numpy()

        if not is_batch:
            final_env_action_np = final_env_action_np[0]
            action_to_store_np = action_to_store_np[0]
            log_prob_to_store_np = log_prob_to_store_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np

    def get_value(self, state):
        with torch.no_grad():
            value = self.Critic(state)
        return value

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        """计算 GAE"""
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
        PPO 更新逻辑 - 纯离散版
        """
        torch.autograd.set_detect_anomaly(True)
        states, values, actions, old_probs, rewards, dones = self.buffer.sample()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)

        # 维度对齐与归一化
        if advantages.ndim == 1: advantages = advantages.reshape(-1, 1)
        if values.ndim == 1: values = values.reshape(-1, 1)

        returns_np = advantages + values
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'ratio': []}

        for _ in range(self.ppo_epoch):
            for batch in self.buffer.generate_batches():
                state = check(states[batch]).to(**ACTOR_PARA.tpdv)
                action_batch = check(actions[batch]).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_probs[batch]).to(**ACTOR_PARA.tpdv).view(-1)
                advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                return_batch = check(returns_np[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                # --- 1. 从 Buffer 获取动作索引 ---
                # 注意：action_batch 的列顺序由 choose_action 中 action_to_store 决定
                discrete_actions_from_buffer = {
                    'trigger': action_batch[:, 0],
                    'salvo_size': action_batch[:, 1].long(),
                    'num_groups': action_batch[:, 2].long(),
                    'inter_interval': action_batch[:, 3].long(),
                }

                ######################### Actor 训练 #########################
                # 2. 新策略评估
                new_dists = self.Actor(state)

                # 3. 计算新 Log Prob (仅离散部分)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_disc

                # 4. 计算熵
                entropy_disc = sum(dist.entropy() for key, dist in new_dists.items())
                total_entropy = entropy_disc.mean()

                # 5. PPO Loss
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)

                # 增大熵系数鼓励探索
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 6. Actor 更新
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                ######################### Critic 训练 #########################
                new_value = self.Critic(state)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_batch)
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # 记录
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['ratio'].append(ratio.mean().item())

        self.buffer.clear_memory()
        for key in train_info: train_info[key] = np.mean(train_info[key])
        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        self.save()
        return train_info

    def prep_training_rl(self):
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        self.Actor.eval()
        self.Critic.eval()

    def save(self, prefix=""):
        try:
            os.makedirs(self.run_save_dir, exist_ok=True)
            os.makedirs(self.win_rate_dir, exist_ok=True)
        except Exception:
            pass

        target_dir = self.win_rate_dir if prefix else self.run_save_dir

        for net_name in ['Actor', 'Critic']:
            try:
                net_model = getattr(self, net_name)
                filename = f"{prefix}_{net_name}.pkl" if prefix else f"{net_name}.pkl"
                torch.save(net_model.state_dict(), os.path.join(target_dir, filename))
            except Exception:
                pass