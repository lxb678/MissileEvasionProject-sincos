# --- START OF FILE Hybrid_PPO_jsbsim_SeparateHeads.py ---

import torch
from torch.nn import *
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
from Interference_code.PPO_model.PPO_evasion_fuza.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza.BufferGRU import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time
from torch.cuda.amp import GradScaler, autocast

# --- 动作空间配置 (与原版相同) ---
CONTINUOUS_DIM = 4
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']
DISCRETE_DIMS = {
    'flare_trigger': 1,
    'salvo_size': 3,
    'intra_interval': 3,
    'num_groups': 3,
    'inter_interval': 3,
}
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())
TOTAL_ACTION_DIM_BUFFER = CONTINUOUS_DIM + len(DISCRETE_DIMS)
DISCRETE_ACTION_MAP = {
    # 'salvo_size': [2, 4, 6],
    # 'intra_interval': [0.02, 0.04, 0.1],
    # 'num_groups': [1, 2, 3],
    # 'inter_interval': [0.5, 1.0, 2.0]
    'salvo_size': [1, 2, 3],  # 修改为发射1、2、3枚
    'intra_interval': [0.05, 0.1, 0.15],
    'num_groups': [1, 2, 3],
    'inter_interval': [0.2, 0.5, 1.0]
}
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},
    'elevator': {'low': -1.0, 'high': 1.0},
    'aileron': {'low': -1.0, 'high': 1.0},
    'rudder': {'low': -1.0, 'high': 1.0},
}

# <<< GRU/RNN 修改 >>>: 新增 RNN 配置
# 这些参数最好也移到 Config.py 中
RNN_HIDDEN_SIZE = 256  # GRU 层的隐藏单元数量
SEQUENCE_LENGTH = 16  # 训练时使用的轨迹片段长度


# ==============================================================================
# Original MLP-based Actor and Critic (保留原始版本以供选择)
# ==============================================================================

class Actor(Module):
    # ... 原版 Actor 代码保持不变 ...
    """
    Actor 网络 (策略网络) - 已更新以支持复杂的五部分离散动作空间
    """

    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # <<< 更改 >>> 输出维度现在是 (连续*2) + 新的logits总数
        # 注意：这里不再需要一个总的 output_dim
        # self.output_dim = (CONTINUOUS_DIM * 2) + TOTAL_DISCRETE_LOGITS
        self.log_std_min = -20.0
        self.log_std_max = 2.0

        # 定义共享骨干网络
        # 负责从原始状态中提取高级特征
        shared_layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim
        self.shared_network = Sequential()
        for i in range(len(shared_layers_dims) - 1):
            # 添加线性层
            self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
            # --- 在此处添加 LayerNorm ---
            # LayerNorm 的输入维度是前一个线性层的输出维度
            self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
            # 添加激活函数
            self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # 骨干网络的输出维度，将作为各个头部的输入
        shared_output_dim = ACTOR_PARA.model_layer_dim[-1]
        # --- 独立头部网络 ---
        # 1. 连续动作头部：输出均值和标准差
        self.continuous_head = Linear(shared_output_dim, CONTINUOUS_DIM * 2)

        # 2. 离散动作头部：输出所有离散决策所需的 logits
        self.discrete_head = Linear(shared_output_dim, TOTAL_DISCRETE_LOGITS)

        # self.init_model()
        # --- 优化器和设备设置 ---
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def forward(self, obs):
        """
        前向传播方法，为每个动作维度创建并返回一个概率分布。
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 通过共享骨干网络提取通用特征
        shared_features = self.shared_network(obs_tensor)

        # 2. 将共享特征分别送入不同的头部网络
        cont_params = self.continuous_head(shared_features)
        all_disc_logits = self.discrete_head(shared_features)

        # ... 后续逻辑与原版完全相同 ...
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=1)
        trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
        has_flares_info = obs_tensor[:, 7]
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min
        mu, log_std = cont_params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        continuous_base_dist = Normal(mu, std)
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
        salvo_size_dist = Categorical(logits=salvo_size_logits)
        intra_interval_dist = Categorical(logits=intra_interval_logits)
        num_groups_dist = Categorical(logits=num_groups_logits)
        inter_interval_dist = Categorical(logits=inter_interval_logits)
        distributions = {
            'continuous': continuous_base_dist,
            'trigger': trigger_dist,
            'salvo_size': salvo_size_dist,
            'intra_interval': intra_interval_dist,
            'num_groups': num_groups_dist,
            'inter_interval': inter_interval_dist
        }
        return distributions


class Critic(Module):
    # ... 原版 Critic 代码保持不变 ...
    """
    Critic 网络 (价值网络)，评估状态的价值 V(s)。
    这个类的结构不受动作空间变化的影响。
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
        self.network = Sequential()
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            self.network.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

    def forward(self, obs):
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


# ==============================================================================
# <<< GRU/RNN 修改 >>>: 定义新的基于 GRU 的 Actor 和 Critic
# ==============================================================================

class Actor_GRU(Module):
    def __init__(self):
        super(Actor_GRU, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.rnn_hidden_size = RNN_HIDDEN_SIZE

        # 1. 共享的 MLP 骨干网络 (与原版相同)
        shared_layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim
        self.shared_network = Sequential()
        for i in range(len(shared_layers_dims) - 1):
            self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
            self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
            self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        shared_output_dim = ACTOR_PARA.model_layer_dim[-1]

        # 2. GRU 层
        # 它接收 MLP 提取的特征作为输入
        self.gru = GRU(shared_output_dim, self.rnn_hidden_size, batch_first=True)

        # 3. 独立头部网络 (与原版相同，但输入维度变为 rnn_hidden_size)
        self.continuous_head = Linear(self.rnn_hidden_size, CONTINUOUS_DIM * 2)
        self.discrete_head = Linear(self.rnn_hidden_size, TOTAL_DISCRETE_LOGITS)

        # 优化器等设置
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                     end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
                                                     total_iters=AGENTPARA.MAX_EXE_NUM)
        self.to(ACTOR_PARA.device)

    def forward(self, obs, hidden_state):
        """
        GRU Actor 的前向传播。
        Args:
            obs (Tensor): 观测值。形状可以是 (batch, features) 用于单步，或 (batch, seq_len, features) 用于序列。
            hidden_state (Tensor): GRU 的隐藏状态。形状是 (1, batch, rnn_hidden_size)。
        Returns:
            tuple: (包含所有动作分布的字典, 新的隐藏状态)
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3  # 检查输入是单个时间步还是序列

        # 统一处理输入形状
        if not is_sequence:
            # 如果是单步 (batch_size, features)，增加一个 seq_len=1 的维度
            obs_tensor = obs_tensor.unsqueeze(1)  # -> (batch_size, 1, features)

        # 1. 通过共享 MLP 提取特征
        # 注意：MLP 只能处理 (N, *, H_in) 形状，所以如果输入是序列，它会独立地处理每个时间步
        shared_features = self.shared_network(obs_tensor)

        # 2. 将特征序列和隐藏状态送入 GRU
        # gru_out 形状: (batch, seq_len, rnn_hidden_size)
        # new_hidden 形状: (1, batch, rnn_hidden_size)
        gru_out, new_hidden = self.gru(shared_features, hidden_state)

        # 如果原始输入是单步，我们也希望输出是单步的
        if not is_sequence:
            gru_out = gru_out.squeeze(1)  # -> (batch_size, rnn_hidden_size)

        # 3. 将 GRU 的输出送入各个头
        cont_params = self.continuous_head(gru_out)
        all_disc_logits = self.discrete_head(gru_out)

        # 后续的分布创建逻辑与原版 Actor 完全相同
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts

        # 动作掩码逻辑 (需要注意在序列情况下正确索引)
        # obs_tensor 此时可能是 (batch, seq_len, features)
        has_flares_info = obs_tensor[..., 7]  # 使用 ... 来处理单步和序列两种情况
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        mu, log_std = cont_params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        continuous_base_dist = Normal(mu, std)

        # Bernoulli 的 logits 需要移除最后一个维度
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
        salvo_size_dist = Categorical(logits=salvo_size_logits)
        intra_interval_dist = Categorical(logits=intra_interval_logits)
        num_groups_dist = Categorical(logits=num_groups_logits)
        inter_interval_dist = Categorical(logits=inter_interval_logits)

        distributions = {
            'continuous': continuous_base_dist,
            'trigger': trigger_dist,
            'salvo_size': salvo_size_dist,
            'intra_interval': intra_interval_dist,
            'num_groups': num_groups_dist,
            'inter_interval': inter_interval_dist
        }

        return distributions, new_hidden


class Critic_GRU(Module):
    def __init__(self):
        super(Critic_GRU, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.rnn_hidden_size = RNN_HIDDEN_SIZE

        # 1. MLP 骨干网络 (与原版 Critic 类似)
        self.network_base = Sequential()
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
        for i in range(len(layers_dims) - 1):
            self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())

        base_output_dim = CRITIC_PARA.model_layer_dim[-1]

        # 2. GRU 层
        self.gru = GRU(base_output_dim, self.rnn_hidden_size, batch_first=True)

        # 3. 输出头
        self.fc_out = Linear(self.rnn_hidden_size, self.output_dim)

        # 优化器等设置
        self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                      end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
                                                      total_iters=AGENTPARA.MAX_EXE_NUM)
        self.to(CRITIC_PARA.device)

    def forward(self, obs, hidden_state):
        """
        GRU Critic 的前向传播。
        """
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3

        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)

        base_features = self.network_base(obs_tensor)
        gru_out, new_hidden = self.gru(base_features, hidden_state)

        if not is_sequence:
            gru_out = gru_out.squeeze(1)

        value = self.fc_out(gru_out)
        return value, new_hidden


# ==============================================================================
# PPO Agent 主类
# ==============================================================================

class PPO_continuous(object):
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = False):
        super(PPO_continuous, self).__init__()

        self.use_rnn = use_rnn
        if self.use_rnn:
            print("--- 初始化 PPO Agent (使用 GRU 模型) ---")
            self.Actor = Actor_GRU()
            self.Critic = Critic_GRU()
        else:
            print("--- 初始化 PPO Agent (使用 MLP 模型) ---")
            self.Actor = Actor()
            self.Critic = Critic()

        self.buffer = Buffer(use_rnn=self.use_rnn)
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        self.training_start_time = time.strftime("PPOGRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../save/save_evade_fuza"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)

        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../../test/test_evade")

    def load_models_from_directory(self, directory_path: str):
        # This function is correct, no changes needed.
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

    def scale_action(self, action_cont_tanh):
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)
        return scaled_action

    def get_initial_hidden_states(self, batch_size=1):
        if not self.use_rnn:
            return None, None
        actor_hidden = torch.zeros((1, batch_size, self.Actor.rnn_hidden_size), device=ACTOR_PARA.device)
        critic_hidden = torch.zeros((1, batch_size, self.Critic.rnn_hidden_size), device=CRITIC_PARA.device)
        return actor_hidden, critic_hidden

    def store_experience(self, state, action, probs, value, reward, done, actor_hidden=None, critic_hidden=None):
        if not np.all(np.isfinite(probs)):
            raise ValueError("在 log_prob 中检测到 NaN/Inf！")
        if self.use_rnn and (actor_hidden is None or critic_hidden is None):
            raise ValueError("使用 RNN 模型时必须存储隐藏状态！")
        self.buffer.store_transition(state, value, action, probs, reward, done, actor_hidden, critic_hidden)

    def choose_action(self, state, actor_hidden, critic_hidden, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            if self.use_rnn:
                value, new_critic_hidden = self.Critic(state_tensor, critic_hidden)
                dists, new_actor_hidden = self.Actor(state_tensor, actor_hidden)
            else:
                value = self.Critic(state_tensor)
                dists = self.Actor(state_tensor)
                new_critic_hidden, new_actor_hidden = None, None

            continuous_base_dist = dists['continuous']
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            action_cont_tanh = torch.tanh(u)
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            sampled_actions_dict = {}
            for key, dist in dists.items():
                if key == 'continuous': continue
                if deterministic:
                    if isinstance(dist, Categorical):
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                    else:
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                else:
                    sampled_actions_dict[key] = dist.sample()

            log_prob_disc = sum(dists[key].log_prob(act) for key, act in sampled_actions_dict.items())
            total_log_prob = log_prob_cont + log_prob_disc

            action_disc_to_store = torch.stack(list(sampled_actions_dict.values()), dim=-1).float()
            action_to_store = torch.cat([u, action_disc_to_store], dim=-1)

            env_action_cont = self.scale_action(action_cont_tanh)
            final_env_action_tensor = torch.cat([env_action_cont, action_disc_to_store], dim=-1)

            value_np = value.cpu().numpy()
            action_to_store_np = action_to_store.cpu().numpy()
            log_prob_to_store_np = total_log_prob.cpu().numpy()
            final_env_action_np = final_env_action_tensor.cpu().numpy()

            if not is_batch:
                final_env_action_np = final_env_action_np[0]
                action_to_store_np = action_to_store_np[0]
                log_prob_to_store_np = log_prob_to_store_np[0]
                value_np = value_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np, value_np, new_actor_hidden, new_critic_hidden

    def cal_gae(self, states, values, actions, probs, rewards, dones):
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
        执行 PPO 的学习和更新步骤。已适配 RNN 模式并修正所有逻辑和维度错误。
        """
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            print(
                f"  [Info] Buffer size ({self.buffer.get_buffer_size()}) < batch size ({BUFFERPARA.BATCH_SIZE}). Skipping.")
            return None

        states, values, actions, old_probs, rewards, dones, _, __ = self.buffer.get_all_data()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        # ✅ 确保 values 是一维数组，与 advantages 对齐
        values = np.squeeze(values)  # 移除多余维度，比如 (N,1) → (N,)

        # ✅ 保证 returns 与 values 形状一致
        returns = advantages + values

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'adv_targ': [], 'ratio': []}

        for _ in range(self.ppo_epoch):
            if self.use_rnn:
                batch_generator = self.buffer.generate_sequence_batches(
                    SEQUENCE_LENGTH, BUFFERPARA.BATCH_SIZE, advantages, returns
                )
            else:
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                if self.use_rnn:
                    state, action_batch, old_prob, advantage, return_, initial_actor_h, initial_critic_h = batch_data
                    state = check(state).to(**ACTOR_PARA.tpdv)
                    action_batch = check(action_batch).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)
                    advantage = check(advantage).to(**ACTOR_PARA.tpdv)
                    return_ = check(return_).to(**CRITIC_PARA.tpdv)
                    initial_actor_h = check(initial_actor_h).to(**ACTOR_PARA.tpdv)
                    initial_critic_h = check(initial_critic_h).to(**CRITIC_PARA.tpdv)
                else:
                    batch_indices = batch_data
                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1)
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'intra_interval': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 3].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 4].long(),
                }

                # Actor Training
                if self.use_rnn:
                    new_dists, _ = self.Actor(state, initial_actor_h)
                else:
                    new_dists = self.Actor(state)

                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc

                total_entropy = (new_dists['continuous'].entropy().sum(dim=-1) + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()

                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                advantage_squeezed = advantage.squeeze(-1) if advantage.dim() > ratio.dim() else advantage
                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                # Critic Training
                if self.use_rnn:
                    new_value, _ = self.Critic(state, initial_critic_h)
                else:
                    new_value = self.Critic(state)

                # ####################################################################
                # # <<< FINAL, DEFINITIVE FIX FOR THE BROADCASTING WARNING >>>
                # ####################################################################
                # We ensure the target `return_` tensor has the same number of dimensions
                # as the network's output `new_value`.
                if new_value.dim() > return_.dim():
                    return_ = return_.unsqueeze(-1)
                # ####################################################################

                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

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

    def prep_training_rl(self):
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        self.Actor.eval()
        self.Critic.eval()

    def save(self, prefix=""):
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