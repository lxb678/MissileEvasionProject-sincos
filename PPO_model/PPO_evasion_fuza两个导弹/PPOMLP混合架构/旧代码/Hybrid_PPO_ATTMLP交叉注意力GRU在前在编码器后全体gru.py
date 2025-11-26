# 文件名: Hybrid_PPO_CrossAttentionMLP实体.py (优化版, 交叉注意力 + GRU)
# 描述: 此版本在交叉注意力架构的基础上增加了 GRU 模块。
#      位置：Attention Fusion (空间特征) -> GRU (时序特征) -> MLP (决策)。
#      这使得模型既能关注当前的威胁分布，也能记忆威胁的历史轨迹（如正在逼近还是远离）。

# 导入 PyTorch 核心库
import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F
# 导入概率分布工具
from torch.distributions import Bernoulli, Categorical, Normal
# 导入配置文件
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
# 导入 Buffer (保持不变)
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.BufferGRUAttn实体 import Buffer
# 导入学习率调度器
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time

# --- 动作空间配置 (保持不变) ---
CONTINUOUS_DIM = 4
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']
DISCRETE_DIMS = {
    'flare_trigger': 1,
    'salvo_size': 3,
    'num_groups': 3,
    'inter_interval': 3,
}
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())
TOTAL_ACTION_DIM_BUFFER = CONTINUOUS_DIM + len(DISCRETE_DIMS)
DISCRETE_ACTION_MAP = {
    'salvo_size': [2, 3, 4],
    'num_groups': [2, 3, 4],
    'inter_interval': [0.2, 0.4, 0.6]
}
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},
    'elevator': {'low': -1.0, 'high': 1.0},
    'aileron': {'low': -1.0, 'high': 1.0},
    'rudder': {'low': -1.0, 'high': 1.0},
}

# --- 实体注意力配置 (保持不变) ---
NUM_MISSILES = 2
MISSILE_FEAT_DIM = 4
AIRCRAFT_FEAT_DIM = 7
FULL_OBS_DIM = (NUM_MISSILES * MISSILE_FEAT_DIM) + AIRCRAFT_FEAT_DIM

ENTITY_EMBED_DIM = 64
ATTN_NUM_HEADS = 2

assert ENTITY_EMBED_DIM % ATTN_NUM_HEADS == 0, "ENTITY_EMBED_DIM must be divisible by ATTN_NUM_HEADS"


# ==============================================================================
#           <<< 核心修改 >>>: 升级架构为交叉注意力 + GRU
# ==============================================================================

class Actor_CrossAttentionMLP(Module):
    """
    Actor 网络 - [架构: 并行Encoder -> 共享GRU -> Attention -> MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=ENTITY_EMBED_DIM):
        super(Actor_CrossAttentionMLP, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.weight_decay = weight_decay
        self.rnn_hidden_dim = rnn_hidden_dim

        # ======================================================================
        # 1. 实体编码器 (MLP) - 位置最前
        # ======================================================================
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU()
        )
        self.aircraft_encoder = Sequential(
            Linear(FULL_OBS_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU()
        )

        # ======================================================================
        # 2. GRU 层 - 在 Encoder 之后
        # ======================================================================

        # 2.1 飞机专用的 GRU
        self.aircraft_gru = nn.GRU(
            input_size=ENTITY_EMBED_DIM,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # 2.2 💥[新增] 导弹专用的 GRU (参数共享给 M1 和 M2)
        # 这样导弹也能提取轨迹特征了
        self.missile_gru = nn.GRU(
            input_size=ENTITY_EMBED_DIM,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # ======================================================================
        # 3. 交叉注意力
        # ======================================================================
        self.attention = MultiheadAttention(
            embed_dim=self.rnn_hidden_dim,
            num_heads=ATTN_NUM_HEADS,
            dropout=0.0,
            batch_first=True
        )

        # ======================================================================
        # 4. MLP 决策层
        # ======================================================================
        mlp_input_dim = self.rnn_hidden_dim
        split_point = 2
        mlp_dims = ACTOR_PARA.model_layer_dim
        base_dims = mlp_dims[:split_point]
        tower_dims = mlp_dims[split_point:]

        self.shared_base_mlp = Sequential()
        base_input_dim = mlp_input_dim
        for i, dim in enumerate(base_dims):
            self.shared_base_mlp.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
            self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', LeakyReLU())
            base_input_dim = dim
        base_output_dim = base_dims[-1] if base_dims else mlp_input_dim

        self.continuous_tower = Sequential()
        tower_input_dim = base_output_dim
        for i, dim in enumerate(tower_dims):
            self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim
        continuous_tower_output_dim = tower_dims[-1] if tower_dims else base_output_dim

        self.discrete_tower = Sequential()
        tower_input_dim = base_output_dim
        for i, dim in enumerate(tower_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim
        discrete_tower_output_dim = tower_dims[-1] if tower_dims else base_output_dim

        self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))

        # --- 优化器设置 ---
        attention_params, gru_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            name_lower = name.lower()
            if any(key in name_lower for key in ['attention', 'attn', 'layer_norm']):
                attention_params.append(param)
            elif 'gru' in name_lower:
                gru_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {'params': attention_params, 'lr': ACTOR_PARA.attention_lr},
            {'params': gru_params, 'lr': ACTOR_PARA.gru_lr},
            {'params': other_params, 'lr': ACTOR_PARA.lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim, start_factor=1.0, end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        # --- 1. 展平并编码 (Encoder-first) ---
        obs_flat = obs_tensor.view(-1, FULL_OBS_DIM)
        missile1_obs_flat = obs_flat[..., 0:MISSILE_FEAT_DIM]
        missile2_obs_flat = obs_flat[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]

        ac_embed_flat = self.aircraft_encoder(obs_flat)
        m1_embed_flat = self.missile_encoder(missile1_obs_flat)
        m2_embed_flat = self.missile_encoder(missile2_obs_flat)

        # --- 2. 还原为序列形状，准备送入 GRU ---
        ac_embed_seq = ac_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)
        m1_embed_seq = m1_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)
        m2_embed_seq = m2_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)

        # --- 3. 隐藏状态管理 ---
        # 注意：现在 rnn_state 包含 3 个部分：Air, M1, M2
        if rnn_state is None:
            h_air, h_m1, h_m2 = None, None, None
        else:
            # 按照 rnn_hidden_dim 进行拆分 (假设三个实体的 hidden_dim 相同)
            h_air, h_m1, h_m2 = torch.split(rnn_state, self.rnn_hidden_dim, dim=-1)
            h_air, h_m1, h_m2 = h_air.contiguous(), h_m1.contiguous(), h_m2.contiguous()

        # --- 4. 通过 GRU 进行时序建模 ---
        # 飞机过飞机GRU
        air_out, next_h_air = self.aircraft_gru(ac_embed_seq, h_air)
        # 导弹过导弹GRU (参数共享)
        m1_out, next_h_m1 = self.missile_gru(m1_embed_seq, h_m1)
        m2_out, next_h_m2 = self.missile_gru(m2_embed_seq, h_m2)

        # 拼接新的 hidden state 返回 (3个部分拼接在一起)
        next_rnn_state = torch.cat([next_h_air, next_h_m1, next_h_m2], dim=-1)

        # --- 5. GRU后的残差连接 ---
        air_out = air_out + ac_embed_seq
        m1_out = m1_out + m1_embed_seq
        m2_out = m2_out + m2_embed_seq

        # --- 6. 准备 Attention 输入 ---
        air_feat_flat = air_out.reshape(-1, self.rnn_hidden_dim)
        m1_feat_flat = m1_out.reshape(-1, self.rnn_hidden_dim)
        m2_feat_flat = m2_out.reshape(-1, self.rnn_hidden_dim)

        # Mask
        inactive_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        is_m1_inactive = torch.all(torch.isclose(missile1_obs_flat, inactive_fingerprint), dim=-1)
        is_m2_inactive = torch.all(torch.isclose(missile2_obs_flat, inactive_fingerprint), dim=-1)
        attention_mask = torch.stack([is_m1_inactive, is_m2_inactive], dim=1)

        query = air_feat_flat.unsqueeze(1)
        keys = torch.stack([m1_feat_flat, m2_feat_flat], dim=1)

        # --- 7. 注意力融合 ---
        attn_output, attn_weights = self.attention(query, keys, keys, key_padding_mask=attention_mask)
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0)

        # Attention后的残差连接
        combined_features = air_feat_flat + attn_output.squeeze(1)

        # --- 8. MLP 决策 ---
        base_features = self.shared_base_mlp(combined_features)
        continuous_features = self.continuous_tower(base_features)
        discrete_features = self.discrete_tower(base_features)

        mu = self.mu_head(continuous_features)
        all_disc_logits = self.discrete_head(discrete_features)

        # 5. 动作掩码处理 (Masking)
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        # --- 修复点：确保 mask 也是展平的 ---
        flare_info_index = 2 * MISSILE_FEAT_DIM + 5
        # 这里 obs_flat 已经是 (B*S, Dim)，所以 has_flares_info 是 (B*S,)
        has_flares_info = obs_flat[..., flare_info_index]

        # <<< 强制 reshape 以防万一 >>>
        # 确保 mask 是 (B*S,) 而不是 (B, S)
        mask = (has_flares_info == 0).view(-1)

        trigger_logits_masked = trigger_logits.clone()

        # 只有当 mask 和 logits 维度匹配时才能赋值
        if torch.any(mask):
            # trigger_logits_masked 是 (B*S, 1)，mask 是 (B*S,)
            # 我们需要把 mask 扩展为 (B*S, 1) 或者直接用 boolean indexing

            # 写法 1: 扩展维度
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)

                # 此时 mask: (640, 1), trigger: (640, 1) -> 这样可以正确索引
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        trigger_probs = torch.sigmoid(trigger_logits_masked)
        no_trigger_mask = (trigger_probs < 0.5)

        # 其他离散动作的 Masking
        NEG_INF = -1e8
        salvo_size_logits_masked = salvo_size_logits.clone()
        forced_salvo = torch.full_like(salvo_size_logits_masked, NEG_INF)
        forced_salvo[..., 0] = 1.0
        salvo_size_logits_masked = torch.where(no_trigger_mask, forced_salvo, salvo_size_logits_masked)

        num_groups_logits_masked = num_groups_logits.clone()
        forced_groups = torch.full_like(num_groups_logits_masked, NEG_INF)
        forced_groups[..., 0] = 1.0
        num_groups_logits_masked = torch.where(no_trigger_mask, forced_groups, num_groups_logits_masked)

        inter_interval_logits_masked = inter_interval_logits.clone()
        forced_interval = torch.full_like(inter_interval_logits_masked, NEG_INF)
        forced_interval[..., 0] = 1.0
        inter_interval_logits_masked = torch.where(no_trigger_mask, forced_interval, inter_interval_logits_masked)

        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        continuous_base_dist = Normal(mu, std)

        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
        salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
        num_groups_dist = Categorical(logits=num_groups_logits_masked)
        inter_interval_dist = Categorical(logits=inter_interval_logits_masked)

        distributions = {
            'continuous': continuous_base_dist,
            'trigger': trigger_dist,
            'salvo_size': salvo_size_dist,
            'num_groups': num_groups_dist,
            'inter_interval': inter_interval_dist
        }

        attention_to_missiles = attn_weights.squeeze(1).view(batch_size, seq_len, 2)
        if seq_len == 1:
            attention_to_missiles = attention_to_missiles.squeeze(1)

        return distributions, attention_to_missiles, next_rnn_state


class Critic_CrossAttentionMLP(Module):
    """
    Critic 网络 - [架构: 并行Encoder -> 共享GRU -> Attention -> MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=ENTITY_EMBED_DIM):
        super(Critic_CrossAttentionMLP, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.weight_decay = weight_decay
        self.rnn_hidden_dim = rnn_hidden_dim

        # ======================================================================
        # 1. 实体编码器 (MLP) - 位置最前
        # ======================================================================
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU()
        )
        self.aircraft_encoder = Sequential(
            Linear(FULL_OBS_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU()
        )

        # ======================================================================
        # 2. GRU 层 - 在 Encoder 之后
        # ======================================================================
        self.aircraft_gru = nn.GRU(
            input_size=ENTITY_EMBED_DIM,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )
        self.missile_gru = nn.GRU(
            input_size=ENTITY_EMBED_DIM,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # ======================================================================
        # 3. 交叉注意力
        # ======================================================================
        self.attention = MultiheadAttention(
            embed_dim=self.rnn_hidden_dim,
            num_heads=ATTN_NUM_HEADS,
            batch_first=True
        )

        # ======================================================================
        # 4. MLP 估值层
        # ======================================================================
        mlp_dims = CRITIC_PARA.model_layer_dim
        self.mlp = Sequential()
        input_dim = self.rnn_hidden_dim
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(f'fc_{i}', Linear(input_dim, dim))
            self.mlp.add_module(f'act_{i}', LeakyReLU())
            input_dim = dim
        self.fc_out = Linear(input_dim, self.output_dim)

        # ======================================================================
        # Optimizer
        # ======================================================================
        attn_params, gru_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'attention' in name.lower():
                attn_params.append(param)
            elif 'gru' in name.lower():
                gru_params.append(param)
            else:
                other_params.append(param)
        self.optim = torch.optim.Adam([
            {'params': attn_params, 'lr': CRITIC_PARA.attention_lr},
            {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
            {'params': other_params, 'lr': CRITIC_PARA.lr}
        ])
        self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                      end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
                                                      total_iters=AGENTPARA.MAX_EXE_NUM)
        self.to(CRITIC_PARA.device)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        # --- 1. 展平并编码 (Encoder-first) ---
        obs_flat = obs_tensor.view(-1, FULL_OBS_DIM)
        missile1_obs_flat = obs_flat[..., 0:MISSILE_FEAT_DIM]
        missile2_obs_flat = obs_flat[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]

        ac_embed_flat = self.aircraft_encoder(obs_flat)
        m1_embed_flat = self.missile_encoder(missile1_obs_flat)
        m2_embed_flat = self.missile_encoder(missile2_obs_flat)

        # --- 2. 还原为序列形状 ---
        ac_embed_seq = ac_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)
        m1_embed_seq = m1_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)
        m2_embed_seq = m2_embed_flat.view(batch_size, seq_len, ENTITY_EMBED_DIM)

        # --- 3. 隐藏状态管理 ---
        if rnn_state is None:
            h_air, h_m1, h_m2 = None, None, None
        else:
            h_air, h_m1, h_m2 = torch.split(rnn_state, self.rnn_hidden_dim, dim=-1)
            h_air, h_m1, h_m2 = h_air.contiguous(), h_m1.contiguous(), h_m2.contiguous()

        # --- 4. 通过 GRU (分别处理) ---
        air_out, next_h_air = self.aircraft_gru(ac_embed_seq, h_air)
        m1_out, next_h_m1 = self.missile_gru(m1_embed_seq, h_m1)
        m2_out, next_h_m2 = self.missile_gru(m2_embed_seq, h_m2)

        next_rnn_state = torch.cat([next_h_air, next_h_m1, next_h_m2], dim=-1)

        # --- 5. GRU后的残差连接 ---
        air_out = air_out + ac_embed_seq
        m1_out = m1_out + m1_embed_seq
        m2_out = m2_out + m2_embed_seq

        # --- 6. 准备 Attention 输入 ---
        air_feat_flat = air_out.reshape(-1, self.rnn_hidden_dim)
        m1_feat_flat = m1_out.reshape(-1, self.rnn_hidden_dim)
        m2_feat_flat = m2_out.reshape(-1, self.rnn_hidden_dim)

        # Mask
        inactive_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        is_m1_inactive = torch.all(torch.isclose(missile1_obs_flat, inactive_fingerprint), dim=-1)
        is_m2_inactive = torch.all(torch.isclose(missile2_obs_flat, inactive_fingerprint), dim=-1)
        attention_mask = torch.stack([is_m1_inactive, is_m2_inactive], dim=1)

        query = air_feat_flat.unsqueeze(1)
        keys = torch.stack([m1_feat_flat, m2_feat_flat], dim=1)

        # --- 7. 注意力融合 ---
        attn_out, _ = self.attention(query, keys, keys, key_padding_mask=attention_mask)
        if torch.isnan(attn_out).any():
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        combined = air_feat_flat + attn_out.squeeze(1)

        # --- 8. MLP 估值 ---
        val = self.fc_out(self.mlp(combined))
        return val, next_rnn_state


class PPO_continuous(object):
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = True):
        super(PPO_continuous, self).__init__()

        # <<< 修改: 启用 RNN 标识 >>>
        self.use_rnn = use_rnn  # True
        print(f"--- 初始化 PPO Agent (Cross-Attention + GRU + MLP) use_rnn={self.use_rnn} ---")

        # <<< 新增配置 >>>
        self.rnn_seq_len = 10  # 序列长度，可调 (如 8, 16, 32)
        self.rnn_batch_size = BUFFERPARA.BATCH_SIZE #64  # 批次大小

        # ... (其他初始化代码保持不变) ...

        # 初始化模型
        self.Actor = Actor_CrossAttentionMLP()
        self.Critic = Critic_CrossAttentionMLP()

        # <<< 新增: 初始化推理用的 Hidden State >>>
        self.actor_rnn_state = None
        self.critic_rnn_state = None

        # Buffer 初始化 (use_attn=True 保持不变，Buffer 内部逻辑暂不改动)
        self.buffer = Buffer(use_rnn=self.use_rnn, use_attn=True)

        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.training_start_time = time.strftime("PPO_EntityCrossATT_GRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../../../save/save_evade_fuza两个导弹"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)
        if load_able:
            if model_dir_path:
                self.load_models_from_directory(model_dir_path)
            else:
                self.load_models_from_directory("../../../../test/test_evade")

    # <<< 新增: 重置 RNN 状态的方法 >>>
    def reset_rnn_state(self):
        self.actor_rnn_state = None
        self.critic_rnn_state = None

    def load_models_from_directory(self, directory_path: str):
        """从指定目录加载 Actor 和 Critic 模型的权重。"""
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
        return lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)

    def store_experience(self, state, action, probs, value, reward, done, attn_weights=None):
        # 使用 choose_action 中保存的 temp_hidden
        self.buffer.store_transition(state, value, action, probs, reward, done,
                                     actor_hidden=self.temp_actor_h,
                                     critic_hidden=self.temp_critic_h,
                                     attn_weights=attn_weights)
        # 如果回合结束，重置 RNN 状态
        if done:
            self.reset_rnn_state()

    def choose_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        # <<< 1. 临时保存更新前的 Hidden State (用于存储) >>>
        current_actor_h = self.actor_rnn_state
        current_critic_h = self.critic_rnn_state

        with torch.no_grad():
            # <<< 核心修改: 传入并更新 RNN 状态 >>>
            value, self.critic_rnn_state = self.Critic(state_tensor, self.critic_rnn_state)

            # Actor 返回: 分布, 注意力权重, 新的 RNN 状态
            dists, attention_weights, self.actor_rnn_state = self.Actor(state_tensor, self.actor_rnn_state)

            attention_weights_for_reward = attention_weights

            # --- 后续动作采样逻辑保持不变 ---
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

            attention_weights_np = attention_weights_for_reward.cpu().numpy() if attention_weights_for_reward is not None else None

            if not is_batch:
                final_env_action_np = final_env_action_np[0]
                action_to_store_np = action_to_store_np[0]
                log_prob_to_store_np = log_prob_to_store_np[0]
                value_np = value_np[0]
                if attention_weights_np is not None:
                    attention_weights_np = attention_weights_np[0]
            # --- 关键修正：移出 if not is_batch 块 ---
            # 这里无论是否是 batch，都需要处理 hidden state 用于存储

            # <<< 2. 保存旧状态到 self，供 store_experience 调用 >>>
            # 如果是 None (第一步)，创建一个全0的 hidden state
            if current_actor_h is None:
                # 注意：这里需要根据实际的 batch_size 创建零状态
                batch_size = state_tensor.shape[0]
                # 计算新的总隐藏维度 (rnn_hidden_dim * 3) -> Air, M1, M2
                total_hidden_dim = self.Actor.rnn_hidden_dim * 3

                # 形状: (Num_Layers=1, Batch, Hidden)
                self.temp_actor_h = torch.zeros(1, batch_size, total_hidden_dim).to(ACTOR_PARA.device)
                self.temp_critic_h = torch.zeros(1, batch_size, total_hidden_dim).to(CRITIC_PARA.device)
            else:
                self.temp_actor_h = current_actor_h
                self.temp_critic_h = current_critic_h

        return final_env_action_np, action_to_store_np, log_prob_to_store_np, value_np, attention_weights_np

    def cal_gae(self, states, values, actions, probs, rewards, dones, next_value=0.0):
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                value_next_step = next_value
            else:
                value_next_step = values[t + 1]

            done_mask = 1.0 - int(dones[t])
            delta = rewards[t] + self.gamma * value_next_step * done_mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantage[t] = gae
        return advantage

    def learn(self, next_visual_value=0.0):
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            return None

        # 1. 获取所有数据
        states, values, actions, old_probs, rewards, dones, _, _, attn_weights = self.buffer.get_all_data()

        # 2. 计算 GAE
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones, next_value=next_visual_value)
        values = np.squeeze(values)
        returns = advantages + values

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

        for _ in range(self.ppo_epoch):
            # -------------------------------------------------------------
            # <<< 核心修改 >>> 根据模式选择生成器
            # -------------------------------------------------------------
            if self.use_rnn:
                batch_generator = self.buffer.generate_sequence_batches(
                    self.rnn_seq_len, self.rnn_batch_size, advantages, returns
                )
            else:
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                # -------------------------------------------------------------
                # 分支 A: RNN 模式 (序列处理)
                # -------------------------------------------------------------
                if self.use_rnn:
                    (b_s, b_a, b_p, b_adv, b_ret, b_h_a, b_h_c, _) = batch_data

                    # 1. 转 Tensor
                    state = torch.FloatTensor(b_s).to(**ACTOR_PARA.tpdv)  # (Batch, Seq, Dim)
                    action_batch = torch.FloatTensor(b_a).to(**ACTOR_PARA.tpdv)  # (Batch, Seq, Act)

                    # 2. 展平 (Flatten) 标签数据，以便后续计算 Loss
                    # 将 (Batch, Seq) 维度合并为 (Batch*Seq)
                    old_prob = torch.FloatTensor(b_p).to(**ACTOR_PARA.tpdv).view(-1)
                    advantage = torch.FloatTensor(b_adv).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = torch.FloatTensor(b_ret).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    # # 3. 处理 Hidden State
                    # # Buffer 存的是 (Batch, Hidden)，PyTorch GRU 需要 (1, Batch, Hidden)
                    # rnn_h_a = torch.FloatTensor(b_h_a).transpose(0, 1).contiguous().to(**ACTOR_PARA.tpdv)
                    # rnn_h_c = torch.FloatTensor(b_h_c).transpose(0, 1).contiguous().to(**CRITIC_PARA.tpdv)

                    # 现在 buffer 吐出来的已经是 (1, Batch, Hidden)，直接转 Tensor 即可
                    rnn_h_a = torch.FloatTensor(b_h_a).to(**ACTOR_PARA.tpdv)
                    rnn_h_c = torch.FloatTensor(b_h_c).to(**CRITIC_PARA.tpdv)
                    # ------------------------------------------------------

                    # 4. 前向传播 (传入序列 + 历史隐藏状态)
                    # 注意：new_dists 里的 tensor 已经是 (Batch*Seq, ...) 形状了
                    new_dists, _, _ = self.Actor(state, rnn_h_a)
                    new_value, _ = self.Critic(state, rnn_h_c)

                    # 5. 关键步骤：展平 Action Batch 以匹配 new_dists
                    # (Batch, Seq, Act_Dim) -> (Batch*Seq, Act_Dim)
                    action_batch = action_batch.view(-1, action_batch.shape[-1])

                # -------------------------------------------------------------
                # 分支 B: MLP 模式 (普通处理)
                # -------------------------------------------------------------
                else:
                    batch_indices = batch_data
                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv)
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    # 前向传播 (无状态)
                    new_dists, _, _ = self.Actor(state)
                    new_value, _ = self.Critic(state)

                # -------------------------------------------------------------
                # 公共 Loss 计算 (删除了原代码中错误的重新 forward 部分)
                # -------------------------------------------------------------
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # 计算概率
                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc

                # 计算熵
                entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                total_entropy = (entropy_cont + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()

                # 计算 Ratio 和 Loss
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 更新 Actor
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=0.5)
                self.Actor.optim.step()

                # 更新 Critic
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)
                self.Critic.optim.step()

                # 记录日志
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['entropy_cont'].append(entropy_cont.mean().item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())

        if not train_info['critic_loss']:
            print("  [Warning] No batches were generated for training.")
            self.buffer.clear_memory()
            return None

        self.buffer.clear_memory()
        for key in train_info:
            train_info[key] = np.mean(train_info[key])

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
        except Exception as e:
            print(f"创建存档目录失败: {e}")

        target_dir = self.win_rate_dir if prefix else self.run_save_dir

        for net_name in ['Actor', 'Critic']:
            try:
                net_model = getattr(self, net_name)
                filename = f"{prefix}_{net_name}.pkl" if prefix else f"{net_name}.pkl"
                full_path = os.path.join(target_dir, filename)
                torch.save(net_model.state_dict(), full_path)
                print(f"  - {filename} 保存成功于 {target_dir}。")
            except Exception as e:
                print(f"  - 保存模型 {net_name} 时发生错误: {e}")