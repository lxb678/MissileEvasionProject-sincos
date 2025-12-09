# 文件名: Hybrid_PPO_PostAttentionGRU.py
# 描述:
#      架构修改为: [Missile/Air Encoder] -> [Cross Attention] -> [Feature Fusion] -> [Global GRU] -> [MLP]
#      GRU 现在位于注意力层之后，用于处理融合了威胁信息的全局上下文序列。

import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
# 导入配置文件
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.BufferGRUAttn实体 import Buffer
from torch.optim import lr_scheduler
import numpy as np
import os
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

ENTITY_EMBED_DIM = 32 #64
ATTN_NUM_HEADS = 2 #4

assert ENTITY_EMBED_DIM % ATTN_NUM_HEADS == 0, "ENTITY_EMBED_DIM must be divisible by ATTN_NUM_HEADS"


# ==============================================================================
#           <<< 核心修改 >>>: Post-Attention GRU 架构 (Actor)
# ==============================================================================

class Actor_PreAttentionGRU(Module):
    """
    架构: [Encoder] -> [GRU] -> [Residual: Concat(Enc, GRU)] -> [Attention] -> [Fusion] -> [MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=32):
        super(Actor_PreAttentionGRU, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim

        # --- 动作方差参数 ---
        self.target_std_min = 0.10
        self.target_std_max = 0.80
        self.target_init_std = 0.75
        self.log_std_min = np.log(self.target_std_min)
        self.log_std_max = np.log(self.target_std_max)
        self.weight_decay = weight_decay

        # --- 维度配置 ---
        # 1. 编码器输出维度 = 32
        self.encoder_hidden_dim = 32
        # 2. GRU 隐藏层维度 = 32
        self.rnn_hidden_dim = 32
        # 3. 残差拼接后的维度 = 32 + 32 = 64
        self.residual_feat_dim = self.rnn_hidden_dim #self.encoder_hidden_dim + self.rnn_hidden_dim

        # --- 网络层定义 ---
        # 1. Encoders
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, self.encoder_hidden_dim),
            # LeakyReLU()
        )
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, self.encoder_hidden_dim),
            # LeakyReLU()
        )

        # 2. GRUs (Pre-Attention)
        self.aircraft_gru = nn.GRU(
            input_size=self.encoder_hidden_dim,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )
        self.missile_gru = nn.GRU(
            input_size=self.encoder_hidden_dim,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # 3. Attention
        # 输入维度必须是 残差拼接后的维度 (64)
        self.attention = MultiheadAttention(
            embed_dim=self.residual_feat_dim,
            num_heads=ATTN_NUM_HEADS,
            dropout=0.0,
            batch_first=True
        )

        # 4. 特征融合
        # 融合了 (Air_Res_Feat) + (Attn_Out) = 64 + 64 = 128
        fusion_total_dim = self.residual_feat_dim * 2
        # self.layer_norm = nn.LayerNorm(fusion_total_dim)  # <--- 强烈建议启用

        # 5. MLP
        split_point = 2
        mlp_dims = ACTOR_PARA.model_layer_dim
        base_dims = mlp_dims[:split_point]
        tower_dims = mlp_dims[split_point:]

        self.shared_base_mlp = Sequential()
        base_input_dim = fusion_total_dim
        for i, dim in enumerate(base_dims):
            self.shared_base_mlp.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
            self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', LeakyReLU())
            base_input_dim = dim
        base_output_dim = base_dims[-1] if base_dims else fusion_total_dim

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

        init_log_std = np.log(self.target_init_std)
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), init_log_std))

        # 优化器
        self._init_optimizer()
        self.to(ACTOR_PARA.device)
        self._init_weights()

    def _init_optimizer(self):
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)
        nn.init.orthogonal_(self.discrete_head.weight, gain=0.01)
        nn.init.constant_(self.discrete_head.bias, 0)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 2: obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        # --- 1. 拆解 Hidden State (维度=32) ---
        if rnn_state is None:
            h_air, h_missiles = None, None
        else:
            # [修正] 加上 .contiguous() 以防万一，或者 GRU 内部可能会警告
            h_air = rnn_state[..., :self.rnn_hidden_dim].contiguous()

            h_miss_raw = rnn_state[..., self.rnn_hidden_dim:]

            # [修正] 报错行：原先是 .view(...)，现在改为 .reshape(...)
            # view() 只能处理内存连续的 Tensor，切片后的 Tensor 往往不连续
            h_missiles = h_miss_raw.reshape(1, batch_size * 2, self.rnn_hidden_dim)

        obs_flat_raw = obs_tensor.view(-1, FULL_OBS_DIM)
        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # --- 2. 飞机特征处理 (Residual: Encoder + GRU) ---
        air_embed_seq = self.aircraft_encoder(aircraft_obs)  # [B, Seq, 32]
        air_gru_out, next_h_air = self.aircraft_gru(air_embed_seq, h_air)  # [B, Seq, 32]

        # [核心] 拼接输入与输出 -> [B, Seq, 64]
        # air_feat_seq = torch.cat([air_embed_seq, air_gru_out], dim=-1)
        air_feat_seq = air_gru_out #+ air_embed_seq

        # --- 3. 导弹特征处理 (Residual: Encoder + GRU) ---
        missiles_raw = torch.cat([missile1_obs, missile2_obs], dim=0)
        missiles_embed_seq = self.missile_encoder(missiles_raw)  # [B*2, Seq, 32]
        missiles_gru_out, next_h_missiles = self.missile_gru(missiles_embed_seq, h_missiles)  # [B*2, Seq, 32]

        # [核心] 拼接 -> [B*2, Seq, 64]
        # missiles_feat_seq = torch.cat([missiles_embed_seq, missiles_gru_out], dim=-1)
        missiles_feat_seq = missiles_gru_out
        m1_feat_seq, m2_feat_seq = torch.split(missiles_feat_seq, batch_size, dim=0)

        # 展平准备 Attention
        air_feat_flat = air_feat_seq.reshape(-1, self.residual_feat_dim)  # [B*S, 64]
        m1_feat_flat = m1_feat_seq.reshape(-1, self.residual_feat_dim)
        m2_feat_flat = m2_feat_seq.reshape(-1, self.residual_feat_dim)

        # --- 4. Attention (输入维度64) ---
        m1_raw = obs_flat_raw[..., 0:4]
        m2_raw = obs_flat_raw[..., 4:8]
        inactive = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        mask = torch.stack([
            torch.all(torch.isclose(m1_raw, inactive), dim=-1),
            torch.all(torch.isclose(m2_raw, inactive), dim=-1)
        ], dim=1)

        query = air_feat_flat.unsqueeze(1)  # [B*S, 1, 64]
        keys = torch.stack([m1_feat_flat, m2_feat_flat], dim=1)  # [B*S, 2, 64]

        attn_out, attn_weights = self.attention(query, keys, keys, key_padding_mask=mask)
        if torch.isnan(attn_out).any(): attn_out = torch.nan_to_num(attn_out, nan=0.0)

        # --- 5. Fusion & MLP ---
        # [B*S, 64] + [B*S, 64] = [B*S, 128]
        fusion_features = torch.cat([air_feat_flat, attn_out.squeeze(1)], dim=-1)
        # fusion_features = self.layer_norm(fusion_features)

        base = self.shared_base_mlp(fusion_features)
        continuous_features = self.continuous_tower(base)
        discrete_features = self.discrete_tower(base)

        mu = self.mu_head(continuous_features)
        # 强行限制均值
        mu = torch.clamp(mu, -3.0, 3.0)

        all_disc_logits = self.discrete_head(discrete_features)

        # Masking 处理 (保持不变)
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        flare_info_index = 2 * MISSILE_FEAT_DIM + 5
        has_flares_info = obs_flat_raw[..., flare_info_index]
        mask = (has_flares_info == 0).view(-1)

        NEG_INF = -1e8
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            mask_expanded = mask.unsqueeze(-1) if mask.dim() < trigger_logits_masked.dim() else mask
            trigger_logits_masked = torch.where(mask_expanded, torch.full_like(trigger_logits, NEG_INF), trigger_logits)

        trigger_probs = torch.sigmoid(trigger_logits_masked)
        no_trigger_mask = (trigger_probs < 0.5)

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

        # 限制标准差应用
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        continuous_base_dist = Normal(mu, std)

        distributions = {
            'continuous': continuous_base_dist,
            'trigger': Bernoulli(logits=trigger_logits_masked.squeeze(-1)),
            'salvo_size': Categorical(logits=salvo_size_logits_masked),
            'num_groups': Categorical(logits=num_groups_logits_masked),
            'inter_interval': Categorical(logits=inter_interval_logits_masked)
        }

        # --- 6. Hidden State 重组 ---
        next_h_miss_reshaped = next_h_missiles.view(1, batch_size, -1)
        next_h_combined = torch.cat([next_h_air, next_h_miss_reshaped], dim=-1)

        # 可视化 Attention
        attn_vis = attn_weights.squeeze(1).view(batch_size, seq_len, 2)
        if seq_len == 1: attn_vis = attn_vis.squeeze(1)

        return distributions, attn_vis, next_h_combined


# ==============================================================================
#           <<< 核心修改 >>>: Post-Attention GRU 架构 (Critic)
# ==============================================================================

class Critic_PreAttentionGRU(Module):
    """
    Critic 网络 - [架构: Encoders -> GRU -> Residual -> Attention -> Fusion -> MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=32):
        super(Critic_PreAttentionGRU, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.weight_decay = weight_decay

        self.encoder_hidden_dim = 32
        self.rnn_hidden_dim = 32
        self.residual_feat_dim = 32 #+ 32  # 64

        # 1. Encoders
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, self.encoder_hidden_dim),
            # LeakyReLU()
        )
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, self.encoder_hidden_dim),
            # LeakyReLU()
        )

        # 2. GRUs (Pre-Attention)
        self.aircraft_gru = nn.GRU(
            input_size=self.encoder_hidden_dim,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )
        self.missile_gru = nn.GRU(
            input_size=self.encoder_hidden_dim,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # 3. Attention (输入维度 64)
        self.attention = MultiheadAttention(
            embed_dim=self.residual_feat_dim,
            num_heads=ATTN_NUM_HEADS,
            batch_first=True
        )

        # 4. Norm (64 + 64 = 128)
        fusion_total_dim = self.residual_feat_dim * 2
        # self.layer_norm = nn.LayerNorm(fusion_total_dim)

        # 5. MLP
        mlp_dims = CRITIC_PARA.model_layer_dim
        self.mlp = Sequential()
        input_dim = fusion_total_dim
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(f'fc_{i}', Linear(input_dim, dim))
            self.mlp.add_module(f'act_{i}', LeakyReLU())
            input_dim = dim
        self.fc_out = Linear(input_dim, self.output_dim)

        # Init & Optimizer
        self._init_optimizer()
        self.to(CRITIC_PARA.device)
        self._init_weights()

    def _init_optimizer(self):
        attention_params, gru_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            name_lower = name.lower()
            if 'attention' in name_lower:
                attention_params.append(param)
            elif 'gru' in name_lower:
                gru_params.append(param)
            else:
                other_params.append(param)
        self.optim = torch.optim.Adam([
            {'params': attention_params, 'lr': CRITIC_PARA.attention_lr},
            {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
            {'params': other_params, 'lr': CRITIC_PARA.lr}
        ])
        self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                      end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
                                                      total_iters=AGENTPARA.MAX_EXE_NUM)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                    elif 'bias' in name: param.data.fill_(0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.fc_out.weight, gain=1.0)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 2: obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        # 1. Hidden Split
        if rnn_state is None:
            h_air, h_missiles = None, None
        else:
            # [修正] 加上 .contiguous()
            h_air = rnn_state[..., :self.rnn_hidden_dim].contiguous()

            # [修正] 报错行：将 .view(...) 改为 .reshape(...)
            # 原代码: h_missiles = rnn_state[..., self.rnn_hidden_dim:].view(1, batch_size * 2, self.rnn_hidden_dim).contiguous()
            h_missiles = rnn_state[..., self.rnn_hidden_dim:].reshape(1, batch_size * 2, self.rnn_hidden_dim)

        # 2. Air Path (Residual)
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]
        air_emb = self.aircraft_encoder(aircraft_obs)
        air_gru, next_h_air = self.aircraft_gru(air_emb, h_air)
        # air_feat = torch.cat([air_emb, air_gru], dim=-1) # 64
        air_feat = air_gru

        # 3. Missile Path (Residual)
        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        m_raw = torch.cat([missile1_obs, missile2_obs], dim=0)
        m_emb = self.missile_encoder(m_raw)
        m_gru, next_h_missiles = self.missile_gru(m_emb, h_missiles)
        # m_feat = torch.cat([m_emb, m_gru], dim=-1) # 64
        m_feat = m_gru
        m1_feat, m2_feat = torch.split(m_feat, batch_size, dim=0)

        # 4. Attention
        obs_flat = obs_tensor.view(-1, FULL_OBS_DIM)
        m1_raw, m2_raw = obs_flat[..., 0:4], obs_flat[..., 4:8]
        inact = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        mask = torch.stack([
            torch.all(torch.isclose(m1_raw, inact), dim=-1),
            torch.all(torch.isclose(m2_raw, inact), dim=-1)
        ], dim=1)

        air_flat = air_feat.reshape(-1, self.residual_feat_dim)
        m1_flat = m1_feat.reshape(-1, self.residual_feat_dim)
        m2_flat = m2_feat.reshape(-1, self.residual_feat_dim)

        q = air_flat.unsqueeze(1)
        k = torch.stack([m1_flat, m2_flat], dim=1)
        attn, _ = self.attention(q, k, k, key_padding_mask=mask)
        if torch.isnan(attn).any(): attn = torch.nan_to_num(attn, nan=0.0)

        # 5. Fusion
        fusion = torch.cat([air_flat, attn.squeeze(1)], dim=-1) # 128
        # fusion = self.layer_norm(fusion)
        val = self.fc_out(self.mlp(fusion))

        # 6. Reassemble
        next_h_comb = torch.cat([next_h_air, next_h_missiles.view(1, batch_size, -1)], dim=-1)
        return val, next_h_comb


class PPO_continuous(object):
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = True):
        super(PPO_continuous, self).__init__()

        self.use_rnn = use_rnn  # True
        print(f"--- 初始化 PPO Agent (Post-Attention GRU) use_rnn={self.use_rnn} ---")

        self.rnn_seq_len = 15 #12 #15 #12 #20 #15 #10 #5 #15 #10 #15 #10 #5 #10
        self.rnn_batch_size = BUFFERPARA.BATCH_SIZE

        # 初始化模型
        # [修改] 使用 PreAttentionGRU 类
        self.Actor = Actor_PreAttentionGRU()
        self.Critic = Critic_PreAttentionGRU()

        self.actor_rnn_state = None
        self.critic_rnn_state = None

        self.buffer = Buffer(use_rnn=self.use_rnn, use_attn=True)

        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.training_start_time = time.strftime("PPO_PostAttnGRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../../../save/save_evade_fuza两个导弹"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)
        if load_able:
            if model_dir_path:
                self.load_models_from_directory(model_dir_path)
            else:
                self.load_models_from_directory("../../../../test/test_evade")

    def reset_rnn_state(self):
        self.actor_rnn_state = None
        self.critic_rnn_state = None

    def load_models_from_directory(self, directory_path: str):
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
        self.buffer.store_transition(state, value, action, probs, reward, done,
                                     actor_hidden=self.temp_actor_h,
                                     critic_hidden=self.temp_critic_h,
                                     attn_weights=attn_weights)
        if done:
            self.reset_rnn_state()

    def choose_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        current_actor_h = self.actor_rnn_state
        current_critic_h = self.critic_rnn_state

        with torch.no_grad():
            value, self.critic_rnn_state = self.Critic(state_tensor, self.critic_rnn_state)
            dists, attention_weights, self.actor_rnn_state = self.Actor(state_tensor, self.actor_rnn_state)

            attention_weights_for_reward = attention_weights

            continuous_base_dist = dists['continuous']
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            action_cont_tanh = torch.tanh(u)
            # ================= [修改开始] =================
            # 1. 计算原始高斯分布的 log_prob
            log_prob_u = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 2. 计算雅可比修正项 (稳定公式)
            # 公式: 2 * (log 2 - u - softplus(-2u))
            # 注意: u 是 pre-tanh 的值
            correction = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u)).sum(dim=-1)

            # 3. 得到最终动作 a = tanh(u) 的 log_prob
            log_prob_cont = log_prob_u - correction
            # ================= [修改结束] =================
            # log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

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

            # ======================================================================
            # 初始化 temp_hidden: 维度修改为所有实体 GRU 隐藏层的总和
            # ======================================================================
            if current_actor_h is None:
                batch_size = state_tensor.shape[0]
                # [核心修改]
                # 实体总数 = 1(飞机) + NUM_MISSILES
                num_entities = 1 + NUM_MISSILES
                total_hidden_dim = self.Actor.rnn_hidden_dim * num_entities  # 32 * 3 = 96

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

        states, values, actions, old_probs, rewards, dones, _, _, attn_weights = self.buffer.get_all_data()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones, next_value=next_visual_value)
        values = np.squeeze(values)
        returns = advantages + values

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

        for _ in range(self.ppo_epoch):
            if self.use_rnn:
                batch_generator = self.buffer.generate_sequence_batches(
                    self.rnn_seq_len, self.rnn_batch_size, advantages, returns
                )
            else:
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                if self.use_rnn:
                    # (b_s, b_a, b_p, b_adv, b_ret, b_h_a, b_h_c, _) = batch_data
                    # === [修改] 解包时增加 b_v (old values) ===
                    # 注意顺序要和 Buffer 修改后的 yield 顺序一致
                    (b_s, b_a, b_p, b_adv, b_ret, b_v, b_h_a, b_h_c, _) = batch_data

                    state = torch.FloatTensor(b_s).to(**ACTOR_PARA.tpdv)
                    action_batch = torch.FloatTensor(b_a).to(**ACTOR_PARA.tpdv)
                    old_prob = torch.FloatTensor(b_p).to(**ACTOR_PARA.tpdv).view(-1)
                    advantage = torch.FloatTensor(b_adv).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = torch.FloatTensor(b_ret).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    # === [新增] 转换 old_values ===
                    old_value = torch.FloatTensor(b_v).to(**CRITIC_PARA.tpdv).view(-1, 1)
                    # =============================

                    rnn_h_a = torch.FloatTensor(b_h_a).to(**ACTOR_PARA.tpdv)
                    rnn_h_c = torch.FloatTensor(b_h_c).to(**CRITIC_PARA.tpdv)

                    new_dists, _, _ = self.Actor(state, rnn_h_a)
                    new_value, _ = self.Critic(state, rnn_h_c)
                    # 确保 new_value 维度对齐
                    new_value = new_value.view(-1, 1)
                    action_batch = action_batch.view(-1, action_batch.shape[-1])

                else:
                    batch_indices = batch_data
                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv)
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                    # === [新增] 获取非 RNN 模式的 old_value ===
                    old_value = check(values[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                    # =========================================

                    new_dists, _, _ = self.Actor(state)
                    new_value, _ = self.Critic(state)

                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # ================= [修改开始] =================
                # --- 第一步: 计算 Log Prob (用于策略更新 Ratio) ---
                # 必须使用 Replay Buffer 中的旧动作 (u_from_buffer)
                # log(pi(a_old)) = log(pi(u_old)) - log_det_J(u_old)

                # 1.1 计算旧动作在新分布下的高斯 Log Prob
                log_prob_u_buffer = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)

                # 1.2 计算旧动作的雅可比修正项
                correction_buffer = 2.0 * (np.log(2.0) - u_from_buffer - F.softplus(-2.0 * u_from_buffer)).sum(dim=-1)

                # 1.3 得到最终用于 Ratio 计算的 Log Prob
                new_log_prob_cont = log_prob_u_buffer - correction_buffer

                # --- 第二步: 计算 Entropy (用于 Loss 惩罚) ---
                # 必须基于当前策略的新分布进行采样 (rsample)，以保留梯度并消除偏差
                # H(pi) = H(u) + E[log_det_J(u)]

                # 2.1 高斯分布的基础熵 (解析解)
                entropy_base = new_dists['continuous'].entropy().sum(dim=-1)

                # 2.2 重采样 (Reparameterization Trick)
                # 这一步至关重要！它建立了 correction 与当前网络参数(mu, sigma)的梯度联系
                u_curr_sample = new_dists['continuous'].rsample()

                # 2.3 计算新采样动作的雅可比修正项期望
                correction_curr = 2.0 * (np.log(2.0) - u_curr_sample - F.softplus(-2.0 * u_curr_sample)).sum(dim=-1)

                # 2.4 得到最终的无偏熵
                entropy_cont = entropy_base + correction_curr

                # 3. 计算离散部分 Log Prob (保持不变)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc
                # ================= [修改结束] =================

                # new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                # new_log_prob_disc = sum(
                #     new_dists[key].log_prob(discrete_actions_from_buffer[key])
                #     for key in discrete_actions_from_buffer
                # )
                # new_prob = new_log_prob_cont + new_log_prob_disc
                #
                # entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                total_entropy = (entropy_cont + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()

                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

                # # ==========================================================
                # # === [修改] Critic Loss 计算 (增加 Value Clip) ===
                # # ==========================================================
                #
                # # 1. 计算未截断的 Loss
                # critic_loss_unclipped = (new_value - return_) ** 2
                #
                # # 2. 计算截断后的 Value
                # # 使用与 Actor 相同的 epsilon (AGENTPARA.epsilon)
                # # clip_param = AGENTPARA.epsilon
                # # 2. 设定适配你奖励量级的 Clip 参数
                # # 你的回报大概是 100，设为 10 到 20 都是安全的。
                # # 设为 10 会比 20 稍微稳一点（保留一点点截断的约束力）。
                # clip_param = 20.0  # 建议取个折中值，比如 15.0
                #
                # v_clipped = old_value + torch.clamp(new_value - old_value, -clip_param, clip_param)
                #
                # # 3. 计算截断后的 Loss
                # critic_loss_clipped = (v_clipped - return_) ** 2
                #
                # # 4. 取两者的最大值 (因为是 Loss，我们在最小化，取 max 意味着更悲观/保守的更新)
                # critic_loss_cost = torch.max(critic_loss_unclipped, critic_loss_clipped)
                #
                # # 5. 求平均作为最终 Loss (通常系数为 0.5)
                # critic_loss = 0.5 * critic_loss_cost.mean()
                #
                # # ==========================================================

                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

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