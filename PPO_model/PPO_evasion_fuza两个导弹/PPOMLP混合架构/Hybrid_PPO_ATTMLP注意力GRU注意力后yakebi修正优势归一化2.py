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
MISSILE_FEAT_DIM = 5  # 导弹特征改为 5 维: [dist_min, dist_max, beta_sin, beta_cos, theta_L]
# AIRCRAFT_FEAT_DIM = 7  # 飞机特征改为 7 维: [av, h, ae, am_sin, am_cos, ir, q]
AIRCRAFT_FEAT_DIM = 6  # <--- 修改这里：去掉 q，维度变为 6: [av, h, ae, am_sin, am_cos, ir]
FULL_OBS_DIM = (NUM_MISSILES * MISSILE_FEAT_DIM) + AIRCRAFT_FEAT_DIM

ENTITY_EMBED_DIM = 64 #32 #64
ATTN_NUM_HEADS = 2

assert ENTITY_EMBED_DIM % ATTN_NUM_HEADS == 0, "ENTITY_EMBED_DIM must be divisible by ATTN_NUM_HEADS"


# ==============================================================================
#           <<< 核心修改 >>>: Post-Attention GRU 架构 (Actor)
# ==============================================================================

class Actor_PostAttentionGRU(Module):
    """
    Actor 网络 - [架构: Encoders -> Attention -> Fusion -> GRU -> MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=ENTITY_EMBED_DIM):
        super(Actor_PostAttentionGRU, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # # ======================================================================
        # # 1. 动作标准差 (Std) 设置 - 核心优化部分
        # # ======================================================================
        # # 下限 0.05: 保持 5% 的底噪，防止策略过早塌缩为确定性，维持鲁棒性
        # self.target_std_min = 0.05
        # # 上限 1.0: 允许少量双峰分布（大机动探索），但避免 1.5 带来的过度 Bang-Bang 控制
        # self.target_std_max = 1.0
        # # 初始 0.6: 位于 0.7 临界点之下，保证初期为单峰分布，飞机飞行平稳
        # self.target_init_std = 0.95

        self.target_std_min = 0.10 #0.20 #0.10 #0.20 #0.05  # 保证底噪
        self.target_std_max = 0.60 #0.80 #0.90 #0.70 #0.80  # 降低上限，避免完全随机
        self.target_init_std = 0.60 #0.75 #0.85 #0.65 #0.75  # 初始值设为中间态，不要设为 max

        # 转换为 Log 空间边界
        self.log_std_min = np.log(self.target_std_min)  # ln(0.05) ≈ -2.99
        self.log_std_max = np.log(self.target_std_max)  # ln(1.0) = 0.0

        self.weight_decay = weight_decay

        # 配置
        self.rnn_hidden_dim = 128 #64 #128 #ENTITY_EMBED_DIM
        self.entity_embed_dim = ENTITY_EMBED_DIM
        self.encoder_hidden_dim = ENTITY_EMBED_DIM

        # # 1. 编码器 (Feature Extraction)
        # # [修改] 恢复飞机编码器，确保 Query 和 Key 在同一语义空间
        # self.missile_encoder = Sequential(
        #     Linear(MISSILE_FEAT_DIM, self.encoder_hidden_dim),
        # )
        # # self.aircraft_encoder = Sequential(
        # #     Linear(FULL_OBS_DIM, self.encoder_hidden_dim),  # 使用全观测或仅飞机特征均可，这里用FULL方便
        # # )
        #
        # # 修改后
        # self.aircraft_encoder = Sequential(
        #     Linear(AIRCRAFT_FEAT_DIM, self.encoder_hidden_dim),
        # )

        # =====================================================================
        # 修改后：增加非线性映射能力
        # =====================================================================
        # # 定义一个中间隐藏层大小，通常可以和 embedding dim 一样，或者稍微小一点
        # encoder_mid_dim = 64

        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU(),
            # nn.LayerNorm(encoder_mid_dim), # 推荐加上 LN，有助于特征标准化
            # Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
            # 这里最后不需要再加激活函数，因为后面马上进 Attention 的 Dot Product
            # 让特征保持线性空间分布对 Attention 计算更有利
        )

        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU(),
            # nn.LayerNorm(encoder_mid_dim),
            # Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )

        # # <<< Pre-Norm 层 >>>
        # # 用于 Attention 的 Query (飞机) 和 Key (导弹)
        # self.q_norm = nn.LayerNorm(self.entity_embed_dim)
        # self.k_norm = nn.LayerNorm(self.entity_embed_dim)

        # 2. 交叉注意力 (保持不变)
        self.attention = MultiheadAttention(
            embed_dim=self.entity_embed_dim,
            num_heads=ATTN_NUM_HEADS,
            dropout=0.0,
            batch_first=True
        )

        # # 注意力拼接后维度 = 2D
        # self.fusion_proj = nn.Linear(self.entity_embed_dim * 2,
        #                              self.entity_embed_dim)

        # self.fusion_norm = nn.LayerNorm(self.entity_embed_dim)

        # # 3. GRU 层 (Global Memory)
        # # [修改] GRU 移到这里。输入维度是 飞机特征 + 注意力上下文
        # self.global_gru = nn.GRU(
        #     input_size=self.entity_embed_dim * 2,  # Concat(Aircraft, Attn_Out)
        #     hidden_size=self.rnn_hidden_dim,
        #     batch_first=True
        # )
        # ===== GRU 后：Skip-Concat + Projection =====
        # self.gru_skipconcat_dim = self.entity_embed_dim + self.rnn_hidden_dim  # 128 + 128 = 256
        # self.post_gru_concat_norm = nn.LayerNorm(self.gru_skipconcat_dim)
        # self.gru_concat_proj = nn.Linear(self.gru_skipconcat_dim, self.rnn_hidden_dim)  # 256 -> 128

        # [修改 2] 添加 Layer Normalization
        # GRU输入维度是 entity_embed_dim * 2，输出是 rnn_hidden_dim
        # 残差连接后的维度是 input_size + hidden_size
        residual_dim = self.entity_embed_dim * 2
        # self.layer_norm = nn.LayerNorm(residual_dim)

        # MLP 决策层
        mlp_input_dim = residual_dim  # 使用残差连接后的维度
        # 4. MLP 决策层
        # [修改] 输入维度现在直接是 GRU 的 hidden_dim
        # mlp_input_dim = self.rnn_hidden_dim

        # gru_input_dim = self.entity_embed_dim * 2
        # mlp_input_dim = self.rnn_hidden_dim + gru_input_dim

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

        # =========================
        # 5) GRU 挪到 shared_base_mlp 后面（关键修改）
        # =========================
        self.global_gru = nn.GRU(
            input_size=base_output_dim,  # ✅ 改成共享层输出维度
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        # =========================
        # 6) Tower 的输入维度改成：base_output_dim + rnn_hidden_dim（如果你要 skip-concat）
        # =========================
        tower_input_dim = base_output_dim + self.rnn_hidden_dim

        # 连续动作塔楼
        self.continuous_tower = nn.Sequential()
        cur_dim = tower_input_dim
        for i, dim in enumerate(tower_dims):
            self.continuous_tower.add_module(f'cont_tower_fc_{i}', nn.Linear(cur_dim, dim))
            self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', nn.LeakyReLU())
            cur_dim = dim
        continuous_tower_output_dim = tower_dims[-1] if tower_dims else tower_input_dim

        # 离散动作塔楼
        self.discrete_tower = nn.Sequential()
        cur_dim = tower_input_dim
        for i, dim in enumerate(tower_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', nn.Linear(cur_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', nn.LeakyReLU())
            cur_dim = dim
        discrete_tower_output_dim = tower_dims[-1] if tower_dims else tower_input_dim

        self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)

        # 初始化为 -0.5 左右 (std ≈ 0.6)，比 1.0 稳健，又比 0.1 有探索性
        init_log_std = np.log(self.target_init_std)
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), init_log_std))
        # # =====================================================
        # # 2. 软限制参数初始化
        # # =====================================================
        # # 计算验证：
        # # Sigmoid(2.0) ≈ 0.88
        # # LogStd ≈ ln(0.05) + 0.88 * (ln(1.5) - ln(0.05)) ≈ 0.0
        # # Std ≈ 1.0 (完美初始值)
        #
        # init_value = 2.5 #2.0
        # self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), init_value))

        # # --- __init__ 中 ---
        # self.pre_gru_norm = nn.LayerNorm(self.entity_embed_dim)
        # # 用于 GRU 残差后的归一化 (Addition 后建议加 LN)
        # self.post_residual_norm = nn.LayerNorm(self.rnn_hidden_dim)

        # 优化器
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
        self._init_weights()  # 必须进行权重初始化

    def _init_weights(self):
        for m in self.modules():
            # 1. 线性层通用初始化
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 2. GRU 特殊初始化 (关键！不要漏掉)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

            # 3. LayerNorm 初始化
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # --- 特殊处理：策略输出头 (最后覆盖前面的通用初始化) ---

        # 连续动作头：确保均值接近 0，避免 Tanh 饱和
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)

        # 离散动作头：确保初始概率均匀 (Max Entropy)
        nn.init.orthogonal_(self.discrete_head.weight, gain=0.01)
        nn.init.constant_(self.discrete_head.bias, 0)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        h_prev = rnn_state

        # --- 数据提取 ---
        obs_flat_raw = obs_tensor.view(-1, FULL_OBS_DIM)
        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # 1. 编码 (Spatial Encoding)
        # 导弹
        missiles_raw = torch.cat([missile1_obs, missile2_obs], dim=0)
        missiles_embed_flat = self.missile_encoder(missiles_raw.view(-1, MISSILE_FEAT_DIM))
        missiles_embed_seq = missiles_embed_flat.view(batch_size * 2, seq_len, self.entity_embed_dim)
        m1_feat_seq, m2_feat_seq = torch.split(missiles_embed_seq, batch_size, dim=0)

        # m1_feat_flat = m1_feat_seq.reshape(-1, self.entity_embed_dim)
        # m2_feat_flat = m2_feat_seq.reshape(-1, self.entity_embed_dim)

        # 飞机 (现在通过Encoder，而不是直接进GRU)
        # air_embed_seq = self.aircraft_encoder(obs_tensor)  # [B, Seq, Dim]
        air_embed_seq = self.aircraft_encoder(aircraft_obs)
        air_embed_flat = air_embed_seq.reshape(-1, self.entity_embed_dim)

        # 2. Attention (Spatial Relation)
        m1_raw = obs_flat_raw[..., 0:MISSILE_FEAT_DIM]
        m2_raw = obs_flat_raw[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        # inactive_fingerprint = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        # <<< 修改开始：更新无效导弹指纹 >>>
        # 原代码可能是 [1.0, 1.0, 0.0, 1.0, 0.0]，这是正确的。
        # 对应环境中的非激活观测值: [dist_min=1, dist_max=1, sin=0, cos=1, theta=0]
        # 确保这里的值与环境代码中的完全一致
        inactive_fingerprint = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        # <<< 修改结束 >>>
        is_m1_inactive = torch.all(torch.isclose(m1_raw, inactive_fingerprint), dim=-1)
        is_m2_inactive = torch.all(torch.isclose(m2_raw, inactive_fingerprint), dim=-1)
        attention_mask = torch.stack([is_m1_inactive, is_m2_inactive], dim=1)

        # query = air_embed_flat.unsqueeze(1)
        # keys = torch.stack([m1_feat_flat, m2_feat_flat], dim=1)
        # [Pre-Norm 步骤 A]: 对 Query (飞机) 进行归一化
        # 这里使用 q_norm 作为 Residual Block 内部的 Pre-Norm
        # air_embed_flat_normed = self.q_norm(air_embed_flat)
        # query = air_embed_flat_normed.unsqueeze(1)  # [B*S, 1, Dim]
        # Query: 飞机 (保留原始物理特征分布)
        # 此时 query 包含绝对的物理量级信息
        query = air_embed_flat.unsqueeze(1)

        # 准备 Keys/Values (Keys 进行 Norm，Values 保持原样或也 Norm 视情况，这里保持原逻辑 K_norm)
        m_entities_raw = torch.stack([m1_feat_seq.reshape(-1, self.entity_embed_dim),
                                      m2_feat_seq.reshape(-1, self.entity_embed_dim)], dim=1)
        keys = m_entities_raw   # 不归一化 self.k_norm(m_entities_raw)
        values = m_entities_raw
        # 2. Attention 计算 (现在的 query 和 keys 都是归一化后的)
        attn_output, attn_weights = self.attention(query, keys, values, key_padding_mask=attention_mask)
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0)

        # 3. 特征融合 (Fusion)
        # 将飞机自身的理解与对环境威胁的理解拼接
        # [Batch*Seq, 1, Dim] -> [Batch*Seq, Dim]
        # 使用原始飞机特征拼接原始威胁特征，保持物理量级一致
        # fusion_features_flat = torch.cat([air_embed_flat, attn_output.squeeze(1)], dim=-1)

        # [Pre-Norm 步骤 B]: 残差连接 (Residual Connection)
        # 公式: x = x + Attention(LN(x))
        # 注意: 加的是原始的 air_embed_flat (未归一化的)，而不是 air_embed_flat_normed
        # fusion_features_flat = air_embed_flat + attn_output.squeeze(1)  # [B*S, Dim]

        attn_flat = attn_output.squeeze(1)
        # 拼接
        fusion_features_flat = torch.cat([air_embed_flat, attn_flat], dim=-1)  # [B*S, 2D]
        # fusion_features_flat = air_embed_flat + attn_output.squeeze(1)  # [B*S, Dim]

        # # 投影回 D 维
        # fusion_features_flat = self.fusion_proj(fusion_cat)

        # # 稳定训练（强烈推荐）
        # fusion_features_flat = self.fusion_norm(fusion_features_flat)

        # 4. GRU (Temporal Processing - Post Attention)
        # 恢复序列维度以进入GRU: [Batch, Seq, Dim*2]
        fusion_features_seq = fusion_features_flat.view(batch_size, seq_len, -1)

        # 先喂共享层（注意：shared_base_mlp 是 Linear/激活，支持 3D 输入）
        base_seq = self.shared_base_mlp(fusion_features_seq)  # [B, T, base_output_dim]

        # fusion_features_seq_norm = self.pre_gru_norm(fusion_features_seq)  # [LN A] 保护 GRU

        gru_out, next_h = self.global_gru(base_seq, h_prev)

        # 可选：像你现在一样做 skip-concat（推荐）
        post_gru_seq = torch.cat([base_seq, gru_out], dim=-1)  # [B, T, base_output_dim + H]

        # 展平给 tower
        tower_input = post_gru_seq.reshape(-1, post_gru_seq.shape[-1])  # [B*T, ...]

        # [修改点 2]：实现残差/跳跃连接
        # 将 GRU 的输出与 GRU 的输入拼接
        # gru_out shape: [Batch, Seq, Hidden]
        # fusion_features_seq shape: [Batch, Seq, Input_Dim]

        # # 5. 残差连接 + LayerNorm (关键修改点)
        # # 将 GRU 的输出与 GRU 的输入拼接
        # residual_features = torch.cat([fusion_features_seq, gru_out], dim=-1)
        #
        # residual_features = self.post_residual_norm(residual_features)  # [LN B] 保护 MLP
        # residual_features = gru_out + fusion_features_seq
        # residual_features = self.post_residual_norm(residual_features)

        # # ===== GRU 后：Skip-Concat + LN + Projection =====
        # residual_features = torch.cat([fusion_features_seq, gru_out], dim=-1)  # [B, T, 256]
        # residual_features = self.post_gru_concat_norm(residual_features)  # 稳定 PPO
        # residual_features = self.gru_concat_proj(residual_features)  # [B, T, 128]

        # [新增] 对拼接后的特征进行 LayerNorm
        # residual_features = self.layer_norm(residual_features)

        # # 展平送入 MLP
        # mlp_input = residual_features.reshape(-1, residual_features.shape[-1])

        # 拼接: [Batch, Seq, Hidden + Input_Dim]
        # residual_features = torch.cat([fusion_features_seq, gru_out], dim=-1)

        # 展平送入 MLP
        # mlp_input = residual_features.reshape(-1, residual_features.shape[-1])

        # 准备进入 MLP 的数据
        # mlp_input = gru_out.reshape(-1, self.rnn_hidden_dim)

        # 5. MLP 决策
        # base_features = self.shared_base_mlp(mlp_input)
        continuous_features = self.continuous_tower(tower_input)
        discrete_features = self.discrete_tower(tower_input)

        mu = self.mu_head(continuous_features)

        # 强行把均值限制在 [-2, 2] 或 [-3, 3] 之间
        # 只要不让它跑到 10 这种离谱的值就行
        mu = torch.clamp(mu, -3.0, 3.0)

        all_disc_logits = self.discrete_head(discrete_features)

        # Masking 处理 (保持不变)
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts
        # <<< 修改开始：更新诱饵弹信息索引 >>>
        # 飞机特征结构: [av, h, ae, am_sin, am_cos, ir, q]
        # o_ir_norm 是第 6 个元素 (索引为 5)
        # 全局索引 = 导弹部分总长 + 飞机内部索引
        flare_info_index = 2 * MISSILE_FEAT_DIM + 5
        has_flares_info = obs_flat_raw[..., flare_info_index]
        # mask = (has_flares_info == 0).view(-1)
        # <<< 修改后 (正确) >>>
        # 因为环境归一化后，0发对应 -1.0。考虑到浮点数误差，我们判断是否小于 -0.99
        mask = (has_flares_info <= -0.99).view(-1)

        NEG_INF = -1e8
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            mask_expanded = mask.unsqueeze(-1) if mask.dim() < trigger_logits_masked.dim() else mask
            trigger_logits_masked = torch.where(mask_expanded,
                                                torch.full_like(trigger_logits, NEG_INF),
                                                trigger_logits)

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

        # =========== 修改: 限制标准差应用 ===========
        # 使用之前计算好的 log 界限进行截断
        # log_std_min = ln(0.01), log_std_max = ln(0.6)
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)

        # # =====================================================
        # # 3. 计算动态标准差 (Soft Mapping)
        # # =====================================================
        # # output = min + (max - min) * sigmoid(param)
        #
        # # 1. 将无界参数压缩到 (0, 1)
        # norm_val = torch.sigmoid(self.log_std_param)
        #
        # # 2. 映射到 log 范围 [log_min, log_max]
        # log_std = self.log_std_min + norm_val * (self.log_std_max - self.log_std_min)
        #
        # # 3. 转回 std
        # std = torch.exp(log_std).expand_as(mu)
        # 此时 std 的值一定在 [0.01, 0.6] 之间
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

        return distributions, attention_to_missiles, next_h


# ==============================================================================
#           <<< 核心修改 >>>: Post-Attention GRU 架构 (Critic)
# ==============================================================================

class Critic_PostAttentionGRU(Module):
    """
    Critic 网络 - [架构: Encoders -> Attention -> Fusion -> GRU -> MLP]
    """

    def __init__(self, weight_decay=1e-4, rnn_hidden_dim=ENTITY_EMBED_DIM):
        super(Critic_PostAttentionGRU, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.weight_decay = weight_decay

        self.rnn_hidden_dim = 128 #ENTITY_EMBED_DIM
        self.entity_embed_dim = ENTITY_EMBED_DIM
        self.encoder_hidden_dim = ENTITY_EMBED_DIM

        # # 1. Encoders
        # self.missile_encoder = Sequential(
        #     Linear(MISSILE_FEAT_DIM, self.encoder_hidden_dim),
        # )
        # # self.aircraft_encoder = Sequential(
        # #     Linear(FULL_OBS_DIM, self.encoder_hidden_dim),
        # # )
        # # 修改后
        # self.aircraft_encoder = Sequential(
        #     Linear(AIRCRAFT_FEAT_DIM, self.encoder_hidden_dim),
        # )

        # =====================================================================
        # 修改后：增加非线性映射能力
        # =====================================================================
        # # 定义一个中间隐藏层大小，通常可以和 embedding dim 一样，或者稍微小一点
        # encoder_mid_dim = 64

        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU(),
            # nn.LayerNorm(encoder_mid_dim), # 推荐加上 LN，有助于特征标准化
            # Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
            # 这里最后不需要再加激活函数，因为后面马上进 Attention 的 Dot Product
            # 让特征保持线性空间分布对 Attention 计算更有利
        )

        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM),
            # LeakyReLU(),
            # nn.LayerNorm(encoder_mid_dim),
            # Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )

        # # <<< Pre-Norm 层 >>>
        # self.q_norm = nn.LayerNorm(self.entity_embed_dim)
        # self.k_norm = nn.LayerNorm(self.entity_embed_dim)

        # 2. Attention
        self.attention = MultiheadAttention(
            embed_dim=self.entity_embed_dim,
            num_heads=ATTN_NUM_HEADS,
            batch_first=True
        )

        # self.fusion_proj = nn.Linear(self.entity_embed_dim * 2,
        #                              self.entity_embed_dim)
        # self.fusion_norm = nn.LayerNorm(self.entity_embed_dim)

        # # 3. Post-Attention GRU
        # self.global_gru = nn.GRU(
        #     input_size=self.entity_embed_dim * 2,  # Concat Input
        #     hidden_size=self.rnn_hidden_dim,
        #     batch_first=True
        # )
        #
        # # ===== GRU 后：Skip-Concat + Projection =====
        # # self.gru_skipconcat_dim = self.entity_embed_dim + self.rnn_hidden_dim
        # # self.post_gru_concat_norm = nn.LayerNorm(self.gru_skipconcat_dim)
        # # self.gru_concat_proj = nn.Linear(self.gru_skipconcat_dim, self.rnn_hidden_dim)
        #
        # # [!!! 修正这里 !!!] 定义 LayerNorm
        # residual_dim =  self.rnn_hidden_dim #+ (self.entity_embed_dim * 2)
        # # self.layer_norm = nn.LayerNorm(residual_dim)
        #
        # # 4. MLP
        # mlp_dims = CRITIC_PARA.model_layer_dim
        # self.mlp = Sequential()
        # # input_dim = self.rnn_hidden_dim  # 来自 GRU 的输出
        # gru_input_dim = self.entity_embed_dim * 2
        # input_dim = self.rnn_hidden_dim + gru_input_dim
        # for i, dim in enumerate(mlp_dims):
        #     self.mlp.add_module(f'fc_{i}', Linear(input_dim, dim))
        #     self.mlp.add_module(f'act_{i}', LeakyReLU())
        #     input_dim = dim
        # self.fc_out = Linear(input_dim, self.output_dim)

        split_point = 2
        mlp_dims = CRITIC_PARA.model_layer_dim
        base_dims = mlp_dims[:split_point]
        post_dims = mlp_dims[split_point:]

        self.shared_base_mlp = nn.Sequential()
        base_input_dim = self.entity_embed_dim * 2  # fusion concat => 2D
        for i, dim in enumerate(base_dims):
            self.shared_base_mlp.add_module(f'base_fc_{i}', nn.Linear(base_input_dim, dim))
            self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', nn.LeakyReLU())
            base_input_dim = dim

        base_output_dim = base_dims[-1] if base_dims else (self.entity_embed_dim * 2)

        self.global_gru = nn.GRU(
            input_size=base_output_dim,  # ✅ 改成共享层输出维度
            hidden_size=self.rnn_hidden_dim,
            batch_first=True
        )

        post_gru_input_dim = base_output_dim + self.rnn_hidden_dim

        self.post_gru_mlp = nn.Sequential()
        cur_dim = post_gru_input_dim
        for i, dim in enumerate(post_dims):
            self.post_gru_mlp.add_module(f'post_fc_{i}', nn.Linear(cur_dim, dim))
            self.post_gru_mlp.add_module(f'post_leakyrelu_{i}', nn.LeakyReLU())
            cur_dim = dim

        post_gru_output_dim = post_dims[-1] if post_dims else post_gru_input_dim

        self.fc_out = nn.Linear(post_gru_output_dim, self.output_dim)

        # # --- __init__ 中 ---
        # self.pre_gru_norm = nn.LayerNorm(self.entity_embed_dim)
        # # 新增
        # self.post_residual_norm = nn.LayerNorm(self.rnn_hidden_dim)

        # Optimizer
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
        self._init_weights()

    def _init_weights(self):
        # 1. 遍历所有模块进行通用初始化
        for m in self.modules():
            # 线性层 (Hidden Layers)：配合 LeakyReLU/ReLU
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # GRU 层：防止梯度消失/爆炸
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

            # LayerNorm 层 (如果你加了的话)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # 2. --- 特殊处理：Critic 输出头 ---
        # 覆盖掉上面的通用初始化
        # 因为 fc_out 后面没有激活函数，所以 gain 使用 1.0 (线性层的标准值)
        # 这样初始的价值估计 V(s) 会在 0 附近波动
        nn.init.orthogonal_(self.fc_out.weight, gain=1.0)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, obs, rnn_state=None):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        batch_size, seq_len, _ = obs_tensor.shape

        h_prev = rnn_state

        obs_flat_raw = obs_tensor.view(-1, FULL_OBS_DIM)
        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # 1. Encoding
        # air_embed_seq = self.aircraft_encoder(obs_tensor)
        air_embed_seq = self.aircraft_encoder(aircraft_obs)
        air_embed_flat = air_embed_seq.reshape(-1, self.entity_embed_dim)

        missiles_raw = torch.cat([missile1_obs, missile2_obs], dim=0)
        missiles_embed_flat = self.missile_encoder(missiles_raw.view(-1, MISSILE_FEAT_DIM))
        missiles_embed_seq = missiles_embed_flat.view(batch_size * 2, seq_len, self.entity_embed_dim)
        m1_feat_seq, m2_feat_seq = torch.split(missiles_embed_seq, batch_size, dim=0)
        # m1_feat_flat = m1_feat_seq.reshape(-1, self.entity_embed_dim)
        # m2_feat_flat = m2_feat_seq.reshape(-1, self.entity_embed_dim)

        # 2. Attention
        m1_raw = obs_flat_raw[..., 0:MISSILE_FEAT_DIM]
        m2_raw = obs_flat_raw[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        # <<< 修改开始：更新无效导弹指纹 (与 Actor 保持一致) >>>
        # 确保这里的值与环境代码中的完全一致: [1.0, 1.0, 0.0, 1.0, 0.0]
        inactive = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        # <<< 修改结束 >>>
        is_m1_in = torch.all(torch.isclose(m1_raw, inactive), dim=-1)
        is_m2_in = torch.all(torch.isclose(m2_raw, inactive), dim=-1)
        mask = torch.stack([is_m1_in, is_m2_in], dim=1)

        # query = air_embed_flat.unsqueeze(1)
        # keys = torch.stack([m1_feat_flat, m2_feat_flat], dim=1)
        # <<< 修改：Attention 前的归一化 >>>
        # # 1. 准备 Q, K, V (V不归一化)
        # # [Pre-Norm 步骤 A]
        # air_embed_flat_normed = self.q_norm(air_embed_flat)
        # query = air_embed_flat_normed.unsqueeze(1)

        query = air_embed_flat.unsqueeze(1)

        m_entities_raw = torch.stack([m1_feat_seq.reshape(-1, self.entity_embed_dim),
                                      m2_feat_seq.reshape(-1, self.entity_embed_dim)], dim=1)
        keys = m_entities_raw #self.k_norm(m_entities_raw)
        values = m_entities_raw

        attn_out, _ = self.attention(query, keys, values, key_padding_mask=mask)
        if torch.isnan(attn_out).any(): attn_out = torch.nan_to_num(attn_out, nan=0.0)

        # 3. Fusion
        # 同样使用未归一化的飞机特征进行拼接
        # fusion_features_flat = torch.cat([air_embed_flat, attn_out.squeeze(1)], dim=-1)

        # [Pre-Norm 步骤 B]: 残差相加
        # x = x + Attention(LN(x))
        # fusion_features_flat = air_embed_flat + attn_out.squeeze(1)
        attn_flat = attn_out.squeeze(1)
        # # --- Fusion: [B*S, 2D] -> [B, T, 2D] ---
        fusion_features_flat = torch.cat([air_embed_flat, attn_flat], dim=-1)  # [B*S, 2D]
        # fusion_features_flat = air_embed_flat + attn_out.squeeze(1)
        fusion_features_seq = fusion_features_flat.view(batch_size, seq_len, -1)  # [B, T, 2D]

        # =========================
        # (A) 先过共享层 Shared Base（关键修改）
        # =========================
        base_seq = self.shared_base_mlp(fusion_features_seq)  # [B, T, base_output_dim]

        # =========================
        # (B) 再过 GRU（关键修改）
        # =========================
        gru_out, next_h = self.global_gru(base_seq, h_prev)  # [B, T, H]

        # =========================
        # (C) Skip-Concat（不投影、不LN，保持你风格）
        # =========================
        post_gru_seq = torch.cat([base_seq, gru_out], dim=-1)  # [B, T, base_output_dim + H]

        # =========================
        # (D) post_gru_mlp + value head
        # =========================
        post_features = self.post_gru_mlp(post_gru_seq)  # [B, T, post_dim]

        if seq_len == 1:
            post_features = post_features.squeeze(1)  # [B, post_dim]
        value = self.fc_out(post_features)  # [B, 1] or [B, output_dim]

        return value, next_h

        # fusion_features_flat = torch.cat([air_embed_flat, attn_flat], dim=-1)
        # # fusion_features_flat = self.fusion_proj(fusion_cat)
        # # fusion_features_flat = self.fusion_norm(fusion_features_flat)
        #
        # # 4. GRU
        # fusion_features_seq = fusion_features_flat.view(batch_size, seq_len, -1)
        #
        # # fusion_features_seq_norm = self.pre_gru_norm(fusion_features_seq)  # [LN A] 保护 GRU
        #
        # gru_out, next_h = self.global_gru(fusion_features_seq, h_prev)
        #
        # # # [修改点 2]：残差拼接
        # # residual_features = torch.cat([fusion_features_seq, gru_out], dim=-1)
        # #
        # # residual_features = self.post_residual_norm(residual_features)  # [LN B] 保护 MLP
        # # residual_features = gru_out + fusion_features_seq
        #
        # residual_features = torch.cat([fusion_features_seq, gru_out], dim=-1)
        # # residual_features = self.post_gru_concat_norm(residual_features)
        # # residual_features = self.gru_concat_proj(residual_features)
        #
        # # 加上这一行：
        # # residual_features = self.post_residual_norm(residual_features)
        #
        # # 5. MLP
        # mlp_input = residual_features.reshape(-1, residual_features.shape[-1])
        #
        # # 5. MLP
        # # mlp_input = gru_out.reshape(-1, self.rnn_hidden_dim)
        # val = self.fc_out(self.mlp(mlp_input))
        #
        # return val, next_h


class PPO_continuous(object):
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = True):
        super(PPO_continuous, self).__init__()

        self.use_rnn = use_rnn  # True
        print(f"--- 初始化 PPO Agent (Post-Attention GRU) use_rnn={self.use_rnn} ---")

        self.rnn_seq_len = 5 #15 #12 #15 #12 #20
        self.rnn_batch_size = BUFFERPARA.BATCH_SIZE

        # 初始化模型
        self.Actor = Actor_PostAttentionGRU()
        self.Critic = Critic_PostAttentionGRU()

        self.actor_rnn_state = None
        self.critic_rnn_state = None

        self.buffer = Buffer(use_rnn=self.use_rnn, use_attn=True)

        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.training_start_time = time.strftime("PPO_PostAttnGRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../../save/save_evade_fuza两个导弹"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)
        if load_able:
            if model_dir_path:
                self.load_models_from_directory(model_dir_path)
            else:
                self.load_models_from_directory("../../../test/test_evade")

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
            # 初始化 temp_hidden: 维度仅为 1 个 hidden_dim
            # ======================================================================
            if current_actor_h is None:
                batch_size = state_tensor.shape[0]
                total_hidden_dim = self.Actor.rnn_hidden_dim
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

    # def learn(self, next_visual_value=0.0):
    #     """
    #     执行 PPO 的学习和更新步骤。
    #     集成特性：
    #     1. Post-Attention GRU 架构 (支持全序列输入)
    #     2. 全局优势归一化
    #     3. 雅可比修正 (Jacobian Correction)
    #     4. 💥 修复版：针对 RNN 模式的维度对齐与最后一步切片训练
    #     """
    #     # 如果 Buffer 中的数据不足一个批次，则跳过学习
    #     if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
    #         return None
    #
    #     # 1. 提取所有数据 (Numpy 数组)
    #     states, values, actions, old_probs, rewards, dones, _, _, attn_weights = self.buffer.get_all_data()
    #
    #     # 2. 计算 GAE 优势
    #     advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones, next_value=next_visual_value)
    #
    #     # 维度对齐处理
    #     values = np.squeeze(values)
    #     if values.ndim == 1:
    #         values = values.reshape(-1, 1)
    #     if advantages.ndim == 1:
    #         advantages = advantages.reshape(-1, 1)
    #
    #     returns = advantages + values
    #     adv_mean = np.mean(advantages)
    #     adv_std = np.std(advantages)
    #     advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    #
    #     train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
    #                   'ratio': []}
    #
    #     # 3. PPO 更新循环
    #     for _ in range(self.ppo_epoch):
    #         if self.use_rnn:
    #             batch_generator = self.buffer.generate_sequence_batches(
    #                 self.rnn_seq_len, self.rnn_batch_size, advantages, returns
    #             )
    #         else:
    #             batch_generator = self.buffer.generate_batches()
    #
    #         for batch_data in batch_generator:
    #             if self.use_rnn:
    #                 # [GRU模式] 解包数据
    #                 (b_s, b_a, b_p, b_adv, b_ret, b_v, b_h_a, b_h_c, _) = batch_data
    #
    #                 # 获取当前的 Batch 和 Seq 长度用于后续还原
    #                 batch_size = b_s.shape[0]
    #                 seq_len = b_s.shape[1]
    #
    #                 # 转 Tensor：保留序列结构 (B, S, ...)
    #                 state = torch.FloatTensor(b_s).to(**ACTOR_PARA.tpdv)
    #                 action_batch = torch.FloatTensor(b_a).to(**ACTOR_PARA.tpdv)
    #                 old_prob = torch.FloatTensor(b_p).to(**ACTOR_PARA.tpdv)  # (B, S)
    #                 advantage = torch.FloatTensor(b_adv).to(**ACTOR_PARA.tpdv)  # (B, S, 1)
    #                 return_ = torch.FloatTensor(b_ret).to(**CRITIC_PARA.tpdv)  # (B, S, 1)
    #
    #                 rnn_h_a = torch.FloatTensor(b_h_a).to(**ACTOR_PARA.tpdv)
    #                 rnn_h_c = torch.FloatTensor(b_h_c).to(**CRITIC_PARA.tpdv)
    #
    #                 # 前向传播：模型内部会处理序列，输出通常是展平的 (B*S, Dim)
    #                 new_dists, _, _ = self.Actor(state, rnn_h_a)
    #                 new_value, _ = self.Critic(state, rnn_h_c)
    #
    #                 # 【💥 关键修复】：将 Buffer 里的动作展平，以匹配 new_dists 的 (B*S) 形状
    #                 u_from_buffer = action_batch[..., :CONTINUOUS_DIM].reshape(-1, CONTINUOUS_DIM)
    #                 discrete_actions_from_buffer = {
    #                     'trigger': action_batch[..., CONTINUOUS_DIM].reshape(-1),
    #                     'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].reshape(-1).long(),
    #                     'num_groups': action_batch[..., CONTINUOUS_DIM + 2].reshape(-1).long(),
    #                     'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].reshape(-1).long(),
    #                 }
    #             else:
    #                 # [MLP模式]
    #                 batch_indices = batch_data
    #                 state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
    #                 action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
    #                 old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv)
    #                 advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
    #                 return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)
    #
    #                 new_dists, _, _ = self.Actor(state)
    #                 new_value, _ = self.Critic(state)
    #
    #                 u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
    #                 discrete_actions_from_buffer = {
    #                     'trigger': action_batch[..., CONTINUOUS_DIM],
    #                     'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
    #                     'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
    #                     'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
    #                 }
    #
    #             # --- 雅可比修正与 LogProb 计算 ---
    #             # 此时所有的输入都是展平后的，计算出的 log_prob 形状为 (B*S) 或 (B)
    #             log_prob_u_buffer = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
    #             correction_buffer = 2.0 * (np.log(2.0) - u_from_buffer - F.softplus(-2.0 * u_from_buffer)).sum(dim=-1)
    #             new_log_prob_cont = log_prob_u_buffer - correction_buffer
    #
    #             entropy_base = new_dists['continuous'].entropy().sum(dim=-1)
    #             u_curr_sample = new_dists['continuous'].rsample()
    #             correction_curr = 2.0 * (np.log(2.0) - u_curr_sample - F.softplus(-2.0 * u_curr_sample)).sum(dim=-1)
    #             entropy_cont = entropy_base + correction_curr
    #
    #             new_log_prob_disc = sum(
    #                 new_dists[key].log_prob(discrete_actions_from_buffer[key]) for key in discrete_actions_from_buffer)
    #
    #             new_prob = new_log_prob_cont + new_log_prob_disc
    #             entropy_disc = sum(dist.entropy() for key, dist in new_dists.items() if key != 'continuous')
    #
    #             # =================== [💥 核心逻辑：还原形状并切片提取最后一步] ===================
    #             if self.use_rnn:
    #                 # 1. 将展平的 (B*S) 还原回 (B, S)
    #                 new_prob = new_prob.view(batch_size, seq_len)
    #                 entropy_cont = entropy_cont.view(batch_size, seq_len)
    #                 entropy_disc = entropy_disc.view(batch_size, seq_len)
    #                 # new_value 从 (B*S, 1) 还原回 (B, S, 1)
    #                 new_value = new_value.view(batch_size, seq_len, 1)
    #
    #                 # 2. 提取最后一步 [:, -1]
    #                 new_prob_train = new_prob[:, -1]
    #                 old_prob_train = old_prob[:, -1]
    #                 advantage_train = advantage[:, -1].squeeze(-1)  # (B, S, 1) -> (B,)
    #
    #                 entropy_cont_train = entropy_cont[:, -1]
    #                 entropy_disc_train = entropy_disc[:, -1]
    #
    #                 new_value_train = new_value[:, -1, :]  # (B, 1)
    #                 return_train = return_[:, -1, :]  # (B, 1)
    #             else:
    #                 new_prob_train = new_prob
    #                 old_prob_train = old_prob
    #                 advantage_train = advantage.squeeze(-1)
    #                 entropy_cont_train = entropy_cont
    #                 entropy_disc_train = entropy_disc
    #                 new_value_train = new_value
    #                 return_train = return_
    #             # ==========================================================================
    #
    #             # 4. 计算最终的 Loss
    #             total_entropy = (entropy_cont_train.mean() + entropy_disc_train.mean())
    #
    #             log_ratio = new_prob_train - old_prob_train
    #             ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
    #
    #             surr1 = ratio * advantage_train
    #             surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_train
    #             actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy
    #
    #             # 更新 Actor
    #             self.Actor.optim.zero_grad()
    #             actor_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
    #             self.Actor.optim.step()
    #
    #             # Critic Loss
    #             critic_loss = torch.nn.functional.mse_loss(new_value_train, return_train)
    #             self.Critic.optim.zero_grad()
    #             critic_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
    #             self.Critic.optim.step()
    #
    #             # 记录信息
    #             train_info['critic_loss'].append(critic_loss.item())
    #             train_info['actor_loss'].append(actor_loss.item())
    #             train_info['dist_entropy'].append(total_entropy.item())
    #             train_info['entropy_cont'].append(entropy_cont_train.mean().item())
    #             train_info['adv_targ'].append(advantage_train.mean().item())
    #             train_info['ratio'].append(ratio.mean().item())
    #
    #     if not train_info['critic_loss']:
    #         print("  [Warning] No batches were generated for training.")
    #         self.buffer.clear_memory()
    #         return None
    #
    #     # 4. 清理与保存
    #     self.buffer.clear_memory()
    #     for key in train_info:
    #         train_info[key] = np.mean(train_info[key])
    #
    #     self.save()
    #     return train_info

    def learn(self, next_visual_value=0.0):
        """
        执行 PPO 的学习和更新步骤。
        集成特性：
        1. Post-Attention GRU 架构
        2. 全局优势归一化 (Global Advantage Normalization)
        3. 雅可比修正 (Jacobian Correction)
        4. 维度对齐防错
        """
        # 如果 Buffer 中的数据不足一个批次，则跳过学习
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            return None

        # 1. 提取所有数据 (此时都在 CPU 上，为 Numpy 数组)
        states, values, actions, old_probs, rewards, dones, _, _, attn_weights = self.buffer.get_all_data()

        # 2. 计算 GAE 优势 (使用传入的 next_visual_value 处理截断)
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones, next_value=next_visual_value)

        # ================= [关键修改：全局优势归一化 & 维度对齐] =================

        # 1. 维度强制对齐 (N,) -> (N, 1)
        # 防止 (N,) + (N, 1) 导致生成 (N, N) 的巨大矩阵
        values = np.squeeze(values)  # 确保是 (N,)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if advantages.ndim == 1:
            advantages = advantages.reshape(-1, 1)

        # 2. 计算 Critic 的目标 Returns (必须使用未归一化的原始数据)
        # Return = Advantage_raw + Value_old
        returns = advantages + values

        # 3. 对 Advantage 进行全局归一化 (用于 Actor 更新)
        # 基于整个 buffer 的统计数据进行归一化，比 mini-batch 归一化更稳定
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # =======================================================================

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

        # 3. PPO 更新循环
        for _ in range(self.ppo_epoch):
            # 根据是否使用 RNN，选择不同的批次生成器
            if self.use_rnn:
                # [GRU模式]
                # 将处理好的全局 advantages 和 returns 传入生成器
                # 生成器内部会根据序列切片提取对应的片段
                batch_generator = self.buffer.generate_sequence_batches(
                    self.rnn_seq_len, self.rnn_batch_size, advantages, returns
                )
            else:
                # [MLP模式]
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                if self.use_rnn:
                    # [GRU模式] 解包数据 (注意：return_ 和 advantage 已经是处理过的了)
                    # 这里的解包需要根据你 Buffer 的具体实现来确定
                    # 假设 Buffer 返回的是 (s, a, p, adv, ret, v, h_a, h_c, mask)
                    (b_s, b_a, b_p, b_adv, b_ret, b_v, b_h_a, b_h_c, _) = batch_data

                    # 转 Tensor
                    state = torch.FloatTensor(b_s).to(**ACTOR_PARA.tpdv)
                    action_batch = torch.FloatTensor(b_a).to(**ACTOR_PARA.tpdv)
                    old_prob = torch.FloatTensor(b_p).to(**ACTOR_PARA.tpdv).view(-1)

                    # 这里的 b_adv 和 b_ret 已经是归一化后的 advantage 和原始 return 了
                    advantage = torch.FloatTensor(b_adv).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = torch.FloatTensor(b_ret).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    # old_value 用于 Value Clipping (可选)
                    old_value = torch.FloatTensor(b_v).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    rnn_h_a = torch.FloatTensor(b_h_a).to(**ACTOR_PARA.tpdv)
                    rnn_h_c = torch.FloatTensor(b_h_c).to(**CRITIC_PARA.tpdv)

                    # 前向传播
                    new_dists, _, _ = self.Actor(state, rnn_h_a)
                    new_value, _ = self.Critic(state, rnn_h_c)

                    # 维度调整
                    new_value = new_value.view(-1, 1)
                    action_batch = action_batch.view(-1, action_batch.shape[-1])

                else:
                    # [MLP模式]
                    batch_indices = batch_data
                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv)

                    # 直接提取已经全局归一化过的 Advantage
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    # 直接提取预计算好的 Return
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    old_value = check(values[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    new_dists, _, _ = self.Actor(state)
                    new_value, _ = self.Critic(state)

                # 解析动作
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # ================= [雅可比修正与 LogProb 计算] =================

                # --- A. 连续动作 Log Prob ---
                # 1. 计算旧动作在高斯分布下的 log_prob
                log_prob_u_buffer = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)

                # 2. 计算雅可比修正项
                correction_buffer = 2.0 * (np.log(2.0) - u_from_buffer - F.softplus(-2.0 * u_from_buffer)).sum(dim=-1)

                # 3. 得到最终 Log Prob (用于 Ratio)
                new_log_prob_cont = log_prob_u_buffer - correction_buffer

                # --- B. 熵计算 (使用重采样技巧) ---
                # 1. 基础高斯熵
                entropy_base = new_dists['continuous'].entropy().sum(dim=-1)

                # 2. 重采样当前策略动作
                u_curr_sample = new_dists['continuous'].rsample()

                # 3. 计算修正期望
                correction_curr = 2.0 * (np.log(2.0) - u_curr_sample - F.softplus(-2.0 * u_curr_sample)).sum(dim=-1)

                # 4. 得到最终熵
                entropy_cont = entropy_base + correction_curr

                # --- C. 离散动作 Log Prob ---
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )

                # 合并 Log Prob
                new_prob = new_log_prob_cont + new_log_prob_disc
                # ==========================================================

                # 计算总熵
                entropy_disc = sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )

                total_entropy = (entropy_cont.mean() + entropy_disc.mean())

                # 计算 Ratio
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                # 计算 Actor Loss (使用归一化后的 advantage)
                if advantage.dim() > ratio.dim():
                    advantage_squeezed = advantage.squeeze(-1)
                else:
                    advantage_squeezed = advantage

                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 更新 Actor
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                # Critic Loss (使用原始 return)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

                # 更新 Critic
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # 记录信息
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

        # 4. 清理与保存
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