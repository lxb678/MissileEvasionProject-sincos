# 文件名: Hybrid_PPO_CrossAttentionMLP实体.py (优化版, 交叉注意力架构)
# 描述: 此版本根据优化思路，将全自注意力模型修改为交叉注意力模型。
#      现在，模型只关注最核心的关系：我方飞机（作为Query）如何关注所有威胁导弹（作为Key/Value）。
#      这种架构更符合任务逻辑，计算效率更高，且能更直接地学习规避策略。
#      注意力权重的输出也变得更直接，与奖励系统完美兼容。

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
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.BufferAttn实体 import Buffer
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
# <<< 核心修改 1: 更新特征维度 >>>
MISSILE_FEAT_DIM = 4 #5  # 原为 3. (1 for dist + 2 for beta_sincos + 2 for theta_sincos)
AIRCRAFT_FEAT_DIM = 7 #8 # 原为 6. (1 av + 1 h + 2 ae_sincos + 2 am_sincos + 1 ir + 1 q)
# <<< 修改结束 >>>
ENTITY_EMBED_DIM = 64 #64 #32 #64 #64 #128
ATTN_NUM_HEADS = 4 #2 #4 #4

assert ENTITY_EMBED_DIM % ATTN_NUM_HEADS == 0, "ENTITY_EMBED_DIM must be divisible by ATTN_NUM_HEADS"


# ==============================================================================
#           <<< 核心修改 >>>: 升级架构为交叉注意力
# ==============================================================================

class Actor_CrossAttentionMLP(Module):
    """
    Actor 网络 - [优化版：交叉注意力 + 完整的混合MLP架构]
    结构为: 实体编码器 -> 交叉注意力(飞机Q, 导弹K/V) -> 提取飞机上下文 -> 共享MLP基座 -> 专用MLP塔楼 -> 独立动作头。
    """

    def __init__(self, weight_decay=1e-4):
        super(Actor_CrossAttentionMLP, self).__init__()
        # --- MLP 和 Head 部分的定义与之前完全相同，无需修改 ---
        self.input_dim = ACTOR_PARA.input_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.weight_decay = weight_decay

        # self.missile_encoder = Sequential(
        #     Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
        #     # GELU()
        #     # Tanh()
        #     LeakyReLU()  # <-- 添加非线性激活
        # )
        # self.aircraft_encoder = Sequential(
        #     Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM),
        #     # GELU()
        #     # Tanh()
        #     LeakyReLU()  # <-- 添加非线性激活
        # )

        encoder_hidden_dim = ENTITY_EMBED_DIM #// 2  # 例如 128

        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, encoder_hidden_dim),
            LeakyReLU(),
            Linear(encoder_hidden_dim, ENTITY_EMBED_DIM),
            # LeakyReLU()
            # Tanh()
        )
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, encoder_hidden_dim),
            LeakyReLU(),
            Linear(encoder_hidden_dim, ENTITY_EMBED_DIM),
            # LeakyReLU()
            # Tanh()
        )

        # self.ln_missile = LayerNorm(ENTITY_EMBED_DIM)
        # self.ln_aircraft = LayerNorm(ENTITY_EMBED_DIM)

        self.attention = MultiheadAttention(embed_dim=ENTITY_EMBED_DIM,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)
        # self.attn_layer_norm = LayerNorm(ENTITY_EMBED_DIM)
        # # ✅ Pre-Norm: 归一化在注意力之前
        # self.attn_pre_norm = LayerNorm(ENTITY_EMBED_DIM)

        # <<< 核心修改 1: 更新MLP的输入维度 >>>
        # 新的输入 = aircraft_context (Embed) + missile1_embed (Embed) + missile2_embed (Embed)
        mlp_input_dim = ENTITY_EMBED_DIM * 2 #* (1 + NUM_MISSILES)  # 64 * 4 = 256

        split_point = 2
        mlp_dims = ACTOR_PARA.model_layer_dim
        base_dims = mlp_dims[:split_point]
        tower_dims = mlp_dims[split_point:]

        self.shared_base_mlp = Sequential()
        base_input_dim = mlp_input_dim #ENTITY_EMBED_DIM
        for i, dim in enumerate(base_dims):
            self.shared_base_mlp.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
            self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', LeakyReLU())
            base_input_dim = dim
        base_output_dim = base_dims[-1] if base_dims else ENTITY_EMBED_DIM

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

        # --- 优化器设置部分也保持不变 ---
        attention_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if any(key in name.lower() for key in ['attention', 'attn', 'layer_norm']):
                attention_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {'params': attention_params, 'lr': ACTOR_PARA.attention_lr},
            {'params': other_params, 'lr': ACTOR_PARA.lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)
        # self._init_weights()  # <--- 添加这行

    def _init_weights(self):
        # 遍历所有模块进行初始化
        for m in self.modules():
            if isinstance(m, Linear):
                # 正交初始化，增益为 sqrt(2) 适合 LeakyReLU/Tanh
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- 特殊处理：策略输出头 ---
        # 让策略输出层初始权重非常小（几乎为0），
        # 这样初始动作概率分布会接近均匀分布（高熵），鼓励初期探索。
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)

        nn.init.orthogonal_(self.discrete_head.weight, gain=0.01)
        nn.init.constant_(self.discrete_head.bias, 0)

    def forward(self, obs):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() > 2:
            raise ValueError(
                f"AttentionMLP model expects non-sequential input (Batch, Dim), but got shape {obs_tensor.shape}")

        # 1. 编码所有实体
        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # # <<< 新增：从观测数据中推断掩码 >>>
        # # 定义无效导弹的“指纹”观测值
        # inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 0.5], device=obs_tensor.device)

        # <<< 核心修改 2: 更新无效导弹的“指纹” >>>
        # 新指纹: [dist=1.0, sin(beta=0)=0, cos(beta=0)=1, sin(theta=0)=0, cos(theta=0)=1]
        # inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device=obs_tensor.device)
        # <<< 修改结束 >>>

        # <<< 核心修正: 更新无效导弹的“指纹”以匹配环境 >>>
        # 新环境的无效指纹: [dist_norm=1.0, sin(beta)=0.0, cos(beta)=1.0, theta_norm=0.0]
        inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        # <<< 修正结束 >>>

        # 检查每个导弹的观测值是否与指纹匹配
        # torch.all(tensor, dim=-1) 会在特征维度上进行比较
        is_m1_inactive = torch.all(torch.isclose(missile1_obs, inactive_missile_fingerprint), dim=-1)
        is_m2_inactive = torch.all(torch.isclose(missile2_obs, inactive_missile_fingerprint), dim=-1)

        # 构建注意力掩码
        # 形状: (Batch, Num_Missiles) -> (B, 2)
        # PyTorch的key_padding_mask要求：True代表“需要被掩盖/忽略”
        attention_mask = torch.stack([is_m1_inactive, is_m2_inactive], dim=1)

        m1_embed = self.missile_encoder(missile1_obs)
        m2_embed = self.missile_encoder(missile2_obs)
        ac_embed = self.aircraft_encoder(aircraft_obs)

        # # 2. <<< 核心修改: 添加 Pre-LayerNorm >>>
        # # 在进入注意力计算前，先进行归一化。这能防止 Q*K 点积过大导致梯度消失。
        # m1_embed = self.ln_missile(m1_embed_raw)
        # m2_embed = self.ln_missile(m2_embed_raw)
        # ac_embed = self.ln_aircraft(ac_embed_raw)

        # # 3. ✅ Pre-Norm: 先归一化飞机表示
        # normalized_aircraft = self.attn_pre_norm(ac_embed)
        #
        # # 4. 交叉注意力（使用归一化后的查询）
        # aircraft_query = normalized_aircraft.unsqueeze(1)

        # <<< 核心修改: 从全自注意力改为交叉注意力 >>>
        # 2. 准备 Query, Key, Value
        # Query: 我方飞机，代表决策主体。增加一个序列维度 (B, Embed) -> (B, 1, Embed)
        aircraft_query = ac_embed.unsqueeze(1)
        # Key/Value: 两个导弹，代表需要被关注的威胁源。(B, Embed) -> (B, 2, Embed)
        missile_entities = torch.stack([m1_embed, m2_embed], dim=1)

        # Value 使用 Raw (关键修改！！！)
        # 这样注意力输出的特征就保留了原始的物理强度信息
        missile_values_raw = torch.stack([m1_embed, m2_embed], dim=1)

        # 3. 执行交叉注意力计算
        # 飞机(Q) 关注 导弹(K,V)，得到融合了导弹信息的飞机上下文
        # attn_output shape: (B, 1, Embed), attn_weights_raw shape: (B, NumHeads, 1, 2)  # attn_weights_raw shape: (B, 1, 2) - 已经平均过heads
        # attn_output, attn_weights_raw = self.attention(query=aircraft_query,
        #                                                key=missile_entities,
        #                                                value=missile_entities)

        # 4. <<< 修改：执行带掩码的交叉注意力计算 >>>
        attn_output, attn_weights_raw = self.attention(query=aircraft_query,
                                                       key=missile_entities,
                                                       value=missile_values_raw, #missile_entities,
                                                       key_padding_mask=attention_mask
                                                       )  # <<< 新增参数

        # <<< 关键修正 1: NaN 保护 >>>
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0)

        # # 4. 残差连接与层归一化
        # # 首先移除 attn_output 多余的序列维度 (B, 1, Embed) -> (B, Embed)
        # attn_output_squeezed = attn_output.squeeze(1)
        # # 将注意力的输出与原始飞机信息结合，防止信息丢失
        # aircraft_context = self.attn_layer_norm(attn_output_squeezed + ac_embed)
        # # <<< 修改结束 >>>
        # # # 5. ✅ 残差连接（使用原始的 ac_embed）
        # # aircraft_context = ac_embed + attn_output_squeezed

        # <<< 核心修改 2: 拼接所有信息作为MLP的输入 >>>
        # 将注意力处理后的飞机上下文 和 原始的导弹信息 全部拼接起来
        # mlp_input = torch.cat([aircraft_context, m1_embed, m2_embed], dim=-1)
        # mlp_input = torch.cat([aircraft_context, ac_embed, m1_embed, m2_embed], dim=-1)

        # 后续的MLP部分，现在使用包含所有信息的 mlp_input
        # base_features = self.shared_base_mlp(mlp_input)

        # --- 修改前 ---
        # attn_output_squeezed = attn_output.squeeze(1)
        # aircraft_context = self.attn_layer_norm(attn_output_squeezed + ac_embed)
        # base_features = self.shared_base_mlp(aircraft_context)

        # --- 修改后建议 ---
        attn_output_squeezed = attn_output.squeeze(1)  # 这是"威胁总览"
        # attn_output_squeezed = self.attn_layer_norm(attn_output_squeezed)

        # # 拼接：[原本的飞机状态, 注意力提取的威胁状态]
        # # 这样 MLP 既知道自己怎么飞(ac_embed)，也知道威胁多严重(attn_output_squeezed)
        # combined_features = torch.cat([ac_embed, attn_output_squeezed], dim=-1)

        # 5. 特征拼接
        # 推荐策略：拼接 [原始未归一化的飞机特征, 归一化且经过注意力的威胁特征]
        # 这样 MLP 既保留了原始幅值信息（可能包含距离绝对值等重要信息），又利用了稳定的注意力特征。
        combined_features = torch.cat([ac_embed, attn_output_squeezed], dim=-1)

        # 注意：需要调整 shared_base_mlp 的输入维度
        # mlp_input_dim = ENTITY_EMBED_DIM * 2
        base_features = self.shared_base_mlp(combined_features)

        # 后续的MLP部分，使用融合了导弹威胁信息的 aircraft_context 作为输入
        # base_features = self.shared_base_mlp(aircraft_context)
        continuous_features = self.continuous_tower(base_features)
        discrete_features = self.discrete_tower(base_features)
        mu = self.mu_head(continuous_features)
        all_disc_logits = self.discrete_head(discrete_features)

        # --- 动作分布和掩码逻辑与之前完全相同，无需修改 ---
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        # # <<< 核心修改：更新诱饵弹信息的索引 >>>
        # # 根据新的观测空间定义 (2*5 + 6)，o_ir_norm 现在是第17个元素，索引为16。
        # # 旧索引: 10
        # # 新索引: 2 * MISSILE_FEAT_DIM + 6
        # flare_info_index = 2 * MISSILE_FEAT_DIM + 6
        # has_flares_info = obs_tensor[..., flare_info_index]  # 原为 obs_tensor[..., 10]
        # # <<< 修改结束 >>>

        # <<< 修改 3: 更新诱饵弹信息的索引 >>>
        # 新的飞机状态向量: [av, h, ae, am_sin, am_cos, ir, q] (7个元素)
        # o_ir_norm 是其中的第6个元素, 索引为 5。
        # 绝对索引 = 导弹部分总维度 + 飞机内部索引 = (2 * MISSILE_FEAT_DIM) + 5
        flare_info_index = 2 * MISSILE_FEAT_DIM + 5  # 原为 +6
        has_flares_info = obs_tensor[..., flare_info_index]
        # <<< 修改结束 >>>

        # has_flares_info现在代表 o_ir_norm, 是一个 [0, 1] 范围的浮点数。
        # 当 o_ir_norm > 0 时表示有弹，o_ir_norm == 0 时表示无弹。
        # 因此，掩码逻辑 `mask = (has_flares_info == 0)` 依然是正确的。
        mask = (has_flares_info == 0)

        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        trigger_probs = torch.sigmoid(trigger_logits_masked)
        no_trigger_mask = (trigger_probs < 0.5)

        salvo_size_logits_masked = salvo_size_logits.clone()
        num_groups_logits_masked = num_groups_logits.clone()
        inter_interval_logits_masked = inter_interval_logits.clone()

        NEG_INF = -1e8
        forced_salvo_logits = torch.full_like(salvo_size_logits_masked, NEG_INF)
        forced_salvo_logits[..., 0] = 1.0
        salvo_size_logits_masked = torch.where(no_trigger_mask, forced_salvo_logits, salvo_size_logits_masked)
        forced_groups_logits = torch.full_like(num_groups_logits_masked, NEG_INF)
        forced_groups_logits[..., 0] = 1.0
        num_groups_logits_masked = torch.where(no_trigger_mask, forced_groups_logits, num_groups_logits_masked)
        forced_interval_logits = torch.full_like(inter_interval_logits_masked, NEG_INF)
        forced_interval_logits[..., 0] = 1.0
        inter_interval_logits_masked = torch.where(no_trigger_mask, forced_interval_logits,
                                                   inter_interval_logits_masked)

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

        # <<< 核心修改 >>>: 直接返回飞机对两个导弹的注意力权重
        # attn_weights_raw shape: (B, NumHeads, 1, 2)
        # 平均掉 head 维度 -> (B, 1, 2)
        # 再去掉 query 维度 -> (B, 2)
        # 这个张量直接代表了飞机对导弹1和导弹2的关注度，可用于奖励函数
        # attention_to_missiles = attn_weights_raw.mean(dim=1).squeeze(1)
        # 推荐的修改 (更清晰、更标准)
        # .squeeze(1) 直接将 (Batch, 1, 2) 变为 (Batch, 2)
        attention_to_missiles = attn_weights_raw.squeeze(1)

        return distributions, attention_to_missiles


class Critic_CrossAttentionMLP(Module):
    """
    Critic 网络 - [优化版：交叉注意力]
    与 Actor 使用相同的交叉注意力机制来评估状态值。
    """

    def __init__(self, weight_decay=1e-4):
        super(Critic_CrossAttentionMLP, self).__init__()
        # --- MLP 和 Head 部分的定义与之前完全相同，无需修改 ---
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.weight_decay = weight_decay

        # self.missile_encoder = Sequential(
        #     Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM),
        #     # GELU()
        #     # Tanh()
        #    LeakyReLU()  # <-- 添加非线性激活
        # )
        # self.aircraft_encoder = Sequential(
        #     Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM),
        #     # GELU()
        #     # Tanh()
        #    LeakyReLU()  # <-- 添加非线性激活
        # )

        encoder_hidden_dim = ENTITY_EMBED_DIM #// 2 #* 2  # 例如 32

        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, encoder_hidden_dim),
            LeakyReLU(),
            Linear(encoder_hidden_dim, ENTITY_EMBED_DIM),
            # Tanh()
        )
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, encoder_hidden_dim),
            LeakyReLU(),
            Linear(encoder_hidden_dim, ENTITY_EMBED_DIM),
            # Tanh()
        )

        # self.ln_missile = LayerNorm(ENTITY_EMBED_DIM)
        # self.ln_aircraft = LayerNorm(ENTITY_EMBED_DIM)

        self.attention = MultiheadAttention(embed_dim=ENTITY_EMBED_DIM,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)
        # self.attn_layer_norm = LayerNorm(ENTITY_EMBED_DIM)
        # # ✅ Pre-Norm: 归一化在注意力之前
        # self.attn_pre_norm = LayerNorm(ENTITY_EMBED_DIM)

        # <<< 核心修改 1: 更新MLP的输入维度 >>>
        mlp_input_dim = ENTITY_EMBED_DIM * 2   #* (1 + NUM_MISSILES)  # 192

        # mlp_input_dim = ENTITY_EMBED_DIM
        mlp_dims = CRITIC_PARA.model_layer_dim

        self.post_attention_mlp = Sequential()
        tower_input_dim = mlp_input_dim
        for i, dim in enumerate(mlp_dims):
            self.post_attention_mlp.add_module(f'mlp_fc_{i}', Linear(tower_input_dim, dim))
            self.post_attention_mlp.add_module(f'mlp_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim
        mlp_output_dim = mlp_dims[-1] if mlp_dims else mlp_input_dim

        self.fc_out = Linear(mlp_output_dim, self.output_dim)

        # --- 优化器设置部分也保持不变 ---
        attention_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if any(key in name.lower() for key in ['attention', 'attn', 'layer_norm']):
                attention_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': attention_params, 'lr': CRITIC_PARA.attention_lr},
            {'params': other_params, 'lr': CRITIC_PARA.lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)
        # self._init_weights()  # <--- 添加这行

    def _init_weights(self):
        # 遍历所有模块进行初始化
        for m in self.modules():
            if isinstance(m, Linear):
                # 正交初始化，增益为 sqrt(2) 适合 LeakyReLU/Tanh
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- Critic 特有：价值头初始化 ---
        # Critic 输出的是价值，不需要像 Actor 那样缩小权重
        # 这里的层名是 self.fc_out
        torch.nn.init.orthogonal_(self.fc_out.weight, gain=1.0)
        torch.nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, obs):
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        if obs_tensor.dim() > 2:
            raise ValueError(
                f"AttentionMLP model expects non-sequential input (Batch, Dim), but got shape {obs_tensor.shape}")

        missile1_obs = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # # <<< 新增：从观测数据中推断掩码 >>>
        # # 定义无效导弹的“指纹”观测值
        # inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 0.5], device=obs_tensor.device)

        # # <<< 核心修改 2: 更新无效导弹的“指纹” >>>
        # # 新指纹: [dist=1.0, sin(beta=0)=0, cos(beta=0)=1, sin(theta=0)=0, cos(theta=0)=1]
        # inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device=obs_tensor.device)
        # # <<< 修改结束 >>>

        # <<< 核心修正: 更新无效导弹的“指纹”以匹配环境 >>>
        # 新环境的无效指纹: [dist_norm=1.0, sin(beta)=0.0, cos(beta)=1.0, theta_norm=0.0]
        inactive_missile_fingerprint = torch.tensor([1.0, 0.0, 1.0, 0.0], device=obs_tensor.device)
        # <<< 修正结束 >>>

        # 检查每个导弹的观测值是否与指纹匹配
        is_m1_inactive = torch.all(torch.isclose(missile1_obs, inactive_missile_fingerprint), dim=-1)
        is_m2_inactive = torch.all(torch.isclose(missile2_obs, inactive_missile_fingerprint), dim=-1)

        # 构建注意力掩码
        attention_mask = torch.stack([is_m1_inactive, is_m2_inactive], dim=1)

        m1_embed = self.missile_encoder(missile1_obs)
        m2_embed = self.missile_encoder(missile2_obs)
        ac_embed = self.aircraft_encoder(aircraft_obs)

        # # 2. <<< 核心修改: 添加 Pre-LayerNorm >>>
        # # 在进入注意力计算前，先进行归一化。这能防止 Q*K 点积过大导致梯度消失。
        # m1_embed = self.ln_missile(m1_embed_raw)
        # m2_embed = self.ln_missile(m2_embed_raw)
        # ac_embed = self.ln_aircraft(ac_embed_raw)

        # # 3. ✅ Pre-Norm: 先归一化飞机表示
        # normalized_aircraft = self.attn_pre_norm(ac_embed)
        #
        # # 4. 交叉注意力（使用归一化后的查询）
        # aircraft_query = normalized_aircraft.unsqueeze(1)

        # <<< 核心修改: 交叉注意力 (与 Actor 相同) >>>
        aircraft_query = ac_embed.unsqueeze(1)
        missile_entities = torch.stack([m1_embed, m2_embed], dim=1)

        # Value 使用 Raw (关键修改！！！)
        # 这样注意力输出的特征就保留了原始的物理强度信息
        missile_values_raw = torch.stack([m1_embed, m2_embed], dim=1)

        # attn_output, _ = self.attention(query=aircraft_query,
        #                                 key=missile_entities,
        #                                 value=missile_entities)

        # 4. <<< 修改：执行带掩码的交叉注意力计算 >>>
        attn_output, _ = self.attention(query=aircraft_query,
                                        key=missile_entities,
                                        value=missile_values_raw, #missile_entities,
                                        key_padding_mask=attention_mask
                                        )  # <<< 新增参数

        # <<< 关键修正 1: NaN 保护 >>>
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0)

        # attn_output_squeezed = attn_output.squeeze(1)
        # # 5. ✅ 残差连接（使用原始的 ac_embed）
        # # aircraft_context = ac_embed + attn_output_squeezed
        # aircraft_context = self.attn_layer_norm(attn_output_squeezed + ac_embed)
        # # <<< 修改结束 >>>

        # <<< 核心修改 2: 拼接所有信息作为MLP的输入 >>>
        # mlp_input = torch.cat([aircraft_context, m1_embed, m2_embed], dim=-1)
        # mlp_input = torch.cat([aircraft_context, ac_embed, m1_embed, m2_embed], dim=-1)

        # 后续的MLP部分，现在使用包含所有信息的 mlp_input
        # mlp_features = self.post_attention_mlp(mlp_input)

        # --- 修改前 ---
        # attn_output_squeezed = attn_output.squeeze(1)
        # aircraft_context = self.attn_layer_norm(attn_output_squeezed + ac_embed)
        # base_features = self.shared_base_mlp(aircraft_context)

        # --- 修改后建议 ---
        attn_output_squeezed = attn_output.squeeze(1)  # 这是"威胁总览"
        # attn_output_squeezed = self.attn_layer_norm(attn_output_squeezed)

        # 拼接：[原本的飞机状态, 注意力提取的威胁状态]
        # 这样 MLP 既知道自己怎么飞(ac_embed)，也知道威胁多严重(attn_output_squeezed)
        combined_features = torch.cat([ac_embed, attn_output_squeezed], dim=-1)

        # 注意：需要调整 shared_base_mlp 的输入维度
        # mlp_input_dim = ENTITY_EMBED_DIM * 2
        mlp_features = self.post_attention_mlp(combined_features)

        # mlp_features = self.post_attention_mlp(aircraft_context)
        value = self.fc_out(mlp_features)

        return value


class PPO_continuous(object):
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = False):
        super(PPO_continuous, self).__init__()

        self.use_rnn = False
        if use_rnn:
            print(
                "[Warning] use_rnn=True was passed, but this PPO version is Attention+MLP only. Forcing use_rnn=False.")

        # <<< 修改 >>>: 初始化新的交叉注意力模型
        print("--- 初始化 PPO Agent (使用 [交叉注意力 -> MLP] 优化模型) ---")
        self.Actor = Actor_CrossAttentionMLP()
        self.Critic = Critic_CrossAttentionMLP()
        # <<< 修改结束 >>>

        self.buffer = Buffer(use_rnn=False, use_attn=True)
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.training_start_time = time.strftime("PPO_EntityCrossATT_MLP_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../save/save_evade_fuza两个导弹"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)
        if load_able:
            if model_dir_path:
                self.load_models_from_directory(model_dir_path)
            else:
                self.load_models_from_directory("../../../../test/test_evade")

    # --- 模型加载、动作缩放、存储经验等方法保持不变 ---
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
        self.buffer.store_transition(state, value, action, probs, reward, done, attn_weights=attn_weights)

    def choose_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.Critic(state_tensor)

            # <<< 核心修改: Actor 现在直接返回所需的注意力权重 >>>
            # attention_weights shape: (Batch, 2)
            dists, attention_weights = self.Actor(state_tensor)

            # <<< 关键适配步骤: 代码变得更简洁 >>>
            # 无需再从 (B, 3, 3) 矩阵中提取，因为 Actor 的输出就是我们想要的
            attention_weights_for_reward = attention_weights
            # <<< 修改结束 >>>

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
                        # print(dist.probs)
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                        # sampled_actions_dict[key] = dist.sample()
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

        return final_env_action_np, action_to_store_np, log_prob_to_store_np, value_np, attention_weights_np

    # --- GAE计算和学习过程的核心逻辑保持不变 ---
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
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            return None

        states, values, actions, old_probs, rewards, dones, _, _, attn_weights = self.buffer.get_all_data()

        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        values = np.squeeze(values)
        returns = advantages + values
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

        for _ in range(self.ppo_epoch):
            batch_generator = self.buffer.generate_batches()

            for batch_indices in batch_generator:
                state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv)
                advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv)
                return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv)

                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # 注意：_ (下划线) 接收了注意力权重，但在训练循环中我们不直接使用它。
                new_dists, _ = self.Actor(state)

                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc

                entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                total_entropy = (entropy_cont + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()

                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                advantage = advantage.view(-1, 1)
                return_ = return_.view(-1, 1)

                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=0.5)
                self.Actor.optim.step()

                new_value = self.Critic(state)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)
                self.Critic.optim.step()

                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['entropy_cont'].append(entropy_cont.mean().item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())

        if not train_info['critic_loss']:
            print("  [Warning] No batches were generated for training. Skipping metrics calculation.")
            self.buffer.clear_memory()
            return None

        self.buffer.clear_memory()
        for key in train_info:
            train_info[key] = np.mean(train_info[key])

        self.save()
        return train_info

    # --- 训练/评估模式切换和保存方法保持不变 ---
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