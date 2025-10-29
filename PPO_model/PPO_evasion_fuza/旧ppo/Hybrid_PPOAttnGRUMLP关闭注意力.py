# --- START OF FILE Hybrid_PPO_jsbsim_SeparateHeads.py ---

import torch
from torch.nn import *
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
# 导入配置文件，其中包含各种超参数
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigAttn import *
# 导入支持 GRU 的经验回放池
from Interference_code.PPO_model.PPO_evasion_fuza.BufferGRUAttn import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time
# 导入 PyTorch 的自动混合精度训练工具，用于加速训练并减少显存占用
from torch.cuda.amp import GradScaler, autocast

# --- 动作空间配置 (与原版相同) ---
# 定义连续动作的维度和键名
CONTINUOUS_DIM = 4
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']
# 定义离散动作的维度。每个键代表一个离散决策，值代表该决策有多少个选项。
DISCRETE_DIMS = {
    'flare_trigger': 1,  # 干扰弹触发，伯努利分布 (是/否)，所以是1个 logit
    'salvo_size': 3,  # 齐射数量，3个选项
    'intra_interval': 3,  # 组内间隔，3个选项
    'num_groups': 3,  # 组数，3个选项
    'inter_interval': 3,  # 组间间隔，3个选项
}
# 计算所有离散动作 logits 的总维度
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())
# 计算存储在 Buffer 中的总动作维度（连续动作 + 离散动作的类别索引）
TOTAL_ACTION_DIM_BUFFER = CONTINUOUS_DIM + len(DISCRETE_DIMS)
# 离散动作的类别索引到实际物理值的映射
DISCRETE_ACTION_MAP = {
    # 'salvo_size': [2, 4, 6],
    # 'intra_interval': [0.02, 0.04, 0.1],
    # 'num_groups': [1, 2, 3],
    # 'inter_interval': [0.5, 1.0, 2.0]
    'salvo_size': [1, 2, 3],  # 修改为发射1、2、3枚
    # 'intra_interval': [0.05, 0.1, 0.15],
    'intra_interval': [0.02, 0.04, 0.08],
    'num_groups': [1, 2, 3],
    'inter_interval': [0.2, 0.5, 1.0]
}
# 连续动作的物理范围，用于将网络输出 (-1, 1) 缩放到实际范围
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},
    'elevator': {'low': -1.0, 'high': 1.0},
    'aileron': {'low': -1.0, 'high': 1.0},
    'rudder': {'low': -1.0, 'high': 1.0},
}

# <<< GRU/RNN/Attention 修改 >>>: 新增 RNN 和 Attention 配置
RNN_HIDDEN_SIZE = 256  # GRU 层的隐藏单元数量
SEQUENCE_LENGTH = 10  # 训练时从经验池中采样的连续轨迹片段的长度
# ATTN_NUM_HEADS = 8     # 注意力机制的头数 (必须能被 MLP 输出维度整除)
ATTN_NUM_HEADS = 2 #3 #4 #8 #4 #1 #2       # <<< 这是您的关键修改

# ==============================================================================
# Original MLP-based Actor and Critic (保留原始版本以供选择)
# ==============================================================================

class Actor(Module):
    # ... 原版 Actor 代码保持不变 ...
    """
       Actor 网络 (策略网络) - 基于 MLP（多层感知机）的版本。
       该网络负责根据当前状态决定要采取的动作策略。
       它具有一个共享的骨干网络，后接两个独立的头部，分别处理连续动作和离散动作。
       """

    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # <<< 更改 >>> 输出维度现在是 (连续*2) + 新的logits总数
        # 注意：这里不再需要一个总的 output_dim
        # self.output_dim = (CONTINUOUS_DIM * 2) + TOTAL_DISCRETE_LOGITS
        self.log_std_min = -20.0 # 限制对数标准差的最小值，防止数值不稳定
        self.log_std_max = 2.0 # 限制对数标准差的最大值

        # 定义共享骨干网络
        # 负责从原始状态中提取高级特征
        shared_layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim
        self.shared_network = Sequential()
        for i in range(len(shared_layers_dims) - 1):
            # 添加线性（全连接）层
            self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
            # --- 在此处添加 LayerNorm ---
            # LayerNorm 的输入维度是前一个线性层的输出维度
            # LayerNorm 对每个样本的特征进行归一化，有助于稳定训练
            self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
            # 添加 LeakyReLU 激活函数，以避免梯度消失问题
            self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # 骨干网络的输出维度，将作为各个头部的输入
        shared_output_dim = ACTOR_PARA.model_layer_dim[-1]
        # --- 独立头部网络 ---
        # 1. 连续动作头部：输出高斯分布的均值 (mu) 和对数标准差 (log_std)，每个连续动作维度都需要这两个参数
        self.continuous_head = Linear(shared_output_dim, CONTINUOUS_DIM * 2)

        # 2. 离散动作头部：输出所有离散决策所需的 logits（未经 Softmax 的原始分数）
        self.discrete_head = Linear(shared_output_dim, TOTAL_DISCRETE_LOGITS)

        # self.init_model() # (被注释掉，因为网络结构在 init 中直接定义)
        # --- 优化器和设备设置 ---
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,# 初始学习率因子
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,# 最终学习率因子
            total_iters=AGENTPARA.MAX_EXE_NUM # 达到最终学习率所需的总迭代次数
        )
        self.to(ACTOR_PARA.device) # 将模型移动到指定的设备 (CPU 或 GPU)

    def forward(self, obs):
        """
        前向传播方法，为每个动作维度创建并返回一个概率分布。
        """
        # 确保输入是 PyTorch 张量并移动到正确的设备
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        # 如果输入是一维的（单个状态），增加一个批次维度
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 通过共享骨干网络提取通用特征
        shared_features = self.shared_network(obs_tensor)

        # 2. 将共享特征分别送入不同的头部网络
        cont_params = self.continuous_head(shared_features) # 获取连续动作的参数
        all_disc_logits = self.discrete_head(shared_features) # 获取所有离散动作的 logits

        # ... 后续逻辑与原版完全相同 ...
        # 根据 DISCRETE_DIMS 的定义，将总的 logits 切分成对应每个离散动作的部分
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=1)
        trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
        # 获取状态中关于干扰弹数量的信息（索引为7）
        has_flares_info = obs_tensor[:, 7]
        # 创建一个掩码，当干扰弹数量为0时为 True
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        # 如果存在干扰弹数量为0的情况，将对应的 trigger_logits 设置为负无穷大
        # 这样在应用 sigmoid/softmax 后，触发概率会趋近于0，实现了动作屏蔽
        if torch.any(mask):
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min
        # 从连续动作参数中分离出均值和对数标准差
        mu, log_std = cont_params.chunk(2, dim=-1)
        # 裁剪对数标准差以保证数值稳定性
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        # 计算标准差
        std = torch.exp(log_std)
        # 创建连续动作的正态分布
        continuous_base_dist = Normal(mu, std)
        # 创建离散动作的分布
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1)) # 伯努利分布
        salvo_size_dist = Categorical(logits=salvo_size_logits) # 分类分布
        intra_interval_dist = Categorical(logits=intra_interval_logits)
        num_groups_dist = Categorical(logits=num_groups_logits)
        inter_interval_dist = Categorical(logits=inter_interval_logits)
        # 将所有分布打包成一个字典返回
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
       这是一个标准的 MLP 模型，负责预测输入状态的期望回报。
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
        """初始化 Critic 的网络结构。"""
        self.network = Sequential()
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            self.network.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
        # 输出层，输出一个标量值，即状态价值 V(s)
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

    def forward(self, obs):
        """前向传播，计算状态价值。"""
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value

# ==============================================================================
# <<< 修改 >>>: 定义新的基于 Attention + GRU 的 Actor 和 Critic
#               [💥 新结构: GRU -> Attention -> MLP -> Heads]
# ==============================================================================

def init_weights(m, gain=1.0):
    """
    一个通用的权重初始化函数。
    :param m: PyTorch module
    :param gain: 正交初始化的增益因子
    """
    if isinstance(m, Linear):
        # 对线性层使用 Kaiming Normal 初始化，适用于 LeakyReLU
        torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, GRU):
        # 对 GRU 的权重使用正交初始化
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'bias' in name:
                param.data.fill_(0)


# ==============================================================================
#           最终版本：特征级注意力 + GRU时序建模
# ==============================================================================
# 注意：此段代码假定你在文件顶部已经做了必要的 import，
# 并且保留了你原来代码里使用的名字（Linear, GRU, LayerNorm, Dropout, LeakyReLU, Sequential, Module, Normal, Bernoulli, Categorical 等）
# 以及 ACTOR_PARA, CRITIC_PARA, RNN_HIDDEN_SIZE, ATTN_NUM_HEADS, CONTINUOUS_DIM, TOTAL_DISCRETE_LOGITS, DISCRETE_DIMS, init_weights, check 等外部定义

class Actor_GRU(Module):
    """
    Actor 网络 (策略网络) - [特征级 Attention -> GRU -> MLP -> 动作头]

    该架构的核心思想是两阶段处理：
    1. 特征交互阶段（空间/特征维度）：在每个独立的时间步，将状态向量中的 D 个特征视为一个序列，
       使用自注意力机制计算这些特征之间的动态关系，从而生成一个上下文感知的状态表示。
    2. 时序建模阶段（时间维度）：将经过特征注意力处理后的一系列状态表示送入 GRU，
       以捕捉状态随时间变化的模式和趋势。
    """
    """
      Actor 网络 (策略网络) - [特征级 Attention -> GRU -> MLP -> 动作头]

      架构特点：
      - 特征交互：在每个时间步，将D个特征视为D个Token，使用自注意力计算特征间的动态关系。
      - 稳定性设计：采用 Pre-and-Post-LN 结构增强训练稳定性。
      - 精细化优化：对注意力相关参数使用独立的学习率和权重衰减。
      """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        super(Actor_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = ACTOR_PARA.input_dim  # D (原始状态特征的数量)
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        # 定义注意力机制和GRU的工作维度
        self.embedding_dim = 64 #128 #RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay  # 保存默认的weight_decay

        # --- 模块定义 ---

        # 1. 特征嵌入层 (Feature Embedding)
        # 作用：将每个独立的标量特征 (维度为1) 投影到一个高维的嵌入空间 (embedding_dim)。
        # 这是实现特征Token化的关键一步。
        self.feature_embed = Linear(1, self.embedding_dim)

        # 2. 特征级自注意力层 (Feature-wise Self-Attention)
        # 作用：在特征维度上进行交互，计算 D 个特征Token之间的相互关系。
        # embed_dim 是每个特征Token被投影后的维度。
        assert self.embedding_dim % ATTN_NUM_HEADS == 0, "embedding_dim must be divisible by num_heads"
        self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.1,
                                            batch_first=True)
        self.attn_dropout = Dropout(p=dropout_rate)
        # LayerNorm 在注意力计算之前应用，作用于每个特征Token的嵌入向量上。
        # self.attention_layernorm = LayerNorm(self.embedding_dim)
        # --- 定义两个LayerNorm ---
        self.attention_layernorm = LayerNorm(self.embedding_dim)  # 这个作为 Pre-LN
        self.post_layernorm = LayerNorm(self.embedding_dim)  # <<< ✅ 新增这个作为 Post-LN >>>

        # 3. GRU 时序建模层
        # 作用：处理由特征注意力模块输出的时间序列，捕捉时间依赖性。
        # 输入维度是 embedding_dim，因为池化后的向量维度与嵌入维度相同。
        self.gru = GRU(self.embedding_dim, self.rnn_hidden_size, batch_first=True)

        # 4. 共享MLP骨干网络
        # 作用：对GRU输出的时序特征进行进一步的非线性变换，提取更高级的决策特征。
        shared_layers_dims = [self.rnn_hidden_size] + ACTOR_PARA.model_layer_dim
        self.shared_network = Sequential()
        for i in range(len(shared_layers_dims) - 1):
            self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
            self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
            self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        mlp_output_dim = ACTOR_PARA.model_layer_dim[-1]

        # 5. 输出头 (Action Heads)
        # 作用：将最终的特征映射到具体的动作分布参数。
        self.mu_head = Linear(mlp_output_dim, CONTINUOUS_DIM)
        self.log_std_param = torch.nn.Parameter(torch.zeros(1, CONTINUOUS_DIM) * -0.5)
        self.discrete_head = Linear(mlp_output_dim, TOTAL_DISCRETE_LOGITS)

        # --- 初始化与优化器 ---
        self.apply(init_weights)
        init_range = 3e-3
        self.mu_head.weight.data.uniform_(-init_range, init_range)
        self.mu_head.bias.data.fill_(0)
        self.discrete_head.weight.data.uniform_(-init_range, init_range)
        self.discrete_head.bias.data.fill_(0)

        # --- 精细化优化器设置 ---

        # 1. 定义不同组的超参数
        attention_weight_decay = 1e-3  # 对注意力相关层使用更强的权重衰减
        default_weight_decay = self.weight_decay

        attention_params, other_params = [], []

        # 2. 遍历并分组参数
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # 将 attention 和 layernorm 相关的参数归为一组
            if any(key in name.lower() for key in ['attention', 'attn', 'layernorm']):
                attention_params.append(param)
            else:
                other_params.append(param)

        # 3. 创建参数组列表
        param_groups = [
            # 组1: 注意力相关参数，使用更强的正则化和更低的学习率
            {'params': attention_params, 'weight_decay': attention_weight_decay, 'lr': ACTOR_PARA.lr * 0.5},
            # 组2: 其他参数，使用默认设置
            {'params': other_params, 'weight_decay': default_weight_decay}
        ]

        # 4. 创建优化器和学习率调度器
        self.optim = torch.optim.Adam(param_groups, lr=ACTOR_PARA.lr)  # 全局lr作为其他组的默认值
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

        # 打印调试信息
        print(f"--- [DEBUG] Actor_GRU Initialized ---")
        print(
            f"[Optimizer] Attention group ({len(attention_params)} params): lr={ACTOR_PARA.lr * 0.5}, decay={attention_weight_decay}")
        print(f"[Optimizer] Other group ({len(other_params)} params): lr={ACTOR_PARA.lr}, decay={default_weight_decay}")

        # self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr, weight_decay=weight_decay)
        # self.actor_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
        #                                              end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
        #                                              total_iters=AGENTPARA.MAX_EXE_NUM)
        # self.to(ACTOR_PARA.device)
        #
        # # 打印调试信息，确认模型配置
        # print(f"--- [DEBUG] Actor_GRU initialized with {self.attention.num_heads} attention heads, "
        #       f"num_feature_tokens = {self.input_dim}, feature_embed_dim = {self.embedding_dim} ---")

    def forward(self, obs, hidden_state):
        """
        前向传播函数。
        obs: 原始状态输入，形状为 (B, D) 或 (B, S, D)。
        hidden_state: GRU的隐藏状态。
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        # 检查输入是否为序列，如果不是（单步推理），则增加一个长度为1的序列维度。
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)  # (B, D) -> (B, 1, D)

        B, S, D = obs_tensor.shape

        # --- 阶段一：特征交互 ---

        # 1. 特征Token化: 将每个时间步的D个特征看作D个独立的Token。
        # (B, S, D) -> (B*S, D, 1)
        feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)

        # 2. 特征嵌入: 将每个维度为1的Token投影到高维空间。
        # (B*S, D, 1) -> (B*S, D, embedding_dim)
        token_embeds = self.feature_embed(feat_tokens)

        # # 3. 特征级自注意力: 计算D个特征Token之间的关系。
        # # 3a. Pre-LayerNorm: 在送入注意力层前进行归一化。
        # normed_tokens = self.attention_layernorm(token_embeds)
        # # 3b. Multi-head Self-attention: 输入形状为(Batch, SeqLen, EmbedDim)，
        # #    在这里 Batch=B*S, SeqLen=D, EmbedDim=embedding_dim。
        # attn_out, _ = self.attention(normed_tokens, normed_tokens, normed_tokens)
        # attn_out = self.attn_dropout(attn_out)  # (B*S, D, embedding_dim)
        #
        # # 3c. 残差连接 + (可选) 后归一化
        # token_context = token_embeds + attn_out

        # 🔇 关闭注意力：直接跳过
        token_context = token_embeds

        token_context = self.post_layernorm(token_context)  # <<< ✅ 新增这行，可选的后归一化层 >>>

        # 4. 池化: 将D个交互后的特征Token聚合成一个单一的上下文向量。
        # 使用平均池化，作用在特征维度(D)上。
        # (B*S, D, embedding_dim) -> (B*S, embedding_dim)
        pooled_context = token_context.mean(dim=1)

        # --- 阶段二：时序建模 ---

        # 5. 序列化: 将处理好的上下文向量恢复成时间序列形式。
        # (B*S, embedding_dim) -> (B, S, embedding_dim)
        contextualized_sequence = pooled_context.view(B, S, self.embedding_dim)

        # 6. GRU处理: 将上下文序列送入GRU进行时序建模。
        # gru_out shape: (B, S, rnn_hidden_size)
        gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)

        # --- 阶段三：决策 ---

        # 7. MLP加工: 对GRU的输出进行深度加工。
        mlp_output = self.shared_network(gru_out)

        # 如果是单步推理，移除序列维度。
        final_features = mlp_output
        if not is_sequence:
            final_features = final_features.squeeze(1)  # (B, S, mlp_dim) -> (B, mlp_dim)

        # 8. 动作头: 生成最终的动作分布。
        mu = self.mu_head(final_features)
        all_disc_logits = self.discrete_head(final_features)

        # (后续动作分布的创建逻辑与原版完全相同)
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts

        has_flares_info = obs_tensor[..., 7]
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
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
        return distributions, new_hidden


class Critic_GRU(Module):
    """
    Critic 网络 (价值网络) - 采用与 Actor 完全相同的 [特征级 Attention -> GRU -> MLP] 结构。
    这保证了 Actor 和 Critic 对状态的理解和处理方式是一致的，有助于稳定训练。
    """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        super(Critic_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        self.embedding_dim = 64 #128 #RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay
        # --- 模块定义 (与 Actor 结构相同) ---

        # 1. 特征嵌入层
        self.feature_embed = Linear(1, self.embedding_dim)

        # 2. 特征级自注意力层
        assert self.embedding_dim % ATTN_NUM_HEADS == 0
        self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.1,
                                            batch_first=True)
        self.attn_dropout = Dropout(p=dropout_rate)
        # --- 定义两个LayerNorm ---
        self.attention_layernorm = LayerNorm(self.embedding_dim)  # 这个作为 Pre-LN
        self.post_layernorm = LayerNorm(self.embedding_dim)  # <<< ✅ 新增这个作为 Post-LN >>>

        # 3. GRU 时序建模层
        self.gru = GRU(self.embedding_dim, self.rnn_hidden_size, batch_first=True)

        # 4. MLP骨干网络
        layers_dims = [self.rnn_hidden_size] + CRITIC_PARA.model_layer_dim
        self.network_base = Sequential()
        for i in range(len(layers_dims) - 1):
            self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # 5. 输出头 (Value Head)
        base_output_dim = CRITIC_PARA.model_layer_dim[-1]
        self.fc_out = Linear(base_output_dim, self.output_dim)

        # --- 初始化与优化器 ---
        self.apply(init_weights)
        init_range = 3e-3
        self.fc_out.weight.data.uniform_(-init_range, init_range)
        self.fc_out.bias.data.fill_(0)

        # --- 精细化优化器设置 (与 Actor 逻辑相同) ---
        attention_weight_decay = 1e-3
        default_weight_decay = self.weight_decay
        attention_params, other_params = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name.lower() for key in ['attention', 'attn', 'layernorm']):
                attention_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': attention_params, 'weight_decay': attention_weight_decay, 'lr': CRITIC_PARA.lr * 0.5},
            {'params': other_params, 'weight_decay': default_weight_decay}
        ]

        self.optim = torch.optim.Adam(param_groups, lr=CRITIC_PARA.lr)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)

        # 打印调试信息
        print(f"--- [DEBUG] Critic_GRU Initialized ---")
        print(
            f"[Optimizer] Attention group ({len(attention_params)} params): lr={CRITIC_PARA.lr * 0.5}, decay={attention_weight_decay}")
        print(
            f"[Optimizer] Other group ({len(other_params)} params): lr={CRITIC_PARA.lr}, decay={default_weight_decay}")

        # self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr, weight_decay=weight_decay)
        # self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
        #                                               end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
        #                                               total_iters=AGENTPARA.MAX_EXE_NUM)
        # self.to(CRITIC_PARA.device)
        #
        # # 打印调试信息
        # print(f"--- [DEBUG] Critic_GRU initialized with {self.attention.num_heads} attention heads, "
        #       f"num_feature_tokens = {self.input_dim}, feature_embed_dim = {self.embedding_dim} ---")

    def forward(self, obs, hidden_state):
        """
        前向传播函数。
        obs: 原始状态输入，形状为 (B, D) 或 (B, S, D)。
        hidden_state: GRU的隐藏状态。
        """
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)

        B, S, D = obs_tensor.shape

        # (数据流与 Actor 完全相同)
        feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)
        token_embeds = self.feature_embed(feat_tokens)

        # normed_tokens = self.attention_layernorm(token_embeds)
        # attn_out, _ = self.attention(normed_tokens, normed_tokens, normed_tokens)
        # attn_out = self.attn_dropout(attn_out)
        #
        # token_context = token_embeds + attn_out

        # 🔇 关闭注意力：直接跳过
        token_context = token_embeds

        token_context = self.post_layernorm(token_context)  # <<< ✅ 新增这行，可选的后归一化层 >>>
        pooled_context = token_context.mean(dim=1)

        contextualized_sequence = pooled_context.view(B, S, self.embedding_dim)

        gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)

        # MLP 加工并输出价值
        base_features_sequence = self.network_base(gru_out)
        value = self.fc_out(base_features_sequence)

        if not is_sequence:
            value = value.squeeze(1)

        return value, new_hidden



# ==============================================================================
# <<< 修改 >>>: 定义新的基于 Attention + GRU 的 Actor 和 Critic
#               [💥 新结构: Attention -> GRU -> MLP -> Heads]
# ==============================================================================

# class Actor_GRU(Module):
#     """
#     Actor 网络 (策略网络) - [新结构: Attention -> GRU -> MLP]
#     结构为: 输入嵌入 -> Self-Attention 上下文感知 -> GRU 时序建模 -> MLP 深度加工 -> 独立动作头。
#     [正则化]: 在 Attention 输出端应用轻量级 Dropout，并使用 Weight Decay。
#     """
#
#     def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
#         super(Actor_GRU, self).__init__()
#         self.input_dim = ACTOR_PARA.input_dim
#         self.log_std_min = -20.0
#         self.log_std_max = 2.0
#         self.rnn_hidden_size = RNN_HIDDEN_SIZE
#         self.embedding_dim = RNN_HIDDEN_SIZE  # Attention 和 GRU 的工作维度
#
#         # 1. 输入嵌入层 (Input Embedding)
#         # 作用：将原始输入维度映射到更高维的嵌入空间，并进行特征抽象
#         self.input_embedding = Sequential(
#             Linear(self.input_dim, self.embedding_dim),
#             LayerNorm(self.embedding_dim),
#             LeakyReLU()
#         )
#
#         # 2. 注意力层 (Self-Attention)
#         assert self.embedding_dim % ATTN_NUM_HEADS == 0
#         self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
#                                             num_heads=ATTN_NUM_HEADS,
#                                             dropout=0.0,  # 避免策略随机性
#                                             batch_first=True)
#         # <<< --- 在这里加入这行代码 --- >>>
#         print(f"--- [DEBUG] Actor_GRU initialized with {self.attention.num_heads} attention heads. ---")
#
#         self.attn_dropout = Dropout(p=dropout_rate)
#         self.attention_layernorm = LayerNorm(self.embedding_dim)
#
#         # 3. GRU 层
#         self.gru = GRU(self.embedding_dim, self.rnn_hidden_size, batch_first=True)
#
#         # 4. 共享的 MLP 骨干网络
#         shared_layers_dims = [self.rnn_hidden_size] + ACTOR_PARA.model_layer_dim
#         self.shared_network = Sequential()
#         for i in range(len(shared_layers_dims) - 1):
#             self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
#             self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
#             self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())
#
#         # # <<< 💥 架构修改开始 >>>
#         #
#         # # 1. 移除了 input_embedding 层
#         #
#         # # 2. 注意力层 (Self-Attention) - 直接处理 input_dim
#         # # <<< 关键约束 >>>
#         # assert self.input_dim % ATTN_NUM_HEADS == 0, \
#         #     f"Input dimension ({self.input_dim}) must be divisible by the number of attention heads ({ATTN_NUM_HEADS})."
#         #
#         # self.attention = MultiheadAttention(embed_dim=self.input_dim,  # <-- 使用 input_dim
#         #                                     num_heads=ATTN_NUM_HEADS,
#         #                                     dropout=0.0,
#         #                                     batch_first=True)
#         # self.attn_dropout = Dropout(p=dropout_rate)
#         # self.attention_layernorm = LayerNorm(self.input_dim)  # <-- 使用 input_dim
#         #
#         # # 3. GRU 层
#         # #    输入维度是 Attention 的输出维度，即 self.input_dim
#         # self.gru = GRU(self.input_dim, self.rnn_hidden_size, batch_first=True)
#         #
#         # # 4. 共享的 MLP 骨干网络
#         # #    输入维度是 GRU 的输出维度，即 self.rnn_hidden_size
#         # shared_layers_dims = [self.rnn_hidden_size] + ACTOR_PARA.model_layer_dim
#         # self.shared_network = Sequential()
#         # for i in range(len(shared_layers_dims) - 1):
#         #     self.shared_network.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
#         #     self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
#         #     self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())
#         #
#         # # <<< 💥 架构修改结束 >>>
#
#         mlp_output_dim = ACTOR_PARA.model_layer_dim[-1]
#
#         # 5. 独立头部网络
#         self.mu_head = Linear(mlp_output_dim, CONTINUOUS_DIM)
#         self.log_std_param = torch.nn.Parameter(torch.zeros(1, CONTINUOUS_DIM) * -0.5)
#         self.discrete_head = Linear(mlp_output_dim, TOTAL_DISCRETE_LOGITS)
#
#         # 初始化
#         self.apply(init_weights)
#         init_range = 3e-3
#         self.mu_head.weight.data.uniform_(-init_range, init_range)
#         self.mu_head.bias.data.fill_(0)
#         self.discrete_head.weight.data.uniform_(-init_range, init_range)
#         self.discrete_head.bias.data.fill_(0)
#
#         # 优化器
#         self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
#         self.actor_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
#                                                      end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
#                                                      total_iters=AGENTPARA.MAX_EXE_NUM)
#         self.to(ACTOR_PARA.device)
#
#     def forward(self, obs, hidden_state):
#         obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
#         is_sequence = obs_tensor.dim() == 3
#         if not is_sequence:
#             obs_tensor = obs_tensor.unsqueeze(1)
#
#         # # 1. 输入嵌入
#         # embedded_input = self.input_embedding(obs_tensor)
#         #
#         # # 2. Self-Attention 模块 (Pre-LN)
#         # normed_input = self.attention_layernorm(embedded_input)
#         # attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         # attn_output = self.attn_dropout(attn_output)
#         # contextualized_sequence = embedded_input + attn_output  # 残差连接
#         #
#         # # 3. GRU 时序建模
#         # gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # <<< 💥 新的前向传播流程 >>>
#
#         # 1. (无嵌入层) 直接使用 obs_tensor
#
#         # # 2. Self-Attention 模块 (Pre-LN)
#         # normed_input = self.attention_layernorm(obs_tensor)  # <-- 直接对输入进行 Norm
#         # attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         # attn_output = self.attn_dropout(attn_output)
#         # contextualized_sequence = obs_tensor + attn_output  # 残差连接
#         #
#         # # 3. GRU 时序建模
#         # gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # --- 1️⃣ 输入准备 ---
#         # 如果是序列训练模式，就直接输入整段序列
#         # 如果是单步推理模式，则加上时间维度 (B,1,D)
#
#         # --- 2️⃣ Attention（只对每个时间步单独处理，不跨时刻）---
#         # 按时间步逐步处理：等价于特征注意力，而非时间注意力
#         # --- 1️⃣ 输入嵌入 ---
#         # 将原始输入映射到更高维的嵌入空间
#         embedded_input = self.input_embedding(obs_tensor)
#
#         # --- 2️⃣ 纯特征注意力（在嵌入空间上，且行为一致）---
#         # 通过 reshape 强制注意力机制只处理单一时间步，确保训练和推理一致
#         B, S, D_embed = embedded_input.shape
#         # 将 (B, S, D_embed) 变形为 (B*S, 1, D_embed)
#         embedded_reshaped = embedded_input.reshape(B * S, 1, D_embed)
#
#         normed_input = self.attention_layernorm(embedded_reshaped)
#         attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         attn_output = self.attn_dropout(attn_output)
#
#         # 残差连接
#         contextualized = embedded_reshaped + attn_output
#
#         # 将形状还原回 (B, S, D_embed)
#         contextualized_sequence = contextualized.view(B, S, D_embed)
#
#         # --- 3️⃣ GRU 时序建模（跨时间依赖）---
#         gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # 4. MLP 深度加工
#         mlp_output = self.shared_network(gru_out)
#
#         final_features = mlp_output
#         if not is_sequence:
#             final_features = final_features.squeeze(1)
#
#         # 5. 动作头和分布创建
#         mu = self.mu_head(final_features)
#         all_disc_logits = self.discrete_head(final_features)
#
#         split_sizes = list(DISCRETE_DIMS.values())
#         logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
#         trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
#
#         has_flares_info = obs_tensor[..., 7]
#         mask = (has_flares_info == 0)
#         trigger_logits_masked = trigger_logits.clone()
#         if torch.any(mask):
#             if mask.dim() < trigger_logits_masked.dim():
#                 mask = mask.unsqueeze(-1)
#             trigger_logits_masked[mask] = torch.finfo(torch.float32).min
#
#         log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std).expand_as(mu)
#         continuous_base_dist = Normal(mu, std)
#
#         trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
#         salvo_size_dist = Categorical(logits=salvo_size_logits)
#         intra_interval_dist = Categorical(logits=intra_interval_logits)
#         num_groups_dist = Categorical(logits=num_groups_logits)
#         inter_interval_dist = Categorical(logits=inter_interval_logits)
#
#         distributions = {
#             'continuous': continuous_base_dist,
#             'trigger': trigger_dist,
#             'salvo_size': salvo_size_dist,
#             'intra_interval': intra_interval_dist,
#             'num_groups': num_groups_dist,
#             'inter_interval': inter_interval_dist
#         }
#         return distributions, new_hidden
#
#
# class Critic_GRU(Module):
#     """
#     Critic 网络 (价值网络) - [新结构: Attention -> GRU -> MLP]
#     结构为: 输入嵌入 -> Self-Attention 上下文感知 -> GRU 时序建模 -> MLP 深度加工 -> 价值输出头。
#     [正则化]: 在 Attention 输出端应用轻量级 Dropout，并使用 Weight Decay。
#     """
#
#     def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
#         super(Critic_GRU, self).__init__()
#         self.input_dim = CRITIC_PARA.input_dim
#         self.output_dim = CRITIC_PARA.output_dim
#         self.rnn_hidden_size = RNN_HIDDEN_SIZE
#         self.embedding_dim = RNN_HIDDEN_SIZE  # Attention 和 GRU 的工作维度
#
#         # 1. 输入嵌入层 (Input Embedding)
#         self.input_embedding = Sequential(
#             Linear(self.input_dim, self.embedding_dim),
#             LayerNorm(self.embedding_dim),
#             LeakyReLU()
#         )
#
#         # 2. 注意力层 (Self-Attention)
#         assert self.embedding_dim % ATTN_NUM_HEADS == 0
#         self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
#                                             num_heads=ATTN_NUM_HEADS,
#                                             dropout=0.0,
#                                             batch_first=True)
#         self.attn_dropout = Dropout(p=dropout_rate)
#         self.attention_layernorm = LayerNorm(self.embedding_dim)
#
#         # 3. GRU 层
#         self.gru = GRU(self.embedding_dim, self.rnn_hidden_size, batch_first=True)
#
#         # 4. MLP 骨干网络
#         layers_dims = [self.rnn_hidden_size] + CRITIC_PARA.model_layer_dim
#         self.network_base = Sequential()
#         for i in range(len(layers_dims) - 1):
#             self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
#             self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
#             self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())
#
#         # # <<< 💥 架构修改开始 >>>
#         #
#         # # 1. 移除了 input_embedding 层
#         #
#         # # 2. 注意力层 (Self-Attention) - 直接处理 input_dim
#         # # <<< 关键约束 >>>
#         # assert self.input_dim % ATTN_NUM_HEADS == 0, \
#         #     f"Input dimension ({self.input_dim}) must be divisible by the number of attention heads ({ATTN_NUM_HEADS})."
#         #
#         # self.attention = MultiheadAttention(embed_dim=self.input_dim,  # <-- 使用 input_dim
#         #                                     num_heads=ATTN_NUM_HEADS,
#         #                                     dropout=0.0,
#         #                                     batch_first=True)
#         # self.attn_dropout = Dropout(p=dropout_rate)
#         # self.attention_layernorm = LayerNorm(self.input_dim)  # <-- 使用 input_dim
#         #
#         # # 3. GRU 层
#         # #    输入维度是 Attention 的输出维度，即 self.input_dim
#         # self.gru = GRU(self.input_dim, self.rnn_hidden_size, batch_first=True)
#         #
#         # # 4. 共享的 MLP 骨干网络
#         # #    输入维度是 GRU 的输出维度，即 self.rnn_hidden_size
#         # shared_layers_dims = [self.rnn_hidden_size] + CRITIC_PARA.model_layer_dim
#         # # <<< ✅ 修正: 变量名从 self.shared_network 改为 self.network_base >>>
#         # self.network_base = Sequential()
#         # for i in range(len(shared_layers_dims) - 1):
#         #     self.network_base.add_module(f'fc_{i}', Linear(shared_layers_dims[i], shared_layers_dims[i + 1]))
#         #     self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
#         #     self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())
#         #
#         # # <<< 💥 架构修改结束 >>>
#
#         base_output_dim = CRITIC_PARA.model_layer_dim[-1]
#
#         # 5. 输出头
#         self.fc_out = Linear(base_output_dim, self.output_dim)
#
#         # 初始化
#         self.apply(init_weights)
#         init_range = 3e-3
#         self.fc_out.weight.data.uniform_(-init_range, init_range)
#         self.fc_out.bias.data.fill_(0)
#
#         # 优化器
#         self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
#         self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
#                                                       end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
#                                                       total_iters=AGENTPARA.MAX_EXE_NUM)
#         self.to(CRITIC_PARA.device)
#
#     def forward(self, obs, hidden_state):
#         obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
#         is_sequence = obs_tensor.dim() == 3
#         if not is_sequence:
#             obs_tensor = obs_tensor.unsqueeze(1)
#
#         # # 1. 输入嵌入
#         # embedded_input = self.input_embedding(obs_tensor)
#         #
#         # # 2. Self-Attention 模块 (Pre-LN)
#         # normed_input = self.attention_layernorm(embedded_input)
#         # attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         # attn_output = self.attn_dropout(attn_output)
#         # contextualized_sequence = embedded_input + attn_output  # 残差连接
#         #
#         # # 3. GRU 时序建模
#         # gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # <<< 💥 新的前向传播流程 >>>
#
#         # 1. (无嵌入层) 直接使用 obs_tensor
#
#         # # 2. Self-Attention 模块 (Pre-LN)
#         # normed_input = self.attention_layernorm(obs_tensor)  # <-- 直接对输入进行 Norm
#         # attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         # attn_output = self.attn_dropout(attn_output)
#         # contextualized_sequence = obs_tensor + attn_output  # 残差连接
#         #
#         # # 3. GRU 时序建模
#         # gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # --- 1️⃣ 输入准备 ---
#         # 如果是序列训练模式，就直接输入整段序列
#         # 如果是单步推理模式，则加上时间维度 (B,1,D)
#
#         # --- 2️⃣ Attention（只对每个时间步单独处理，不跨时刻）---
#         # 按时间步逐步处理：等价于特征注意力，而非时间注意力
#         # --- 1️⃣ 输入嵌入 ---
#         # 将原始输入映射到更高维的嵌入空间
#         embedded_input = self.input_embedding(obs_tensor)
#
#         # --- 2️⃣ 纯特征注意力（在嵌入空间上，且行为一致）---
#         # 通过 reshape 强制注意力机制只处理单一时间步，确保训练和推理一致
#         B, S, D_embed = embedded_input.shape
#         # 将 (B, S, D_embed) 变形为 (B*S, 1, D_embed)
#         embedded_reshaped = embedded_input.reshape(B * S, 1, D_embed)
#
#         normed_input = self.attention_layernorm(embedded_reshaped)
#         attn_output, _ = self.attention(normed_input, normed_input, normed_input)
#         attn_output = self.attn_dropout(attn_output)
#
#         # 残差连接
#         contextualized = embedded_reshaped + attn_output
#
#         # 将形状还原回 (B, S, D_embed)
#         contextualized_sequence = contextualized.view(B, S, D_embed)
#
#         # --- 3️⃣ GRU 时序建模（跨时间依赖）---
#         gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # 4. MLP 深度加工
#         base_features_sequence = self.network_base(gru_out)
#
#         # 5. 价值输出头
#         value = self.fc_out(base_features_sequence)
#
#         if not is_sequence:
#             value = value.squeeze(1)
#
#         return value, new_hidden

# ==============================================================================
# PPO Agent 主类
# ==============================================================================

class PPO_continuous(object):
    """
    PPO 智能体主类。
    通过 `use_rnn` 标志来决定是使用 MLP 模型还是 GRU 模型。
    """
    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = False):
        super(PPO_continuous, self).__init__()
        # 根据 use_rnn 标志，实例化对应的 Actor 和 Critic 网络
        self.use_rnn = use_rnn
        if self.use_rnn:
            print("--- 初始化 PPO Agent (使用 [Attention -> GRU -> MLP] 模型) ---")
            # 实例化新的 Actor 和 Critic，并传入正则化参数
            self.Actor = Actor_GRU(dropout_rate=0.1, weight_decay=1e-4)
            self.Critic = Critic_GRU(dropout_rate=0.1, weight_decay=1e-4)
        else:
            print("--- 初始化 PPO Agent (使用 MLP 模型) ---")
            self.Actor = Actor()
            self.Critic = Critic()
        # 实例化经验回放池，并告知它是否需要处理 RNN 隐藏状态
        self.buffer = Buffer(use_rnn=self.use_rnn)
        # 从配置中加载 PPO 算法的超参数
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        self.training_start_time = time.strftime("PPOGRU_Attn_%Y-%m-%d_%H-%M-%S") # <<< 修改 >>> 更新存档文件夹名称
        self.base_save_dir = "../../../save/save_evade_fuza"
        # 为本次运行创建一个唯一的存档文件夹
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        # 如果需要加载预训练模型
        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../../../test/test_evade")

    def load_models_from_directory(self, directory_path: str):
        # This function is correct, no changes needed.
        """从指定目录加载 Actor 和 Critic 模型的权重。"""
        if not os.path.isdir(directory_path):
            print(f"[错误] 模型加载失败：提供的路径 '{directory_path}' 不是一个有效的文件夹。")
            return
        files = os.listdir(directory_path)
        # 优先尝试加载带有前缀的模型文件 (例如 "1000_Actor.pkl")
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
        # 如果没有带前缀的文件，则尝试加载默认的 "Actor.pkl" 和 "Critic.pkl"
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
        """将 tanh 输出的连续动作 (-1, 1) 缩放到环境定义的实际范围。"""
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)
        return scaled_action

    def get_initial_hidden_states(self, batch_size=1):
        """为 GRU 模型生成初始的零隐藏状态。"""
        if not self.use_rnn:
            return None, None
        actor_hidden = torch.zeros((1, batch_size, self.Actor.rnn_hidden_size), device=ACTOR_PARA.device)
        critic_hidden = torch.zeros((1, batch_size, self.Critic.rnn_hidden_size), device=CRITIC_PARA.device)
        return actor_hidden, critic_hidden

    def store_experience(self, state, action, probs, value, reward, done, actor_hidden=None, critic_hidden=None):
        """将单步经验存储到 Buffer 中。"""
        # 检查 log_prob 是否包含无效值（NaN 或 Inf）
        if not np.all(np.isfinite(probs)):
            raise ValueError("在 log_prob 中检测到 NaN/Inf！")
        # 如果使用 RNN，必须提供隐藏状态
        if self.use_rnn and (actor_hidden is None or critic_hidden is None):
            raise ValueError("使用 RNN 模型时必须存储隐藏状态！")
        self.buffer.store_transition(state, value, action, probs, reward, done, actor_hidden, critic_hidden)

    def choose_action(self, state, actor_hidden, critic_hidden, deterministic=False):
        """
        根据当前状态选择动作。
        :param deterministic: 如果为 True，则选择最可能的动作（用于评估），否则进行随机采样（用于训练）。
        :return: A tuple containing the action for the environment, action to store in buffer,
                 log probability, state value, and new hidden states.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        # 检查输入是否是批处理数据
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():# 在此块中不计算梯度
            if self.use_rnn:
                # GRU 模型需要传入隐藏状态
                value, new_critic_hidden = self.Critic(state_tensor, critic_hidden)
                dists, new_actor_hidden = self.Actor(state_tensor, actor_hidden)
            else:
                # MLP 模型不需要隐藏状态
                value = self.Critic(state_tensor)
                dists = self.Actor(state_tensor)
                new_critic_hidden, new_actor_hidden = None, None
            # 从连续动作分布中采样
            continuous_base_dist = dists['continuous']
            # u 是正态分布的原始样本，tanh(u) 是为了限制范围并引入非线性
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            action_cont_tanh = torch.tanh(u)
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)
            # 从所有离散动作分布中采样
            sampled_actions_dict = {}
            for key, dist in dists.items():
                if key == 'continuous': continue
                if deterministic:
                    if isinstance(dist, Categorical): # 分类分布：取概率最大的类别
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                    else: # 伯努利分布：取概率大于0.5的
                        # sampled_actions_dict[key] = (dist.probs > 0.5).float()
                        sampled_actions_dict[key] = dist.sample()
                else:
                    sampled_actions_dict[key] = dist.sample()
            # 计算总的对数概率
            log_prob_disc = sum(dists[key].log_prob(act) for key, act in sampled_actions_dict.items())
            total_log_prob = log_prob_cont + log_prob_disc
            # 准备要存储在 Buffer 中的动作（连续部分是原始样本 u，离散部分是类别索引）
            action_disc_to_store = torch.stack(list(sampled_actions_dict.values()), dim=-1).float()
            action_to_store = torch.cat([u, action_disc_to_store], dim=-1)
            # 准备要发送给环境的动作（连续部分是缩放后的动作，离散部分是类别索引）
            env_action_cont = self.scale_action(action_cont_tanh)
            final_env_action_tensor = torch.cat([env_action_cont, action_disc_to_store], dim=-1)
            # 将所有结果转换为 numpy 数组
            value_np = value.cpu().numpy()
            action_to_store_np = action_to_store.cpu().numpy()
            log_prob_to_store_np = total_log_prob.cpu().numpy()
            final_env_action_np = final_env_action_tensor.cpu().numpy()
            # 如果输入不是批处理，则移除批次维度
            if not is_batch:
                final_env_action_np = final_env_action_np[0]
                action_to_store_np = action_to_store_np[0]
                log_prob_to_store_np = log_prob_to_store_np[0]
                value_np = value_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np, value_np, new_actor_hidden, new_critic_hidden

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        """计算广义优势估计 (Generalized Advantage Estimation, GAE)。"""
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        # 从后向前遍历轨迹
        for t in reversed(range(len(rewards))):
            # 获取下一个状态的价值，如果是最后一步，则为 0
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            # done_mask 用于在回合结束时切断价值的传播
            done_mask = 1.0 - int(dones[t])
            # 计算 TD-error (delta)
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            # 递归计算 GAE
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantage[t] = gae
        return advantage

    def learn(self):
        """
        执行 PPO 的学习和更新步骤。已适配 RNN 模式并修正所有逻辑和维度错误。
        """
        # 如果 Buffer 中的数据不足一个批次，则跳过学习
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            print(
                f"  [Info] Buffer size ({self.buffer.get_buffer_size()}) < batch size ({BUFFERPARA.BATCH_SIZE}). Skipping.")
            return None
        # 1. 数据准备
        states, values, actions, old_probs, rewards, dones, _, __ = self.buffer.get_all_data()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        # ✅ 确保 values 是一维数组，与 advantages 对齐
        values = np.squeeze(values)  # 移除多余维度，比如 (N,1) → (N,)

        # ✅ 保证 returns 与 values 形状一致
        # returns (即 G_t) 是优势函数的 unbiased estimator
        returns = advantages + values
        # 用于记录训练过程中的各种损失和指标
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [],'entropy_cont': [], 'adv_targ': [], 'ratio': []}
        # 2. PPO 训练循环
        for _ in range(self.ppo_epoch):
            # 根据是否使用 RNN，选择不同的批次生成器
            if self.use_rnn:
                # 为 RNN 生成连续的序列批次
                batch_generator = self.buffer.generate_sequence_batches(
                    SEQUENCE_LENGTH, BUFFERPARA.BATCH_SIZE, advantages, returns
                )
            else:
                # 为 MLP 生成随机的批次索引
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                # 3. 批次数据处理
                if self.use_rnn:
                    # 解包 RNN 的序列批次数据
                    state, action_batch, old_prob, advantage, return_, initial_actor_h, initial_critic_h = batch_data
                    # 将 numpy 数组转换为 tensor 并移动到设备
                    state = check(state).to(**ACTOR_PARA.tpdv)
                    action_batch = check(action_batch).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)
                    advantage = check(advantage).to(**ACTOR_PARA.tpdv)
                    return_ = check(return_).to(**CRITIC_PARA.tpdv)
                    initial_actor_h = check(initial_actor_h).to(**ACTOR_PARA.tpdv)
                    initial_critic_h = check(initial_critic_h).to(**CRITIC_PARA.tpdv)
                else:
                    # 处理 MLP 的批次索引
                    batch_indices = batch_data
                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1)
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                # # 从批次动作中解析出连续和离散部分
                # 💥 不再需要切片出最后一个时间步，直接使用完整的序列
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(), # 类别索引需要是 long 类型
                    'intra_interval': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 3].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 4].long(),
                }

                # 4. Actor (策略) 网络训练
                if self.use_rnn:
                    new_dists, _ = self.Actor(state, initial_actor_h)
                else:
                    new_dists = self.Actor(state)
                # 计算新策略下，旧动作的对数概率
                # 💥 现在 u_from_buffer 和 new_dists['continuous'] 的形状都是 (B, 4)，可以匹配了
                # 计算 log_prob，所有张量都是序列，维度可以正确匹配
                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc
                # 计算策略的熵，以鼓励探索
                entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                total_entropy = (entropy_cont + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()
                # 计算新旧策略的比率 (importance sampling ratio)
                # 💥 使用完整的序列 old_prob 和 advantage
                # 注意：需要确保它们的形状与 new_prob 匹配
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0)) # clamp 防止数值溢出
                # advantage 可能是 (B, S, 1) 或 (B, S)，需要与 ratio (B, S) 对齐
                advantage_squeezed = advantage.squeeze(-1) if advantage.dim() > ratio.dim() else advantage
                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed
                # Actor 的损失是裁剪后的目标函数的负值，加上熵的正则项
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy
                # 反向传播和优化
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)  # 梯度裁剪
                self.Actor.optim.step()

                # 5. Critic (价值) 网络训练
                if self.use_rnn:
                    # 对于 RNN 模型，Critic 输出的是聚合后的单一价值
                    # new_value shape: (B, 1)
                    new_value, _ = self.Critic(state, initial_critic_h)
                else:
                    # 对于 MLP 模型，逻辑保持不变
                    # new_value shape: (B, 1)
                    new_value = self.Critic(state)

                # # ####################################################################
                # # # <<< FINAL, DEFINITIVE FIX FOR THE BROADCASTING WARNING >>>
                # # ####################################################################
                # # We ensure the target `return_` tensor has the same number of dimensions
                # # as the network's output `new_value`.
                # # 确保目标值 `return_` 和网络输出 `new_value` 的维度一致，以避免 PyTorch 的广播警告。
                # # 例如，`new_value` 可能是 (B, S, 1)，而 `return_` 是 (B, S)，这会导致不明确的广播。
                # # 通过 unsqueeze(-1) 将 `return_` 变为 (B, S, 1)，使其形状完全匹配。
                # 💥 确保 return_ 维度与 new_value 匹配
                if new_value.dim() > return_.dim():
                    return_ = return_.unsqueeze(-1)
                # ####################################################################
                # Critic 的损失是预测值和真实回报（returns）之间的均方误差
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                # 反向传播和优化
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()
                # 记录该批次的训练信息
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['entropy_cont'].append(entropy_cont.mean().item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())
            # 6. 更新学习率、清理和返回

            # <<< ✅ 在此处更新 Actor 的学习率 >>>
        # 在完成整个批次数据的学习后，让调度器步进一次
        self.Actor.actor_scheduler.step()

        # <<< ✅ 在此处更新 Critic 的学习率 >>>
        self.Critic.critic_scheduler.step()

        # 6. 清理和返回
        self.buffer.clear_memory() # 完成一轮学习后清空 Buffer
        # 计算整个 epoch 的平均指标
        for key in train_info:
            train_info[key] = np.mean(train_info[key])
        # 记录当前学习率
        # train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        # train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        # 记录 Actor 的学习率
        # param_groups[0] 是注意力组, param_groups[1] 是其他参数（主）组
        train_info['actor_lr_attn'] = self.Actor.optim.param_groups[0]['lr']
        train_info['actor_lr_main'] = self.Actor.optim.param_groups[1]['lr']

        # 记录 Critic 的学习率
        train_info['critic_lr_attn'] = self.Critic.optim.param_groups[0]['lr']
        train_info['critic_lr_main'] = self.Critic.optim.param_groups[1]['lr']
        self.save() # 保存模型
        return train_info

    def prep_training_rl(self):
        """将 Actor 和 Critic 网络设置为训练模式。"""
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        """将 Actor 和 Critic 网络设置为评估模式。"""
        self.Actor.eval()
        self.Critic.eval()

    def save(self, prefix=""):
        """保存 Actor 和 Critic 模型的权重。"""
        try:
            # 确保保存目录存在
            os.makedirs(self.run_save_dir, exist_ok=True)
            print(f"模型将被保存至: {self.run_save_dir}")
        except Exception as e:
            print(f"创建模型文件夹 {self.run_save_dir} 失败: {e}")
            return
        # 分别保存 Actor 和 Critic
        for net in ['Actor', 'Critic']:
            try:
                filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"
                full_path = os.path.join(self.run_save_dir, filename)
                torch.save(getattr(self, net).state_dict(), full_path)
                print(f"  - {filename} 保存成功。")
            except Exception as e:
                print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")