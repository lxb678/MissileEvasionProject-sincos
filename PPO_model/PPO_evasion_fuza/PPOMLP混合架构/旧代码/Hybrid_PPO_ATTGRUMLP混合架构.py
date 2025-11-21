# 导入 PyTorch 核心库
import torch
from torch.nn import *
# 导入概率分布工具，用于构建策略网络的输出
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
# 导入配置文件，其中包含各种超参数
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigAttn import *
# 导入支持 GRU 的经验回放池
from Interference_code.PPO_model.PPO_evasion_fuza.BufferGRUAttn import *
# 导入学习率调度器
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
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']# 对应飞机的油门、升降舵、副翼、方向舵
# 定义离散动作的维度。每个键代表一个离散决策，值代表该决策有多少个选项。
DISCRETE_DIMS = {
    'flare_trigger': 1,  # 干扰弹触发，伯努利分布 (是/否)，所以是1个 logit
    'salvo_size': 3,  # 齐射数量，3个选项，对应一个3类的分类分布
    # 'intra_interval': 3,  # 组内间隔，3个选项
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
    # 'salvo_size': [1, 2, 3],  # 修改为发射1、2、3枚
    # # 'intra_interval': [0.05, 0.1, 0.15],
    # 'intra_interval': [0.02, 0.04, 0.08],
    # 'num_groups': [1, 2, 3],
    # 'inter_interval': [0.2, 0.5, 1.0]

    'salvo_size': [2, 3, 4],  # 修改为发射2、3、4枚
    # 'intra_interval': [0.05, 0.1, 0.15],
    # 'intra_interval': [0.02, 0.04, 0.06],
    'num_groups': [2, 3, 4],
    'inter_interval': [0.2, 0.4, 0.6]
}
# 连续动作的物理范围，用于将网络输出 (-1, 1) 缩放到实际范围
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},
    'elevator': {'low': -1.0, 'high': 1.0},
    'aileron': {'low': -1.0, 'high': 1.0},
    'rudder': {'low': -1.0, 'high': 1.0},
}

# <<< GRU/RNN/Attention 修改 >>>: 新增 RNN 和 Attention 配置
RNN_HIDDEN_SIZE = 128 #64  # GRU 层的隐藏单元数量
SEQUENCE_LENGTH = 5 #10  # 训练时从经验池中采样的连续轨迹片段的长度
# ATTN_NUM_HEADS = 8     # 注意力机制的头数 (必须能被 MLP 输出维度整除)
ATTN_NUM_HEADS = 1  # 2 #3 #4 #8 #4 #1 #2       # <<< 这是您的关键修改：设置注意力机制的头数


# ==============================================================================
# Original MLP-based Actor and Critic (保留原始版本以供选择)
# ==============================================================================

class Actor(Module):
    """
    Actor 网络 (策略网络) - 基于 MLP（多层感知机）的版本。
    该网络负责根据当前状态决定要采取的动作策略。
    它具有一个共享的骨干网络，后接两个独立的头部，分别处理连续动作和离散动作。
    """

    def __init__(self):
        """初始化 Actor 网络的结构和优化器。"""
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim  # 输入维度，即状态空间的维度
        # <<< 更改 >>> 输出维度现在是 (连续*2) + 新的logits总数
        # 注意：这里不再需要一个总的 output_dim
        # self.output_dim = (CONTINUOUS_DIM * 2) + TOTAL_DISCRETE_LOGITS
        self.log_std_min = -20.0  # 限制对数标准差的最小值，防止数值不稳定
        self.log_std_max = 2.0  # 限制对数标准差的最大值

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
            # self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
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
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)  # 使用 Adam 优化器
        # 定义学习率调度器，实现学习率的线性衰减
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,  # 初始学习率因子
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,  # 最终学习率因子
            total_iters=AGENTPARA.MAX_EXE_NUM  # 达到最终学习率所需的总迭代次数
        )
        self.to(ACTOR_PARA.device)  # 将模型移动到指定的设备 (CPU 或 GPU)

    def forward(self, obs):
        """
        前向传播方法，为每个动作维度创建并返回一个概率分布。
        :param obs: 状态观测值
        :return: 一个包含所有动作概率分布的字典
        """
        # 确保输入是 PyTorch 张量并移动到正确的设备
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        # 如果输入是一维的（单个状态），增加一个批次维度
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 通过共享骨干网络提取通用特征
        shared_features = self.shared_network(obs_tensor)

        # 2. 将共享特征分别送入不同的头部网络
        cont_params = self.continuous_head(shared_features)  # 获取连续动作的参数
        all_disc_logits = self.discrete_head(shared_features)  # 获取所有离散动作的 logits

        # ... 后续逻辑与原版完全相同 ...
        # 根据 DISCRETE_DIMS 的定义，将总的 logits 切分成对应每个离散动作的部分
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=1)
        trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts

        # 获取状态中关于干扰弹数量的信息（状态向量的第8个元素，索引为7）
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
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))  # 伯努利分布
        salvo_size_dist = Categorical(logits=salvo_size_logits)  # 分类分布
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
        """初始化 Critic 网络的结构和优化器。"""
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
            # self.network.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
        # 输出层，输出一个标量值，即状态价值 V(s)
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

    def forward(self, obs):
        """前向传播，计算状态价值。"""
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


def init_weights(m, gain=1.0):
    """
    一个通用的权重初始化函数。
    :param m: PyTorch module
    :param gain: 正交初始化的增益因子
    """
    if isinstance(m, Linear):
        # 对线性层使用 Kaiming Normal 初始化，适用于 LeakyReLU 激活函数
        torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            # 将偏置初始化为0
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, GRU):
        # 对 GRU 的权重使用正交初始化，有助于稳定 RNN 的训练
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏层的权重
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'bias' in name:  # 偏置
                param.data.fill_(0)


# # ==============================================================================
# #           最终版本：特征级注意力 + GRU时序建模 (修改版，无池化瓶颈)
# # ==============================================================================
#
# class Actor_GRU(Module):
#     """
#     Actor 网络 (策略网络) - [修改版 V3: 有嵌入层, 无池化瓶颈]
#     架构流程: [特征嵌入 -> 原始特征级 Attention -> GRU -> MLP -> 动作头]
#
#     架构特点：
#     - 特征嵌入：将每个状态特征（标量）映射到一个高维向量空间，使自注意力能够在此空间中捕捉更丰富的关系。
#     - 无池化瓶颈：注意力模块的输出（每个特征的上下文感知嵌入）被完全保留。它们被展平(flatten)成一个长向量，
#                   然后送入GRU。这避免了池化操作可能导致的信息损失。
#     - 多头注意力：由于特征被嵌入到高维空间，现在可以使用多头注意力机制来并行地关注来自不同表示子空间的信息。
#     """
#     """
#         Actor 网络 - [最终架构: Attention -> GRU -> Hybrid MLP -> Heads]
#         """
#
#     def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
#         """初始化网络的各个模块、权重和优化器。"""
#         super(Actor_GRU, self).__init__()
#         # --- 基础参数定义 ---
#         self.input_dim = ACTOR_PARA.input_dim  # D, 状态特征的数量
#         self.log_std_min = -20.0
#         self.log_std_max = 2.0
#         self.rnn_hidden_size = self.input_dim #RNN_HIDDEN_SIZE
#         self.weight_decay = weight_decay
#         # <<< MODIFIED 1/6 >>>: 定义一个新的嵌入维度
#         self.embedding_dim = 1 #8 #16 #1 #32  # 每个特征将被映射到这个维度
#
#         # <<< MODIFIED 2/6 >>>: 检查多头注意力的约束
#         # 多头注意力的一个要求是：嵌入维度必须能被头的数量整除
#         assert self.embedding_dim % ATTN_NUM_HEADS == 0, \
#             f"embedding_dim ({self.embedding_dim}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"
#
#         # --- 模块定义 ---
#         # <<< ADDED BACK 3/6 >>>: 重新引入特征嵌入层
#         # 将每个1维特征映射到 embedding_dim 维
#         # self.feature_embed = Linear(1, self.embedding_dim)
#
#         # 1. 特征级自注意力层
#         # 在 D 个特征的嵌入向量之间计算注意力，以捕捉特征间的相互关系
#         self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
#                                             num_heads=ATTN_NUM_HEADS,
#                                             dropout=0.0,  # 在策略网络中通常不使用 dropout
#                                             batch_first=True)  # 输入输出格式为 (Batch, Seq, Feature)
#         # self.attention_layernorm = LayerNorm(self.embedding_dim, elementwise_affine=False) # 可选的层归一化
#
#
#         # 2. GRU 时序建模层
#         # <<< MODIFIED 5/6 >>>: GRU的输入维度现在是 D * embedding_dim
#         # 因为所有特征的上下文感知嵌入被展平后送入GRU
#         gru_input_dim = self.input_dim * self.embedding_dim
#         # self.rnn_hidden_size = gru_input_dim
#         self.gru = GRU(gru_input_dim, self.rnn_hidden_size, batch_first=True)
#
#         # --- 3. 混合 MLP 架构 (处理 GRU 的输出) ---
#         #    在这里应用你的混合架构设计
#         split_point = 2  # 假设 ACTOR_PARA.model_layer_dim 有3层或更多
#         base_dims = ACTOR_PARA.model_layer_dim[:split_point]
#         continuous_tower_dims = ACTOR_PARA.model_layer_dim[split_point:]
#         discrete_tower_dims = continuous_tower_dims
#         # discrete_tower_dims = [dim // 2 for dim in continuous_tower_dims]
#
#         # 4a. 构建共享MLP基座 (Shared Base MLP)
#         self.shared_base_mlp = Sequential()
#         base_input_dim = self.rnn_hidden_size  # MLP的输入是 GRU 的输出
#         for i, dim in enumerate(base_dims):
#             self.shared_base_mlp.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
#             self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', LeakyReLU())
#             base_input_dim = dim
#         base_output_dim = base_dims[-1] if base_dims else self.rnn_hidden_size
#
#         # 4b. 构建连续动作塔楼 (Continuous Tower)
#         self.continuous_tower = Sequential()
#         tower_input_dim = base_output_dim
#         for i, dim in enumerate(continuous_tower_dims):
#             self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         continuous_tower_output_dim = continuous_tower_dims[-1] if continuous_tower_dims else base_output_dim
#
#         # 4c. 构建离散动作塔楼 (Discrete Tower)
#         self.discrete_tower = Sequential()
#         tower_input_dim = base_output_dim
#         for i, dim in enumerate(discrete_tower_dims):
#             self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         discrete_tower_output_dim = discrete_tower_dims[-1] if discrete_tower_dims else base_output_dim
#
#         # 5. 定义最终的输出头 (Heads)
#         self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
#         self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
#         self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))
#
#         # --- <<< MODIFICATION START: 精细化优化器设置 >>> ---
#         # 1. 初始化空的参数列表
#         gru_params = []
#         attention_params = []
#         other_params = []
#
#         # 2. 遍历所有命名参数，并将它们分配到对应的组中
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue
#
#             # 根据参数名称中的关键词进行分组
#             if 'gru' in name.lower():
#                 gru_params.append(param)
#             elif any(key in name.lower() for key in ['attention', 'attn', 'layernorm']):
#                 # 注意：你之前的代码把 layernorm 也归入了 attention 组，这里保持一致
#                 attention_params.append(param)
#             else:
#                 other_params.append(param)
#
#         # 3. 创建参数组 (parameter groups) 列表
#         param_groups = [
#             {
#                 'params': gru_params,
#                 'lr': ACTOR_PARA.gru_lr  # 使用为 GRU 定义的专属学习率
#             },
#             {
#                 'params': attention_params,
#                 'lr': ACTOR_PARA.attention_lr  # 使用为 Attention 定义的专属学习率
#             },
#             {
#                 'params': other_params  # 其他所有参数
#                 # 这里不指定 'lr'，它们将使用优化器构造函数中的默认 lr
#             }
#         ]
#
#         # 4. 使用参数组初始化优化器
#         # 传入的 lr=ACTOR_PARA.lr 将作为 "other_params" 组的默认学习率
#         self.optim = torch.optim.Adam(param_groups, lr=ACTOR_PARA.lr, weight_decay=self.weight_decay)
#
#         print("--- Actor Optimizer Initialized with Parameter Groups ---")
#         print(f"  - GRU Params LR: {ACTOR_PARA.gru_lr}")
#         print(f"  - Attention Params LR: {ACTOR_PARA.attention_lr}")
#         print(f"  - Other Params LR: {ACTOR_PARA.lr}")
#
#         self.actor_scheduler = lr_scheduler.LinearLR(
#             self.optim,
#             start_factor=1.0,
#             end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
#             total_iters=AGENTPARA.MAX_EXE_NUM
#         )
#         self.to(ACTOR_PARA.device)
#         # print(f"--- [DEBUG] Actor_GRU Initialized (No Pooling Bottleneck) ---")
#
#     def forward(self, obs, hidden_state):
#         """
#         前向传播方法。
#         :param obs: 状态观测值，形状为 (B, D) 或 (B, S, D)
#         :param hidden_state: GRU 的隐藏状态
#         :return: 包含动作分布的字典和新的 GRU 隐藏状态
#         """
#         obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
#         # 检查输入是否为序列。如果不是（例如，单步推理），则增加一个长度为1的序列维度。
#         is_sequence = obs_tensor.dim() == 3
#         if not is_sequence:
#             obs_tensor = obs_tensor.unsqueeze(1)  # (B, D) -> (B, 1, D)
#         B, S, D = obs_tensor.shape  # B:批大小, S:序列长度, D:特征维度
#
#         # --- 阶段一：特征交互 ---
#         # 1. 准备特征 token: (B, S, D) -> (B*S, D, 1)
#         feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)
#
#         # <<< ADDED BACK 6/6, part 1 >>>: 应用嵌入层
#         # 2. 特征嵌入: (B*S, D, 1) -> (B*S, D, embedding_dim)
#         # token_embeds = self.feature_embed(feat_tokens)
#         token_embeds = feat_tokens
#         #层归一化
#         # token_embeds = self.attention_layernorm(token_embeds)
#
#
#         # 3. 自注意力计算
#         # query, key, value 都是 token_embeds，进行自注意力计算
#         attn_out, _ = self.attention(token_embeds, token_embeds, token_embeds)
#
#         # 4. 残差连接：将注意力输出与原始嵌入相加，有助于梯度传播
#         token_context = token_embeds + attn_out  # 形状: (B*S, D, embedding_dim)
#
#         # 层归一化
#         # token_context = self.attention_layernorm(token_context)
#
#         # <<< MODIFIED 6/6, part 2 >>>: 重塑以保留所有信息
#         # 5. 展平上下文向量: (B*S, D, embedding_dim) -> (B*S, D * embedding_dim)
#         flattened_context = token_context.contiguous().view(B * S, -1)
#
#         # 6. 恢复序列结构: (B*S, D * embedding_dim) -> (B, S, D * embedding_dim)
#         contextualized_sequence = flattened_context.view(B, S, -1)
#
#         # --- 阶段二：时序建模 ---
#         # 7. GRU处理，现在输入的是经过特征交互和展平后的完整序列
#         gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # --- 阶段三：决策 (Hybrid MLP) ---
#         # 1. GRU 的输出流经共享 MLP 基座
#         base_features = self.shared_base_mlp(gru_out)
#
#         # 2. 共享特征被分别送入两个专用塔楼
#         continuous_features = self.continuous_tower(base_features)
#         discrete_features = self.discrete_tower(base_features)
#
#         # 3. 如果是单步输入，压缩特征维度以匹配头部
#         if not is_sequence:
#             continuous_features = continuous_features.squeeze(1)
#             discrete_features = discrete_features.squeeze(1)
#
#         # 4. 每个头部接收来自其专属塔楼的特征
#         mu = self.mu_head(continuous_features)
#         all_disc_logits = self.discrete_head(discrete_features)
#
#         # (后续创建分布的逻辑与 MLP 版本完全相同)
#         split_sizes = list(DISCRETE_DIMS.values())
#         logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
#         # trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
#         trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts
#
#         has_flares_info = obs_tensor[..., 7]
#         mask = (has_flares_info == 0)
#         trigger_logits_masked = trigger_logits.clone()
#         if torch.any(mask):
#             if mask.dim() < trigger_logits_masked.dim():
#                 mask = mask.unsqueeze(-1)
#             trigger_logits_masked[mask] = torch.finfo(torch.float32).min
#
#         # ===============================================================
#         # 5️⃣ 触发器层次控制：当“不投放”时，屏蔽其他离散动作 logits
#         # ===============================================================
#         # 先得到触发器分布
#         trigger_probs = torch.sigmoid(trigger_logits_masked)  # shape: [B,1]
#
#         # 如果 trigger_probs < 0.5，说明模型倾向于“不投放”
#         # 我们用这个条件生成一个 mask（True=不投放）
#         no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)  # shape: [B]
#
#         # 创建 logits 的副本，避免原地操作污染梯度
#         salvo_size_logits_masked = salvo_size_logits.clone()
#         # intra_interval_logits_masked = intra_interval_logits.clone()
#         num_groups_logits_masked = num_groups_logits.clone()
#         inter_interval_logits_masked = inter_interval_logits.clone()
#         # ===============================================================
#         # 当 trigger 不投放时，将其他离散动作 logits 强制为 index=0 (one-hot 形式)
#         # ===============================================================
#         if torch.any(no_trigger_mask):
#             INF = 1e6
#             NEG_INF = -1e6
#             for logits_tensor in [
#                 salvo_size_logits_masked,
#                 # intra_interval_logits_masked,
#                 num_groups_logits_masked,
#                 inter_interval_logits_masked,
#             ]:
#                 logits_sub = logits_tensor[no_trigger_mask]
#                 if logits_sub.numel() > 0:
#                     logits_sub[:] = NEG_INF  # 全部置为极小值
#                     logits_sub[:, 0] = INF  # 仅 index=0 置为极大值
#                     logits_tensor[no_trigger_mask] = logits_sub
#
#         log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std).expand_as(mu)
#         continuous_base_dist = Normal(mu, std)
#
#         trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
#         salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
#         # intra_interval_dist = Categorical(logits=intra_interval_logits_masked)
#         num_groups_dist = Categorical(logits=num_groups_logits_masked)
#         inter_interval_dist = Categorical(logits=inter_interval_logits_masked)
#
#         distributions = {
#             'continuous': continuous_base_dist,
#             'trigger': trigger_dist,
#             'salvo_size': salvo_size_dist,
#             # 'intra_interval': intra_interval_dist,
#             'num_groups': num_groups_dist,
#             'inter_interval': inter_interval_dist
#         }
#         return distributions, new_hidden
#
#
# class Critic_GRU(Module):
#     """
#     Critic 网络 (价值网络) - 采用与 Actor 相似的 GRU 结构，但为了简化，这里没有使用 Attention。
#     这是一个 GRU -> MLP -> Value Head 的结构。
#     """
#
#     def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
#         """初始化 Critic 网络的模块、权重和优化器。"""
#         super(Critic_GRU, self).__init__()
#         # --- 基础参数定义 ---
#         self.input_dim = CRITIC_PARA.input_dim  # D
#         self.output_dim = CRITIC_PARA.output_dim
#         self.rnn_hidden_size = self.input_dim #RNN_HIDDEN_SIZE
#         self.weight_decay = weight_decay
#
#         # <<< MODIFIED 1/6 >>>: 定义一个新的嵌入维度
#         self.embedding_dim = 1  # 8 #16 #1 #32  # 每个特征将被映射到这个维度
#
#         # <<< MODIFIED 2/6 >>>: 检查多头注意力的约束
#         # 多头注意力的一个要求是：嵌入维度必须能被头的数量整除
#         assert self.embedding_dim % ATTN_NUM_HEADS == 0, \
#             f"embedding_dim ({self.embedding_dim}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"
#
#         # --- 模块定义 ---
#         # <<< ADDED BACK 3/6 >>>: 重新引入特征嵌入层
#         # 将每个1维特征映射到 embedding_dim 维
#         # self.feature_embed = Linear(1, self.embedding_dim)
#
#         # 1. 特征级自注意力层
#         # 在 D 个特征的嵌入向量之间计算注意力，以捕捉特征间的相互关系
#         self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
#                                             num_heads=ATTN_NUM_HEADS,
#                                             dropout=0.0,  # 在策略网络中通常不使用 dropout
#                                             batch_first=True)  # 输入输出格式为 (Batch, Seq, Feature)
#
#         # 2. GRU 时序建模层
#         # <<< MODIFIED 5/6 >>>: GRU的输入维度现在是 D * embedding_dim
#         # 因为所有特征的上下文感知嵌入被展平后送入GRU
#         gru_input_dim = self.input_dim * self.embedding_dim
#
#         # --- 模块定义 ---
#         # 1. GRU 时序建模层
#         # 输入维度是原始状态维度 D (self.input_dim)
#         self.gru = GRU(gru_input_dim, self.rnn_hidden_size, batch_first=True)
#
#         # 2. MLP骨干网络 (不变)
#         layers_dims = [self.rnn_hidden_size] + CRITIC_PARA.model_layer_dim
#         self.network_base = Sequential()
#         for i in range(len(layers_dims) - 1):
#             self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
#             # self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
#             self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())
#
#         # 3. 输出头 (Value Head) (不变)
#         base_output_dim = CRITIC_PARA.model_layer_dim[-1]
#         self.fc_out = Linear(base_output_dim, self.output_dim)
#
#         # # --- [新增] 应用初始化 ---
#         # self.apply(init_weights)  # 对所有子模块应用通用初始化
#
#         # # --- [新增] 对输出层进行特殊初始化 ---
#         # # 这样做是为了在训练开始时有更稳定的价值估计
#         # init_range = 3e-3
#         # self.fc_out.weight.data.uniform_(-init_range, init_range)
#         # self.fc_out.bias.data.fill_(0)
#         # # --- 初始化结束 ---
#
#         # --- <<< MODIFICATION START: 精细化优化器设置 >>> ---
#         # 1. 初始化空的参数列表
#         gru_params = []
#         attention_params = []
#         other_params = []
#
#         # 2. 遍历所有命名参数，并将它们分配到对应的组中
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue
#
#             # 根据参数名称中的关键词进行分组
#             if 'gru' in name.lower():
#                 gru_params.append(param)
#             elif any(key in name.lower() for key in ['attention', 'attn', 'layernorm']):
#                 # 注意：你之前的代码把 layernorm 也归入了 attention 组，这里保持一致
#                 attention_params.append(param)
#             else:
#                 other_params.append(param)
#
#         # 3. 创建参数组 (parameter groups) 列表
#         param_groups = [
#             {
#                 'params': gru_params,
#                 'lr': CRITIC_PARA.gru_lr  # 使用为 GRU 定义的专属学习率
#             },
#             {
#                 'params': attention_params,
#                 'lr': CRITIC_PARA.attention_lr  # 使用为 Attention 定义的专属学习率
#             },
#             {
#                 'params': other_params  # 其他所有参数
#                 # 这里不指定 'lr'，它们将使用优化器构造函数中的默认 lr
#             }
#         ]
#
#         # 4. 使用参数组初始化优化器
#         # 传入的 lr=CRITIC_PARA.lr 将作为 "other_params" 组的默认学习率
#         self.optim = torch.optim.Adam(param_groups, lr=CRITIC_PARA.lr, weight_decay=self.weight_decay)
#
#         print("--- Actor Optimizer Initialized with Parameter Groups ---")
#         print(f"  - GRU Params LR: {CRITIC_PARA.gru_lr}")
#         print(f"  - Attention Params LR: {CRITIC_PARA.attention_lr}")
#         print(f"  - Other Params LR: {CRITIC_PARA.lr}")
#
#         self.critic_scheduler = lr_scheduler.LinearLR(
#             self.optim,
#             start_factor=1.0,
#             end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
#             total_iters=AGENTPARA.MAX_EXE_NUM
#         )
#         self.to(CRITIC_PARA.device)
#         # print(f"--- [DEBUG] Critic_GRU Initialized (No Pooling Bottleneck) ---")
#
#     def forward(self, obs, hidden_state):
#         """
#         前向传播方法。
#         :param obs: 状态观测值，形状为 (B, D) 或 (B, S, D)
#         :param hidden_state: GRU 的隐藏状态
#         :return: 状态价值和新的 GRU 隐藏状态
#         """
#         obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
#         # 检查输入是否为序列。如果不是（例如，单步推理），则增加一个长度为1的序列维度。
#         is_sequence = obs_tensor.dim() == 3
#         if not is_sequence:
#             obs_tensor = obs_tensor.unsqueeze(1) # (B, D) -> (B, 1, D)
#         B, S, D = obs_tensor.shape  # B:批大小, S:序列长度, D:特征维度
#         # --- 阶段一：特征交互 ---
#         # 1. 准备特征 token: (B, S, D) -> (B*S, D, 1)
#         feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)
#
#         # <<< ADDED BACK 6/6, part 1 >>>: 应用嵌入层
#         # 2. 特征嵌入: (B*S, D, 1) -> (B*S, D, embedding_dim)
#         # token_embeds = self.feature_embed(feat_tokens)
#         token_embeds = feat_tokens
#         # 层归一化
#         # token_embeds = self.attention_layernorm(token_embeds)
#
#         # 3. 自注意力计算
#         # query, key, value 都是 token_embeds，进行自注意力计算
#         attn_out, _ = self.attention(token_embeds, token_embeds, token_embeds)
#
#         # 4. 残差连接：将注意力输出与原始嵌入相加，有助于梯度传播
#         token_context = token_embeds + attn_out  # 形状: (B*S, D, embedding_dim)
#
#         # 层归一化
#         # token_context = self.attention_layernorm(token_context)
#
#         # <<< MODIFIED 6/6, part 2 >>>: 重塑以保留所有信息
#         # 5. 展平上下文向量: (B*S, D, embedding_dim) -> (B*S, D * embedding_dim)
#         flattened_context = token_context.contiguous().view(B * S, -1)
#
#         # 6. 恢复序列结构: (B*S, D * embedding_dim) -> (B, S, D * embedding_dim)
#         contextualized_sequence = flattened_context.view(B, S, -1)
#
#         # --- 阶段二：时序建模 ---
#         # 7. GRU处理，现在输入的是经过特征交互和展平后的完整序列
#         gru_out, new_hidden = self.gru(contextualized_sequence, hidden_state)
#
#         # # --- 阶段一：时序建模 ---
#         # # 直接将原始状态序列送入GRU
#         # gru_out, new_hidden = self.gru(obs_tensor, hidden_state)
#
#         # --- 阶段二：价值评估 (不变) ---
#         # MLP加工
#         base_features_sequence = self.network_base(gru_out)
#         # 输出价值
#         value = self.fc_out(base_features_sequence)
#
#         # 如果是单步推理，移除序列维度
#         if not is_sequence:
#             value = value.squeeze(1)
#
#         return value, new_hidden


# ==============================================================================
#           完整新架构: Attention -> MLP -> GRU -> MLP_Towers -> Heads
# ==============================================================================

class Actor_GRU(Module):
    """
    Actor 网络 - [完整版 V2: Attention -> MLP -> GRU -> Hybrid MLP]
    结构为: 跨特征注意力 -> 共享MLP特征提取 -> GRU 序列处理 -> 专用MLP塔楼 -> 独立动作头。
    """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        """初始化网络的各个模块、权重和优化器。"""
        super(Actor_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = ACTOR_PARA.input_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay
        self.embedding_dim = 1  # 每个原始特征的嵌入维度

        assert self.embedding_dim % ATTN_NUM_HEADS == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"

        # --- 1. 特征级自注意力层 (处理原始输入) ---
        self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)

        # --- 2. 定义 GRU 之前的共享 MLP (Pre-GRU MLP) ---
        # 这个MLP的输入维度是Attention模块的输出维度
        pre_gru_mlp_input_dim = self.input_dim * self.embedding_dim
        pre_gru_mlp_layers = 2
        pre_gru_dims = ACTOR_PARA.model_layer_dim[:pre_gru_mlp_layers]

        self.pre_gru_mlp = Sequential()
        mlp_input_dim = pre_gru_mlp_input_dim
        for i, dim in enumerate(pre_gru_dims):
            self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(mlp_input_dim, dim))
            self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
            mlp_input_dim = dim

        # Pre-GRU MLP 的输出维度，将作为 GRU 的输入维度
        pre_gru_output_dim = pre_gru_dims[-1] if pre_gru_dims else pre_gru_mlp_input_dim

        # --- 3. GRU 时序建模层 ---
        self.gru = GRU(pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)

        # --- 4. 定义 GRU 之后的 MLP 塔楼 (Post-GRU Towers) ---
        post_gru_dims = ACTOR_PARA.model_layer_dim[pre_gru_mlp_layers:]

        # 4a. 构建连续动作塔楼
        self.continuous_tower = Sequential()
        tower_input_dim = self.rnn_hidden_size
        for i, dim in enumerate(post_gru_dims):
            self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim
        continuous_tower_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size

        # 4b. 构建离散动作塔楼
        self.discrete_tower = Sequential()
        tower_input_dim = self.rnn_hidden_size
        for i, dim in enumerate(post_gru_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim
        discrete_tower_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size

        # --- 5. 定义最终的输出头 (Heads) ---
        self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))

        # --- 6. 优化器设置 ---
        gru_params, attention_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'gru' in name.lower():
                gru_params.append(param)
            elif any(key in name.lower() for key in ['attention', 'attn']):
                attention_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': gru_params, 'lr': ACTOR_PARA.gru_lr},
            {'params': attention_params, 'lr': ACTOR_PARA.attention_lr},
            {'params': other_params, 'lr': ACTOR_PARA.lr}
        ]

        self.optim = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def forward(self, obs, hidden_state):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)
        B, S, D = obs_tensor.shape

        # --- 阶段一: 特征交互 (Attention) ---
        # 1. 准备特征 token 并应用 Attention
        feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)
        token_embeds = feat_tokens
        attn_out, _ = self.attention(token_embeds, token_embeds, token_embeds)
        token_context = token_embeds + attn_out

        # 2. 展平并恢复序列结构
        flattened_context = token_context.contiguous().view(B * S, -1)
        contextualized_sequence = flattened_context.view(B, S, -1)

        # --- 阶段二: 特征提取 (Pre-GRU MLP) ---
        # 3. 将经过 Attention 的序列送入 MLP
        mlp_sequence = self.pre_gru_mlp(contextualized_sequence)

        # --- 阶段三: 时序建模 (GRU) ---
        # 4. 将 MLP 处理后的序列送入 GRU
        gru_out, new_hidden = self.gru(mlp_sequence, hidden_state)

        # --- 阶段四: 决策 (Post-GRU MLP Towers) ---
        # 5. GRU 的输出送入后续的塔楼和头部
        base_features = gru_out
        continuous_features = self.continuous_tower(base_features)
        discrete_features = self.discrete_tower(base_features)

        if not is_sequence:
            continuous_features = continuous_features.squeeze(1)
            discrete_features = discrete_features.squeeze(1)

        mu = self.mu_head(continuous_features)
        all_disc_logits = self.discrete_head(discrete_features)

        # --- 阶段五: 创建概率分布 (完整代码) ---
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        has_flares_info = obs_tensor[..., 11]
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        # 触发器层次控制
        trigger_probs = torch.sigmoid(trigger_logits_masked)
        no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)

        salvo_size_logits_masked = salvo_size_logits.clone()
        num_groups_logits_masked = num_groups_logits.clone()
        inter_interval_logits_masked = inter_interval_logits.clone()

        if torch.any(no_trigger_mask):
            INF = 1e6
            NEG_INF = -1e6
            for logits_tensor in [salvo_size_logits_masked, num_groups_logits_masked, inter_interval_logits_masked]:
                # 仅对需要屏蔽的样本进行操作
                logits_sub = logits_tensor[no_trigger_mask]
                if logits_sub.numel() > 0:
                    logits_sub[:] = NEG_INF  # 全部置为极小值
                    logits_sub[:, 0] = INF  # 仅 index=0 置为极大值
                    logits_tensor[no_trigger_mask] = logits_sub

        # 创建连续动作分布
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        continuous_base_dist = Normal(mu, std)

        # 创建离散动作分布
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
        salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
        num_groups_dist = Categorical(logits=num_groups_logits_masked)
        inter_interval_dist = Categorical(logits=inter_interval_logits_masked)

        # 打包所有分布
        distributions = {
            'continuous': continuous_base_dist,
            'trigger': trigger_dist,
            'salvo_size': salvo_size_dist,
            'num_groups': num_groups_dist,
            'inter_interval': inter_interval_dist
        }

        return distributions, new_hidden


# ==============================================================================
#           完整新架构: Attention -> MLP -> GRU -> MLP -> Head
# ==============================================================================

class Critic_GRU(Module):
    """
    Critic 网络 - [完整版 V2: Attention -> MLP -> GRU -> MLP]
    结构为: 跨特征注意力 -> 共享MLP特征提取 -> GRU 序列处理 -> MLP -> 输出头。
    """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        """初始化 Critic 网络的模块、权重和优化器。"""
        super(Critic_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay
        self.embedding_dim = 1

        assert self.embedding_dim % ATTN_NUM_HEADS == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"

        # --- 1. 特征级自注意力层 ---
        self.attention = MultiheadAttention(embed_dim=self.embedding_dim,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)

        # --- 2. 定义 GRU 之前的共享 MLP (Pre-GRU MLP) ---
        pre_gru_mlp_input_dim = self.input_dim * self.embedding_dim
        pre_gru_mlp_layers = 2
        pre_gru_dims = CRITIC_PARA.model_layer_dim[:pre_gru_mlp_layers]

        self.pre_gru_mlp = Sequential()
        mlp_input_dim = pre_gru_mlp_input_dim
        for i, dim in enumerate(pre_gru_dims):
            self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(mlp_input_dim, dim))
            self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
            mlp_input_dim = dim

        pre_gru_output_dim = pre_gru_dims[-1] if pre_gru_dims else pre_gru_mlp_input_dim

        # --- 3. GRU 时序建模层 ---
        self.gru = GRU(pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)

        # --- 4. 定义 GRU 之后的 MLP (Post-GRU MLP) ---
        post_gru_dims = CRITIC_PARA.model_layer_dim[pre_gru_mlp_layers:]
        self.post_gru_mlp = Sequential()
        mlp_input_dim = self.rnn_hidden_size
        for i, dim in enumerate(post_gru_dims):
            self.post_gru_mlp.add_module(f'post_gru_fc_{i}', Linear(mlp_input_dim, dim))
            self.post_gru_mlp.add_module(f'post_gru_leakyrelu_{i}', LeakyReLU())
            mlp_input_dim = dim

        post_gru_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size

        # --- 5. 最终的输出头 (Head) ---
        self.fc_out = Linear(post_gru_output_dim, self.output_dim)

        # --- 6. 优化器设置 ---
        gru_params, attention_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'gru' in name.lower():
                gru_params.append(param)
            elif any(key in name.lower() for key in ['attention', 'attn']):
                attention_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
            {'params': attention_params, 'lr': CRITIC_PARA.attention_lr},
            {'params': other_params, 'lr': CRITIC_PARA.lr}
        ]

        self.optim = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)

    def forward(self, obs, hidden_state):
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)
        B, S, D = obs_tensor.shape

        # --- 阶段一: 特征交互 (Attention) ---
        feat_tokens = obs_tensor.contiguous().view(B * S, D, 1)
        token_embeds = feat_tokens
        attn_out, _ = self.attention(token_embeds, token_embeds, token_embeds)
        token_context = token_embeds + attn_out
        flattened_context = token_context.contiguous().view(B * S, -1)
        contextualized_sequence = flattened_context.view(B, S, -1)

        # --- 阶段二: 特征提取 (Pre-GRU MLP) ---
        mlp_sequence = self.pre_gru_mlp(contextualized_sequence)

        # --- 阶段三: 时序建模 (GRU) ---
        gru_out, new_hidden = self.gru(mlp_sequence, hidden_state)

        # --- 阶段四: 价值评估 (Post-GRU MLP) ---
        post_gru_features = self.post_gru_mlp(gru_out)
        value = self.fc_out(post_gru_features)

        if not is_sequence:
            value = value.squeeze(1)

        return value, new_hidden



# ==============================================================================
# PPO Agent 主类
# ==============================================================================

class PPO_continuous(object):
    """
    PPO 智能体主类。
    通过 `use_rnn` 标志来决定是使用 MLP 模型还是 GRU+Attention 模型。
    """

    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = False):
        """
        初始化PPO智能体。
        :param load_able: 是否加载预训练模型。
        :param model_dir_path: 预训练模型的文件夹路径。
        :param use_rnn: 是否使用基于 GRU+Attention 的模型。
        """
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
        self.gamma = AGENTPARA.gamma  # 折扣因子
        self.gae_lambda = AGENTPARA.lamda  # GAE 的 lambda 参数
        self.ppo_epoch = AGENTPARA.ppo_epoch  # 每次学习时，数据被重复使用的次数
        self.total_steps = 0
        # 为本次训练创建一个带时间戳的存档文件夹名
        self.training_start_time = time.strftime("PPO_ATT_GRU_%Y-%m-%d_%H-%M-%S")  # <<< 修改 >>> 更新存档文件夹名称
        self.base_save_dir = "../../../../save/save_evade_fuza"
        win_rate_subdir = "胜率模型"
        # 拼接成完整的存档路径
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)

        # 如果需要加载预训练模型
        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../../../../test/test_evade")

    def load_models_from_directory(self, directory_path: str):
        """从指定目录加载 Actor 和 Critic 模型的权重。"""
        # 检查路径是否存在且为文件夹
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
                    # 加载权重到模型
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
        # 线性插值公式: low + (tanh_out + 1) / 2 * (high - low)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)
        return scaled_action

    def get_initial_hidden_states(self, batch_size=1):
        """为 GRU 模型生成初始的零隐藏状态。"""
        if not self.use_rnn:
            return None, None
        # GRU 隐藏状态的形状是 (num_layers, batch_size, hidden_size)
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
        :param state: 当前状态观测值。
        :param actor_hidden: Actor GRU 的隐藏状态。
        :param critic_hidden: Critic GRU 的隐藏状态。
        :param deterministic: 如果为 True，则选择最可能的动作（用于评估），否则进行随机采样（用于训练）。
        :return: (环境动作, 存储动作, 对数概率, 状态价值, 新的actor隐藏状态, 新的critic隐藏状态)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        # 检查输入是否是批处理数据
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():  # 在此块中不计算梯度，以提高性能
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
            # u 是正态分布的原始样本
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            # tanh(u) 是为了限制范围并引入非线性，这是 SAC 和 TQC 等算法中常用的技巧
            action_cont_tanh = torch.tanh(u)
            # 计算连续动作的对数概率
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 从所有离散动作分布中采样
            sampled_actions_dict = {}
            for key, dist in dists.items():
                if key == 'continuous': continue
                if deterministic:  # 确定性模式
                    if isinstance(dist, Categorical):  # 分类分布：取概率最大的类别
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                    else:  # 伯努利分布：取概率大于0.5的
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                else:  # 随机模式
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
            # 获取下一个状态的价值，如果是最后一步，则价值为 0
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
        执行 PPO 的学习和更新步骤。已适配 RNN 模式。
        """
        # 如果 Buffer 中的数据不足一个批次，则跳过学习
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            print(
                f"  [Info] Buffer size ({self.buffer.get_buffer_size()}) < batch size ({BUFFERPARA.BATCH_SIZE}). Skipping.")
            return None
        # 1. 数据准备
        states, values, actions, old_probs, rewards, dones, _, __ = self.buffer.get_all_data()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        values = np.squeeze(values)  # 移除多余维度，比如 (N,1) → (N,) 确保 values 是一维数组，与 advantages 对齐

        # returns (即 G_t，目标价值) 是优势函数 + 状态价值   保证 returns 与 values 形状一致
        # returns (即 G_t) 是优势函数的 unbiased estimator
        returns = advantages + values
        # 用于记录训练过程中的各种损失和指标
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

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

                # 从批次动作中解析出连续和离散部分 💥 不再需要切片出最后一个时间步，直接使用完整的序列
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),  # 类别索引需要是 long 类型
                    # 'intra_interval': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # 4. Actor (策略) 网络训练
                if self.use_rnn:
                    new_dists, _ = self.Actor(state, initial_actor_h)
                else:
                    new_dists = self.Actor(state)

                # 计算新策略下，旧动作的对数概率
                # 现在 u_from_buffer 和 new_dists['continuous'] 的形状都是 (B, 4)，可以匹配了
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
                # 使用完整的序列 old_prob 和 advantage
                # 注意：需要确保它们的形状与 new_prob 匹配
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))  # clamp 防止数值溢出

                # 确保 advantage 和 ratio 的形状可以广播  advantage可能是 (B, S, 1) 或 (B, S)，需要与 ratio (B, S) 对齐
                advantage_squeezed = advantage.squeeze(-1) if advantage.dim() > ratio.dim() else advantage

                # PPO 的裁剪目标函数
                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed

                # Actor 的损失是裁剪后的目标函数的负值，加上熵的正则项
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 反向传播和优化
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
                self.Actor.optim.step()

                # 5. Critic (价值) 网络训练
                if self.use_rnn:
                    # 对于 RNN 模型，Critic 输出的是聚合后的单一价值
                    # new_value shape: (B, 1)
                    new_value, _ = self.Critic(state, initial_critic_h)
                else:
                    # new_value shape: (B, 1)
                    new_value = self.Critic(state)

                # 确保目标值 `return_` 和网络输出 `new_value` 的维度一致，以避免 PyTorch 的广播警告。
                if new_value.dim() > return_.dim():
                    return_ = return_.unsqueeze(-1)

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
        # # 在完成整个批次数据的学习后，让调度器步进一次
        # self.Actor.actor_scheduler.step()
        # self.Critic.critic_scheduler.step()

        # 6. 清理和返回
        self.buffer.clear_memory()  # 完成一轮学习后清空 Buffer
        # 计算整个 epoch 的平均指标
        for key in train_info:
            train_info[key] = np.mean(train_info[key])

        self.save()  # 保存模型
        return train_info

    def prep_training_rl(self):
        """将 Actor 和 Critic 网络设置为训练模式。"""
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        """将 Actor 和 Critic 网络设置为评估模式（例如，关闭 Dropout）。"""
        self.Actor.eval()
        self.Critic.eval()

    def save(self, prefix=""):
        """
        保存 Actor 和 Critic 模型的权重。
        - 如果提供了 prefix，则保存到 '胜率模型' 子目录中。
        - 否则，保存到主运行目录中。
        """
        # --- 3. 初始化时创建所有需要的目录 (推荐做法) ---
        try:
            os.makedirs(self.run_save_dir, exist_ok=True)
            os.makedirs(self.win_rate_dir, exist_ok=True)
            print(f"训练存档目录已创建: {self.run_save_dir}")
        except Exception as e:
            print(f"创建存档目录失败: {e}")

        # --- 1. 根据 prefix 确定目标保存目录 (逻辑更清晰) ---
        if prefix:
            target_dir = self.win_rate_dir
            print(f"胜率模型将被保存至: {target_dir}")
        else:
            target_dir = self.run_save_dir
            print(f"常规模型将被保存至: {target_dir}")

        # --- 2. 循环保存模型 (代码无重复) ---
        for net_name in ['Actor', 'Critic']:
            try:
                # 获取模型对象
                net_model = getattr(self, net_name)

                # 构造文件名
                filename = f"{prefix}_{net_name}.pkl" if prefix else f"{net_name}.pkl"
                full_path = os.path.join(target_dir, filename)

                # 保存模型的状态字典
                torch.save(net_model.state_dict(), full_path)
                print(f"  - {filename} 保存成功。")

            except AttributeError:
                print(f"  - 错误: 找不到名为 '{net_name}' 的模型。")
            except Exception as e:
                print(f"  - 保存模型 {net_name} 到 {full_path} 时发生错误: {e}")

    # def save(self, prefix=""):
    #     """保存 Actor 和 Critic 模型的权重。"""
    #     try:
    #         # 确保保存目录存在
    #         os.makedirs(self.run_save_dir, exist_ok=True)
    #         print(f"模型将被保存至: {self.run_save_dir}")
    #     except Exception as e:
    #         print(f"创建模型文件夹 {self.run_save_dir} 失败: {e}")
    #         return
    #     # 分别保存 Actor 和 Critic
    #     for net in ['Actor', 'Critic']:
    #         try:
    #             # 构造文件名
    #             # filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"
    #             # full_path = os.path.join(self.run_save_dir, filename)
    #             if prefix:
    #                 filename = f"{prefix}_{net}.pkl"
    #                 full_path = os.path.join(self.run_save_dir2, filename)
    #             else:
    #                 filename = f"{net}.pkl"
    #                 full_path = os.path.join(self.run_save_dir, filename)
    #             # 保存模型的状态字典
    #             torch.save(getattr(self, net).state_dict(), full_path)
    #             print(f"  - {filename} 保存成功。")
    #         except Exception as e:
    #             print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")