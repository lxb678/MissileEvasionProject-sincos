# --- START OF FILE Hybrid_PPO_jsbsim_SeparateHeads.py ---

import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
# 导入配置文件，其中包含各种超参数
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigGRU import *
# 导入支持 GRU 的经验回放池
from Interference_code.PPO_model.PPO_evasion_fuza.BufferGRU import *
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

# <<< GRU/RNN 修改 >>>: 新增 RNN 配置
# 这些参数最好也移到 Config.py 中
RNN_HIDDEN_SIZE = 128 #64 #128 #64  #128 #64 #128 #64 #9 #9 #32 #9  # GRU 层的隐藏单元数量
SEQUENCE_LENGTH =  5 #10 #5 #8 #2 #5 #15   # 训练时从经验池中采样的连续轨迹片段的长度


class Actor_GRU(Module):
    """
    Actor 网络 (策略网络) - [架构: 共享MLP -> GRU (+残差) -> 塔楼MLP]
    严格保持原配置层数：
    1. Pre-GRU: 使用 model_layer_dim 的前2层
    2. GRU: 处理时序
    3. Residual: LayerNorm(Pre-GRU输出 + GRU输出)
    4. Post-GRU: 使用 model_layer_dim 的剩余层
    """

    def __init__(self):
        super(Actor_GRU, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # self.log_std_min = -20.0
        # self.log_std_max = 2.0

        # # 1. 设定物理意义上的标准差范围
        # # =====================================================
        # # 下限 0.05: 保持 5% 的底噪，防止过拟合，增加策略鲁棒性
        # self.target_std_min = 0.01 #0.05
        # # 上限 1.5: 覆盖 Tanh 有效区，防止过多无效探索
        # self.target_std_max = 1.5
        #
        # # 自动转换为 log 空间 (因为网络参数训练 log 值更稳定)
        # # log(0.01) ≈ -4.605, log(0.6) ≈ -0.51
        # self.log_std_min = np.log(self.target_std_min)
        # self.log_std_max = np.log(self.target_std_max)

        self.target_std_min = 0.10 #0.20 #0.10 #0.05  # 保证底噪
        self.target_std_max = 0.70 #0.80  # 0.90 #0.70 #0.80  # 降低上限，避免完全随机
        self.target_init_std = 0.60 #0.75  # 0.85 #0.65 #0.75  # 初始值设为中间态，不要设为 max

        # 转换为 Log 空间边界
        self.log_std_min = np.log(self.target_std_min)  # ln(0.05) ≈ -2.99
        self.log_std_max = np.log(self.target_std_max)  # ln(1.0) = 0.0

        self.rnn_hidden_size = RNN_HIDDEN_SIZE

        # =====================================================================
        # 1. 共享 MLP (Pre-GRU) - 严格使用配置的前2层
        # =====================================================================
        split_point = 2  # 在第2层切分
        pre_gru_dims = ACTOR_PARA.model_layer_dim[:split_point]

        self.pre_gru_mlp = Sequential()
        input_dim = self.input_dim
        for i, dim in enumerate(pre_gru_dims):
            self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(input_dim, dim))
            self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
            input_dim = dim

        # 记录 Pre-GRU 的输出维度
        self.pre_gru_output_dim = input_dim

        # =====================================================================
        # 2. GRU 层
        # =====================================================================
        self.gru = GRU(self.pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)
        # # 3. [修正] 归一化层 (必须启用)
        # # 用于 GRU 输入前，防止 MLP 输出数值过大
        # self.pre_gru_norm = nn.LayerNorm(self.pre_gru_output_dim)
        # # 用于残差连接后，稳定后续 MLP 的输入
        # self.post_residual_norm = nn.LayerNorm(self.rnn_hidden_size)

        # # =====================================================================
        # # 3. 残差连接适配器 & 归一化
        # # =====================================================================
        # # 如果 Pre-GRU 输出维度 != GRU 隐藏层维度，需要投影才能相加
        # if self.pre_gru_output_dim != self.rnn_hidden_size:
        #     self.residual_projection = Linear(self.pre_gru_output_dim, self.rnn_hidden_size)
        # else:
        #     self.residual_projection = nn.Identity()

        # self.ln_residual = LayerNorm(self.rnn_hidden_size)

        # =====================================================================
        # 4. 专用 MLP 塔楼 (Post-GRU) - 使用配置的剩余层
        # =====================================================================
        post_gru_dims = ACTOR_PARA.model_layer_dim[split_point:]

        # 🔥 [修改点 1]：塔楼的输入维度 = GRU输出维度 + GRU输入特征维度
        # 这样实现了 Skip Connection 的拼接
        tower_input_dim = self.rnn_hidden_size + self.pre_gru_output_dim
        # self.layer_norm = nn.LayerNorm(tower_input_dim)

        # self.post_concat_norm = nn.LayerNorm(tower_input_dim)
        # self.tower_proj = nn.Linear(tower_input_dim, self.rnn_hidden_size)  # 投影回 128（或你的hidden）
        # tower_input_dim = self.rnn_hidden_size

        # 连续动作塔楼
        self.continuous_tower = Sequential()
        # 注意：这里需要一个临时变量 current_dim 来构建塔楼，因为 tower_input_dim 在循环中会变
        current_dim = tower_input_dim
        for i, dim in enumerate(post_gru_dims):
            self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(current_dim, dim))
            self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
            current_dim = dim
        continuous_tower_output_dim = post_gru_dims[-1] if post_gru_dims else tower_input_dim

        # 离散动作塔楼
        self.discrete_tower = Sequential()
        current_dim = tower_input_dim  # 重置维度
        for i, dim in enumerate(post_gru_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(current_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
            current_dim = dim
        discrete_tower_output_dim = post_gru_dims[-1] if post_gru_dims else tower_input_dim

        # =====================================================================
        # 5. 输出头
        # =====================================================================
        self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
        # self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))
        # =========== 修改: 初始化 Std 参数 ===========
        # 初始化为 log_std_max (即 std=0.6)，让智能体刚开始时有最大的探索能力
        # self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), self.log_std_max))
        # =====================================================
        # 2. 软限制参数初始化
        # =====================================================
        # 计算验证：
        # Sigmoid(2.0) ≈ 0.88
        # LogStd ≈ ln(0.05) + 0.88 * (ln(1.5) - ln(0.05)) ≈ 0.0
        # Std ≈ 1.0 (完美初始值)

        # init_value = 2.5 #2.0
        # self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), init_value))

        # 初始化为 -0.5 左右 (std ≈ 0.6)，比 1.0 稳健，又比 0.1 有探索性
        init_log_std = np.log(self.target_init_std)
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), init_log_std))

        # # --- [新增] 定义归一化层 ---
        # # 1. 用于进入 GRU 前，稳定 MLP 提取的特征
        # self.pre_gru_norm = nn.LayerNorm(self.pre_gru_output_dim)
        #
        # # # 2. 用于残差拼接后，稳定进入塔楼的混合特征
        # # # 维度是：GRU隐藏层维度 + Pre-GRU输出维度
        # # self.post_concat_norm = nn.LayerNorm(self.rnn_hidden_size + self.pre_gru_output_dim)

        # 初始化
        # self.apply(init_weights)
        self._init_weights()  # 必须进行权重初始化
        # 优化器设置
        gru_params = list(self.gru.parameters())
        other_params = (
                list(self.pre_gru_mlp.parameters()) +
                list(self.continuous_tower.parameters()) +
                list(self.discrete_tower.parameters()) +
                list(self.mu_head.parameters()) +
                list(self.discrete_head.parameters()) +
                [self.log_std_param]
        )
        # if isinstance(self.residual_projection, Linear):
        #     other_params.extend(list(self.residual_projection.parameters()))
        # other_params.extend(list(self.ln_residual.parameters()))

        param_groups = [
            {'params': gru_params, 'lr': ACTOR_PARA.gru_lr},
            {'params': other_params, 'lr': ACTOR_PARA.lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        self.actor_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                     end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
                                                     total_iters=AGENTPARA.MAX_EXE_NUM)
        self.to(ACTOR_PARA.device)

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

    def forward(self, obs, hidden_state):
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)

        # 1. 共享 MLP 特征提取
        # Input: (B, S, Input_Dim) -> Output: (B, S, Pre_Dim)
        features = self.pre_gru_mlp(obs_tensor)

        # 2. [修正] GRU 前归一化 (Pre-Norm) - 启用！
        # 这一步保证进入 GRU 的数据分布是稳定的
        # features_normed = self.pre_gru_norm(features)

        # 2. GRU 时序记忆
        # Input: (B, S, Pre_Dim) -> Output: (B, S, RNN_Hidden)
        gru_out, new_hidden = self.gru(features, hidden_state)

        # # 3. 🔥 残差连接 + LayerNorm 🔥
        # combined_features = gru_out + features
        # 再次归一化 (非常重要，否则值会越加越大)
        # combined_features = self.post_residual_norm(combined_features)

        combined_features = torch.cat([features, gru_out], dim=-1)  # skip-concat
        # combined_features = self.post_concat_norm(combined_features)  # 稳定
        # combined_features = self.tower_proj(combined_features)  # 回到 hidden 维

        # # 3. 🔥 [修改点 2] 拼接 (Concatenation) 🔥
        # # 将 "当前时刻特征(features)" 和 "历史记忆(gru_out)" 在最后一个维度拼接
        # # features shape: (B, S, pre_gru_dim)
        # # gru_out shape:  (B, S, rnn_hidden_size)
        # combined_features = torch.cat([features, gru_out], dim=-1)
        #
        # # --- [新增] 归一化位置 B (最关键！) ---
        # combined_features = self.post_concat_norm(combined_features)
        # # combined_features = self.layer_norm(combined_features)
        # # combined_features =  gru_out

        # result shape:   (B, S, pre_gru_dim + rnn_hidden_size)

        # 4. 塔楼处理
        continuous_features = self.continuous_tower(combined_features)
        discrete_features = self.discrete_tower(combined_features)

        # continuous_features = self.continuous_tower(gru_out)
        # discrete_features = self.discrete_tower(gru_out)

        # 5. 单步处理适配
        if not is_sequence:
            continuous_features = continuous_features.squeeze(1)
            discrete_features = discrete_features.squeeze(1)

        # 6. 输出头
        mu = self.mu_head(continuous_features)

        # 强行把均值限制在 [-2, 2] 或 [-3, 3] 之间
        # 只要不让它跑到 10 这种离谱的值就行
        mu = torch.clamp(mu, -2.0, 2.0)
        # # =====================================================
        # # 1. 连续动作均值 mu: 使用 Tanh 替代 Clamp 防止梯度死亡
        # # =====================================================
        # # 原始的 mu_head 输出通过 tanh 压缩到 [-1, 1]，再放大到 [-3, 3]
        # mu_raw = self.mu_head(continuous_features)
        # mu = torch.tanh(mu_raw) * 3.0  # 强行且平滑地把均值限制在 [-3, 3] 之间

        all_disc_logits = self.discrete_head(discrete_features)

        # --- 以下逻辑保持不变 ---
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        has_flares_info = obs_tensor[..., 10]  # 请确保索引对应正确
        # mask = (has_flares_info == 0)
        # <<< 修改后 (正确) >>>
        # 因为环境归一化后，0发对应 -1.0。考虑到浮点数误差，我们判断是否小于 -0.99
        mask = (has_flares_info <= -0.99)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min
        # 离散动作的 Logits (直接克隆，不再做基于概率的干预)
        salvo_size_logits_masked = salvo_size_logits.clone()
        num_groups_logits_masked = num_groups_logits.clone()
        inter_interval_logits_masked = inter_interval_logits.clone()

        # trigger_probs = torch.sigmoid(trigger_logits_masked)
        # no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)
        # if torch.any(no_trigger_mask):
        #     INF = 1e6
        #     NEG_INF = -1e6
        #     for logits_tensor in [salvo_size_logits_masked, num_groups_logits_masked, inter_interval_logits_masked]:
        #         logits_sub = logits_tensor[no_trigger_mask]
        #         if logits_sub.numel() > 0:
        #             logits_sub[:] = NEG_INF
        #             logits_sub[:, 0] = INF
        #             logits_tensor[no_trigger_mask] = logits_sub

        # log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        # =====================================================
        # 3. 计算动态标准差 (Soft Mapping)
        # =====================================================
        # output = min + (max - min) * sigmoid(param)

        # # 1. 将无界参数压缩到 (0, 1)
        # norm_val = torch.sigmoid(self.log_std_param)
        #
        # # 2. 映射到 log 范围 [log_min, log_max]
        # log_std = self.log_std_min + norm_val *  (self.log_std_max - self.log_std_min)

        # # =========== 修改: 限制标准差应用 ===========
        # # 使用之前计算好的 log 界限进行截断
        # # log_std_min = ln(0.01), log_std_max = ln(0.6)
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)

        # 3. 转回 std
        std = torch.exp(log_std).expand_as(mu)
        continuous_base_dist = Normal(mu, std)

        # # =====================================================
        # # 2. 连续动作标准差 log_std: 使用可导的 Soft Mapping
        # # =====================================================
        # # 1. 将无界的网络参数自适应地平滑压缩到 (0, 1) 区间
        # norm_val = torch.sigmoid(self.log_std_param)
        #
        # # 2. 映射到 log 范围 [log_min, log_max]，全程保持可导，永不卡死
        # log_std = self.log_std_min + norm_val * (self.log_std_max - self.log_std_min)
        #
        # # 3. 转回 std
        # std = torch.exp(log_std).expand_as(mu)
        #
        # # =====================================================
        # # 3. 构建所有的概率分布
        # # =====================================================
        # continuous_base_dist = Normal(mu, std)

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

        return distributions, new_hidden


class Critic_GRU(Module):
    """
    Critic 网络 (价值网络) - [架构: 共享MLP -> GRU (+残差) -> 后置MLP]
    严格保持原配置层数。
    """

    def __init__(self):
        super(Critic_GRU, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.rnn_hidden_size = RNN_HIDDEN_SIZE

        # =====================================================================
        # 1. 共享 MLP (Pre-GRU)
        # =====================================================================
        split_point = 2
        pre_gru_dims = CRITIC_PARA.model_layer_dim[:split_point]

        self.pre_gru_mlp = Sequential()
        input_dim = self.input_dim
        for i, dim in enumerate(pre_gru_dims):
            self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(input_dim, dim))
            self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
            input_dim = dim

        self.pre_gru_output_dim = input_dim

        # =====================================================================
        # 2. GRU 层
        # =====================================================================
        self.gru = GRU(self.pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)
        # # 3. [修正] 归一化层 (启用)
        # self.pre_gru_norm = nn.LayerNorm(self.pre_gru_output_dim)
        # self.post_residual_norm = nn.LayerNorm(self.rnn_hidden_size)

        # # =====================================================================
        # # 3. 残差连接适配器 & 归一化
        # # =====================================================================
        # if self.pre_gru_output_dim != self.rnn_hidden_size:
        #     self.residual_projection = Linear(self.pre_gru_output_dim, self.rnn_hidden_size)
        # else:
        #     self.residual_projection = nn.Identity()

        # self.ln_residual = LayerNorm(self.rnn_hidden_size)

        # =====================================================================
        # 4. 后置 MLP (Post-GRU)
        # =====================================================================
        post_gru_dims = CRITIC_PARA.model_layer_dim[split_point:]

        # 🔥 [修改点 1]：输入维度变为拼接后的维度
        tower_input_dim = self.rnn_hidden_size + self.pre_gru_output_dim
        # self.layer_norm = nn.LayerNorm(tower_input_dim)
        # self.post_concat_norm = nn.LayerNorm(tower_input_dim)
        # self.tower_proj = nn.Linear(tower_input_dim, self.rnn_hidden_size)  # 投影回 128（或你的hidden）
        # tower_input_dim = self.rnn_hidden_size

        self.post_gru_mlp = Sequential()
        current_dim = tower_input_dim
        for i, dim in enumerate(post_gru_dims):
            self.post_gru_mlp.add_module(f'post_gru_fc_{i}', Linear(current_dim, dim))
            self.post_gru_mlp.add_module(f'post_gru_leakyrelu_{i}', LeakyReLU())
            current_dim = dim

        post_gru_output_dim = post_gru_dims[-1] if post_gru_dims else tower_input_dim

        # =====================================================================
        # 5. 输出头
        # =====================================================================
        self.fc_out = Linear(post_gru_output_dim, self.output_dim)

        # # --- [新增] 定义归一化层 ---
        # self.pre_gru_norm = nn.LayerNorm(self.pre_gru_output_dim)
        # # self.post_concat_norm = nn.LayerNorm(self.rnn_hidden_size + self.pre_gru_output_dim)

        # 初始化
        # self.apply(init_weights)
        self._init_weights()

        # 优化器
        gru_params = list(self.gru.parameters())
        other_params = (
                list(self.pre_gru_mlp.parameters()) +
                list(self.post_gru_mlp.parameters()) +
                list(self.fc_out.parameters())
        )
        # if isinstance(self.residual_projection, Linear):
        #     other_params.extend(list(self.residual_projection.parameters()))
        # other_params.extend(list(self.ln_residual.parameters()))

        param_groups = [
            {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
            {'params': other_params, 'lr': CRITIC_PARA.lr}
        ]
        self.optim = torch.optim.Adam(param_groups)
        self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
                                                      end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
                                                      total_iters=AGENTPARA.MAX_EXE_NUM)
        self.to(CRITIC_PARA.device)

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

    def forward(self, obs, hidden_state):
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)

        # 1. Pre-GRU
        features = self.pre_gru_mlp(obs_tensor)

        # 2. [修正] Pre-Norm (启用)
        # features_normed = self.pre_gru_norm(features)

        # 2. GRU
        gru_out, new_hidden = self.gru(features, hidden_state)

        # # 3. 🔥 Residual 🔥
        # # features_projected = self.residual_projection(features)
        # # combined_features = gru_out
        # # 3. 🔥 [修改点 2] 拼接 (Skip Connection) 🔥
        combined_features = torch.cat([features, gru_out], dim=-1)
        # # --- [新增] 归一化位置 B ---
        # combined_features = self.post_concat_norm(combined_features)
        # # combined_features = self.layer_norm(combined_features)
        # # combined_features = gru_out

        # 3. 🔥 [新增] 残差相加 🔥
        # combined_features = gru_out + features
        # 归一化
        # combined_features = self.post_residual_norm(combined_features)

        # combined_features = torch.cat([features, gru_out], dim=-1)  # skip-concat
        # combined_features = self.post_concat_norm(combined_features)  # 稳定
        # combined_features = self.tower_proj(combined_features)  # 回到 hidden 维

        # 4. Post-GRU
        post_features = self.post_gru_mlp(combined_features)



        # post_features = self.post_gru_mlp(gru_out)

        if not is_sequence:
            post_features = post_features.squeeze(1)

        # 5. Head
        value = self.fc_out(post_features)

        return value, new_hidden

# ==============================================================================
# Original MLP-based Actor and Critic (保留原始版本以供选择)
# ==============================================================================

class Actor(Module):
    # ... 原版 Actor 代码保持不变 ...
    """
       Actor 网络 (策略网络) - 基于 MLP（多层感知机）的版本。
       该网络负责根据当前状态决定要采取的动作策略。
       它具有一个共享的骨干网络，后接两个独立的头部，分别处理连续动作和离散动作。
        [💥 修改] 连续动作部分：mu 头状态依赖，log_std 为全局可学习参数。
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
            # self.shared_network.add_module(f'LayerNorm_{i}', LayerNorm(shared_layers_dims[i + 1]))
            # 添加 LeakyReLU 激活函数，以避免梯度消失问题
            self.shared_network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # 骨干网络的输出维度，将作为各个头部的输入
        shared_output_dim = ACTOR_PARA.model_layer_dim[-1]
        # --- 独立头部网络 ---
        # # 1. 连续动作头部：输出高斯分布的均值 (mu) 和对数标准差 (log_std)，每个连续动作维度都需要这两个参数
        # self.continuous_head = Linear(shared_output_dim, CONTINUOUS_DIM * 2)
        # --- 💥 [修改] 独立头部网络 ---
        # mu 头部
        self.mu_head = Linear(shared_output_dim, CONTINUOUS_DIM)
        # log_std 作为独立的、与状态无关的可学习参数
        self.log_std_param = torch.nn.Parameter(torch.zeros(1, CONTINUOUS_DIM) * -0.5)

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
        # cont_params = self.continuous_head(shared_features) # 获取连续动作的参数
        # --- 💥 [修改] 从不同头获取参数 ---
        mu = self.mu_head(shared_features)
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
        # # 从连续动作参数中分离出均值和对数标准差
        # mu, log_std = cont_params.chunk(2, dim=-1)
        # 裁剪对数标准差以保证数值稳定性
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        # # 计算标准差
        # std = torch.exp(log_std)

        # --- 💥 [修改] 创建连续动作分布 ---
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
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
# <<< GRU/RNN 修改 >>>: 定义新的基于 GRU 的 Actor 和 Critic
#                       [💥 新结构: GRU -> MLP -> Heads]
# ==============================================================================

def init_weights(m, gain=1.0):
    """
    一个通用的权重初始化函数。
    :param m: PyTorch module
    :param gain: 正交初始化的增益因子
    """
    # if isinstance(m, Linear):
    #     # 对线性层使用 Kaiming Normal 初始化，适用于 LeakyReLU
    #     torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
    #     if m.bias is not None:
    #         torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, GRU):
        # 对 GRU 的权重使用正交初始化，这是 RNN 的最佳实践
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=gain)
            elif 'bias' in name:
                param.data.fill_(0)
# # ==============================================================================
# # <<< GRU/RNN 修改 >>>: 定义新的基于 GRU 的 Actor 和 Critic
# #                       [💥 新结构: GRU -> MLP -> Heads]
# # ==============================================================================
#
# class Actor_GRU(Module):
#     """
#         Actor 网络 (策略网络) - 基于 GRU 的版本。
#         结构为: MLP 特征提取 -> GRU 序列处理 -> 独立动作头。
#         这种结构能够捕捉状态序列中的时间依赖关系。
#         [💥 修改] 连续动作部分：mu 头状态依赖，log_std 为全局可学习参数。
#         """
#     """
#     Actor 网络 (策略网络) - [最终混合架构: GRU -> Hybrid MLP]
#     结构为: GRU 序列处理 -> 共享MLP基座 -> 专用MLP塔楼 -> 独立动作头。
#     """
#     def __init__(self):
#         super(Actor_GRU, self).__init__()
#         self.input_dim = ACTOR_PARA.input_dim
#         self.log_std_min = -20.0
#         self.log_std_max = 2.0
#         self.rnn_hidden_size =  RNN_HIDDEN_SIZE #RNN_HIDDEN_SIZE  self.input_dim
#
#         # 1. GRU 层作为第一层，直接处理原始状态输入
#         self.gru = GRU(self.input_dim, self.rnn_hidden_size, batch_first=True)
#
#         # --- 移植过来的混合架构定义 ---
#         # 这个混合架构现在处理的是 GRU 的输出，而不是原始输入
#         # 2. 定义 MLP 各部分的维度
#         #    假设 model_layer_dim = [256, 256，256], split_point = 1
#         split_point = 2  # 在 MLP 的第2层后拆分
#         base_dims = ACTOR_PARA.model_layer_dim[:split_point]  # 例如: [256,256]
#         continuous_tower_dims = ACTOR_PARA.model_layer_dim[split_point:]  # 例如: [256]
#         discrete_tower_dims = continuous_tower_dims
#         # 让离散塔楼的维度是连续塔楼的一半
#         # discrete_tower_dims = [dim // 2 for dim in continuous_tower_dims]  # 例如: [128]
#
#         # 3. 构建共享MLP基座 (Shared Base MLP)
#         self.shared_base_mlp = Sequential()
#         # MLP的输入维度是 GRU 的隐藏层大小
#         base_input_dim = self.rnn_hidden_size
#         for i, dim in enumerate(base_dims):
#             self.shared_base_mlp.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
#             self.shared_base_mlp.add_module(f'base_leakyrelu_{i}', LeakyReLU())
#             base_input_dim = dim
#         base_output_dim = base_dims[-1] if base_dims else self.rnn_hidden_size
#
#         # 4. 构建连续动作塔楼 (Continuous Tower)
#         self.continuous_tower = Sequential()
#         tower_input_dim = base_output_dim
#         for i, dim in enumerate(continuous_tower_dims):
#             self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         continuous_tower_output_dim = continuous_tower_dims[-1] if continuous_tower_dims else base_output_dim
#
#         # 5. 构建离散动作塔楼 (Discrete Tower)
#         self.discrete_tower = Sequential()
#         tower_input_dim = base_output_dim
#         for i, dim in enumerate(discrete_tower_dims):
#             self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         discrete_tower_output_dim = discrete_tower_dims[-1] if discrete_tower_dims else base_output_dim
#
#         # 6. 定义最终的输出头 (Heads)
#         self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
#         self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
#         self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))
#
#         # <<< MODIFICATION START: 精细化优化器设置 >>>
#         # 1. 将参数分为 GRU 参数 和 其他参数
#         gru_params = []
#         other_params = []
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue
#             # 根据参数名中是否包含 'gru' 来进行分组
#             if 'gru' in name.lower():
#                 gru_params.append(param)
#             else:
#                 other_params.append(param)
#
#         # 2. 创建参数组 (parameter groups) 列表
#         param_groups = [
#             {
#                 'params': gru_params,
#                 'lr': ACTOR_PARA.gru_lr  # 为 GRU 参数设置专属学习率
#             },
#             {
#                 'params': other_params  # 其他所有参数 (MLP, Heads)
#                 # 不指定 lr，将使用下面 Adam 构造函数中的默认 lr
#             }
#         ]
#
#         # 3. 使用参数组初始化优化器
#         # 默认 lr 将用于 'other_params' 组
#         self.optim = torch.optim.Adam(param_groups, lr=ACTOR_PARA.lr)
#
#         print("--- Actor_GRU Optimizer Initialized with Parameter Groups ---")
#         print(f"  - GRU Params LR: {ACTOR_PARA.gru_lr}")
#         print(f"  - Other (MLP) Params LR: {ACTOR_PARA.lr}")
#         # <<< MODIFICATION END >>>
#
#         # 优化器等设置
#         # self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
#         self.actor_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
#                                                      end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
#                                                      total_iters=AGENTPARA.MAX_EXE_NUM)
#         self.to(ACTOR_PARA.device)
#
#     def forward(self, obs, hidden_state):
#         """
#         GRU Actor 的前向传播。
#         这个方法被设计为可以同时处理单个时间步的输入（用于与环境交互）和序列输入（用于训练）。
#         Args:
#             obs (Tensor): 观测值。形状可以是 (batch, features) 用于单步，或 (batch, seq_len, features) 用于序列。
#             hidden_state (Tensor): GRU 的隐藏状态。形状是 (num_layers=1, batch, rnn_hidden_size)。
#         Returns:
#             tuple: (包含所有动作分布的字典, 新的隐藏状态)
#         """
#         obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
#         # 检查输入是单个时间步还是序列，通过判断张量的维度
#         is_sequence = obs_tensor.dim() == 3
#
#         # 统一处理输入形状，使其符合 GRU 的输入要求 (batch, seq_len, features)
#         if not is_sequence:
#             # 如果是单步 (batch_size, features)，增加一个 seq_len=1 的维度
#             obs_tensor = obs_tensor.unsqueeze(1)  # -> (batch_size, 1, features)
#
#         # 1. 原始状态序列首先通过 GRU
#         gru_out, new_hidden = self.gru(obs_tensor, hidden_state)
#
#         # --- 新的混合 MLP 数据流 ---
#         # 2. GRU 的输出流经共享 MLP 基座
#         base_features = self.shared_base_mlp(gru_out)
#
#         # 3. 共享特征被分别送入两个专用塔楼
#         continuous_features = self.continuous_tower(base_features)
#         discrete_features = self.discrete_tower(base_features)
#
#         # 4. 如果是单步输入，压缩特征维度以匹配头部
#         if not is_sequence:
#             continuous_features = continuous_features.squeeze(1)
#             discrete_features = discrete_features.squeeze(1)
#
#         # 5. 每个头部接收来自其专属塔楼的特征
#         mu = self.mu_head(continuous_features)
#         all_disc_logits = self.discrete_head(discrete_features)
#
#         # 后续的分布创建逻辑与原版 Actor 完全相同
#         split_sizes = list(DISCRETE_DIMS.values())
#         logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
#         # trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
#         trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts
#         # 关键：动作掩码依赖于原始输入 obs_tensor，而不是 GRU 的输出
#         # 动作掩码逻辑 (需要注意在序列情况下正确索引)
#         # obs_tensor 此时可能是 (batch, seq_len, features) 或 (batch, features)
#         # 使用 ... (Ellipsis) 可以优雅地处理这两种情况，它代表任意数量的前导维度。
#         has_flares_info = obs_tensor[..., 7]  # 使用 ... 来处理单步和序列两种情况
#         mask = (has_flares_info == 0)
#         trigger_logits_masked = trigger_logits.clone()
#         if torch.any(mask):
#             # unsqueeze a dim to match the mask shape with trigger_logits_masked if they are different
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
#
#         # mu, log_std = cont_params.chunk(2, dim=-1)
#         # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
#         # std = torch.exp(log_std)
#         # --- 💥 [修改] 5. 创建连续动作分布 ---
#         # 使用全局可学习的 log_std 参数
#         log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
#         # 使用 .expand_as(mu) 来匹配批次和序列维度
#         std = torch.exp(log_std).expand_as(mu)
#         continuous_base_dist = Normal(mu, std)
#
#         # Bernoulli 的 logits 需要移除最后一个维度（如果它存在且为1）
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
#
#         return distributions, new_hidden
#
#
# class Critic_GRU(Module):
#     """
#     Critic 网络 (价值网络) - 基于 GRU 的版本。
#     结构为: MLP 特征提取 -> GRU 序列处理 -> 输出头。
#     用于评估状态序列的价值。
#     """
#     """
#        Critic 网络 (价值网络) - [新结构: GRU -> MLP]
#        结构为: GRU 序列处理 -> MLP 特征提取 -> 输出头。
#        """
#     def __init__(self):
#         super(Critic_GRU, self).__init__()
#         self.input_dim = CRITIC_PARA.input_dim
#         self.output_dim = CRITIC_PARA.output_dim
#         self.rnn_hidden_size = RNN_HIDDEN_SIZE #RNN_HIDDEN_SIZE  self.input_dim
#
#         # # 1. MLP 骨干网络 (与原版 Critic 类似)
#         # self.network_base = Sequential()
#         # layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
#         # for i in range(len(layers_dims) - 1):
#         #     self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
#         #     self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
#         #     self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())
#         #
#         # base_output_dim = CRITIC_PARA.model_layer_dim[-1]
#         #
#         # # 2. GRU 层
#         # self.gru = GRU(base_output_dim, self.rnn_hidden_size, batch_first=True)
#         #
#         # # 3. 输出头，将 GRU 的输出映射到最终的价值 V(s)
#         # self.fc_out = Linear(self.rnn_hidden_size, self.output_dim)
#
#         # 1. GRU 层作为第一层
#         self.gru = GRU(self.input_dim, self.rnn_hidden_size, batch_first=True)
#
#         # # 💥 在 GRU 和 MLP 之间新增一个 LayerNorm
#         # self.mlp_input_layernorm = LayerNorm(self.rnn_hidden_size)
#
#         # 2. MLP 骨干网络，接收 GRU 的输出
#         # MLP的输入维度是 GRU 的隐藏层大小
#         layers_dims = [self.rnn_hidden_size] + CRITIC_PARA.model_layer_dim
#         self.network_base = Sequential()
#         for i in range(len(layers_dims) - 1):
#             self.network_base.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
#             # self.network_base.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
#             self.network_base.add_module(f'LeakyReLU_{i}', LeakyReLU())
#
#         # MLP 的输出维度
#         base_output_dim = CRITIC_PARA.model_layer_dim[-1]
#
#         # 3. 输出头，接收 MLP 的输出
#         self.fc_out = Linear(base_output_dim, self.output_dim)
#
#         # # --- [新增] 应用初始化 ---
#         # self.apply(init_weights)  # 对所有子模块应用通用初始化
#         #
#         # # --- [新增] 对输出层进行特殊初始化 ---
#         # # 这样做是为了在训练开始时有更稳定的价值估计
#         # init_range = 3e-3
#         # self.fc_out.weight.data.uniform_(-init_range, init_range)
#         # self.fc_out.bias.data.fill_(0)
#         # # --- 初始化结束 ---
#
#         # <<< MODIFICATION START: 精细化优化器设置 >>>
#         # 1. 将参数分为 GRU 参数 和 其他参数
#         gru_params = []
#         other_params = []
#         for name, param in self.named_parameters():
#             if not param.requires_grad:
#                 continue
#             if 'gru' in name.lower():
#                 gru_params.append(param)
#             else:
#                 other_params.append(param)
#
#         # 2. 创建参数组 (parameter groups) 列表
#         param_groups = [
#             {
#                 'params': gru_params,
#                 'lr': CRITIC_PARA.gru_lr  # 为 GRU 参数设置专属学习率
#             },
#             {
#                 'params': other_params  # 其他所有参数 (MLP, Head)
#             }
#         ]
#
#         # 3. 使用参数组初始化优化器
#         self.optim = torch.optim.Adam(param_groups, lr=CRITIC_PARA.lr)
#
#         print("--- Critic_GRU Optimizer Initialized with Parameter Groups ---")
#         print(f"  - GRU Params LR: {CRITIC_PARA.gru_lr}")
#         print(f"  - Other (MLP) Params LR: {CRITIC_PARA.lr}")
#         # <<< MODIFICATION END >>>
#
#         # 优化器等设置
#         # self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
#         self.critic_scheduler = lr_scheduler.LinearLR(self.optim, start_factor=1.0,
#                                                       end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
#                                                       total_iters=AGENTPARA.MAX_EXE_NUM)
#         self.to(CRITIC_PARA.device)
#
#     def forward(self, obs, hidden_state):
#         """
#         GRU Critic 的前向传播。
#         同样支持单步和序列输入。
#         """
#         obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
#         is_sequence = obs_tensor.dim() == 3
#
#         if not is_sequence:
#             obs_tensor = obs_tensor.unsqueeze(1)
#         # # 1. MLP 特征提取
#         # base_features = self.network_base(obs_tensor)
#         # # 2. GRU 序列处理
#         # gru_out, new_hidden = self.gru(base_features, hidden_state)
#         #
#         # if not is_sequence:
#         #     gru_out = gru_out.squeeze(1)
#         # # 3. 输出头计算价值
#         # value = self.fc_out(gru_out)
#
#         # 1. [新流程] 原始状态序列首先通过 GRU
#         gru_out, new_hidden = self.gru(obs_tensor, hidden_state)
#
#         # # 2. 💥 [新流程] 将 GRU 的输出通过新增的 LayerNorm
#         # normed_gru_out = self.mlp_input_layernorm(gru_out)
#         # # 3. [新流程] GRU 的输出（记忆向量）再通过 MLP 进行特征提取
#         # base_features = self.network_base(normed_gru_out)  # MLP 的输入是归一化后的 gru_out
#
#         # 2. [新流程] GRU 的输出再通过 MLP
#         base_features = self.network_base(gru_out)
#
#         if not is_sequence:
#             base_features = base_features.squeeze(1)
#
#         # 3. [新流程] MLP 的输出送入输出头计算价值
#         value = self.fc_out(base_features)
#         return value, new_hidden
#
# # ==============================================================================
# # <<< 新架构 >>>: 定义基于 MLP -> GRU 的 Actor
# #                       [💥 新结构: 共享MLP -> GRU -> 塔楼MLP -> Heads]
# # ==============================================================================
#
# class Actor_GRU(Module):
#     """
#     Actor 网络 (策略网络) - [混合架构: MLP -> GRU]
#     结构为: 共享MLP基座 -> GRU 序列处理 -> 专用MLP塔楼 -> 独立动作头。
#     """
#
#     def __init__(self):
#         super(Actor_GRU, self).__init__()
#         self.input_dim = ACTOR_PARA.input_dim
#         self.log_std_min = -20.0
#         self.log_std_max = 2.0
#         self.rnn_hidden_size = RNN_HIDDEN_SIZE
#
#         # --- 1. 定义 GRU 之前的共享 MLP 特征提取器 (Pre-GRU MLP) ---
#         # 假设我们使用 model_layer_dim 的前两层作为 Pre-GRU MLP
#         # 例如，如果 model_layer_dim = [256, 256, 256]，这里就是 [256, 256]
#         pre_gru_mlp_layers = 2
#         pre_gru_dims = ACTOR_PARA.model_layer_dim[:pre_gru_mlp_layers]  # -> [256, 256]
#
#         self.pre_gru_mlp = Sequential()
#         mlp_input_dim = self.input_dim
#         for i, dim in enumerate(pre_gru_dims):
#             self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(mlp_input_dim, dim))
#             self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
#             mlp_input_dim = dim
#
#         # Pre-GRU MLP 的输出维度，将作为 GRU 的输入维度
#         pre_gru_output_dim = pre_gru_dims[-1] if pre_gru_dims else self.input_dim
#
#         # # --- 新增: GRU 前的归一化 ---
#         # self.ln_pre_gru = LayerNorm(pre_gru_output_dim)
#
#         # --- 2. GRU 层 ---
#         # GRU 的输入维度现在是 Pre-GRU MLP 的输出维度
#         self.gru = GRU(pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)
#
#         # # --- 新增: GRU 后的归一化 ---
#         # self.ln_post_gru = LayerNorm(self.rnn_hidden_size)
#
#         # --- 3. 定义 GRU 之后的 MLP 塔楼 (Post-GRU Towers) ---
#         # 这部分与你原有的 Actor_GRU 类似，但输入维度是 GRU 的 hidden_size
#         post_gru_dims = ACTOR_PARA.model_layer_dim[pre_gru_mlp_layers:]  # -> [256]
#
#         # 连续动作塔楼
#         self.continuous_tower = Sequential()
#         tower_input_dim = self.rnn_hidden_size
#         for i, dim in enumerate(post_gru_dims):
#             self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         continuous_tower_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size
#
#         # 离散动作塔楼 (可以和连续塔楼结构相同或不同)
#         self.discrete_tower = Sequential()
#         tower_input_dim = self.rnn_hidden_size
#         for i, dim in enumerate(post_gru_dims):
#             self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
#             self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#         discrete_tower_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size
#
#         # --- 4. 定义最终的输出头 (Heads) ---
#         self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
#         self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
#         self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))
#
#         # --- 必须添加这行 ---
#         self.apply(init_weights)
#
#         # --- 5. 优化器设置 (可以保持不变) ---
#         # 仍然可以将 GRU 参数和其他 MLP 参数分开设置不同的学习率
#         gru_params = list(self.gru.parameters())
#         other_params = (
#                 list(self.pre_gru_mlp.parameters()) +
#                 list(self.continuous_tower.parameters()) +
#                 list(self.discrete_tower.parameters()) +
#                 list(self.mu_head.parameters()) +
#                 list(self.discrete_head.parameters()) +
#                 [self.log_std_param]
#         )
#
#         param_groups = [
#             {'params': gru_params, 'lr': ACTOR_PARA.gru_lr},
#             {'params': other_params, 'lr': ACTOR_PARA.lr}
#         ]
#         self.optim = torch.optim.Adam(param_groups)
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
#         # 1. 原始状态序列首先通过 Pre-GRU MLP 进行特征提取
#         # Sequential 会自动地将 MLP 应用于序列的最后一个维度
#         # 输入: (batch, seq_len, features) -> 输出: (batch, seq_len, pre_gru_output_dim)
#         features_sequence = self.pre_gru_mlp(obs_tensor)
#
#         # # --- 应用 LayerNorm ---
#         # features_sequence = self.ln_pre_gru(features_sequence)
#
#         # 2. 将提取出的特征序列送入 GRU
#         # 输入: (batch, seq_len, pre_gru_output_dim) -> 输出: (batch, seq_len, rnn_hidden_size)
#         gru_out, new_hidden = self.gru(features_sequence, hidden_state)
#
#         # # --- 应用 LayerNorm ---
#         # gru_out = self.ln_post_gru(gru_out)
#
#         # 3. GRU 的输出被分别送入两个专用塔楼
#         continuous_features = self.continuous_tower(gru_out)
#         discrete_features = self.discrete_tower(gru_out)
#
#         # 4. 如果是单步输入，压缩特征维度以匹配头部
#         if not is_sequence:
#             continuous_features = continuous_features.squeeze(1)
#             discrete_features = discrete_features.squeeze(1)
#
#         # 5. 每个头部接收来自其专属塔楼的特征
#         mu = self.mu_head(continuous_features)
#         all_disc_logits = self.discrete_head(discrete_features)
#
#         # --- 后续的分布创建和掩码逻辑与原版 Actor_GRU 完全相同 ---
#         # ... (此处省略与 Actor_GRU 中完全相同的掩码和分布创建代码) ...
#         split_sizes = list(DISCRETE_DIMS.values())
#         logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
#         trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts
#
#         has_flares_info = obs_tensor[..., 9] # 原为 obs_tensor[..., 11]
#         mask = (has_flares_info == 0)
#         trigger_logits_masked = trigger_logits.clone()
#         if torch.any(mask):
#             if mask.dim() < trigger_logits_masked.dim():
#                 mask = mask.unsqueeze(-1)
#             trigger_logits_masked[mask] = torch.finfo(torch.float32).min
#
#         # (省略层次化控制逻辑，因为和原来一样)
#         trigger_probs = torch.sigmoid(trigger_logits_masked)
#         no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)
#         salvo_size_logits_masked = salvo_size_logits.clone()
#         num_groups_logits_masked = num_groups_logits.clone()
#         inter_interval_logits_masked = inter_interval_logits.clone()
#         if torch.any(no_trigger_mask):
#             INF = 1e6
#             NEG_INF = -1e6
#             for logits_tensor in [salvo_size_logits_masked, num_groups_logits_masked, inter_interval_logits_masked]:
#                 logits_sub = logits_tensor[no_trigger_mask]
#                 if logits_sub.numel() > 0:
#                     logits_sub[:] = NEG_INF
#                     logits_sub[:, 0] = INF
#                     logits_tensor[no_trigger_mask] = logits_sub
#
#         log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std).expand_as(mu)
#         continuous_base_dist = Normal(mu, std)
#
#         trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
#         salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
#         num_groups_dist = Categorical(logits=num_groups_logits_masked)
#         inter_interval_dist = Categorical(logits=inter_interval_logits_masked)
#
#         distributions = {
#             'continuous': continuous_base_dist,
#             'trigger': trigger_dist,
#             'salvo_size': salvo_size_dist,
#             'num_groups': num_groups_dist,
#             'inter_interval': inter_interval_dist
#         }
#
#         return distributions, new_hidden
#
#
# # ==============================================================================
# # <<< 新架构 >>>: 定义基于 MLP -> GRU 的 Critic
# #                       [💥 新结构: 共享MLP -> GRU -> MLP -> Head]
# # ==============================================================================
#
# class Critic_GRU(Module):
#     """
#     Critic 网络 (价值网络) - [混合架构: MLP -> GRU]
#     结构为: 共享MLP特征提取 -> GRU 序列处理 -> MLP -> 输出头。
#     与 Actor_MLP_GRU 的主干结构保持一致。
#     """
#
#     def __init__(self):
#         super(Critic_GRU, self).__init__()
#         self.input_dim = CRITIC_PARA.input_dim
#         self.output_dim = CRITIC_PARA.output_dim
#         self.rnn_hidden_size = RNN_HIDDEN_SIZE
#
#         # --- 1. 定义 GRU 之前的共享 MLP 特征提取器 (Pre-GRU MLP) ---
#         # 使用 Critic 配置中的 MLP 层定义
#         pre_gru_mlp_layers = 2  # 与 Actor 保持一致
#         pre_gru_dims = CRITIC_PARA.model_layer_dim[:pre_gru_mlp_layers]
#
#         self.pre_gru_mlp = Sequential()
#         mlp_input_dim = self.input_dim
#         for i, dim in enumerate(pre_gru_dims):
#             self.pre_gru_mlp.add_module(f'pre_gru_fc_{i}', Linear(mlp_input_dim, dim))
#             self.pre_gru_mlp.add_module(f'pre_gru_leakyrelu_{i}', LeakyReLU())
#             mlp_input_dim = dim
#
#         pre_gru_output_dim = pre_gru_dims[-1] if pre_gru_dims else self.input_dim
#
#         # # --- 新增: GRU 前的归一化 ---
#         # self.ln_pre_gru = LayerNorm(pre_gru_output_dim)
#
#         # --- 2. GRU 层 ---
#         self.gru = GRU(pre_gru_output_dim, self.rnn_hidden_size, batch_first=True)
#
#         # # --- 新增: GRU 后的归一化 ---
#         # self.ln_post_gru = LayerNorm(self.rnn_hidden_size)
#
#         # --- 3. 定义 GRU 之后的 MLP (Post-GRU MLP) ---
#         post_gru_dims = CRITIC_PARA.model_layer_dim[pre_gru_mlp_layers:]
#
#         self.post_gru_mlp = Sequential()
#         tower_input_dim = self.rnn_hidden_size
#         for i, dim in enumerate(post_gru_dims):
#             self.post_gru_mlp.add_module(f'post_gru_fc_{i}', Linear(tower_input_dim, dim))
#             self.post_gru_mlp.add_module(f'post_gru_leakyrelu_{i}', LeakyReLU())
#             tower_input_dim = dim
#
#         post_gru_output_dim = post_gru_dims[-1] if post_gru_dims else self.rnn_hidden_size
#
#         # --- 4. 最终的输出头 (Head) ---
#         self.fc_out = Linear(post_gru_output_dim, self.output_dim)
#
#         # --- 必须添加这行 ---
#         self.apply(init_weights)
#
#         # --- 5. 优化器设置 (与 Actor 类似，分离 GRU 和其他参数) ---
#         gru_params = list(self.gru.parameters())
#         other_params = (
#                 list(self.pre_gru_mlp.parameters()) +
#                 list(self.post_gru_mlp.parameters()) +
#                 list(self.fc_out.parameters())
#         )
#
#         param_groups = [
#             {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
#             {'params': other_params, 'lr': CRITIC_PARA.lr}
#         ]
#         self.optim = torch.optim.Adam(param_groups)
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
#         # 1. 原始状态序列通过 Pre-GRU MLP
#         features_sequence = self.pre_gru_mlp(obs_tensor)
#
#         # # --- 应用 LayerNorm ---
#         # features_sequence = self.ln_pre_gru(features_sequence)
#
#         # 2. 特征序列送入 GRU
#         gru_out, new_hidden = self.gru(features_sequence, hidden_state)
#
#         # # --- 应用 LayerNorm ---
#         # gru_out = self.ln_post_gru(gru_out)
#
#         # 3. GRU 输出通过 Post-GRU MLP
#         post_gru_features = self.post_gru_mlp(gru_out)
#
#         # 处理单步输入的情况
#         if not is_sequence:
#             post_gru_features = post_gru_features.squeeze(1)
#
#         # 4. MLP 输出送入输出头计算价值
#         value = self.fc_out(post_gru_features)
#
#         return value, new_hidden


# # ==============================================================================
# # <<< 新架构 >>>: 定义基于 128 -> GRU -> 128 的 Actor 和 Critic
# #                       [💥 新结构: MLP -> GRU -> 共享MLP -> 塔楼MLP -> Heads]
# # ==============================================================================
#
# class Actor_GRU(Module):
#     """
#     Actor 网络 (策略网络) - [自定义混合架构: 128 -> GRU -> 128 (共享) -> 128 (塔楼)]
#     """
#
#     def __init__(self):
#         super(Actor_GRU, self).__init__()
#         self.input_dim = ACTOR_PARA.input_dim
#         self.log_std_min = -20.0
#         self.log_std_max = 2.0
#         self.rnn_hidden_size = 128  # GRU的隐藏层大小固定为128
#
#         # 定义网络各部分的维度
#         pre_gru_dim = 128
#         post_gru_shared_dim = 128
#         tower_dim = 128
#
#         # --- 1. Pre-GRU MLP (1层, 输出128) ---
#         self.pre_gru_mlp = Sequential(
#             Linear(self.input_dim, pre_gru_dim),
#             LeakyReLU()
#         )
#
#         # --- 2. GRU 层 (输入128, 隐藏128) ---
#         self.gru = GRU(pre_gru_dim, self.rnn_hidden_size, batch_first=True)
#
#         # --- 3. Post-GRU 共享 MLP (1层, 输出128) ---
#         self.post_gru_shared_mlp = Sequential(
#             Linear(self.rnn_hidden_size, post_gru_shared_dim),
#             LeakyReLU()
#         )
#
#         # --- 4. 专用 MLP 塔楼 (每条路1层, 输出128) ---
#         # 连续动作塔楼
#         self.continuous_tower = Sequential(
#             Linear(post_gru_shared_dim, tower_dim),
#             LeakyReLU()
#         )
#         # 离散动作塔楼
#         self.discrete_tower = Sequential(
#             Linear(post_gru_shared_dim, tower_dim),
#             LeakyReLU()
#         )
#
#         # --- 5. 最终的输出头 (Heads) ---
#         self.mu_head = Linear(tower_dim, CONTINUOUS_DIM)
#         self.discrete_head = Linear(tower_dim, TOTAL_DISCRETE_LOGITS)
#         self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))
#
#         # --- 6. 优化器设置 ---
#         gru_params = list(self.gru.parameters())
#         other_params = (
#                 list(self.pre_gru_mlp.parameters()) +
#                 list(self.post_gru_shared_mlp.parameters()) +
#                 list(self.continuous_tower.parameters()) +
#                 list(self.discrete_tower.parameters()) +
#                 list(self.mu_head.parameters()) +
#                 list(self.discrete_head.parameters()) +
#                 [self.log_std_param]
#         )
#         param_groups = [
#             {'params': gru_params, 'lr': ACTOR_PARA.gru_lr},
#             {'params': other_params, 'lr': ACTOR_PARA.lr}
#         ]
#         self.optim = torch.optim.Adam(param_groups)
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
#         # 1. 原始状态序列通过 Pre-GRU MLP
#         features1 = self.pre_gru_mlp(obs_tensor)
#
#         # 2. 特征序列送入 GRU
#         gru_out, new_hidden = self.gru(features1, hidden_state)
#
#         # 3. GRU 输出通过 Post-GRU 共享 MLP
#         shared_features = self.post_gru_shared_mlp(gru_out)
#
#         # 4. 共享特征被分别送入两个专用塔楼
#         continuous_features = self.continuous_tower(shared_features)
#         discrete_features = self.discrete_tower(shared_features)
#
#         # 5. 如果是单步输入，压缩特征维度以匹配头部
#         if not is_sequence:
#             continuous_features = continuous_features.squeeze(1)
#             discrete_features = discrete_features.squeeze(1)
#
#         # 6. 每个头部接收来自其专属塔楼的特征
#         mu = self.mu_head(continuous_features)
#         all_disc_logits = self.discrete_head(discrete_features)
#
#         # --- 后续的分布创建和掩码逻辑与之前完全相同 ---
#         # ... (此处省略与之前 Actor_GRU 中完全相同的掩码和分布创建代码) ...
#         split_sizes = list(DISCRETE_DIMS.values())
#         logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
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
#         trigger_probs = torch.sigmoid(trigger_logits_masked)
#         no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)
#         salvo_size_logits_masked = salvo_size_logits.clone()
#         num_groups_logits_masked = num_groups_logits.clone()
#         inter_interval_logits_masked = inter_interval_logits.clone()
#         if torch.any(no_trigger_mask):
#             INF = 1e6
#             NEG_INF = -1e6
#             for logits_tensor in [salvo_size_logits_masked, num_groups_logits_masked, inter_interval_logits_masked]:
#                 logits_sub = logits_tensor[no_trigger_mask]
#                 if logits_sub.numel() > 0:
#                     logits_sub[:] = NEG_INF
#                     logits_sub[:, 0] = INF
#                     logits_tensor[no_trigger_mask] = logits_sub
#
#         log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std).expand_as(mu)
#         continuous_base_dist = Normal(mu, std)
#
#         trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
#         salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
#         num_groups_dist = Categorical(logits=num_groups_logits_masked)
#         inter_interval_dist = Categorical(logits=inter_interval_logits_masked)
#
#         distributions = {
#             'continuous': continuous_base_dist,
#             'trigger': trigger_dist,
#             'salvo_size': salvo_size_dist,
#             'num_groups': num_groups_dist,
#             'inter_interval': inter_interval_dist
#         }
#
#         return distributions, new_hidden
#
#
# class Critic_GRU(Module):
#     """
#     Critic 网络 (价值网络) - [自定义混合架构]
#     与 Actor 的主干结构保持一致，以实现更好的特征共享和表示学习。
#     """
#
#     def __init__(self):
#         super(Critic_GRU, self).__init__()
#         self.input_dim = CRITIC_PARA.input_dim
#         self.output_dim = CRITIC_PARA.output_dim
#         self.rnn_hidden_size = 128  # 与 Actor 保持一致
#
#         # 定义网络各部分的维度
#         pre_gru_dim = 128
#         post_gru_shared_dim = 128
#         final_mlp_dim = 128
#
#         # --- 1. Pre-GRU MLP (与 Actor 结构相同) ---
#         self.pre_gru_mlp = Sequential(
#             Linear(self.input_dim, pre_gru_dim),
#             LeakyReLU()
#         )
#
#         # --- 2. GRU 层 (与 Actor 结构相同) ---
#         self.gru = GRU(pre_gru_dim, self.rnn_hidden_size, batch_first=True)
#
#         # --- 3. Post-GRU 共享 MLP (与 Actor 结构相同) ---
#         self.post_gru_shared_mlp = Sequential(
#             Linear(self.rnn_hidden_size, post_gru_shared_dim),
#             LeakyReLU()
#         )
#
#         # --- 4. 最终的价值处理 MLP ---
#         self.final_mlp = Sequential(
#             Linear(post_gru_shared_dim, final_mlp_dim),
#             LeakyReLU()
#         )
#
#         # --- 5. 最终的输出头 (Head) ---
#         self.fc_out = Linear(final_mlp_dim, self.output_dim)
#
#         # --- 6. 优化器设置 ---
#         gru_params = list(self.gru.parameters())
#         other_params = (
#                 list(self.pre_gru_mlp.parameters()) +
#                 list(self.post_gru_shared_mlp.parameters()) +
#                 list(self.final_mlp.parameters()) +
#                 list(self.fc_out.parameters())
#         )
#         param_groups = [
#             {'params': gru_params, 'lr': CRITIC_PARA.gru_lr},
#             {'params': other_params, 'lr': CRITIC_PARA.lr}
#         ]
#         self.optim = torch.optim.Adam(param_groups)
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
#         # 1. 原始状态序列通过 Pre-GRU MLP
#         features1 = self.pre_gru_mlp(obs_tensor)
#
#         # 2. 特征序列送入 GRU
#         gru_out, new_hidden = self.gru(features1, hidden_state)
#
#         # 3. GRU 输出通过 Post-GRU 共享 MLP
#         shared_features = self.post_gru_shared_mlp(gru_out)
#
#         # 4. 共享特征送入最终的MLP
#         final_features = self.final_mlp(shared_features)
#
#         # 处理单步输入的情况
#         if not is_sequence:
#             final_features = final_features.squeeze(1)
#
#         # 5. MLP 输出送入输出头计算价值
#         value = self.fc_out(final_features)
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
            print("--- 初始化 PPO Agent (使用 GRU 模型) ---")
            self.Actor = Actor_GRU()
            self.Critic = Critic_GRU()
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
        self.training_start_time = time.strftime("PPOGRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../save/save_evade_fuza"
        win_rate_subdir = "胜率模型"
        # 为本次运行创建一个唯一的存档文件夹
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)

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

            # ================= [修改开始] =================
            # 1. 计算原始高斯分布的 log_prob
            log_prob_u = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 2. 计算雅可比修正项 (稳定公式)
            # 公式: 2 * (log 2 - u - softplus(-2u))
            # 注意: u 是 pre-tanh 的值
            # correction = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u)).sum(dim=-1)

            # 3. 得到最终动作 a = tanh(u) 的 log_prob
            log_prob_cont = log_prob_u #- correction
            # ================= [修改结束] =================

            # log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 从所有离散动作分布中采样
            sampled_actions_dict = {}
            for key, dist in dists.items():
                if key == 'continuous': continue
                if deterministic:
                    if isinstance(dist, Categorical): # 分类分布：取概率最大的类别
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                    else: # 伯努利分布：取概率大于0.5的
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                        # sampled_actions_dict[key] = dist.sample()
                else:
                    sampled_actions_dict[key] = dist.sample()

            # 提取 Trigger 动作用于掩码
            trigger_action = sampled_actions_dict['trigger']

            # ================= [3. 离散动作 Log Prob (Ratio 一致性修复)] =================
            # 1. Trigger 的 Log Prob 永远计算
            log_prob_trigger = dists['trigger'].log_prob(trigger_action)

            # 2. 计算所有子动作的 Log Prob 总和
            sub_log_prob_sum = 0.0
            for key in sampled_actions_dict:
                if key == 'trigger': continue
                sub_log_prob_sum += dists[key].log_prob(sampled_actions_dict[key])

            # 3. 🌟 灵魂掩码 🌟：如果 trigger=0，子动作概率不计入 (即 log_prob = 0)
            # 确保 trigger_action 是 float 类型用于乘法
            valid_sub_log_prob = sub_log_prob_sum * trigger_action

            # 4. 合并离散部分
            log_prob_disc = log_prob_trigger + valid_sub_log_prob
            # ===========================================================

            # 计算总的对数概率
            # log_prob_disc = sum(dists[key].log_prob(act) for key, act in sampled_actions_dict.items())
            total_log_prob = log_prob_cont + log_prob_disc
            # 准备要存储在 Buffer 中的动作（连续部分是原始样本 u，离散部分是类别索引）
            # action_disc_to_store = torch.stack(list(sampled_actions_dict.values()), dim=-1).float()
            # 强制指定顺序，必须与 learn 函数中 discrete_actions_from_buffer 的解析顺序一一对应！
            # 顺序: [Trigger, Salvo_Size, Num_Groups, Inter_Interval]
            ordered_disc_actions = [
                sampled_actions_dict['trigger'],
                sampled_actions_dict['salvo_size'],
                sampled_actions_dict['num_groups'],
                sampled_actions_dict['inter_interval']
            ]
            action_disc_to_store = torch.stack(ordered_disc_actions, dim=-1).float()
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
        执行 PPO 的学习和更新步骤。
        已集成：
        1. 全局优势归一化 (Global Advantage Normalization)
        2. 维度对齐 (防止 Broadcasting Error)
        3. 雅可比修正 (Jacobian Correction)
        4. 混合架构支持 (MLP/GRU)
        """
        # 如果 Buffer 中的数据不足一个批次，则跳过学习
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            # print(f"  [Info] Buffer size ({self.buffer.get_buffer_size()}) < batch size ({BUFFERPARA.BATCH_SIZE}). Skipping.")
            return None

        # 1. 数据准备
        # 注意：这里我们获取了所有数据，此时它们都是 Numpy 数组
        states, values, actions, old_probs, rewards, dones, _, __ = self.buffer.get_all_data()

        # 计算 GAE 优势
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)

        # ================= [关键修改：全局优势归一化 & 维度对齐] =================

        # 1. 维度强制对齐 (N,) -> (N, 1)
        # 这步至关重要，防止 (N,) + (N, 1) 导致生成 (N, N) 的巨大矩阵
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

        # 用于记录训练过程中的各种损失和指标
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': [],'mean_entropy_trigger': [], 'mean_entropy_sub': []}

        # 2. PPO 训练循环
        for _ in range(self.ppo_epoch):
            # 根据是否使用 RNN，选择不同的批次生成器
            if self.use_rnn:
                # [GRU模式]
                # 将处理好的全局 advantages 和 returns 传入生成器
                # 生成器内部会根据序列切片提取对应的片段
                batch_generator = self.buffer.generate_sequence_batches(
                    SEQUENCE_LENGTH, BUFFERPARA.BATCH_SIZE, advantages, returns
                )
            else:
                # [MLP模式]
                # 生成随机索引
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                # 3. 批次数据提取
                if self.use_rnn:
                    # [GRU模式] 解包数据 (注意：return_ 和 advantage 已经是处理过的了)
                    state, action_batch, old_prob, advantage, return_, initial_actor_h, initial_critic_h = batch_data

                    # 转 Tensor
                    state = check(state).to(**ACTOR_PARA.tpdv)
                    action_batch = check(action_batch).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)  # RNN下通常不需要 view(-1)
                    advantage = check(advantage).to(**ACTOR_PARA.tpdv)
                    return_ = check(return_).to(**CRITIC_PARA.tpdv)
                    initial_actor_h = check(initial_actor_h).to(**ACTOR_PARA.tpdv)
                    initial_critic_h = check(initial_critic_h).to(**CRITIC_PARA.tpdv)
                else:
                    # [MLP模式] 使用索引切片
                    batch_indices = batch_data

                    state = check(states[batch_indices]).to(**ACTOR_PARA.tpdv)
                    action_batch = check(actions[batch_indices]).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_probs[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1)

                    # 直接提取已经全局归一化过的 Advantage
                    advantage = check(advantages[batch_indices]).to(**ACTOR_PARA.tpdv).view(-1, 1)
                    # 直接提取预计算好的 Return
                    return_ = check(returns[batch_indices]).to(**CRITIC_PARA.tpdv).view(-1, 1)

                    initial_actor_h, initial_critic_h = None, None

                # 解析动作
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # 4. Actor 前向传播
                if self.use_rnn:
                    new_dists, _ = self.Actor(state, initial_actor_h)
                else:
                    new_dists = self.Actor(state)

                # ================= [修改开始：雅可比修正 + Ratio一致性 + 条件熵] =================

                # --- A. 连续动作 Log Prob (带雅可比修正) ---
                log_prob_u_buffer = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                # correction_buffer = 2.0 * (np.log(2.0) - u_from_buffer - F.softplus(-2.0 * u_from_buffer)).sum(
                #     dim=-1)
                new_log_prob_cont = log_prob_u_buffer #- correction_buffer

                # --- B. 离散动作 Log Prob (Ratio 一致性修复) ---
                # 1. 提取 Buffer 中记录的真实开火情况 (用于 Mask LogProb)
                actual_triggers = discrete_actions_from_buffer['trigger']  # [B, S] or [B]

                # 2. Trigger 的 log_prob 永远要算
                new_log_prob_trigger = new_dists['trigger'].log_prob(actual_triggers)

                # 3. 子动作原始 log_prob 总和
                sub_log_probs_raw = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer if key != 'trigger'
                )

                # 4. 🌟 灵魂掩码 (Log Prob) 🌟：如果历史数据没开火，子动作概率变化不应影响 Ratio
                valid_sub_log_probs = sub_log_probs_raw * actual_triggers

                # 5. 合并离散 log_prob
                new_log_prob_disc = new_log_prob_trigger + valid_sub_log_probs

                # --- C. 计算总 Log Prob (用于计算 Ratio) ---
                new_prob = new_log_prob_cont + new_log_prob_disc

                # # 在计算熵之前插入
                # # 检查 Buffer 里到底有没有 1
                # debug_trigger_sum = actual_triggers.sum().item()
                # if debug_trigger_sum == 0:
                #     print(f"[严重警告] Batch 里的 trigger 全是 0！Step: {self.total_steps}")
                #     # 打印一下 trigger 的原始值看看是不是被 mask 搞坏了
                #     # print(action_batch[..., CONTINUOUS_DIM])
                # else:
                #     print(f"[正常] Batch 里发现了 {debug_trigger_sum} 次开火。")

                # --- D. 熵计算 (条件熵逻辑) ---

                # D.1 连续动作熵 (带雅可比修正的重采样)
                entropy_base = new_dists['continuous'].entropy().sum(dim=-1)
                # entropy_base = new_dists['continuous'].entropy().mean(dim=-1)  # 动作维度求平均
                u_curr_sample = new_dists['continuous'].rsample()
                correction_curr = 2.0 * (np.log(2.0) - u_curr_sample - F.softplus(-2.0 * u_curr_sample)).sum(dim=-1)
                entropy_cont = entropy_base + correction_curr
                mean_entropy_cont = entropy_cont.mean()

                # D.2 Trigger 熵
                entropy_trigger = new_dists['trigger'].entropy()
                mean_entropy_trigger = entropy_trigger.mean()

                # ==========================================================
                # D.3 🌟 移除子动作的熵掩码，强制保持后台探索欲 🌟
                # ==========================================================

                # 1. 计算所有子动作的原始熵 (Raw)
                entropy_sub_actions_raw = sum(
                    dist.entropy() for key, dist in new_dists.items()
                    if key not in ['continuous', 'trigger']
                )

                # 2. 直接求平均！不要乘 actual_triggers！
                # 让网络始终保持对子动作选项的好奇心，哪怕它当前不想按 Trigger。
                mean_entropy_sub = entropy_sub_actions_raw.mean()

                # ==========================================================
                # D.3 🌟 灵魂掩码 (Entropy) 🌟：子动作条件均值熵
                # [修正]: 必须使用 Buffer 中的 actual_triggers (历史动作) 作为掩码
                # 这样才能与 Ratio 的掩码逻辑保持严格一致，且避免阈值跳变带来的抖动。
                # ==========================================================

                # # 1. 计算所有子动作的熵 (Raw) -> Shape: [Batch, Seq]
                # entropy_sub_actions_raw = sum(
                #     dist.entropy() for key, dist in new_dists.items()
                #     if key not in ['continuous', 'trigger']
                # )
                # # 1. 计算所有子动作的熵 (Raw) 并按动作数量求平均 -> Shape: [Batch, Seq]
                # sub_action_keys = [key for key in new_dists.items() if key[0] not in ['continuous', 'trigger']]
                # num_sub_actions = len(sub_action_keys)  # 当前为 3 (salvo_size, num_groups, inter_interval)
                #
                # entropy_sub_actions_raw = sum(
                #     dist.entropy() for key, dist in sub_action_keys
                # ) / num_sub_actions

                # # 2. 使用 Buffer 中的真实触发记录作为掩码 -> Shape: [Batch, Seq]
                # # (不需要再从 new_dists['trigger'].probs 里去算阈值了)
                # mask_trigger = actual_triggers  # 0.0 或 1.0
                #
                # # 3. 统计有效的触发次数 (整个 Batch * Sequence 中发射的总次数)
                # num_triggered = mask_trigger.sum()
                #
                # # 4. 计算条件均值熵
                # if num_triggered > 0:
                #     # 只计算那些【历史上确实发射了】的样本的子动作熵
                #     # 重点：除以 num_triggered 而不是 (Batch * Seq)
                #     mean_entropy_sub = (entropy_sub_actions_raw * mask_trigger).sum() / num_triggered
                # else:
                #     mean_entropy_sub = torch.tensor(0.0, device=ACTOR_PARA.device)

                # # 3. 🌟 灵魂掩码 (Entropy) 🌟：改为全局平均
                # # 逻辑：(子动作熵 * 真实开火掩码).mean()
                # # 效果：不开火的时候熵贡献为0，分母为总步数 (Batch * Seq)。
                # # 结果：如果开火很稀疏，这个值会非常小（这是正常的）。
                # mean_entropy_sub = (entropy_sub_actions_raw * actual_triggers).mean()

                # D.4 加权总熵 Bonus
                # COEF_CONT = 1.0   # 连续动作主导
                # COEF_TRIG = 10.0  # 鼓励 Trigger 探索
                # COEF_SUB  = 0.5   # 子动作熵
                # COEF_CONT = AGENTPARA.entropy * 1.0
                # COEF_TRIG = AGENTPARA.entropy * 1.0
                # COEF_SUB = AGENTPARA.entropy * 1.0

                # total_entropy_bonus = (COEF_CONT * mean_entropy_cont) + \
                #                       (COEF_TRIG * mean_entropy_trigger) + \
                #                       (COEF_SUB * mean_entropy_sub)

                total_entropy_bonus = mean_entropy_cont + mean_entropy_trigger + mean_entropy_sub #* 0.1
                # ==========================================================

                # # ================= [雅可比修正与 LogProb 计算] =================
                #
                # # --- A. 连续动作 Log Prob ---
                # # 1. 计算旧动作在高斯分布下的 log_prob
                # log_prob_u_buffer = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                #
                # # 2. 计算雅可比修正项
                # correction_buffer = 2.0 * (np.log(2.0) - u_from_buffer - F.softplus(-2.0 * u_from_buffer)).sum(dim=-1)
                #
                # # 3. 得到最终 Log Prob (用于 Ratio)
                # new_log_prob_cont = log_prob_u_buffer - correction_buffer
                #
                # # --- B. 熵计算 (使用重采样技巧) ---
                # # 1. 基础高斯熵
                # entropy_base = new_dists['continuous'].entropy().sum(dim=-1)
                #
                # # 2. 重采样当前策略动作
                # u_curr_sample = new_dists['continuous'].rsample()
                #
                # # 3. 计算修正期望
                # correction_curr = 2.0 * (np.log(2.0) - u_curr_sample - F.softplus(-2.0 * u_curr_sample)).sum(dim=-1)
                #
                # # 4. 得到最终熵
                # entropy_cont = entropy_base + correction_curr
                #
                # # --- C. 离散动作 Log Prob ---
                # new_log_prob_disc = sum(
                #     new_dists[key].log_prob(discrete_actions_from_buffer[key])
                #     for key in discrete_actions_from_buffer
                # )
                #
                # # 合并 Log Prob
                # new_prob = new_log_prob_cont + new_log_prob_disc
                # # ==========================================================
                #
                # # 计算总熵
                # entropy_disc = sum(
                #     dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                # )
                #
                # # # =================== [💥 核心修改点：针对 RNN 提取最后一步] ===================
                # # if self.use_rnn:
                # #     # 提取最后一步的数据用于 Loss 计算
                # #     # new_prob 形状从 (B, S) 变为 (B,)
                # #     new_prob = new_prob[:, -1]
                # #     old_prob = old_prob[:, -1]
                # #     advantage = advantage[:, -1]
                # #     # 熵也只计算最后一步
                # #     entropy_cont = entropy_cont[:, -1]
                # #     entropy_disc = entropy_disc[:, -1]
                # # # ==========================================================================
                #
                # # 平均熵 (注意：entropy_cont 已经在上面算好了)
                # # 这里的 mean() 是对 batch 维度取平均
                # total_entropy_val = (entropy_cont.mean() + entropy_disc.mean())

                # 计算 Ratio
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                # 计算 Actor Loss (使用归一化后的 advantage)
                # 确保 advantage 维度匹配
                if advantage.dim() > ratio.dim():
                    advantage_squeezed = advantage.squeeze(-1)
                else:
                    advantage_squeezed = advantage

                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed

                # Loss = -Policy_Loss - Entropy_Bonus
                actor_loss = -torch.min(surr1, surr2).mean() - (AGENTPARA.con_entropy * mean_entropy_cont + AGENTPARA.dis_entropy * (1.0 * mean_entropy_trigger + 1.0 * mean_entropy_sub))#AGENTPARA.entropy * (mean_entropy_cont + 1.0 * mean_entropy_trigger + 1.0 * mean_entropy_sub)#AGENTPARA.entropy * total_entropy_bonus

                # 更新 Actor
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                # 5. Critic 更新
                if self.use_rnn:
                    new_value, _ = self.Critic(state, initial_critic_h)
                    # # =================== [💥 核心修改点：提取最后一步] ===================
                    # # 确保 new_value 和 return_ 都是 (B, 1)
                    # new_value = new_value[:, -1, :]
                    # return_ = return_[:, -1, :]
                    # # ==================================================================
                else:
                    new_value = self.Critic(state)

                # 确保维度匹配 (B, S, 1) vs (B, S, 1)
                if new_value.dim() > return_.dim():
                    return_ = return_.unsqueeze(-1)
                elif new_value.dim() < return_.dim():
                    new_value = new_value.unsqueeze(-1)

                # 计算 Critic Loss (使用原始 returns)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

                # 更新 Critic
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # 记录信息
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['mean_entropy_trigger'].append(mean_entropy_trigger.item())
                train_info['mean_entropy_sub'].append(mean_entropy_sub.item())
                train_info['dist_entropy'].append(total_entropy_bonus.item())
                train_info['entropy_cont'].append(entropy_cont.mean().item())
                train_info['adv_targ'].append(advantage.mean().item())  # 这里的 advantage 已经是归一化后的，接近0是正常的
                train_info['ratio'].append(ratio.mean().item())

        # 6. 清理
        self.buffer.clear_memory()
        # 防御性计算均值，防止空列表报错
        for key in train_info:
            if len(train_info[key]) > 0:
                train_info[key] = np.mean(train_info[key])
            else:
                train_info[key] = 0.0
        # for key in train_info:
        #     train_info[key] = np.mean(train_info[key])

        # train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        # train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        self.save()

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
    #             filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"
    #             full_path = os.path.join(self.run_save_dir, filename)
    #             torch.save(getattr(self, net).state_dict(), full_path)
    #             print(f"  - {filename} 保存成功。")
    #         except Exception as e:
    #             print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")