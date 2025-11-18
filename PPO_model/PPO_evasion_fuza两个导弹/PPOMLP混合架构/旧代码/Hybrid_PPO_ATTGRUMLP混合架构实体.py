# 文件: Hybrid_PPO_ATTGRUMLP混合架构.py (实体注意力修改版)

# 导入 PyTorch 核心库
import torch
from torch.nn import *
import torch.nn.functional as F  # <<< 新增 >>> 导入 functional 库以使用 softmax
# 导入概率分布工具，用于构建策略网络的输出
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
# 导入配置文件，其中包含各种超参数
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigAttn import *
# 导入支持 GRU 的经验回放池
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.BufferGRUAttn实体 import *
# 导入学习率调度器
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time
# 导入 PyTorch 的自动混合精度训练工具，用于加速训练并减少显存占用
from torch.cuda.amp import GradScaler, autocast

# --- 动作空间配置 (保持不变) ---
# 定义连续动作的维度和键名
CONTINUOUS_DIM = 4
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']  # 对应飞机的油门、升降舵、副翼、方向舵
# 定义离散动作的维度。每个键代表一个离散决策，值代表该决策有多少个选项。
DISCRETE_DIMS = {
    'flare_trigger': 1,  # 干扰弹触发，伯努利分布 (是/否)，所以是1个 logit
    'salvo_size': 3,  # 齐射数量，3个选项，对应一个3类的分类分布
    'num_groups': 3,  # 组数，3个选项
    'inter_interval': 3,  # 组间间隔，3个选项
}
# 计算所有离散动作 logits 的总维度
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())
# 计算存储在 Buffer 中的总动作维度（连续动作 + 离散动作的类别索引）
TOTAL_ACTION_DIM_BUFFER = CONTINUOUS_DIM + len(DISCRETE_DIMS)
# 离散动作的类别索引到实际物理值的映射
DISCRETE_ACTION_MAP = {
    'salvo_size': [2, 3, 4],  # 修改为发射2、3、4枚
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

# <<< 实体注意力修改 >>>: 定义实体和嵌入的维度
NUM_MISSILES = 2
MISSILE_FEAT_DIM = 3  # 每个导弹的观测维度: [o_dis, o_beta, o_theta_L]
AIRCRAFT_FEAT_DIM = 6  # 飞机自身观测维度: [o_av, o_h, o_ae, o_am, o_ir, o_q]
ENTITY_EMBED_DIM = 64  # 将每个实体的原始观测编码到的高维空间维度

RNN_HIDDEN_SIZE = 128  # GRU 层的隐藏单元数量
SEQUENCE_LENGTH = 10  # 训练时从经验池中采样的连续轨迹片段的长度
ATTN_NUM_HEADS = 4  # 注意力机制的头数 (实体嵌入维度较高，可以使用多头注意力)

# 确保嵌入维度可以被头数整除，这是多头注意力的要求
assert ENTITY_EMBED_DIM % ATTN_NUM_HEADS == 0, "ENTITY_EMBED_DIM must be divisible by ATTN_NUM_HEADS"


# ==============================================================================
#           实体注意力新架构: EntityEncoders -> Attention -> GRU -> MLP_Towers -> Heads
# ==============================================================================

class Actor_GRU(Module):
    """
    Actor 网络 - [实体注意力版: EntityEncoders -> Attention -> GRU -> Hybrid MLP]
    结构为: 实体编码器 -> 跨实体注意力 -> GRU 序列处理 -> 专用MLP塔楼 -> 独立动作头。
    """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        """初始化网络的各个模块、权重和优化器。"""
        super(Actor_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = ACTOR_PARA.input_dim  # 仍然使用总输入维度进行记录
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay

        # --- 1. <<< 新增 >>> 实体编码器 (Entity Encoders) ---
        # 作用: 将不同实体的低维原始观测值，映射到统一的高维特征空间（嵌入空间）
        # 为导弹创建一个共享的编码器 MLP
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM // 2),
            LeakyReLU(),
            Linear(ENTITY_EMBED_DIM // 2, ENTITY_EMBED_DIM)
        )
        # 为飞机创建一个独立的编码器 MLP
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM // 2),
            LeakyReLU(),
            Linear(ENTITY_EMBED_DIM // 2, ENTITY_EMBED_DIM)
        )

        # --- 2. <<< 修改 >>> 实体间自注意力层 ---
        # 作用: 计算飞机（Query）对各个导弹（Keys/Values）的关注程度
        # embed_dim 现在是实体嵌入的维度 ENTITY_EMBED_DIM
        self.attention = MultiheadAttention(embed_dim=ENTITY_EMBED_DIM,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)  # 输入输出格式为 (Batch, Seq, Feature)
        # 在Attention之后和残差连接之前使用LayerNorm稳定训练
        self.attn_layer_norm = LayerNorm(ENTITY_EMBED_DIM)

        # --- 3. <<< 修改 >>> GRU 时序建模层 ---
        # 作用: 处理经过注意力加权后的飞机状态序列，捕捉时间动态
        # GRU的输入是飞机上下文感知的嵌入向量，维度为 ENTITY_EMBED_DIM
        self.gru = GRU(ENTITY_EMBED_DIM, self.rnn_hidden_size, batch_first=True)

        # --- 4. 定义 GRU 之后的 MLP 塔楼 (Post-GRU Towers) ---
        # 作用: 解码GRU输出的时序信息，分别生成连续和离散动作的决策依据
        # (这部分结构可以保持不变，因为它的输入是GRU的输出)
        post_gru_mlp_layers = 2  # 假设总共有4层MLP，GRU后用2层
        post_gru_dims = ACTOR_PARA.model_layer_dim[-post_gru_mlp_layers:]

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

        # --- 5. 定义最终的输出头 (Heads) (保持不变) ---
        # 作用: 将塔楼的输出转换为最终的动作分布参数
        self.mu_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)
        # 连续动作的对数标准差，作为可学习的参数，与状态无关
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))

        # --- 6. 优化器设置 (分组学习率) ---
        gru_params, attention_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'gru' in name.lower():
                gru_params.append(param)
            elif any(key in name.lower() for key in ['attention', 'attn', 'layer_norm']):
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
        """
        前向传播方法。
        :param obs: 状态观测值，形状为 (B, D) 或 (B, S, D)
        :param hidden_state: GRU 的隐藏状态
        :return: (包含动作分布的字典, 新的 GRU 隐藏状态, 实体注意力权重)
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)  # (B, D) -> (B, 1, D)
        B, S, D = obs_tensor.shape  # B:批大小, S:序列长度, D:特征维度

        # --- <<< 核心修改: 阶段一 - 实体编码 >>> ---
        # 1. 从扁平化的观测中切分出各个实体的部分
        # 假设 obs 顺序: m1_obs (3), m2_obs (3), ac_obs (6)
        missile1_obs_seq = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs_seq = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs_seq = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        # 2. 将每个实体的观测序列通过各自的编码器，得到高维嵌入
        # (B, S, Feat_Dim) -> (B, S, Embed_Dim)
        m1_embed_seq = self.missile_encoder(missile1_obs_seq)
        m2_embed_seq = self.missile_encoder(missile2_obs_seq)
        ac_embed_seq = self.aircraft_encoder(aircraft_obs_seq)

        # --- <<< 核心修改: 阶段二 - 跨实体注意力 >>> ---
        # 3. 准备 Attention 输入
        # Query: 来自飞机,因为它需要决定关注哪个导弹
        # Keys/Values: 来自导弹,因为它们是被关注的对象
        query = ac_embed_seq

        # 将导弹嵌入堆叠成一个序列，方便处理
        # (B, S, Embed_Dim) -> (B, S, 2, Embed_Dim)
        missile_embeds_seq = torch.stack([m1_embed_seq, m2_embed_seq], dim=2)
        keys = missile_embeds_seq
        values = missile_embeds_seq

        # 为了使用 batch_first 的 MultiheadAttention, 我们需要将 B 和 S 合并
        # 这样可以一次性处理所有时间步的注意力计算
        query = query.contiguous().view(B * S, 1, -1)  # (B*S, 1, Embed_Dim)
        keys = keys.contiguous().view(B * S, NUM_MISSILES, -1)  # (B*S, 2, Embed_Dim)
        values = values.contiguous().view(B * S, NUM_MISSILES, -1)  # (B*S, 2, Embed_Dim)

        # 4. 计算注意力
        # attn_output: (B*S, 1, Embed_Dim), attn_weights: (B*S, 1, 2)
        # attn_output是根据权重加权求和后的Value(导弹信息)
        # attn_weights是飞机对两个导弹的关注度
        attn_output, attn_weights = self.attention(query, keys, values)

        # 5. 残差连接和层归一化
        # 将注意力输出（融合了导弹信息）与原始飞机嵌入相加，让飞机在关注威胁的同时不丢失自身状态
        context_vector = self.attn_layer_norm(attn_output.squeeze(1) + query.squeeze(1))

        # 6. 恢复序列结构，为GRU做准备
        # (B*S, Embed_Dim) -> (B, S, Embed_Dim)
        contextualized_ac_seq = context_vector.view(B, S, -1)

        # --- 阶段三: 时序建模 (GRU) ---
        # 7. 将融合了威胁信息的飞机上下文序列送入 GRU
        gru_out, new_hidden = self.gru(contextualized_ac_seq, hidden_state)

        # --- 阶段四: 决策 (Post-GRU MLP Towers) (保持不变) ---
        base_features = gru_out
        continuous_features = self.continuous_tower(base_features)
        discrete_features = self.discrete_tower(base_features)

        if not is_sequence:
            continuous_features = continuous_features.squeeze(1)
            discrete_features = discrete_features.squeeze(1)

        mu = self.mu_head(continuous_features)
        all_disc_logits = self.discrete_head(discrete_features)

        # --- 阶段五: 创建概率分布 (保持不变) ---
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=-1)
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        # 动作屏蔽：如果没有干扰弹，则不能触发投放
        has_flares_info = obs_tensor[..., 10]  # 干扰弹数量在观测向量的第11个位置（索引10）
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            if mask.dim() < trigger_logits_masked.dim():
                mask = mask.unsqueeze(-1)
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min

        # 触发器层次控制：如果不投放，则其他离散动作参数无效
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
                    logits_sub[:] = NEG_INF
                    logits_sub[:, 0] = INF
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

        # <<< 新增 >>> 返回注意力权重，用于与奖励函数结合或进行分析
        # (B*S, 1, 2) -> (B, S, 2)
        final_attn_weights = attn_weights.view(B, S, -1)
        if not is_sequence:
            final_attn_weights = final_attn_weights.squeeze(1)

        return distributions, new_hidden, final_attn_weights


class Critic_GRU(Module):
    """
    Critic 网络 - [实体注意力版]
    与Actor采用几乎相同的结构，以保证价值评估和策略决策基于相同的信息处理流。
    """

    def __init__(self, dropout_rate=0.05, weight_decay=1e-4):
        """初始化 Critic 网络的模块、权重和优化器。"""
        super(Critic_GRU, self).__init__()
        # --- 基础参数定义 ---
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        self.rnn_hidden_size = RNN_HIDDEN_SIZE
        self.weight_decay = weight_decay

        # --- 1. 实体编码器 (与 Actor 共享结构) ---
        self.missile_encoder = Sequential(
            Linear(MISSILE_FEAT_DIM, ENTITY_EMBED_DIM // 2), LeakyReLU(),
            Linear(ENTITY_EMBED_DIM // 2, ENTITY_EMBED_DIM)
        )
        self.aircraft_encoder = Sequential(
            Linear(AIRCRAFT_FEAT_DIM, ENTITY_EMBED_DIM // 2), LeakyReLU(),
            Linear(ENTITY_EMBED_DIM // 2, ENTITY_EMBED_DIM)
        )

        # --- 2. 实体间自注意力层 ---
        self.attention = MultiheadAttention(embed_dim=ENTITY_EMBED_DIM,
                                            num_heads=ATTN_NUM_HEADS,
                                            dropout=0.0,
                                            batch_first=True)
        self.attn_layer_norm = LayerNorm(ENTITY_EMBED_DIM)

        # --- 3. GRU 时序建模层 ---
        self.gru = GRU(ENTITY_EMBED_DIM, self.rnn_hidden_size, batch_first=True)

        # --- 4. 定义 GRU 之后的 MLP (Post-GRU MLP) ---
        # Critic 使用一个统一的MLP即可
        post_gru_mlp_layers = 2
        post_gru_dims = CRITIC_PARA.model_layer_dim[-post_gru_mlp_layers:]
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
            elif any(key in name.lower() for key in ['attention', 'attn', 'layer_norm']):
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
        """前向传播，计算状态价值。"""
        obs_tensor = check(obs).to(**CRITIC_PARA.tpdv)
        is_sequence = obs_tensor.dim() == 3
        if not is_sequence:
            obs_tensor = obs_tensor.unsqueeze(1)
        B, S, D = obs_tensor.shape

        # --- 阶段一和二: 实体编码与注意力 (与 Actor 完全相同) ---
        missile1_obs_seq = obs_tensor[..., 0:MISSILE_FEAT_DIM]
        missile2_obs_seq = obs_tensor[..., MISSILE_FEAT_DIM:2 * MISSILE_FEAT_DIM]
        aircraft_obs_seq = obs_tensor[..., 2 * MISSILE_FEAT_DIM:]

        m1_embed_seq = self.missile_encoder(missile1_obs_seq)
        m2_embed_seq = self.missile_encoder(missile2_obs_seq)
        ac_embed_seq = self.aircraft_encoder(aircraft_obs_seq)

        query = ac_embed_seq.contiguous().view(B * S, 1, -1)
        missile_embeds_seq = torch.stack([m1_embed_seq, m2_embed_seq], dim=2)
        keys = missile_embeds_seq.contiguous().view(B * S, NUM_MISSILES, -1)
        values = missile_embeds_seq.contiguous().view(B * S, NUM_MISSILES, -1)

        attn_output, _ = self.attention(query, keys, values)
        context_vector = self.attn_layer_norm(attn_output.squeeze(1) + query.squeeze(1))
        contextualized_ac_seq = context_vector.view(B, S, -1)

        # --- 阶段三: 时序建模 (GRU) ---
        gru_out, new_hidden = self.gru(contextualized_ac_seq, hidden_state)

        # --- 阶段四: 价值评估 (Post-GRU MLP) ---
        post_gru_features = self.post_gru_mlp(gru_out)
        value = self.fc_out(post_gru_features)

        if not is_sequence:
            value = value.squeeze(1)

        return value, new_hidden


# ==============================================================================
# PPO Agent 主类 (已适配实体注意力模型)
# ==============================================================================
class PPO_continuous(object):
    """
    PPO 智能体主类。
    """

    def __init__(self, load_able: bool, model_dir_path: str = None, use_rnn: bool = True):
        super(PPO_continuous, self).__init__()
        self.use_rnn = use_rnn
        if self.use_rnn:
            print("--- 初始化 PPO Agent (使用 [实体注意力 -> GRU -> MLP] 模型) ---")
            self.Actor = Actor_GRU()
            self.Critic = Critic_GRU()
        else:
            # 保留纯MLP作为备选方案
            print("--- 初始化 PPO Agent (使用 MLP 模型) ---")
            # self.Actor = Actor()
            # self.Critic = Critic()
            raise NotImplementedError("MLP path is not fully supported in this version.")

        self.buffer = Buffer(use_rnn=self.use_rnn, use_attn=True)  # <<< 修改 >>> 告知Buffer需要存储注意力权重
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        # ... (存储路径等初始化代码保持不变) ...
        self.training_start_time = time.strftime("PPO_EntityATT_GRU_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../save/save_evade_fuza两个导弹"
        win_rate_subdir = "胜率模型"
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)
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


    def store_experience(self, state, action, probs, value, reward, done, actor_hidden=None, critic_hidden=None,
                         attn_weights=None):
        """将单步经验（包括注意力权重）存储到 Buffer 中。"""
        if not np.all(np.isfinite(probs)):
            raise ValueError("在 log_prob 中检测到 NaN/Inf！")
        if self.use_rnn and (actor_hidden is None or critic_hidden is None):
            raise ValueError("使用 RNN 模型时必须存储隐藏状态！")
        # <<< 核心修改 >>>: 调用 Buffer 的 store_transition，它现在能处理 attn_weights
        self.buffer.store_transition(state, value, action, probs, reward, done, actor_hidden, critic_hidden,
                                     attn_weights)

    def choose_action(self, state, actor_hidden, critic_hidden, deterministic=False):
        """
        根据当前状态选择动作，并返回实体注意力权重。
        :return: (环境动作, 存储动作, 对数概率, 状态价值, 新actor隐藏状态, 新critic隐藏状态, 注意力权重)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            if self.use_rnn:
                value, new_critic_hidden = self.Critic(state_tensor, critic_hidden)
                # <<< 核心修改 >>>: Actor 现在返回三个值
                dists, new_actor_hidden, attention_weights = self.Actor(state_tensor, actor_hidden)
            else:
                # MLP 分支
                value = self.Critic(state_tensor)
                dists = self.Actor(state_tensor)
                new_critic_hidden, new_actor_hidden, attention_weights = None, None, None

            # 从连续动作分布中采样 (逻辑不变)
            continuous_base_dist = dists['continuous']
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            action_cont_tanh = torch.tanh(u)
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 从所有离散动作分布中采样 (逻辑不变)
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

            # 计算总的对数概率 (逻辑不变)
            log_prob_disc = sum(dists[key].log_prob(act) for key, act in sampled_actions_dict.items())
            total_log_prob = log_prob_cont + log_prob_disc

            # 准备要存储和发送的动作 (逻辑不变)
            action_disc_to_store = torch.stack(list(sampled_actions_dict.values()), dim=-1).float()
            action_to_store = torch.cat([u, action_disc_to_store], dim=-1)
            env_action_cont = self.scale_action(action_cont_tanh)
            final_env_action_tensor = torch.cat([env_action_cont, action_disc_to_store], dim=-1)

            # 将所有结果转换为 numpy 数组 (逻辑不变)
            value_np = value.cpu().numpy()
            action_to_store_np = action_to_store.cpu().numpy()
            log_prob_to_store_np = total_log_prob.cpu().numpy()
            final_env_action_np = final_env_action_tensor.cpu().numpy()

            # <<< 新增 >>>: 处理注意力权重的返回
            attention_weights_np = attention_weights.cpu().numpy() if attention_weights is not None else None

            # 如果输入不是批处理，则移除批次维度 (逻辑不变)
            if not is_batch:
                final_env_action_np = final_env_action_np[0]
                action_to_store_np = action_to_store_np[0]
                log_prob_to_store_np = log_prob_to_store_np[0]
                value_np = value_np[0]
                if attention_weights_np is not None:
                    attention_weights_np = attention_weights_np[0]

        # <<< 核心修改 >>>: 返回值增加 attention_weights_np
        return final_env_action_np, action_to_store_np, log_prob_to_store_np, value_np, new_actor_hidden, new_critic_hidden, attention_weights_np

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        # (此函数代码无需修改)
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
        执行 PPO 的学习和更新步骤。已适配实体注意力 RNN 模式。
        """
        if self.buffer.get_buffer_size() < BUFFERPARA.BATCH_SIZE:
            return None

        # <<< 核心修改 >>>: Buffer 现在返回 attn_weights
        states, values, actions, old_probs, rewards, dones, actor_hiddens, critic_hiddens, attn_weights = self.buffer.get_all_data()

        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        values = np.squeeze(values)
        returns = advantages + values
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'entropy_cont': [], 'adv_targ': [],
                      'ratio': []}

        for _ in range(self.ppo_epoch):
            if self.use_rnn:
                batch_generator = self.buffer.generate_sequence_batches(
                    SEQUENCE_LENGTH, BUFFERPARA.BATCH_SIZE, advantages, returns
                )
            else:
                batch_generator = self.buffer.generate_batches()

            for batch_data in batch_generator:
                if self.use_rnn:
                    # <<< 核心修改 >>>: 解包批次数据，包含 attn_weights
                    state, action_batch, old_prob, advantage, return_, initial_actor_h, initial_critic_h, old_attn_weights = batch_data
                    state = check(state).to(**ACTOR_PARA.tpdv)
                    action_batch = check(action_batch).to(**ACTOR_PARA.tpdv)
                    old_prob = check(old_prob).to(**ACTOR_PARA.tpdv)
                    advantage = check(advantage).to(**ACTOR_PARA.tpdv)
                    return_ = check(return_).to(**CRITIC_PARA.tpdv)
                    initial_actor_h = check(initial_actor_h).to(**ACTOR_PARA.tpdv)
                    initial_critic_h = check(initial_critic_h).to(**CRITIC_PARA.tpdv)
                    old_attn_weights = check(old_attn_weights).to(**ACTOR_PARA.tpdv)  # 也转换为tensor
                else:  # MLP path
                    # ...
                    pass

                # 从批次动作中解析出连续和离散部分 (逻辑不变)
                u_from_buffer = action_batch[..., :CONTINUOUS_DIM]
                discrete_actions_from_buffer = {
                    'trigger': action_batch[..., CONTINUOUS_DIM],
                    'salvo_size': action_batch[..., CONTINUOUS_DIM + 1].long(),
                    'num_groups': action_batch[..., CONTINUOUS_DIM + 2].long(),
                    'inter_interval': action_batch[..., CONTINUOUS_DIM + 3].long(),
                }

                # 4. Actor (策略) 网络训练
                if self.use_rnn:
                    new_dists, _, new_attn_weights = self.Actor(state, initial_actor_h)
                else:
                    new_dists = self.Actor(state)
                    new_attn_weights = None

                # 计算新策略下，旧动作的对数概率 (逻辑不变)
                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc

                # 计算策略的熵 (逻辑不变)
                entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                total_entropy = (entropy_cont + sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )).mean()

                # 计算 PPO 裁剪目标函数 (逻辑不变)
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                advantage_squeezed = advantage.squeeze(-1) if advantage.dim() > ratio.dim() else advantage
                surr1 = ratio * advantage_squeezed
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage_squeezed
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 反向传播和优化 (逻辑不变)
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                # 5. Critic (价值) 网络训练 (逻辑不变)
                if self.use_rnn:
                    new_value, _ = self.Critic(state, initial_critic_h)
                else:
                    new_value = self.Critic(state)

                if new_value.dim() > return_.dim():
                    return_ = return_.unsqueeze(-1)

                critic_loss = torch.nn.functional.mse_loss(new_value, return_)

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

        self.buffer.clear_memory()
        for key in train_info:
            train_info[key] = np.mean(train_info[key])

        self.save()
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