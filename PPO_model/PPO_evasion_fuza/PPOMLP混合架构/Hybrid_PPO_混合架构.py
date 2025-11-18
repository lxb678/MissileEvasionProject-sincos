import torch
from torch import nn
from torch.nn import *
# <<< 更改 >>> 导入 Categorical 分布用于多分类离散动作
from torch.distributions import Bernoulli, Categorical
from torch.distributions import Normal
from Interference_code.PPO_model.PPO_evasion_fuza.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza.Buffer import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re
import time
from torch.cuda.amp import GradScaler, autocast

# --- 动作空间配置 ---
# 定义连续动作的维度
CONTINUOUS_DIM = 4  # 油门, 升降舵, 副翼, 方向舵
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']

# <<< 更改 >>> 根据新要求重新定义多离散动作空间
# 这个字典定义了策略网络需要做出的所有离散决策
DISCRETE_DIMS = {
    'flare_trigger': 1,  # 是否投放: 1个logit -> Bernoulli (是/否)
    'salvo_size': 3,  # 一组的数量: 3个logit -> Categorical (例如: 2, 4, 6 发)
    # 'intra_interval': 3,  # 组内每发间隔: 3个logit -> Categorical (例如: 0.1s, 0.2s, 0.5s)
    'num_groups': 3,  # 投放组数: 3个logit -> Categorical (例如: 1, 2, 3 组)
    'inter_interval': 3,  # 组间隔:   3个logit -> Categorical (例如: 0.5s, 1.0s, 2.0s)
}
# <<< 更改 >>> 重新计算离散部分总共需要的网络输出数量
TOTAL_DISCRETE_LOGITS = sum(DISCRETE_DIMS.values())  # 1 + 3 + 3 + 3 + 3 = 13

# <<< 更改 >>> 重新计算存储在 Buffer 中动作的总维度
# 存储的动作包括：4个连续动作的原始值(u) + 5个离散动作的采样索引
TOTAL_ACTION_DIM_BUFFER = CONTINUOUS_DIM + len(DISCRETE_DIMS)  # 4 + 5 = 9

# <<< 新增 >>> 离散动作索引到物理值的映射 (非常重要！)
# 这个映射表用于在环境端将智能体输出的索引(0, 1, 2)转换为JSBSim可以理解的实际参数。
# 例如，如果智能体为'salvo_size'选择了索引1，环境会将其解释为投放4发。
# DISCRETE_ACTION_MAP = {
#     'salvo_size': [2, 4, 6],  # index 0 -> 2发, index 1 -> 4发, index 2 -> 6发
#     'intra_interval': [0.1, 0.2, 0.5],  # index 0 -> 0.1s 间隔, ...
#     'num_groups': [1, 2, 3],  # index 0 -> 1组, ...
#     'inter_interval': [0.5, 1.0, 2.0]  # index 0 -> 0.5s 组间隔, ...
# }
DISCRETE_ACTION_MAP = {
#     'salvo_size': [2, 4, 6],          # 投放数量：低、中、高强度
#     'intra_interval': [0.02, 0.04, 0.1], # 组内间隔(秒)：密集(欺骗)、标准、稍疏
# # 'intra_interval': [0.05, 0.1, 0.2], # 组内间隔(秒)：密集(欺骗)、标准、稍疏
#     'num_groups': [1, 2, 3],          # 投放组数：单次反应、标准程序、持续程序
#     'inter_interval': [0.5, 1.0, 2.0]  # 组间间隔(秒)：紧急连续、标准连续、预防/遮蔽
#     'salvo_size': [1, 2, 3],  # 修改为发射1、2、3枚
#     # 'intra_interval': [0.05, 0.1, 0.15],
#     'intra_interval': [0.02, 0.04, 0.08],
#     'num_groups': [1, 2, 3],
#     'inter_interval': [0.2, 0.5, 1.0]
    'salvo_size': [2, 3, 4],  # 修改为发射2、3、4枚
    # 'intra_interval': [0.05, 0.1, 0.15],
    # 'intra_interval': [0.02, 0.04, 0.06],
    'num_groups': [2, 3, 4],
    'inter_interval': [0.2, 0.4, 0.6]
}
# <<< 新增 >>> 定义固定的组内投放间隔
# FIXED_INTRA_INTERVAL = 0.05

# 'salvo_size': [2, 3, 4],  # 修改为发射2、3、4枚
#     'intra_interval': [0.05],
#     'num_groups': [2, 3, 4],
#     'inter_interval': [0.2, 0.4, 0.6]

# --- 动作范围定义 (仅用于连续动作缩放) ---
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},
    'elevator': {'low': -1.0, 'high': 1.0},
    'aileron': {'low': -1.0, 'high': 1.0},
    'rudder': {'low': -1.0, 'high': 1.0},
}

def init_weights(m):
    """
    一个通用的权重初始化函数。
    """
    if isinstance(m, Linear):
        # 对线性层使用 Kaiming Normal 初始化，适用于 LeakyReLU
        torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Actor(Module):
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

        # --- 混合架构定义 ---

        # 1. 为每个部分定义层的维度
        #    !! 重要：请根据你的 Config.py 来调整这里的切片 !!
        #    假设 model_layer_dim = [256, 256，256]，我们在第2层后拆分
        split_point = 2  # 在第2层后拆分
        base_dims = ACTOR_PARA.model_layer_dim[:split_point]  # 例如: [256，256]
        # tower_dims = ACTOR_PARA.model_layer_dim[split_point:]  # 例如: [256]
        continuous_tower_dims = ACTOR_PARA.model_layer_dim[split_point:] #[256]  # 连续动作塔楼的维度
        #【修正后的代码】让离散塔楼的每一层维度都是连续塔楼对应层的一半
        # 使用列表推导式来实现
        discrete_tower_dims = continuous_tower_dims
        # discrete_tower_dims = [dim // 2 for dim in continuous_tower_dims]  # 例如: [128, 64]

        # 2. 构建共享基座网络 (Shared Base)
        self.shared_base = Sequential()
        base_input_dim = self.input_dim
        for i, dim in enumerate(base_dims):
            self.shared_base.add_module(f'base_fc_{i}', Linear(base_input_dim, dim))
            self.shared_base.add_module(f'base_leakyrelu_{i}', LeakyReLU())
            base_input_dim = dim

        # 共享基座的输出维度
        base_output_dim = base_dims[-1]

        # 3. 构建连续动作塔楼 (Continuous Tower)
        self.continuous_tower = Sequential()
        tower_input_dim = base_output_dim
        for i, dim in enumerate(continuous_tower_dims):
            self.continuous_tower.add_module(f'cont_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.continuous_tower.add_module(f'cont_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim

        # 4. 构建离散动作塔楼 (Discrete Tower)
        self.discrete_tower = Sequential()
        tower_input_dim = base_output_dim  # 输入同样来自共享基座
        for i, dim in enumerate(discrete_tower_dims):
            self.discrete_tower.add_module(f'disc_tower_fc_{i}', Linear(tower_input_dim, dim))
            self.discrete_tower.add_module(f'disc_tower_leakyrelu_{i}', LeakyReLU())
            tower_input_dim = dim

        # 5. 定义最终的输出头 (Heads)
        #    头部的输入维度现在是它们各自塔楼的输出维度
        continuous_tower_output_dim = continuous_tower_dims[-1] if continuous_tower_dims else base_output_dim
        self.continuous_head = Linear(continuous_tower_output_dim, CONTINUOUS_DIM)
        discrete_tower_output_dim = discrete_tower_dims[-1] if discrete_tower_dims else base_output_dim
        self.discrete_head = Linear(discrete_tower_output_dim, TOTAL_DISCRETE_LOGITS)

        # 6. 定义与状态无关的连续动作 log_std
        # self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), -0.5))
        self.log_std_param = torch.nn.Parameter(torch.full((1, CONTINUOUS_DIM), 0.0))

        # --- 优化器和其他设置 (保持不变) ---
        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)
    def init_model(self):
        """初始化神经网络结构"""
        self.network = Sequential()
        layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim + [self.output_dim]
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            if i < len(layers_dims) - 2:
                self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())

    def forward(self, obs):
        """
        前向传播方法，为每个动作维度创建并返回一个概率分布。
        """
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 数据首先流经共享基座
        base_features = self.shared_base(obs_tensor)

        # 2. 共享特征被分别送入两个专用塔楼
        continuous_features = self.continuous_tower(base_features)
        discrete_features = self.discrete_tower(base_features)

        # 3. 每个头部接收来自其专属塔楼的特征
        mu = self.continuous_head(continuous_features)
        all_disc_logits = self.discrete_head(discrete_features)

        # 3. 按 DISCRETE_DIMS 结构将所有logits切分为5个部分
        split_sizes = list(DISCRETE_DIMS.values())
        logits_parts = torch.split(all_disc_logits, split_sizes, dim=1)

        # # 为每个部分分配合理的变量名
        # trigger_logits, salvo_size_logits, intra_interval_logits, num_groups_logits, inter_interval_logits = logits_parts
        # <<< 修改 >>> 变量解包，现在只有4个部分
        trigger_logits, salvo_size_logits, num_groups_logits, inter_interval_logits = logits_parts

        # 4. 【动作掩码】逻辑 (只作用于触发器，如果没诱饵弹则不能投放)
        # 假设 obs_tensor 的第 7 个特征 (索引为7) 代表红外诱饵弹数量
        has_flares_info = obs_tensor[:, 11]
        mask = (has_flares_info == 0)
        trigger_logits_masked = trigger_logits.clone()
        if torch.any(mask):
            # 将没有诱饵弹的样本对应的 logit 设置为一个极小值，阻止选择该动作
            trigger_logits_masked[mask] = torch.finfo(torch.float32).min
            # --- 这是修改后的正确代码 ---
            # 获取 trigger_logits_masked 张量自身的数据类型 (dtype)
            # 然后获取该 dtype 对应的极小值
            # fill_value = torch.finfo(trigger_logits_masked.dtype).min
            # trigger_logits_masked[mask] = fill_value
            # 使用一个在 FP16 表示范围内且足够小的安全值
            # trigger_logits_masked[mask] = -1e4  # -10000.0

        # ===============================================================
        # 6️⃣ <<< 新增：层级控制逻辑 >>>
        # 当“不投放”时，屏蔽其他离散动作的 logits
        # ===============================================================
        # 先根据触发器的 logits 计算出其概率
        trigger_probs = torch.sigmoid(trigger_logits_masked)  # shape: [B,1]

        # 如果 trigger_probs < 0.5，说明模型倾向于“不投放”
        # 我们用这个条件生成一个 mask（True=不投放）
        no_trigger_mask = (trigger_probs < 0.5).squeeze(-1)  # shape: [B]

        # 创建 logits 的副本，避免原地操作污染梯度
        salvo_size_logits_masked = salvo_size_logits.clone()
        # intra_interval_logits_masked = intra_interval_logits.clone()
        num_groups_logits_masked = num_groups_logits.clone()
        inter_interval_logits_masked = inter_interval_logits.clone()
        # ===============================================================
        # 当 trigger 不投放时，将其他离散动作 logits 强制为 index=0 (one-hot 形式)
        # ===============================================================
        if torch.any(no_trigger_mask):
            # print("<<<<<<<<<<<<<<<<<< 警告：强制Mask逻辑被触发！ >>>>>>>>>>>>>>>>>>")
            INF = 1e6
            NEG_INF = -1e6
            # 遍历所有依赖于触发器的 logits 张量
            for logits_tensor in [
                salvo_size_logits_masked,
                # intra_interval_logits_masked,
                num_groups_logits_masked,
                inter_interval_logits_masked,
            ]:
                # 选出那些需要被屏蔽的行 (样本)
                logits_sub = logits_tensor[no_trigger_mask]
                # 如果确实有需要屏蔽的行
                if logits_sub.numel() > 0:
                    # 将这些行的所有 logits 都设为极小值
                    logits_sub[:] = NEG_INF  # 全部置为极小值
                    # 然后只把第 0 列 (对应索引 0) 的 logit 设为极大值
                    logits_sub[:, 0] = INF  # 仅 index=0 置为极大值
                    # 将修改后的 logits 写回原张量
                    logits_tensor[no_trigger_mask] = logits_sub

        # 5. 创建所有动作分布对象
        # 5.1 连续动作分布
        # 使用全局可学习的 log_std 参数
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        continuous_base_dist = Normal(mu, std)

        # 5.2  创建新的5个离散分布
        trigger_dist = Bernoulli(logits=trigger_logits_masked.squeeze(-1))
        salvo_size_dist = Categorical(logits=salvo_size_logits_masked)
        # intra_interval_dist = Categorical(logits=intra_interval_logits_masked)
        num_groups_dist = Categorical(logits=num_groups_logits_masked)
        inter_interval_dist = Categorical(logits=inter_interval_logits_masked)

        # 6.返回包含所有新分布的字典
        distributions = {
            'continuous': continuous_base_dist,
            'trigger': trigger_dist,
            'salvo_size': salvo_size_dist,
            # 'intra_interval': intra_interval_dist,
            'num_groups': num_groups_dist,
            'inter_interval': inter_interval_dist
        }
        return distributions


class Critic(Module):
    """
    Critic 网络 (价值网络)，评估状态的价值 V(s)。
    这个类的结构不受动作空间变化的影响。
    """

    def __init__(self):
        super(Critic, self).__init__()
        self.input_dim = CRITIC_PARA.input_dim
        self.output_dim = CRITIC_PARA.output_dim
        # 1. 定义网络结构
        self.network = Sequential()
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            # self.network.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
        # 添加最后的输出层
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

        # # --- [新增] 应用参数初始化 ---
        # # 首先对所有层应用通用的初始化策略
        # self.network.apply(init_weights)
        #
        # # 然后对输出层(fc_out)进行特殊的、小范围的初始化，以保证初始价值估计的稳定性
        # # self.network[-1] 指的是 Sequential 中的最后一个模块，即 fc_out
        # init_range = 3e-3
        # self.network[-1].weight.data.uniform_(-init_range, init_range)
        # self.network[-1].bias.data.fill_(0)
        # # --- 初始化结束 ---

        # 2. 定义优化器
        self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        self.critic_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(CRITIC_PARA.device)

        # self.init_model()
        # self.optim = torch.optim.Adam(self.parameters(), CRITIC_PARA.lr)
        # self.critic_scheduler = lr_scheduler.LinearLR(
        #     self.optim,
        #     start_factor=1.0,
        #     end_factor=AGENTPARA.mini_lr / CRITIC_PARA.lr,
        #     total_iters=AGENTPARA.MAX_EXE_NUM
        # )
        # self.to(CRITIC_PARA.device)

    def init_model(self):
        """初始化 Critic 网络结构，并加入 LayerNorm 以增强稳定性"""
        self.network = Sequential()

        # 1. 定义所有层的维度列表，从输入层到最后一个隐藏层
        # 例如: input_dim=10, model_layer_dim=[256, 256]
        # -> layers_dims 会是 [10, 256, 256]
        layers_dims = [self.input_dim] + CRITIC_PARA.model_layer_dim

        # 2. 循环构建所有的隐藏层
        # 这个循环将构建从 input->256 和 256->256 的部分
        for i in range(len(layers_dims) - 1):
            # 添加线性层
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))

            # 在线性层之后、激活函数之前添加 LayerNorm
            # LayerNorm 的维度是其前面线性层的输出维度
            # self.network.add_module(f'LayerNorm_{i}', LayerNorm(layers_dims[i + 1]))

            # 添加激活函数
            self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # 3. 单独添加最后的输出层
        # 它的输入是最后一个隐藏层的维度，输出是 self.output_dim (通常为1)
        # 最后的输出层通常不需要 LayerNorm 或激活函数
        self.network.add_module('fc_out', Linear(layers_dims[-1], self.output_dim))

    def forward(self, obs):
        """前向传播，计算状态价值"""
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


class PPO_continuous(object):
    """
    PPO 智能体主类，整合 Actor 和 Critic 并实现 PPO 算法核心逻辑。
    """

    def __init__(self, load_able: bool, model_dir_path: str = None):
        super(PPO_continuous, self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()


        # 只需要在 agent 初始化时创建一次
        self.actor_scaler = GradScaler()
        self.critic_scaler = GradScaler()

        self.buffer = Buffer()
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        self.training_start_time = time.strftime("PPO_%Y-%m-%d_%H-%M-%S")
        self.base_save_dir = "../../../save/save_evade_fuza"
        win_rate_subdir = "胜率模型"
        # 拼接成完整的存档路径
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)
        self.win_rate_dir = os.path.join(self.run_save_dir, win_rate_subdir)

        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                self.load_models_from_directory(model_dir_path)
            else:
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../../../test/test_evade")

    def load_models_from_directory(self, directory_path: str):
        """从指定的文件夹路径加载模型，能自动识别多种命名格式。"""
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

    def store_experience(self, state, action, probs, value, reward, done):
        """存储经验到 Buffer，并在存储前进行数值检查。"""
        if not np.all(np.isfinite(probs)):
            print("=" * 50)
            print(f"!!! 严重错误: 在 log_prob 中检测到非有限值 (NaN/Inf) !!!")
            print(f"Log_prob 值: {probs}")
            print(f"导致错误的状态: {state}")
            print(f"导致错误的动作: {action}")
            print("=" * 50)
            raise ValueError("在 log_prob 中检测到 NaN/Inf！")
        self.buffer.store_transition(state, value, action, probs, reward, done)

    def scale_action(self, action_cont_tanh):
        """将tanh压缩后的连续动作 [-1, 1] 缩放到环境的实际物理范围。"""
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)
        return scaled_action

    def check_numerics(self, name, tensor, state=None, action=None, threshold=1e4):
        """检查 tensor 是否存在 NaN、Inf 或异常大值，用于调试。"""
        arr = tensor.detach().cpu().numpy()
        if not np.all(np.isfinite(arr)):
            print(f"[数值错误] {name} 出现 NaN/Inf. 值: {arr}")
            if state is not None: print(f"对应 state: {state}")
            if action is not None: print(f"对应 action: {action}")
            raise ValueError(f"NaN/Inf detected in {name}")
        if np.any(np.abs(arr) > threshold):
            print(f"[警告] {name} 数值过大 (> {threshold}). 值: {arr.max()}, {arr.min()}")

    def map_discrete_actions(self, discrete_actions_indices):
        """
        将离散动作的索引张量映射到其物理值。
        Args:
            discrete_actions_indices (dict): 一个包含各离散动作索引张量的字典。
        Returns:
            np.ndarray: 一个包含所有离散动作物理值的 numpy 数组。
        """
        # 0. 获取触发器动作 (形状: [B])
        trigger_action = discrete_actions_indices['trigger'].cpu().numpy()

        # 1. 初始化一个用于存放物理值的数组 (形状: [B, 4])
        # 对应 salvo_size, intra_interval, num_groups, inter_interval
        batch_size = trigger_action.shape[0]
        physical_actions = np.zeros((batch_size, 4), dtype=np.float32)

        # 2. 遍历每个 Categorical 动作
        action_keys = ['salvo_size', 'intra_interval', 'num_groups', 'inter_interval']
        for i, key in enumerate(action_keys):
            # 获取该动作的索引 (形状: [B])
            indices = discrete_actions_indices[key].cpu().numpy()
            # 获取映射表
            mapping = DISCRETE_ACTION_MAP[key]
            # 使用索引从映射表中查找物理值
            physical_actions[:, i] = np.array([mapping[idx] for idx in indices])

        # 3. 应用触发器逻辑：如果 trigger=0，则所有其他参数无效 (可以设为0)
        # 触发器为0的位置的掩码 (broadcastable to physical_actions)
        trigger_mask = (trigger_action == 0)[:, np.newaxis]
        physical_actions[np.repeat(trigger_mask, 4, axis=1)] = 0.0

        # 4. 最终将 trigger (0/1) 和其他物理值组合起来
        # 最终输出的离散动作数组 (形状: [B, 5])
        final_discrete_env_actions = np.hstack([
            trigger_action[:, np.newaxis],
            physical_actions
        ])

        return final_discrete_env_actions

    def choose_action(self, state, deterministic=False):
        """
        根据当前状态选择动作，这是与环境交互的核心。
        - 支持确定性/随机性采样。
        - 支持批处理。
        - 当不投放诱饵弹时，智能地将相关参数置零。
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # 1. 从Actor获取包含所有动作分布的字典
            dists = self.Actor(state_tensor)

            # 2. 处理连续动作
            continuous_base_dist = dists['continuous']
            # 采样原始动作 u (pre-tanh)
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            # 计算 squashed 后的动作 a (post-tanh)
            action_cont_tanh = torch.tanh(u)
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # # --- 💥 新的、正确的 log_prob 计算 (带雅可比修正) ---
            # # 基础的高斯 log_prob: log p(u)
            # log_prob_u = continuous_base_dist.log_prob(u)
            #
            # # 稳定的雅可比修正项: log(1 - tanh(u)^2)
            # # 使用公式: 2 * (log(2) - u - softplus(-2*u))
            # # torch.nn.functional.softplus(x) = log(1 + exp(x))
            # log_prob_correction = 2 * (np.log(2.0) - u - torch.nn.functional.softplus(-2 * u))
            #
            # # 最终的 log_prob: log p(a) = log p(u) - log(1 - a^2)
            # log_prob_cont = (log_prob_u - log_prob_correction).sum(dim=-1)

            # # --- 为了调试，我们提前获取 mean 和 log_std ---
            # action_mean, log_std = continuous_base_dist.mean, continuous_base_dist.scale.log()
            #
            # u = action_mean if deterministic else continuous_base_dist.rsample()
            # action_cont_tanh = torch.tanh(u)
            # log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)

            # 3. 从所有5个离散分布中采样或选择最优动作
            sampled_actions_dict = {}
            for key, dist in dists.items():
                if key == 'continuous':
                    continue
                if deterministic:
                    if isinstance(dist, Categorical):
                        # 选择概率最高的动作索引
                        sampled_actions_dict[key] = torch.argmax(dist.probs, dim=-1)
                        # sampled_actions_dict[key] = dist.sample()
                    elif isinstance(dist, Bernoulli):
                        # 选择概率大于0.5的动作 (0或1)
                        sampled_actions_dict[key] = (dist.probs > 0.5).float()
                        # 按概率采样 (0或1)
                        # sampled_actions_dict[key] = dist.sample()
                else:
                    # 随机采样
                    sampled_actions_dict[key] = dist.sample()

            # 为了方便后续使用，将字典中的动作解包到单独的变量
            trigger_action = sampled_actions_dict['trigger']
            salvo_size_action = sampled_actions_dict['salvo_size']
            # intra_interval_action = sampled_actions_dict['intra_interval']
            num_groups_action = sampled_actions_dict['num_groups']
            inter_interval_action = sampled_actions_dict['inter_interval']

            # 4. 计算并加总所有5个离散动作的对数概率
            log_prob_disc = (dists['trigger'].log_prob(trigger_action) +
                             dists['salvo_size'].log_prob(salvo_size_action) +
                             # dists['intra_interval'].log_prob(intra_interval_action) +
                             dists['num_groups'].log_prob(num_groups_action) +
                             dists['inter_interval'].log_prob(inter_interval_action))

            # 5. 计算总的对数概率
            total_log_prob = log_prob_cont + log_prob_disc

            # 6. 准备要存入Buffer的动作向量 (存储原始值u和原始采样索引)
            action_disc_to_store = torch.stack([
                trigger_action, salvo_size_action, #intra_interval_action,
                num_groups_action, inter_interval_action
            ], dim=-1).float()
            action_to_store = torch.cat([u, action_disc_to_store], dim=-1)

            # 7. 准备发送到环境的最终动作向量 (应用置零逻辑)
            env_action_cont = self.scale_action(action_cont_tanh)

            # 创建一个掩码，用于识别哪些样本的 trigger_action 为 0
            zero_mask = (trigger_action == 0)

            # 使用 clone() 避免原地修改影响 buffer 中存储的动作
            env_salvo_size_action = salvo_size_action.clone()
            # env_intra_interval_action = intra_interval_action.clone()
            env_num_groups_action = num_groups_action.clone()
            env_inter_interval_action = inter_interval_action.clone()

            # <<< 新增 >>> 创建一个与其它动作形状相同、值为固定值的张量
            # fixed_intra_interval_action = torch.full_like(trigger_action, FIXED_INTRA_INTERVAL, dtype=torch.float32)

            # print(fixed_intra_interval_action)

            # 应用掩码，将不投放的样本的参数置零
            env_salvo_size_action[zero_mask] = 0
            # env_intra_interval_action[zero_mask] = 0
            env_num_groups_action[zero_mask] = 0
            env_inter_interval_action[zero_mask] = 0

            # 拼接成最终发送给环境的离散动作部分
            final_env_action_disc = torch.stack([
                trigger_action, env_salvo_size_action, #fixed_intra_interval_action, #env_intra_interval_action,
                env_num_groups_action, env_inter_interval_action
            ], dim=-1).float()

            # 拼接连续和离散部分
            final_env_action_tensor = torch.cat([env_action_cont, final_env_action_disc], dim=-1)

        # 8. 将所有结果转换为Numpy数组
        action_to_store_np = action_to_store.cpu().numpy()
        log_prob_to_store_np = total_log_prob.cpu().numpy()
        final_env_action_np = final_env_action_tensor.cpu().numpy()

    #     # ###############################################################
    #     # #                  在这里插入我们的诊断代码                     #
    #     # ###############################################################
    #
    #     print("\n" + "=" * 20 + " ACTION DEBUG " + "=" * 20)
    #     # 1) 查看 mean/std 分布
    #     print("mean (mu): mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
    #         action_mean.mean().item(), action_mean.std().item(), action_mean.min().item(), action_mean.max().item()
    #     ))
    #     print("log_std:   mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
    #         log_std.mean().item(), log_std.std().item(), log_std.min().item(), log_std.max().item()
    #     ))
    #
    #     # 2) 查看 u 的范围（未 squash）
    #     u_np = u.detach().cpu().numpy().ravel()
    #     print("u (pre-tanh): min {:.3f}, max {:.3f}, mean_abs {:.3f}".format(
    #         u_np.min(), u_np.max(), np.abs(u_np).mean()
    #     ))
    #
    #     # 3) 查看 tanh 后动作以及放大后范围
    #     act_tanh_np = action_cont_tanh.detach().cpu().numpy().ravel()
    #     print("action (post-tanh): min {:.3f}, max {:.3f}".format(
    #         act_tanh_np.min(), act_tanh_np.max()
    #     ))
    #
    #     # 由于你的 scale_action 是在 agent 内部实现的，这里我们直接打印缩放后的动作
    #     scaled_act_np = env_action_cont.detach().cpu().numpy().ravel()
    #     print("action (scaled for env): min {:.3f}, max {:.3f}".format(
    #         scaled_act_np.min(), scaled_act_np.max()
    #     ))
    #
    #     # 4) 统计被饱和的比例（接近 ±1）
    #     saturated_count = np.sum(np.abs(act_tanh_np) > 0.98)
    #     saturated_ratio = saturated_count / act_tanh_np.size
    #     print("Saturated ratio (>0.98): {:.2f}% ({}/{})".format(
    #         saturated_ratio * 100, saturated_count, act_tanh_np.size
    #     ))
    #     print("=" * 54 + "\n")
    # # ###############################################################

        # 如果输入不是批处理，则移除批次维度
        if not is_batch:
            final_env_action_np = final_env_action_np[0]
            action_to_store_np = action_to_store_np[0]
            log_prob_to_store_np = log_prob_to_store_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np

    def get_value(self, state):
        """使用 Critic 网络获取给定状态的价值。"""
        with torch.no_grad():
            value = self.Critic(state)
        return value

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        """计算广义优势估计 (GAE)。"""
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
        执行 PPO 的学习和更新步骤，已适配多部分离散动作空间。
        """
        torch.autograd.set_detect_anomaly(True)
        states, values, actions, old_probs, rewards, dones = self.buffer.sample()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)

        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [],'entropy_cont': [], 'adv_targ': [], 'ratio': []}

        for _ in range(self.ppo_epoch):
            for batch in self.buffer.generate_batches():
                state = check(states[batch]).to(**ACTOR_PARA.tpdv)
                action_batch = check(actions[batch]).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_probs[batch]).to(**ACTOR_PARA.tpdv).view(-1)
                advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv)
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # Mini-batch advantage normalization
                advantage = advantage.view(-1, 1)

                # --- 1. 从 Buffer 中分离出连续和离散动作 ---
                u_from_buffer = action_batch[:, :CONTINUOUS_DIM]
                # 离散动作的索引从第 CONTINUOUS_DIM 列开始
                discrete_actions_from_buffer = {
                    'trigger': action_batch[:, CONTINUOUS_DIM],
                    'salvo_size': action_batch[:, CONTINUOUS_DIM + 1].long(),
                    # 'intra_interval': action_batch[:, CONTINUOUS_DIM + 2].long(),
                    'num_groups': action_batch[:, CONTINUOUS_DIM + 2].long(),# 索引从 +3 改为 +2
                    'inter_interval': action_batch[:, CONTINUOUS_DIM + 3].long(), # 索引从 +4 改为 +3
                }

                # ######################### Actor 训练 #########################
                # # 1. 将前向传播包裹在 autocast 中
                # with autocast():
                #     # 2. 使用当前策略重新评估旧动作的概率
                #     new_dists = self.Actor(state)
                #
                #     # 3. 重新计算新策略下，旧动作的组合 log_prob
                #     new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)
                #     new_log_prob_disc = sum(
                #         new_dists[key].log_prob(discrete_actions_from_buffer[key])
                #         for key in discrete_actions_from_buffer
                #     )
                #     new_prob = new_log_prob_cont + new_log_prob_disc
                #
                #     # 4. 计算组合策略熵
                #     entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                #     entropy_disc = sum(
                #         dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                #     )
                #     total_entropy = (entropy_cont + entropy_disc).mean()
                #
                #     # 5. 计算重要性采样比率和PPO clipped loss (后续逻辑不变)
                #     log_ratio = new_prob - old_prob
                #     ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                #     surr1 = ratio * advantage.squeeze(-1)
                #     surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                #     actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy
                #
                # # 6. Actor梯度更新, 使用 scaler
                # self.Actor.optim.zero_grad(set_to_none=True) # 使用 set_to_none=True 略微提升性能
                # # 用 scaler.scale 来缩放 loss
                # self.actor_scaler.scale(actor_loss).backward()
                # # --- 正确的梯度裁剪流程 ---
                # # 1. Unscale 梯度
                # self.actor_scaler.unscale_(self.Actor.optim)
                # # 2. 在 unscaled 的梯度上进行裁剪
                # torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                # # 3. 执行优化器步骤和 scaler 更新
                # self.actor_scaler.step(self.Actor.optim)
                # self.actor_scaler.update()
                #
                #
                # ######################### Critic 训练 #########################
                # # 3. 将 Critic 的前向传播也包裹在 autocast 中
                # with autocast():
                #     # 7. 计算价值目标并更新Critic (逻辑不变)
                #     old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                #     return_ = advantage + old_value_from_buffer
                #     new_value = self.Critic(state)
                #     critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                # # 4. Critic 梯度更新，使用另一个 scaler
                # self.Critic.optim.zero_grad(set_to_none=True)
                # self.critic_scaler.scale(critic_loss).backward()
                # # --- 同样地，对 Critic 也执行正确的裁剪流程 ---
                # # 1. Unscale 梯度
                # self.critic_scaler.unscale_(self.Critic.optim)
                # # 2. 在 unscaled 的梯度上进行裁剪
                # torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                # # 3. 执行优化器步骤和 scaler 更新
                # self.critic_scaler.step(self.Critic.optim)
                # self.critic_scaler.update()

                ######################### Actor 训练 #########################
                # 2. 使用当前策略重新评估旧动作的概率
                new_dists = self.Actor(state)

                # 3. 重新计算新策略下，旧动作的组合 log_prob
                new_log_prob_cont = new_dists['continuous'].log_prob(u_from_buffer).sum(dim=-1)

                # # --- 💥 重新计算新策略下，旧动作的组合 log_prob (带雅可比修正) ---
                # # 连续部分
                # new_log_prob_u = new_dists['continuous'].log_prob(u_from_buffer)
                # new_log_prob_correction = 2 * (np.log(2.0) - u_from_buffer - torch.nn.functional.softplus(-2 * u_from_buffer))
                # new_log_prob_cont = (new_log_prob_u - new_log_prob_correction).sum(dim=-1)

                new_log_prob_disc = sum(
                    new_dists[key].log_prob(discrete_actions_from_buffer[key])
                    for key in discrete_actions_from_buffer
                )
                new_prob = new_log_prob_cont + new_log_prob_disc

                # 4. 计算组合策略熵
                entropy_cont = new_dists['continuous'].entropy().sum(dim=-1)
                entropy_disc = sum(
                    dist.entropy() for key, dist in new_dists.items() if key != 'continuous'
                )
                total_entropy = (entropy_cont + entropy_disc).mean()
                # --- 核心修改 ---
                # 定义不同的熵系数
                CONTINUOUS_ENTROPY_COEFF = AGENTPARA.entropy  # e.g., 0.01
                DISCRETE_ENTROPY_COEFF = AGENTPARA.entropy #0.05  # 给离散动作一个高 5 倍的熵系数来鼓励探索

                # 计算加权的熵奖励
                total_entropy_bonus = (CONTINUOUS_ENTROPY_COEFF * entropy_cont.mean() +
                                       DISCRETE_ENTROPY_COEFF * entropy_disc.mean())

                # 5. 计算重要性采样比率和PPO clipped loss (后续逻辑不变)
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - total_entropy_bonus #AGENTPARA.entropy * total_entropy

                # 6. Actor梯度更新, 使用 scaler
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                self.Actor.optim.step()

                ######################### Critic 训练 #########################
                # 7. 计算价值目标并更新Critic (逻辑不变)
                old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                return_ = advantage + old_value_from_buffer
                new_value = self.Critic(state)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # # 1️⃣ 计算旧 value（来自 buffer）
                # old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                #
                # # 2️⃣ 目标回报 (return_t = advantage + V_old)
                # return_ = advantage + old_value_from_buffer
                #
                # # 3️⃣ 重新评估新 value
                # new_value = self.Critic(state)
                #
                # # 4️⃣ --- Value Clipping ---
                # # # 限制新预测与旧值的偏差不超过 epsilon
                # # value_clipped = old_value_from_buffer + torch.clamp(
                # #     new_value - old_value_from_buffer,
                # #     -AGENTPARA.epsilon,
                # #     AGENTPARA.epsilon
                # #     # - AGENTPARA.epsilon * old_value_from_buffer.abs(),  # 相对比例 clip,
                # #     # AGENTPARA.epsilon * old_value_from_buffer.abs()  # 相对比例 clip
                # # )
                #
                # #保持原 reward 量级，但改成“相对比例裁剪”解释：
                # # 当 V_old=50 时，允许变化 ±10；
                # # 当 V_old=2 时，允许变化 ±0.4；
                # # 当 V_old≈0 时，会太小，可以加一个下限：
                # scale = torch.clamp(torch.abs(old_value_from_buffer), min=1.0)  # 防止太小
                # value_clipped = old_value_from_buffer + torch.clamp(
                #     new_value - old_value_from_buffer,
                #     -AGENTPARA.epsilon * scale,
                #     +AGENTPARA.epsilon * scale
                # )
                #
                # # 5️⃣ --- 计算两种误差 ---
                # # 普通 MSE loss
                # value_losses = (new_value - return_) ** 2
                # # 裁剪后的 loss（使用裁剪值）
                # value_losses_clipped = (value_clipped - return_) ** 2
                #
                # # 6️⃣ --- 取两者的最大值（防止过度更新）---
                # critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                #
                # # 7️⃣ --- 优化器更新 ---
                # self.Critic.optim.zero_grad()
                # critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                # self.Critic.optim.step()

                # 8. 记录训练信息
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['entropy_cont'].append(entropy_cont.mean().item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())

        self.buffer.clear_memory()
        for key in train_info: train_info[key] = np.mean(train_info[key])
        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        self.save()
        return train_info

    def prep_training_rl(self):
        """将网络设置为训练模式"""
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        """将网络设置为评估模式"""
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
    #     """将模型保存到以训练开始时间命名的专属文件夹中。"""
    #     try:
    #         os.makedirs(self.run_save_dir, exist_ok=True)
    #         print(f"模型将被保存至: {self.run_save_dir}")
    #     except Exception as e:
    #         print(f"创建模型文件夹 {self.run_save_dir} 失败: {e}")
    #         return
    #
    #     for net in ['Actor', 'Critic']:
    #         try:
    #             filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"
    #             full_path = os.path.join(self.run_save_dir, filename)
    #             torch.save(getattr(self, net).state_dict(), full_path)
    #             print(f"  - {filename} 保存成功。")
    #         except Exception as e:
    #             print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")