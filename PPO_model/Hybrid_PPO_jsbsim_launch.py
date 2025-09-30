import torch
from torch.nn import *
# <<< 混合动作空间 >>> 导入伯努利分布用于离散动作
from torch.distributions import Bernoulli, Categorical, TransformedDistribution
from torch.distributions import Normal, TanhTransform
from Interference_code.PPO_model.Config_launch import *
from Interference_code.PPO_model.Buffer_launch import *
from torch.optim import lr_scheduler
import numpy as np
import re
import os  # <<< 1. 导入 os 库
import time # <<< 2. 导入 time 库
##PPO算法：解决梯度爆炸   动作采用tanh缩放的话，经验池存储原始动作u，可以不用雅可比修正动作对数概率log. （用了雅可比修正也不会爆炸，暂时不清楚为什么），如果用反tanh算原始动作u就会有误差，就会梯度爆炸
## 如果动作为加速度，观测状态中有速度，可以对速度进行裁剪，不会爆炸；如果观测状态没有速度，对环境的速度进行裁剪会爆炸，因为同一观测状态导致不同结果
# 优势标准化不用了、Minibatch可以留

# --- 动作空间配置 ---
# 定义连续和离散动作的维度
CONTINUOUS_DIM = 4  # 更改为 4: 油门, 升降舵, 副翼, 方向舵
DISCRETE_DIM = 1    # 离散动作的数量 (fire_missile)
# 定义连续动作的键名，用于后续的动作缩放
# <<< 更改 >>> 定义连续动作的键名
CONTINUOUS_ACTION_KEYS = ['throttle', 'elevator', 'aileron', 'rudder']

# --- 动作范围定义 (已更新) ---
# <<< 更改 >>> 定义每个动作的物理范围
# JSBSim 的 fcs/*-cmd-norm 属性通常接受归一化的输入
ACTION_RANGES = {
    'throttle': {'low': 0.0, 'high': 1.0},   # 油门指令范围 [0, 1]
    'elevator': {'low': -1.0, 'high': 1.0},  # 升降舵指令范围 [-1, 1]
    'aileron':  {'low': -1.0, 'high': 1.0},  # 副翼指令范围 [-1, 1]
    'rudder':   {'low': -1.0, 'high': 1.0},  # 方向舵指令范围 [-1, 1]
    'missile':    {'low': 0.0,  'high': 1.0}        # 离散动作的逻辑范围 (这里用不上，但保持完整性)
}

class Actor(Module):
    """
   Actor 网络 (策略网络)

   该网络负责根据当前状态(observation)生成动作。
   它特别为【混合动作空间】设计：
   1.  输出用于构建连续动作（正态分布）和离散动作（伯努利分布）的参数。
   2.  在前向传播中，它会返回两个独立的分布对象，而不是直接返回动作。
   3.  实现了【动作掩码】，当特定条件不满足时（如没有红外诱饵弹），会阻止选择对应的离散动作。
   """
    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # print(f"Actor 网络输入维度: {self.input_dim}")
        # <<< 混合动作空间 >>> output_dim 是所有动作分布所需的参数总数
        # 输出维度 = (连续动作数量 * 2 (均值mu, 标准差log_std)) + (离散动作数量 (logits))
        self.output_dim = (CONTINUOUS_DIM * 2) + DISCRETE_DIM
        # 定义标准差的对数值 log_std 的范围，防止其过大或过小导致数值不稳定
        self.log_std_min = -20.0
        self.log_std_max =  2.0 #2.0
        # 初始化网络模型、优化器和学习率调度器
        self.init_model()
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
            # 最后一层是输出层，不需要激活函数
            if i < len(layers_dims) - 2:
                self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())
    def forward(self, obs):
        """
       前向传播方法

       Args:
           obs (torch.Tensor): 输入的状态观测值

       Returns:
           tuple: (continuous_base_dist, discrete_dist)
                  - continuous_base_dist: 用于连续动作的【基础】正态分布对象 (pre-tanh)
                  - discrete_dist: 用于离散动作的伯努利分布对象
       """
        # 注意：这里的 obs 是已经转换为 tensor 的
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        # 如果输入是单个样本，增加一个批次维度
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 获取神经网络的原始输出
        output = self.network(obs_tensor)
        # 2. 将输出切分为连续动作参数和离散动作参数
        cont_params = output[:, :CONTINUOUS_DIM * 2]
        disc_logits = output[:, CONTINUOUS_DIM * 2:]  # shape: (B, 1)

        # 3. 【动作掩码】逻辑
        # 假设 obs_tensor 的第 7 个特征 (索引为7) 代表红外诱饵弹数量
        # has_flares_info 的形状是 (B)
        has_flares_info = obs_tensor[:, 4]
        # 创建一个掩码，当没有诱饵弹时为 True
        mask = (has_flares_info == 0)

        disc_logits_masked = disc_logits.clone()  # 使用 .clone() 避免原地修改  # 我们使用 .clone() 来避免原地修改可能引发的梯度问题
        if torch.any(mask):
            # 将没有诱饵弹的样本对应的 logit 设置为一个极小值（等效于负无穷），
            # 这样通过 sigmoid 后概率会趋近于 0，从而阻止智能体选择该动作。
            # disc_logits_masked[mask] = -1e9  # 一个足够小的数，等效于负无穷
            disc_logits_masked[mask] = torch.finfo(torch.float32).min

        # 4. 创建动作分布对象
        # 4.1 创建连续动作的【基础】正态分布
        #     注意：这里创建的是变换前 (pre-tanh) 的分布。实际的动作会通过 tanh 函数进行压缩。
        mu, log_std = cont_params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # 限制 log_std 范围
        std = torch.exp(log_std)
        continuous_base_dist = Normal(mu, std)
        # 4.2 创建离散动作的伯努利分布（使用掩码后的 logits）
        # squeeze(-1) 将 [B, 1] 形状的 logits 变为 [B]，以匹配 Bernoulli 分布的输入要求
        discrete_dist = Bernoulli(logits=disc_logits_masked.squeeze(-1))

        return continuous_base_dist, discrete_dist

# Critic 类的定义完全保持不变
class Critic(Module):
    """
   Critic 网络 (价值网络)

   该网络负责评估当前状态(observation)的价值 V(s)，即从当前状态开始，
   遵循当前策略所能获得的期望回报。它的输出是一个标量。
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
        """初始化神经网络结构"""
        self.network = Sequential()
        for i in range(len(CRITIC_PARA.model_layer_dim) + 1):
            if i == 0:
                self.network.add_module('fc_{}'.format(i), Linear(self.input_dim, CRITIC_PARA.model_layer_dim[0]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())
            elif i < len(CRITIC_PARA.model_layer_dim):
                self.network.add_module('fc_{}'.format(i),
                                        Linear(CRITIC_PARA.model_layer_dim[i - 1], CRITIC_PARA.model_layer_dim[i]))
                self.network.add_module('LeakyReLU_{}'.format(i), LeakyReLU())
            else:
                self.network.add_module('fc_{}'.format(i), Linear(CRITIC_PARA.model_layer_dim[-1], self.output_dim))

    def forward(self, obs):
        """前向传播，计算状态价值"""
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


class PPO_continuous(object):
    """
   PPO 智能体主类

   该类整合了 Actor 和 Critic 网络，并实现了 PPO 算法的核心逻辑，
   包括动作选择、经验存储、优势计算和网络更新。
   """
    """
   初始化PPO智能体。

   Args:
       load_able (bool): 是否需要加载预训练模型。
       model_dir_path (str, optional): 【新】包含模型文件的【文件夹路径】。
                                       如果提供，将从此文件夹加载。
   """

    def __init__(self, load_able: bool, model_dir_path: str = None):
        super(PPO_continuous, self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()
        self.buffer = Buffer()
        # PPO 算法超参数
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0

        # <<< 3. 在初始化时，为本次训练运行生成一个唯一的时间戳 >>>
        # 格式为 "年-月-日_时-分-秒"，例如 "2023-10-27_15-30-45"
        self.training_start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # 定义基础保存目录
        self.base_save_dir = "save_launch"

        # 构建本次训练专用的文件夹路径
        self.run_save_dir = os.path.join(self.base_save_dir, self.training_start_time)

        # # 加载预训练模型
        # if load_able:
        #     # 推荐使用新的加载逻辑，如果旧逻辑仍需保留，可以取消注释
        #     self.load_model_from_save_folder()
        #     # for net in ['Actor', 'Critic']:
        #     #     try:
        #     #         path = f"save/{net}.pkl"
        #     #         getattr(self, net).load_state_dict(torch.load(path, weights_only=True))
        #     #         print(f"成功加载模型: {path}")
        #     #     except Exception as e:
        #     #         print(f"加载模型 {net}.pkl 失败: {e}")

        # --- <<< 核心修改 2: 更新模型加载逻辑 >>> ---
        if load_able:
            if model_dir_path:
                print(f"--- 正在从指定文件夹加载模型: {model_dir_path} ---")
                # 调用我们新的、更强大的加载函数
                self.load_models_from_directory(model_dir_path)
            else:
                # 保留旧的回退逻辑，如果需要的话
                print("--- 未指定模型文件夹，尝试从默认文件夹 'test' 加载 ---")
                self.load_models_from_directory("../test")

        # --- <<< 核心修改 3: 实现新的、通用的加载函数 >>> ---

    def load_models_from_directory(self, directory_path: str):
        """
        从指定的文件夹路径加载模型，能自动识别多种命名格式。
        - 格式1 (带前缀): "prefix_Actor.pkl", "prefix_Critic.pkl"
        - 格式2 (无前缀): "Actor.pkl", "Critic.pkl"
        """
        if not os.path.isdir(directory_path):
            print(f"[错误] 模型加载失败：提供的路径 '{directory_path}' 不是一个有效的文件夹。")
            return

        files = os.listdir(directory_path)

        # 优先级 1: 查找带前缀的 Actor 文件 (e.g., "best_Actor.pkl")
        actor_files_with_prefix = [f for f in files if f.endswith("_Actor.pkl")]
        if len(actor_files_with_prefix) > 0:
            # 如果有多个，优先选择第一个找到的
            actor_filename = actor_files_with_prefix[0]
            # 从文件名中提取前缀
            prefix = actor_filename.replace("_Actor.pkl", "")
            critic_filename = f"{prefix}_Critic.pkl"
            print(f"  - 检测到前缀 '{prefix}'，准备加载模型...")

            # 检查对应的 Critic 文件是否存在
            if critic_filename in files:
                actor_full_path = os.path.join(directory_path, actor_filename)
                critic_full_path = os.path.join(directory_path, critic_filename)

                try:
                    self.Actor.load_state_dict(torch.load(actor_full_path, map_location=ACTOR_PARA.device))
                    print(f"    - 成功加载 Actor: {actor_full_path}")
                    self.Critic.load_state_dict(torch.load(critic_full_path, map_location=CRITIC_PARA.device))
                    print(f"    - 成功加载 Critic: {critic_full_path}")
                    return  # 成功加载，结束函数
                except Exception as e:
                    print(f"    - [错误] 加载带前缀的模型时失败: {e}")
            else:
                print(f"    - [警告] 找到了 '{actor_filename}' 但未找到对应的 '{critic_filename}'。")

        # 优先级 2: 如果没找到带前缀的，就查找无前缀的 "Actor.pkl"
        if "Actor.pkl" in files and "Critic.pkl" in files:
            print("  - 检测到无前缀格式，准备加载 'Actor.pkl' 和 'Critic.pkl'...")
            actor_full_path = os.path.join(directory_path, "Actor.pkl")
            critic_full_path = os.path.join(directory_path, "Critic.pkl")
            try:
                self.Actor.load_state_dict(torch.load(actor_full_path, map_location=ACTOR_PARA.device))
                print(f"    - 成功加载 Actor: {actor_full_path}")
                self.Critic.load_state_dict(torch.load(critic_full_path, map_location=CRITIC_PARA.device))
                print(f"    - 成功加载 Critic: {critic_full_path}")
                return  # 成功加载，结束函数
            except Exception as e:
                print(f"    - [错误] 加载无前缀模型时失败: {e}")

        # 如果以上两种方式都失败
        print(f"[错误] 模型加载失败：在文件夹 '{directory_path}' 中未找到任何有效的 Actor/Critic 模型对。")

    def load_model_from_save_folder(self):
        """
          从指定文件夹加载模型，能自动识别新旧两种命名格式。
          - 新格式: "prefix_Actor.pkl", "prefix_Critic.pkl"
          - 旧格式: "Actor.pkl", "Critic.pkl"
        """
        save_dir = "test"
        files = os.listdir(save_dir)

        # 找唯一的 *_Actor.pkl 文件
        # ------- 情况 1：查找 *_Actor.pkl 文件 -------
        actor_files = [f for f in files if f.endswith("_Actor.pkl")]
        if len(actor_files) == 1:
            match = re.match(r"(.+)_Actor\.pkl", actor_files[0])
            if not match:
                print("文件名格式错误")
                return
            prefix = match.group(1)
            for net in ['Actor', 'Critic']:
                try:
                    filename = os.path.join(save_dir, f"{prefix}_{net}.pkl")
                    getattr(self, net).load_state_dict(torch.load(filename, weights_only=True))
                    print(f"成功加载模型: {filename}")
                except Exception as e:
                    print(f"加载失败: {filename}，原因: {e}")
            return  # 成功加载后退出

        # ------- 情况 2：查找老格式 Actor.pkl 和 Critic.pkl -------
        elif "Actor.pkl" in files and "Critic.pkl" in files:
            for net in ['Actor', 'Critic']:
                try:
                    filename = os.path.join(save_dir, f"{net}.pkl")
                    getattr(self, net).load_state_dict(torch.load(filename, weights_only=True))
                    print(f"成功加载旧格式模型: {filename}")
                except Exception as e:
                    print(f"加载失败: {filename}，原因: {e}")
            return  # 成功加载后退出

        # ------- 都不符合，报错 -------
        else:
            print("模型加载错误：未找到符合要求的模型文件，请确保 save/ 中存在一对 Actor/Critic 模型")
            return

    def store_experience(self, state, action, probs, value, reward, done):
        """
        存储经验到 Buffer，并在存储前进行数值检查。
        """
        # 检查 log_prob 是否为 NaN 或无穷大，这是训练不稳定的常见信号
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
        """
        <<< 混合动作空间 >>>
        只对动作的连续部分进行缩放，从 [-1, 1] 映射到环境的实际范围。
        """
        # 从配置中获取每个连续动作的最小/最大值
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        # 线性插值公式: low + (value_in_0_1) * (high - low)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)

        return scaled_action

    def check_numerics(self, name, tensor, state=None, action=None, threshold=1e4):
        """检查 tensor 是否存在 NaN、Inf 或异常大值"""
        arr = tensor.detach().cpu().numpy()

        if not np.all(np.isfinite(arr)):
            print("=" * 50)
            print(f"[数值错误] {name} 出现 NaN/Inf")
            print(f"值: {arr}")
            if state is not None:
                print(f"对应的 state: {state}")
            if action is not None:
                print(f"对应的 action: {action}")
            print("=" * 50)
            raise ValueError(f"NaN/Inf detected in {name}")

        if np.any(np.abs(arr) > threshold):
            print("=" * 50)
            print(f"[警告] {name} 数值过大 (> {threshold})")
            print(f"最大值: {arr.max()}, 最小值: {arr.min()}")
            if state is not None:
                print(f"对应的 state: {state}")
            if action is not None:
                print(f"对应的 action: {action}")
            print("=" * 50)
            # 这里不一定要 raise，可以选择直接返回，让训练继续

    def choose_action(self, state, deterministic=False):
        """
        根据当前状态选择动作。这是与环境交互的核心。

        这个函数实现了 Tanh-Normal 分布的完整采样和对数概率计算流程，以处理有界连续动作空间。
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # 1. Actor 现在返回基础分布(Normal, Bernoulli)
            continuous_base_dist, discrete_dist = self.Actor(state_tensor)
            # --- 连续动作处理 ---
            # 2. 采样或获取原始动作 u (pre-tanh)(形状: [B, 3])
            u = continuous_base_dist.mean if deterministic else continuous_base_dist.rsample()
            # 3. 通过 tanh 函数将 u 压缩到 [-1, 1] 范围内，得到 a = tanh(u)  计算最终动作 a = tanh(u) 计算用于环境的动作 a (形状: [B, 3])
            action_cont_tanh = torch.tanh(u)
            # 4. 计算 log_prob，现在它就是基础正态分布的 log_prob
            log_prob_cont = continuous_base_dist.log_prob(u).sum(dim=-1)
            # (雅可比修正项被完全移除)
            # --- 离散动作处理 ---
            # 5. 从伯努利分布中采样离散动作
            if deterministic:
                # probs > 0.5 结果是 True/False，转成 float 就是 1.0/0.0
                # action_disc = (discrete_dist.probs > 0.5).float().unsqueeze(-1)
                action_disc = discrete_dist.sample().unsqueeze(-1)
            else:
                action_disc = discrete_dist.sample().unsqueeze(-1)
            # 6. 计算离散动作的 log_prob (输入为 [B])
            log_prob_disc = discrete_dist.log_prob(action_disc.squeeze(-1))

            # --- 组合与输出 ---
            # 7. 组合总的对数概率
            total_log_prob = log_prob_cont + log_prob_disc
            # print(f"log_prob_cont: {log_prob_cont}, log_prob_disc: {log_prob_disc}, total_log_prob: {total_log_prob}")
            # 数值检查
            self.check_numerics("log_prob_cont", log_prob_cont, state_tensor, action_cont_tanh)
            self.check_numerics("log_prob_disc", log_prob_disc, state_tensor, action_disc)
            self.check_numerics("total_log_prob", total_log_prob, state_tensor)
            # 8. 【核心】准备存储到 Buffer 的动作
            #    我们必须存储原始的 u (pre-tanh)，而不是 tanh(u)。
            #    因为在学习步骤中，我们需要用 u 来重新计算其在新策略下的概率。
            # 之前: action_to_store = torch.cat([action_cont_tanh, ...])
            # 现在: 我们直接存储原始的 u，而不是 tanh 后的 a
            action_to_store = torch.cat([u, action_disc], dim=-1)
            # 9. 准备发送到环境的最终动作
            #    连续部分需要被缩放到环境的实际范围
            env_action_cont = self.scale_action(action_cont_tanh)
            # 创建一个全零的 "flare" 占位符
            dummy_flare_action = torch.zeros_like(action_disc)
            final_env_action_tensor = torch.cat([env_action_cont, dummy_flare_action, action_disc], dim=-1)
        # 将 Tensors 转换为 Numpy 数组
        action_to_store_np = action_to_store.cpu().numpy()
        log_prob_to_store_np = total_log_prob.cpu().numpy()
        final_env_action_np = final_env_action_tensor.cpu().numpy()
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
        """
       计算广义优势估计 (Generalized Advantage Estimation, GAE)。
       GAE 是对优势函数 A(s,a) = Q(s,a) - V(s) 的一种估计，它通过 gamma 和 lambda 参数
       在无偏（高方差）和有偏（低方差）之间进行权衡。
       """
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            # 获取下一个状态的价值，如果是轨迹的最后一步，则 next_value 为 0
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            #  # done_mask 用于在回合结束时切断价值的传播
            done_mask = 1.0 - int(dones[t])
            # 计算 TD-error (delta)
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            # GAE 递推公式: gae_t = delta_t + gamma * lambda * gae_{t+1}
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantage[t] = gae

        # 可选：对优势进行标准化，通常能提升训练稳定性
        # advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        return advantage

    def learn(self):
        """
        执行 PPO 的学习和更新步骤。
        """
        # 开启异常检测，有助于调试梯度问题
        torch.autograd.set_detect_anomaly(True)
        # 1. 从 Buffer 中采样经验并计算优势
        states, values, actions, old_probs, rewards, dones = self.buffer.sample()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)
        # print(f"advantages: {advantages}")
        # 存储训练过程中的统计信息
        train_info = {'critic_loss': [], 'actor_loss': [], 'dist_entropy': [], 'adv_targ': [], 'ratio': []}
        # 2. 多轮 PPO Epoch 迭代更新
        for _ in range(self.ppo_epoch):
            # 3. 从完整数据中生成小批量 (mini-batch)
            for batch in self.buffer.generate_batches():
                # 提取 mini-batch 数据并转换为 Tensor
                state = check(states[batch]).to(**ACTOR_PARA.tpdv)
                action_batch = check(actions[batch]).to(**ACTOR_PARA.tpdv) # 包含 [u_cont, action_disc]
                # --- (核心通用化修改 1) 强制重塑关键张量 ---
                # old_prob: 确保是一维向量 [B]
                old_prob = check(old_probs[batch]).to(**ACTOR_PARA.tpdv).view(-1)
                # advantage: 确保是列向量 [B, 1]
                advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv).view(-1, 1)

                # advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv)  # 这里 advantage 还是 (B,)
                # # 在这里进行 mini-batch 内部的优势标准化
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                # advantage = advantage.unsqueeze(-1)  # 然后再增加维度

                # 从 Buffer 中分离出连续动作的原始值 u 和离散动作
                u_from_buffer = action_batch[:, :CONTINUOUS_DIM]
                action_disc = action_batch[:, CONTINUOUS_DIM:]

                ######################### Actor 训练 #########################
                # 4. 使用当前策略重新评估旧动作的概率
                new_cont_base_dist, new_disc_dist = self.Actor(state)
                # 5. 重新计算新策略下，旧动作 u 的 log_prob
                new_log_prob_cont = new_cont_base_dist.log_prob(u_from_buffer).sum(dim=-1)
                # (雅可比修正项被完全移除)
                new_log_prob_disc = new_disc_dist.log_prob(action_disc.squeeze(-1))
                new_prob = new_log_prob_cont + new_log_prob_disc
                # 6. 计算策略熵，作为探索的正则化项
                entropy_cont = new_cont_base_dist.entropy().sum(dim=-1)
                entropy_disc = new_disc_dist.entropy()  # Bernoulli.entropy() 直接返回 (B,)
                total_entropy = (entropy_cont + entropy_disc).mean()
                # 7. 计算重要性采样比率 ratio = pi_new(a|s) / pi_old(a|s)
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))# 限制防止数值爆炸

                self.check_numerics("new_prob", new_prob, state, action_batch)
                self.check_numerics("ratio", ratio, state, action_batch)

                # 8. 计算 PPO 的 clip 代理目标函数
                # --- (核心通用化修改 2) 确保 ratio 和 advantage 的形状匹配 ---
                # surr1/surr2 的计算需要 ratio 和 advantage 的维度匹配
                # ratio 是 [B], advantage 是 [B, 1]。
                # 我们需要将 ratio 变成 [B, 1] 或者将 advantage 变回 [B]。
                # 推荐将 advantage squeeze掉，因为损失最后是标量。
                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # 9. Actor 梯度更新
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0) # 梯度裁剪
                self.Actor.optim.step()

                ######################### Critic 训练 #########################
                # 10. 计算价值目标 (TD-target / GAE-target)
                # old_value_from_buffer: 确保是列向量 [B, 1]
                old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv).view(-1, 1)
                # advantage [B, 1] + old_value_from_buffer [B, 1] -> return_ [B, 1]
                return_ = advantage + old_value_from_buffer
                # 11. 使用当前 Critic 评估状态价值
                new_value = self.Critic(state)
                # 12. 计算 Critic 的均方误差损失
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)
                # 13. Critic 梯度更新
                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=1.0)
                self.Critic.optim.step()

                # 记录训练信息
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())

        # 清空 Buffer，为下一次数据收集做准备
        self.buffer.clear_memory()
        # 计算本次学习的平均统计信息
        for key in train_info:
            train_info[key] = np.mean(train_info[key])

        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        # 保存模型
        self.save()
        return train_info

    # --- 实用方法 ---
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
        将模型保存到以训练开始时间命名的专属文件夹中。

        Args:
            prefix (str, optional): 文件名的前缀，可用于标记回合数或最佳模型。
                                    例如: save(prefix="episode_5000")
        """
        try:
            # <<< 4. 确保本次训练的专属文件夹存在，如果不存在则创建 >>>
            # os.makedirs(..., exist_ok=True) 是一个安全的操作，
            # 如果文件夹已存在，它不会报错。
            os.makedirs(self.run_save_dir, exist_ok=True)
            print(f"模型将被保存至: {self.run_save_dir}")

        except Exception as e:
            print(f"创建模型文件夹 {self.run_save_dir} 失败: {e}")
            return  # 如果文件夹创建失败，则不继续执行

        # 循环保存 Actor 和 Critic 网络
        for net in ['Actor', 'Critic']:
            try:
                # <<< 5. 构建带有前缀和网络名的完整文件名 >>>
                # 如果 prefix 为空, 文件名为 "Actor.pkl" 或 "Critic.pkl"
                # 如果 prefix 为 "best", 文件名为 "best_Actor.pkl"
                filename = f"{prefix}_{net}.pkl" if prefix else f"{net}.pkl"

                # <<< 6. 构建最终的完整文件路径 >>>
                full_path = os.path.join(self.run_save_dir, filename)

                # 执行保存操作
                torch.save(getattr(self, net).state_dict(), full_path)
                print(f"  - {filename} 保存成功。")

            except Exception as e:
                print(f"  - 保存模型 {net} 到 {full_path} 时发生错误: {e}")