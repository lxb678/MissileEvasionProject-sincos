import torch
from torch.nn import *
# <<< 混合动作空间 >>> 导入伯努利分布用于离散动作
from torch.distributions import Bernoulli, Categorical, TransformedDistribution
from torch.distributions import Normal, TanhTransform
from PPO_model.Config import *
from PPO_model.Buffer import *
from torch.optim import lr_scheduler
import numpy as np
import os
import re

# <<< 混合动作空间 >>> 假设这些常量在 Config.py 中定义
# from PPO_model.Config import CONTINUOUS_DIM, DISCRETE_DIM, CONTINUOUS_ACTION_KEYS

# 如果 Config 文件中没有定义，也可以在这里定义
CONTINUOUS_DIM = 3
DISCRETE_DIM = 1
CONTINUOUS_ACTION_KEYS = ['nx', 'nz', 'p_cmd']

# --- 动作范围定义 ---
# 根据你的环境来定义每个动作的最小值和最大值
ACTION_RANGES = {
    'nx':       {'low': -1.0, 'high': 2.0},
    'nz':       {'low': -5.0, 'high': 9.0},
    # 'phi_cmd':  {'low': -np.pi, 'high': np.pi},   # 假设滚转指令是-180到+180度
    # 假设滚转角速度范围是 -240度/秒 到 +240度/秒  这个值需要根据你的飞机模型进行调整
    'p_cmd':    {'low': -4.0 * np.pi / 3.0, 'high': 4.0 * np.pi / 3.0},
    'flare':    {'low': 0.0,  'high': 1.0}        # 输出一个[0,1]的倾向值
}


# SquashedNormal 类保持不变
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        transforms = [TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        return torch.tanh(self.loc)


class Actor(Module):
    '''
    <<< 混合动作空间 >>>
    用于混合动作空间的 Actor 网络。
    1. 输出用于连续和离散两种分布的参数。
    2. forward 前向传播方法返回两个独立的分布对象。
    '''

    def __init__(self):
        super(Actor, self).__init__()
        self.input_dim = ACTOR_PARA.input_dim
        # <<< 混合动作空间 >>> output_dim 是所有动作分布所需的参数总数
        # (连续动作的 mu, log_std) + (离散动作的 logits)
        self.output_dim = (CONTINUOUS_DIM * 2) + DISCRETE_DIM
        self.init_model()

        self.log_std_min = -3.0
        self.log_std_max = 0.5

        self.optim = torch.optim.Adam(self.parameters(), ACTOR_PARA.lr)
        self.actor_scheduler = lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=AGENTPARA.mini_lr / ACTOR_PARA.lr,
            total_iters=AGENTPARA.MAX_EXE_NUM
        )
        self.to(ACTOR_PARA.device)

    def init_model(self):
        # 网络主体结构可以保持不变
        self.network = Sequential()
        # 简化网络构建过程
        layers_dims = [self.input_dim] + ACTOR_PARA.model_layer_dim + [self.output_dim]
        for i in range(len(layers_dims) - 1):
            self.network.add_module(f'fc_{i}', Linear(layers_dims[i], layers_dims[i + 1]))
            # 最后一层不需要激活函数
            if i < len(layers_dims) - 2:
                self.network.add_module(f'LeakyReLU_{i}', LeakyReLU())

        # <<< 掩码修改 >>> forward 方法现在需要接收原始的 obs 来进行掩码

    def forward(self, obs):
        # 注意：这里的 obs 是已经转换为 tensor 的
        obs_tensor = check(obs).to(**ACTOR_PARA.tpdv)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # 1. 正常通过网络获取所有输出
        output = self.network(obs_tensor)
        cont_params = output[:, :CONTINUOUS_DIM * 2]
        disc_logits = output[:, CONTINUOUS_DIM * 2:]  # shape: (B, 1)

        # 2. <<< 掩码修改 >>> 应用 logits 掩码
        # 假设 obs_tensor 的第7个特征 (索引为7) 是红外诱饵弹数量
        # has_flares_info 的形状是 (B)
        has_flares_info = obs_tensor[:, 7]
        # 创建一个掩码，当没有诱饵弹时为 True
        mask = (has_flares_info == 0)

        if torch.any(mask):
            # 将对应位置的 logit 设置为一个非常小的数
            # 我们使用 .clone() 来避免原地修改可能引发的梯度问题
            disc_logits_masked = disc_logits.clone()
            # disc_logits_masked[mask] = -1e9  # 一个足够小的数，等效于负无穷
            disc_logits_masked[mask] = torch.finfo(torch.float32).min
        else:
            disc_logits_masked = disc_logits

        # 3. 创建分布 (使用掩码后的 logits)
        # 创建连续分布 (不变)
        mu, log_std = cont_params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        continuous_dist = SquashedNormal(mu, std)

        # 创建离散分布 (使用 squeeze 后的 masked_logits)
        discrete_dist = Bernoulli(logits=disc_logits_masked.squeeze(-1))

        return continuous_dist, discrete_dist


# Critic 类的定义完全保持不变
class Critic(Module):
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
        obs = check(obs).to(**CRITIC_PARA.tpdv)
        value = self.network(obs)
        return value


class PPO_continuous(object):
    def __init__(self, load_able):
        super(PPO_continuous, self).__init__()
        self.Actor = Actor()
        self.Critic = Critic()
        self.buffer = Buffer()
        self.gamma = AGENTPARA.gamma
        self.gae_lambda = AGENTPARA.lamda
        self.ppo_epoch = AGENTPARA.ppo_epoch
        self.total_steps = 0
        # if load_able:
        #     for net in ['Actor', 'Critic']:
        #         try:
        #             getattr(self, net).load_state_dict(
        #                 torch.load("save/" + net + '.pkl', weights_only=True))
        #         except:
        #             print("load error")
        #     pass

        if load_able:
            self.load_model_from_save_folder()

    # store_experience, get_value, cal_gae, load_model 等方法保持不变
    def load_model_from_save_folder(self):
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

        # --- 【在这里添加断言检查】 ---
        # 这里的 probs 就是 log_prob
        # 使用 np.isneginf 专门检查负无穷
        # or np.isinf 检查正负无穷
        # 也可以使用一个更通用的检查，检查 NaN 和 Inf
        if not np.all(np.isfinite(probs)):
            print("=" * 50)
            print("!!! CRITICAL ERROR: Non-finite value (NaN or Inf) detected in log_prob.")
            print(f"Log_prob value: {probs}")
            print(f"State that caused this: {state}")
            print(f"Action that caused this: {action}")
            print("=" * 50)
            raise ValueError("NaN/Inf in log_prob detected!")
        # --- ---------------------- ---

        self.buffer.store_transition(state, value, action, probs, reward, done)

    def scale_action(self, action_cont_tanh):
        """
        <<< 混合动作空间 >>>
        只对动作的连续部分进行缩放，从 [-1, 1] 映射到环境的实际范围。
        """
        lows = torch.tensor([ACTION_RANGES[k]['low'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        highs = torch.tensor([ACTION_RANGES[k]['high'] for k in CONTINUOUS_ACTION_KEYS], **ACTOR_PARA.tpdv)
        scaled_action = lows + (action_cont_tanh + 1.0) * 0.5 * (highs - lows)
        # eps = 1e-6
        # scaled_action = lows + (action_cont_tanh + 1 - eps) * 0.5 * (highs - lows)

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
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=ACTOR_PARA.device)
        is_batch = state_tensor.dim() > 1
        if not is_batch:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # <<< 掩码修改 >>> Actor 现在需要原始 state_tensor 来进行内部掩码
            continuous_dist, discrete_dist = self.Actor(state_tensor)

            # 采样连续动作
            action_cont_tanh = continuous_dist.mean if deterministic else continuous_dist.rsample()

            # 采样离散动作
            if deterministic:
                # action_disc = torch.round(discrete_dist.probs).unsqueeze(-1)
                # probs > 0.5 结果是 True/False，转成 float 就是 1.0/0.0
                # action_disc = (discrete_dist.probs > 0.5).float().unsqueeze(-1)
                action_disc = discrete_dist.sample().unsqueeze(-1)
            else:
                action_disc = discrete_dist.sample().unsqueeze(-1)

            # 计算对数概率 (逻辑也不需要手动掩码了)
            # 连续动作 log_prob
            eps = 1e-3# 比 1e-6 更稳妥
            safe_action_cont = torch.clamp(action_cont_tanh, -1 + eps, 1 - eps)
            log_prob_cont = continuous_dist.log_prob(safe_action_cont).sum(dim=-1)

            # log_prob_cont = continuous_dist.log_prob(action_cont_tanh).sum(dim=-1)
            log_prob_disc = discrete_dist.log_prob(action_disc.squeeze(-1))
            # print("log_prob_cont:",log_prob_cont)
            # print("log_prob_disc:",log_prob_disc)

            total_log_prob = log_prob_cont + log_prob_disc

            self.check_numerics("log_prob_cont", log_prob_cont, state_tensor, action_cont_tanh)
            self.check_numerics("log_prob_disc", log_prob_disc, state_tensor, action_disc)
            self.check_numerics("total_log_prob", total_log_prob, state_tensor)

            # 后续代码不变...
            action_to_store = torch.cat([safe_action_cont, action_disc], dim=-1)
            env_action_cont = self.scale_action(action_cont_tanh)
            final_env_action_tensor = torch.cat([env_action_cont, action_disc], dim=-1)

        # ...后续代码不变...
        action_to_store_np = action_to_store.cpu().numpy()
        log_prob_to_store_np = total_log_prob.cpu().numpy()
        final_env_action_np = final_env_action_tensor.cpu().numpy()

        if not is_batch:
            final_env_action_np = final_env_action_np[0]
            action_to_store_np = action_to_store_np[0]
            log_prob_to_store_np = log_prob_to_store_np[0]

        return final_env_action_np, action_to_store_np, log_prob_to_store_np

    def get_value(self, state):
        value = self.Critic(state)
        return value

    def cal_gae(self, states, values, actions, probs, rewards, dones):
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            # 如果是最后一步，则 next_value 为 0
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            # 结束标志 done 的掩码
            done_mask = 1.0 - int(dones[t])
            delta = rewards[t] + self.gamma * next_value * done_mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantage[t] = gae

        # 优势标准化
        # advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        return advantage

    def learn(self):
        torch.autograd.set_detect_anomaly(True)

        states, values, actions, old_probs, rewards, dones = self.buffer.sample()
        advantages = self.cal_gae(states, values, actions, old_probs, rewards, dones)

        train_info = {
            'critic_loss': [], 'actor_loss': [], 'dist_entropy': [],
            'adv_targ': [], 'ratio': []
        }

        for _ in range(self.ppo_epoch):
            for batch in self.buffer.generate_batches():
                # ... (数据准备不变) ...
                state = check(states[batch]).to(**ACTOR_PARA.tpdv)
                action_batch = check(actions[batch]).to(**ACTOR_PARA.tpdv)
                old_prob = check(old_probs[batch]).to(**ACTOR_PARA.tpdv)
                advantage = check(advantages[batch]).to(**ACTOR_PARA.tpdv).unsqueeze(-1)

                action_cont_tanh = action_batch[:, :CONTINUOUS_DIM]
                action_disc = action_batch[:, CONTINUOUS_DIM:]  # shape: (B, 1)

                ######################### Actor 训练 #########################
                new_cont_dist, new_disc_dist = self.Actor(state)

                eps = 1e-3 # 比 1e-6 更稳妥
                action_cont_tanh_clipped = torch.clamp(action_cont_tanh, -1+eps, 1-eps)
                new_log_prob_cont = new_cont_dist.log_prob(action_cont_tanh_clipped).sum(dim=-1)
                #对 log_prob 做上下界裁剪（防止极端值）
                new_log_prob_cont = torch.clamp(new_log_prob_cont, min=-1e3, max=1e3)
                # 相同地处理 discrete 部分与 new_prob 总和

                # ---【修正】---
                # new_disc_dist 的 log_prob 需要一个 (B) 形状的输入
                new_log_prob_disc = new_disc_dist.log_prob(action_disc.squeeze(-1))
                new_log_prob_disc = torch.clamp(new_log_prob_disc, min=-1e3, max=1e3)
                # 输出直接是 (B)

                # 两个 (B) 张量相加
                new_prob = new_log_prob_cont + new_log_prob_disc
                # new_prob = torch.clamp(new_prob, min=-1e3, max=1e3)
                # ---【修正】---
                # new_disc_dist.entropy() 现在直接返回 (B)，不再需要 squeeze
                entropy_cont = new_cont_dist.base_dist.entropy().sum(dim=-1)
                entropy_disc = new_disc_dist.entropy()
                total_entropy = (entropy_cont + entropy_disc).mean()

                # 现在 old_prob(B) 和 new_prob(B) 形状保证一致
                log_ratio = new_prob - old_prob
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                self.check_numerics("new_prob", new_prob, state, action_batch)
                self.check_numerics("ratio", ratio, state, action_batch)

                surr1 = ratio * advantage.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - AGENTPARA.epsilon, 1.0 + AGENTPARA.epsilon) * advantage.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - AGENTPARA.entropy * total_entropy

                # ... (后续代码基本不变，但为了保险，我重新检查了 Critic 部分)
                self.Actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=0.5)
                self.Actor.optim.step()

                ######################### Critic 训练 #########################
                # 检查一下 old_value 的形状
                old_value_from_buffer = check(values[batch]).to(**CRITIC_PARA.tpdv)  # Critic输出(N,1),所以这里是(B,1)
                return_ = advantage + old_value_from_buffer  # (B, 1) + (B, 1) -> OK
                new_value = self.Critic(state)  # (B, 1)
                critic_loss = torch.nn.functional.mse_loss(new_value, return_)  # OK

                self.Critic.optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)
                self.Critic.optim.step()

                # 记录训练信息... (保持不变)
                train_info['critic_loss'].append(critic_loss.item())
                train_info['actor_loss'].append(actor_loss.item())
                train_info['dist_entropy'].append(total_entropy.item())
                train_info['adv_targ'].append(advantage.mean().item())
                train_info['ratio'].append(ratio.mean().item())
                # train_info['torch.min(surr1, surr2).mean()'] = torch.min(surr1, surr2).mean().item()

        # ... 后续代码保持不变 ...
        self.buffer.clear_memory()

        for key in train_info:
            train_info[key] = np.mean(train_info[key])

        train_info['actor_lr'] = self.Actor.optim.param_groups[0]['lr']
        train_info['critic_lr'] = self.Critic.optim.param_groups[0]['lr']
        self.save1()
        return train_info

    # 其他方法如 prep_training_rl, prep_eval_rl, save 保持不变
    def prep_training_rl(self):
        self.Actor.train()
        self.Critic.train()

    def prep_eval_rl(self):
        self.Actor.eval()
        self.Critic.eval()

    def save1(self):
        for net in ['Actor', 'Critic']:
            try:
                torch.save(getattr(self, net).state_dict(), "save/" + net + ".pkl")
            except Exception as e:
                print(f"模型保存失败: {e}")
    def save(self, prefix=""):
        for net in ['Actor', 'Critic']:
            try:
                filename = f"save/{prefix}_{net}.pkl" if prefix else f"save/{net}.pkl"
                torch.save(getattr(self, net).state_dict(), filename)
            except:
                print("write_error")