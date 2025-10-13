# 文件: option_critic_agent.py

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import numpy as np
from enum import Enum
import os

# --- 导入您的PPO模型类，仅用于加载权重 ---
from Interference_code.PPO_model.旧文件.Hybrid_PPO_jsbsim_launch import Actor as AttackActor
from Interference_code.PPO_model.旧文件.Hybrid_PPO_jsbsim import Actor as EvadeActor

# --- 定义超参数 ---
STATE_DIM = 14  # 使用主环境的14维观测
MANEUVER_DIM = 4  # 4个连续机动动作
DISCRETE_DIM = 1  # 每个选项有1个离散动作
NUM_OPTIONS = 2  # 两个选项: ATTACK 和 EVADE
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
TERMINATION_COST = 0.01  # 鼓励选项持续更长时间的成本


class Option(Enum):
    ATTACK = 0
    EVADE = 1


class OptionCriticNetwork(nn.Module):
    """
    统一的Option-Critic神经网络。
    接收一个14维状态，输出所有决策所需信息。
    """

    def __init__(self):
        super().__init__()

        # --- 共享主干网络 ---
        self.shared_layers = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # --- 分支输出头 ---
        # 1. 选项价值头 (Q_omega): 输出每个选项的Q值
        self.q_heads = nn.Linear(256, NUM_OPTIONS)

        # 2. 终止概率头 (beta): 输出每个选项的终止概率
        self.beta_heads = nn.Linear(256, NUM_OPTIONS)

        # 3. 选项内部策略头 (pi_omega): 为每个选项创建一个独立的演员头
        # 每个头负责输出 4个连续机动动作 + 1个离散动作 的参数
        self.actor_heads = nn.ModuleList([
            nn.Linear(256, (MANEUVER_DIM * 2) + DISCRETE_DIM) for _ in range(NUM_OPTIONS)
        ])

        # 为连续动作定义可学习的标准差
        self.log_std_min, self.log_std_max = -20.0, 2.0

    def forward(self, state):
        features = self.shared_layers(state)

        q_values = self.q_heads(features)
        termination_probs = torch.sigmoid(self.beta_heads(features))

        # 为每个选项的策略生成分布
        distributions = []
        for i in range(NUM_OPTIONS):
            params = self.actor_heads[i](features)

            # 分离连续和离散动作的参数
            cont_params = params[:, :MANEUVER_DIM * 2]
            disc_logits = params[:, MANEUVER_DIM * 2:]

            # 创建连续动作分布 (pre-tanh)
            mu, log_std = cont_params.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            cont_dist = Normal(mu, std)

            # 创建离散动作分布 (伯努利)
            disc_dist = Bernoulli(logits=disc_logits.squeeze(-1))

            distributions.append((cont_dist, disc_dist))

        return q_values, termination_probs, distributions


class OptionCriticAgent:
    def __init__(self, device='cpu', load_from_experts=False, attack_model_dir=None, evade_model_dir=None):
        self.device = torch.device(device)
        self.network = OptionCriticNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        # 经验回放缓冲区
        self.buffer = []  # 简化版buffer，实际应用中应使用更高效的实现

        if load_from_experts:
            self.initialize_from_experts(attack_model_dir, evade_model_dir)

    def initialize_from_experts(self, attack_model_dir, evade_model_dir):
        """用预训练的专家模型权重初始化选项内部策略头"""
        print("--- 正在从专家模型初始化策略头 ---")
        try:
            # 1. 加载攻击专家模型
            attack_actor_temp = AttackActor()
            attack_actor_path = os.path.join(attack_model_dir, "Actor.pkl")  # 假设是这个文件名
            attack_actor_temp.load_state_dict(torch.load(attack_actor_path, map_location=self.device))

            # 2. 加载规避专家模型
            evade_actor_temp = EvadeActor()
            evade_actor_path = os.path.join(evade_model_dir, "Actor.pkl")  # 假设是这个文件名
            evade_actor_temp.load_state_dict(torch.load(evade_actor_path, map_location=self.device))

            # 3. 复制权重
            # 共享层权重可以用任一专家模型的权重来初始化
            self.network.shared_layers.load_state_dict(attack_actor_temp.network[:4].state_dict())

            # 复制攻击策略头权重 (Option 0)
            self.network.actor_heads[Option.ATTACK.value].load_state_dict(attack_actor_temp.network[4].state_dict())

            # 复制规避策略头权重 (Option 1)
            self.network.actor_heads[Option.EVADE.value].load_state_dict(evade_actor_temp.network[4].state_dict())

            print("--- 专家模型权重初始化成功！ ---")

        except Exception as e:
            print(f"[错误] 初始化专家模型失败: {e}")
            print("将使用随机权重进行训练。")

    def _scale_maneuver_actions(self, maneuver_actions_tanh):
        """将[-1, 1]的机动动作缩放到实际范围"""
        # 油门: [0, 1], 其他: [-1, 1]
        lows = torch.tensor([0.0, -1.0, -1.0, -1.0], device=self.device)
        highs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        return lows + (maneuver_actions_tanh + 1.0) * 0.5 * (highs - lows)

    def get_action(self, state, prev_option, deterministic=False):
        """核心决策函数"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values, term_probs, distributions = self.network(state_tensor)

            # --- 1. 高层决策：决定是否终止并选择新选项 ---
            current_option = prev_option
            if prev_option is None or torch.rand(1, device=self.device) < term_probs[0, prev_option.value]:
                # 如果是初始步或当前选项决定终止，则选择一个新选项
                # 使用 epsilon-greedy 策略选择新选项
                if deterministic or np.random.rand() > 0.05:
                    current_option = Option(torch.argmax(q_values).item())
                else:
                    current_option = Option(np.random.randint(0, NUM_OPTIONS))

            # --- 2. 低层执行：根据当前选项采样动作 ---
            cont_dist, disc_dist = distributions[current_option.value]

            # 采样原始机动动作 u (pre-tanh)
            u_maneuver = cont_dist.mean if deterministic else cont_dist.sample()

            # 采样离散动作
            action_discrete = disc_dist.mean if deterministic else disc_dist.sample()

            # --- 3. 动作整合与输出 ---
            # a) 计算用于存储的log_prob
            log_prob_cont = cont_dist.log_prob(u_maneuver).sum(dim=-1)
            log_prob_disc = disc_dist.log_prob(action_discrete)
            total_log_prob = log_prob_cont + log_prob_disc

            # b) 准备存储到buffer的动作 [u_maneuver, discrete_action]
            action_to_store = torch.cat([u_maneuver, action_discrete.unsqueeze(-1)], dim=-1)

            # c) 准备发送到环境的6维动作
            maneuver_tanh = torch.tanh(u_maneuver)
            env_maneuver = self._scale_maneuver_actions(maneuver_tanh)

            if current_option == Option.ATTACK:
                # 攻击选项: [机动, 诱饵弹(0), 发射导弹]
                env_action = torch.cat(
                    [env_maneuver, torch.zeros(1, 1, device=self.device), action_discrete.unsqueeze(-1)], dim=-1)
            else:  # Option.EVADE
                # 规避选项: [机动, 诱饵弹, 发射导弹(0)]
                env_action = torch.cat(
                    [env_maneuver, action_discrete.unsqueeze(-1), torch.zeros(1, 1, device=self.device)], dim=-1)

        return (
            env_action.squeeze(0).cpu().numpy(),
            action_to_store.squeeze(0).cpu().numpy(),
            total_log_prob.item(),
            current_option
        )

    def store_experience(self, state, option, action, log_prob, reward, done, next_state):
        self.buffer.append((state, option.value, action, log_prob, reward, done, next_state))

    def _compute_advantages_and_returns(self, rewards, dones, values, next_values, term_probs):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        # 价值函数的 TD 目标
        # target = r + gamma * ( (1-beta) * Q(s', w) + beta * max_w' Q(s', w') )
        # V(s') = max_w' Q(s', w')
        next_q_values = next_values.max(axis=-1)  # V(s')

        # 计算下一状态的期望价值 (考虑了终止概率)
        next_expected_values = (1 - term_probs) * next_values[
            np.arange(len(rewards)), self.buffer_options] + term_probs * next_q_values

        gae = 0
        for t in reversed(range(len(rewards))):
            done_mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_expected_values[t] * done_mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * done_mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def learn(self):
        if len(self.buffer) < MINIBATCH_SIZE:
            return None

        # 1. 解包数据
        states, options, actions, old_log_probs, rewards, dones, next_states = zip(*self.buffer)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.buffer_options = np.array(options)  # 保存为numpy数组，方便索引
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        # 2. 计算优势和回报
        with torch.no_grad():
            q_values, term_probs, _ = self.network(states)
            next_q_values, _, _ = self.network(next_states)

            # 获取当前选项的Q值 V_omega(s)
            options_tensor = torch.tensor(self.buffer_options, dtype=torch.long, device=self.device).unsqueeze(1)
            values_omega = q_values.gather(1, options_tensor).squeeze(1)

        advantages, returns = self._compute_advantages_and_returns(
            rewards, dones,
            values_omega.cpu().numpy(),
            next_q_values.cpu().numpy(),
            term_probs.detach().cpu().numpy()
        )

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 3. PPO 学习循环
        for _ in range(PPO_EPOCHS):
            # 创建minibatch索引
            indices = np.random.permutation(len(self.buffer))
            for start in range(0, len(self.buffer), MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                batch_indices = indices[start:end]

                # 提取minibatch数据
                b_states = states[batch_indices]
                b_options = self.buffer_options[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                # --- 核心学习步骤 ---
                q_omega, term_probs, distributions = self.network(b_states)

                b_options = torch.tensor(b_options, dtype=torch.long, device=self.device).unsqueeze(1)

                # --- A. Critic (Q-value) Loss ---
                values_pred = q_omega.gather(1, torch.tensor(b_options, device=self.device).unsqueeze(1)).squeeze(1)
                critic_loss = nn.functional.mse_loss(values_pred, b_returns)

                # --- B. Termination (beta) Loss ---
                # 终止优势 A_beta = Q(s, w) - V(s)
                state_values = q_omega.max(dim=1)[0].detach()  # V(s)
                termination_advantage = (values_pred.detach() - state_values)
                # 终止损失：(终止概率 * 终止优势 + 终止成本)
                termination_loss = (
                            term_probs.gather(1, torch.tensor(b_options, device=self.device).unsqueeze(1)).squeeze(
                                1) * termination_advantage + TERMINATION_COST).mean()

                # --- C. Actor (pi_omega) Loss ---
                # 标准化优势函数
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                log_probs = torch.zeros_like(b_old_log_probs)
                entropies = torch.zeros_like(b_old_log_probs)

                # 分别计算每个选项的损失，然后加权
                for option_idx in range(NUM_OPTIONS):
                    mask = (b_options == option_idx)
                    if not mask.any(): continue

                    cont_dist, disc_dist = distributions[option_idx]

                    # 重新计算log_prob
                    u_maneuver = b_actions[mask, :MANEUVER_DIM]
                    action_disc = b_actions[mask, MANEUVER_DIM]

                    log_prob_cont = cont_dist.log_prob(u_maneuver)[mask].sum(dim=-1)
                    log_prob_disc = disc_dist.log_prob(action_disc)[mask]
                    log_probs[mask] = log_prob_cont + log_prob_disc

                    # 计算熵
                    entropy_cont = cont_dist.entropy()[mask].sum(dim=-1)
                    entropy_disc = disc_dist.entropy()[mask]
                    entropies[mask] = entropy_cont + entropy_disc

                # PPO Clipped Objective
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- 总损失和优化 ---
                total_loss = actor_loss + 0.5 * critic_loss + termination_loss - ENTROPY_COEF * entropies.mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "termination_loss": termination_loss.item(),
            "entropy": entropies.mean().item()
        }