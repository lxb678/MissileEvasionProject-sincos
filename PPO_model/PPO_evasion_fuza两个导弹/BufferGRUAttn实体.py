import numpy as np
# 导入配置文件
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *


class Buffer(object):
    '''
    支持 MLP、RNN (GRU/LSTM) 和 Attention 的通用经验回放池。
    核心升级：增加了对 RNN 隐藏状态的存储，以及按序列长度 (Seq_Len) 采样的方法。
    '''

    def __init__(self, use_rnn=False, use_attn=False):
        """
        :param use_rnn: 是否启用 RNN 模式 (存储 Hidden States, 采样序列)
        :param use_attn: 是否启用 Attention 模式 (存储 Attention Weights)
        """
        self.batch_size = BUFFERPARA.BATCH_SIZE
        self.use_rnn = use_rnn
        self.use_attn = use_attn
        self.clear_memory()

    def clear_memory(self):
        """清空所有存储的经验"""
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        if self.use_rnn:
            self.actor_hidden_states = []
            self.critic_hidden_states = []
        else:
            self.actor_hidden_states = None
            self.critic_hidden_states = None

        if self.use_attn:
            self.attention_weights = []
        else:
            self.attention_weights = None

    def store_transition(self, state, value, action, probs, reward, done,
                         actor_hidden=None, critic_hidden=None, attn_weights=None):
        """
        存储单步交互数据。
        如果 use_rnn=True，必须传入 actor_hidden 和 critic_hidden。
        """
        self.states.append(state)
        self.values.append(value)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)

        # --- 核心修改：处理 RNN 隐藏状态 ---
        if self.use_rnn:
            if actor_hidden is None or critic_hidden is None:
                raise ValueError("[Buffer Error] use_rnn=True 但未传入 hidden_state!")

            # 隐藏状态通常在 GPU 上，形状为 (1, 1, Hidden)，我们需要将其转为 numpy 并去掉多余维度
            # 存入形状: (Hidden_Dim, )
            self.actor_hidden_states.append(actor_hidden.detach().cpu().numpy().squeeze())
            self.critic_hidden_states.append(critic_hidden.detach().cpu().numpy().squeeze())

        # --- 处理 Attention 权重 ---
        if self.use_attn:
            if attn_weights is None:
                # 兼容性处理：如果没传权重但开了attn，暂时存0或者报错，这里选择存None并在取数据时处理
                pass
            else:
                # 假设 attn_weights 已经是 numpy 或 tensor
                if hasattr(attn_weights, 'cpu'):
                    attn_weights = attn_weights.detach().cpu().numpy()
                self.attention_weights.append(attn_weights)

    def get_all_data(self):
        """
        返回所有数据，用于计算 GAE (Generalized Advantage Estimation)。
        此时返回的是由单步数据组成的完整轨迹。
        """
        # 将列表转换为 Numpy 数组
        s = np.array(self.states, dtype=np.float32)
        v = np.array(self.values, dtype=np.float32)
        a = np.array(self.actions, dtype=np.float32)
        p = np.array(self.probs, dtype=np.float32)
        r = np.array(self.rewards, dtype=np.float32)
        d = np.array(self.dones, dtype=np.bool_)

        actor_h = np.array(self.actor_hidden_states, dtype=np.float32) if self.use_rnn else None
        critic_h = np.array(self.critic_hidden_states, dtype=np.float32) if self.use_rnn else None
        attn_w = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None

        return s, v, a, p, r, d, actor_h, critic_h, attn_w

    def generate_batches(self):
        """
        [非 RNN 模式]
        随机打乱所有数据点，生成 Mini-batch。
        适用于 MLP / Attention 模型。
        """
        n_states = len(self.states)
        if n_states < self.batch_size:
            return

        indices = np.arange(n_states)
        np.random.shuffle(indices)

        for i in range(0, n_states, self.batch_size):
            yield indices[i:i + self.batch_size]

    def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns):
        """
        [优化版] 为 GRU/RNN 模型生成连续的序列批次。
        结合了 B 的严谨性（NumPy预处理、Hidden自动转置）和 A 的功能（支持 Attention）。
        """
        if not self.use_rnn:
            raise RuntimeError("调用了 generate_sequence_batches，但 Buffer 初始化时 use_rnn=False")

        n_transitions = len(self.states)

        # -------------------------------------------------------------
        # 1. [B的优化] 预先将所有 List 转换为 NumPy Array
        #    这避免了在循环中对 List 进行切片（慢且易错），确保操作的是 View
        # -------------------------------------------------------------
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.float32)
        probs_np = np.array(self.probs, dtype=np.float32)

        # 确保 advantages 和 returns 也被视为 NumPy 数组
        adv_np = np.array(advantages, dtype=np.float32)
        ret_np = np.array(returns, dtype=np.float32)

        # 隐藏状态通常是 (N, Layers, Hidden)
        actor_h_np = np.array(self.actor_hidden_states, dtype=np.float32)
        critic_h_np = np.array(self.critic_hidden_states, dtype=np.float32)

        # [A的功能] 如果有 Attention，也转为 NumPy
        attn_w_np = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None

        # -------------------------------------------------------------
        # 2. [B的逻辑] 识别 Episode 边界并计算合法的序列起始点
        # -------------------------------------------------------------
        episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]

        valid_seq_starts = []
        for start_idx in episode_starts:
            # 找到当前回合的结束点
            try:
                end_idx = episode_starts[episode_starts.index(start_idx) + 1]
            except IndexError:
                end_idx = n_transitions

            episode_len = end_idx - start_idx

            # 只有当回合长度 >= 序列长度时，才能提取数据
            if episode_len >= sequence_length:
                # 滑动窗口：保证序列的最后一步不超过 end_idx
                valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))

        if not valid_seq_starts:
            return

        # 打乱起始索引
        np.random.shuffle(valid_seq_starts)

        # -------------------------------------------------------------
        # 3. 生成 Batch
        # -------------------------------------------------------------
        for i in range(0, len(valid_seq_starts), batch_size):
            batch_start_indices = valid_seq_starts[i: i + batch_size]

            if len(batch_start_indices) == 0:
                continue

            # 初始化 Batch 容器
            b_states, b_actions, b_probs = [], [], []
            b_adv, b_ret = [], []
            b_actor_h, b_critic_h = [], []
            b_attn = []

            for start_idx in batch_start_indices:
                end_idx = start_idx + sequence_length

                # [B的优化] 在 NumPy 数组上进行切片
                b_states.append(states_np[start_idx:end_idx])
                b_actions.append(actions_np[start_idx:end_idx])
                b_probs.append(probs_np[start_idx:end_idx])
                b_adv.append(adv_np[start_idx:end_idx])
                b_ret.append(ret_np[start_idx:end_idx])

                # Hidden State 只取序列第一步
                b_actor_h.append(actor_h_np[start_idx])
                b_critic_h.append(critic_h_np[start_idx])

                # [A的功能] Attention 切片
                if self.use_attn:
                    b_attn.append(attn_w_np[start_idx:end_idx])

            # -------------------------------------------------------------
            # 4. [B的优化] 堆叠并处理 Hidden State 维度
            # -------------------------------------------------------------
            # 普通数据堆叠: (Batch, Seq, Dim)
            stacked_states = np.stack(b_states)
            stacked_actions = np.stack(b_actions)
            stacked_probs = np.stack(b_probs)
            stacked_adv = np.stack(b_adv)
            stacked_ret = np.stack(b_ret)

            # [A的功能] Attention 堆叠
            stacked_attn = np.stack(b_attn) if self.use_attn else None

            # [B的关键特性] Hidden State 处理
            # 1. 先堆叠 -> (Batch, Layers, Hidden)
            stacked_actor_h_raw = np.stack(b_actor_h)
            stacked_critic_h_raw = np.stack(b_critic_h)

            # --- 修复开始 ---
            # 检查维度。如果只有 2 维 (Batch, Hidden)，说明缺少了 Layers 维度
            if stacked_actor_h_raw.ndim == 2:
                # 手动增加 Layers 维度 -> (Batch, 1, Hidden)
                stacked_actor_h_raw = np.expand_dims(stacked_actor_h_raw, axis=1)

            if stacked_critic_h_raw.ndim == 2:
                stacked_critic_h_raw = np.expand_dims(stacked_critic_h_raw, axis=1)
            # --- 修复结束 ---

            # 2. 再转置 -> (Layers, Batch, Hidden)
            # 这样输出的数据可以直接喂给 PyTorch 的 GRU，不需要在外部 unsqueeze 或 transpose
            final_actor_h = np.transpose(stacked_actor_h_raw, (1, 0, 2))
            final_critic_h = np.transpose(stacked_critic_h_raw, (1, 0, 2))

            yield (
                stacked_states,
                stacked_actions,
                stacked_probs,
                stacked_adv,
                stacked_ret,
                final_actor_h,  # 已转置好的 (Layers, Batch, Hidden)
                final_critic_h,  # 已转置好的 (Layers, Batch, Hidden)
                stacked_attn  # Attention 权重
            )

    def get_buffer_size(self):
        return len(self.states)