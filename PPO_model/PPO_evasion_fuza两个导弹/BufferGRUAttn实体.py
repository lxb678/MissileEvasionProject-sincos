# --- START OF FILE BufferGRUAttn.py (实体注意力修改版) ---

import numpy as np
# 从配置文件导入 BUFFERPARA，其中包含了像 BATCH_SIZE 这样的超参数。
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *


class Buffer(object):
    '''
    一个通用的经验回放池（Replay Buffer）。
    这个 Buffer 设计得非常灵活，通过布尔标志 `use_rnn` 和 `use_attn` 来决定其工作模式，
    可以支持 MLP、RNN 以及需要处理注意力权重的模型。
    '''

    # <<< 核心修改 1/5 >>>: __init__ 方法增加 use_attn 标志
    def __init__(self, use_rnn=False, use_attn=False):
        """
       Buffer 类的构造函数。
       :param use_rnn: 布尔值，如果为 True，则 Buffer 会额外存储和处理 RNN 的隐藏状态。
       :param use_attn: 布尔值，如果为 True，则 Buffer 会额外存储和处理注意力权重。
        """
        self.batch_size = BUFFERPARA.BATCH_SIZE
        self.use_rnn = use_rnn
        self.use_attn = use_attn  # 新增标志
        self.clear_memory()

    # <<< 核心修改 2/5 >>>: get_all_data 方法增加返回 attention_weights
    def get_all_data(self):
        """
        将所有在 Python 列表中存储的经验数据，统一转换为 NumPy 数组格式并返回。
        """
        actor_hiddens = np.array(self.actor_hidden_states, dtype=np.float32) if self.use_rnn else None
        critic_hiddens = np.array(self.critic_hidden_states, dtype=np.float32) if self.use_rnn else None
        # 如果使用注意力模型，则将存储的权重列表转换为 NumPy 数组
        attention_weights = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.bool_),
            actor_hiddens,
            critic_hiddens,
            attention_weights  # 新增返回值
        )

    def generate_batches(self):
        """
        为 MLP 模型生成随机批次（Batches）。 (此函数无需修改)
        """
        n_states = len(self.states)
        if n_states < self.batch_size:
            return
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        for i in range(0, n_states, self.batch_size):
            yield indices[i:i + self.batch_size]

    # <<< 核心修改 3/5 >>>: generate_sequence_batches 方法增加处理和返回 attention_weights
    def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns):
        """
        为 GRU/RNN 模型生成连续的序列批次。
        现在也支持同时提供注意力权重的序列。
        """
        n_transitions = len(self.states)
        episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]

        valid_seq_starts = []
        for start_idx in episode_starts:
            end_idx = n_transitions
            try:
                end_idx = episode_starts[episode_starts.index(start_idx) + 1]
            except IndexError:
                pass
            episode_len = end_idx - start_idx
            if episode_len >= sequence_length:
                valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))

        if not valid_seq_starts:
            return

        np.random.shuffle(valid_seq_starts)

        # 预先将所有列表转换为 NumPy 数组，以保证高效和正确的切片操作。
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.float32)
        probs_np = np.array(self.probs, dtype=np.float32)
        actor_hiddens_np = np.array(self.actor_hidden_states, dtype=np.float32)
        critic_hiddens_np = np.array(self.critic_hidden_states, dtype=np.float32)
        advantages_np = np.array(advantages, dtype=np.float32)
        returns_np = np.array(returns, dtype=np.float32)
        # 新增：将注意力权重也转换为 NumPy 数组
        attn_weights_np = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None

        for i in range(0, len(valid_seq_starts), batch_size):
            batch_start_indices = valid_seq_starts[i: i + batch_size]
            if len(batch_start_indices) == 0: continue

            state_batch, action_batch, prob_batch = [], [], []
            advantage_batch, return_batch = [], []
            initial_actor_h_batch, initial_critic_h_batch = [], []
            # 新增：初始化用于存储注意力权重批次的列表
            attn_weights_batch = [] if self.use_attn else None

            for start_idx in batch_start_indices:
                end_idx = start_idx + sequence_length

                state_batch.append(states_np[start_idx:end_idx])
                action_batch.append(actions_np[start_idx:end_idx])
                prob_batch.append(probs_np[start_idx:end_idx])
                advantage_batch.append(advantages_np[start_idx:end_idx])
                return_batch.append(returns_np[start_idx:end_idx])
                initial_actor_h_batch.append(actor_hiddens_np[start_idx])
                initial_critic_h_batch.append(critic_hiddens_np[start_idx])
                # 新增：如果使用注意力，则切片出对应的权重序列
                if self.use_attn:
                    attn_weights_batch.append(attn_weights_np[start_idx:end_idx])

            stacked_actor_h = np.stack(initial_actor_h_batch)
            stacked_critic_h = np.stack(initial_critic_h_batch)
            # 新增：如果使用注意力，则将权重批次也堆叠成一个 NumPy 数组
            # 最终 attn_weights_batch 的形状为 (batch_size, sequence_length, num_entities)
            stacked_attn_weights = np.stack(attn_weights_batch) if self.use_attn else None

            # 使用 yield 返回一个完整的批次数据，现在包含了注意力权重
            yield (np.stack(state_batch),
                   np.stack(action_batch),
                   np.stack(prob_batch),
                   np.stack(advantage_batch),
                   np.stack(return_batch),
                   np.transpose(stacked_actor_h, (1, 0, 2)),
                   np.transpose(stacked_critic_h, (1, 0, 2)),
                   stacked_attn_weights  # 新增返回值
                   )

    # <<< 核心修改 4/5 >>>: store_transition 方法增加 attn_weights 参数
    def store_transition(self, state, value, action, probs, reward, done, actor_hidden=None, critic_hidden=None,
                         attn_weights=None):
        """
       向 Buffer 中存储单步的经验数据（一个 transition）。
       现在可以同时存储注意力权重。
       """
        self.states.append(state)
        self.values.append(value)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)

        if self.use_rnn:
            if actor_hidden is None or critic_hidden is None:
                raise ValueError("在 RNN 模式下，隐藏状态必须被提供给 Buffer。")
            self.actor_hidden_states.append(actor_hidden.detach().cpu().numpy().squeeze(1))
            self.critic_hidden_states.append(critic_hidden.detach().cpu().numpy().squeeze(1))

        # 新增：如果使用注意力模型，则存储注意力权重
        if self.use_attn:
            if attn_weights is None:
                raise ValueError("在 Attention 模式下，注意力权重必须被提供给 Buffer。")
            # 注意力权重通常已经是 numpy 数组，无需转换
            self.attention_weights.append(attn_weights)

    # <<< 核心修改 5/5 >>>: clear_memory 方法增加清空 attention_weights
    def clear_memory(self):
        """
        清空 Buffer 中的所有经验数据。
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

        if self.use_rnn:
            self.actor_hidden_states = []
            self.critic_hidden_states = []

        # 新增：如果使用注意力模型，也清空其列表
        if self.use_attn:
            self.attention_weights = []

    def get_buffer_size(self):
        """
        获取当前 Buffer 中存储的转换（transition）数量。
        """
        return len(self.states)