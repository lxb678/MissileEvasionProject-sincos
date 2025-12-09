# --- START OF FILE BufferGRUAttn实体.py (通用适配版) ---

import numpy as np
# 从配置文件导入 BUFFERPARA，其中包含了像 BATCH_SIZE 这样的超参数。
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *


class Buffer(object):
    '''
    一个通用的经验回放池（Replay Buffer）。
    这个 Buffer 设计得非常灵活，通过布尔标志 `use_rnn` 和 `use_attn` 来决定其工作模式，
    可以支持 MLP、RNN 以及需要处理注意力权重的模型。
    '''

    def __init__(self, use_rnn=False, use_attn=False):
        """
       Buffer 类的构造函数。
       :param use_rnn: 布尔值，如果为 True，则 Buffer 会额外存储和处理 RNN 的隐藏状态。
       :param use_attn: 布尔值，如果为 True，则 Buffer 会额外存储和处理注意力权重。
        """
        self.batch_size = BUFFERPARA.BATCH_SIZE
        self.use_rnn = use_rnn
        self.use_attn = use_attn
        self.clear_memory()

    def get_all_data(self):
        """
        将所有在 Python 列表中存储的经验数据，统一转换为 NumPy 数组格式并返回。
        """
        # <<< 核心修改 1/4 >>>: 确保在非RNN模式下 actor_hiddens 和 critic_hiddens 是 None
        actor_hiddens = np.array(self.actor_hidden_states, dtype=np.float32) if self.use_rnn and self.actor_hidden_states else None
        critic_hiddens = np.array(self.critic_hidden_states, dtype=np.float32) if self.use_rnn and self.critic_hidden_states else None
        attention_weights = np.array(self.attention_weights, dtype=np.float32) if self.use_attn and self.attention_weights else None

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.bool_),
            actor_hiddens,
            critic_hiddens,
            attention_weights
        )

    def generate_batches(self, verbose=False):
        """
        [MLP 模式] 生成随机 Batch
        修改点：
        1. 丢弃末尾不足 batch_size 的数据 (Drop Last)
        2. 增加详细的 Debug 输出
        """
        n_states = len(self.states)

        # 计算完整 Batch 数
        n_full_batches = n_states // self.batch_size
        remainder = n_states % self.batch_size

        # --- [Debug 输出] ---
        if verbose:
            print(f"\n{'=' * 20} Buffer 采样统计 (MLP) {'=' * 20}")
            print(f"原始步数:       {n_states}")
            print(f"Batch Size:     {self.batch_size}")
            print(f"计划生成 Batch: {n_full_batches}")
            print(f"丢弃余数数据:   {remainder} 条")

            if n_full_batches == 0:
                print("❌ [警告] 数据不足一个 Batch，本次不更新！")
            print("-" * 60)

        if n_full_batches == 0:
            return

        indices = np.arange(n_states)
        np.random.shuffle(indices)

        for i in range(n_full_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size

            batch_indices = indices[start_idx: end_idx]

            if verbose:
                # 打印当前 Batch 的序号和实际大小 (应该永远等于 Batch Size)
                print(f"  -> Yielding Batch {i + 1}/{n_full_batches} | Size: {len(batch_indices)}")

            yield batch_indices

        if verbose:
            print(f"{'=' * 20} 采样结束 {'=' * 20}\n")

    def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns, verbose=True):
        """
        [RNN 模式] 生成序列 Batch
        修改点：
        1. 增加 verbose 输出
        2. 丢弃末尾不足 batch_size 的序列 (Drop Last)
        """
        if not self.use_rnn:
            raise RuntimeError("调用了 generate_sequence_batches，但 use_rnn=False")

        n_transitions = len(self.states)

        # 1. 确定 Episode 边界
        episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]

        valid_seq_starts = []
        for start_idx in episode_starts:
            try:
                end_idx = episode_starts[episode_starts.index(start_idx) + 1]
            except IndexError:
                end_idx = n_transitions

            episode_len = end_idx - start_idx
            if episode_len >= sequence_length:
                # 依然使用滑动窗口采样 (Stride=1)，最大化利用数据
                valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))

        # --- [Debug 输出] ---
        if verbose:
            print(f"\n{'=' * 20} Buffer 采样统计 (RNN+Attn) {'=' * 20}")
            print(f"原始步数: {n_transitions}")
            print(f"有效序列数: {len(valid_seq_starts)} (Seq_Len={sequence_length})")

        if not valid_seq_starts:
            if verbose: print("❌ 无有效序列，跳过训练。")
            return

        np.random.shuffle(valid_seq_starts)

        # 2. 转为 NumPy (加速切片)
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.float32)
        probs_np = np.array(self.probs, dtype=np.float32)

        # Values 如果需要切片给外面用，也可以加进来
        # values_np = np.array(self.values, dtype=np.float32)

        adv_np = np.array(advantages, dtype=np.float32)
        ret_np = np.array(returns, dtype=np.float32)

        actor_h_np = np.array(self.actor_hidden_states, dtype=np.float32)
        critic_h_np = np.array(self.critic_hidden_states, dtype=np.float32)
        attn_w_np = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None

        # 3. 计算完整的 Batch 数量 (Drop Last)
        n_sequences = len(valid_seq_starts)
        n_full_batches = n_sequences // batch_size
        remainder = n_sequences % batch_size

        if verbose:
            print(f"Batch Size: {batch_size}")
            print(f"计划生成 Batch 数: {n_full_batches}")
            print(f"丢弃余数序列: {remainder}")
            if n_full_batches == 0:
                print("❌ [警告] 有效序列不足一个 Batch，本次不更新！")
            print("-" * 60)

        if n_full_batches == 0:
            return

        # 4. 生成 Batch
        for i in range(n_full_batches):
            batch_idx_start = i * batch_size
            batch_idx_end = (i + 1) * batch_size

            batch_start_indices = valid_seq_starts[batch_idx_start: batch_idx_end]

            # 初始化容器
            b_states, b_actions, b_probs = [], [], []
            b_adv, b_ret = [], []
            b_actor_h, b_critic_h = [], []
            b_attn = [] if self.use_attn else None

            for start_idx in batch_start_indices:
                end_idx = start_idx + sequence_length

                b_states.append(states_np[start_idx:end_idx])
                b_actions.append(actions_np[start_idx:end_idx])
                b_probs.append(probs_np[start_idx:end_idx])
                b_adv.append(adv_np[start_idx:end_idx])
                b_ret.append(ret_np[start_idx:end_idx])

                # Hidden State 只取序列第一步
                b_actor_h.append(actor_h_np[start_idx])
                b_critic_h.append(critic_h_np[start_idx])

                # Attention 也要切片
                if self.use_attn:
                    b_attn.append(attn_w_np[start_idx:end_idx])

            # 堆叠
            # 状态: (Batch, Seq, Dim)
            stacked_states = np.stack(b_states)
            stacked_actions = np.stack(b_actions)
            stacked_probs = np.stack(b_probs)
            stacked_adv = np.stack(b_adv)
            stacked_ret = np.stack(b_ret)

            # Hidden: (Batch, Layers, Hidden) -> 转置为 (Layers, Batch, Hidden)
            stacked_actor_h = np.transpose(np.stack(b_actor_h), (1, 0, 2))
            stacked_critic_h = np.transpose(np.stack(b_critic_h), (1, 0, 2))

            # Attn: (Batch, Seq, Attn_Dim)
            stacked_attn = np.stack(b_attn) if self.use_attn else None

            if verbose:
                print(f"  -> Batch {i + 1}/{n_full_batches} | "
                      f"State:{stacked_states.shape} | "
                      f"Hidden:{stacked_actor_h.shape}")

            yield (
                stacked_states,
                stacked_actions,
                stacked_probs,
                stacked_adv,
                stacked_ret,
                stacked_actor_h,
                stacked_critic_h,
                stacked_attn
            )

    # def generate_batches(self):
    #     """
    #     为 MLP / Attention+MLP 模型生成随机批次（Batches）。
    #     """
    #     n_states = len(self.states)
    #
    #     # self.batch_size = n_states
    #
    #     if n_states < self.batch_size:
    #         return
    #     indices = np.arange(n_states)
    #     np.random.shuffle(indices)
    #     for i in range(0, n_states, self.batch_size):
    #         yield indices[i:i + self.batch_size]
    #
    # def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns):
    #     """
    #     为 GRU/RNN 模型生成连续的序列批次。
    #     """
    #     # <<< 核心修改 2/4 >>>: 如果不是RNN模式，直接抛出错误，防止被误用
    #     if not self.use_rnn:
    #         raise RuntimeError("generate_sequence_batches should only be called when use_rnn is True.")
    #
    #     n_transitions = len(self.states)
    #     episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]
    #
    #     valid_seq_starts = []
    #     for start_idx in episode_starts:
    #         end_idx = n_transitions
    #         try:
    #             end_idx = episode_starts[episode_starts.index(start_idx) + 1]
    #         except IndexError:
    #             pass
    #         episode_len = end_idx - start_idx
    #         if episode_len >= sequence_length:
    #             valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))
    #
    #     if not valid_seq_starts:
    #         return
    #
    #     np.random.shuffle(valid_seq_starts)
    #
    #     states_np = np.array(self.states, dtype=np.float32)
    #     actions_np = np.array(self.actions, dtype=np.float32)
    #     probs_np = np.array(self.probs, dtype=np.float32)
    #     advantages_np = np.array(advantages, dtype=np.float32)
    #     returns_np = np.array(returns, dtype=np.float32)
    #     # 只有在 use_rnn 为 True 时，这些列表才会有内容
    #     actor_hiddens_np = np.array(self.actor_hidden_states, dtype=np.float32)
    #     critic_hiddens_np = np.array(self.critic_hidden_states, dtype=np.float32)
    #     attn_weights_np = np.array(self.attention_weights, dtype=np.float32) if self.use_attn else None
    #
    #     for i in range(0, len(valid_seq_starts), batch_size):
    #         batch_start_indices = valid_seq_starts[i: i + batch_size]
    #         if len(batch_start_indices) == 0: continue
    #
    #         state_batch, action_batch, prob_batch = [], [], []
    #         advantage_batch, return_batch = [], []
    #         initial_actor_h_batch, initial_critic_h_batch = [], []
    #         attn_weights_batch = [] if self.use_attn else None
    #
    #         for start_idx in batch_start_indices:
    #             end_idx = start_idx + sequence_length
    #             state_batch.append(states_np[start_idx:end_idx])
    #             action_batch.append(actions_np[start_idx:end_idx])
    #             prob_batch.append(probs_np[start_idx:end_idx])
    #             advantage_batch.append(advantages_np[start_idx:end_idx])
    #             return_batch.append(returns_np[start_idx:end_idx])
    #             initial_actor_h_batch.append(actor_hiddens_np[start_idx])
    #             initial_critic_h_batch.append(critic_hiddens_np[start_idx])
    #             if self.use_attn:
    #                 attn_weights_batch.append(attn_weights_np[start_idx:end_idx])
    #
    #         stacked_actor_h = np.stack(initial_actor_h_batch)
    #         stacked_critic_h = np.stack(initial_critic_h_batch)
    #         stacked_attn_weights = np.stack(attn_weights_batch) if self.use_attn else None
    #
    #         yield (np.stack(state_batch),
    #                np.stack(action_batch),
    #                np.stack(prob_batch),
    #                np.stack(advantage_batch),
    #                np.stack(return_batch),
    #                np.transpose(stacked_actor_h, (1, 0, 2)),
    #                np.transpose(stacked_critic_h, (1, 0, 2)),
    #                stacked_attn_weights
    #                )

    # <<< 核心修改 3/4 >>>: store_transition 方法参数顺序调整，使其更通用
    def store_transition(self, state, value, action, probs, reward, done,
                         actor_hidden=None, critic_hidden=None, attn_weights=None):
        """
       向 Buffer 中存储单步的经验数据（一个 transition）。
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

        if self.use_attn:
            if attn_weights is None:
                raise ValueError("在 Attention 模式下，注意力权重必须被提供给 Buffer。")
            self.attention_weights.append(attn_weights)

    # <<< 核心修改 4/4 >>>: clear_memory 方法，根据标志初始化列表
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

        # 根据标志决定是否需要初始化对应的列表
        if self.use_rnn:
            self.actor_hidden_states = []
            self.critic_hidden_states = []
        else: # 明确地将它们设为None，避免混淆
            self.actor_hidden_states = None
            self.critic_hidden_states = None

        if self.use_attn:
            self.attention_weights = []
        else:
            self.attention_weights = None


    def get_buffer_size(self):
        """
        获取当前 Buffer 中存储的转换（transition）数量。
        """
        return len(self.states)