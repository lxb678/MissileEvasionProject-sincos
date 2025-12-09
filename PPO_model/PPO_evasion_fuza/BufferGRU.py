# --- START OF FILE BufferGRU.py (最终修正版) ---

import numpy as np
# 从配置文件导入 BUFFERPARA，其中包含了像 BATCH_SIZE 这样的超参数。
from Interference_code.PPO_model.PPO_evasion_fuza.ConfigGRU import *


class Buffer(object):
    '''
    一个通用的经验回放池（Replay Buffer）。
    这个 Buffer 设计得非常灵活，通过一个布尔标志 `use_rnn` 来决定其工作模式，
    从而可以同时支持简单的多层感知机（MLP）模型和需要处理序列数据的循环神经网络（如 GRU）模型。
    '''

    def __init__(self, use_rnn=False):
        """
       Buffer 类的构造函数。
       :param use_rnn: 布尔值，如果为 True，则 Buffer 会额外存储和处理 RNN 的隐藏状态。
        """
        # 从配置中获取批次大小，用于后续生成批次数据。
        self.batch_size = BUFFERPARA.BATCH_SIZE
        # 记录是否为 RNN 模式。
        self.use_rnn = use_rnn
        # 初始化所有存储列表，清空经验池。
        self.clear_memory()

    def get_all_data(self):
        """
        将所有在 Python 列表中存储的经验数据，统一转换为 NumPy 数组格式并返回。
        这样做是为了方便后续进行高效的数值计算（例如在 PyTorch 或 TensorFlow 中）。
        :return: 一个包含所有经验数据的元组。
        """
        # 如果是 RNN 模式，则将存储的隐藏状态列表转换为 NumPy 数组。
        # 否则，将隐藏状态设为 None。
        actor_hiddens = np.array(self.actor_hidden_states, dtype=np.float32) if self.use_rnn else None
        critic_hiddens = np.array(self.critic_hidden_states, dtype=np.float32) if self.use_rnn else None
        # 返回包含所有转换后数据的元组。
        return (
            np.array(self.states, dtype=np.float32),  # 状态
            np.array(self.values, dtype=np.float32),  # 状态值 (V-value)
            np.array(self.actions, dtype=np.float32),  # 采取的动作
            np.array(self.probs, dtype=np.float32),  # 采取动作的对数概率
            np.array(self.rewards, dtype=np.float32),  # 奖励
            np.array(self.dones, dtype=np.bool_),  # 是否结束的标志
            actor_hiddens,  # Actor 的隐藏状态 (RNN 模式)
            critic_hiddens  # Critic 的隐藏状态 (RNN 模式)
        )

    # def generate_batches(self):
    #     """
    #     为 MLP 模型生成随机批次（Batches）。
    #     这种方法通过随机打乱索引来破坏数据间的时间连续性，适用于不依赖于序列信息的模型。
    #     :return: 一个批次数据的索引生成器。
    #     """
    #     # 获取当前存储的转换（transition）总数。
    #     n_states = len(self.states)
    #     # 如果数据量不足一个批次，则不生成。
    #     if n_states < self.batch_size:
    #         return
    #     # 生成一个从 0 到 n_states-1 的索引数组。
    #     indices = np.arange(n_states)
    #     # 随机打乱索引数组，实现无放回的随机抽样。
    #     np.random.shuffle(indices)
    #     # 以 batch_size 为步长，遍历打乱后的索引数组。
    #     for i in range(0, n_states, self.batch_size):
    #         # 使用 yield 关键字返回一个批次的索引，这使得函数成为一个生成器，节省内存。
    #         yield indices[i:i + self.batch_size]
    def generate_batches(self):
        """
        [MLP 模式] 生成随机批次。
        修改点：丢弃末尾不足 batch_size 的数据。
        """
        n_states = len(self.states)

        # 1. 计算有多少个完整的 Batch
        n_full_batches = n_states // self.batch_size

        if n_full_batches == 0:
            return

        indices = np.arange(n_states)
        np.random.shuffle(indices)

        # 2. 只遍历完整的 Batch 数量
        for i in range(n_full_batches):
            # 显式计算起始和结束索引
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size

            # 这样切片出来的长度永远等于 self.batch_size
            yield indices[start_idx: end_idx]

    def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns, verbose=False):
        """
        [RNN 模式] 生成序列批次。
        修改点：丢弃末尾不足 batch_size 的序列数据。
        """
        n_transitions = len(self.states)

        # 1. 确定所有回合的起始点
        episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]

        valid_seq_starts = []
        for start_idx in episode_starts:
            try:
                end_idx = episode_starts[episode_starts.index(start_idx) + 1]
            except IndexError:
                end_idx = n_transitions

            episode_len = end_idx - start_idx
            if episode_len >= sequence_length:
                # 依然使用滑动窗口采样 (stride=1)，最大限度利用数据
                valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))

        # --- [Debug 输出 1] 数据总量统计 ---
        if verbose:
            print(f"\n{'=' * 20} Buffer 采样统计 {'=' * 20}")
            print(f"原始步数 (Transitions): {n_transitions}")
            print(f"有效序列数 (Valid Seqs): {len(valid_seq_starts)} (Seq_Len={sequence_length})")

        # 如果没有有效序列，退出
        if not valid_seq_starts:
            if verbose: print("❌ 没有有效序列，跳过训练。")
            return

        # 如果没有有效序列，退出
        if not valid_seq_starts:
            return

        # 2. 打乱序列起始索引
        np.random.shuffle(valid_seq_starts)

        # 3. 转换为 NumPy 数组 (加速切片)
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.float32)
        probs_np = np.array(self.probs, dtype=np.float32)
        # 注意: 如果 advantages/returns 传入时已经是 numpy，这里会自动处理，如果是 list 则转换
        adv_np = np.array(advantages, dtype=np.float32)
        ret_np = np.array(returns, dtype=np.float32)

        actor_h_np = np.array(self.actor_hidden_states, dtype=np.float32)
        critic_h_np = np.array(self.critic_hidden_states, dtype=np.float32)

        # 4. [核心修改] 计算完整的 Batch 数量 (Drop Last 逻辑)
        n_sequences = len(valid_seq_starts)
        n_full_batches = n_sequences // batch_size
        remainder = n_sequences % batch_size

        # --- [Debug 输出 2] Batch 规划 ---
        if verbose:
            print(f"Batch Size 设置:      {batch_size}")
            print(f"计划生成 Batch 数:    {n_full_batches}")
            print(f"丢弃余数序列:         {remainder} 条")
            if n_full_batches == 0:
                print("❌ [警告] 有效序列不足一个 Batch，本次不进行更新！")
            print("-" * 60)

        if n_full_batches == 0:
            # print("数据不足一个 Batch，跳过训练")
            return

        # 5. 循环生成
        for i in range(n_full_batches):
            # 计算切片范围
            batch_idx_start = i * batch_size
            batch_idx_end = (i + 1) * batch_size

            # 获取这一个 Batch 的所有序列起始点
            batch_start_indices = valid_seq_starts[batch_idx_start: batch_idx_end]

            # 容器初始化
            b_states, b_actions, b_probs = [], [], []
            b_adv, b_ret = [], []
            b_actor_h, b_critic_h = [], []

            for start_idx in batch_start_indices:
                end_idx = start_idx + sequence_length

                b_states.append(states_np[start_idx:end_idx])
                b_actions.append(actions_np[start_idx:end_idx])
                b_probs.append(probs_np[start_idx:end_idx])
                b_adv.append(adv_np[start_idx:end_idx])
                b_ret.append(ret_np[start_idx:end_idx])

                # Hidden state 只取序列第一步
                b_actor_h.append(actor_h_np[start_idx])
                b_critic_h.append(critic_h_np[start_idx])

            # 堆叠
            stacked_actor_h = np.stack(b_actor_h)
            stacked_critic_h = np.stack(b_critic_h)

            # 堆叠与转置
            # 最终状态形状: (Batch, Seq_Len, State_Dim)
            final_states = np.stack(b_states)
            final_actions = np.stack(b_actions)
            final_probs = np.stack(b_probs)
            final_adv = np.stack(b_adv)
            final_ret = np.stack(b_ret)

            # --- [Debug 输出 3] 每个 Batch 的具体形状 ---
            if verbose:
                print(f"  -> Yielding Batch {i + 1}/{n_full_batches}")
                print(f"     States Shape: {final_states.shape} (Batch, Seq, Dim)")
                print(f"     Actor H Shape: {stacked_actor_h.shape} (Layers, Batch, Hidden)")

            yield (
                np.stack(b_states),
                np.stack(b_actions),
                np.stack(b_probs),
                np.stack(b_adv),
                np.stack(b_ret),
                # 转置 Hidden State 为 (Layers, Batch, Hidden) 以适配 PyTorch
                np.transpose(stacked_actor_h, (1, 0, 2)),
                np.transpose(stacked_critic_h, (1, 0, 2))
            )


    # def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns):
    #     """
    #     为 GRU/RNN 模型生成连续的序列批次。
    #     这个方法的核心是保证每个批次中的数据都是时间上连续的片段，并且不会跨越回合（episode）的边界。
    #     :param sequence_length: 每个序列的长度。
    #     :param batch_size: 每个批次包含多少个序列。
    #     :param advantages: 优势函数值的列表或数组。
    #     :param returns: 回报（G_t）的列表或数组。
    #     :return: 一个序列批次数据的生成器。
    #     """
    #     # 获取存储的转换总数。
    #     n_transitions = len(self.states)
    #     # 找到所有回合的起始点。一个回合的结束（done=True）意味着下一个时间步是新回合的开始。
    #     # 我们将 0 作为第一个回合的起点，然后将每个 done=True 的下一个索引也标记为起点。
    #     episode_starts = [0] + [i + 1 for i, done in enumerate(self.dones) if done and i < n_transitions - 1]
    #     # 存储所有有效的序列起始索引。
    #     valid_seq_starts = []
    #     # 遍历每个回合的起始点，以确定该回合内所有可能的、长度为 `sequence_length` 的序列。
    #     for start_idx in episode_starts:
    #         # 确定当前回合的结束点。
    #         end_idx = n_transitions
    #         try:
    #             # 找到下一个回合的起点，它就是当前回合的终点。
    #             end_idx = episode_starts[episode_starts.index(start_idx) + 1]
    #         except IndexError:
    #             # 如果是最后一个回合，则终点就是数据总长度。
    #             pass
    #         # 计算当前回合的长度。
    #         episode_len = end_idx - start_idx
    #         # 如果回合长度足够容纳一个序列，则计算该回合内所有可能的序列起始点。
    #         if episode_len >= sequence_length:
    #             # 例如，回合长度为10，序列长度为4，则有效的起始点为 0, 1, 2, 3, 4, 5, 6。
    #             valid_seq_starts.extend(range(start_idx, end_idx - sequence_length + 1))
    #     # 如果找不到任何有效的序列，则直接返回。
    #     if not valid_seq_starts:
    #         return
    #     # 随机打乱所有有效序列的起始点。
    #     # 注意：这只是打乱了序列的顺序，而每个序列内部的时间连续性得到了保留。
    #     np.random.shuffle(valid_seq_starts)
    #
    #     # ##############################################################
    #     # # <<< 这里是问题的根源和最终的修复 >>>
    #     # ##############################################################
    #     # 将所有列表预先转换为 NumPy 数组，以保证后续切片操作的正确性。
    #     # 之前的 Bug 是因为 advantages 和 returns 作为函数参数传入，它们可能是 Python 列表，
    #     # 在循环中对列表进行切片比对 NumPy 数组进行切片效率低得多，且行为可能不完全一致。
    #     # 统一转换为 NumPy 数组可以确保高效和正确的切片操作。
    #     states_np = np.array(self.states, dtype=np.float32)
    #     actions_np = np.array(self.actions, dtype=np.float32)
    #     probs_np = np.array(self.probs, dtype=np.float32)
    #     actor_hiddens_np = np.array(self.actor_hidden_states, dtype=np.float32)
    #     critic_hiddens_np = np.array(self.critic_hidden_states, dtype=np.float32)
    #     # 确保 advantages 和 returns 也被视为 NumPy 数组进行切片
    #     advantages_np = np.array(advantages, dtype=np.float32)
    #     returns_np = np.array(returns, dtype=np.float32)
    #     # ##############################################################
    #     # 以 batch_size 为步长，遍历所有打乱后的序列起始点，构造批次。
    #     for i in range(0, len(valid_seq_starts), batch_size):
    #         # 获取当前批次的所有序列的起始索引。
    #         batch_start_indices = valid_seq_starts[i: i + batch_size]
    #         # 如果该批次为空，则跳过。
    #         if len(batch_start_indices) == 0: continue
    #         # 初始化用于存储当前批次数据的列表。
    #         state_batch, action_batch, prob_batch = [], [], []
    #         advantage_batch, return_batch = [], []
    #         initial_actor_h_batch, initial_critic_h_batch = [], []
    #         # 遍历当前批次中的每个序列起始点。
    #         for start_idx in batch_start_indices:
    #             # 计算序列的结束点。
    #             end_idx = start_idx + sequence_length
    #             # 从 NumPy 数组中切片出整个序列的数据。
    #             state_batch.append(states_np[start_idx:end_idx])
    #             action_batch.append(actions_np[start_idx:end_idx])
    #             prob_batch.append(probs_np[start_idx:end_idx])
    #             # 现在，我们是在 NumPy 数组上进行正确的切片操作
    #             advantage_batch.append(advantages_np[start_idx:end_idx])
    #             return_batch.append(returns_np[start_idx:end_idx])
    #             # 对于 RNN，我们需要提供序列开始时的初始隐藏状态。
    #             initial_actor_h_batch.append(actor_hiddens_np[start_idx])
    #             initial_critic_h_batch.append(critic_hiddens_np[start_idx])
    #         # 使用 np.stack 将列表中的多个隐藏状态数组堆叠成一个更高维度的数组。
    #         # 此时形状为 (batch_size, num_layers, hidden_dim)。
    #         stacked_actor_h = np.stack(initial_actor_h_batch)
    #         stacked_critic_h = np.stack(initial_critic_h_batch)
    #         # 使用 yield 返回一个完整的批次数据。
    #         # np.stack 将序列列表（例如 state_batch）堆叠成一个大的 NumPy 数组。
    #         # 最终 state_batch 的形状为 (batch_size, sequence_length, state_dim)。
    #         # 对隐藏状态进行转置（transpose），将其形状从 (batch_size, num_layers, hidden_dim)
    #         # 变为 (num_layers, batch_size, hidden_dim)，以匹配 PyTorch GRU 层的输入要求。
    #         yield (np.stack(state_batch),
    #                np.stack(action_batch),
    #                np.stack(prob_batch),
    #                np.stack(advantage_batch),
    #                np.stack(return_batch),
    #                np.transpose(stacked_actor_h, (1, 0, 2)),
    #                np.transpose(stacked_critic_h, (1, 0, 2)))

    def store_transition(self, state, value, action, probs, reward, done, actor_hidden=None, critic_hidden=None):
        """
       向 Buffer 中存储单步的经验数据（一个 transition）。
       """
        self.states.append(state)
        self.values.append(value)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)
        # 如果是 RNN 模式，则必须存储隐藏状态。
        if self.use_rnn:
            if actor_hidden is None or critic_hidden is None:
                raise ValueError("在 RNN 模式下，隐藏状态必须被提供给 Buffer。")
            # 在存储隐藏状态前，进行处理：
            # .detach(): 从计算图中分离张量，防止梯度信息被存储。
            # .cpu(): 将张量移动到 CPU（如果它在 GPU 上）。
            # .numpy(): 将 PyTorch 张量转换为 NumPy 数组。
            # .squeeze(1): 移除批次维度。因为在采集数据时，批次大小通常为1，
            #             隐藏状态的形状可能是 (num_layers, 1, hidden_dim)，squeeze(1) 后变为 (num_layers, hidden_dim)。
            self.actor_hidden_states.append(actor_hidden.detach().cpu().numpy().squeeze(1))
            self.critic_hidden_states.append(critic_hidden.detach().cpu().numpy().squeeze(1))

    def clear_memory(self):
        """
                清空 Buffer 中的所有经验数据。
                通常在完成一次学习更新（例如一个 PPO epoch）后调用。
                """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        # 如果是 RNN 模式，也清空隐藏状态列表。
        if self.use_rnn:
            self.actor_hidden_states = []
            self.critic_hidden_states = []

    def get_buffer_size(self):
        """
               获取当前 Buffer 中存储的转换（transition）数量。
               :return: 存储的转换数量。
               """
        return len(self.states)