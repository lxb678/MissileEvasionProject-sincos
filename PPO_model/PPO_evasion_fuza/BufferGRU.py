# --- START OF FILE BufferGRU.py (最终修正版) ---

import numpy as np
from Interference_code.PPO_model.PPO_evasion_fuza.Config import *


class Buffer(object):
    '''
    一个通用的经验池，通过 use_rnn 标志来支持 MLP 和 GRU 模型。
    '''

    def __init__(self, use_rnn=False):
        self.batch_size = BUFFERPARA.BATCH_SIZE
        self.use_rnn = use_rnn
        self.clear_memory()

    def get_all_data(self):
        """将所有存储的数据转换为 NumPy 数组并返回。"""
        actor_hiddens = np.array(self.actor_hidden_states, dtype=np.float32) if self.use_rnn else None
        critic_hiddens = np.array(self.critic_hidden_states, dtype=np.float32) if self.use_rnn else None

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.bool_),
            actor_hiddens,
            critic_hiddens
        )

    def generate_batches(self):
        """为 MLP 模型生成随机批次。"""
        n_states = len(self.states)
        if n_states < self.batch_size:
            return

        indices = np.arange(n_states)
        np.random.shuffle(indices)
        for i in range(0, n_states, self.batch_size):
            yield indices[i:i + self.batch_size]

    def generate_sequence_batches(self, sequence_length, batch_size, advantages, returns):
        """为 GRU/RNN 模型生成连续的序列批次。"""
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

        # ##############################################################
        # # <<< 这里是问题的根源和最终的修复 >>>
        # ##############################################################
        # 将所有列表预先转换为 NumPy 数组，以保证后续切片操作的正确性。
        # 之前的 Bug 是因为 advantages 和 returns 作为函数参数传入，没有被转换。
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.float32)
        probs_np = np.array(self.probs, dtype=np.float32)
        actor_hiddens_np = np.array(self.actor_hidden_states, dtype=np.float32)
        critic_hiddens_np = np.array(self.critic_hidden_states, dtype=np.float32)
        # 确保 advantages 和 returns 也被视为 NumPy 数组进行切片
        advantages_np = np.array(advantages, dtype=np.float32)
        returns_np = np.array(returns, dtype=np.float32)
        # ##############################################################

        for i in range(0, len(valid_seq_starts), batch_size):
            batch_start_indices = valid_seq_starts[i: i + batch_size]
            if len(batch_start_indices) == 0: continue

            state_batch, action_batch, prob_batch = [], [], []
            advantage_batch, return_batch = [], []
            initial_actor_h_batch, initial_critic_h_batch = [], []

            for start_idx in batch_start_indices:
                end_idx = start_idx + sequence_length

                state_batch.append(states_np[start_idx:end_idx])
                action_batch.append(actions_np[start_idx:end_idx])
                prob_batch.append(probs_np[start_idx:end_idx])
                # 现在，我们是在 NumPy 数组上进行正确的切片操作
                advantage_batch.append(advantages_np[start_idx:end_idx])
                return_batch.append(returns_np[start_idx:end_idx])

                initial_actor_h_batch.append(actor_hiddens_np[start_idx])
                initial_critic_h_batch.append(critic_hiddens_np[start_idx])

            stacked_actor_h = np.stack(initial_actor_h_batch)
            stacked_critic_h = np.stack(initial_critic_h_batch)

            yield (np.stack(state_batch),
                   np.stack(action_batch),
                   np.stack(prob_batch),
                   np.stack(advantage_batch),
                   np.stack(return_batch),
                   np.transpose(stacked_actor_h, (1, 0, 2)),
                   np.transpose(stacked_critic_h, (1, 0, 2)))

    def store_transition(self, state, value, action, probs, reward, done, actor_hidden=None, critic_hidden=None):
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

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        if self.use_rnn:
            self.actor_hidden_states = []
            self.critic_hidden_states = []

    def get_buffer_size(self):
        return len(self.states)