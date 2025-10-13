from Interference_code.PPO_model.PPO_evasion_fuza.Config import *

class Buffer(object):
    '''
    经验池 不指定经验池大小，而是根据对抗结束的次数来触发训练  只需要确定采样的样本批次大小
    '''
    def __init__(self):
        self.batch_size = BUFFERPARA.BATCH_SIZE
        self.states = []
        self.values = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def sample(self):
        return (
            np.array(self.states),
            np.array(self.values),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.rewards),
            np.array(self.dones),
        )

    def generate_batches(self):
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        ids = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(ids)
        batches = [
            ids[i * self.batch_size: (i + 1) * self.batch_size]
            for i in range(n_batches)
        ]
        return batches

    def store_transition(self, state, value, action, probs, reward, done):
        self.states.append(state)
        self.values.append(value)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []






