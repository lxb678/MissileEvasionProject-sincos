import torch
import numpy as np
class AGENTPARA:
    '''
    OBS_DIM=输入观测状态维度
    ACTION_DIM=输出动作维度
    MAX_EXE_NUM=最大的训练次数
    RANDOM_SEED=随机种子 用于复现
    epsilon=clip的参数
    gamma=TD折扣参数
    lamda=GAE折扣参数
    entropy=损失中的熵权重
    mini_lr=最小的学习率
    ppo_epoch =每次训练的代数
    '''
    OBS_DIM= 11 #13 #9
    ACTION_DIM= 8 #9
    MAX_EXE_NUM=5e5
    RANDOM_SEED=1
    epsilon=0.2
    gamma=0.99
    lamda=0.95
    entropy= 0.01 #0.01 #0.001 #0.005
    mini_lr=5e-6
    ppo_epoch = 5 #10 #5

class BUFFERPARA:
    '''
    BATCH_SIZE = 样本批次大小
    '''
    BATCH_SIZE = 256


class MODELPARA:
    def __init__(self):
        super(MODELPARA,self).__init__()
        '''
        self.lr =学习率
        self.tau = 在更新模型的时候稳定训练用 
        self.use_PopArt = None   改进PPO 查PopArt
        self.model_layer_dim  每层的维度
        self.device = 训练平台 cuda为gpu cpu为cpu
        self.tpdv = 输入数据的类型
        self.input_dim = 输入数据维度
        self.output_dim = 输出数据维度
        '''
        self.lr = None
        self.gru_lr = None
        self.attention_lr = None
        self.tau = None
        self.use_PopArt = None
        self.model_layer_dim = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.actiontpdv = dict(dtype=torch.int32, device=self.device)
        self.input_dim = None
        self.output_dim = None
def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
ACTOR_PARA = MODELPARA()
ACTOR_PARA.input_dim = AGENTPARA.OBS_DIM
# ACTOR_PARA.model_layer_dim = [256,128,64,32]
# ACTOR_PARA.model_layer_dim = [256,256,256,256]
# ACTOR_PARA.model_layer_dim = [256,256] #[128,128,128] #[256,256]
ACTOR_PARA.model_layer_dim = [128,128,128]#[512,256,128]  #[128,128,128]
# ACTOR_PARA.model_layer_dim = [512,256,128,64]
ACTOR_PARA.output_dim = AGENTPARA.ACTION_DIM   #多维的动作 每一维的动作输出0/1
ACTOR_PARA.lr = 3e-4 #3e-4 #5e-4 #3e-4  #1e-5
ACTOR_PARA.gru_lr = 3e-4 #1e-4 #3e-4       # GRU 层的专属学习率 (通常可以设得小一些)

CRITIC_PARA = MODELPARA()
CRITIC_PARA.input_dim = AGENTPARA.OBS_DIM
# CRITIC_PARA.model_layer_dim = [256,128,64,32]
# CRITIC_PARA.model_layer_dim = [256,256,256,256]
# CRITIC_PARA.model_layer_dim = [256,256] #[128,128,128] #[256,256]
CRITIC_PARA.model_layer_dim = [128,128,128]#[512,256,128] #[128,128,128]
# CRITIC_PARA.model_layer_dim = [512,256,128,64]
CRITIC_PARA.output_dim = 1
CRITIC_PARA.lr = 1e-3 #3e-4 #1e-3
CRITIC_PARA.gru_lr = 1e-3 #3e-4 #1e-3 #1e-3 #1e-4 #ACTOR_PARA.gru_lr