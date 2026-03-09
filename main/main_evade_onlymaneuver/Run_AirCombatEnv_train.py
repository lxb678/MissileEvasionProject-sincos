import sys
import os

# 获取当前脚本的绝对路径，并向上推导到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_folder -> main -> Interference_code -> 规避导弹项目sincos
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)
import gym
import torch
import numpy as np
import random
from Interference_code.PPO_model.PPO_evasion_onlymaneuver.Hybrid_PPO_jsbsim import *
from Interference_code.PPO_model.PPO_evasion_onlymaneuver.Config import *
from torch.utils.tensorboard import SummaryWriter
#from env.AirCombatEnv import *
from Interference_code.env.AirCombatEnv6_maneuver_flare import *
from Interference_code.env.missile_evasion_environment.missile_evasion_environment import *
from Interference_code.env.missile_evasion_environment_jsbsim_onlymaneuver.Vec_missile_evasion_environment_jsbsim import *
import time

LOAD_ABLE = False  #鏄惁浣跨敤save鏂囦欢澶逛腑鐨勬ā鍨?

# <<<--- Tacview 鍙鍖栧紑鍏?---<<<
# 灏嗘椤硅涓?True 鍗冲彲鍦ㄨ缁冩椂寮€鍚?Tacview
TACVIEW_ENABLED_DURING_TRAINING = False
# ---
def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    ''' 璁剧疆闅忔満绉嶅瓙 '''
    #env.action_space.seed(seed)   #鍙敞閲?
    #env.reset(seed=seed)          #鍙敞閲?
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#璁板綍璁粌鐨勬崯澶辩瓑鏁板€硷紝鐢ㄤ簬缁樺埗鍥捐〃  浣跨敤tensorboard --logdir= 璺緞 鐨勫懡浠ょ粯鍒?鏂囦欢鍚嶆槸闅忔満绉嶅瓙-璁粌鏃ユ湡-鏄惁浣跨敤鍌ㄥ瓨鐨勬ā鍨?
writer = SummaryWriter(log_dir='../../log/log_evade_onlymaneuver/Trans_seed{}_time_{}_loadable_{}'.format(AGENTPARA.RANDOM_SEED,time.strftime("%m_%d_%H_%M_%S", time.localtime()),LOAD_ABLE))


env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)   #鐜 鍚庣画鐢ㄨ嚜宸卞啓鐨勭幆澧冩浛鎹? 涓嶄娇鐢╣ym 杩欓噷鍙槸浣滀负楠岃瘉
set_seed(env)   #璁剧疆鐜鐨勯殢鏈虹瀛愶紝濡傛灉杩欓噷鎶ラ敊  娉ㄩ噴鎺夊嚱鏁颁腑鏍囪鐨勯儴鍒?
agent = PPO_continuous(LOAD_ABLE)
global_step = 0               #鎬讳氦浜掓鏁?
success_num = 0
for i_episode in range(100000):
    observation, info = env.reset(seed=AGENTPARA.RANDOM_SEED)   #鑷繁鍐欑殑鐜鍙互鏀逛负 env.reset()   杩欎釜鍑芥暟杩斿洖鍒濆鍖栨椂鐨勮娴嬬姸鎬?

    # <<<--- 鍦ㄨ繖閲屾鏌?---<<<
    if np.isnan(observation).any():
        print("!!! FATAL: env.reset() returned NaN!")
        exit()  # 鎴栬€?raise an error

    # rewards = 0
    done = False
    step = 0
    global_reward = 0
    action_list = []
    for t in range(10000):
        agent.prep_eval_rl()    #浜や簰缁忛獙鐨勬椂鍊欎笉浼氫紶姊害
        with torch.no_grad():   #浜や簰鏃舵搴︿笉鍥炰紶
            env_action, action_tanh, prob = agent.choose_action(observation)
            value = agent.get_value(observation).cpu().detach().numpy()
            state = observation
        observation, reward, done, _, _ = env.step(env_action)
        global_reward += reward
        agent.buffer.store_transition(state, value, action_tanh, prob, reward, done)  # 鏀堕泦缁忛獙

        global_step += 1
        step += 1

        if done:
            print("Episode {} finished after {} timesteps,浠跨湡鏃堕棿t = {}s,鍥炲悎濂栧姳:{}".format(i_episode+1, step,round(env.t_now, 2), global_reward))
            if env.success:
                success_num += 1
            if (i_episode + 1) % 100 == 0:
                print("姣忎竴鐧惧洖鍚堥鏈哄瓨娲绘鏁皗} ".format(success_num))
                if success_num >= 90:
                    # agent.save()
                    agent.save(f"success_{success_num}_ep{i_episode + 1}")
                writer.add_scalar('success_num',
                                  success_num,
                                  global_step=global_step
                                  )
                success_num = 0
            break
    #env.render()        #杩欎釜鏄彲瑙嗗寲鐨? 鑷繁鍐欑殑鐜鍙互娉ㄩ噴鎺?
    if i_episode % 10 == 0 and i_episode != 0:  # 姣忎氦浜?0灞€璁粌涓€娆?
        print("train, global_step:{}".format(global_step))
        agent.prep_training_rl()
        train_info = agent.learn()  # 鑾峰緱璁粌涓殑淇℃伅锛岀敤浜庣粯鍒跺浘琛?
        for item in list(train_info.keys()):
            writer.add_scalar(f"train/{item}",  # 鍔犱笂 "train/" 鍓嶇紑
                              train_info[item],
                              global_step=global_step
                              )
        #璁粌瀹屼箣鍚庯紝闇€瑕侀獙璇佹ā鍨嬶紝缁樺埗濂栧姳鏇茬嚎(杩欎釜娴嬭瘯鐜鐨勫鍔辨洸绾夸娇鐢ㄥ箷濂栧姳鎬诲拰锛屽湪椤圭洰涓彲浠ヨ€冭檻浣跨敤骞曞钩鍧囧鍔?
        agent.prep_eval_rl()
        print("eval, global_step:{}".format(global_step))
        with torch.no_grad():
            done_eval = False
            observation_eval, info = env.reset(seed=AGENTPARA.RANDOM_SEED)  # 鑷繁鍐欑殑鐜鍙互鏀逛负 env.reset()   杩欎釜鍑芥暟杩斿洖鍒濆鍖栨椂鐨勮娴嬬姸鎬?
            reward_sum = 0
            step = 0
            reward_eval = 0
            for t in range(10000):
                action_eval, _, _ = agent.choose_action(observation_eval, deterministic=True)
                observation_eval, reward_eval, done_eval, _, _ = env.step(action_eval)
                reward_sum += reward_eval
                step += 1
                if done_eval:
                    break
            writer.add_scalar('reward_sum',
                              reward_sum,
                              global_step=global_step
                              )

