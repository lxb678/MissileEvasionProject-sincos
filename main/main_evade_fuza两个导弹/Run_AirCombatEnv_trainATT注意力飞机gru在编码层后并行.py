import random
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import os
import math
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

# ------------------- 导入模型和配置 -------------------
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_ATTMLP注意力GRU注意力后yakebi修正优势归一化2 import *
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.ConfigAttn import *
from Interference_code.env.missile_evasion_environment_jsbsim_fuza两个导弹.Vec_missile_evasion_environment_jsbsim实体2 import *

# ------------------- 全局配置 -------------------
LOAD_ABLE = False
USE_RNN_MODEL = True
TACVIEW_ENABLED_DURING_TRAINING = False

# <<< 并行化配置 >>>
UPDATE_CYCLE = 10  # 每收集 10 个 Episode 更新一次网络
NUM_WORKERS = 10  # 并行进程数
ROUNDS_PER_UPDATE = int(math.ceil(UPDATE_CYCLE / NUM_WORKERS))


def set_seed(env, seed=AGENTPARA.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if env is not None and hasattr(env, 'action_space'):
        env.action_space.seed(seed)


def pack_action_into_dict(flat_action_np: np.ndarray, attn_weights: Optional[np.ndarray] = None) -> dict:
    continuous_part = flat_action_np[:CONTINUOUS_DIM]
    discrete_part = flat_action_np[CONTINUOUS_DIM:].astype(int)
    return {
        "continuous_actions": continuous_part,
        "discrete_actions": discrete_part,
        "attention_weights": attn_weights
    }


def _get_pre_hidden_numpy_safe(local_agent):
    """ 获取当前 RNN 隐状态并转为 numpy 以便 IPC 传输 """
    h_act = getattr(local_agent, 'temp_actor_h', None)
    h_cri = getattr(local_agent, 'temp_critic_h', None)

    if h_act is None or h_cri is None:
        dim_a = local_agent.Actor.rnn_hidden_dim
        dim_c = local_agent.Critic.rnn_hidden_dim
        # 注意：这里需要与 Actor/Critic 内部定义的层数匹配，通常是 (1, 1, Dim)
        h_act_np = np.zeros((1, 1, dim_a), dtype=np.float32)
        h_cri_np = np.zeros((1, 1, dim_c), dtype=np.float32)
    else:
        h_act_np = h_act.detach().cpu().numpy()
        h_cri_np = h_cri.detach().cpu().numpy()

    return h_act_np, h_cri_np


def store_experience_to_buffer(agent, s, a, p, v, r, d, attn, h_act_np, h_cri_np):
    """ 将 Worker 传回的数据存入 Master 的 Buffer """
    h_act = torch.from_numpy(h_act_np).to(ACTOR_PARA.device)
    h_cri = torch.from_numpy(h_cri_np).to(CRITIC_PARA.device)

    # 这里的 agent.buffer.store_transition 会处理 numpy 到 tensor 的最终转换
    agent.buffer.store_transition(
        state=s, value=v, action=a, probs=p, reward=r, done=d,
        actor_hidden=h_act,
        critic_hidden=h_cri,
        attn_weights=attn
    )


# =================================================================
#                   Worker Process (子进程逻辑)
# =================================================================
def worker_process(rank, pipe):
    # 子进程强制使用 CPU，避免显存冲突和非法的 CUDA 调用
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    ACTOR_PARA.device = torch.device("cpu")
    CRITIC_PARA.device = torch.device("cpu")
    ACTOR_PARA.tpdv = {'dtype': torch.float32, 'device': torch.device("cpu")}
    CRITIC_PARA.tpdv = {'dtype': torch.float32, 'device': torch.device("cpu")}

    try:
        env = AirCombatEnv(tacview_enabled=False)
        local_agent = PPO_continuous(load_able=False, model_dir_path=None, use_rnn=USE_RNN_MODEL)

        while True:
            cmd, packet = pipe.recv()
            if cmd == 'EXIT':
                env.close()
                break

            elif cmd == 'RUN_EPISODE':
                weights_dict, episode_seed = packet
                local_agent.Actor.load_state_dict(weights_dict['actor'])
                local_agent.Critic.load_state_dict(weights_dict['critic'])
                local_agent.Actor.eval()
                local_agent.Critic.eval()

                set_seed(env, episode_seed)
                observation, info = env.reset(seed=episode_seed)

                if USE_RNN_MODEL:
                    local_agent.reset_rnn_state()

                transitions = []
                episode_reward = 0
                step_count = 0
                v_last_step = 0.0  # 用于存储截断时的引导价值

                for t in range(10000):
                    state_to_store = observation
                    with torch.no_grad():
                        # 获取执行动作前的隐藏状态（存入 Buffer 需要）
                        h_act_np, h_cri_np = _get_pre_hidden_numpy_safe(local_agent)

                        env_action_flat, action_to_store, prob, value, attn_weights = local_agent.choose_action(
                            state_to_store)

                    action_dict = pack_action_into_dict(env_action_flat, attn_weights)
                    observation, reward, terminated, truncated, info = env.step(action_dict)

                    done = terminated or truncated
                    episode_reward += reward

                    # 【修正 1】: Reward 保持原样，不要在 transitions 里加 V(s_next)
                    transitions.append([
                        state_to_store, action_to_store, prob, value, reward, done, attn_weights, h_act_np, h_cri_np
                    ])

                    step_count += 1
                    if done:
                        # 【修正 2】: 如果是 Truncated，计算 Bootstrap 价值并传回给 Master
                        if truncated:
                            with torch.no_grad():
                                last_obs_t = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                                # 注意：RNN 模型计算 V(s_next) 必须使用当前最新的隐状态
                                v_next, _ = local_agent.Critic(last_obs_t, local_agent.critic_rnn_state)
                                v_last_step = v_next.item()

                        success = info.get('success', False)
                        break

                pipe.send({
                    'transitions': transitions,
                    'episode_reward': episode_reward,
                    'success': success,
                    'steps': step_count,
                    'v_last_step': v_last_step  # 传回引导值
                })

    except Exception as e:
        import traceback
        pipe.send({'error': str(e), 'trace': traceback.format_exc()})


# =================================================================
#                   Master Process (主训练逻辑)
# =================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    set_seed(None, AGENTPARA.RANDOM_SEED)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f'../../log/log_evade_fuza两个导弹/PPO_Parallel_ATT_PostGRU_{timestamp}')

    # 主进程 Agent (在 GPU 上更新)
    agent = PPO_continuous(load_able=LOAD_ABLE, model_dir_path=None, use_rnn=USE_RNN_MODEL)

    workers, pipes = [], []
    for i in range(NUM_WORKERS):
        p_conn, c_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(i, c_conn))
        p.start()
        workers.append(p)
        pipes.append(p_conn)

    global_step = 0
    total_episodes = 0
    success_num = 0
    eval_env = AirCombatEnv(tacview_enabled=TACVIEW_ENABLED_DURING_TRAINING)

    try:
        # 计算总更新次数
        num_updates = 20000 // UPDATE_CYCLE
        for update_idx in range(num_updates):
            # 将最新的模型权重拷贝到 CPU 准备分发
            full_weights_cpu = {
                'actor': {k: v.cpu() for k, v in agent.Actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in agent.Critic.state_dict().items()}
            }

            collected_this_cycle = 0
            last_bootstrap_value = 0.0  # 记录最后一条轨迹的引导值

            # A. 数据收集阶段
            for round_idx in range(ROUNDS_PER_UPDATE):
                remaining = UPDATE_CYCLE - collected_this_cycle
                if remaining <= 0: break
                active = min(NUM_WORKERS, remaining)

                for i in range(active):
                    abs_id = total_episodes + i
                    pipe_seed = AGENTPARA.RANDOM_SEED + abs_id
                    pipes[i].send(('RUN_EPISODE', (full_weights_cpu, pipe_seed)))

                for i in range(active):
                    res = pipes[i].recv()
                    if 'error' in res:
                        print(f"Worker Error:\n{res['trace']}")
                        raise RuntimeError(res['error'])

                    # 将数据存入 Buffer
                    for trans in res['transitions']:
                        s, a, p, v, r, d, attn, h_act_np, h_cri_np = trans
                        store_experience_to_buffer(agent, s, a, p, v, r, d, attn, h_act_np, h_cri_np)
                        global_step += 1

                    total_episodes += 1
                    collected_this_cycle += 1

                    # 记录这一局的引导值，作为 learn() 的参数
                    last_bootstrap_value = res['v_last_step']

                    print(
                        f"Ep {total_episodes} | Rew: {res['episode_reward']:.2f} | Steps: {res['steps']} | Success: {res['success']}")
                    writer.add_scalar('Episode/Reward', res['episode_reward'], global_step)
                    if res['success']: success_num += 1

                    # 成功率统计
                    if total_episodes % 100 == 0:
                        sr = success_num / 100.0
                        writer.add_scalar('Episode/Success_Rate_per_100_ep', sr, total_episodes)
                        if sr >= 0.90: agent.save(prefix=f"success_{int(sr * 100)}_ep{total_episodes}")
                        success_num = 0

            # B. 学习阶段
            print(f"\n--- Episode {total_episodes}: 开始周期性训练 ---")
            agent.prep_training_rl()

            # 【核心修正 3】: 使用 Worker 传回的引导值进行学习
            # 注意：由于 PPO 这里的 learn 是计算整个 Buffer 的 GAE，
            # 理想情况下应该每条轨迹独立计算，但按你目前的 learn 结构，
            # 传入最后一条轨迹的引导值是保证超参数一致性的权宜之计（Reward 已经干净了）。
            train_info = agent.learn(next_visual_value=last_bootstrap_value)

            if train_info:
                for k, v in train_info.items():
                    writer.add_scalar(f"Train/{k}", v, global_step)

            # C. 评估阶段 (保持串行代码中的逻辑)
            agent.prep_eval_rl()
            TEST_SEED_OFFSET = 100000
            e_seed = AGENTPARA.RANDOM_SEED + TEST_SEED_OFFSET + total_episodes
            set_seed(eval_env, e_seed)
            e_obs, _ = eval_env.reset(seed=e_seed)
            if USE_RNN_MODEL: agent.reset_rnn_state()
            e_rew_sum = 0
            for _ in range(10000):
                with torch.no_grad():
                    e_a, _, _, _, e_at = agent.choose_action(e_obs, deterministic=True)
                e_obs, e_r, e_d, e_t, _ = eval_env.step(pack_action_into_dict(e_a, e_at))
                e_rew_sum += e_r
                if e_d or e_t: break
            writer.add_scalar('Eval/Reward_Sum', e_rew_sum, global_step)
            print(f">>> 评估结束 | 种子: {e_seed} | 奖励: {e_rew_sum:.2f}\n")

    except KeyboardInterrupt:
        print("用户中断，正在安全退出...")
    finally:
        # 关闭子进程
        for p_conn in pipes:
            try:
                p_conn.send(('EXIT', None))
            except:
                pass
        for p in workers:
            p.join(timeout=1)
            if p.is_alive(): p.terminate()
        writer.close()
        eval_env.close()