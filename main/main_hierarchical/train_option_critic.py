# 文件: train_option_critic.py

import torch

# --- 导入您的环境和新的HRL智能体 ---
from Interference_code.env.Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim_pointmass import \
    AirCombatEnv
from option_critic_agent import OptionCriticAgent  # <<< 导入我们新的智能体
from Interference_code.main.main_attack.blue_ai_rules import get_blue_ai_action

# --- 训练设置 ---
TOTAL_TRAIN_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 5000
LEARN_INTERVAL_STEPS = 2048  # 每收集这么多步数据，就执行一次学习
INITIALIZE_FROM_EXPERTS = True  # <<< 关键开关：是否从专家模型加载权重

# 模型路径 (仅当 INITIALIZE_FROM_EXPERTS=True 时使用)
ATTACK_MODEL_FOLDER = "../../test/test_hierarchical_model/attack_model"
EVADE_MODEL_FOLDER = "../../test/test_hierarchical_model/evade_model"

if __name__ == '__main__':
    # 1. 初始化主空战环境
    env = AirCombatEnv(tacview_enabled=False)  # 训练时通常关闭Tacview以提高速度

    # 2. 初始化Option-Critic智能体 (红方)
    agent = OptionCriticAgent(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        load_from_experts=INITIALIZE_FROM_EXPERTS,
        attack_model_dir=ATTACK_MODEL_FOLDER,
        evade_model_dir=EVADE_MODEL_FOLDER
    )

    total_steps = 0
    print(f"\n--- 开始Option-Critic智能体训练 (设备: {agent.device}) ---")

    for i_episode in range(TOTAL_TRAIN_EPISODES):
        # 重置环境和智能体内部状态
        obs_dict = env.reset()
        state = obs_dict['red_agent']

        current_option = None  # 回合开始时没有预设选项
        episode_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            # --- 1. 红方HRL智能体决策 ---
            env_action, action_to_store, log_prob, current_option = agent.get_action(state, current_option)

            # --- 2. 蓝方规则AI决策 ---
            blue_obs = obs_dict['blue_agent']
            blue_action = get_blue_ai_action(blue_obs, env)

            # --- 3. 环境交互 ---
            actions = {'red_agent': env_action, 'blue_agent': blue_action}
            next_obs_dict, rewards, dones, info = env.step(actions)

            # --- 4. 存储经验 ---
            next_state = next_obs_dict['red_agent']
            reward = rewards['red_agent']
            done = dones['__all__']

            agent.store_experience(state, current_option, action_to_store, log_prob, reward, done, next_state)

            # --- 5. 状态更新与学习 ---
            state = next_state
            episode_reward += reward
            total_steps += 1

            # 检查是否达到学习条件
            if total_steps % LEARN_INTERVAL_STEPS == 0 and len(agent.buffer) > 0:
                print(f"\n--- 达到 {LEARN_INTERVAL_STEPS} 步，开始学习... ---")
                train_stats = agent.learn()
                if train_stats:
                    print(
                        f"学习完成. Actor Loss: {train_stats['actor_loss']:.4f}, Critic Loss: {train_stats['critic_loss']:.4f}")

            if done:
                break

        print(f"回合 {i_episode + 1}/{TOTAL_TRAIN_EPISODES} | 总步数: {total_steps} | 回合奖励: {episode_reward:.2f}")

        # (可选) 定期保存模型
        if (i_episode + 1) % 100 == 0:
            # agent.save(prefix=f"episode_{i_episode+1}") # 需要在 agent 类中实现 save 方法
            print(f"--- Episode {i_episode + 1}, 模型已保存 ---")