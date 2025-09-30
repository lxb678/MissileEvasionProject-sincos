# 文件: hierarchical_agent.py

import numpy as np

# --- 导入您的PPO模型类 ---
# (请确保这里的路径和文件名是正确的)
from PPO_model.Hybrid_PPO_jsbsim_launch import PPO_continuous as AttackPPO
# 假设规避模型的类名为 EvadePPO，请根据实际情况修改
from PPO_model.Hybrid_PPO_jsbsim import PPO_continuous as EvadePPO

from Interference_code.fire_control_rules_all import can_launch_missile_with_pk, fire_control_timers, FIRE_COOLDOWN_S


class HierarchicalAgent:
    """
    一个分层决策智能体，整合了攻击和规避导弹两个专家模型。
    """

    def __init__(self, attack_model_dir: str, evade_model_dir: str, load_models: bool = True):
        """
       初始化分层智能体。

       Args:
           attack_model_dir (str): 包含攻击模型的【文件夹】路径。
           evade_model_dir (str): 包含规避模型的【文件夹】路径。
           load_models (bool): 是否加载预训练模型。
       """
        print("--- 正在初始化分层决策智能体 ---")

        # 1. 加载专家模型
        # (重要) 这里的 PPO 类初始化参数需要与您训练时使用的完全一致
        # 例如，如果需要传入 state_dim, action_dim 等，请在这里加上
        print("  - 正在加载攻击专家模型...")
        self.attack_agent = AttackPPO(load_able=load_models, model_dir_path=attack_model_dir)
        self.attack_agent.prep_eval_rl()  # 设置为评估模式

        print("  - 正在加载规避导弹专家模型...")
        self.evade_agent = EvadePPO(load_able=load_models, model_dir_path=evade_model_dir)
        self.evade_agent.prep_eval_rl()  # 设置为评估模式

        # 2. 定义高层决策逻辑的参数
        self.missile_threat_threshold = 0.98  # 威胁距离观测值的阈值，小于此值则激活规避

        # --- <<< 核心修改 1: 新增一个状态变量来记录上一次的模式 >>> ---
        self.last_mode = None  # 'ATTACK' or 'EVADE'

        print("--- 分层智能体初始化完成 ---")

    def reset(self):
        # 重置全局的计时器
        fire_control_timers['red'] = -FIRE_COOLDOWN_S
        fire_control_timers['blue'] = -FIRE_COOLDOWN_S
        self.last_mode = None

    def get_action(self, combat_obs: np.ndarray, evade_obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """
        顶层决策函数 (有限状态机)。
        它会调用专家模型生成5维动作，然后手动添加“发射导弹”维度，以匹配环境的6维输入。

        :param combat_obs: 用于攻击Agent的14维观测向量。
        :param evade_obs: 用于规避Agent的9维观测向量。
        :param env: 环境对象，用于调用火控逻辑。
        :param deterministic: 是否使用确定性动作。
        :return: 一个【长度为6】的动作向量。
        """
        # --- 状态机决策逻辑 ---
        # 规则：只要有导弹威胁，就无条件激活规避Agent。

        # 从 combat_obs 中提取威胁距离 (第10个元素, 索引为9)
        # o_threat_dist: 1.0 (无威胁) -> 0.0 (非常近)
        threat_dist_obs = combat_obs[9]

        if threat_dist_obs < self.missile_threat_threshold:
            # --- 状态: 导弹规避 (MISSILE_EVADE) ---
            # --- 状态: 导弹规避 ---
            # 规避Agent生成5维动作 [油门, 升降舵, 副翼, 方向舵, 诱饵弹]
            # print("  [决策]: 导弹威胁！激活规避Agent") # (可选) 调试时取消注释
            current_mode = 'EVADE'
            # --- <<< 核心修改 1: 仅在状态切换时打印 >>> ---
            if self.last_mode != current_mode:
                print(f"\n--- [决策切换]: 导弹威胁出现！激活【规避Agent】 (威胁距离观测值: {threat_dist_obs:.2f}) ---\n")
                self.last_mode = current_mode

            # 使用规避Agent和其专属的观测数据
            action_5d, _, _ = self.evade_agent.choose_action(evade_obs, deterministic=deterministic)

            # 在规避时，我们通常不发射导弹
            fire_missile_cmd = 0.0

        else:
            # --- 状态: 攻击/占位 (WVR_ENGAGE) ---
            # 攻击Agent生成5维动作
            # print("  [决策]: 无威胁，执行攻击任务") # (可选) 调试时取消注释
            current_mode = 'ATTACK'
            # --- <<< 核心修改 1: 仅在状态切换时打印 >>> ---
            if self.last_mode != current_mode:
                print(f"\n--- [决策切换]: 威胁解除，切换回【攻击Agent】 ---\n")
                self.last_mode = current_mode

            # 使用攻击Agent和其专属的观测数据
            action_5d, _, _ = self.attack_agent.choose_action(combat_obs, deterministic=deterministic)
            # --- 在攻击模式下，使用您的专家火控规则来决定是否发射导弹 ---

            # 调用通用函数
            if can_launch_missile_with_pk(launcher_ac=env.red_aircraft,
                                          target_ac=env.blue_aircraft,
                                          current_sim_time=env.t_now):
                fire_missile_cmd = 1.0
            else:
                fire_missile_cmd = 0.0

        # --- 手动拼接成6维动作 ---
        # 动作的前4维是连续机动
        continuous_actions = action_5d[:4]
        # 第5维是诱饵弹
        flare_action = action_5d[4]

        # 按照环境期望的顺序 [油门, 升降舵, 副翼, 方向舵, 诱饵弹, 发射导弹] 重新组合
        action_6d = np.array([
            continuous_actions[0],  # 油门
            continuous_actions[1],  # 升降舵
            continuous_actions[2],  # 副翼
            continuous_actions[3],  # 方向舵
            flare_action,  # 投放诱饵
            fire_missile_cmd  # 发射导弹
        ])

        return action_6d