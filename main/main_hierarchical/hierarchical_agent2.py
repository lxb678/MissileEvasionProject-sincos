# 文件: hierarchical_agent.py

import numpy as np
from enum import Enum, auto
from typing import Tuple

# --- 导入您的PPO模型类 ---
# (请确保这里的路径和文件名是正确的)
from Interference_code.PPO_model.旧文件.Hybrid_PPO_jsbsim_launch import PPO_continuous as AttackPPO
from Interference_code.PPO_model.旧文件.Hybrid_PPO_jsbsim import PPO_continuous as EvadePPO

# --- 导入火控逻辑 ---
from Interference_code.main.main_attack.fire_control_rules_all import can_launch_missile_with_pk, fire_control_timers, \
    FIRE_COOLDOWN_S


# [IMPROVEMENT] 使用枚举来定义智能体的工作模式，更安全、更清晰
class AgentMode(Enum):
    """定义智能体的工作模式"""
    ATTACK = auto()
    EVADE = auto()


class HierarchicalAgent:
    """
    一个分层决策智能体，通过一个基于规则的高层控制器，
    在“攻击”和“规避”两个专家模型之间进行切换。
    """

    def __init__(self, attack_model_dir: str, evade_model_dir: str, load_models: bool = True):
        """
        初始化分层智能体。

        Args:
            attack_model_dir (str): 包含攻击模型的文件夹路径。
            evade_model_dir (str): 包含规避模型的文件夹路径。
            load_models (bool): 是否加载预训练模型。
        """
        print("--- 正在初始化分层决策智能体 ---")

        # 1. 加载专家模型
        # (重要) 这里的 PPO 类初始化参数需要与您训练时使用的完全一致
        print(f"  - 正在从 '{attack_model_dir}' 加载攻击专家模型...")
        self.attack_agent = AttackPPO(load_able=load_models, model_dir_path=attack_model_dir)
        self.attack_agent.prep_eval_rl()

        print(f"  - 正在从 '{evade_model_dir}' 加载规避导弹专家模型...")
        self.evade_agent = EvadePPO(load_able=load_models, model_dir_path=evade_model_dir)
        self.evade_agent.prep_eval_rl()

        # 2. 定义高层决策逻辑的参数
        self.missile_threat_threshold = 0.98  # 威胁距离观测值的阈值，小于此值则激活规避

        # [IMPROVEMENT] 使用枚举类型来跟踪当前模式，初始化为None
        self.current_mode: AgentMode = None

        print("--- 分层智能体初始化完成 ---")

    def reset(self):
        """重置智能体状态和火控计时器，在每个episode开始时调用。"""
        # 重置全局的计时器
        fire_control_timers['red'] = -FIRE_COOLDOWN_S
        fire_control_timers['blue'] = -FIRE_COOLDOWN_S
        self.current_mode = None
        print("--- 智能体状态已重置 ---")

    def _decide_mode(self, combat_obs: np.ndarray) -> AgentMode:
        """
        [高层决策逻辑] 根据观测信息决定当前应该处于攻击模式还是规避模式。
        """
        # o_threat_dist: 1.0 (无威胁) -> 0.0 (非常近)
        threat_dist_obs = combat_obs[9]  # 提取威胁距离 (第10个元素)

        if threat_dist_obs < self.missile_threat_threshold:
            return AgentMode.EVADE
        else:
            return AgentMode.ATTACK

    def _execute_evade_policy(self, evade_obs: np.ndarray, deterministic: bool) -> Tuple[np.ndarray, float]:
        """
        [低层策略执行] 调用规避模型生成动作。
        """
        # 规避Agent生成5维动作 [油门, 升降舵, 副翼, 方向舵, 诱饵弹]
        action_5d, _, _ = self.evade_agent.choose_action(evade_obs, deterministic=deterministic)

        # 在规避时，不主动发射导弹
        fire_missile_cmd = 0.0

        return action_5d, fire_missile_cmd

    def _execute_attack_policy(self, combat_obs: np.ndarray, env, deterministic: bool) -> Tuple[np.ndarray, float]:
        """
        [低层策略执行] 调用攻击模型生成动作，并结合火控规则。
        """
        # 攻击Agent生成5维机动动作
        action_5d, _, _ = self.attack_agent.choose_action(combat_obs, deterministic=deterministic)

        # 结合专家火控规则决定是否发射导弹
        if can_launch_missile_with_pk(launcher_ac=env.red_aircraft,
                                      target_ac=env.blue_aircraft,
                                      current_sim_time=env.t_now):
            fire_missile_cmd = 1.0
        else:
            fire_missile_cmd = 0.0

        return action_5d, fire_missile_cmd

    def get_action(self, combat_obs: np.ndarray, evade_obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """
        顶层决策函数，整合了高层决策和低层策略执行。

        Args:
            combat_obs (np.ndarray): 用于攻击Agent的14维观测向量。
            evade_obs (np.ndarray): 用于规避Agent的9维观测向量。
            env: 环境对象，用于调用火控逻辑。
            deterministic (bool): 是否使用确定性动作。

        Returns:
            np.ndarray: 一个长度为6的最终动作向量，用于输入环境。
        """
        # 1. 高层决策：决定当前模式
        new_mode = self._decide_mode(combat_obs)

        # 监控模式切换，并打印日志
        if new_mode != self.current_mode:
            if new_mode == AgentMode.EVADE:
                threat_dist_obs = combat_obs[9]
                print(f"\n--- [决策切换]: 导弹威胁出现！激活【规避模式】 (威胁距离观测值: {threat_dist_obs:.2f}) ---\n")
            else:
                print(f"\n--- [决策切换]: 威胁解除，切换回【攻击模式】 ---\n")
            self.current_mode = new_mode

        # 2. 低层执行：根据当前模式调用相应的专家策略
        if self.current_mode == AgentMode.EVADE:
            action_5d, fire_missile_cmd = self._execute_evade_policy(evade_obs, deterministic)
        else:  # AgentMode.ATTACK
            action_5d, fire_missile_cmd = self._execute_attack_policy(combat_obs, env, deterministic)

        # 3. 动作整合：将专家策略输出的5维动作和火控指令拼接成环境所需的6维动作
        # 动作顺序: [油门, 升降舵, 副翼, 方向舵, 诱饵弹, 发射导弹]
        final_action = np.array([
            action_5d[0],  # 油门
            action_5d[1],  # 升降舵
            action_5d[2],  # 副翼
            action_5d[3],  # 方向舵
            action_5d[4],  # 投放诱饵
            fire_missile_cmd  # 发射导弹
        ])

        return final_action