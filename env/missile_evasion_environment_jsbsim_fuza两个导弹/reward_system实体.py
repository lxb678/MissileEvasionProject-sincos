# 文件: reward_system.py (实体注意力修改版)

import numpy as np
import math
from typing import List, Optional

# (中文) 导入对象类，以便进行类型提示和访问状态
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile


class RewardCalculator:
    """
    <<< 实体注意力修改版 >>>
    封装了所有与奖励计算相关的逻辑。
    此版本能处理多个导弹，并能使用神经网络提供的注意力权重来动态计算融合奖励。
    """

    def __init__(self):
        # --- 所有超参数保持不变 ---
        self.W = 20
        self.U = -20
        self.SAFE_ALTITUDE_M = 3000.0
        # --- 所有超参数 ---
        # 确保所有这些 self.xxx = ... 的行都在您的代码中
        self.W = 20
        self.U = -20
        self.SAFE_ALTITUDE_M = 3000.0
        self.DANGER_ALTITUDE_M = 1500.0
        self.KV_MS = 0.2 * 340
        self.MAX_ALTITUDE_M = 12000.0
        self.OVER_ALTITUDE_PENALTY_FACTOR = 0.5
        self.ASPECT_REWARD_EFFECTIVE_RANGE = 10000.0
        self.ASPECT_REWARD_WIDTH_DEG = 45.0  # <--- 这是出错的那个属性
        self.ASPECT_PITCH_THRESHOLD_DEG = 70.0
        self.MAX_PHYSICAL_ROLL_RATE_RAD_S = np.deg2rad(240.0)
        self.OPTIMAL_SPEED_FLOOR_MACH = 0.8
        self.K_SPEED_FLOOR_PENALTY = -2.0
        self.DIVE_OUTER_BOUNDARY_M = 7000.0
        self.DIVE_INNER_BOUNDARY_M = 3000.0
        self.MAX_DIVE_SPEED_MS = 300.0
        self.OPTIMAL_DIVE_ANGLE_DEG = -30.0
        self.DIVE_ANGLE_WIDTH_DEG = 30.0
        self.COORDINATION_BONUS_FACTOR = 0.5
        self.MAX_G_LIMIT = -5.0
        self.FLARE_EFFECTIVE_DISTANCE_M = 3000.0
        self.REWARD_FLARE_IN_WINDOW = 0.5
        self.PENALTY_FLARE_OUT_WINDOW = -1.0
        self.AIRCRAFT_MAX_G = 9.0
        self.HIGH_G_SENSITIVITY = 4.0

        # <<< 新增 >>>: 用于混合策略的 alpha 参数
        # alpha 控制奖励权重在 "基于规则" 和 "基于注意力" 之间的插值。
        # alpha = 0.0: 完全基于规则 (Tau)
        # alpha = 1.0: 完全基于神经网络的注意力
        # 这个值应该在主训练循环中随着训练步数从 0 逐渐增加到 1。
        self.attention_blending_alpha = 0.0  # 默认为0，即从规则开始

        # --- 状态变量需要支持多导弹 ---
        self.history = {}

        # <<< 新增：定义关注无效目标的惩罚值 >>>
        self.PENALTY_FOR_ATTENDING_INACTIVE = -1.0  # 这是一个可以调整的超参数

    def reset(self):
        """为新回合重置状态变量。"""
        self.history.clear()

    # <<< 新增 >>>: 用于在外部更新 alpha 的接口
    def set_attention_blending_alpha(self, alpha: float):
        """
        设置注意力混合权重 alpha。
        :param alpha: 一个在 [0, 1] 范围内的浮点数。
        """
        self.attention_blending_alpha = np.clip(alpha, 0.0, 1.0)

    # --- 稀疏奖励接口 (逻辑不变) ---
    def get_sparse_reward(self, miss_distance: float, R_kill: float) -> float:
        if miss_distance > R_kill:
            return self.W
        else:
            return self.U

    # <<< 全新的 calculate_dense_reward 实现 >>>
    def calculate_dense_reward(self, aircraft: Aircraft, missiles: List[Missile],
                               remaining_flares: int, total_flares: int, action: dict, t_now: float) -> float:
        """
        计算并返回总密集奖励 (方案C: 惩罚机制)。
        - 不对权重进行重新归一化。
        - 对关注无效目标的行为施加惩罚。
        """
        flare_trigger_action = action['discrete_actions'][0]

        # --- 1. 获取神经网络输出的原始注意力权重 ---
        # 注意：这里的权重与 self.missiles 列表是一一对应的
        num_missiles = len(missiles)
        default_attn_weights = np.full(num_missiles, 1.0 / num_missiles) if num_missiles > 0 else np.array([])
        nn_attention_weights = action.get("attention_weights", default_attn_weights)

        # 健壮性检查
        if len(nn_attention_weights) != num_missiles:
            nn_attention_weights = default_attn_weights

        # --- 2. 遍历所有导弹，计算各自的奖励/惩罚 ---
        single_missile_rewards = []
        for i, missile in enumerate(missiles):
            # 检查导弹是否是“有效威胁”
            is_active = (not missile.terminated and t_now >= missile.launch_time)

            if is_active:
                # 如果是有效威胁，正常计算奖励
                reward = self._calculate_reward_for_one_missile(aircraft, missile, flare_trigger_action)
                single_missile_rewards.append(reward)
            else:
                # 如果是无效威胁，其“奖励”就是一个固定的惩罚值
                # single_missile_rewards.append(self.PENALTY_FOR_ATTENDING_INACTIVE)
                single_missile_rewards.append(0.0)

        # --- 3. 使用原始权重进行加权求和，不归一化 ---
        # 注意：这里我们只使用神经网络的权重，不再混合基于规则的权重，
        # 因为混合规则权重会破坏梯度信号的纯粹性。
        # 如果需要，alpha混合需要更复杂的逻辑，暂时简化。
        weighted_missile_reward = sum(w * r for w, r in zip(nn_attention_weights, single_missile_rewards))

        # --- 4. 计算与导弹无关的通用奖励 ---
        # 为了给通用奖励找到一个参考“主威胁”，我们仍然可以找出权重最高的活跃导弹
        primary_threat = None
        active_missiles = [m for m in missiles if not m.terminated and t_now >= m.launch_time]
        if active_missiles:
            # 找出与活跃导弹对应的权重，并找到最大值
            active_weights = [nn_attention_weights[i] for i, m in enumerate(missiles) if m in active_missiles]
            if active_weights:
                primary_threat = active_missiles[np.argmax(active_weights)]

        final_dense_reward = self._calculate_general_rewards(
            aircraft, flare_trigger_action, remaining_flares, total_flares,
            weighted_missile_reward, primary_threat
        )

        return final_dense_reward

    # # <<< 核心修改 >>>: 全新的密集奖励接口
    # def calculate_dense_reward(self, aircraft: Aircraft, missiles: List[Missile],
    #                            remaining_flares: int, total_flares: int, action: dict, t_now: float) -> float:
    #     """
    #     计算并返回总密集奖励。
    #     此版本实现了基于规则和基于注意力的混合加权奖励。
    #     """
    #     flare_trigger_action = action['discrete_actions'][0]
    #
    #     # --- 1. 筛选出所有活动的导弹 ---
    #     active_missiles = [m for m in missiles if not m.terminated and t_now >= m.launch_time]
    #
    #     if not active_missiles:
    #         return self._calculate_general_rewards(aircraft, flare_trigger_action, remaining_flares, total_flares, 0.0,
    #                                                None)
    #
    #     # --- 2. 计算与导弹相关的奖励 ---
    #
    #     # 2a. 基于物理规则 (Tau) 计算威胁权重
    #     rule_based_weights = self._calculate_rule_based_threat_weights(aircraft, active_missiles)
    #
    #     # 2b. 从 action 字典中获取神经网络的注意力权重
    #     # 提供一个默认的均等权重，以防万一没有提供
    #     num_active = len(active_missiles)
    #     default_attn_weights = np.full(num_active, 1.0 / num_active) if num_active > 0 else np.array([])
    #     nn_attention_weights = action.get("attention_weights", default_attn_weights)
    #
    #     # 确保权重维度匹配
    #     if len(nn_attention_weights) != num_active:
    #         nn_attention_weights = default_attn_weights  # 如果不匹配，使用默认值
    #
    #     # 2c. 使用 alpha 进行混合，得到最终的奖励权重
    #     final_reward_weights = (1 - self.attention_blending_alpha) * rule_based_weights + \
    #                            self.attention_blending_alpha * nn_attention_weights
    #
    #     # 归一化以确保权重和为1
    #     if np.sum(final_reward_weights) > 1e-6:
    #         final_reward_weights /= np.sum(final_reward_weights)
    #     else:  # 如果所有权重都为0, 则使用均等权重
    #         final_reward_weights = default_attn_weights
    #
    #     # 2d. 为每个活动导弹计算单项奖励
    #     single_missile_rewards = []
    #     for missile in active_missiles:
    #         reward = self._calculate_reward_for_one_missile(aircraft, missile, flare_trigger_action)
    #         single_missile_rewards.append(reward)
    #
    #     # 2e. 使用最终的混合权重进行加权求和
    #     weighted_missile_reward = sum(w * r for w, r in zip(final_reward_weights, single_missile_rewards))
    #
    #     # # <<< 核心修改：选择最大权重导弹 >>>
    #     # if len(active_missiles) > 0:
    #     #     primary_threat_idx = np.argmax(final_reward_weights)
    #     #     primary_missile = active_missiles[primary_threat_idx]
    #     #
    #     #     # 只计算主要威胁的奖励
    #     #     missile_reward = self._calculate_reward_for_one_missile(
    #     #         aircraft, primary_missile, flare_trigger_action
    #     #     )
    #     # else:
    #     #     missile_reward = 0.0
    #     #     primary_missile = None
    #
    #     # --- 3. 计算与导弹无关的通用奖励 ---
    #     # 并将加权后的导弹奖励传入
    #     final_dense_reward = self._calculate_general_rewards(
    #         aircraft, flare_trigger_action, remaining_flares, total_flares,
    #         weighted_missile_reward, active_missiles[np.argmax(final_reward_weights)] if num_active > 0 else None
    #     )
    #
    #     return final_dense_reward

    # --- 辅助函数 ---

    def _calculate_reward_for_one_missile(self, aircraft: Aircraft, missile: Missile,
                                          flare_trigger_action: float) -> float:
        """计算针对单个导弹的所有相关奖励项的总和。"""
        reward_posture = 1.0 * self._compute_missile_posture_reward_blend(aircraft, missile)
        # reward_closing_velocity = 0.5 * self._reward_for_closing_velocity_change(aircraft, missile, dt=0.2)
        reward_ata_rate = 1.0 * self._reward_for_ata_rate(aircraft, missile, dt=0.2)
        reward_flare_timing = 1.0 * self._compute_flare_timing_reward(flare_trigger_action, aircraft, missile)

        # 高G机动奖励可以与机动有效性（ATA Rate）相乘
        reward_high_g = 1.0 * self._reward_for_high_g_maneuver(aircraft)
        final_high_g_reward = 0.5 * reward_high_g * reward_ata_rate

        # # <<< 新增：调用分离加速度奖励 >>>
        # reward_separation_accel = 0.5 * self._reward_for_separation_acceleration(aircraft, missile)

        return reward_posture + reward_flare_timing + final_high_g_reward

    def _calculate_general_rewards(self, aircraft: Aircraft, flare_trigger_action: float,
                                   remaining_flares: int, total_flares: int,
                                   missile_related_reward: float, primary_threat: Optional[Missile]) -> float:
        """计算与特定导弹无关的通用奖励/惩罚。"""
        reward_altitude = 0.5 * self._compute_altitude_reward(aircraft)  #0.5
        reward_resource = 0.2 * self._compute_resource_penalty(flare_trigger_action, remaining_flares, total_flares)
        reward_roll_penalty = 0.5 * self._penalty_for_roll_rate_magnitude(aircraft) #0.5
        reward_coordinated_turn = 1.0 * self._reward_for_coordinated_turn(aircraft, 0.2)  # 降低权重
        reward_speed = 0.5 * self._reward_for_maintaining_speed(aircraft)

        return (
                missile_related_reward +
                reward_altitude +
                reward_resource +
                reward_roll_penalty +
                reward_coordinated_turn +
                reward_speed
        )

    def _calculate_tau(self, aircraft: Aircraft, missile: Missile) -> float:
        """计算单个导弹的预估命中时间 (Tau)。"""
        relative_pos_vec = aircraft.pos - missile.pos
        relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
        distance = np.linalg.norm(relative_pos_vec)
        if distance < 1e-6:
            return 0.0

        closing_velocity = -np.dot(relative_vel_vec, relative_pos_vec) / distance

        if closing_velocity > 1.0:
            return distance / closing_velocity
        else:
            # 如果没有在接近，返回一个极大的Tau值，表示威胁很低
            return float('inf')

    def _calculate_rule_based_threat_weights(self, aircraft: Aircraft, active_missiles: List[Missile]) -> np.ndarray:
        """基于物理规则（Tau）计算每个活动导弹的威胁权重。"""
        if not active_missiles:
            return np.array([])

        threat_values = []
        for missile in active_missiles:
            tau = self._calculate_tau(aircraft, missile)
            # 威胁值是 Tau 的倒数，Tau 越小，威胁值越大
            # 加上一个小的 epsilon 防止除以零，并平滑函数
            threat = 1.0 / (tau + 0.1)
            threat_values.append(threat)

        threat_values = np.array(threat_values)
        total_threat = np.sum(threat_values)

        # 归一化威胁值，得到权重
        if total_threat > 1e-6:
            return threat_values / total_threat
        else:
            # 如果所有威胁值都接近于0（例如所有导弹都在远离），则给予均等权重
            num_active = len(active_missiles)
            return np.full(num_active, 1.0 / num_active) if num_active > 0 else np.array([])

    def _get_missile_history(self, missile_id: int):
        """辅助函数，用于获取或创建特定导弹的历史状态字典。"""
        if missile_id not in self.history:
            self.history[missile_id] = {
                'prev_closing_velocity': None,
                'prev_ata_rad': None,
                'prev_separation_velocity': None,
            }
        return self.history[missile_id]

    def _evaluate_threat(self, aircraft: Aircraft, missiles: List[Missile], t_now: float):
        """
        [旧方法 - 保留但不再直接用于奖励计算]
        评估并返回当前威胁最大的活动导弹。
        """
        # (此函数的代码保持不变，但现在主要用于调试或作为备用逻辑)
        active_missiles = [m for m in missiles if not m.terminated and t_now >= m.launch_time]
        if not active_missiles:
            return None

        min_tau = float('inf')
        primary_threat = active_missiles[0]

        for missile in active_missiles:
            tau = self._calculate_tau(aircraft, missile)
            if tau < min_tau:
                min_tau = tau
                primary_threat = missile

        # 如果所有导弹Tau都是inf, 则选择距离最近的
        if min_tau == float('inf'):
            min_dist = float('inf')
            for missile in active_missiles:
                dist = np.linalg.norm(aircraft.pos - missile.pos)
                if dist < min_dist:
                    min_dist = dist
                    primary_threat = missile

        return primary_threat

    # =========================================================================
    # 以下是所有基础奖励计算函数，它们保持不变，因为它们只处理
    # (aircraft, missile) 对，或者只处理 aircraft 自身状态。
    # =========================================================================

    def _reward_for_maintaining_speed(self, aircraft: Aircraft) -> float:
        """
        如果飞机速度高于等于0.8马-赫，则给予固定的正奖励。
        如果低于0.8马赫，则根据速度差施加惩罚。
        """
        # 1. 定义速度阈值 (单位: m/s)
        SPEED_THRESHOLD_MS = 0.8 * 340.0

        # 2. 定义固定的正奖励值
        # 当速度达标时，给予这个奖励。
        REWARD_FOR_SAFE_SPEED = 0.5  # 可以调整这个值的大小

        # 3. 获取飞机当前速度
        current_speed_ms = aircraft.velocity

        # 4. 判断并计算奖励/惩罚
        if current_speed_ms >= SPEED_THRESHOLD_MS:
            # --- 情况 1: 速度达标 ---
            # 直接返回固定的正奖励
            return REWARD_FOR_SAFE_SPEED
        else:
            # --- 情况 2: 速度不达标 ---
            # a. 计算速度差 (这将是一个负数)
            speed_shortfall = current_speed_ms - SPEED_THRESHOLD_MS

            # b. 将速度差转换为惩罚值
            # 我们需要一个基准来归一化惩罚。
            # 假设当速度比阈值低150 m/s时，我们希望惩罚达到最大值-1.0。
            # 这个 MAX_SHORTFALL_FOR_PENALTY 可以根据战术需求调整。
            MAX_SHORTFALL_FOR_PENALTY = 120.0

            # c. 计算惩罚
            # speed_shortfall / MAX_SHORTFALL_FOR_PENALTY 会得到一个 (-1, 0) 范围的值
            # 例如, 如果欠速 75m/s, 惩罚就是 -75 / 150 = -0.5
            # 使用 np.clip 确保惩罚不会超过-1.0
            penalty = np.clip(speed_shortfall / MAX_SHORTFALL_FOR_PENALTY, -1.0, 0)

            return penalty

    def _reward_for_separation_acceleration(self, aircraft: Aircraft, missile: Missile) -> float:
        """
        (V3 - Multi-Missile Aware) 奖励“分离速度”的增加量，但仅在战术上合理时生效。
        此版本使用 per-missile 历史记录。
        """
        # 获取该特定导弹的历史记录
        missile_hist = self._get_missile_history(missile.id)

        # --- 1. 计算基础几何关系 (逻辑不变) ---
        aircraft_v_vec = aircraft.get_velocity_vector()
        missile_v_vec = missile.get_velocity_vector()
        los_vec_m_to_a = aircraft.pos - missile.pos
        distance = np.linalg.norm(los_vec_m_to_a)

        # --- 2. 距离门控 (逻辑不变) ---
        INNER_BOUNDARY_M = 3000.0
        OUTER_BOUNDARY_M = 4000.0
        distance_weight = 0.0
        if distance > OUTER_BOUNDARY_M:
            distance_weight = 1.0
        elif distance > INNER_BOUNDARY_M:
            distance_weight = (distance - INNER_BOUNDARY_M) / (OUTER_BOUNDARY_M - INNER_BOUNDARY_M)

        if distance_weight <= 0:
            # <<< 修改 >>> 使用 per-missile 历史记录
            missile_hist['prev_separation_velocity'] = None
            return 0.0

        # --- 3. 位置门控 (逻辑不变) ---
        if distance < 1e-6:
            # <<< 修改 >>> 使用 per-missile 历史记录
            missile_hist['prev_separation_velocity'] = None
            return 0.0

        los_unit_vec = los_vec_m_to_a / distance
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if norm_missile_v > 1e-6:
            cos_ata = np.clip(np.dot(los_unit_vec, missile_v_vec) / norm_missile_v, -1.0, 1.0)
            if cos_ata < 0:
                # <<< 修改 >>> 使用 per-missile 历史记录
                missile_hist['prev_separation_velocity'] = None
                return 0.0

        # --- 4. 核心逻辑：计算奖励 ---
        current_separation_velocity = np.dot(aircraft_v_vec, los_unit_vec)

        reward = 0.0
        # <<< 修改 >>> 读取 per-missile 历史记录
        prev_vel = missile_hist['prev_separation_velocity']
        if prev_vel is not None:
            delta_v_sep = current_separation_velocity - prev_vel
            reward_base = delta_v_sep
            SENSITIVITY = 0.1
            reward_scaled = np.tanh(reward_base * SENSITIVITY)
            reward = reward_scaled * distance_weight

        # --- 5. 更新历史状态 ---
        # <<< 修改 >>> 写入 per-missile 历史记录
        missile_hist['prev_separation_velocity'] = current_separation_velocity

        return reward

    def _reward_for_closing_velocity_change(self, aircraft: Aircraft, missile: Missile, dt: float):
        """(V3 - 多导弹版)"""
        missile_hist = self._get_missile_history(missile.id)
        relative_pos_vec = aircraft.pos - missile.pos
        relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
        distance = np.linalg.norm(relative_pos_vec)
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if distance < 1e-6 or norm_missile_v < 1e-6:
            missile_hist['prev_closing_velocity'] = None
            return 0.0
        cos_ata = np.dot(relative_pos_vec, missile_v_vec) / (distance * norm_missile_v)
        if cos_ata < 0:
            missile_hist['prev_closing_velocity'] = None
            return 1.0
        closing_velocity_current = -np.dot(relative_vel_vec, relative_pos_vec) / (distance + 1e-6)
        reward = 0.0
        prev_vel = missile_hist['prev_closing_velocity']
        if prev_vel is not None:
            delta_V_close = closing_velocity_current - prev_vel
            reward_base = -delta_V_close
            SENSITIVITY = 0.1
            reward = np.tanh(reward_base * SENSITIVITY)
        missile_hist['prev_closing_velocity'] = closing_velocity_current
        return max(0.0, reward)

    def _reward_for_ata_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """(V3 - 多导弹版)"""
        missile_hist = self._get_missile_history(missile.id)
        current_los_vec = aircraft.pos - missile.pos
        missile_v_vec = missile.get_velocity_vector()
        norm_los = np.linalg.norm(current_los_vec)
        norm_v = np.linalg.norm(missile_v_vec)
        if norm_los < 1e-6 or norm_v < 1e-6:
            missile_hist['prev_ata_rad'] = None
            return 0.0
        cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)
        if cos_ata < 0:
            missile_hist['prev_ata_rad'] = None
            return 1.0
        current_ata_rad = np.arccos(np.clip(cos_ata, -1.0, 1.0))
        prev_ata = missile_hist['prev_ata_rad']
        if prev_ata is None:
            missile_hist['prev_ata_rad'] = current_ata_rad
            return 0.0
        delta_ata_rad = abs(current_ata_rad - prev_ata)
        ata_rate_rad_s = delta_ata_rad / dt if dt > 1e-6 else 0.0
        SENSITIVITY = 10.0
        normalized_reward = math.tanh(ata_rate_rad_s * SENSITIVITY)
        missile_hist['prev_ata_rad'] = current_ata_rad
        return normalized_reward

    def _compute_missile_posture_reward_blend(self, aircraft: Aircraft, missile: Missile):
        aircraft_v_vec = aircraft.get_velocity_vector()
        los_vec_m_to_a = aircraft.pos - missile.pos
        distance = np.linalg.norm(los_vec_m_to_a)
        missile_v_vec = missile.get_velocity_vector()
        norm_los = np.linalg.norm(los_vec_m_to_a)
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if norm_los > 1e-6 and norm_missile_v > 1e-6:
            cos_ata = np.clip(np.dot(los_vec_m_to_a, missile_v_vec) / (norm_los * norm_missile_v), -1.0, 1.0)
            if cos_ata < 0:
                return 1.0
        norm_v = np.linalg.norm(aircraft_v_vec)
        if norm_v < 1e-6 or norm_los < 1e-6:
            return 0.0
        cos_angle = np.clip(np.dot(aircraft_v_vec, los_vec_m_to_a) / (norm_v * norm_los), -1.0, 1.0)
        reward_tail_chase = cos_angle
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.rad2deg(angle_rad)
        angle_error_deg = abs(angle_deg - 90)
        reward_three_nine = math.exp(-(angle_error_deg ** 2) / (2 * self.ASPECT_REWARD_WIDTH_DEG ** 2))

        # <<< 核心修改：增加俯仰角限制 >>>
        # 1. 获取飞机的俯仰角 (pitch)，并转换为度
        # 通过 attitude_rad 元组的第一个元素获取俯仰角
        aircraft_pitch_deg = np.rad2deg(aircraft.attitude_rad[0])

        # 2. 如果俯仰角小于-70度（即机头朝下超过70度），则取消三九线奖励
        if aircraft_pitch_deg < -self.ASPECT_PITCH_THRESHOLD_DEG:  # 使用超参数 -70.0
            reward_three_nine = 0.0
        # <<< 修改结束 >>>

        # print("aircraft_pitch_deg", aircraft_pitch_deg)
        # print("reward_three_nine", reward_three_nine)

        if distance <= 3000.0:
            return reward_three_nine
        elif distance >= 4000.0:
            return reward_tail_chase
        else:
            alpha = (distance - 3000.0) / 1000.0
            return (1 - alpha) * reward_three_nine + alpha * reward_tail_chase

    def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
        try:
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            return 0.0
        if nz <= 0: return 0.0
        if self.AIRCRAFT_MAX_G <= 1e-6: return 0.0
        normalized_g = np.clip(nz / self.AIRCRAFT_MAX_G, 0.0, 1.0)
        return normalized_g

    def _compute_flare_timing_reward(self, flare_trigger_action: float, aircraft: Aircraft, missile: Missile) -> float:
        if flare_trigger_action < 0.5: return 0.0
        distance = np.linalg.norm(aircraft.pos - missile.pos)
        if distance < self.FLARE_EFFECTIVE_DISTANCE_M:
            return self.REWARD_FLARE_IN_WINDOW
        else:
            return self.PENALTY_FLARE_OUT_WINDOW

    def _compute_altitude_reward(self, aircraft: Aircraft):
        altitude_m = aircraft.pos[1]
        v_vertical_ms = aircraft.get_velocity_vector()[1]
        Pv = 0.0
        if altitude_m <= self.SAFE_ALTITUDE_M and v_vertical_ms < 0:
            descent_speed = abs(v_vertical_ms)
            penalty_factor = (descent_speed / self.KV_MS) * ((self.SAFE_ALTITUDE_M - altitude_m) / self.SAFE_ALTITUDE_M)
            Pv = -np.clip(penalty_factor, 0.0, 1.0)
        PH = 0.0
        if altitude_m <= self.DANGER_ALTITUDE_M:
            PH = np.clip(altitude_m / self.DANGER_ALTITUDE_M, 0.0, 1.0) - 1.0
        P_over = 0.0
        if altitude_m > self.MAX_ALTITUDE_M:
            P_over = -((altitude_m - self.MAX_ALTITUDE_M) / 1000.0) * self.OVER_ALTITUDE_PENALTY_FACTOR
        return Pv + PH + P_over

    def _compute_resource_penalty(self, release_flare_action, remaining_flares, total_flares):
        if release_flare_action > 0.5:
            alphaR = 2.0
            k1 = 3.0
            fraction_remaining = remaining_flares / total_flares
            penalty = -alphaR * (1 + k1 * (1 - fraction_remaining))
            return penalty
        return 0.0

    def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
        p_real_rad_s = aircraft.roll_rate_rad_s
        if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6: return 0.0
        normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
        return (normalized_roll_rate) * -1.0

    def _reward_for_coordinated_turn(self, aircraft: Aircraft, dt: float):
        try:
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            return 0.0
        positive_g = nz
        g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
        g_factor = np.clip(g_factor, 0, 1.0)
        reward = -1.0 * g_factor
        return reward