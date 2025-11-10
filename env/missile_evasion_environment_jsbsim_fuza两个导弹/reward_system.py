# 文件: reward_system.py (已适配多导弹场景)

import numpy as np
import math
from typing import List

# (中文) 导入对象类，以便进行类型提示和访问状态
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile


class RewardCalculator:
    """
    <<< 多导弹更改 >>>
    封装了所有与奖励计算相关的逻辑。
    此版本能处理多个导弹，通过威胁评估来计算奖励。
    """

    def __init__(self):
        # --- 所有超参数保持不变 ---
        self.W = 20
        self.U = -20
        self.SAFE_ALTITUDE_M = 3000.0
        self.DANGER_ALTITUDE_M = 1500.0
        self.KV_MS = 0.2 * 340
        self.MAX_ALTITUDE_M = 12000.0
        self.OVER_ALTITUDE_PENALTY_FACTOR = 0.5
        self.ASPECT_REWARD_EFFECTIVE_RANGE = 10000.0
        self.ASPECT_REWARD_WIDTH_DEG = 45.0
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

        # --- <<< 多导弹更改 >>> 状态变量需要支持多导弹 ---
        # 我们使用字典来存储每个导弹的历史状态，以导弹的唯一ID作为键
        self.history = {}

    def reset(self):
        """为新回合重置状态变量。"""
        # <<< 多导弹更改 >>> 直接清空历史字典
        self.history.clear()

    # --- 稀疏奖励接口 (逻辑不变) ---
    def get_sparse_reward(self, miss_distance: float, R_kill: float) -> float:
        if miss_distance > R_kill:
            return self.W
        else:
            return self.U

    # --- <<< 多导弹更改 >>> 密集奖励接口 ---
    def calculate_dense_reward(self, aircraft: Aircraft, missiles: List[Missile],
                               remaining_flares: int, total_flares: int, action: dict, t_now: float) -> float:
        """
        计算并返回总密集奖励。
        1. 评估哪个导弹是当前最大威胁。
        2. 基于最大威胁计算所有相关奖励。
        """
        flare_trigger_action = action['discrete_actions'][0]

        # --- 1. <<< 核心修改 >>> 威胁评估 ---
        primary_threat = self._evaluate_threat(aircraft, missiles, t_now)

        # # 如果没有活动的导弹（例如，都还没发射或都已被规避），则只给予生存奖励
        if primary_threat is None:
            return 0.0  # 生存奖励

        # --- 2. 基于最大威胁导弹计算所有奖励组件 ---
        #    所有原来需要 `missile` 参数的函数，现在都传入 `primary_threat`
        reward_posture = 1.0 * self._compute_missile_posture_reward_blend(aircraft, primary_threat)
        reward_altitude = 0.5 * self._compute_altitude_reward(aircraft)
        reward_resource = 0.2 * self._compute_resource_penalty(flare_trigger_action, remaining_flares, total_flares)
        reward_roll_penalty = 0.5 * self._penalty_for_roll_rate_magnitude(aircraft)
        reward_coordinated_turn = self._reward_for_coordinated_turn(aircraft, 0.2)

        # <<< 多导弹更改 >>> 确保历史状态是针对特定导弹的
        reward_closing_velocity = 0.5 * self._reward_for_closing_velocity_change(aircraft, primary_threat, dt=0.2)
        reward_ata_rate = 1.0 * self._reward_for_ata_rate(aircraft, primary_threat, dt=0.2)

        # 干扰弹时机奖励现在也应基于最大威胁
        reward_flare_timing = 0.5 * self._compute_flare_timing_reward(flare_trigger_action, aircraft, primary_threat)

        # 高G机动奖励本身与导弹无关，但可以与机动有效性（ATA Rate）相乘
        reward_high_g = 1.0 * self._reward_for_high_g_maneuver(aircraft)
        final_high_g_reward = 0.5 * reward_high_g * reward_ata_rate

        # 3. 加权求和 (逻辑不变)
        final_dense_reward = (
                reward_posture +
                reward_altitude +
                reward_resource +
                reward_roll_penalty +
                reward_coordinated_turn +
                reward_closing_velocity +
                reward_flare_timing +
                final_high_g_reward
        )

        # (可选) 调试打印，显示当前的主要威胁
        # print(f"Primary Threat: Missile {primary_threat.id + 1}")

        return final_dense_reward

    def _get_missile_history(self, missile_id: int):
        """辅助函数，用于获取或创建特定导弹的历史状态字典。"""
        if missile_id not in self.history:
            self.history[missile_id] = {
                'prev_closing_velocity': None,
                'prev_ata_rad': None,
                # ... 其他需要为每个导弹单独记录的历史状态 ...
            }
        return self.history[missile_id]

    def _evaluate_threat(self, aircraft: Aircraft, missiles: List[Missile], t_now: float):
        """
        评估并返回当前威胁最大的活动导弹。
        评估标准：最小的命中时间 (Tau)。如果无法计算Tau，则使用最小距离。
        """
        active_missiles = [m for m in missiles if not m.terminated and t_now >= m.launch_time]
        if not active_missiles:
            return None

        min_tau = float('inf')
        primary_threat = active_missiles[0]  # 默认是第一个活动的导弹

        for missile in active_missiles:
            # 计算 Tau
            relative_pos_vec = aircraft.pos - missile.pos
            relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
            distance = np.linalg.norm(relative_pos_vec)
            if distance < 1e-6: continue

            closing_velocity = -np.dot(relative_vel_vec, relative_pos_vec) / distance

            # 只有正在接近的导弹才计算 Tau
            if closing_velocity > 1.0:
                tau = distance / closing_velocity
                if tau < min_tau:
                    min_tau = tau
                    primary_threat = missile
            # 如果所有导弹都在远离，则威胁最大的就是距离最近的那个
            elif min_tau == float('inf'):
                if distance < np.linalg.norm(aircraft.pos - primary_threat.pos):
                    primary_threat = missile

        return primary_threat

    # --- <<< 多导弹更改 >>> 修改所有需要历史状态的函数 ---

    def _reward_for_closing_velocity_change(self, aircraft: Aircraft, missile: Missile, dt: float):
        """(V3 - 多导弹版)"""
        # <<< 核心修改 >>> 获取该导弹的专属历史记录
        missile_hist = self._get_missile_history(missile.id)

        # ... (函数其余部分的计算逻辑完全不变) ...
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
        # <<< 核心修改 >>> 从专属历史记录中读取
        prev_vel = missile_hist['prev_closing_velocity']
        if prev_vel is not None:
            delta_V_close = closing_velocity_current - prev_vel
            reward_base = -delta_V_close
            SENSITIVITY = 0.1
            reward = np.tanh(reward_base * SENSITIVITY)

        # <<< 核心修改 >>> 更新到专属历史记录中
        missile_hist['prev_closing_velocity'] = closing_velocity_current

        return max(0.0, reward)  # 保持原来的逻辑，只返回正奖励

    def _reward_for_ata_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """(V3 - 多导弹版)"""
        # <<< 核心修改 >>> 获取该导弹的专属历史记录
        missile_hist = self._get_missile_history(missile.id)

        # ... (函数其余部分的计算逻辑完全不变) ...
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

        # <<< 核心修改 >>> 从专属历史记录中读取
        prev_ata = missile_hist['prev_ata_rad']
        if prev_ata is None:
            missile_hist['prev_ata_rad'] = current_ata_rad
            return 0.0

        delta_ata_rad = abs(current_ata_rad - prev_ata)
        ata_rate_rad_s = delta_ata_rad / dt if dt > 1e-6 else 0.0

        SENSITIVITY = 10.0
        normalized_reward = math.tanh(ata_rate_rad_s * SENSITIVITY)

        # <<< 核心修改 >>> 更新到专属历史记录中
        missile_hist['prev_ata_rad'] = current_ata_rad

        return normalized_reward

    # =========================================================================
    # 以下函数不需要修改，因为它们不依赖于历史状态，或者只依赖飞机状态
    # 为保持代码简洁，这里只列出函数签名，您可以直接复制粘贴它们
    # =========================================================================

    def _compute_missile_posture_reward_blend(self, aircraft: Aircraft, missile: Missile):
        # (此函数逻辑不变，因为它只计算当前帧的状态)
        # ... (代码与您原来的版本完全相同)
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

        if distance <= 3000.0:
            return reward_three_nine
        elif distance >= 4000.0:
            return reward_tail_chase
        else:
            alpha = (distance - 3000.0) / 1000.0
            return (1 - alpha) * reward_three_nine + alpha * reward_tail_chase

    def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
        # (此函数逻辑不变)
        # ... (代码与您原来的版本完全相同)
        try:
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            return 0.0
        if nz <= 0: return 0.0
        if self.AIRCRAFT_MAX_G <= 1e-6: return 0.0
        normalized_g = np.clip(nz / self.AIRCRAFT_MAX_G, 0.0, 1.0)
        return normalized_g

    def _compute_flare_timing_reward(self, flare_trigger_action: float, aircraft: Aircraft, missile: Missile) -> float:
        # (此函数逻辑不变，现在它接收的是 primary_threat)
        # ... (代码与您原来的版本完全相同)
        if flare_trigger_action < 0.5: return 0.0
        distance = np.linalg.norm(aircraft.pos - missile.pos)
        if distance < self.FLARE_EFFECTIVE_DISTANCE_M:
            return self.REWARD_FLARE_IN_WINDOW
        else:
            return self.PENALTY_FLARE_OUT_WINDOW

    def _compute_altitude_reward(self, aircraft: Aircraft):
        # (此函数逻辑不变)
        # ... (代码与您原来的版本完全相同)
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
        # (此函数逻辑不变)
        # ... (代码与您原来的版本完全相同)
        if release_flare_action > 0.5:
            alphaR = 2.0
            k1 = 3.0
            fraction_remaining = remaining_flares / total_flares
            penalty = -alphaR * (1 + k1 * (1 - fraction_remaining))
            return penalty
        return 0.0

    def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
        # (此函数逻辑不变)
        # ... (代码与您原来的版本完全相同)
        p_real_rad_s = aircraft.roll_rate_rad_s
        if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6: return 0.0
        normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
        return (normalized_roll_rate) * -1.0

    def _reward_for_coordinated_turn(self, aircraft: Aircraft, dt: float):
        # (此函数逻辑不变)
        # ... (代码与您原来的版本完全相同)
        try:
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            return 0.0
        positive_g = nz
        g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
        g_factor = np.clip(g_factor, 0, 1.0)
        reward = -1.0 * g_factor
        return reward