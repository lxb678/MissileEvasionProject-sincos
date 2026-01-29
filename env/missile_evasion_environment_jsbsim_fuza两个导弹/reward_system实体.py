# 文件: reward_system.py (实体注意力修改版)

import numpy as np
import math
from typing import List, Optional

# (中文) 导入对象类，以便进行类型提示和访问状态
from .AircraftJSBSim_DirectControl import Aircraft
from .missile2 import Missile


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

        # <<< 新增 >>>: 单枚导弹成功规避的补偿奖励
        # 即使任务失败，每躲过一枚导弹，惩罚减少 5.0
        # 逻辑: 2枚导弹，全中=-20，躲过1枚=-10，全躲过=SUCCESS(+20)
        self.PARTIAL_EVASION_BONUS = 10.0

        # self.SAFE_ALTITUDE_M = 3000.0
        # --- 所有超参数 ---
        # 确保所有这些 self.xxx = ... 的行都在您的代码中

        self.SAFE_ALTITUDE_M = 3000.0
        self.DANGER_ALTITUDE_M = 2000.0
        self.KV_MS = 50.0 #0.2 * 340
        self.MAX_ALTITUDE_M = 10000.0 #12000.0
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

    # # --- 稀疏奖励接口 (逻辑不变) ---
    # def get_sparse_reward(self, miss_distance: float, R_kill: float) -> float:
    #     if miss_distance > R_kill:
    #         return self.W
    #     else:
    #         return self.U

    # <<< 核心修改：基于标志位的稀疏奖励 >>>
    def get_sparse_reward(self, success: bool, missiles: List[Missile]) -> float:
        """
        计算稀疏奖励。
        :param success: 回合是否判定为完全成功
        :param missiles: 导弹对象列表 (检查 terminated 和 caused_hit 标志)
        :return: 最终奖励值
        """
        if success:
            # 这里的 success 是 environment 判定的（通常意味着所有导弹terminated且未命中）
            return self.W
        else:
            # --- 失败情况 (被命中 或 撞地) ---
            total_reward = self.U  # 基础惩罚 -20

            # 计算补偿：统计有多少枚导弹是被成功“耗死”或“躲过”的
            evaded_count = 0
            for m in missiles:
                # 条件1: 导弹已经停止活动 (terminated == True)
                # 条件2: 导弹不是导致坠机的凶手 (没有 caused_hit 属性 或 为 False)
                is_dead = m.terminated
                is_killer = getattr(m, 'caused_hit', False)

                if is_dead and not is_killer:
                    evaded_count += 1

            # 添加补偿 (每躲过一枚加 5 分)
            bonus = evaded_count * self.PARTIAL_EVASION_BONUS  # 建议设为 5.0

            total_reward += bonus

            # 打印调试信息（可选）
            # if evaded_count > 0:
            #     print(f"[Reward] 任务失败，但成功规避了 {evaded_count} 枚导弹。总分: {min(total_reward, -5.0)}")

            # 同样限制最高分为 -5.0，保证失败永远比成功分低
            return min(total_reward, -5.0)

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
        # reward_posture = 1.0 * self._compute_missile_posture_reward_blend(aircraft, missile)  #去掉看效果
        # reward_closing_velocity = 0.5 * self._reward_for_closing_velocity_change(aircraft, missile, dt=0.2)
        # reward_ata_rate = 1.0 * self._reward_for_los_rate(aircraft, missile, dt=0.2)
        # reward_flare_timing = 1.0 * self._compute_flare_timing_reward(flare_trigger_action, aircraft, missile) #去掉看效果
        reward_los_rate = 1.0 * self._reward_for_los_rate(aircraft, missile, dt=0.2)
        # 高G机动奖励可以与机动有效性（ATA Rate）相乘
        reward_high_g = 1.0 * self._reward_for_high_g_maneuver(aircraft)
        final_high_g_reward = 1.0 * reward_high_g * reward_los_rate

        # # <<< 新增：调用分离加速度奖励 >>>
        # reward_separation_accel = 0.5 * self._reward_for_separation_acceleration(aircraft, missile)

        return final_high_g_reward #+ reward_los_rate

    def _calculate_general_rewards(self, aircraft: Aircraft, flare_trigger_action: float,
                                   remaining_flares: int, total_flares: int,
                                   missile_related_reward: float, primary_threat: Optional[Missile]) -> float:
        """计算与特定导弹无关的通用奖励/惩罚。"""
        reward_altitude = 0.5 * self._compute_altitude_reward(aircraft)  #0.5
        reward_resource = 1.0 * self._compute_resource_penalty(flare_trigger_action, remaining_flares, total_flares)
        reward_roll_penalty = 0.5 * self._penalty_for_roll_rate_magnitude(aircraft) #0.5
        # reward_coordinated_turn = 0.5 * self._reward_for_coordinated_turn(aircraft, 0.2)  # 降低权重
        reward_punish_push_down = 0.5 * self._reward_for_punish_push_down(aircraft)
        # reward_speed = 0.5 * self._reward_for_maintaining_speed(aircraft)
        reward_survivaltime = 0.5 #0.2  # 每步存活奖励

        return (
                missile_related_reward +
                reward_altitude +
                reward_resource +
                reward_roll_penalty +
                # reward_coordinated_turn +
                reward_punish_push_down +
                # reward_speed +
                reward_survivaltime
        )

    def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float) -> float:
        """
        (V3 - 解析法 + 线性截断 - 多导弹适配) 奖励视线矢量的角速度。
        使用物理公式 |r x v| / r^2 直接计算。
        特性：无状态 (Stateless)，天然支持多导弹并行，无需历史记录。
        """
        # 1. 基础向量获取
        r_vec = aircraft.pos - missile.pos
        # 相对速度 = 飞机速度 - 导弹速度
        # 注意：v_rel 的方向反过来只会改变叉积向量的方向，不改变其模长，所以顺序不影响 los_rate 的大小
        v_rel = aircraft.get_velocity_vector() - missile.get_velocity_vector()

        # 距离平方 (r^2)
        r_sq = float(np.dot(r_vec, r_vec))
        r_norm = math.sqrt(r_sq)  # 需要用这个做 ATA 检查

        # 2. 基础数据准备 & 门控
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = float(np.linalg.norm(missile_v_vec))

        # 避免除零错误
        if r_sq < 1.0 or norm_missile_v < 0.1:
            return 0.0

        # --- 核心门控：后半球直接给满分 (1.0) ---
        # cos_ata = (R . Vm) / (|R|*|Vm|)
        cos_ata = np.dot(r_vec, missile_v_vec) / (r_norm * norm_missile_v)

        if cos_ata < 0:
            return 1.0  # 安全状态，威胁解除

        # --- 3. 解析法核心计算 ---
        # 公式: Omega = |r x v_rel| / r^2
        # 计算叉积 (Cross Product)
        cross_prod = np.cross(r_vec, v_rel)
        # 计算叉积的模长
        cross_norm = float(np.linalg.norm(cross_prod))

        # 得到瞬时视线角速率 (rad/s)
        los_rate = cross_norm / r_sq

        # --- 4. 线性截断归一化 ---
        # 设定目标阈值 (0.2 rad/s ≈ 11.5 deg/s)
        # 这个值代表了“优秀的规避机动”标准
        TARGET_LOS_RATE = 0.2

        # 线性映射
        normalized_reward = los_rate #/ TARGET_LOS_RATE

        # 截断到 [0, 1]
        reward = np.clip(normalized_reward, 0.0, 1.0)

        return float(reward)

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float) -> float:
    #     """
    #     (V4.1 - 持续奖励版 - 多导弹适配)
    #     奖励 LOS(视线矢量) 的旋转速率。
    #
    #     注意：此版本在后半球(cos_ata < 0)时会持续返回 1.2 奖励。
    #     已适配多导弹：使用 missile_hist 隔离状态，互不干扰。
    #     """
    #     # [关键点1] 获取当前这枚导弹的专属历史记录
    #     missile_hist = self._get_missile_history(missile.id)
    #
    #     # -----------------------------
    #     # 0) dt 安全
    #     # -----------------------------
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # -----------------------------
    #     # 1) 基础向量
    #     # -----------------------------
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_los = float(np.linalg.norm(current_los_vec))
    #
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_v = float(np.linalg.norm(missile_v_vec))
    #
    #     # [数值保护]
    #     if norm_los < 1.0 or norm_v < 0.1:
    #         missile_hist['prev_los_vec'] = None
    #         return 0.0
    #
    #     # -----------------------------
    #     # 2) 门控：后半球判断
    #     # -----------------------------
    #     cos_ata = float(np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v))
    #
    #     # [关键点2] 你要求的逻辑：持续给予 1.2 奖励
    #     if cos_ata < 0.0:
    #         # 必须清除 missile_hist 中的记录，而不是 self.prev_los_vec
    #         missile_hist['prev_los_vec'] = None
    #         return 1.2  # 只要在后半球，每一步都给 1.2
    #
    #     # -----------------------------
    #     # 3) 初始化/检查历史记录
    #     # -----------------------------
    #     # 从字典中读取上一帧向量
    #     prev_los_vec = missile_hist['prev_los_vec']
    #
    #     if prev_los_vec is None:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 0.0
    #
    #     prev_norm = float(np.linalg.norm(prev_los_vec))
    #     if prev_norm < 1.0:
    #         missile_hist['prev_los_vec'] = None  # 坏历史，重置
    #         return 0.0
    #
    #     # -----------------------------
    #     # 4) 计算 LOS 夹角 (atan2 版本)
    #     # -----------------------------
    #     current_u = current_los_vec / norm_los
    #     prev_u = prev_los_vec / prev_norm
    #
    #     dot_uv = float(np.clip(np.dot(current_u, prev_u), -1.0, 1.0))
    #     cross_norm = float(np.linalg.norm(np.cross(prev_u, current_u)))
    #
    #     delta_angle_rad = float(np.arctan2(cross_norm, dot_uv))
    #
    #     # [关键点3] 更新字典中的历史记录
    #     missile_hist['prev_los_vec'] = current_los_vec
    #
    #     # -----------------------------
    #     # 5) LOS rate + 抖动抑制
    #     # -----------------------------
    #     los_rate = delta_angle_rad / safe_dt
    #
    #     # 限制最大值 (抑制物理引擎噪声)
    #     rate_cap = 2.0
    #     los_rate = min(los_rate, rate_cap)
    #
    #     # -----------------------------
    #     # 6) 奖励映射
    #     # -----------------------------
    #     SENSITIVITY = 1.0 #10.0
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #
    #     return float(reward)

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float) -> float:
    #     """
    #     (V4.1 - 最小改动稳定版 - 多导弹适配)
    #     奖励 LOS(视线矢量) 的旋转速率，抑制刷分与抖动。
    #
    #     特性：
    #       1. Latch机制：后半球给予一次性大额奖励，避免Agent在安全区刷分。
    #       2. atan2计算：解决小角度下的数值不稳定性。
    #       3. Rate Cap：限制最大奖励幅度，过滤物理引擎噪声。
    #     """
    #     # 获取该特定导弹的历史记录
    #     missile_hist = self._get_missile_history(missile.id)
    #
    #     # -----------------------------
    #     # 0) dt 安全
    #     # -----------------------------
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # -----------------------------
    #     # 1) 基础向量
    #     # -----------------------------
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_los = float(np.linalg.norm(current_los_vec))
    #
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_v = float(np.linalg.norm(missile_v_vec))
    #
    #     # [数值保护]：距离过近或速度异常时不计算
    #     # 如果距离太近(<1.0m)，物理计算可能发散，直接视为0
    #     if norm_los < 1.0 or norm_v < 0.1:
    #         missile_hist['prev_los_vec'] = None
    #         missile_hist['los_safe_latch'] = False
    #         return 0.0
    #
    #     # -----------------------------
    #     # 2) 门控：后半球一次性奖励 (Latch机制)
    #     # -----------------------------
    #     # dot > 0: 前半球(威胁); dot < 0: 后半球(相对安全)
    #     cos_ata = float(np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v))
    #
    #     if cos_ata < 0.0:
    #         # --- 进入后半球 (安全区) ---
    #         # 逻辑：只有在"刚进入"的那一帧给予一次性大奖 (Bonus)
    #         # 之后只要保持在后半球，Latch 为 True，奖励为 0，防止刷分
    #         if missile_hist['los_safe_latch'] is False:
    #             bonus = 1.0
    #             missile_hist['los_safe_latch'] = True
    #             missile_hist['prev_los_vec'] = None  # 重置历史，避免切回前半球时角度突变
    #             return float(bonus)
    #         else:
    #             # 已经在后半球了，保持安静，专注于其他任务
    #             missile_hist['prev_los_vec'] = None
    #             return 0.0
    #     else:
    #         # --- 回到前半球 (威胁区) ---
    #         missile_hist['los_safe_latch'] = False
    #
    #     # -----------------------------
    #     # 3) 初始化/检查历史记录
    #     # -----------------------------
    #     prev_los_vec = missile_hist['prev_los_vec']
    #
    #     if prev_los_vec is None:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 0.0
    #
    #     prev_norm = float(np.linalg.norm(prev_los_vec))
    #     if prev_norm < 1.0:
    #         missile_hist['prev_los_vec'] = None  # 坏历史，重置
    #         return 0.0
    #
    #     # -----------------------------
    #     # 4) 计算 LOS 单位向量夹角 (atan2 版本)
    #     # -----------------------------
    #     current_u = current_los_vec / norm_los
    #     prev_u = prev_los_vec / prev_norm
    #
    #     # 点积 (cos)
    #     dot_uv = float(np.clip(np.dot(current_u, prev_u), -1.0, 1.0))
    #     # 叉积模长 (sin)
    #     cross_norm = float(np.linalg.norm(np.cross(prev_u, current_u)))
    #
    #     # 使用 atan2 计算精确的弧度变化 [0, pi]
    #     delta_angle_rad = float(np.arctan2(cross_norm, dot_uv))
    #
    #     # 更新历史
    #     missile_hist['prev_los_vec'] = current_los_vec
    #
    #     # -----------------------------
    #     # 5) LOS rate + 抖动抑制 (Rate Cap)
    #     # -----------------------------
    #     los_rate = delta_angle_rad / safe_dt
    #
    #     # rate_cap: 2.0 rad/s 约等于 115度/秒，超过这个值的通常是物理引擎碰撞噪声
    #     rate_cap = 2.0
    #     los_rate = min(los_rate, rate_cap)
    #
    #     # -----------------------------
    #     # 6) 奖励映射
    #     # -----------------------------
    #     # LOS rate 通常在 0.05 ~ 0.5 之间。
    #     # 0.1 rad/s * 10.0 = 1.0 -> tanh(1.0) ≈ 0.76 (显著奖励)
    #     SENSITIVITY = 10.0
    #
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #
    #     return float(reward)

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V3 - 优化版 - 适配多导弹) 奖励天线训练角 (ATA) 的变化率。
    #     核心逻辑：奖励 ATA 角的增大（Situation Improving），使用 tanh(rate * 10.0) 放大信号。
    #     """
    #     # --- 0. 获取该特定导弹的历史记录 (适配多导弹架构) ---
    #     missile_hist = self._get_missile_history(missile.id)
    #
    #     # --- 1. 获取视线矢量和导弹速度矢量 ---
    #     current_los_vec = aircraft.pos - missile.pos
    #     missile_v_vec = missile.get_velocity_vector()
    #
    #     norm_los = np.linalg.norm(current_los_vec)
    #     norm_v = np.linalg.norm(missile_v_vec)
    #
    #     # [优化1] 数值保护：增加极小值防止除零，过滤无意义的物理状态
    #     if norm_los < 1e-1 or norm_v < 1e-1:
    #         missile_hist['prev_ata_rad'] = None  # 重置该导弹的历史
    #         return 0.0
    #
    #     # --- 2. 前半球门控 ---
    #     # 计算 ATA 角的余弦值
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)
    #
    #     # 如果飞机在导弹后半球 (ATA > 90度, cos_ata < 0)，则威胁解除
    #     # [逻辑确认] 这表示导弹正在飞离飞机，给予最大奖励鼓励保持
    #     if cos_ata < 0:
    #         missile_hist['prev_ata_rad'] = None  # 重置该导弹的历史
    #         return 1.0
    #
    #     # 3. 计算当前的 ATA 角
    #     current_ata_rad = np.arccos(np.clip(cos_ata, -1.0, 1.0))
    #
    #     # 4. 初始化历史记录
    #     if missile_hist['prev_ata_rad'] is None:
    #         missile_hist['prev_ata_rad'] = current_ata_rad
    #         return 0.0
    #
    #     # --- 3. 计算变化率 ---
    #     # 读取历史
    #     prev_ata = missile_hist['prev_ata_rad']
    #
    #     # 计算差值
    #     delta_ata_rad = current_ata_rad - prev_ata
    #
    #     # [优化2] dt 安全检查：防止 dt 为 0 或极小时导致梯度爆炸
    #     safe_dt = max(dt, 1e-4)
    #     rate = delta_ata_rad / safe_dt
    #
    #     reward = 0.0
    #
    #     # --- 4. 核心奖励逻辑 ---
    #     if delta_ata_rad > 0:
    #         # ATA 正在增大 (Situation Improving)
    #
    #         # [优化3] 调整敏感度系数 (Scaling Factor)
    #         # 理由：ATA的变化率通常很小。如果不放大，tanh后奖励太微弱。
    #         # 设为 10.0，让有效的规避动作能拿到接近 1.0 的奖励。
    #         SENSITIVITY = 10.0
    #         reward = math.tanh(rate * SENSITIVITY)
    #     else:
    #         # ATA 正在减小 (Situation Worsening)
    #         # 保持为 0.0，不惩罚，避免AI陷入局部最优（不敢动）
    #         reward = 0.0
    #
    #     # 5. 更新该导弹的历史记录
    #     missile_hist['prev_ata_rad'] = current_ata_rad
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 几何修正版) 奖励视线矢量的角速度。
    #     使用几何点积计算实际转动角度，并使用 tanh 限制最大奖励幅度。
    #     """
    #     # 获取该特定导弹的历史记录 (适配多导弹架构)
    #     missile_hist = self._get_missile_history(missile.id)
    #
    #     # 1. 计算当前的视线矢量
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # --- 门控逻辑 ---
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     # 异常值处理
    #     if norm_current < 1e-6 or norm_missile_v < 1e-6:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 0.0
    #
    #     # 判断是否在导弹后方 (安全区)
    #     # cos_ata < 0 说明导弹正在远离或飞机在导弹后方，此时威胁解除
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #     if cos_ata < 0:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 1.0  # 威胁解除，给满分
    #
    #     # 初始化历史记录
    #     if missile_hist['prev_los_vec'] is None:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 0.0
    #
    #     # 2. 计算几何角度变化
    #     prev_los_vec = missile_hist['prev_los_vec']
    #     norm_prev = np.linalg.norm(prev_los_vec)
    #
    #     if norm_prev < 1e-6:
    #         missile_hist['prev_los_vec'] = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = prev_los_vec / norm_prev
    #
    #     # 点积计算夹角
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     # 计算角速率
    #     los_angular_rate_rad_s = angle_rad / dt if dt > 1e-6 else 0.0
    #
    #     # 5. 更新历史记录
    #     missile_hist['prev_los_vec'] = current_los_vec
    #
    #     # --- [关键修改] 使用 tanh 进行归一化 ---
    #     # 为什么要改？因为在交汇瞬间 rate 可能极高。
    #     # scaling_factor: 调节敏感度。
    #     # 如果 los_rate 为 0.1 rad/s (~5.7度/s)，scaling=10.0 -> tanh(1.0) ≈ 0.76
    #     scaling_factor = 1.0  # 建议根据训练情况调整，例如 10.0 或 20.0
    #     reward = math.tanh(los_angular_rate_rad_s * scaling_factor)
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 切向速度最终版) 奖励视线角速率 (LOS Rate)。
    #     适用于实体关注机制：无状态，计算快，信号稳健。
    #
    #     核心原理：
    #     不计算几何角度差，而是计算【切向速度】(Tangential Velocity)。
    #     切向速度越大 => 视线转动越快 => 导弹必须拉更大的过载。
    #     """
    #     # --- 1. 计算基础矢量 ---
    #     # 视线矢量 (从导弹指向飞机)
    #     r_vec = aircraft.pos - missile.pos
    #     distance = np.linalg.norm(r_vec)
    #
    #     # 避免除零
    #     if distance < 1e-6:
    #         return 0.0
    #
    #     # 视线单位矢量
    #     los_unit_vec = r_vec / distance
    #
    #     # --- 2. 前半球安全门控 ---
    #     # 这一步非常关键：如果导弹已经飞过了（在飞机屁股后面），
    #     # 就不需要再做机动了，直接给满分奖励。
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     if norm_missile_v > 1e-6:
    #         # 计算 ATA (Antenna Train Angle) 的余弦
    #         cos_ata = np.dot(los_unit_vec, missile_v_vec) / norm_missile_v
    #
    #         # 如果 cos_ata < 0，说明飞机在导弹后方 (或侧后方)，威胁已解除
    #         # 此时给予最大奖励，鼓励保持这种安全状态
    #         if cos_ata < 0:
    #             return 1.0
    #
    #     # --- 3. 计算切向速度 (核心逻辑) ---
    #     aircraft_v_vec = aircraft.get_velocity_vector()
    #     aircraft_speed = np.linalg.norm(aircraft_v_vec)
    #
    #     # 如果飞机静止，无法产生 LOS Rate
    #     if aircraft_speed < 1e-6:
    #         return 0.0
    #
    #     # 分解速度矢量：
    #     # 径向速度 (Radial) = 速度在视线方向上的投影
    #     # 切向速度 (Tangential/Transverse) = 总速度 - 径向速度
    #
    #     # v_radial = (v . u) * u
    #     radial_velocity_component = np.dot(aircraft_v_vec, los_unit_vec) * los_unit_vec
    #
    #     # v_tan = v - v_radial
    #     transverse_velocity_vec = aircraft_v_vec - radial_velocity_component
    #
    #     transverse_speed = np.linalg.norm(transverse_velocity_vec)
    #
    #     # --- 4. 计算奖励 ---
    #     # 归一化方法: reward = V_tan / V_total = sin(Aspect Angle)
    #     # 当飞机垂直于视线飞行时 (Beaming/Notching)，奖励接近 1.0
    #     # 当飞机迎头或尾追时，奖励接近 0.0
    #     reward = transverse_speed / aircraft_speed
    #
    #     return reward


    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     奖励视线角变化率 |λ_dot|（比例导引 PN 的核心量之一）
    #     计算公式（3D通用）:
    #         |λ_dot| = || r × v_rel || / ||r||^2
    #     其中:
    #         r     = target_pos - missile_pos
    #         v_rel = target_vel - missile_vel
    #     奖励输出范围: [0, 1)（tanh 归一化）
    #     此方法无需历史状态，天然支持多导弹并行计算。
    #     """
    #
    #     # --- 1) 相对位置 r 与相对速度 v_rel ---
    #     r = aircraft.pos - missile.pos
    #     v_m = missile.get_velocity_vector()
    #     v_t = aircraft.get_velocity_vector()
    #     v_rel = v_t - v_m
    #
    #     r_norm = np.linalg.norm(r)
    #     if r_norm < 1e-6:
    #         return 0.0
    #
    #     # --- 2) 前/后半球门控 ---
    #     # 用导弹速度与 LOS 的夹角判断“导弹是否仍在追向目标”
    #     v_m_norm = np.linalg.norm(v_m)
    #     if v_m_norm > 1e-6:
    #         # 计算 ATA (Antenna Train Angle) 的余弦
    #         cos_look = float(np.dot(r, v_m) / (r_norm * v_m_norm))
    #
    #         # 防抖：给一点裕度，避免 cos 在 0 附近抖动
    #         REAR_GATE = -0.05  # 小于这个才认为进入后半球 (ATA > 90度)
    #         if cos_look < REAR_GATE:
    #             return 1.0  # 威胁解除，给予最大奖励
    #
    #     # --- 3) 计算 LOS 角速率幅值 |λ_dot| ---
    #     # 公式: |r x v| / |r|^2
    #     cross = np.cross(r, v_rel)
    #     los_rate = np.linalg.norm(cross) / (r_norm ** 2)  # 单位: rad/s
    #
    #     # --- 4) 归一化：tanh ---
    #     # SENSITIVITY 决定多大 los_rate 会接近饱和
    #     # 举例：如果 los_rate 常见量级 ~0.05 rad/s，设 20 让 tanh(1)=0.76 左右
    #     SENSITIVITY = 20.0
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     奖励视线角变化率 |λ_dot|（比例导引 PN 的核心量之一）
    #     计算公式（3D通用）:
    #         |λ_dot| = || r × v_rel || / ||r||^2
    #     其中:
    #         r     = target_pos - missile_pos
    #         v_rel = target_vel - missile_vel
    #     奖励输出范围: [0, 1)（tanh 归一化）
    #     """
    #
    #     # --- 1) 相对位置 r 与相对速度 v_rel ---
    #     r = aircraft.pos - missile.pos
    #     v_m = missile.get_velocity_vector()
    #     v_t = aircraft.get_velocity_vector()  # 如果你没有这个函数，见下方说明
    #     v_rel = v_t - v_m
    #
    #     r_norm = np.linalg.norm(r)
    #     if r_norm < 1e-6:
    #         return 0.0
    #
    #     # --- 2) 前/后半球门控（可选，但你之前就想要） ---
    #     # 用导弹速度与 LOS 的夹角判断“导弹是否仍在追向目标”
    #     v_m_norm = np.linalg.norm(v_m)
    #     if v_m_norm > 1e-6:
    #         cos_look = float(np.dot(r, v_m) / (r_norm * v_m_norm))
    #         # 防抖：给一点裕度，避免 cos 在 0 附近抖动
    #         REAR_GATE = -0.05  # 小于这个才认为进入后半球
    #         if cos_look < REAR_GATE:
    #             return 1.0  # 你想要“脱离威胁给最大奖励”
    #     # 如果导弹速度无效，就不门控，继续算 LOS rate
    #
    #     # --- 3) 计算 LOS 角速率幅值 |λ_dot| ---
    #     cross = np.cross(r, v_rel)
    #     los_rate = np.linalg.norm(cross) / (r_norm ** 2)  # rad/s
    #
    #     # --- 4) 归一化：tanh（避免 dt 太小导致爆炸，这里已经不除 dt 了）---
    #     # SENSITIVITY 决定多大 los_rate 会接近饱和
    #     # 举例：如果 los_rate 常见量级 ~0.05 rad/s，设 20 让 tanh(1)=0.76 左右
    #     SENSITIVITY = 20.0
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #
    #     return reward

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
                # --- V4.1 新增状态 ---
                'prev_los_vec': None,  # 记录上一帧视线矢量
                'los_safe_latch': False,  # 记录是否已判定为安全（防止反复刷分）
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
        SPEED_THRESHOLD_MS = 250.0 #0.6 * 340.0 #0.8 * 340.0

        # 2. 定义固定的正奖励值
        # 当速度达标时，给予这个奖励。
        REWARD_FOR_SAFE_SPEED = 0.0 #0.5  # 可以调整这个值的大小

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
            MAX_SHORTFALL_FOR_PENALTY = 100.0 #50 #120.0

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

    # def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
    #     try:
    #         nz = aircraft.get_normal_g_force()
    #     except AttributeError:
    #         return 0.0
    #     if nz <= 0: return 0.0
    #     if self.AIRCRAFT_MAX_G <= 1e-6: return 0.0
    #     normalized_g = np.clip(nz / self.AIRCRAFT_MAX_G, 0.0, 1.0)
    #     return normalized_g

    def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
        """
        (修改版) 奖励飞机执行高G机动 (基于总过载 Total G)。
        注意：这包含了升力(Nz)、阻力(Nx)和侧向力(Ny)。
        """
        try:
            # 1. 获取三个轴向的分量 (注意：这里已经是归一化后的G值)
            # 你需要确保 Aircraft 类里有这个 get_total_g_force 方法
            total_g, (g_x, g_y, g_z) = aircraft.get_total_g_force()

            # 调试：观察主要是哪个轴在贡献G力
            # print(f"Tx:{g_x:.1f} Ty:{g_y:.1f} Tz:{g_z:.1f} | Total:{total_g:.1f}")

        except AttributeError:
            print("Aircraft类缺少 get_total_g_force 方法")
            return 0.0

        # # 2. 门槛过滤
        # # 平飞时 total_g 约为 1.0 (重力)
        # # 我们只奖励超过 1G 的机动
        # if total_g <= 1.0:
        #     return 0.0

        # 3. 归一化 (Mapping 1.0 ~ Max_G -> 0.0 ~ 1.0)
        min_g_threshold = 0.0 #1.0

        # 防止分母为0或负数
        if self.AIRCRAFT_MAX_G <= min_g_threshold:
            return 0.0

        normalized_g = (total_g - min_g_threshold) / (self.AIRCRAFT_MAX_G - min_g_threshold)
        normalized_g = np.clip(normalized_g, 0.0, 1.0)

        # 4. 奖励计算
        # 使用平方，鼓励高G爆发
        # reward = np.power(normalized_g, 2.0) * 1.0
        reward = normalized_g * 1.0

        # --- [可选安全锁] ---
        # 如果你担心 AI 靠“急刹车”(Nx) 骗分，可以加一个惩罚项：
        # 如果主要过载不是来自 Nz (拉杆)，则打折奖励。
        # 判断标准：如果 |Nz| 占比小于 80%，说明飞行动作很怪。
        # ratio = abs(g_z) / (total_g + 1e-6)
        # if ratio < 0.8:
        #     reward *= 0.5

        return reward

    # def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
    #     """
    #     奖励飞机执行高G机动 (基于法向过载 Nz)。
    #     修正：处理 JSBSim Z轴向下导致的负值问题。
    #     """
    #     try:
    #         # 1. 获取 Z 轴过载
    #         # JSBSim 中拉杆产生向上的升力，Z轴向下，所以拉杆时值为负 (例如 -5.0)
    #         g_z_raw = aircraft.fdm.get_property_value('accelerations/n-pilot-z-norm')
    #
    #         # 2. 【关键修正】取反，转换为符合直觉的 "Pilot G"
    #         # 现在：平飞 ≈ 1.0，拉杆 ≈ 5.0，推杆 < 1.0
    #         current_nz = -g_z_raw
    #
    #     except AttributeError:
    #         return 0.0
    #
    #     # 3. 过滤：只奖励有效的正向机动
    #     # 平飞是 1G。如果没有拉杆 (<=1G) 或者在推杆 (负G)，则不给奖励
    #     if current_nz <= 1.0:
    #         return 0.0
    #
    #     # 4. 归一化 (Mapping 1.0 ~ Max_G -> 0.0 ~ 1.0)
    #     # 假设 AIRCRAFT_MAX_G 设置为 9.0
    #     min_g_threshold = 1.0
    #
    #     if self.AIRCRAFT_MAX_G <= min_g_threshold:
    #         return 0.0
    #
    #     # 计算归一化值
    #     normalized_g = (current_nz - min_g_threshold) / (self.AIRCRAFT_MAX_G - min_g_threshold)
    #
    #     # 截断到 [0, 1] 之间，防止超过最大G时奖励溢出
    #     normalized_g = np.clip(normalized_g, 0.0, 1.0)
    #
    #     # 5. 奖励计算
    #     # 使用平方 (power 2) 鼓励 AI 只有在真正需要急转弯（高G）时才获得高分
    #     # 也可以直接用 linear: reward = normalized_g * 1.0
    #     reward = np.power(normalized_g, 2.0) * 1.0
    #     # reward = normalized_g * 1.0
    #
    #     # 调试打印 (可选)
    #     # print(f"Raw: {g_z_raw:.2f}, PilotG: {current_nz:.2f}, Rew: {reward:.3f}")
    #
    #     return reward

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
            # Pv = - penalty_factor
            Pv = -np.clip(penalty_factor, 0.0, 1.0)
        PH = 0.0
        if altitude_m <= self.DANGER_ALTITUDE_M:
            PH = np.clip(altitude_m / self.DANGER_ALTITUDE_M, 0.0, 1.0) - 1.0
        P_over = 0.0
        if altitude_m > self.MAX_ALTITUDE_M:
            P_over = -((altitude_m - self.MAX_ALTITUDE_M) / 1000.0) * self.OVER_ALTITUDE_PENALTY_FACTOR
        return Pv + PH + P_over

    # def _compute_altitude_reward(self, aircraft: Aircraft):
    #     """
    #     (V2 改进版) 增强低空生存本能，引入非线性惩罚。
    #     """
    #     altitude_m = aircraft.pos[1]
    #     # 即使 aircraft.pos[1] 理论上是 h_sl，但为了防止穿地后的数值异常，截断一下
    #     altitude_m = max(0.0, altitude_m)
    #     v_vertical_ms = aircraft.get_velocity_vector()[1]
    #
    #     # --- 全局惩罚权重 (关键！) ---
    #     # 建议设置为与 self.W (20) 相当的量级，确保生存优先级高于攻击
    #     ALTITUDE_PENALTY_WEIGHT = 1.0  # 5.0
    #
    #     # -----------------------------------------------------------
    #     # 1. 低空俯冲惩罚 (Pv) - 只有在向下冲时才惩罚
    #     # -----------------------------------------------------------
    #     Pv = 0.0
    #     if altitude_m <= self.SAFE_ALTITUDE_M and v_vertical_ms < -0.1:  # 增加一个小死区防止抖动
    #         descent_speed = abs(v_vertical_ms)
    #
    #         # 归一化高度因子: 地面=1.0, 安全高度=0.0
    #         h_factor = (self.SAFE_ALTITUDE_M - altitude_m) / self.SAFE_ALTITUDE_M
    #
    #         # 归一化速度因子: 比如 KV_MS (68m/s) 时为 1.0
    #         v_factor = descent_speed / self.KV_MS
    #
    #         # 移除 clip 的上限，让高速俯冲变得极其昂贵
    #         # 例如: 高度50m (h~1.0), 速度 200m/s (v~3.0) -> raw = 3.0
    #         raw_pv = h_factor * v_factor
    #
    #         # 使用平方让它更敏感，或者直接线性
    #         Pv = -1.0 * raw_pv
    #
    #         # -----------------------------------------------------------
    #     # 2. 绝对高度惩罚 (PH) - 接近地面时的指数级恐慌
    #     # -----------------------------------------------------------
    #     PH = 0.0
    #     if altitude_m <= self.DANGER_ALTITUDE_M:
    #         # 使用指数惩罚：在 DANGER_ALTITUDE_M 处为 0，在 0m 处急剧增大
    #         # 公式: (1 - h/h_danger)^2  -> 抛物线增长
    #         ratio = altitude_m / self.DANGER_ALTITUDE_M
    #         # 当 altitude_m = 0, penalty = -1.0
    #         # 当 altitude_m = 1500, penalty = 0.0
    #         # 使用平方 (pow 2) 让接近 0m 时的梯度更陡峭
    #         PH = -1.0 * ((1.0 - ratio) ** 2)
    #
    #     # -----------------------------------------------------------
    #     # 3. 超高空惩罚 (P_over) - 保持线性即可
    #     # -----------------------------------------------------------
    #     P_over = 0.0
    #     if altitude_m > self.MAX_ALTITUDE_M:
    #         diff = altitude_m - self.MAX_ALTITUDE_M
    #         # 每高出 1000米，惩罚 -0.5 (保持原逻辑，这属于软约束)
    #         P_over = -(diff / 1000.0) * self.OVER_ALTITUDE_PENALTY_FACTOR
    #
    #     # -----------------------------------------------------------
    #     # 4. 总计
    #     # -----------------------------------------------------------
    #     # Pv 和 PH 是生存关键，乘以大权重
    #     # P_over 是任务边界，权重小一点也没关系，或者统一乘
    #
    #     # 假设 ALTITUDE_PENALTY_WEIGHT = 5.0
    #     # 撞地瞬间 (h=0, v=-100):
    #     # PH = -1.0
    #     # Pv = -1.0 * (1.0 * 1.5) = -1.5
    #     # Total = 5.0 * (-2.5) = -12.5
    #     # 这个量级 (-12.5) 足以抵消掉大部分非致命的攻击奖励 (+5 ~ +10)，迫使AI拉起。
    #
    #     total_reward = ALTITUDE_PENALTY_WEIGHT * (Pv + PH) + P_over
    #
    #     return total_reward

    # def _compute_resource_penalty(self, release_flare_action, remaining_flares, total_flares):
    #     if release_flare_action > 0.5:
    #         alphaR = 2.0
    #         k1 = 3.0
    #         fraction_remaining = remaining_flares / total_flares
    #         penalty = -alphaR * (1 + k1 * (1 - fraction_remaining))
    #         return penalty
    #     return 0.0

    def _compute_resource_penalty(self, release_flare_action, remaining_flares, total_flares):
        """
        线性资源惩罚函数 (Linear Penalty)
        范围: 0.0 (满弹) -> -1.0 (空弹)

        逻辑:
        惩罚力度均匀增加。
        每发射一枚，惩罚增加的幅度是固定的 (1 / total_flares)。
        """
        # 1. 动作阈值判断
        if release_flare_action <= 0.5:
            return 0.0

        # 2. 安全处理
        if total_flares <= 0:
            return -1.0

        # 3. 计算使用比例
        # fraction_used = 1.0 - (剩余 / 总数)
        # 满弹时为 0.0，空弹时为 1.0
        fraction_used = 1.0 - (remaining_flares / total_flares)

        # 4. 线性计算 (直接取负)
        penalty = -fraction_used

        return penalty

    # def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
    #     p_real_rad_s = aircraft.roll_rate_rad_s
    #     if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6: return 0.0
    #     # if np.rad2deg(abs(p_real_rad_s)) < 120: return 0.0
    #     # print(np.rad2deg(p_real_rad_s))
    #     normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
    #     return (normalized_roll_rate) * -1.0

    def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
        p_real_rad_s = aircraft.roll_rate_rad_s

        # 防御性检查：避免除以0
        if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6:
            return 0.0

        # 1. 归一化 (Normalization)
        # 结果通常在 [-1, 1] 之间。
        # 即使模拟器出现Bug导致超过最大值，下一步的平方也能处理，但建议clip一下防止数值爆炸。
        # normalized_roll_rate = p_real_rad_s / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
        # normalized_roll_rate = np.clip(normalized_roll_rate, -1.5, 1.5)  # 防止物理引擎抽风产生极大值

        normalized_roll_rate = p_real_rad_s * 1.0 / 4.0

        # 2. 平方惩罚 (Quadratic Penalty)
        # 核心修改：去掉 abs，改为平方 (**2)
        # 效果：0.1的滚转只惩罚 0.01（几乎忽略，允许微调），0.9的滚转惩罚 0.81（重罚）
        penalty_base = normalized_roll_rate ** 2

        # 3. 权重调节 (Weighting)
        # 之前的 -1.0 可能过于武断。你需要根据你的 Total Reward 范围来调整这个 k。
        # 假设你躲避导弹成功的奖励是 +10，这里建议 k 取 0.5 左右。
        # 如果躲避成功只有 +1，这里 k 建议取 0.05 ~ 0.1。
        k = 1.0 #0.5

        return -1.0 * k * penalty_base

    # def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
    #     p_real_rad_s = aircraft.roll_rate_rad_s
    #
    #     # --- 参数设置 ---
    #     # sensitivity (敏感度): 控制惩罚上升的快慢。
    #     # 假设我们认为 60度/秒 (~1.0 rad/s) 是比较显著的滚转。
    #     # 如果 sensitivity = 0.5:
    #     #   1.0 rad/s -> tanh(0.5) = 0.46 (中等惩罚)
    #     #   3.0 rad/s -> tanh(1.5) = 0.90 (接近最大惩罚)
    #     #   10.0 rad/s -> tanh(5.0) = 0.999 (最大也就这样了，不会爆炸)
    #     sensitivity = 0.2
    #
    #     # 1. 计算 Tanh 值 (范围 0 到 1)
    #     # 使用绝对值，因为滚转方向不重要
    #     normalized_penalty = np.tanh(abs(p_real_rad_s) * sensitivity)
    #
    #     # 2. 施加死区 (可选，建议加上)
    #     # 如果滚转非常小（比如 < 0.1 rad/s），直接忽略，避免飞机发生高频微颤
    #     if abs(p_real_rad_s) < 0.1:
    #         return 0.0
    #
    #     # 3. 最终权重
    #     k = 0.5
    #
    #     return -k * normalized_penalty

    def _reward_for_punish_push_down(self, aircraft: Aircraft):
        """
        惩罚压杆（负G）行为。
        JSBSim Z轴向下：正值代表压杆 (Pilot Negative G)。
        平飞时'accelerations/n-pilot-z-norm'是-1，那么应该是和飞行员操作直接相关，大于0就是在压杆。
        """
        try:
            # 1. 获取原始 Nz (拉杆为负，压杆为正)
            # 例如：推杆产生 -2G (Pilot)，JSBSim 读数为 +2.0
            nz_raw = aircraft.fdm.get_property_value('accelerations/n-pilot-z-norm')
        except AttributeError:
            return 0.0

        # 2. 设定容忍阈值
        # 有时候轻微的推杆（0G 到 1G 之间，即失重）是为了加速，可以允许。
        # 但如果 Pilot G < 0 (即 nz_raw > 0)，就是真正的负G机动，需要惩罚。
        # 甚至更严格一点：Pilot G < 0.5 (即 nz_raw > -0.5) 就开始惩罚，防止AI习惯低G飞行

        # 这里演示：只惩罚 Pilot G < 0 (即 nz_raw > 0) 的情况
        if nz_raw <= 0:
            return 0.0

        # 3. 归一化惩罚
        # 负G的极限通常比正G小得多，一般 F16 限制在 -3G 左右
        MAX_NEGATIVE_G = 5.0

        # 计算惩罚系数 (越大的正 nz_raw，惩罚越重)
        penalty_factor = nz_raw / MAX_NEGATIVE_G
        penalty_factor = np.clip(penalty_factor, 0.0, 1.0)

        # 4. 返回负奖励
        # 系数 1.0 可以根据需要调整权重
        reward = -1.0 * penalty_factor

        # print(f"Pushing Down! Raw: {nz_raw:.2f}, Penalty: {reward:.3f}")
        return reward

    def _reward_for_coordinated_turn(self, aircraft: Aircraft, dt: float):
        # try:
        #     nz = aircraft.get_normal_g_force()
        # except AttributeError:
        #     return 0.0
        # positive_g = nz
        # g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
        # g_factor = np.clip(g_factor, 0, 1.0)
        # reward = -1.0 * g_factor

        # # 1. 计算滚转因子 (请务必最后一次确认滚转角的正确索引，很可能是[3]！)
        # try:
        #     roll_rad = aircraft.state_vector[6]  # <--- 滚转角 phi
        # except IndexError:
        #     return 0.0
        #
        # roll_factor = math.sin(abs(roll_rad))
        # # if abs(np.rad2deg(roll_rad)) > 90:
        # #     roll_factor = math.sin(math.pi - abs(roll_rad))
        # roll_factor = np.clip(roll_factor, 0, 1.0)

        # 2. 直接从飞机对象获取法向G力 (nz)
        try:
            nz = aircraft.get_normal_g_force()
            # nz2 = -aircraft.get_pilot_normal_g()
        except AttributeError:
            print("错误: 您的 Aircraft 类中没有 get_normal_g_force() 方法。请先实现它。")
            return 0.0

        # 3. 根据JSBSim的符号约定计算G力因子
        #    平飞: nz = -1.0 G -> positive_g = 1.0 G
        #    2G转弯: nz = -2.0 G -> positive_g = 2.0 G
        positive_g = nz

        # G力因子是超过1G的部分
        g_factor = min(0, positive_g) / (self.MAX_G_LIMIT)
        # g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
        g_factor = np.clip(g_factor, 0, 1.0)

        # 4. 最终奖励
        # reward = -1.0 * roll_factor * g_factor
        reward = -1.0  * g_factor

        # 调试打印
        # print(f"RollF: {roll_factor:.2f} | NZ1: {nz:.2f} | G_F: {g_factor:.2f} | Reward: {reward:.3f}")

        return reward