# 文件: reward_system.py (已修正，精确匹配您的代码)

import numpy as np
import math

# (中文) 导入对象类，以便进行类型提示和访问状态
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile


class RewardCalculator:
    """
    封装了所有与奖励计算相关的逻辑。
    所有逻辑和参数均从您最新的 AirCombatEnv 文件中精确提取。
    """

    def __init__(self):
        # --- 在这里集中定义所有奖励相关的超参数 ---

        # [稀疏奖励参数]
        self.W = 20 #50  # 成功奖励基准
        self.U = -20 #-50  # 失败固定惩罚

        # [高度惩罚参数]
        self.SAFE_ALTITUDE_M = 3000.0 #1000.0
        self.DANGER_ALTITUDE_M = 1500.0 #500.0
        self.KV_MS = 0.2 * 340
        self.MAX_ALTITUDE_M = 12000.0
        self.OVER_ALTITUDE_PENALTY_FACTOR = 0.5

        # [三九线奖励参数]
        self.ASPECT_REWARD_EFFECTIVE_RANGE = 10000.0 #5000.0
        self.ASPECT_REWARD_WIDTH_DEG = 45.0
        self.ASPECT_PITCH_THRESHOLD_DEG = 75.0

        # [滚转惩罚参数] (无阈值版本)
        self.MAX_PHYSICAL_ROLL_RATE_RAD_S = np.deg2rad(240.0)
        # self.MAX_PHYSICAL_ROLL_RATE_RAD_S = np.deg2rad(324.0)

        # [速度惩罚参数]
        self.OPTIMAL_SPEED_FLOOR_MACH = 0.8 #0.8
        self.K_SPEED_FLOOR_PENALTY = -2.0

        self.DIVE_OUTER_BOUNDARY_M = 7000.0  # 奖励开始出现的外部边界
        self.DIVE_INNER_BOUNDARY_M = 3000.0  # 奖励达到最大权重的内部边界
        self.MAX_DIVE_SPEED_MS = 300.0

        self.OPTIMAL_DIVE_ANGLE_DEG = -30.0  # 定义我们认为最理想的俯冲角
        self.DIVE_ANGLE_WIDTH_DEG = 30.0  # 定义奖励曲线的“宽度”，越大越宽容

        self.COORDINATION_BONUS_FACTOR = 0.5

        # --- 状态变量 ---
        self.prev_missile_v_mag = None
        self.prev_los_vec = None
        self.prev_velocity_vec = None
        self.MAX_G_LIMIT = -5.0  # 假设飞机的结构极限

        # --- 为 Tau 加速度奖励新增的状态变量 ---
        self.prev_tau = None  # 上一时间步的Tau值
        self.prev_delta_tau = None  # 上一时间步的Tau变化量 (delta_tau)
        # 在 RewardCalculator 类的 __init__ 方法中添加：
        self.prev_ata_rad = None
        # <<< 新增 >>> 为新的奖励函数创建历史状态变量
        self.prev_taa_rad = None
        # 为此函数添加一个历史状态变量
        self.prev_closing_velocity = None

        # <<< 新增：为干扰弹投放奖励定义参数 >>>
        self.FLARE_EFFECTIVE_DISTANCE_M = 3000.0  # 干扰弹有效投放距离阈值 (3km)
        self.REWARD_FLARE_IN_WINDOW = 0.5  # 在有效窗口内投放的奖励值
        self.PENALTY_FLARE_OUT_WINDOW = -1.0 #-0.2  # 在窗口外投放的惩罚值
        # <<< 新增结束 >>>

        # <<< 新增：为高G机动奖励定义参数 >>>
        self.AIRCRAFT_MAX_G = 9.0  # 飞机的最大正G力过载
        self.HIGH_G_SENSITIVITY = 4.0  # 奖励函数的敏感度，值越大，低G奖励越少，高G奖励越多
        # <<< 新增结束 >>>

        # <<< 新增：为分离加速度奖励创建历史状态变量 >>>
        self.prev_separation_velocity = None

    def reset(self):
            """为新回合重置状态变量。"""
            self.prev_missile_v_mag = None
            self.prev_los_vec = None
            # --- 重置 Tau 加速度奖励的状态 ---
            self.prev_tau = None
            self.prev_delta_tau = None
            # 在 RewardCalculator 类的 reset 方法中添加：
            self.prev_ata_rad = None
            # <<< 新增 >>> 在每回合开始时重置
            self.prev_taa_rad = None
            # 为此函数添加一个历史状态变量
            self.prev_closing_velocity = None
            # 注意：持续滚转的计时器现在不在这个类里，因为它依赖于环境的dt，
            # 最好由主环境管理。或者在这里接收dt进行更新。为简化，我们先假设它在主环境中。

            # <<< 新增：重置历史状态变量 >>>
            self.prev_separation_velocity = None

    # --- (中文) 稀疏奖励接口 ---
    def get_sparse_reward(self, miss_distance: float, R_kill: float) -> float:
        """
        计算最终的事件奖励。
        这个方法精确地复制了您原来的 compute_sparse_reward_1 逻辑。
        """
        if miss_distance > R_kill:
            return self.W
        else:
            return self.U

    # --- (中文) 密集奖励接口 ---
    def calculate_dense_reward(self, aircraft: Aircraft, missile: Missile,
                               remaining_flares: int, total_flares: int, action: dict) -> float: # <<< 更改 >>> 明确 action 是一个字典
        """
        计算并返回当前时间步的总密集奖励。
        这是从主环境调用的唯一接口。
        """
        # --- <<< 核心修正 >>> ---
        # 从新的动作字典中提取投放决策
        # 'discrete_actions' 的第一个元素 (索引0) 是 'flare_trigger'
        flare_trigger_action = action['discrete_actions'][0]

        # 1. 计算所有独立的奖励/惩罚组件
        reward_posture = 1.0 * self._compute_missile_posture_reward_blend(aircraft,missile) #尾后和三九

        # reward_posture = 1.0 * self._compute_missile_posture_reward_pure_posture(missile, aircraft) # 纯尾后
        reward_altitude = 0.5 * self._compute_altitude_reward(aircraft)  #高度惩罚阶跃
        # <<< 核心修正 >>> ---
        # 将提取出的 flare_trigger_action 传递给资源惩罚函数
        reward_resource = 0.2 * self._compute_resource_penalty(flare_trigger_action, remaining_flares, total_flares) #感觉0.2有点大，智能体很小心的投放
        reward_roll_penalty = 0.5 * self._penalty_for_roll_rate_magnitude(aircraft)
        # reward_speed_penalty = 0.5 * self._penalty_for_dropping_below_speed_floor(aircraft)  #1.0速度惩罚还可以再大点感觉，或者加一个速度奖励
        # reward_survivaltime = 0.2  # 每步存活奖励
        # reward_los = self._reward_for_los_rate(aircraft, missile, 0.2)  # LOS变化率奖励
        # reward_dive = self._reward_for_tactical_dive_smooth(aircraft, missile)
        reward_coordinated_turn = self._reward_for_coordinated_turn(aircraft, 0.2)
        # reward_dive = self._reward_for_optimal_dive_angle(aircraft, missile)
        # reward_dive = 0.0

        # # 核心威胁降低奖励
        # w_increase_tau = 4.0  # [核心] 增加命中时间是首要目标
        # # reward_increase_tau = w_increase_tau * self._reward_for_increasing_tau(aircraft, missile)
        # # 使用新的 tau 加速度奖励
        # reward_tau_accel = self._reward_for_tau_acceleration(aircraft, missile)

        # [新] 使用您指定的 ATA Rate 奖励
        w_ata_rate = 1.0  # [新] 为 ATA Rate 设置一个高权重
        reward_ata_rate = w_ata_rate * self._reward_for_ata_rate(aircraft, missile, 0.2)

        # [新] 使用 TAA Rate 奖励 (基于飞机速度矢量)
        # w_taa_rate = 1.0 #0.6 #1.0  # [新] 为 TAA Rate 设置一个高权重
        # reward_taa_rate = w_taa_rate * self._reward_for_taa_rate(aircraft, missile, 0.2)

        # #接近速度奖励
        # reward_closing_velocity = 0.5 * self._reward_for_closing_velocity_change(aircraft, missile, dt = 0.2)

        # <<< 新增：调用干扰弹时机奖励函数 >>>
        # 将新奖励的权重也在这里设置，例如 1.0
        reward_flare_timing = 1.0 * self._compute_flare_timing_reward(flare_trigger_action, aircraft, missile)

        # <<< 新增：调用高G机动奖励函数 >>>
        # 在这里设置新奖励的权重，例如 0.5
        reward_high_g = 1.0 * self._reward_for_high_g_maneuver(aircraft)

        # 在主函数 calculate_dense_reward 中
        # reward_ata_rate = self._reward_for_ata_rate(...) # 假设范围是 [0, 1]
        # reward_high_g_base = self._reward_for_high_g_maneuver(...) # 假设范围是 [0, 1]
        # 最终奖励 = 基础G力奖励 * 机动有效性
        final_high_g_reward = 0.5 * reward_high_g * reward_ata_rate

        # # <<< 新增 >>> 调用新的分离速度奖励函数，权重可以设高一些，因为它代表了核心生存策略
        # reward_separation = 1.0 * self._reward_for_separation_velocity(aircraft, missile)
        # <<< 核心修改：只使用分离加速度奖励 >>>
        # 这个奖励直接鼓励“加速”这个行为
        reward_separation_accel = 0.5 * self._reward_for_separation_acceleration(aircraft, missile)

        # 2. 将所有组件按权重加权求和 (权重直接在此处定义，与您的代码一致)
        final_dense_reward = (
                reward_posture +
                reward_altitude +
                reward_resource +
                reward_roll_penalty +  # 惩罚项权重应为负数, reward_F_roll_penalty基准是正的
                # reward_speed_penalty  # reward_for_optimal_speed基准是负的
                # + reward_survivaltime
                # + reward_los
                # + reward_dive
                + reward_coordinated_turn
                # + reward_increase_tau
                # + reward_tau_accel
                # + reward_closing_velocity
                #     + reward_separation
            + reward_separation_accel   # <<< 已替换 >>>
                # + reward_ata_rate
                # + reward_taa_rate
                + reward_flare_timing
                # + reward_high_g  # <<< 新增：将高G奖励加入总和 >>>
                +final_high_g_reward
        )
        # print(
        #     f"reward_posture: {reward_posture:.2f}",
        #       f"reward_altitude: {reward_altitude:.2f}",
        #       f"reward_resource: {reward_resource:.2f}",
        #       f"reward_roll_penalty: {reward_roll_penalty:.2f}",
        #       # f"reward_speed_penalty: {reward_speed_penalty:.2f}",
        #       #   f"reward_survivaltime: {reward_survivaltime:.2f}",
        #       #   f"reward_los: {reward_los:.2f}",
        #       #   f"reward_dive: {reward_dive:.2f}",
        #         f"reward_coordinated_turn: {reward_coordinated_turn:.2f}",
        #       #   f"reward_increase_tau: {reward_increase_tau:.2f}",
        #       #   f"reward_tau_accel: {reward_tau_accel:.2f}",
        #       #   f"reward_closing_velocity: {reward_closing_velocity:.2f}",
        #     # f"reward_separation: {reward_separation:.2f}",
        #     f"reward_separation_accel: {reward_separation_accel:.2f}",
        #         # f"reward_ata_rate: {reward_ata_rate:.2f}",
        #         # f"reward_taa_rate: {reward_taa_rate:.2f}",
        #     f"reward_flare_timing: {reward_flare_timing:.2f}",
        #     # f"reward_high_g: {reward_high_g:.2f}",
        #     f"final_high_g_reward: {final_high_g_reward:.2f}",
        #       f"final_dense_reward: {final_dense_reward:.2f}")

        return final_dense_reward

    # <<< 核心修改：具备战术情境感知的版本 >>>
    def _reward_for_separation_acceleration(self, aircraft: Aircraft, missile: Missile) -> float:
        """
        (V2 - Context-Aware) 奖励“分离速度”的增加量，但仅在战术上合理时生效。

        核心逻辑：
        1.  [距离门控]: 仅在远距离 (>3km) 生效，权重随距离平滑增加，
                        以避免与近距离的“三九线”战术发生冲突。
        2.  [位置门控]: 仅在飞机处于导弹前半球 (ATA < 90度) 时生效，
                        专注于奖励“逃离当前威胁”的行为。
        3.  [行为奖励]: 在满足以上条件时，奖励分离速度的增加（加速），惩罚减小（减速）。
        """
        # --- 1. 计算基础几何关系 ---
        aircraft_v_vec = aircraft.get_velocity_vector()
        missile_v_vec = missile.get_velocity_vector()
        los_vec_m_to_a = aircraft.pos - missile.pos
        distance = np.linalg.norm(los_vec_m_to_a)

        # --- 2. 距离门控：计算距离权重 ---
        # 和姿态奖励函数保持一致的边界，以确保战术协同
        INNER_BOUNDARY_M = 3000.0
        OUTER_BOUNDARY_M = 4000.0

        distance_weight = 0.0
        if distance > OUTER_BOUNDARY_M:
            distance_weight = 1.0
        elif distance > INNER_BOUNDARY_M:
            # 在 3-4 km 之间，权重从 0 线性增加到 1
            distance_weight = (distance - INNER_BOUNDARY_M) / (OUTER_BOUNDARY_M - INNER_BOUNDARY_M)

        # 如果权重为0 (距离太近)，则此奖励无效，提前返回
        if distance_weight <= 0:
            self.prev_separation_velocity = None  # 重置历史，因为情境已改变
            return 0.0

        # --- 3. 位置门控：检查是否在前半球 ---
        if distance < 1e-6:
            self.prev_separation_velocity = None
            return 0.0

        los_unit_vec = los_vec_m_to_a / distance
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if norm_missile_v > 1e-6:
            cos_ata = np.clip(np.dot(los_unit_vec, missile_v_vec) / norm_missile_v, -1.0, 1.0)
            # 如果飞机在导弹后半球 (cos_ata < 0)，威胁解除，此奖励不适用
            if cos_ata < 0:
                self.prev_separation_velocity = None  # 重置历史
                return 0.0

        # --- 4. 核心逻辑：计算分离加速度奖励 ---
        current_separation_velocity = np.dot(aircraft_v_vec, los_unit_vec)

        reward = 0.0
        if self.prev_separation_velocity is not None:
            delta_v_sep = current_separation_velocity - self.prev_separation_velocity
            reward_base = delta_v_sep

            SENSITIVITY = 0.1
            # 计算基础奖励，范围 [-1, 1]
            reward_scaled = np.tanh(reward_base * SENSITIVITY)

            # 将基础奖励与距离权重相乘
            reward = reward_scaled * distance_weight

        # --- 5. 更新历史状态 ---
        self.prev_separation_velocity = current_separation_velocity

        return reward
    # <<< 新增：分离速度奖励函数 >>>
    def _reward_for_separation_velocity(self, aircraft: Aircraft, missile: Missile) -> float:
        """
        (V1 - 独立版) 专门奖励飞机沿视线方向的分离速度。
        这个奖励直接衡量飞机的逃逸努力，且不受导弹自身减速的影响。

        核心逻辑：
        - 只在远距离 (>3km) 生效，避免与近距离的三九线战术冲突。
        - 在导弹后半球时，给予最大奖励，鼓励维持安全状态。
        - 奖励大小与飞机沿视线远离导弹的速度分量成正比。
        """
        # --- 1. 计算基础几何关系 ---
        aircraft_v_vec = aircraft.get_velocity_vector()
        missile_v_vec = missile.get_velocity_vector()
        los_vec_m_to_a = aircraft.pos - missile.pos
        distance = np.linalg.norm(los_vec_m_to_a)

        # --- 2. 距离门控：只在远距离激活此奖励 ---
        # 定义奖励开始生效的内部边界和完全生效的外部边界
        SEPARATION_REWARD_INNER_BOUNDARY_M = 3000.0
        SEPARATION_REWARD_OUTER_BOUNDARY_M = 4000.0

        if distance < SEPARATION_REWARD_INNER_BOUNDARY_M:
            return 0.0  # 距离太近，不奖励分离速度，优先三九线机动

        # 处理除零风险
        if distance < 1e-6:
            return 0.0

        los_unit_vec = los_vec_m_to_a / distance

        # --- 3. 安全门控：优先进行后半球检查 ---
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if norm_missile_v > 1e-6:
            cos_ata = np.clip(np.dot(los_unit_vec, missile_v_vec) / norm_missile_v, -1.0, 1.0)
            if cos_ata < 0:
                # 飞机在导弹后半球，威胁解除，给予最大奖励
                return 1.0

        # --- 4. 计算核心奖励：分离速度 ---
        # 分离速度 = 飞机速度矢量在视线单位矢量上的投影
        separation_velocity = np.dot(aircraft_v_vec, los_unit_vec)

        # 归一化：用一个合理的飞机最大速度作为基准
        MAX_AIRCRAFT_SPEED_MS = 400.0  # 约 1.2 马赫
        # 我们只奖励正的分离速度（正在远离），如果是负的（正在靠近），奖励为0
        reward_base = np.clip(separation_velocity / MAX_AIRCRAFT_SPEED_MS, 0.0, 1.0)

        # --- 5. 应用平滑的距离权重 ---
        if distance >= SEPARATION_REWARD_OUTER_BOUNDARY_M:
            distance_weight = 1.0
        else:
            # 在 3-4 km 之间，权重从 0 线性增加到 1
            distance_weight = (distance - SEPARATION_REWARD_INNER_BOUNDARY_M) / \
                              (SEPARATION_REWARD_OUTER_BOUNDARY_M - SEPARATION_REWARD_INNER_BOUNDARY_M)

        final_reward = reward_base * distance_weight

        return final_reward

    def _compute_missile_posture_reward_blend(self, aircraft: Aircraft, missile: Missile):
        """
        (最终版) “先置尾再三九”，两者均基于“飞机速度 vs 机弹位置”夹角。
        <<< 核心修改：如果飞机在导弹后半球，直接奖励1.0 >>>
        """
        # --- 1. 计算基础几何关系 ---
        # a) 获取飞机速度矢量
        aircraft_v_vec = aircraft.get_velocity_vector()
        # b) 获取机弹位置矢量 (从导弹指向飞机)
        los_vec_m_to_a = aircraft.pos - missile.pos
        # c) 计算距离
        distance = np.linalg.norm(los_vec_m_to_a)

        # --- 2. <<< 新增：优先进行后半球安全检查 >>> ---
        #    这是最优先的判断。如果飞机已经机动到导弹后方，
        #    视为取得了决定性优势，直接给予最大姿态奖励并返回。
        missile_v_vec = missile.get_velocity_vector()
        norm_los = np.linalg.norm(los_vec_m_to_a)
        norm_missile_v = np.linalg.norm(missile_v_vec)

        if norm_los > 1e-6 and norm_missile_v > 1e-6:
            cos_ata = np.clip(np.dot(los_vec_m_to_a, missile_v_vec) / (norm_los * norm_missile_v), -1.0, 1.0)
            if cos_ata < 0:
                # 如果飞机在导弹后半球 (ATA > 90度)，直接返回 1.0 的高额奖励
                return 1.0
        # --- 后半球检查结束 ---

        # --- 3. 计算“飞机速度矢量”与“机弹位置矢量”夹角的余弦值 ---
        #    (只有在前半球时，才会执行到这里)
        norm_v = np.linalg.norm(aircraft_v_vec)
        if norm_v < 1e-6 or norm_los < 1e-6:
            return 0.0  # 无法计算，返回0

        cos_angle = np.clip(np.dot(aircraft_v_vec, los_vec_m_to_a) / (norm_v * norm_los), -1.0, 1.0)

        # --- 4. 基于同一个cos_angle，计算两种战术奖励 ---

        # a) 远距离“置尾”奖励 (Tail-chase reward)
        #    目标: 夹角为0° (cos=1)，即直接飞离导弹。
        reward_tail_chase = cos_angle

        # b) 近距离“三九线”奖励 (Beaming reward)
        #    目标: 夹角为90° (cos=0)。我们用高斯函数来奖励这个姿态。
        angle_rad = np.arccos(cos_angle)  # 范围 [0, pi]
        angle_deg = np.rad2deg(angle_rad)  # 范围 [0, 180]
        angle_error_deg = abs(angle_deg - 90)  # 与90度的误差
        reward_three_nine = math.exp(-(angle_error_deg ** 2) / (2 * self.ASPECT_REWARD_WIDTH_DEG ** 2))

        # <<< 核心修改：增加俯仰角限制 >>>
        # 1. 获取飞机的俯仰角 (pitch)，并转换为度
        # 通过 attitude_rad 元组的第一个元素获取俯仰角
        aircraft_pitch_deg = np.rad2deg(aircraft.attitude_rad[0])

        # 2. 如果俯仰角小于-70度（即机头朝下超过70度），则取消三九线奖励
        if aircraft_pitch_deg < -self.ASPECT_PITCH_THRESHOLD_DEG:  # 使用超参数 -70.0
            reward_three_nine = 0.0
        # <<< 修改结束 >>>

        # --- 5. 根据距离进行平滑融合 ---
        #    (原有的安全检查逻辑已移到函数开头并被强化)
        if distance <= 3000.0: #2000.0:
            return reward_three_nine
        elif distance >= 4000.0: #3000.0:
            return reward_tail_chase
        else:
            # 在 3-4 km 之间线性混合
            alpha = (distance - 3000.0) / 1000.0  # 0->1
            # 距离越近，越偏向"三九"；距离越远，越偏向"置尾"
            return (1 - alpha) * reward_three_nine + alpha * reward_tail_chase

    # def _compute_missile_posture_reward_blend(self, aircraft: Aircraft, missile: Missile):
    #     """
    #     (最终版) “先置尾再三九”，两者均基于“飞机速度 vs 机弹位置”夹角。
    #     """
    #     # --- 1. 计算基础几何关系 ---
    #     # a) 获取飞机速度矢量
    #     aircraft_v_vec = aircraft.get_velocity_vector()
    #     # b) 获取机弹位置矢量 (从导弹指向飞机)
    #     los_vec_m_to_a = aircraft.pos - missile.pos
    #     # c) 计算距离
    #     distance = np.linalg.norm(los_vec_m_to_a)
    #
    #     # d) 计算“飞机速度矢量”与“机弹位置矢量”夹角的余弦值
    #     norm_v = np.linalg.norm(aircraft_v_vec)
    #     norm_los = np.linalg.norm(los_vec_m_to_a)
    #     if norm_v < 1e-6 or norm_los < 1e-6:
    #         return 0.0  # 无法计算，返回0
    #
    #     cos_angle = np.clip(np.dot(aircraft_v_vec, los_vec_m_to_a) / (norm_v * norm_los), -1.0, 1.0)
    #
    #     # --- 2. 基于同一个cos_angle，计算两种战术奖励 ---
    #
    #     # a) 远距离“置尾”奖励 (Tail-chase reward)
    #     #    目标: 夹角为0° (cos=1)，即直接飞离导弹。
    #     reward_tail_chase = cos_angle
    #
    #     # b) 近距离“三九线”奖励 (Beaming reward)
    #     #    目标: 夹角为90° (cos=0)。我们用高斯函数来奖励这个姿态。
    #     angle_rad = np.arccos(cos_angle)  # 范围 [0, pi]
    #     angle_deg = np.rad2deg(angle_rad)  # 范围 [0, 180]
    #     angle_error_deg = abs(angle_deg - 90)  # 与90度的误差
    #     reward_three_nine = math.exp(-(angle_error_deg ** 2) / (2 * self.ASPECT_REWARD_WIDTH_DEG ** 2))
    #
    #     # --- 3. 应用安全检查 ---
    #     #    如果飞机已在导弹后半球，则取消任何负面惩罚 (仅对"置尾"奖励有效)
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #     if norm_missile_v > 1e-6:
    #         cos_ata = np.dot(los_vec_m_to_a, missile_v_vec) / (norm_los * norm_missile_v)
    #         if cos_ata < 0:
    #             # 如果飞机在导弹后半球 (ATA > 90度)，直接返回 1.0 的高额奖励
    #             return 1.0
    #         # if cos_ata < 0 and reward_tail_chase < 0:
    #         #     reward_tail_chase = 0.0
    #
    #     # --- 4. 根据距离进行平滑融合 ---
    #     if distance <= 2000.0:
    #         return reward_three_nine
    #     elif distance >= 3000.0:
    #         return reward_tail_chase
    #     else:
    #         # 在 2-3 km 之间线性混合
    #         alpha = (distance - 2000.0) / 1000.0  # 0->1
    #         # 距离越近，越偏向"三九"；距离越远，越偏向"置尾"
    #         return (1 - alpha) * reward_three_nine + alpha * reward_tail_chase

    # <<< 新增：高G机动奖励函数 >>>
    def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
        """
        奖励飞机执行高G机动。
        奖励的大小与总法向过载正相关。
        """
        try:
            # 1. 获取当前的法向G力 (nz)
            #    假设：平飞时 nz=1.0, 3G转弯时 nz=3.0
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            print("错误: 您的 Aircraft 类中没有 get_normal_g_force() 方法。")
            return 0.0

        # 2. 我们只奖励正G力，不奖励负G（推杆）机动
        if nz <= 0:
            return 0.0

        # 3. <<< 核心修改：直接用总G力进行归一化 >>>
        #    将当前G力映射到 [0, 1] 区间
        if self.AIRCRAFT_MAX_G <= 1e-6:
            return 0.0  # 避免除零

        normalized_g = nz / self.AIRCRAFT_MAX_G
        normalized_g = np.clip(normalized_g, 0.0, 1.0)

        # 4. 使用 tanh 函数进行平滑塑形
        reward = normalized_g * 1.0
        # reward = np.tanh(normalized_g * 1.0)

        # # 调试打印
        # print(f"NZ: {nz:.2f} G, Norm G: {normalized_g:.2f}, Reward: {reward:.3f}")

        return reward

        # <<< 新增：干扰弹投放时机奖励的计算函数 >>>

    def _compute_flare_timing_reward(self, flare_trigger_action: float, aircraft: Aircraft, missile: Missile) -> float:
        """
        根据飞机与导弹的距离，计算干扰弹投放时机的奖励或惩罚。
        """
        # 1. 检查是否执行了投放动作
        #    我们假设 action > 0.5 意味着投放
        if flare_trigger_action < 0.5:
            return 0.0  # 如果没有投放，则没有时机奖励/惩罚

        # 2. 计算当前飞机与导弹的距离
        distance = np.linalg.norm(aircraft.pos - missile.pos)

        # 3. 根据距离判断并返回相应的奖励或惩罚
        if distance < self.FLARE_EFFECTIVE_DISTANCE_M:
            # 距离在有效窗口内，给予正奖励
            return self.REWARD_FLARE_IN_WINDOW
        else:
            # 距离在有效窗口外，给予负奖励 (惩罚)
            return self.PENALTY_FLARE_OUT_WINDOW

    # <<< 新增结束 >>>

    # --- (中文) 下面是所有从您主环境文件中迁移过来的、正在使用的私有奖励计算方法 ---

    def _reward_for_closing_velocity_change(self, aircraft: Aircraft, missile: Missile, dt: float):
        """
        (V2 - 健壮版) 奖励“接近速度”的减小。
        - 移除了硬编码的dt。
        - 简化了归一化，使用tanh和敏感度参数。
        - (可选) 增加了前半球门控。
        """
        # --- 1. 计算当前“实际”接近速度 ---
        relative_pos_vec = aircraft.pos - missile.pos
        relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
        distance = np.linalg.norm(relative_pos_vec)

        # --- (推荐) 2. 前半球门控 ---
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = np.linalg.norm(missile_v_vec)
        if distance < 1e-6 or norm_missile_v < 1e-6:
            self.prev_closing_velocity = None
            return 0.0

        cos_ata = np.dot(relative_pos_vec, missile_v_vec) / (distance * norm_missile_v)
        if cos_ata < 0:  # 飞机在导弹后半球，威胁解除
            self.prev_closing_velocity = None  # 重置历史
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # return 0.0  # 威胁解除时，不计算此奖励

        # --- 3. 计算接近速度 ---
        # 避免在距离为0时除零
        closing_velocity_current = -np.dot(relative_vel_vec, relative_pos_vec) / (distance + 1e-6)

        # --- 4. 计算奖励 ---
        reward = 0.0
        if self.prev_closing_velocity is not None:
            # a) 计算接近速度的变化量
            delta_V_close = closing_velocity_current - self.prev_closing_velocity

            # b) 核心逻辑：奖励与“接近速度的减小量”成正比
            #    即奖励与 -delta_V_close 成正比
            reward_base = -delta_V_close

            # c) (新) 使用敏感度参数和tanh进行平滑、稳定的缩放
            #    SENSITIVITY决定了多大的速度变化能让奖励饱和。
            #    例如，SENSITIVITY=0.1 意味着 +/- 10 m/s 的速度变化就能让奖励接近 +/- 1.0
            #    这个值需要根据您的环境进行调整。
            SENSITIVITY = 0.1
            reward = np.tanh(reward_base * SENSITIVITY)

        # --- 5. 更新历史状态以备下一步使用 ---
        self.prev_closing_velocity = closing_velocity_current

        if reward < 0.0:
            return 0.0
        else:
            return reward

    # def _reward_for_closing_velocity_change(self, aircraft: Aircraft, missile: Missile):
    #     """
    #     (V10 - 时间变化版)
    #     奖励“接近速度”相对于上一时间步的减小量。
    #     这是一个简单、自适应且有效的奖励塑形方法。
    #     """
    #     # --- 1. 计算当前“实际”接近速度 ---
    #     # a) 相对位置和速度矢量
    #     relative_pos_vec = aircraft.pos - missile.pos
    #     relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
    #
    #     distance = np.linalg.norm(relative_pos_vec)
    #     if distance < 1e-6:
    #         self.prev_closing_velocity = None  # 重置历史
    #         return 0.0
    #
    #     # b) 接近速度 = - (相对速度在视线方向上的投影)
    #     closing_velocity_current = -np.dot(relative_vel_vec, relative_pos_vec) / distance
    #
    #     # --- 2. 计算奖励 ---
    #     reward = 0.0
    #     if self.prev_closing_velocity is not None:
    #         # a) 计算接近速度的变化量
    #         delta_V_close = closing_velocity_current - self.prev_closing_velocity
    #
    #         # b) 核心逻辑：我们奖励“负的变化”，即接近速度的减小
    #         #    所以奖励与 -delta_V_close 成正比
    #         reward_base = -delta_V_close
    #
    #         # if reward_base <= 0:
    #         #     reward = 0.0
    #         # else:
    #         # c) 缩放奖励
    #         #    归一化因子可以是导弹速度 * dt，代表了在一个步长时间内可能的最大速度变化尺度
    #         dt = 0.2  # <--- 关键！需要传入决策步长 dt
    #         normalization_factor = np.linalg.norm(missile.get_velocity_vector()) * dt + 1e-6
    #
    #         normalized_reward = reward_base / normalization_factor
    #
    #         reward = np.tanh(normalized_reward * 5.0)  # 乘以系数并用tanh平滑
    #
    #     # # 调试打印
    #     # if self.prev_closing_velocity is not None:
    #     #     print(
    #     #         f"PrevV_c:{self.prev_closing_velocity:.1f} | CurrV_c:{closing_velocity_current:.1f} | Delta:{delta_V_close:.1f} | Reward:{reward:.3f}")
    #
    #     # --- 3. 更新历史状态以备下一步使用 ---
    #     self.prev_closing_velocity = closing_velocity_current
    #
    #     return reward



    def _reward_for_taa_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """
        (V3 - 基于飞机速度版) 奖励目标姿态角 (TAA) 的变化率。
        这个函数逻辑上与旧的 _reward_for_ata_rate 完全相同，
        但计算的是机弹视线矢量与【飞机速度矢量】之间的夹角变化率。
        """
        # --- 1. 获取视线矢量和【飞机】速度矢量 ---
        current_los_vec = aircraft.pos - missile.pos
        # <<< 核心修改 >>>
        aircraft_v_vec = aircraft.get_velocity_vector_from_jsbsim()

        norm_los = np.linalg.norm(current_los_vec)
        # <<< 核心修改 >>>
        norm_v = np.linalg.norm(aircraft_v_vec)

        if norm_los < 1e-6 or norm_v < 1e-6:
            self.prev_taa_rad = None  # 重置历史，因为当前值无效
            return 0.0

        # --- 2. 计算 TAA 角的余弦值 ---
        # (这里的门控逻辑可以保留，也可以去掉，取决于您的战术意图)
        # 保留它的意义是：只在飞机大致朝向导弹时（前半球）才奖励机动。
        # 去掉它的意义是：无论飞机朝向哪里，只要改变姿态就奖励。
        # 这里我们先按您的要求，保持逻辑不变，所以保留门控。
        cos_taa = np.dot(current_los_vec, aircraft_v_vec) / (norm_los * norm_v)

        # --- 2. <<< 核心修正：增加前半球门控 >>> ---
        # a) 计算 ATA 角的余弦值，用于判断前后半球
        missile_v_vec = missile.get_velocity_vector()
        cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)

        # b) 如果飞机在导弹后半球 (ATA > 90度, cos_ata < 0)，则威胁解除
        if cos_ata < 0:
            self.prev_taa_rad = None  # 重置历史，下次进入前半球时重新计算
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # return 0.0  # 不给予任何奖励
        # --- 修正结束 ---


        # 3. 计算当前的 TAA 角
        # 注意：这个角度现在是 Target Aspect Angle
        current_taa_rad = np.arccos(np.clip(cos_taa, -1.0, 1.0))

        # 4. 计算 TAA 角的变化率
        if self.prev_taa_rad is None:
            self.prev_taa_rad = current_taa_rad
            return 0.0

        delta_taa_rad = abs(current_taa_rad - self.prev_taa_rad)
        taa_rate_rad_s = delta_taa_rad / dt if dt > 1e-6 else 0.0

        # 5. 更新历史记录并返回奖励
        self.prev_taa_rad = current_taa_rad

        # 权重可能需要重新调整，因为 TAA 的变化率可能与 ATA 的变化率尺度不同
        SCALING_FACTOR = 5.0
        return np.tanh(taa_rate_rad_s * SCALING_FACTOR)

    def _reward_for_ata_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """
        (V2 - 增加前半球门控) 奖励天线训练角 (ATA) 的变化率。
        """
        # --- 1. 获取视线矢量和导弹速度矢量 ---
        current_los_vec = aircraft.pos - missile.pos
        missile_v_vec = missile.get_velocity_vector()

        norm_los = np.linalg.norm(current_los_vec)
        norm_v = np.linalg.norm(missile_v_vec)

        if norm_los < 1e-6 or norm_v < 1e-6:
            self.prev_ata_rad = None  # 重置历史，因为当前值无效
            return 0.0

        # --- 2. <<< 核心修正：增加前半球门控 >>> ---
        # a) 计算 ATA 角的余弦值，用于判断前后半球
        cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)

        # b) 如果飞机在导弹后半球 (ATA > 90度, cos_ata < 0)，则威胁解除
        if cos_ata < 0:
            self.prev_ata_rad = None  # 重置历史，下次进入前半球时重新计算
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # return 0.0  # 不给予任何奖励
        # --- 修正结束 ---

        # 3. 计算当前的 ATA 角
        current_ata_rad = np.arccos(np.clip(cos_ata, -1.0, 1.0))

        # 4. 计算 ATA 角的变化率
        if self.prev_ata_rad is None:
            self.prev_ata_rad = current_ata_rad
            return 0.0

        delta_ata_rad = abs(current_ata_rad - self.prev_ata_rad)
        ata_rate_rad_s = delta_ata_rad / dt if dt > 1e-6 else 0.0

        # 5. <<< 核心修改：使用 tanh 函数进行归一化 >>>
        # a) 定义一个敏感度参数。这个参数决定了多大的 ata_rate 能让奖励接近饱和(1.0)。
        #    你需要根据你的环境来调整这个值。
        #    例如，如果 ata_rate 通常在 0.1 rad/s 左右，可以设 SENSITIVITY = 10.0
        SENSITIVITY = 10.0

        normalized_reward = math.tanh(ata_rate_rad_s * SENSITIVITY)

        # 6. 更新历史记录并返回奖励
        self.prev_ata_rad = current_ata_rad

        # SCALING_FACTOR 可以设为1，因为奖励已经被归一化到 [0, 1] 区间了
        # 当然也可以再乘以一个因子来调整权重
        return normalized_reward

        # # 5. 更新历史记录并返回奖励
        # self.prev_ata_rad = current_ata_rad
        #
        # SCALING_FACTOR = 10.0
        # return ata_rate_rad_s * SCALING_FACTOR

    # 新的奖励函数:
    def _reward_for_increasing_tau(self, aircraft: Aircraft, missile: Missile):
        """
        奖励“命中时间（Tau）”的增加。这是衡量威胁降低的核心指标。
        """
        # a) 计算相对位置和速度
        relative_pos_vec = aircraft.pos - missile.pos
        relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()

        distance = np.linalg.norm(relative_pos_vec)

        # --- 2. <<< 核心修正：增加前半球门控 >>> ---
        missile_v_vec = missile.get_velocity_vector()
        norm_v_missile = np.linalg.norm(missile_v_vec)
        if distance < 1e-6 or norm_v_missile < 1e-6:
            self.prev_tau = None
            return 0.0

        cos_ata = np.dot(relative_pos_vec, missile_v_vec) / (distance * norm_v_missile)

        if cos_ata < 0:  # 飞机在后半球
            self.prev_tau = None  # 重置历史
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # 在这种情况下，威胁已经解除，我们可以给予一个固定的、小的正奖励来鼓励维持这种状态
            # 或者直接返回0，避免干扰其他奖励。返回0更安全。
            # return 0.0
            # --- 修正结束 ---

        # b) 计算径向趋近速度 (closing velocity)
        #    正值表示正在靠近，负值表示正在远离
        closing_velocity = -np.dot(relative_vel_vec, relative_pos_vec) / (distance + 1e-6)

        # c) 计算 Tau
        if closing_velocity < 1.0:  # 如果不是在靠近，或速度很慢
            # 视为威胁解除，给予一个大的、固定的奖励
            # 我们可以用一个很大的Tau值来代表这种情况
            current_tau = 100.0
        else:
            current_tau = distance / closing_velocity

        # d) 计算奖励
        reward = 0.0
        if self.prev_tau is not None:
            # 核心逻辑：奖励Tau的增加量
            delta_tau = current_tau - self.prev_tau
            # 使用 tanh 进行平滑缩放，避免奖励爆炸
            reward = np.tanh(delta_tau)

            # e) 更新状态
        self.prev_tau = current_tau

        return reward

    def _compute_missile_posture_reward(self, missile: Missile, aircraft: Aircraft):
        """
                (v2 - 修正版) 计算导弹姿态奖励。
                新增了安全检查：如果导弹已经失锁，则不应有任何惩罚。
                """
        """
        (v3 - 简化失锁判断) 计算导弹姿态奖励。
        使用一个简化的几何条件来判断导弹是否“失锁”，并在此情况下清除负奖励。
        """
        # 1. 获取当前导弹的速度大小 (magnitude)
        missile_v_mag = missile.velocity
        # 2. 如果是第一步，无法计算速度变化，则初始化并返回0
        if self.prev_missile_v_mag is None:
            self.prev_missile_v_mag = missile_v_mag
            return 0.0

        # 3. 计算导弹速度大小的衰减量，并进行缩放
        v_decrease = self.prev_missile_v_mag - missile_v_mag
        v_decrease_scaled = v_decrease * 0.5

        # 4. 获取速度向量
        missile_v_vec = missile.get_velocity_vector()
        aircraft_v_vec = aircraft.get_velocity_vector()

        # 5. 计算速度向量夹角的余弦值
        norm_product = np.linalg.norm(missile_v_vec) * np.linalg.norm(aircraft_v_vec) + 1e-6
        angle_cos = np.dot(missile_v_vec, aircraft_v_vec) / norm_product
        angle_cos = np.clip(angle_cos, -1.0, 1.0)

        # 6. 根据原始逻辑计算基础奖励 (保持不变)
        reward = 0
        if angle_cos < 0:  # 迎头或对冲态势
            reward = angle_cos / (max(v_decrease_scaled, 0) + 1)
        else:  # 尾追或同向态势
            reward = angle_cos * max(v_decrease_scaled, 0)

        # --- (中文) 7. 核心修正：使用简化的几何条件进行安全检查 ---

        # a) 计算导弹到飞机的视线矢量
        los_vec_m_to_a = aircraft.pos[0:3] - missile.pos[0:3]

        # b) 计算视线矢量与导弹速度矢量的夹角余弦值
        #    这个夹角被称为 "Antenna Train Angle" (ATA) 或 "Angle-Off"
        norm_product_ata = np.linalg.norm(los_vec_m_to_a) * np.linalg.norm(missile_v_vec) + 1e-6
        cos_ata = np.dot(los_vec_m_to_a, missile_v_vec) / norm_product_ata
        cos_ata = np.clip(cos_ata, -1.0, 1.0)

        # c) 判断飞机是否在导弹的后半球 (夹角 > 90度)
        #    如果夹角 > 90度，那么其余弦值 cos_ata 会是负数。
        is_aircraft_behind_missile = cos_ata < 0

        # d) 如果飞机已经在导弹的后半球 (视为失锁)，并且计算出的奖励是负的，则强制改为0
        if is_aircraft_behind_missile and reward < 0:
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # reward = 0.0

        # 8. 更新上一步的速度 (原步骤7)
        self.prev_missile_v_mag = missile_v_mag

        # 9. 返回最终计算出的奖励值 (原步骤8)
        return reward * 1.0

    def _compute_missile_posture_reward_pure_posture(self, missile: Missile, aircraft: Aircraft):
        """
        (v5 - 纯姿态版) 计算导弹姿态奖励。
        这个版本移除了所有关于导弹自身速度衰减的考量，
        奖励完全由导弹与飞机速度矢量的夹角决定。
        同时，保留了关键的“失锁”安全检查。
        """
        # 1. 获取速度向量
        missile_v_vec = missile.get_velocity_vector()
        aircraft_v_vec = aircraft.get_velocity_vector()

        # 2. 计算速度向量夹角的余弦值 (核心姿态指标)
        norm_product = np.linalg.norm(missile_v_vec) * np.linalg.norm(aircraft_v_vec) + 1e-6
        angle_cos = np.dot(missile_v_vec, aircraft_v_vec) / norm_product
        angle_cos = np.clip(angle_cos, -1.0, 1.0)

        # 3. 基于姿态计算奖励
        # 简化逻辑：直接使用速度夹角余弦值作为奖励基础。
        # - angle_cos > 0 (尾追/同向): 奖励为正，对准越好(越接近1)奖励越高。
        # - angle_cos < 0 (迎头/对冲): 奖励为负 (即惩罚)，对准越好(越接近-1)惩罚越大。
        # 注意：迎头情况下的惩罚逻辑可能需要根据您的具体任务目标进行微调。
        reward = angle_cos

        # --- 4. 核心修正：使用简化的几何条件进行安全检查 ---

        # a) 计算导弹到飞机的视线矢量
        los_vec_m_to_a = aircraft.pos[0:3] - missile.pos[0:3]

        # b) 计算视线矢量与导弹速度矢量的夹角余弦值 (ATA)
        norm_product_ata = np.linalg.norm(los_vec_m_to_a) * np.linalg.norm(missile_v_vec) + 1e-6
        cos_ata = np.dot(los_vec_m_to_a, missile_v_vec) / norm_product_ata
        cos_ata = np.clip(cos_ata, -1.0, 1.0)

        # c) 判断飞机是否在导弹的后半球 (视为失锁)
        is_aircraft_behind_missile = cos_ata < 0

        # d) 如果视为失锁，并且计算出的奖励是负的，则强制改为0
        if is_aircraft_behind_missile and reward < 0:
            reward = 0.0

        # 5. 返回最终计算出的奖励值
        return reward * 1.0

    # def _compute_missile_posture_reward_blend(self, missile: Missile, aircraft: Aircraft):
    #     """
    #     (v6 - 平滑融合版)
    #     距离阈值：
    #         d <= 4 km : 仅使用三九奖励
    #         d >= 5 km : 仅使用纯姿态奖励
    #         4 km < d < 5 km : 两者线性混合
    #     """
    #     # 0. 距离计算
    #     los_vec_m_to_a = aircraft.pos[0:3] - missile.pos[0:3]
    #     distance = np.linalg.norm(los_vec_m_to_a)
    #
    #     # --- 三九奖励 ---
    #     reward_three_nine = self._reward_for_aspect_angle(aircraft, missile)
    #
    #     # --- 纯姿态奖励 ---
    #     missile_v_vec = missile.get_velocity_vector()
    #     aircraft_v_vec = aircraft.get_velocity_vector()
    #     norm_product = np.linalg.norm(missile_v_vec) * np.linalg.norm(aircraft_v_vec) + 1e-6
    #     angle_cos = np.dot(missile_v_vec, aircraft_v_vec) / norm_product
    #     angle_cos = np.clip(angle_cos, -1.0, 1.0)
    #     reward_pure = angle_cos
    #
    #     # ATA失锁检查
    #     norm_product_ata = np.linalg.norm(los_vec_m_to_a) * np.linalg.norm(missile_v_vec) + 1e-6
    #     cos_ata = np.dot(los_vec_m_to_a, missile_v_vec) / norm_product_ata
    #     cos_ata = np.clip(cos_ata, -1.0, 1.0)
    #     if cos_ata < 0 and reward_pure < 0:
    #         reward_pure = 0.0
    #
    #     # --- 平滑融合 ---
    #     if distance <= 2000.0: #4000.0:
    #         return reward_three_nine
    #     elif distance >= 3000.0: #5000.0:
    #         return reward_pure
    #     else:
    #         # 在 4-5 km 之间线性混合
    #         alpha = (distance - 2000.0) / 1000.0  # 0~1
    #         return (1 - alpha) * reward_three_nine + alpha * reward_pure

    def _compute_altitude_reward(self, aircraft: Aircraft):
        """
        计算高度奖励（实际为惩罚）。
        惩罚危险的低空飞行和不经济的超高空飞行。
        """
        # 1. 获取飞机当前的高度 (y坐标) 和垂直速度
        altitude_m = aircraft.pos[1]
        v_vertical_ms = aircraft.get_velocity_vector()[1]
        # --- 低空惩罚逻辑  ---
        # 2. 计算速度惩罚 (Pv)
        Pv = 0.0
        if altitude_m <= self.SAFE_ALTITUDE_M and v_vertical_ms < 0:
            descent_speed = abs(v_vertical_ms)
            penalty_factor = (descent_speed / self.KV_MS) * ((self.SAFE_ALTITUDE_M - altitude_m) / self.SAFE_ALTITUDE_M)
            Pv = -np.clip(penalty_factor, 0.0, 1.0)
        # 3. 计算绝对高度惩罚 (PH)
        PH = 0.0
        if altitude_m <= self.DANGER_ALTITUDE_M:
            PH = np.clip(altitude_m / self.DANGER_ALTITUDE_M, 0.0, 1.0) - 1.0
        # --- 新增：超高空惩罚逻辑 ---
        # 4. 计算超高空惩罚 (P_over)
        P_over = 0.0
        if altitude_m > self.MAX_ALTITUDE_M:
            # 惩罚值与超出高度成正比
            # 例如，在13000米时，惩罚为 -(13000-12000)/1000 * 0.5 = -0.5
            # 这样设计可以平滑地增加惩罚
            P_over = -((altitude_m - self.MAX_ALTITUDE_M) / 1000.0) * self.OVER_ALTITUDE_PENALTY_FACTOR
        # 5. 合并所有惩罚项，并乘以全局缩放系数
        #    将低空惩罚和高空惩罚相加

        return Pv + PH + P_over

    def _compute_resource_penalty(self, release_flare_action, remaining_flares, total_flares):
        """干扰资源使用过量惩罚函数"""
        if release_flare_action > 0.5:
            # (中文) 您原始代码中的 alphaR 和 k1
            alphaR = 2.0
            k1 = 3.0
            fraction_remaining = remaining_flares / total_flares
            penalty = -alphaR * (1 + k1 * (1 - fraction_remaining))
            return penalty
        return 0.0

    def _reward_for_aspect_angle(self, aircraft: Aircraft, missile: Missile):
        """(组件A - v5版，增加了对垂直机动的抑制)"""
        current_R_rel = np.linalg.norm(aircraft.pos - missile.pos)
        # distance_weight = 1.0 - np.clip(current_R_rel / self.ASPECT_REWARD_EFFECTIVE_RANGE, 0.0, 1.0)
        distance_weight = 1.0
        if distance_weight <= 0.0: return 0.0
        # --- (中文) 核心修正：只有在非极端俯仰角时才计算该奖励 ---# 定义一个俯仰角阈值，例如70度。只有当飞机不那么“垂直”时，三九线奖励才有效。
        pitch_rad = aircraft.state_vector[4]
        if np.rad2deg(pitch_rad) > self.ASPECT_PITCH_THRESHOLD_DEG: return 0.0  # (中文) 如果飞机太垂直，直接返回0，不给任何三九线奖励

        threat_angle_rad = self._compute_relative_beta(aircraft.state_vector, missile.state_vector)
        threat_angle_deg = np.rad2deg(threat_angle_rad)

        is_on_right_side = (threat_angle_deg > 0) and (threat_angle_deg < 180)
        if is_on_right_side:
            angle_error_deg = abs(threat_angle_deg - 90)
        else:
            angle_error_deg = abs(threat_angle_deg - 270)

        base_reward = math.exp(-(angle_error_deg ** 2) / (2 * self.ASPECT_REWARD_WIDTH_DEG ** 2))
        return base_reward * distance_weight

    def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
        """
        (最终版 - 无阈值) 直接惩罚任何非零的滚转角速度。
        惩罚的大小与滚转角速度的绝对值成正比。
        """
        # 从状态向量中获取当前的实际滚转角速度
        p_real_rad_s = aircraft.roll_rate_rad_s
        if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6: return 0.0
        # if np.rad2deg(abs(p_real_rad_s)) < 120 : return 0.0
        # 将当前的滚转速率归一化到 [0, 1] 范围
        #    (当前速率 / 最大速率)
        #    防止除零错误
        normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
        return (normalized_roll_rate) * -1.0  # 返回基准值[0,1]

    def _penalty_for_dropping_below_speed_floor(self, aircraft: Aircraft):
        """
        (速度下限惩罚模型) 只有当飞机速度低于最优机动速度下限时，才施加惩罚。
        速度越高，越没有惩罚。
        """
        # 计算当前马赫数
        Vt = aircraft.velocity
        speed_of_sound = 340.0
        current_mach = Vt / speed_of_sound
        # 检查速度是否低于了下限
        if current_mach < self.OPTIMAL_SPEED_FLOOR_MACH:
            # a) 计算低于下限的差距
            mach_deficit = self.OPTIMAL_SPEED_FLOOR_MACH - current_mach
            # b) (推荐) 对差距进行归一化，使其尺度可控
            #    分母就是我们的下限本身。
            #    例如，如果速度掉到 0.35 马赫，差距是 0.35，归一化后就是 0.5。
            normalized_deficit = mach_deficit / self.OPTIMAL_SPEED_FLOOR_MACH
            # c) 计算最终惩罚
            #    可以使用线性或二次惩罚。二次惩罚 (平方) 会让惩罚随着速度的降低而急剧增加，
            #    这能更强烈地阻止AI进入危险的低速区。
            penalty_base = normalized_deficit  # 您的代码使用的是二次惩罚
            return self.K_SPEED_FLOOR_PENALTY * penalty_base
        return 0.0

    def _compute_relative_beta(self, x_target, x_missile):
        """
       (v3 - 修正版) 计算导弹相对于飞机机头的方位角 (0-2pi)。
       该版本使用坐标系旋转和 arctan2，比旧的 arccos+cross_product 方法更健壮。
       """
        # 1. 获取飞机的偏航角 (从北向东为正)
        psi_t = x_target[5]
        # 2. 计算从世界坐标系 (NUE) 旋转到以飞机机头为前方的参考系所需要的旋转矩阵
        # 这是一个绕 y 轴 (天轴) 的二维旋转
        cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)  # 需要旋转 -psi_t 才能让机头朝向新的 x' 轴
        # 旋转矩阵
        # R = [[cos, -sin],
        #      [sin,  cos]]
        # 我们只关心水平面 xz (北-东)
        R_vec = x_missile[3:6] - x_target[0:3]
        # 3. 计算飞机指向导弹的相对位置矢量，并投影到水平面
        R_proj_world = np.array([R_vec[0], R_vec[2]])  # 水平面上的 (x, z) 分量
        # 4. 将这个相对位置矢量旋转到以飞机为参考的坐标系下
        # [x_rel_body] = [cos, -sin] * [x_rel_world]
        # [z_rel_body]   [sin,  cos]   [z_rel_world]
        x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
        z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
        # 5. 使用 arctan2 直接计算角度
        # 在这个新的参考系下：
        # x_rel_body > 0 表示在前半球, < 0 表示在后半球
        # z_rel_body > 0 表示在右半侧(东), < 0 表示在左半侧(西)
        # arctan2(z, x) 会给出从 x 轴正方向 (机头) 到该向量的角度
        threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
        # 6. 将结果从 [-pi, pi] 转换为 [0, 2pi]
        if threat_angle_rad < 0:
            threat_angle_rad += 2 * np.pi
        # print("threat_angle_rad:", np.rad2deg(threat_angle_rad))

        return threat_angle_rad

    # 新的奖励函数:
    def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """
        (V2 - 修正版) 奖励视线矢量的角速度。
        新增了安全门控：只在飞机处于导弹前半球时才计算奖励。
        """
        # 1. 计算当前的视线矢量 (从导弹指向飞机)
        current_los_vec = aircraft.pos - missile.pos
        norm_current = np.linalg.norm(current_los_vec)

        # --- 核心修正：增加前向扇区门控 ---
        # a) 获取导弹的速度矢量
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = np.linalg.norm(missile_v_vec)

        # b) 避免除零错误
        if norm_current < 1e-6 or norm_missile_v < 1e-6:
            # 如果距离过近或导弹静止，则不计算奖励
            self.prev_los_vec = current_los_vec  # 别忘了更新状态
            return 0.0

        # c) 计算ATA角的余弦值
        # cos(ATA) = (LOS · V_missile) / (|LOS| * |V_missile|)
        cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)

        # d) 如果飞机在导弹后半球 (ATA > 90度, cos(ATA) < 0)，则奖励为0
        if cos_ata < 0:
            self.prev_los_vec = current_los_vec  # 仍然需要更新状态以备下一帧
            return 1.0  # <<< MODIFIED >>> 给予最大奖励
            # return 0.0
        # --- 修正结束 ---

        # 如果是第一步，则初始化并返回0
        if self.prev_los_vec is None:
            self.prev_los_vec = current_los_vec
            return 0.0

        # 2. 计算角速度 (原逻辑)
        norm_prev = np.linalg.norm(self.prev_los_vec)
        if norm_prev < 1e-6:
            self.prev_los_vec = current_los_vec
            return 0.0

        current_los_unit_vec = current_los_vec / norm_current
        prev_los_unit_vec = self.prev_los_vec / norm_prev

        dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        los_angular_rate_rad_s = angle_rad / dt if dt > 1e-6 else 0.0

        # 5. 更新历史记录
        self.prev_los_vec = current_los_vec

        # 6. 返回缩放后的奖励值
        return los_angular_rate_rad_s * 0.5

    def _reward_for_tactical_dive_smooth(self, aircraft: Aircraft, missile: Missile):
        """
        (平滑版) 在导弹接近时，奖励向下的俯冲机动。
        奖励在内外边界之间平滑增长。
        """
        distance = np.linalg.norm(aircraft.pos - missile.pos)

        # 1. 计算平滑的距离权重
        if distance > self.DIVE_OUTER_BOUNDARY_M:
            distance_weight = 0.0
        elif distance < self.DIVE_INNER_BOUNDARY_M:
            distance_weight = 1.0
        else:
            # 在 [DIVE_INNER_BOUNDARY_M, DIVE_OUTER_BOUNDARY_M] 区间内线性插值
            # 公式: 1 - (当前值 - 下限) / (上限 - 下限)
            distance_weight = 1.0 - (distance - self.DIVE_INNER_BOUNDARY_M) / (
                        self.DIVE_OUTER_BOUNDARY_M - self.DIVE_INNER_BOUNDARY_M)

        if distance_weight <= 0:
            return 0.0

        # 2. 获取并检查垂直速度
        vertical_velocity = aircraft.get_velocity_vector()[1]
        if vertical_velocity >= 0:
            return 0.0  # 只有俯冲时才奖励

        # 3. 计算基础的速度奖励
        dive_speed = -vertical_velocity
        normalized_speed_reward = np.clip(dive_speed / self.MAX_DIVE_SPEED_MS, 0, 1.0)

        # 4. 最终奖励 = 速度奖励 * 平滑的距离权重
        reward = normalized_speed_reward * distance_weight
        return reward

    def _reward_for_optimal_dive_angle(self, aircraft: Aircraft, missile: Missile):
        """
        (V2 - 角度优化版) 奖励飞机以一个最优角度进行战术俯冲。
        """
        distance = np.linalg.norm(aircraft.pos - missile.pos)

        # 1. 计算平滑的距离权重 (这部分逻辑保持不变)
        if distance > self.DIVE_OUTER_BOUNDARY_M:
            distance_weight = 0.0
        elif distance < self.DIVE_INNER_BOUNDARY_M:
            distance_weight = 1.0
        else:
            distance_weight = 1.0 - (distance - self.DIVE_INNER_BOUNDARY_M) / (
                    self.DIVE_OUTER_BOUNDARY_M - self.DIVE_INNER_BOUNDARY_M)

        if distance_weight <= 0:
            return 0.0

        # 2. 获取飞机俯冲角并计算角度奖励因子
        try:
            pitch_rad = aircraft.state_vector[4]  # 获取俯仰角 theta
            pitch_deg = np.rad2deg(pitch_rad)
        except IndexError:
            return 0.0

        # a) 只在俯冲时（俯仰角为负）才计算奖励
        if pitch_deg >= 0:
            return 0.0

        # b) 计算当前角度与最优角度的差距
        angle_error_deg = abs(pitch_deg - self.OPTIMAL_DIVE_ANGLE_DEG)

        # c) 使用高斯函数计算角度因子
        #    当 angle_error_deg = 0 时, 因子为 exp(0) = 1 (最大)
        #    当 angle_error_deg 变大时, 因子平滑下降
        angle_factor = math.exp(-(angle_error_deg ** 2) / (2 * self.DIVE_ANGLE_WIDTH_DEG ** 2))

        # 3. 最终奖励 = 角度因子 * 平滑的距离权重
        reward = angle_factor * distance_weight
        # print(f"Distance: {distance:.1f} m, Weight: {distance_weight:.2f}, Pitch: {pitch_deg:.1f} deg, AngleErr: {angle_error_deg:.1f} deg, AngleF: {angle_factor:.2f}, Reward: {reward:.3f}")
        return reward

        # 这是一个辅助函数，如果您的飞机类没有提供旋转矩阵，可以用这个
        # 您需要把它放在 RewardCalculator 类的内部

    def _get_rotation_matrix_from_euler(self, roll, pitch, yaw):
        """从欧拉角（phi, theta, psi）创建从机体到世界的旋转矩阵。"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # ZYX 顺序 (Yaw-Pitch-Roll)
        R = np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp]
        ])
        return R

    def _reward_for_coordinated_turn(self, aircraft: Aircraft, dt: float):
        """
        (V5 - 最终版) 直接从JSBSim读取nz，只奖励正确的拉杆转弯。
        这个版本简单、直接且不会出错。
        """
        # # 1. 计算滚转因子 (请务必最后一次确认滚转角的正确索引，很可能是[3]！)
        # try:
        #     roll_rad = aircraft.state_vector[6]  # <--- 滚转角 phi
        # except IndexError:
        #     return 0.0
        #
        # roll_factor = math.sin(abs(roll_rad))
        # if abs(np.rad2deg(roll_rad)) > 90:
        #     roll_factor = math.sin(math.pi - abs(roll_rad))
        # roll_factor = np.clip(roll_factor, 0, 1.0)

        # 2. 直接从飞机对象获取法向G力 (nz)
        try:
            nz = aircraft.get_normal_g_force()
        except AttributeError:
            print("错误: 您的 Aircraft 类中没有 get_normal_g_force() 方法。请先实现它。")
            return 0.0

        # 3. 根据JSBSim的符号约定计算G力因子
        #    平飞: nz = -1.0 G -> positive_g = 1.0 G
        #    2G转弯: nz = -2.0 G -> positive_g = 2.0 G
        positive_g = nz

        # G力因子是超过1G的部分
        g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
        g_factor = np.clip(g_factor, 0, 1.0)

        # 4. 最终奖励
        # reward = -1.0 * roll_factor * g_factor
        reward = -1.0  * g_factor

        # 调试打印
        # print(f"RollF: {roll_factor:.2f} | NZ: {nz:.2f} | G_F: {g_factor:.2f} | Reward: {reward:.3f}")

        return reward

    def _reward_for_tau_acceleration(self, aircraft: Aircraft, missile: Missile):
        """
        (V2 - 加速度版) 奖励“命中时间（Tau）”下降趋势的减缓。
        核心逻辑：奖励 Tau 变化率的“加速度”。如果 Tau 的下降变慢了，就给予正奖励。
        """
        # --- 步骤 1: 计算当前的 Tau 值 ---

        # a) 计算相对位置和速度矢量
        relative_pos_vec = aircraft.pos - missile.pos
        relative_vel_vec = aircraft.get_velocity_vector() - missile.get_velocity_vector()
        distance = np.linalg.norm(relative_pos_vec)

        # b) 安全检查 & 前半球门控 (与您之前的函数逻辑一致)
        missile_v_vec = missile.get_velocity_vector()
        norm_v_missile = np.linalg.norm(missile_v_vec)
        if distance < 1e-6 or norm_v_missile < 1e-6:
            # 状态无效，重置所有历史记录并返回0
            self.prev_tau = None
            self.prev_delta_tau = None
            return 0.0

        cos_ata = np.dot(relative_pos_vec, missile_v_vec) / (distance * norm_v_missile)
        if cos_ata < 0:  # 飞机在导弹后半球，威胁解除
            self.prev_tau = None
            self.prev_delta_tau = None
            # 威胁解除时，这是一个非常好的状态，可以给一个小的正奖励
            # 也可以返回0，让其他奖励函数（如生存奖励）来处理
            return 0.0

            # c) 计算径向趋近速度
        closing_velocity = -np.dot(relative_vel_vec, relative_pos_vec) / (distance + 1e-6)

        # d) 计算当前 Tau
        if closing_velocity < 1.0:  # 如果正在远离或速度很慢
            current_tau = 100.0  # 使用一个大的常数代表“安全”
        else:
            current_tau = distance / closing_velocity

        # --- 步骤 2: 计算奖励 ---

        # a) 如果没有上一步的 Tau 值，则无法计算变化量
        if self.prev_tau is None:
            self.prev_tau = current_tau
            # self.prev_delta_tau 保持为 None
            return 0.0

        # b) 计算当前步的 Tau 变化量
        delta_tau = current_tau - self.prev_tau

        # c) 如果没有上一步的 Tau 变化量，则无法计算“加速度”
        if self.prev_delta_tau is None:
            self.prev_tau = current_tau
            self.prev_delta_tau = delta_tau
            return 0.0

        # d) 核心逻辑：计算 Tau 的“加速度”
        #    tau_acceleration > 0 意味着 Tau 的下降趋势正在减缓，或者正在转为上升
        tau_acceleration = delta_tau - self.prev_delta_tau

        # e) 使用 tanh 进行平滑缩放
        #    需要一个缩放因子，因为 tau_acceleration 的值通常很小
        ACCELERATION_SCALING_FACTOR = 10.0
        reward = np.tanh(tau_acceleration * ACCELERATION_SCALING_FACTOR)

        # --- 步骤 3: 更新所有历史状态以备下一步使用 ---
        self.prev_tau = current_tau
        self.prev_delta_tau = delta_tau

        return reward
