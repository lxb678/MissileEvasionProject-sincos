# 文件: reward_system.py (已修正，精确匹配您的代码)

import numpy as np
import math

# (中文) 导入对象类，以便进行类型提示和访问状态
from .AircraftJSBSim_DirectControl import Aircraft
from .missile2 import Missile


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
        self.SAFE_ALTITUDE_M = 3000.0
        self.DANGER_ALTITUDE_M = 2000.0 #1500.0
        self.KV_MS = 50.0 #0.2 * 340
        self.MAX_ALTITUDE_M = 10000.0 #12000.0
        self.OVER_ALTITUDE_PENALTY_FACTOR = 0.5

        # [三九线奖励参数]
        self.ASPECT_REWARD_EFFECTIVE_RANGE = 10000.0 #5000.0
        self.ASPECT_REWARD_WIDTH_DEG = 45.0
        self.ASPECT_PITCH_THRESHOLD_DEG = 70.0

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
        # reward_posture = 1.0 * self._compute_missile_posture_reward_blend(aircraft,missile) #尾后和三九

        # reward_posture = 1.0 * self._compute_missile_posture_reward_pure_posture(missile, aircraft) # 纯尾后
        reward_altitude = 0.5 * self._compute_altitude_reward(aircraft)  #高度惩罚阶跃
        # <<< 核心修正 >>> ---
        # 将提取出的 flare_trigger_action 传递给资源惩罚函数
        reward_resource = 1.0 * self._compute_resource_penalty(flare_trigger_action, remaining_flares, total_flares) #感觉0.2有点大，智能体很小心的投放
        reward_roll_penalty = 0.5 * self._penalty_for_roll_rate_magnitude(aircraft)  #1.14修改，原来是0.5  1.15号，感觉0.1好像不太行
        # reward_speed_penalty = 0.5 * self._penalty_for_dropping_below_speed_floor(aircraft)  #1.0速度惩罚还可以再大点感觉，或者加一个速度奖励
        reward_survivaltime = 0.5 #0.2  # 每步存活奖励
        # reward_los = self._reward_for_los_rate(aircraft, missile, 0.2)  # LOS变化率奖励
        # reward_dive = self._reward_for_tactical_dive_smooth(aircraft, missile)
        # reward_coordinated_turn = 0.5 * self._reward_for_coordinated_turn(aircraft, 0.2)
        reward_punish_push_down = 0.5 * self._reward_for_punish_push_down(aircraft)
        # reward_dive = self._reward_for_optimal_dive_angle(aircraft, missile)
        # reward_dive = 0.0

        # # 核心威胁降低奖励
        # w_increase_tau = 4.0  # [核心] 增加命中时间是首要目标
        # # reward_increase_tau = w_increase_tau * self._reward_for_increasing_tau(aircraft, missile)
        # # 使用新的 tau 加速度奖励
        # reward_tau_accel = self._reward_for_tau_acceleration(aircraft, missile)

        # [新] 使用您指定的 ATA Rate 奖励
        w_ata_rate = 1.0 #1.0 # [新] 为 ATA Rate 设置一个高权重
        # reward_ata_rate = w_ata_rate * self._reward_for_los_rate(aircraft, missile, 0.2)
        reward_los_rate = w_ata_rate * self._reward_for_los_rate(aircraft, missile, 0.2)

        # [新] 使用 TAA Rate 奖励 (基于飞机速度矢量)
        # w_taa_rate = 1.0 #0.6 #1.0  # [新] 为 TAA Rate 设置一个高权重
        # reward_taa_rate = w_taa_rate * self._reward_for_taa_rate(aircraft, missile, 0.2)

        # #接近速度奖励
        # reward_closing_velocity = 0.5 * self._reward_for_closing_velocity_change(aircraft, missile, dt = 0.2)

        # <<< 新增：调用干扰弹时机奖励函数 >>>
        # 将新奖励的权重也在这里设置，例如 1.0
        # reward_flare_timing = 1.0 * self._compute_flare_timing_reward(flare_trigger_action, aircraft, missile)

        # <<< 新增：调用高G机动奖励函数 >>>
        # 在这里设置新奖励的权重，例如 0.5
        reward_high_g = 1.0 * self._reward_for_high_g_maneuver(aircraft)

        # 在主函数 calculate_dense_reward 中
        # reward_ata_rate = self._reward_for_ata_rate(...) # 假设范围是 [0, 1]
        # reward_high_g_base = self._reward_for_high_g_maneuver(...) # 假设范围是 [0, 1]
        # 最终奖励 = 基础G力奖励 * 机动有效性
        final_high_g_reward = 1.0 * reward_high_g * reward_los_rate  #1.8号修改，原来是0.5

        # # <<< 新增 >>> 调用新的分离速度奖励函数，权重可以设高一些，因为它代表了核心生存策略
        # reward_separation = 1.0 * self._reward_for_separation_velocity(aircraft, missile)
        # <<< 核心修改：只使用分离加速度奖励 >>>
        # 这个奖励直接鼓励“加速”这个行为
        # reward_separation_accel = 0.5 * self._reward_for_separation_acceleration(aircraft, missile)

        # reward_speed = 0.5 * self._reward_for_maintaining_speed(aircraft)

        # reward_head_on_penalty = 0.5 * self._compute_missile_head_on_penalty(aircraft, missile)  #迎头惩罚

        # 2. 将所有组件按权重加权求和 (权重直接在此处定义，与您的代码一致)
        final_dense_reward = (
                # reward_posture +    #去掉看效果
            # reward_head_on_penalty + #迎头惩罚
                reward_altitude +
                reward_resource +
                reward_roll_penalty +  # 惩罚项权重应为负数, reward_F_roll_penalty基准是正的   #去掉看效果   还是很有必要的
                # reward_speed_penalty  # reward_for_optimal_speed基准是负的   #速度暂时不需要了
                + reward_survivaltime
                # + reward_los
                # + reward_dive
                # + reward_coordinated_turn   #去掉看效果   还有很有必要的
                + reward_punish_push_down
                # + reward_increase_tau
                # + reward_tau_accel
                # + reward_closing_velocity
                #     + reward_separation
            # + reward_separation_accel   # <<< 已替换 >>>
            #     + reward_speed    #去掉看效果
            #     + reward_los_rate  #添加los奖励
                # + reward_taa_rate
                # + reward_flare_timing  #去掉看效果
                # + reward_high_g  # <<< 新增：将高G奖励加入总和 >>>
                + final_high_g_reward
        )
        # print(
        #     #f"reward_posture: {reward_posture:.2f}",
        #      #f"reward_head_on_penalty: {reward_head_on_penalty:.2f}",
        #       f"reward_altitude: {reward_altitude:.2f}",
        #       f"reward_resource: {reward_resource:.2f}",
        #       f"reward_roll_penalty: {reward_roll_penalty:.2f}",
        #       # f"reward_speed_penalty: {reward_speed_penalty:.2f}",
        #       #   f"reward_survivaltime: {reward_survivaltime:.2f}",
        #       #   f"reward_los: {reward_los:.2f}",
        #       #   f"reward_dive: {reward_dive:.2f}",
        #       #   f"reward_coordinated_turn: {reward_coordinated_turn:.2f}",
        #     f"reward_punish_push_down: {reward_punish_push_down:.2f}",
        #       #   f"reward_increase_tau: {reward_increase_tau:.2f}",
        #       #   f"reward_tau_accel: {reward_tau_accel:.2f}",
        #       #   f"reward_closing_velocity: {reward_closing_velocity:.2f}",
        #     # f"reward_separation: {reward_separation:.2f}",
        #     #f"reward_separation_accel: {reward_separation_accel:.2f}",
        #     f"reward_los_rate: {reward_los_rate:.2f}",
        #     # f"reward_ata_rate: {reward_ata_rate:.2f}",
        #         # f"reward_taa_rate: {reward_taa_rate:.2f}",
        #     #f"reward_flare_timing: {reward_flare_timing:.2f}",
        #     f"reward_high_g: {reward_high_g:.2f}",
        #     # f"reward_speed: {reward_speed:.2f}",
        #     f"final_high_g_reward: {final_high_g_reward:.2f}",
        #       f"final_dense_reward: {final_dense_reward:.2f}")

        return final_dense_reward

    def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
        """
        (V3 - 解析法 + 线性截断) 奖励视线矢量的角速度。
        使用物理公式 |r x v| / r^2 直接计算，无需历史状态。
        """
        # 1. 基础向量获取
        r_vec = aircraft.pos - missile.pos
        # 相对速度 = 飞机速度 - 导弹速度
        v_rel = aircraft.get_velocity_vector() - missile.get_velocity_vector()

        # 距离平方 (r^2)
        r_sq = np.dot(r_vec, r_vec)
        r_norm = np.sqrt(r_sq)

        # 2. 基础数据准备 & 门控
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = np.linalg.norm(missile_v_vec)

        # 避免除零错误 (距离过近或导弹静止)
        if r_sq < 1.0 or norm_missile_v < 0.1:
            return 0.0

        # --- 核心门控：后半球直接给满分 (1.0) ---
        # 计算 ATA 余弦
        cos_ata = np.dot(r_vec, missile_v_vec) / (r_norm * norm_missile_v)

        if cos_ata < 0:
            return 1.0  # 安全状态，威胁解除

        # --- 3. 解析法核心计算 ---
        # 公式: Omega = |r x v_rel| / r^2
        # 计算叉积
        cross_prod = np.cross(r_vec, v_rel)
        # 计算叉积的模长
        cross_norm = np.linalg.norm(cross_prod)

        # 得到瞬时视线角速率 (rad/s)
        los_rate = cross_norm / r_sq

        # --- 4. 线性截断归一化 ---
        # 设定目标阈值 (0.2 rad/s ≈ 11.5 deg/s)
        TARGET_LOS_RATE = 0.2

        # 线性映射
        normalized_reward = los_rate #/ TARGET_LOS_RATE

        # 截断到 [0, 1]
        # reward = los_rate #np.clip(normalized_reward, 0.0, 1.0)
        reward = np.clip(normalized_reward, 0.0, 1.0)

        return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 截断归一化版) 奖励视线矢量的角速度。
    #     使用线性映射 + 截断 (Clip)。
    #     """
    #     # 0. dt 安全保护
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # 1. 计算当前的视线矢量
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # 2. 基础数据准备 & 门控
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     # 避免除零
    #     if norm_current < 1.0 or norm_missile_v < 0.1:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # --- 核心门控：后半球直接给满分 (1.0) ---
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #
    #     if cos_ata < 0:
    #         self.prev_los_vec = current_los_vec
    #         return 1.0  # 安全状态
    #
    #     # 初始化历史
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 3. 计算角速度
    #     prev_norm = np.linalg.norm(self.prev_los_vec)
    #     if prev_norm < 1.0:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = self.prev_los_vec / prev_norm
    #
    #     # 计算夹角
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     # 计算角速率 (rad/s)
    #     los_angular_rate_rad_s = angle_rad / safe_dt
    #
    #     # 4. 更新历史
    #     self.prev_los_vec = current_los_vec
    #
    #     # --- 5. 线性截断核心 ---
    #     # 设定一个满分阈值。
    #     # 0.2 rad/s 约为 11.5度/秒。
    #     # 这是一个非常剧烈的机动，足以造成导弹脱锁或能量耗尽。
    #     TARGET_LOS_RATE = 0.2
    #
    #     # 计算比例： rate / 0.2
    #     normalized_reward = los_angular_rate_rad_s #/ TARGET_LOS_RATE
    #
    #     # 截断到 [0, 1] 之间
    #     reward = np.clip(normalized_reward, 0.0, 1.0)
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 修正版) 奖励视线矢量的角速度。
    #     包含安全门控：飞机在导弹后半球时直接给 1.0。
    #     """
    #     # 0. dt 安全保护
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # 1. 计算当前的视线矢量 (从导弹指向飞机)
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # --- 核心修正：增加前向扇区门控 ---
    #     # a) 获取导弹的速度矢量
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     # b) 避免除零错误 (距离过近或速度异常)
    #     if norm_current < 1.0 or norm_missile_v < 0.1:
    #         self.prev_los_vec = current_los_vec  # 更新状态
    #         return 0.0
    #
    #     # c) 计算ATA角的余弦值
    #     # cos(ATA) = (LOS · V_missile) / (|LOS| * |V_missile|)
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #
    #     # d) 如果飞机在导弹后半球 (ATA > 90度, cos(ATA) < 0)，则奖励为 1.0
    #     if cos_ata < 0:
    #         self.prev_los_vec = current_los_vec  # 仍然需要更新状态以备下一帧
    #         return 1.0  # <<< MODIFIED >>> 给予最大奖励
    #
    #     # 如果是第一步，则初始化并返回0
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 2. 计算角速度
    #     prev_norm = np.linalg.norm(self.prev_los_vec)
    #     if prev_norm < 1.0:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = self.prev_los_vec / prev_norm
    #
    #     # 使用 clip 防止浮点误差导致 arccos 报错
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     # 计算角速率
    #     los_angular_rate_rad_s = angle_rad / safe_dt
    #
    #     # 5. 更新历史记录
    #     self.prev_los_vec = current_los_vec
    #
    #     # # --- 5. 归一化核心 ---
    #     # # 设置敏感度系数。
    #     # # 系数越大，越容易拿满分；系数越小，奖励增长越平缓。
    #     # # 设为 5.0 时：
    #     # #   rate = 0.1 rad/s (约6度/秒) -> reward ≈ 0.46
    #     # #   rate = 0.2 rad/s (约11度/秒) -> reward ≈ 0.76
    #     # #   rate > 0.5 rad/s -> reward ≈ 0.99 (接近满分)
    #     # SENSITIVITY = 1.0
    #     #
    #     # reward = math.tanh(los_angular_rate_rad_s * SENSITIVITY)
    #     #
    #     # return reward
    #
    #     # 6. 返回缩放后的奖励值 (您设定的系数 0.5)
    #     return los_angular_rate_rad_s #* 0.5

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 切向速度最终版) 奖励视线角速率 (LOS Rate)。
    #
    #     核心原理：
    #     不计算几何角度差，而是计算【切向速度】(Tangential Velocity)。
    #     切向速度越大 => 视线转动越快 => 导弹必须拉更大的过载。
    #
    #     优点：
    #     1. 信号平滑：没有 1/r^2 的奇点，没有 dt 导致的数值爆炸。
    #     2. 因果明确：直接奖励飞机产生的“横向逃逸分量”，不受导弹修正动作的干扰。
    #     3. 无状态：不需要维护 prev_los_vec，代码更干净。
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
    #     # --- 2. 前半球安全门控 (保留你之前的逻辑) ---
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
    #             # 这里不需要更新 prev_los_vec 了，因为这个版本是无状态的
    #             return 1.0
    #
    #             # --- 3. 计算切向速度 (核心逻辑) ---
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
    #     radial_velocity_component = np.dot(aircraft_v_vec, los_unit_vec) * los_unit_vec
    #     transverse_velocity_vec = aircraft_v_vec - radial_velocity_component
    #
    #     transverse_speed = np.linalg.norm(transverse_velocity_vec)
    #
    #     # --- 4. 计算奖励 ---
    #     # 归一化方法 A (推荐):
    #     # 奖励 sin(Aspect Angle)。
    #     # 当飞机垂直于视线飞行时 (Beaming/Notching)，transverse_speed 接近 aircraft_speed，奖励接近 1.0
    #     # 当飞机迎头或尾追时，transverse_speed 接近 0，奖励接近 0.0
    #     reward = transverse_speed / aircraft_speed
    #
    #     # 归一化方法 B (如果你希望同时鼓励飞机保持高速):
    #     # 使用一个参考最大速度 (例如 340m/s) 进行归一化
    #     # reward = np.clip(transverse_speed / 340.0, 0.0, 1.0)
    #
    #     # 这里使用方法 A，因为它纯粹衡量“几何机动效率”，
    #     # 速度的大小已经由单独的 `reward_speed` 或 `reward_energy` 来管理了。
    #     return reward

    # def _reward_for_los_rate(self, aircraft: "Aircraft", missile: "Missile", dt: float) -> float:
    #     """
    #     (V4.1 - 最小改动稳定版) 奖励 LOS(视线矢量) 的旋转速率，抑制刷分与抖动。
    #     相比 V4 的最小改动点：
    #       1) 后半球奖励改为“一次性 bonus”（latch），避免每步刷 1.0
    #       2) 夹角用 atan2(||u×v||, u·v) 替代 arccos，更数值稳定（小角度更线性）
    #       3) los_rate 做上限裁剪（rate_cap），抑制高频抖动/数值噪声刷分
    #       4) prev_los_vec 异常时重置，避免坏历史污染
    #     依赖的成员变量（若没有会自动创建/使用）：
    #       - self.prev_los_vec: np.ndarray 或 None
    #       - self._los_safe_latch: bool（记录上一帧是否已处于后半球安全态）
    #     """
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
    #     # 数值保护：距离过近或速度异常时不计算
    #     if norm_los < 1.0 or norm_v < 0.1:
    #         self.prev_los_vec = None
    #         # 进入异常态，安全 latch 也清掉（防止卡住）
    #         if hasattr(self, "_los_safe_latch"):
    #             self._los_safe_latch = False
    #         return 0.0
    #
    #     # -----------------------------
    #     # 2) 门控：后半球一次性奖励（避免每步刷满分）
    #     # -----------------------------
    #     # dot > 0: 前半球(威胁)； dot < 0: 后半球(相对安全)
    #     cos_ata = float(np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v))
    #
    #     if not hasattr(self, "_los_safe_latch"):
    #         self._los_safe_latch = False
    #
    #     # if cos_ata < 0.0:
    #     #     # 一次性 bonus：只有“刚进入后半球”的那个 step 给
    #     #     bonus = 1.0 if (self._los_safe_latch is False) else 0.0
    #     #     self._los_safe_latch = True
    #     #     self.prev_los_vec = None  # 重置历史，避免下一步夹角乱跳
    #     #     return float(bonus)
    #     # else:
    #     #     # 回到前半球（仍有威胁）
    #     #     self._los_safe_latch = False
    #
    #     # SAFE_REWARD = 0.2  # 每步小奖励：鼓励“保持在后半球”
    #     # if cos_ata < 0.0:
    #     #     bonus = 1.0 if (self._los_safe_latch is False) else 0.0
    #     #     self._los_safe_latch = True
    #     #     self.prev_los_vec = None
    #     #     return float(bonus + SAFE_REWARD)
    #     # else:
    #     #     self._los_safe_latch = False
    #
    #     if cos_ata < 0.0:
    #         self.prev_los_vec = None  # 重置
    #         return 1.2  # 巨大奖励：成功甩掉导弹
    #
    #     # -----------------------------
    #     # 3) 初始化历史
    #     # -----------------------------
    #     if getattr(self, "prev_los_vec", None) is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     prev_norm = float(np.linalg.norm(self.prev_los_vec))
    #     if prev_norm < 1.0:
    #         # 坏历史，重置
    #         self.prev_los_vec = None
    #         return 0.0
    #
    #     # -----------------------------
    #     # 4) 计算 LOS 单位向量夹角（更稳的 atan2 版本）
    #     # -----------------------------
    #     current_u = current_los_vec / norm_los
    #     prev_u = self.prev_los_vec / prev_norm
    #
    #     dot_uv = float(np.clip(np.dot(current_u, prev_u), -1.0, 1.0))
    #     cross_norm = float(np.linalg.norm(np.cross(prev_u, current_u)))
    #     delta_angle_rad = float(np.arctan2(cross_norm, dot_uv))  # [0, pi]
    #
    #     # 更新历史
    #     self.prev_los_vec = current_los_vec
    #
    #     # -----------------------------
    #     # 5) LOS rate + 抖动抑制（裁剪上限）
    #     # -----------------------------
    #     los_rate = delta_angle_rad / safe_dt
    #
    #     # rate_cap：抑制高频抖动/噪声刷分
    #     # 你可以按 dt 与场景调：0.5~3.0 rad/s 常见
    #     rate_cap = 2.0
    #     los_rate = min(los_rate, rate_cap)
    #
    #     # -----------------------------
    #     # 6) 奖励映射（避免过早饱和，可稍降灵敏度）
    #     # -----------------------------
    #     # 如果你 dt=0.02s，且 delta_angle 常在 0.005~0.02rad，
    #     # los_rate 就可能在 0.25~1.0rad/s；此时 10 会比 15 更不容易全饱和。
    #     SENSITIVITY = 1.0
    #
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #     return float(reward)

    # def _reward_for_los_rate_aligned(self, aircraft, missile, dt: float) -> float:
    #     # 0) dt保护
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # 1) 相对位置
    #     r = aircraft.pos - missile.pos
    #     Rx, Ry, Rz = float(r[0]), float(r[1]), float(r[2])
    #     R = float(np.linalg.norm(r))
    #     if R < 1.0:
    #         self.prev_theta_L = None
    #         self.prev_phi_L = None
    #         return 0.0
    #
    #     # 2) LOS角（更稳的 atan2 定义）
    #     Rh = float(np.hypot(Rx, Rz))
    #     theta_L = float(np.arctan2(Ry, max(Rh, 1e-9)))
    #     phi_L = float(np.arctan2(Rz, Rx))
    #
    #     # 3) 后半球门控（你可以继续用一次性 latch 逻辑）
    #     v_m = missile.get_velocity_vector()
    #     norm_v = float(np.linalg.norm(v_m))
    #     if norm_v > 1e-6:
    #         cos_ata = float(np.dot(r, v_m) / (R * norm_v))
    #         if cos_ata < 0.0:
    #             self.prev_theta_L = None
    #             self.prev_phi_L = None
    #             return 1.0  # 建议你还是用一次性 latch
    #
    #     # 4) 差分计算 theta_dot / phi_dot
    #     if getattr(self, "prev_theta_L", None) is None or getattr(self, "prev_phi_L", None) is None:
    #         self.prev_theta_L = theta_L
    #         self.prev_phi_L = phi_L
    #         return 0.0
    #
    #     theta_L_dot = (theta_L - self.prev_theta_L) / safe_dt
    #     dphi = np.arctan2(np.sin(phi_L - self.prev_phi_L), np.cos(phi_L - self.prev_phi_L))
    #     phi_L_dot = float(dphi) / safe_dt
    #
    #     self.prev_theta_L = theta_L
    #     self.prev_phi_L = phi_L
    #
    #     # 5) 等效 LOS rate（对齐导引律输入）
    #     omega_los = float(np.sqrt(theta_L_dot ** 2 + (phi_L_dot * np.cos(theta_L)) ** 2))
    #
    #     # 6) 映射为奖励（你可沿用 tanh）
    #     SENSITIVITY = 10.0
    #     return float(math.tanh(omega_los * SENSITIVITY))

    # def _reward_for_los_rate(self, aircraft, missile, dt: float) -> float:
    #     """
    #     (V4.2 - 解析法 / 导引律对齐版)
    #     奖励“等效 LOS 角速率 omega_LOS”，与 PN / OGL 的真实驱动量一致。
    #
    #     保留的稳定性改进：
    #       1) 后半球一次性奖励（latch），避免刷分
    #       2) 解析法计算 LOS rate（不使用差分，不放大噪声）
    #       3) rate 上限裁剪，抑制高频抖动
    #       4) tanh 映射，奖励有界
    #     """
    #
    #     # -----------------------------
    #     # 0) dt 安全（解析法理论上不依赖 dt，但用于逻辑统一）
    #     # -----------------------------
    #     safe_dt = max(float(dt), 1e-4)
    #
    #     # -----------------------------
    #     # 1) 相对位置 / 速度
    #     # -----------------------------
    #     r = aircraft.pos - missile.pos
    #     Rx, Ry, Rz = float(r[0]), float(r[1]), float(r[2])
    #     R = float(np.linalg.norm(r))
    #
    #     if R < 1.0:
    #         if hasattr(self, "_los_safe_latch"):
    #             self._los_safe_latch = False
    #         return 0.0
    #
    #     v_m = missile.get_velocity_vector()
    #     v_a = aircraft.get_velocity_vector()
    #     v_rel = v_a - v_m
    #
    #     Vx_rel, Vy_rel, Vz_rel = float(v_rel[0]), float(v_rel[1]), float(v_rel[2])
    #
    #     # -----------------------------
    #     # 2) 后半球门控（一次性奖励）
    #     # -----------------------------
    #     norm_v = float(np.linalg.norm(v_m))
    #     if not hasattr(self, "_los_safe_latch"):
    #         self._los_safe_latch = False
    #
    #     if norm_v > 1e-6:
    #         cos_ata = float(np.dot(r, v_m) / (R * norm_v))
    #         if cos_ata < 0.0:
    #             bonus = 1.0 if (self._los_safe_latch is False) else 0.0
    #             self._los_safe_latch = True
    #             return float(bonus)
    #         else:
    #             self._los_safe_latch = False
    #
    #     # -----------------------------
    #     # 3) 解析法 LOS 角速率
    #     # -----------------------------
    #     Rh_sq = Rx * Rx + Rz * Rz
    #     Rh = np.sqrt(Rh_sq)
    #     R_sq = Rh_sq + Ry * Ry
    #
    #     # --- 偏航 LOS rate φ_dot ---
    #     if Rh_sq < 1e-9:
    #         phi_L_dot = 0.0
    #     else:
    #         phi_L_dot = (Rx * Vz_rel - Rz * Vx_rel) / Rh_sq
    #
    #     # --- 俯仰 LOS rate θ_dot ---
    #     if Rh < 1e-9:
    #         theta_L_dot = 0.0
    #     else:
    #         Rh_dot = (Rx * Vx_rel + Rz * Vz_rel) / Rh
    #         theta_L_dot = (Rh * Vy_rel - Ry * Rh_dot) / (R_sq + 1e-9)
    #
    #     # -----------------------------
    #     # 4) 等效 LOS rate（对齐 PN / OGL）
    #     # -----------------------------
    #     theta_L = np.arctan2(Ry, max(Rh, 1e-9))
    #     omega_los = np.sqrt(theta_L_dot ** 2 +
    #                         (phi_L_dot * np.cos(theta_L)) ** 2)
    #
    #     # -----------------------------
    #     # 5) 抖动抑制：rate 上限裁剪
    #     # -----------------------------
    #     rate_cap = 2.0  # rad/s，可按场景微调
    #     omega_los = min(float(omega_los), rate_cap)
    #
    #     # -----------------------------
    #     # 6) 奖励映射
    #     # -----------------------------
    #     SENSITIVITY = 10.0
    #     reward = math.tanh(omega_los * SENSITIVITY)
    #
    #     return float(reward)

    # def _reward_for_los_rate(self, aircraft, missile, dt: float) -> float:
    #     """
    #     (V5 - 终极解析矢量版)
    #     利用叉乘原理直接计算 LOS 角速率。
    #     数学原理：Omega = |r x v| / |r|^2
    #     优势：
    #       1. 包含 V4.2 的所有物理精度（无滞后，瞬时值）。
    #       2. 消除 V4.2 的球坐标奇点风险（无需处理头顶/脚下情况）。
    #       3. 计算开销最小。
    #     """
    #
    #     # -----------------------------
    #     # 1) 相对向量
    #     # -----------------------------
    #     r_vec = aircraft.pos - missile.pos
    #     # R squared (距离的平方)
    #     r_sq = float(np.dot(r_vec, r_vec))
    #
    #     # 保护：防止距离过近除零 (撞击或重合)
    #     if r_sq < 1.0:
    #         if hasattr(self, "_los_safe_latch"):
    #             self._los_safe_latch = False
    #         return 0.0
    #
    #     v_m = missile.get_velocity_vector()
    #     v_a = aircraft.get_velocity_vector()
    #     # 相对速度：注意是被减数 (飞机 - 导弹) 还是 (导弹 - 飞机)
    #     # 对计算模长没影响，但物理上通常指 r_dot
    #     v_rel = v_a - v_m
    #
    #     # -----------------------------
    #     # 2) 后半球门控 (Latch)
    #     # -----------------------------
    #     # 使用 r 和 v_m 判断 ATA
    #     norm_v = float(np.linalg.norm(v_m))
    #     norm_r = math.sqrt(r_sq)
    #
    #     if not hasattr(self, "_los_safe_latch"):
    #         self._los_safe_latch = False
    #
    #     if norm_v > 0.1:
    #         # cos_ata = (r . v_m) / (|r|*|v_m|)
    #         cos_ata = float(np.dot(r_vec, v_m) / (norm_r * norm_v))
    #
    #         if cos_ata < 0.0:
    #             bonus = 1.0 if (self._los_safe_latch is False) else 0.0
    #             self._los_safe_latch = True
    #             return float(bonus)
    #         else:
    #             self._los_safe_latch = False
    #
    #     # -----------------------------
    #     # 3) 核心：解析法计算 LOS Rate
    #     # -----------------------------
    #     # 公式：omega = |r x v| / r^2
    #     # 物理意义：提取垂直于视线的速度分量，除以距离
    #     cross_prod = np.cross(r_vec, v_rel)
    #     cross_norm = float(np.linalg.norm(cross_prod))
    #
    #     omega_los = cross_norm / r_sq
    #
    #     # -----------------------------
    #     # 4) 抑制与映射
    #     # -----------------------------
    #     # 截断上限 (2.0 rad/s)
    #     omega_los = min(omega_los, 2.0)
    #
    #     # Tanh 映射
    #     SENSITIVITY = 1.0
    #     reward = math.tanh(omega_los * SENSITIVITY)
    #
    #     return float(reward)


    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 修正版) 奖励视线矢量的角速度。
    #     新增了安全门控：只在飞机处于导弹前半球时才计算奖励。
    #     """
    #     # 1. 计算当前的视线矢量 (从导弹指向飞机)
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # --- 核心修正：增加前向扇区门控 ---
    #     # a) 获取导弹的速度矢量
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     # b) 避免除零错误
    #     if norm_current < 1e-6 or norm_missile_v < 1e-6:
    #         # 如果距离过近或导弹静止，则不计算奖励
    #         self.prev_los_vec = current_los_vec  # 别忘了更新状态
    #         return 0.0
    #
    #     # c) 计算ATA角的余弦值
    #     # cos(ATA) = (LOS · V_missile) / (|LOS| * |V_missile|)
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #
    #     # d) 如果飞机在导弹后半球 (ATA > 90度, cos(ATA) < 0)，则奖励为0
    #     if cos_ata < 0:
    #         self.prev_los_vec = current_los_vec  # 仍然需要更新状态以备下一帧
    #         return 1.0  # <<< MODIFIED >>> 给予最大奖励
    #         # return 0.0
    #     # --- 修正结束 ---
    #
    #     # 如果是第一步，则初始化并返回0
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 2. 计算角速度 (原逻辑)
    #     norm_prev = np.linalg.norm(self.prev_los_vec)
    #     if norm_prev < 1e-6:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = self.prev_los_vec / norm_prev
    #
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     los_angular_rate_rad_s = angle_rad / dt if dt > 1e-6 else 0.0
    #
    #     # 5. 更新历史记录
    #     self.prev_los_vec = current_los_vec
    #
    #     # 6. 返回缩放后的奖励值
    #     return los_angular_rate_rad_s * 0.5

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V4 - 融合终极版) 奖励视线矢量 (LOS) 的旋转速率。
    #     逻辑核心：最大化 LOS Rate 以迫使导弹过载饱和。
    #     实现细节：采用 V3 的数值保护和敏感度缩放。
    #     """
    #     # --- 1. 获取基础向量 ---
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_los = np.linalg.norm(current_los_vec)
    #
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_v = np.linalg.norm(missile_v_vec)
    #
    #     # [数值保护] 距离过近或速度异常时不计算
    #     if norm_los < 1.0 or norm_v < 0.1:
    #         self.prev_los_vec = None
    #         return 0.0
    #
    #     # --- 2. 门控逻辑：判断威胁是否解除 ---
    #     # 计算 ATA 余弦：判断飞机是否在导弹屁股后面
    #     # dot > 0: 前半球(威胁)； dot < 0: 后半球(安全)
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)
    #
    #     if cos_ata < 0:
    #         self.prev_los_vec = None  # 重置
    #         return 1.0  # 巨大奖励：成功甩掉导弹
    #
    #     # --- 3. 初始化历史 ---
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # --- 4. 计算 LOS 矢量的几何旋转角 (V2 核心) ---
    #     # 即使导弹对得准，只要飞机横向跑得快，LOS就在转，这就是我们想要的
    #     prev_norm = np.linalg.norm(self.prev_los_vec)
    #     if prev_norm < 1.0:
    #         return 0.0
    #
    #     # 归一化向量进行点积
    #     current_u = current_los_vec / norm_los
    #     prev_u = self.prev_los_vec / prev_norm
    #
    #     # 计算夹角 (使用 clip 防止浮点误差导致的 NaN)
    #     # dot_product = 1.0 代表没动，angle = 0
    #     dot_product = np.clip(np.dot(current_u, prev_u), -1.0, 1.0)
    #     delta_angle_rad = np.arccos(dot_product)
    #
    #     # 更新历史
    #     self.prev_los_vec = current_los_vec
    #
    #     # --- 5. 计算奖励 (V3 的健壮性) ---
    #     safe_dt = max(dt, 1e-4)
    #     los_rate = delta_angle_rad / safe_dt
    #
    #     # [参数调整]
    #     # LOS rate 通常在 0.01 ~ 0.5 之间。
    #     # 我们希望当 rate = 0.1 rad/s (约5.7度/秒) 时，能拿到不错的奖励。
    #     # tanh(0.1 * 10) = tanh(1.0) ≈ 0.76 (不错)
    #     # tanh(0.1 * 20) = tanh(2.0) ≈ 0.96 (接近满分)
    #     SENSITIVITY = 15.0
    #
    #     # 奖励总是非负的，因为任何方向的 LOS 旋转都是对导弹的消耗
    #     reward = math.tanh(los_rate * SENSITIVITY)
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V3 - 优化版) 奖励天线训练角 (ATA) 的变化率。
    #     """
    #     # --- 1. 获取视线矢量和导弹速度矢量 ---
    #     current_los_vec = aircraft.pos - missile.pos
    #     missile_v_vec = missile.get_velocity_vector()
    #
    #     norm_los = np.linalg.norm(current_los_vec)
    #     norm_v = np.linalg.norm(missile_v_vec)
    #
    #     # [优化1] 数值保护：增加极小值防止除零
    #     if norm_los < 1e-1 or norm_v < 1e-1:
    #         self.prev_ata_rad = None
    #         return 0.0
    #
    #     # --- 2. 前半球门控 ---
    #     # 计算 ATA 角的余弦值
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_los * norm_v)
    #
    #     # 如果飞机在导弹后半球 (ATA > 90度, cos_ata < 0)，则威胁解除
    #     # [逻辑确认] 这表示导弹正在飞离飞机，给予最大奖励鼓励保持
    #     if cos_ata < 0:
    #         self.prev_ata_rad = None  # 重置历史
    #         return 1.0
    #
    #     # 3. 计算当前的 ATA 角
    #     current_ata_rad = np.arccos(np.clip(cos_ata, -1.0, 1.0))
    #
    #     # 4. 初始化历史记录
    #     if self.prev_ata_rad is None:
    #         self.prev_ata_rad = current_ata_rad
    #         return 0.0
    #
    #     # --- 3. 计算变化率 ---
    #     delta_ata_rad = current_ata_rad - self.prev_ata_rad
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
    #         # 理由：ATA的变化率通常很小 (0.0x ~ 0.x rad/s)。
    #         # 如果系数是 1.0，tanh(0.1) ≈ 0.1，奖励太小，AI 学得慢。
    #         # 建议设为 10.0 到 20.0，让良好的规避动作能拿到接近 1.0 的奖励。
    #         SENSITIVITY = 10.0
    #         reward = math.tanh(rate * SENSITIVITY)
    #     else:
    #         # ATA 正在减小 (Situation Worsening)
    #         # 保持为 0.0，不惩罚，避免AI因为无法改变导弹物理性能而陷入局部最优（不敢动）
    #         reward = 0.0
    #
    #     # 5. 更新历史记录
    #     self.prev_ata_rad = current_ata_rad
    #
    #     return reward

    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 几何修正版) 奖励视线矢量的角速度。
    #     使用几何点积计算实际转动角度，并使用 tanh 限制最大奖励幅度。
    #     """
    #     # 1. 计算当前的视线矢量
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # --- 门控逻辑 ---
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     if norm_current < 1e-6 or norm_missile_v < 1e-6:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 判断是否在导弹后方 (安全区)
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #     if cos_ata < 0:
    #         self.prev_los_vec = current_los_vec
    #         return 1.0  # 威胁解除，给满分
    #
    #     # 初始化历史
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 2. 计算几何角度变化
    #     norm_prev = np.linalg.norm(self.prev_los_vec)
    #     if norm_prev < 1e-6:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = self.prev_los_vec / norm_prev
    #
    #     # 点积计算夹角
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     # 计算角速率
    #     los_angular_rate_rad_s = angle_rad / dt if dt > 1e-6 else 0.0
    #
    #     # 5. 更新历史记录
    #     self.prev_los_vec = current_los_vec
    #
    #     # --- [关键修改] 使用 tanh 进行归一化 ---
    #     # 为什么要改？因为在交汇瞬间 rate 可能极高。
    #     # 设置 scaling_factor: 假设我们认为 5度/秒 (约0.08 rad/s) 就是很棒的机动了。
    #     # 0.08 * 20 = 1.6 -> tanh(1.6) ≈ 0.9 (接近满分)
    #     scaling_factor = 1.0 #20.0
    #     reward = math.tanh(los_angular_rate_rad_s * scaling_factor)
    #
    #     return reward


    # 新的奖励函数:
    # def _reward_for_los_rate(self, aircraft: Aircraft, missile: Missile, dt: float):
    #     """
    #     (V2 - 修正版) 奖励视线矢量的角速度。
    #     新增了安全门控：只在飞机处于导弹前半球时才计算奖励。
    #     """
    #     # 1. 计算当前的视线矢量 (从导弹指向飞机)
    #     current_los_vec = aircraft.pos - missile.pos
    #     norm_current = np.linalg.norm(current_los_vec)
    #
    #     # --- 核心修正：增加前向扇区门控 ---
    #     # a) 获取导弹的速度矢量
    #     missile_v_vec = missile.get_velocity_vector()
    #     norm_missile_v = np.linalg.norm(missile_v_vec)
    #
    #     # b) 避免除零错误
    #     if norm_current < 1e-6 or norm_missile_v < 1e-6:
    #         # 如果距离过近或导弹静止，则不计算奖励
    #         self.prev_los_vec = current_los_vec  # 别忘了更新状态
    #         return 0.0
    #
    #     # c) 计算ATA角的余弦值
    #     # cos(ATA) = (LOS · V_missile) / (|LOS| * |V_missile|)
    #     cos_ata = np.dot(current_los_vec, missile_v_vec) / (norm_current * norm_missile_v)
    #
    #     # d) 如果飞机在导弹后半球 (ATA > 90度, cos(ATA) < 0)，则奖励为0
    #     if cos_ata < 0:
    #         self.prev_los_vec = current_los_vec  # 仍然需要更新状态以备下一帧
    #         return 1.0  # <<< MODIFIED >>> 给予最大奖励
    #         # return 0.0
    #     # --- 修正结束 ---
    #
    #     # 如果是第一步，则初始化并返回0
    #     if self.prev_los_vec is None:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     # 2. 计算角速度 (原逻辑)
    #     norm_prev = np.linalg.norm(self.prev_los_vec)
    #     if norm_prev < 1e-6:
    #         self.prev_los_vec = current_los_vec
    #         return 0.0
    #
    #     current_los_unit_vec = current_los_vec / norm_current
    #     prev_los_unit_vec = self.prev_los_vec / norm_prev
    #
    #     dot_product = np.clip(np.dot(current_los_unit_vec, prev_los_unit_vec), -1.0, 1.0)
    #     angle_rad = np.arccos(dot_product)
    #
    #     los_angular_rate_rad_s = angle_rad / dt if dt > 1e-6 else 0.0
    #
    #     # 5. 更新历史记录
    #     self.prev_los_vec = current_los_vec
    #
    #     # 6. 返回缩放后的奖励值
    #     return los_angular_rate_rad_s * 0.5

    def _reward_for_maintaining_speed(self, aircraft: Aircraft) -> float:
        """
        如果飞机速度高于等于0.8马-赫，则给予固定的正奖励。
        如果低于0.8马赫，则根据速度差施加惩罚。
        """
        # 1. 定义速度阈值 (单位: m/s)
        SPEED_THRESHOLD_MS = 250.0 #0.7 * 340.0 #0.8 * 340.0

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

    def _compute_missile_head_on_penalty(self, aircraft: Aircraft, missile: Missile):
        """
        计算飞机相对于导弹的“迎头惩罚” (Head-on Penalty)。

        逻辑：
        1. 如果导弹在飞离飞机（飞机在导弹后半球），不予惩罚（安全）。
        2. 如果飞机正对导弹飞行（迎头），给予最大负值惩罚 (-1.0)。
        3. 使用高斯函数使得偏离迎头角度时惩罚平滑衰减。
        """
        # --- 1. 计算基础几何关系 (与原代码一致) ---
        aircraft_v_vec = aircraft.get_velocity_vector()
        los_vec_m_to_a = aircraft.pos - missile.pos  # 从导弹指向飞机

        distance = np.linalg.norm(los_vec_m_to_a)
        norm_v = np.linalg.norm(aircraft_v_vec)
        norm_los = np.linalg.norm(los_vec_m_to_a)

        # 异常保护
        if norm_v < 1e-6 or norm_los < 1e-6:
            return 0.0

        # --- 2. 安全检查：如果导弹正在飞离飞机，则不惩罚 ---
        # (即飞机处于导弹的后半球)
        missile_v_vec = missile.get_velocity_vector()
        norm_missile_v = np.linalg.norm(missile_v_vec)

        if norm_missile_v > 1e-6:
            # 计算 ATA (Antenna Train Angle) 相关的余弦
            cos_ata = np.clip(np.dot(los_vec_m_to_a, missile_v_vec) / (norm_los * norm_missile_v), -1.0, 1.0)
            if cos_ata < 0:
                # 飞机在导弹后半球，导弹在远离，此时即使机头对着导弹(追击)也是安全的
                return 0.0

        # --- 3. 计算“飞机速度”与“机弹连线”的夹角余弦 ---
        # cos = 1.0  -> 飞机背对导弹 (拖尾)
        # cos = 0.0  -> 飞机垂直导弹 (三九)
        # cos = -1.0 -> 飞机正对导弹 (迎头) -> 这是我们要惩罚的
        cos_angle = np.clip(np.dot(aircraft_v_vec, los_vec_m_to_a) / (norm_v * norm_los), -1.0, 1.0)

        # --- 4. 如果不是迎头趋势，直接返回 0 ---
        # 我们可以设置一个阈值，例如 cos > -0.5 (夹角小于120度) 就不算迎头，不惩罚
        if cos_angle > -0.2:  # 稍微宽松一点，只有比较明显的迎头才惩罚
            return 0.0

        # --- 5. 计算高斯惩罚 ---
        # 目标角度是 180 度 (背对连线矢量，即正对导弹)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.rad2deg(angle_rad)

        # 计算与 180度 (迎头) 的误差
        angle_error_deg = abs(angle_deg - 180.0)

        # 定义迎头判定的宽窄 (你可以将此作为类属性 self.HEAD_ON_PENALTY_WIDTH_DEG)
        # 建议值: 45.0 到 60.0 度，越小表示只惩罚极度精准的迎头
        penalty_width_deg = 60.0

        # 计算高斯值 (0.0 到 1.0)
        penalty_factor = math.exp(-(angle_error_deg ** 2) / (2 * penalty_width_deg ** 2))

        # --- 6. 距离权重调整 (可选) ---
        # 距离越近，迎头越危险，惩罚应该越重
        # 如果距离很远(例如 >8km)，迎头可能是为了进攻，惩罚可以减轻
        dist_weight = 1.0
        if distance > 8000:
            dist_weight = 0.2
        elif distance > 5000:
            dist_weight = 0.5

        # 返回负值作为惩罚
        return -1.0 * penalty_factor * dist_weight

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

    # # <<< 新增：高G机动奖励函数 >>>
    # def _reward_for_high_g_maneuver(self, aircraft: Aircraft) -> float:
    #     """
    #     奖励飞机执行高G机动。
    #     奖励的大小与总法向过载正相关。   总过载
    #     """
    #     try:
    #         # 1. 获取当前的法向G力 (nz)
    #         #    假设：平飞时 nz=1.0, 3G转弯时 nz=3.0
    #         nz = aircraft.get_normal_g_force()  #法向过载
    #         # n_total = aircraft.get_total_g_force()   #飞行员总过载
    #         # nz_p = aircraft.get_nz_g_force()
    #
    #         # print(nz,nz_p)
    #
    #     except AttributeError:
    #         print("错误: 您的 Aircraft 类中没有 get_normal_g_force() 方法。")
    #         return 0.0
    #
    #     # 2. 我们只奖励正G力，不奖励负G（推杆）机动
    #     if nz <= 0:
    #         return 0.0
    #
    #     # 3. <<< 核心修改：直接用总G力进行归一化 >>>
    #     #    将当前G力映射到 [0, 1] 区间
    #     if self.AIRCRAFT_MAX_G <= 1e-6:
    #         return 0.0  # 避免除零
    #
    #     normalized_g = nz / self.AIRCRAFT_MAX_G
    #     normalized_g = np.clip(normalized_g, 0.0, 1.0)
    #
    #     # 4. 使用 tanh 函数进行平滑塑形
    #     reward = normalized_g * 1.0
    #     # reward = np.tanh(normalized_g * 1.0)
    #
    #     # # 调试打印
    #     # print(f"NZ: {nz:.2f} G, Norm G: {normalized_g:.2f}, Reward: {reward:.3f}")
    #
    #     return reward

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
            # Pv = - penalty_factor
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
    #     ALTITUDE_PENALTY_WEIGHT = 1.0 #5.0
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
    #     """干扰资源使用过量惩罚函数"""
    #     if release_flare_action > 0.5:
    #         # (中文) 您原始代码中的 alphaR 和 k1
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

    # def _penalty_for_roll_rate_magnitude(self, aircraft: Aircraft):
    #     """
    #     (最终版 - 无阈值) 直接惩罚任何非零的滚转角速度。
    #     惩罚的大小与滚转角速度的绝对值成正比。
    #     """
    #     # 从状态向量中获取当前的实际滚转角速度
    #     p_real_rad_s = aircraft.roll_rate_rad_s
    #     if self.MAX_PHYSICAL_ROLL_RATE_RAD_S < 1e-6: return 0.0
    #     # print(np.rad2deg(p_real_rad_s))
    #     # if np.rad2deg(abs(p_real_rad_s)) < 120 : return 0.0
    #     # 将当前的滚转速率归一化到 [0, 1] 范围
    #     #    (当前速率 / 最大速率)
    #     #    防止除零错误
    #
    #     normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
    #     return (normalized_roll_rate) * -1.0  # 返回基准值[0,1]


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
    #     # print(np.rad2deg(p_real_rad_s), 0.5* -k * normalized_penalty)
    #
    #     return -k * normalized_penalty

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

    # def _reward_for_coordinated_turn(self, aircraft: Aircraft, dt: float):
    #     """
    #     (V5 - 最终版) 直接从JSBSim读取nz，惩罚压杆行为。
    #     这个版本简单、直接且不会出错。
    #     """
    #     # # 1. 计算滚转因子 (请务必最后一次确认滚转角的正确索引，很可能是[3]！)
    #     # try:
    #     #     roll_rad = aircraft.state_vector[6]  # <--- 滚转角 phi
    #     # except IndexError:
    #     #     return 0.0
    #     #
    #     # roll_factor = math.sin(abs(roll_rad))
    #     # # if abs(np.rad2deg(roll_rad)) > 90:
    #     # #     roll_factor = math.sin(math.pi - abs(roll_rad))
    #     # roll_factor = np.clip(roll_factor, 0, 1.0)
    #
    #     # 2. 直接从飞机对象获取法向G力 (nz)
    #     try:
    #         nz = aircraft.get_normal_g_force()
    #         # nz2 = aircraft.get_pilot_normal_g()
    #         # print(nz,nz2)
    #     except AttributeError:
    #         print("错误: 您的 Aircraft 类中没有 get_normal_g_force() 方法。请先实现它。")
    #         return 0.0
    #
    #     # 3. 根据JSBSim的符号约定计算G力因子
    #     #    平飞: nz = -1.0 G -> positive_g = 1.0 G
    #     #    2G转弯: nz = -2.0 G -> positive_g = 2.0 G
    #     positive_g = nz
    #
    #     # G力因子是超过1G的部分
    #     g_factor = min(0, positive_g) / (self.MAX_G_LIMIT)
    #     # g_factor = min(0, positive_g - 1) / (self.MAX_G_LIMIT - 1)
    #     g_factor = np.clip(g_factor, 0, 1.0)
    #
    #     # 4. 最终奖励
    #     # reward = -1.0 * roll_factor * g_factor
    #     reward = -1.0  * g_factor
    #
    #     # 调试打印
    #     # print(f"RollF: {roll_factor:.2f} | NZ1: {nz:.2f} | G_F: {g_factor:.2f} | Reward: {reward:.3f}")
    #
    #     return reward

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
