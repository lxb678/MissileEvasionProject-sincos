# 文件: reward_system.py (已修正，精确匹配您的代码)

import numpy as np
import math

# (中文) 导入对象类，以便进行类型提示和访问状态
from .aircraft import Aircraft
from .missile import Missile


class RewardCalculator:
    """
    封装了所有与奖励计算相关的逻辑。
    所有逻辑和参数均从您最新的 AirCombatEnv 文件中精确提取。
    """

    def __init__(self):
        # --- 在这里集中定义所有奖励相关的超参数 ---

        # [稀疏奖励参数]
        self.W = 100  # 成功奖励基准
        self.U = -100  # 失败固定惩罚

        # [高度惩罚参数]
        self.SAFE_ALTITUDE_M = 1000.0
        self.DANGER_ALTITUDE_M = 500.0
        self.KV_MS = 0.2 * 340
        self.MAX_ALTITUDE_M = 12000.0
        self.OVER_ALTITUDE_PENALTY_FACTOR = 0.5

        # [三九线奖励参数]
        self.ASPECT_REWARD_EFFECTIVE_RANGE = 5000.0
        self.ASPECT_REWARD_WIDTH_DEG = 45.0
        self.ASPECT_PITCH_THRESHOLD_DEG = 70.0

        # [滚转惩罚参数] (无阈值版本)
        self.MAX_PHYSICAL_ROLL_RATE_RAD_S = np.deg2rad(240.0)

        # [速度惩罚参数]
        self.OPTIMAL_SPEED_FLOOR_MACH = 0.8
        self.K_SPEED_FLOOR_PENALTY = -2.0

        # --- 状态变量 ---
        self.prev_missile_v_mag = None

    def reset(self):
        """为新回合重置状态变量。"""
        self.prev_missile_v_mag = None
        # 注意：持续滚转的计时器现在不在这个类里，因为它依赖于环境的dt，
        # 最好由主环境管理。或者在这里接收dt进行更新。为简化，我们先假设它在主环境中。

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
                               remaining_flares: int, total_flares: int, action: list) -> float:
        """
        计算并返回当前时间步的总密集奖励。
        这是从主环境调用的唯一接口。
        """
        # 1. 计算所有独立的奖励/惩罚组件
        reward_posture = self._compute_missile_posture_reward(missile, aircraft)
        reward_altitude = self._compute_altitude_reward(aircraft)
        reward_resource = self._compute_resource_penalty(action[3], remaining_flares, total_flares)
        reward_aspect = self._reward_for_aspect_angle(aircraft, missile)
        reward_roll_penalty = self._penalty_for_roll_rate_magnitude(aircraft)
        reward_speed_penalty = self._penalty_for_dropping_below_speed_floor(aircraft)

        # 2. 将所有组件按权重加权求和 (权重直接在此处定义，与您的代码一致)
        final_dense_reward = (
                1.0 * reward_posture +
                0.5 * reward_altitude +
                0.2 * reward_resource +
                1.0 * reward_aspect +
                0.8 * reward_roll_penalty +  # 惩罚项权重应为负数, reward_F_roll_penalty基准是正的
                1.0 * reward_speed_penalty  # reward_for_optimal_speed基准是负的
        )
        # print(f"reward_posture: {1.0 * reward_posture:.2f}",
        #       f"reward_altitude: {0.5 * reward_altitude:.2f}",
        #       f"reward_resource: {0.2 * reward_resource:.2f}",
        #       f"reward_aspect: {1.0 * reward_aspect:.2f}",
        #       f"reward_roll_penalty: {0.8 * reward_roll_penalty:.2f}",
        #       f"reward_speed_penalty: {1.0 * reward_speed_penalty:.2f}",
        #       f"final_dense_reward: {final_dense_reward:.2f}")

        return final_dense_reward

    # --- (中文) 下面是所有从您主环境文件中迁移过来的、正在使用的私有奖励计算方法 ---

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
            reward = 0.0

        # 8. 更新上一步的速度 (原步骤7)
        self.prev_missile_v_mag = missile_v_mag

        # 9. 返回最终计算出的奖励值 (原步骤8)
        return reward * 1.0

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
        distance_weight = 1.0 - np.clip(current_R_rel / self.ASPECT_REWARD_EFFECTIVE_RANGE, 0.0, 1.0)
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
        # 将当前的滚转速率归一化到 [0, 1] 范围
        #    (当前速率 / 最大速率)
        #    防止除零错误
        normalized_roll_rate = abs(p_real_rad_s) / self.MAX_PHYSICAL_ROLL_RATE_RAD_S
        return normalized_roll_rate * -1.0  # 返回基准值[0,1]

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
            penalty_base = normalized_deficit ** 2  # 您的代码使用的是二次惩罚
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