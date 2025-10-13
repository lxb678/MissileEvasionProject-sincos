# 文件: missile.py

import numpy as np


class Missile:
    """
    封装了AIM-9X导弹所有状态、物理参数、导引和动力学模型的类。
    """

    def __init__(self, initial_state: np.ndarray):
        """
        使用一个初始状态向量来初始化导弹。

        Args:
            initial_state (np.ndarray): 包含6个元素的初始状态向量
                [V(速度), theta(俯仰角), psi_c(偏航角), x(北), y(上), z(东)]
        """
        # --- 核心状态 ---
        self.state = np.array(initial_state, dtype=float)

        # --- 物理与导引参数 (源自您的 AirCombatEnv) ---
        self.g = 9.81
        self.N = 5.0  # 比例导引系数
        self.mass = 60.0  # (kg) 导弹质量
        self.diameter = 0.127  # (m) 导弹直径
        self.max_g_load = 50.0  # 导弹最大过载

        # --- 导引头/引信参数 ---
        self.seeker_max_range = 30000.0  # 导引头最大搜索范围 (m)
        self.seeker_fov_rad = np.deg2rad(90)  # 导引头最大视场角度 (弧度)
        self.seeker_max_omega_rad_s = 12.0  # 导引头最大角速度 (弧度/秒)
        self.seeker_max_time = 60.0  # 导引头最大搜索时间 (秒)

    def update_with_los_rate(self, dt: float, theta_L_dot: float, phi_L_dot: float):
        """
        根据外部传入的视线角速率，更新导弹在一个时间步 dt 后的状态。
        """
        """
        根据等效目标位置，更新导弹在一个时间步 dt 后的状态。

        Args:
            dt (float): 仿真时间步长 (秒)。
            target_pos_equiv (np.ndarray): 导引头当前锁定的等效红外质心位置 [x, y, z]。
            prev_los_angles (tuple): 上一步的视线角 (prev_theta_L, prev_phi_L)。

        Returns:
            tuple: 新的视线角 (theta_L, phi_L)，用于在主环境中更新。
        """
        # # --- 1. 计算视线角速率 (Line-of-Sight Rate) ---
        # theta_L, phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
        #     target_pos_equiv, prev_los_angles, dt
        # )

        # --- 2. 调用动力学函数计算新状态 ---
        self.state = self._missile_dynamics(self.state, dt, theta_L_dot, phi_L_dot)


    # --- 属性访问器 (Property Accessors) ---
    @property
    def pos(self) -> np.ndarray:
        """返回导弹的位置向量 [x, y, z] (北, 天, 东)"""
        return self.state[3:6]

    @property
    def velocity(self) -> float:
        """返回导弹的总速度大小 (V)"""
        return self.state[0]

    @property
    def state_vector(self) -> np.ndarray:
        """返回完整的状态向量"""
        return self.state

    def get_velocity_vector(self) -> np.ndarray:
        """计算并返回导弹在世界坐标系(NUE)下的速度矢量"""
        V, theta, psi = self.state[0], self.state[1], self.state[2]
        vx = V * np.cos(theta) * np.cos(psi)
        vy = V * np.sin(theta)
        # 注意：您的导弹动力学中，偏航角psi是围绕y轴（天轴）的，
        # 且z轴是东。这与标准的航空坐标系有差异，可能导致vz为负。
        # 这里我们严格按照您的公式来，psi=0时朝北，psi=90时朝东。
        vz = V * np.cos(theta) * np.sin(psi)
        return np.array([vx, vy, vz])

    def check_seeker_lock(self, target_aircraft, t_now: float) -> bool:
        """
        检查导引头是否能锁定目标飞机。
        """
        # 这是一个简化的接口，完整的检查逻辑在主环境类中，因为它需要两个对象。
        # 实际使用中，一个独立的 check_seeker_lock 函数会更好。
        # 这里我们只提供一个框架。
        R_vec = target_aircraft.pos - self.pos
        R_mag = np.linalg.norm(R_vec)

        if R_mag > self.seeker_max_range or t_now > self.seeker_max_time:
            return False

        missile_vel_vec = self.get_velocity_vector()
        cos_angle = np.dot(R_vec, missile_vel_vec) / (R_mag * self.velocity + 1e-6)

        if cos_angle < np.cos(self.seeker_fov_rad):  # 简化为视场角判断
            return False

        return True

    # --- 私有辅助方法 ---
    def _calculate_los_rate(self, target_pos_equiv, prev_los_angles, dt):
        """计算视线角和视线角速率"""
        prev_theta_L, prev_phi_L = prev_los_angles

        Rx = target_pos_equiv[0] - self.state[3]
        Ry = target_pos_equiv[1] - self.state[4]
        Rz = target_pos_equiv[2] - self.state[5]
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) + 1e-6

        theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
        phi_L = np.arctan2(Rz, Rx)

        if prev_theta_L is None or prev_phi_L is None:
            theta_L_dot, phi_L_dot = 0.0, 0.0
        else:
            theta_L_dot = (theta_L - prev_theta_L) / dt
            dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
            phi_L_dot = dphi / dt

        return theta_L, phi_L, theta_L_dot, phi_L_dot

    def _missile_dynamics(self, state, dt, theta_L_dot, phi_L_dot):
        """
        导弹的核心动力学积分器。这是一个纯函数。
        """

        # def get_Cx_AIM9X(Ma: float) -> float:
        #     if Ma <= 0.9:
        #         return 0.20
        #     elif Ma <= 1.2:
        #         return 0.20 + (0.38 - 0.20) * (Ma - 0.9) / 0.3
        #     elif Ma <= 4.0:
        #         return 0.38 * np.exp(-0.35 * (Ma - 1.2)) + 0.15
        #     else:
        #         return 0.15

        # AIM-9X 阻力系数模型 更大
        def get_Cx_AIM9X(Ma: float) -> float:
            """
            根据马赫数 Ma 返回导弹阻力系数 Cd
            分段拟合：
              1) Ma <= 0.8 : 常数
              2) 0.8 < Ma <= 1.1 : 线性
              3) 1.1 < Ma <= 4.0 : 三次多项式
            """
            Ma1, Ma2, Ma3 = 0.8, 1.1, 4.0

            if Ma <= Ma1:
                return 0.5
            elif Ma <= Ma2:
                # 线性段 (0.8,0.5) -> (1.1,0.78)
                return 0.9333 * Ma - 0.2466
            elif Ma <= Ma3:
                # 三次多项式段
                return (-0.0004537 * Ma ** 3
                        + 0.0290835 * Ma ** 2
                        - 0.2867967 * Ma
                        + 1.0608893)
            else:
                # 超出拟合范围可选择外推或固定
                return (-0.0004537 * Ma ** 3
                        + 0.0290835 * Ma ** 2
                        - 0.2867967 * Ma
                        + 1.0608893)

        V, theta, psi_c, x_pos, y_pos, z_pos = state

        # === 1. 过载计算 ===
        ny = self.N * V * theta_L_dot / self.g + np.cos(theta)
        # nz = self.N * V * phi_L_dot * np.cos(theta) / self.g  # (修正) 传统导引律中, nz = N*V*phi_dot*cos(theta)/g
        # (中文) 核心恢复：移除 nz 计算中的 cos(theta) 项
        nz = self.N * V * phi_L_dot / self.g

        n_total_cmd = np.sqrt(ny ** 2 + nz ** 2)
        if n_total_cmd > self.max_g_load:
            scale = self.max_g_load / n_total_cmd
            ny *= scale
            nz *= scale

        # === 2. 空气动力学计算 ===
        H = y_pos
        Temper = 15.0
        T_H = 273 + Temper - 0.6 * H / 100
        P_H = (1 - H / 44300) ** 5.256
        rho = 1.293 * P_H * (273 / T_H)
        Ma = V / 340

        Cx = get_Cx_AIM9X(Ma)
        S = np.pi * (self.diameter ** 2) / 4
        q = 0.5 * rho * V ** 2
        F_drag = Cx * q * S

        # === 3. 积分与状态更新 ===
        v_dot = -F_drag / self.mass - self.g * np.sin(theta)
        V_next = V + v_dot * dt + 1e-8

        # 防止速度过小导致除零错误
        # V_for_calc = max(V_next, 10.0)

        theta_dot = (ny * self.g - self.g * np.cos(theta)) / V_next
        psi_c_dot = (nz * self.g) / (V_next * np.cos(theta))

        theta_next = theta + theta_dot * dt
        psi_c_next = psi_c + psi_c_dot * dt

        theta_next = np.clip(theta_next, -np.deg2rad(89), np.deg2rad(89))

        if psi_c_next > np.pi:
            psi_c_next -= 2 * np.pi
        if psi_c_next < -np.pi:
            psi_c_next += 2 * np.pi

        # (中文) 您的旧代码中，位置更新使用的是积分前的速度矢量。
        # 更精确的做法是使用平均速度或积分后的速度矢量。这里我们保持与您旧代码一致的简化。
        dx = V_next * np.cos(theta_next) * np.cos(psi_c_next)
        dy = V_next * np.sin(theta_next)
        dz = V_next * np.cos(theta_next) * np.sin(psi_c_next)

        pos_next = np.array([x_pos, y_pos, z_pos]) + np.array([dx, dy, dz]) * dt

        return np.array([V_next, theta_next, psi_c_next, *pos_next])