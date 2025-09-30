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

        # --- <<< 新增：状态标志和辅助属性 >>> ---
        self.is_active = True
        self.flight_time = 0.0
        self.max_flight_time = 60.0  # 导弹最大飞行时间 (s)
        self.target = None  # 用于存储导弹的目标对象 (在发射时由环境设置)
        self.id = "missile_default_id"  # 导弹的唯一ID (在发射时由环境设置)
        self.name = "Missile"  # Tacview显示的名称
        self.color = "Orange"  # Tacview显示的颜色

        # --- <<< 新增：用于计算视线角速率的历史变量 >>> ---
        self.prev_theta_L = None
        self.prev_phi_L = None

        # --- 物理与导引参数 (源自您的 AirCombatEnv) ---
        self.g = 9.81
        self.N = 5.0  # 比例导引系数
        self.mass = 90.0  # (kg) AIM-9X 质量约为 85kg，我们取一个整数
        self.diameter = 0.127  # (m) 导弹直径
        self.max_g_load = 50.0  # 导弹最大过载

        # --- <<< 新增：发动机参数 >>> ---
        self.thrust = 12000.0  #15000.0 # (N) 助推器推力, 这是一个估算值
        self.boost_time = 3.0  # (s) 助推器工作时间

        # --- <<< 新增：从规避环境中移植的计时器 >>> ---
        self.prev_dist_to_target = None  # 用于计算 range_rate
        self.escape_timer = 0.0
        self.lost_and_separating_duration = 0.0

        # --- <<< 新增：失锁自毁计时器 >>> ---
        self.time_since_lock_lost = 0.0

        # --- <<< 新增：用于检测状态变化的标志 >>> ---
        self.was_active_in_prev_frame = True
        # --- <<< 新增：记录失效原因 >>> ---
        self.inactive_reason = "In-Flight"

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

    # --- <<< 改造：主更新方法 >>> ---
    def update(self, dt: float, target_pos_equiv: np.ndarray or None):
        """
        根据等效目标位置，更新导弹在一个时间步 dt 后的状态。
        这个方法现在封装了所有逻辑。

        Args:
            dt (float): 仿真时间步长 (秒)。
            target_pos_equiv (np.ndarray or None): 导引头当前锁定的等效红外质心位置 [x, y, z]。
                                                   如果为 None，表示导引头失锁。
        """
        # 1. 检查导弹是否还应继续飞行
        self.flight_time += dt
        if not self.is_active or self.flight_time > self.max_flight_time:
            self.is_active = False
            self.inactive_reason = "Flight Time Exceeded"  # <--- 记录原因
            return

        # 2. <<< 新增：检查失锁自毁逻辑 >>>
        if target_pos_equiv is None:
            # 如果当前帧失锁，累加计时器
            self.time_since_lock_lost += dt
        else:
            # 如果当前帧有目标，清零计时器
            self.time_since_lock_lost = 0.0
        if self.time_since_lock_lost > 2.0:
            # if self.is_active:  # 确保只打印一次
                # print(f">>> 导弹 {self.name} 因持续失锁超过2秒而自毁。")
            self.is_active = False
            self.inactive_reason = "Target Lost"  # <--- 记录原因
            return  # 自毁后，直接返回，不再进行后续计算

        # 2. 计算视线角速率 (Line-of-Sight Rate)
        if target_pos_equiv is not None:
            # 如果导引头锁定目标，正常计算LOS速率
            theta_L, phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
                target_pos_equiv, (self.prev_theta_L, self.prev_phi_L), dt
            )
            # 更新历史值，为下一步计算做准备
            self.prev_theta_L = theta_L
            self.prev_phi_L = phi_L
        else:
            # 如果导引头失锁 (target_pos_equiv is None)，导弹按直线飞行
            # 此时视线角速率为0，即没有制导指令
            theta_L_dot, phi_L_dot = 0.0, 0.0

        # 3. 调用动力学函数计算新状态
        # 调用动力学模型时，需要传入当前飞行时间来判断发动机是否工作
        self.state = self._missile_dynamics(self.state, dt, theta_L_dot, phi_L_dot, self.flight_time)

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

    # --- <<< 核心修改：更新动力学模型 >>> ---
    def _missile_dynamics(self, state, dt, theta_L_dot, phi_L_dot, current_flight_time):
        """
        导弹的核心动力学积分器，增加了助推段逻辑。
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

        # 1. 过载计算 (不变)
        ny = self.N * V * theta_L_dot / self.g + np.cos(theta)
        nz = self.N * V * phi_L_dot / self.g
        n_total_cmd = np.sqrt(ny ** 2 + nz ** 2)
        if n_total_cmd > self.max_g_load:
            scale = self.max_g_load / n_total_cmd
            ny, nz = ny * scale, nz * scale

        # 2. 空气动力学计算 (不变)
        H = y_pos
        T_H = 288.15 - 0.0065 * H
        rho = 1.225 * (T_H / 288.15) ** (self.g / (287 * 0.0065) - 1)
        a_sound = np.sqrt(1.4 * 287 * T_H)
        Ma = V / (a_sound + 1e-6)
        Cx = get_Cx_AIM9X(Ma)
        S = np.pi * (self.diameter ** 2) / 4
        q = 0.5 * rho * V ** 2
        F_drag = Cx * q * S

        # 3. <<< 新增：判断发动机是否工作 >>>
        current_thrust = self.thrust if current_flight_time <= self.boost_time else 0.0

        # 4. 积分与状态更新 (合力计算发生变化)
        # v_dot = (合力) / 质量
        # 合力 = 推力 - 阻力 - 重力在速度方向的分量
        v_dot = (current_thrust - F_drag) / self.mass - self.g * np.sin(theta)
        V_next = V + v_dot * dt

        # 为了防止仿真不稳定，可以在这里加一个马赫数上限
        if V_next > 2.5 * a_sound:
            V_next = 2.5 * a_sound

        V_next = max(V_next, 1.0)

        theta_dot = (ny * self.g - self.g * np.cos(theta)) / V_next
        psi_c_dot = (nz * self.g) / (V_next * np.cos(theta) + 1e-6)

        theta_next = np.clip(theta + theta_dot * dt, -np.pi / 2, np.pi / 2)
        psi_c_next = (psi_c + psi_c_dot * dt + np.pi) % (2 * np.pi) - np.pi

        dx = V_next * np.cos(theta_next) * np.cos(psi_c_next)
        dy = V_next * np.sin(theta_next)
        dz = V_next * np.cos(theta_next) * np.sin(psi_c_next)
        pos_next = np.array([x_pos, y_pos, z_pos]) + np.array([dx, dy, dz]) * dt

        return np.array([V_next, theta_next, psi_c_next, *pos_next])