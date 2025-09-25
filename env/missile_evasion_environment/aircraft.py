# 文件: aircraft.py

import numpy as np

class Aircraft:
    """
    封装了F-16飞机所有状态、物理参数和动力学模型的类。
    """

    def __init__(self, initial_state: np.ndarray):
        """
        使用一个初始状态向量来初始化飞机。

        Args:
            initial_state (np.ndarray): 包含8个元素的初始状态向量
                [x(北), y(天), z(东), Vt(速度), theta(俯仰), psi(偏航), phi(滚转), p_real(滚转速率)]
        """
        # --- 核心状态 ---
        self.state = np.array(initial_state, dtype=float)

        # --- 飞机物理与气动参数 (源自您的 AirCombatEnv) ---
        self.g = 9.81
        self.m = 15000.0  # (kg) 飞机质量
        self.S = 27.87  # (m^2) 机翼参考面积
        self.MAX_TWR = 1.2  # 最大推重比

        self.SEA_LEVEL_STATIC_THRUST = self.MAX_TWR * self.m * self.g  # 海平面最大静推力 (N)
        self.RHO_0 = 1.225  # 海平面标准大气密度 (kg/m^3)
        # 经验系数, 用于模拟推力随高度和马赫数的变化
        self.THRUST_ALT_EXP = 0.7  # 推力随密度变化的指数 (alpha)
        self.THRUST_MACH_K1 = 0.6  # 马赫数一次项系数 (用于模拟冲压效应)
        self.THRUST_MACH_K2 = 0.18  # 马赫数二次项系数 (用于模拟高马赫数下的损失)

        # 滚转响应时间常数
        self.tau_roll = 0.2

        # 飞行动作的物理限制
        self.min_velocity_ms = 100.0
        self.max_velocity_ms = 400.0

    # --- (中文) 核心修正：添加这个缺失的属性 ---
    @property
    def attitude_rad(self) -> tuple:
        """
        以属性的方式，返回飞机的姿态（弧度）。
        返回一个包含 (theta, psi, phi) 的元组。
        """
        # 状态向量: [x, y, z, Vt, theta, psi, phi, p_real]
        # theta (俯仰) 在索引 4
        # psi   (偏航) 在索引 5
        # phi   (滚转) 在索引 6
        return self.state[4], self.state[5], self.state[6]

    def update(self, dt: float, action: list):
        """
        根据AI的动作指令，更新飞机在一个时间步 dt 后的状态。

        Args:
            dt (float): 仿真时间步长 (秒)。
            action (list): 来自AI的动作指令 [nx, nz, p_cmd, ...]。
                           nx in [-1, 1], nz in [-5, 9], p_cmd in [-240, 240] deg/s
        """
        nx_cmd, nz_cmd, p_cmd = action[0], action[1], action[2]

        # 调用独立的动力学函数来计算新状态
        self.state = self._aircraft_dynamics(self.state, dt, nx_cmd, nz_cmd, p_cmd)

        # 应用飞行包线限制
        self.state[3] = np.clip(self.state[3], self.min_velocity_ms, self.max_velocity_ms)

    # --- 属性访问器 (Property Accessors) ---
    @property
    def pos(self) -> np.ndarray:
        """返回飞机的位置向量 [x, y, z] (北, 天, 东)"""
        return self.state[:3]

    @property
    def velocity(self) -> float:
        """返回飞机的总速度大小 (Vt)"""
        return self.state[3]

    @property
    def roll_rate_rad_s(self) -> float:
        """返回飞机的实际滚转角速度 (p_real)"""
        return self.state[7]

    @property
    def state_vector(self) -> np.ndarray:
        """返回完整的状态向量"""
        return self.state

    def get_velocity_vector(self) -> np.ndarray:
        """计算并返回飞机在世界坐标系(NUE)下的速度矢量"""
        Vt, theta, psi = self.state[3], self.state[4], self.state[5]
        vx = Vt * np.cos(theta) * np.cos(psi)
        vy = Vt * np.sin(theta)
        vz = Vt * np.cos(theta) * np.sin(psi)
        return np.array([vx, vy, vz])

    # --- 核心动力学模型 (作为私有方法) ---
    def _aircraft_dynamics(self, state: np.ndarray, dt: float, nx: float, nz: float, p_cmd: float) -> np.ndarray:
        """
        飞机的核心动力学积分器。这是一个纯函数，不修改类自身的状态。
        所有逻辑精确地从您提供的 AirCombatEnv6_maneuver_flare.py 中提取。
        """
        # ================== 坐标系定义 ==================
        # 惯性坐标系 (Inertial Frame):
        #   - 外部状态表示: 北-天-东 (North-Up-East, NUE) -> 用于 state 向量和绘图
        #   - 内部物理计算: 北-东-地 (North-East-Down, NED) -> 用于核心物理引擎
        # 机体坐标系 (Body Frame): 前-右-下 (Forward-Right-Down, FRD)
        # =========================================================

        # ================== 四元数与旋转矩阵函数 (保持不变) ==================
        # 这些函数依然基于 NED/FRD 的欧拉角定义 (phi-roll, theta-pitch, psi-yaw)

        # --- 内部辅助函数 ---
        def normalize(q):
            norm = np.linalg.norm(q)
            return q / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])

        def euler_to_quaternion(phi, theta, psi):
            cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
            cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
            cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
            q0 = cr * cp * cy + sr * sp * sy
            q1 = sr * cp * cy - cr * sp * sy
            q2 = cr * sp * cy + sr * cp * sy
            q3 = cr * cp * sy - sr * sp * cy
            return normalize(np.array([q0, q1, q2, q3]))

        def quaternion_to_rotation_matrix(q):
            q0, q1, q2, q3 = q
            # 从机体系(FRD)到惯性系(NED)的旋转矩阵 R_frd_to_ned
            return np.array([
                [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
            ])

        def quaternion_to_continuous_euler(q):
            R = quaternion_to_rotation_matrix(q)
            sin_theta = -R[2, 0]
            theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
            cos_theta = np.cos(theta)
            if np.abs(cos_theta) < 1e-6:
                phi, psi = 0.0, np.arctan2(-R[0, 1], R[1, 1])
            else:
                psi, phi = np.arctan2(R[1, 0], R[0, 0]), np.arctan2(R[2, 1], R[2, 2])
            return phi, theta, psi

        def get_total_drag_coefficient(Ma):
            """
                (最简化模型 v2 - 已修正数值)
                根据马赫数(Ma)，直接估算一个“中等强度机动”状态下的总阻力系数(C_D)。
                """
            if Ma < 0.9:
                # 亚音速机动状态，升致阻力已经比较显著
                C_D = 0.10
            elif Ma < 1.2:
                # 跨声速区域，激波阻力叠加升致阻力，达到峰值
                # (线性插值到一个很高的峰值，例如 0.30)
                C_D = 0.10 + (0.16 - 0.10) * ((Ma - 0.9) / 0.3)
            elif Ma < 2.0:
                # 超声速区域，虽然升力效率变差，但阻力仍然很高
                C_D = 0.16 + (0.25 - 0.16) * ((Ma - 1.2) / 0.8)
            else:
                # 高超声速区域，阻力稳定在一个较高的水平
                C_D = 0.25

            return C_D

        # MODIFIED: 状态向量定义更新为 北-天-东 (NUE)
        # 状态向量: [x(北), y(上), z(东), Vt, theta(俯仰), psi(偏航), phi(滚转)]

        # === 1. 解包当前状态 ===
        x_nue, y_nue, z_nue, Vt, theta, psi, phi, p_real = state

        # === 坐标转换: 从 NUE (状态) 到 NED (物理) ===
        # pos_ned = [North, East, Down]
        # x_ned = x_nue
        # y_ned = z_nue
        # z_ned = -y_nue
        pos_ned = np.array([x_nue, z_nue, -y_nue])

        # === 转换 (欧拉角到旋转矩阵) ===
        q_current = euler_to_quaternion(phi, theta, psi)
        R_frd_to_ned = quaternion_to_rotation_matrix(q_current)

        # === 2. 滚转动力学 ===
        p_dot = (1 / self.tau_roll) * (p_cmd - p_real)  # p_real 的变化率
        p_real_new = p_real + p_dot * dt
        p_body = np.clip(p_real_new, -np.deg2rad(240), np.deg2rad(240))  # 使用实际滚转速率并加以限制

        # === 3. 核心物理计算 ===
        H = y_nue  # 高度 (米)
        Temper = 15.0 # 温度 (摄氏度)
        T_H = 273 + Temper - 0.6 * H / 100 # 温度 (K)
        P_H = (1 - H / 44300) ** 5.256 # 压力系数
        rho = 1.293 * P_H * (273 / T_H)  # 密度 (kg/m^3)
        Ma = Vt / 340.0  # 简化马赫数计算
        q = 0.5 * rho * Vt ** 2  # 动压

        lift = nz * self.m * self.g  # 升力 (N)  # 这里的 lift 是AI指令产生的总升力

        C_D = get_total_drag_coefficient(Ma)  #总阻力系数 C_D
        drag = q * self.S * C_D  # 阻力 (N)

        # thrust = 0.0
        # # --- 将过载指令和计算出的阻力转换为力 ---
        # # 现在的 nx_cmd 代表的是 "推力过载" (Thrust-to-Weight Ratio)
        # # 我们需要计算出实际的推力
        # if nx >= 0:
        #     # --- 推力计算 ---
        #     # 将 nx_cmd (0 to 1) 映射到推力百分比 (0 to 1)
        #     # 这里的映射关系可以更复杂，但线性映射是一个好的开始
        #     # 假设 nx_cmd=1 对应最大推力
        #     thrust = (self.MAX_TWR * self.m * self.g) * nx  # 假设 nx in [0, 1]
        # else:  # nx < 0, 对应减速板
        #     # 减速板不产生推力，而是增加阻力
        #     # --- 减速板阻力计算 ---
        #     # 将 nx_cmd (-1 to 0) 映射到减速板开启程度 (1 to 0)
        #     # 减速板会增加一个巨大的额外阻力
        #     thrust = nx * 0.8 * self.m * self.g

        # --- 推力计算 (更真实的模型) ---
        # 1. 计算海平面标准密度与当前高度密度的比值
        density_ratio = rho / self.RHO_0

        # 2. 根据高度和马赫数计算最大可用推力的修正系数
        #    这个公式是一个经验模型，模拟了冲压效应和高空/高速损失
        thrust_correction_factor = (density_ratio ** self.THRUST_ALT_EXP) * \
                                   (1 + self.THRUST_MACH_K1 * Ma - self.THRUST_MACH_K2 * Ma ** 2)
        # 保证系数不为负
        thrust_correction_factor = max(0, thrust_correction_factor)

        # 3. 计算当前条件下的最大可用推力
        max_available_thrust = self.SEA_LEVEL_STATIC_THRUST * thrust_correction_factor

        thrust = 0.0
        if nx >= 0:
            # 实际推力 = 油门指令(0-1) * 当前最大可用推力
            thrust = max_available_thrust * nx
        else:  # nx < 0, 对应减速板
            # 减速板逻辑可以保持不变，它增加阻力，不产生推力
            # 为了更清晰，我们明确推力为0，额外阻力在后面计算
            thrust = 0
            # 额外阻力可以加到 drag 上，或者像您一样用一个负推力等效
            # 这里我们采用您的负推力等效方式，但注意这只是一个效果模拟
            thrust = nx * 0.5 * self.m * self.g  # 这里的模型也可以再细化，但暂时保持

        #净前向力 = 推力 - 阻力
        net_forward_force = thrust - drag

        # 2. MODIFIED: 在机体系(FRD: 前-右-下)中定义气动力
        # 升力 (lift) 产生向上的力, 在 FRD 的 Z 轴 (向下) 是负方向
        F_aero_frd = np.array([net_forward_force, 0, -lift]) # 机体系中的气动力
        F_gravity_ned = np.array([0, 0, self.m * self.g]) # 惯性系中的重力
        F_total_ned = R_frd_to_ned @ F_aero_frd + F_gravity_ned # 总力在惯性系中的表示
        V_vec_ned = R_frd_to_ned @ np.array([Vt, 0, 0])   # 速度矢量在机体系中总是指向前方

        # === 4. 积分与状态更新 ===
        # F_total_ned 中垂直于 V_vec_ned 的分力导致了速度方向的改变
        # F_perp_ned = F_total_ned - np.dot(F_total_ned, V_vec_ned / Vt) * (V_vec_ned / Vt)
        # a_perp = F_perp / m, R = V^2 / a_perp, omega = V / R = a_perp / V
        omega_ned = np.cross(V_vec_ned, F_total_ned) / (self.m * Vt ** 2 + 1e-8) if Vt > 1e-3 else np.zeros(3)
        #  转换回机体系 (FRD)
        omega_frd = R_frd_to_ned.T @ omega_ned

        p, q_body, r_body = p_body, omega_frd[1], omega_frd[2]  # 俯仰率 q (绕 Y 轴) 和 偏航率 r (绕 Z 轴) 由动力学决定
        # --- 动力学与运动学积分 (仍然在 NED/FRD 框架下) ---
        dq_dt = 0.5 * np.array([-q_current[1] * p - q_current[2] * q_body - q_current[3] * r_body,
                                q_current[0] * p + q_current[2] * r_body - q_current[3] * q_body,
                                q_current[0] * q_body - q_current[1] * r_body + q_current[3] * p,
                                q_current[0] * r_body + q_current[1] * q_body - q_current[2] * p])
        q_new = normalize(q_current + dt * dq_dt)
        phi_new, theta_new, psi_new = quaternion_to_continuous_euler(q_new)

        acceleration_ned = F_total_ned / self.m
        V_unit_vec_ned = V_vec_ned / Vt if Vt > 1e-3 else np.array([1., 0., 0.])
        Vt_new = Vt + dt * np.dot(acceleration_ned, V_unit_vec_ned)
        # 计算 NED 坐标下的位移
        pos_ned_new = pos_ned + V_vec_ned * dt

        # 返回新的状态向量 (NUE 格式)
        return np.array(
            [pos_ned_new[0], -pos_ned_new[2], pos_ned_new[1], Vt_new, theta_new, psi_new, phi_new, p_real_new])