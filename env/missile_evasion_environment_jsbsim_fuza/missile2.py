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

        # --- <<< 新增 >>> 最优导引律(OGL) 参数 ---
        self.K1 = 8.0  # 纵向增益
        self.K2 = 8.0  # 侧向增益
        self.lambda1_ = 0.1  # 能量/脱靶量 权重系数

        # --- 导引头/引信参数 ---
        self.seeker_max_range = 30000.0  # 导引头最大搜索范围 (m)
        self.seeker_fov_rad = np.deg2rad(90)  # 导引头最大视场角度 (弧度)
        self.seeker_max_omega_rad_s = np.deg2rad(90.0)#12.0  # 导引头最大角速度 (弧度/秒)
        self.seeker_max_time = 60.0  # 导引头最大搜索时间 (秒)

        # --- 历史状态记录 (用于计算导引律所需的微分量) ---
        self.prev_theta_L = None
        self.prev_phi_L = None

    def update_OGL(self,
                   dt: float,
                   target_pos: np.ndarray,
                   target_vel: np.ndarray,
                   target_pos_equiv: np.ndarray = None):
        """
        基于最优导引律 (OGL, Optimal Guidance Law) 更新导弹状态。
        该版本特点：
          1) 视线角 (LOS angles) 使用 atan2 定义，更数值稳定：
             - phi_L   = atan2(Rz, Rx)                     # 偏航视线角
             - theta_L = atan2(Ry, sqrt(Rx^2 + Rz^2))      # 俯仰视线角
          2) 视线角速率 (LOS rates) 使用解析法（相对速度直接求导），无需历史差分：
             - phi_L_dot   = (Rx*Vz_rel - Rz*Vx_rel)/(Rx^2+Rz^2)
             - theta_L_dot = (Rh*Vy_rel - Ry*Rh_dot) / (R^2)
               其中 Rh_dot = (Rx*Vx_rel + Rz*Vz_rel)/Rh
          3) t_go 使用“物理目标”计算的接近率 real_Rdot，避免诱饵等效点导致 t_go 乱跳。
          4) 使用 effective_V = max(|real_Rdot|, 0.5*missile_speed) 防止低接近率下增益塌陷。
          5) 失锁时：只保留重力补偿 ny=cos(theta)，侧向 nz=0。

        Args:
            dt: 时间步长 (s)
            target_pos: 目标真实物理位置 (用于真实 Rdot / t_go)
            target_vel: 目标真实速度向量
            target_pos_equiv: 导引头看到的“等效目标点”（有干扰/诱饵时会用它算 LOS）
                              若为 None 表示失锁（或不机动）。
        """

        # -----------------------------
        # 0) 基础保护
        # -----------------------------
        if dt <= 1e-6:
            # dt 太小会导致积分/数值异常，直接不更新（或你也可以强制用一个最小 dt）
            return

        # -----------------------------
        # 1) 选择“导引目标点”
        #    - guide_target_pos：用于 LOS 角与 LOS 角速率（导引头几何）
        #    - phys_target_pos ：用于真实 Rdot / t_go（拦截时间估计更可靠）
        # -----------------------------
        guide_target_pos = target_pos_equiv if target_pos_equiv is not None else target_pos
        phys_target_pos = target_pos

        # -----------------------------
        # 2) 解包导弹状态
        # -----------------------------
        V, theta, psi_c = float(self.state[0]), float(self.state[1]), float(self.state[2])

        # -----------------------------
        # 3) 相对位置 / 速度（用于 LOS 与 LOS rate）
        # -----------------------------
        # r_g: 导引几何用的相对位置（指向等效点或真实点）
        r_g = guide_target_pos - self.pos
        Rx, Ry, Rz = float(r_g[0]), float(r_g[1]), float(r_g[2])

        # 水平距离 Rh = sqrt(Rx^2 + Rz^2)
        Rh_sq = Rx * Rx + Rz * Rz
        Rh = np.sqrt(Rh_sq)

        # 总距离 R = ||r||
        R_sq = Rh_sq + Ry * Ry
        R = np.sqrt(R_sq)

        # 防止除 0
        if R < 1e-6:
            # 导弹已经到达目标附近（或数值异常），此时 OGL 意义不大
            # 你可以在外层做命中判定，这里直接给一个稳定更新
            ny = np.cos(theta)
            nz = 0.0
            self.state = self._missile_dynamics_integration(self.state, dt, ny, nz)
            return

        # 相对速度：目标速度 - 导弹速度（解析 LOS rate 需要）
        missile_vel_vec = self.get_velocity_vector()
        missile_speed = float(self.velocity)  # 你原代码一直用 self.velocity
        v_rel = target_vel - missile_vel_vec
        Vx_rel, Vy_rel, Vz_rel = float(v_rel[0]), float(v_rel[1]), float(v_rel[2])

        # -----------------------------
        # 4) 计算 LOS 角（用于调试/记录；解析 rate 不依赖差分）
        #    采用 atan2 定义，避免 asin 在高仰角附近敏感
        # -----------------------------
        phi_L = np.arctan2(Rz, Rx)  # 偏航视线角
        theta_L = np.arctan2(Ry, max(Rh, 1e-9))  # 俯仰视线角（Rh=0 时保护）

        # -----------------------------
        # 5) 计算 LOS 角速率（解析法）
        # -----------------------------
        # 5.1 偏航视线角速率 phi_L_dot
        #     phi = atan2(Rz,Rx) => phi_dot = (Rx*Vz - Rz*Vx)/(Rx^2+Rz^2)
        if Rh_sq < 1e-12:
            # 目标几乎在正上/正下方，偏航角不可定义（奇异）
            phi_L_dot = 0.0
        else:
            phi_L_dot = (Rx * Vz_rel - Rz * Vx_rel) / Rh_sq

        # 5.2 俯仰视线角速率 theta_L_dot
        #     theta = atan2(Ry, Rh)
        #     theta_dot = (Rh*Vy - Ry*Rh_dot) / (Rh^2 + Ry^2) = (Rh*Vy - Ry*Rh_dot)/R^2
        #     Rh_dot = (Rx*Vx + Rz*Vz)/Rh
        if Rh < 1e-9:
            # 同样处于奇异附近，直接置 0（或你也可以只保留 Vy/R 的近似）
            theta_L_dot = 0.0
        else:
            Rh_dot = (Rx * Vx_rel + Rz * Vz_rel) / Rh
            theta_L_dot = (Rh * Vy_rel - Ry * Rh_dot) / (R_sq + 1e-9)

        # （可选）如果你想限制导引头角速度能力，可加限幅：
        # omega_los = sqrt(theta_L_dot^2 + (phi_L_dot*cos(theta_L))^2) 等定义很多
        # 这里不强加，避免改变你原系统行为。

        # -----------------------------
        # 6) 计算真实接近率 real_Rdot（用于 t_go）
        #    用真实目标位置（phys_target_pos）更稳，不被等效点跳变影响
        # -----------------------------
        r_phys = guide_target_pos - self.pos
        R_phys = np.linalg.norm(r_phys) + 1e-9
        # real_Rdot = (r · v_rel) / ||r||
        # 注意：当 real_Rdot < 0 表示在接近（距离在减小）
        real_Rdot = float(np.dot(r_phys, v_rel) / R_phys)

        # -----------------------------
        # 7) 估算 t_go（保持你原来的“低接近率保底”思想）
        # -----------------------------
        # 如果接近率太小或在远离（real_Rdot > -10），用“用自身速度飞完距离”的虚拟时间
        # 否则用物理公式 t_go = -R/real_Rdot (real_Rdot 为负)
        if real_Rdot > -10.0:
            t_go = R_phys / (missile_speed + 1e-9)
        else:
            t_go = -R_phys / (real_Rdot - 1e-9)

        # 防止 t_go 过小导致 (3*lambda1 + t_go^3) 数值不稳定
        t_go = max(0.5, float(t_go))

        # -----------------------------
        # 8) OGL 计算过载指令 ny, nz
        # -----------------------------
        # 失锁：不机动，只做重力补偿（保持基本弹道稳定）
        if target_pos_equiv is None:
            ny = float(np.cos(theta))
            nz = 0.0
            self.state = self._missile_dynamics_integration(self.state, dt, ny, nz)
            return

        # 锁定：执行 OGL
        denom = self.g * (3.0 * self.lambda1_ + t_go ** 3)

        # effective_V：保证低接近率/大离轴时仍有足够“转弯增益”
        effective_V = max(abs(real_Rdot), 0.5 * missile_speed)

        # ny：纵向过载（含重力补偿 +cos(theta)）
        ny = (self.K1 * (t_go ** 3) * effective_V * theta_L_dot) / (denom + 1e-9) + np.cos(theta)

        # nz：侧向过载
        nz = (self.K2 * (t_go ** 3) * effective_V * phi_L_dot) / (denom + 1e-9)

        # -----------------------------
        # 9) 交给统一的动力学积分（内部包含过载限幅、气动阻力等）
        # -----------------------------
        self.state = self._missile_dynamics_integration(self.state, dt, float(ny), float(nz))

        # （可选）如需记录 LOS 角用于调试/输出：
        # self.prev_theta_L = theta_L
        # self.prev_phi_L   = phi_L

    # def update_OGL(self, dt: float, target_pos: np.ndarray, target_vel: np.ndarray, target_pos_equiv=None):
    #     # 1. 确定导引目标点
    #     if target_pos_equiv is not None:
    #         guide_target_pos = target_pos_equiv
    #     else:
    #         guide_target_pos = target_pos
    #
    #     # 解包导弹状态
    #     V, theta, psi_c = self.state[0], self.state[1], self.state[2]
    #     x_m, y_m, z_m = self.state[3], self.state[4], self.state[5]
    #
    #     # 2. 计算相对位置矢量 (R_vec)
    #     R_vec = guide_target_pos - self.pos
    #     Rx, Ry, Rz = R_vec[0], R_vec[1], R_vec[2]
    #
    #     # 水平距离平方 (用于计算偏航率)
    #     R_horiz_sq = Rx ** 2 + Rz ** 2
    #     # 总距离平方
    #     R_sq = R_horiz_sq + Ry ** 2
    #     R = np.sqrt(R_sq) + 1e-6
    #
    #     # 3. 计算视线角 (LOS Angles) - 仅用于记录或调试，解析法计算Rate不需要它
    #     theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
    #     phi_L = np.arctan2(Rz, Rx)
    #
    #     # <<< 步骤提前 >>>：先计算相对速度，因为解析法求导需要用到它
    #     missile_vel_vec = self.get_velocity_vector()
    #     missile_speed = self.velocity
    #
    #     # 注意：如果是追踪红外质心(equiv)，严格来说应该用质心的移动速度。
    #     # 但质心速度很难求(因为质心会跳变)，这里用物理目标速度近似是工程上最稳健的做法。
    #     V_rel = target_vel - missile_vel_vec
    #     Vx_rel, Vy_rel, Vz_rel = V_rel[0], V_rel[1], V_rel[2]
    #
    #     # 计算接近速率 (R_dot) = (R . V_rel) / |R|
    #     # 用于后续 t_go 计算和 theta_L_dot 计算
    #     Rdot = (Rx * Vx_rel + Ry * Vy_rel + Rz * Vz_rel) / R
    #
    #     # 4. <<< 核心修改：使用解析法计算视线角速率 >>>
    #
    #     # --- 计算偏航视线角速率 (phi_L_dot) ---
    #     # 公式: d(arctan(z/x))/dt = (x*z_dot - z*x_dot) / (x^2 + z^2)
    #     if R_horiz_sq < 1e-6:
    #         phi_L_dot = 0.0  # 就在正上方/正下方，无偏航概念
    #     else:
    #         phi_L_dot = (Rx * Vz_rel - Rz * Vx_rel) / R_horiz_sq
    #
    #     # --- 计算俯仰视线角速率 (theta_L_dot) ---
    #     # 公式: d(arcsin(y/R))/dt = (R*y_dot - y*R_dot) / (R * sqrt(x^2+z^2))
    #     R_horiz = np.sqrt(R_horiz_sq)
    #     if R_horiz < 1e-6:
    #         theta_L_dot = 0.0
    #     else:
    #         # 分子: R * Vy_rel - Ry * Rdot
    #         numerator = R * Vy_rel - Ry * Rdot
    #         # 分母: R^2 * cos(theta_L) = R * R_horiz
    #         denominator = R * R_horiz
    #         theta_L_dot = numerator / denominator
    #
    #     # 更新历史视线角 (虽然解析法不用它计算Rate，但为了保持状态一致性还是更新一下)
    #     self.prev_theta_L = theta_L
    #     self.prev_phi_L = phi_L
    #
    #     # 5. (原来的步骤5已合并到上面)
    #     # 但我们需要保留计算 t_go 用的 "物理 Rdot"
    #     # 你的原代码区分了 "guide_target" 和 "phys_target"。
    #     # 如果 guide_target 就是 phys_target，那上面的 Rdot 就是物理 Rdot。
    #     # 如果有诱饵弹干扰，建议还是重新算一下物理 Rdot 用于 t_go 估算，或者直接用上面的 Rdot 近似也行。
    #     # 为了保险，保留你原来的物理 Rdot 计算逻辑用于 t_go：
    #
    #     R_vec_phys = target_pos - self.pos
    #     R_phys = np.linalg.norm(R_vec_phys) + 1e-6
    #     real_Rdot = np.dot(R_vec_phys, V_rel) / R_phys
    #
    #     # 6. 估算剩余飞行时间 (t_go) - 保持之前的修正逻辑
    #     if real_Rdot > -10.0:
    #         t_go = R_phys / (missile_speed + 1e-6)
    #     else:
    #         t_go = -R_phys / (real_Rdot - 1e-6)
    #     t_go = max(0.5, t_go)
    #
    #     # 7. OGL 核心公式 - 保持之前的修正逻辑
    #     if target_pos_equiv is not None:
    #         denom = self.g * (3 * self.lambda1_ + t_go ** 3)
    #
    #         # 使用有效速度保底
    #         effective_V = max(abs(real_Rdot), 0.5 * missile_speed)
    #
    #         ny = (self.K1 * t_go ** 3 * effective_V * theta_L_dot) / denom + np.cos(theta)
    #         nz = (self.K2 * t_go ** 3 * effective_V * phi_L_dot) / denom
    #     else:
    #         ny = np.cos(theta)
    #         nz = 0.0
    #
    #     # 8. 动力学积分
    #     self.state = self._missile_dynamics_integration(self.state, dt, ny, nz)

    # def update_OGL(self, dt: float, target_pos: np.ndarray, target_vel: np.ndarray, target_pos_equiv=None):
    #     """
    #     <<< 新增 >>> 基于最优导引律(OGL)的状态更新函数。
    #
    #     Args:
    #         dt: 时间步长
    #         target_pos: 目标真实物理位置 [x, y, z] (用于计算 R_dot)
    #         target_vel: 目标真实速度矢量 [vx, vy, vz] (用于计算 R_dot)
    #         target_pos_equiv: 导引头看到的等效目标位置 (用于计算视线角)
    #                           如果为None (失锁)，则默认沿当前方向或飞向最后已知点。
    #     """
    #     # 1. 确定导引目标点 (如果有红外干扰，导引头指向等效质心；否则指向真实目标)
    #     if target_pos_equiv is not None:
    #         guide_target_pos = target_pos_equiv
    #     else:
    #         guide_target_pos = target_pos
    #
    #     # print("target_pos_equiv", target_pos_equiv)
    #
    #     # 解包导弹状态
    #     V, theta, psi_c = self.state[0], self.state[1], self.state[2]
    #     x_m, y_m, z_m = self.state[3], self.state[4], self.state[5]
    #
    #     # 2. 计算相对运动学参数
    #     # R_vec: 指向导引目标
    #     R_vec = guide_target_pos - self.pos
    #     R = np.linalg.norm(R_vec) + 1e-6
    #
    #     # 3. 计算视线角 (LOS Angles)
    #     # theta_L: 俯仰视线角, phi_L: 偏航视线角
    #     # 注意：这里假设 y轴为高 (North-Up-East 或 similar)
    #     Rx, Ry, Rz = R_vec[0], R_vec[1], R_vec[2]
    #     theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
    #     phi_L = np.arctan2(Rz, Rx)
    #
    #     # 4. 计算视线角速率 (LOS Rates)
    #     if self.prev_theta_L is None:
    #         theta_L_dot = 0.0
    #         phi_L_dot = 0.0
    #     else:
    #         theta_L_dot = (theta_L - self.prev_theta_L) / dt
    #         # 处理角度跳变 (-pi 到 pi)
    #         dphi = np.arctan2(np.sin(phi_L - self.prev_phi_L), np.cos(phi_L - self.prev_phi_L))
    #         phi_L_dot = dphi / dt
    #
    #     # 更新历史视线角
    #     self.prev_theta_L = theta_L
    #     self.prev_phi_L = phi_L
    #
    #     # 5. 计算接近速率 (R_dot) 用于估算 t_go
    #     # 注意：R_dot 需要用到真实的物理相对速度，而不是等效目标的
    #     missile_vel_vec = self.get_velocity_vector()
    #     missile_speed = self.velocity  # 获取导弹自身速率 (例如 900m/s)
    #     V_rel = target_vel - missile_vel_vec
    #     # R_dot = (R_vec . V_rel) / R
    #     # 这里为了稳健性，使用物理目标的相对位置来计算 R_dot 方向
    #     R_vec_phys = target_pos - self.pos
    #     R_phys = np.linalg.norm(R_vec_phys) + 1e-6
    #     Rdot = np.dot(R_vec_phys, V_rel) / R_phys
    #
    #     # print("Rdot", Rdot)
    #
    #     # 6. 估算剩余飞行时间 (t_go)
    #
    #     # <<< 修改开始：智能 t_go 计算 >>>
    #     # 如果 Rdot > -10 (表示正在远离，或者接近速度极慢)
    #     if Rdot > -10.0:
    #         # 此时物理上的撞击时间是无穷大或未定义的。
    #         # 我们给导引律一个 "虚拟 t_go"：假设导弹以当前全速飞完这段距离所需的时间。
    #         # 这样导引律会认为：“还有很长路要走，我有足够时间修正，赶紧掉头！”
    #         t_go = R_phys / (missile_speed + 1e-6)
    #     else:
    #         # 正常接近中，使用物理公式
    #         t_go = -R_phys / (Rdot - 1e-6)
    #
    #     # 防止 t_go 过小导致奇异值 (保留这个保底)
    #     t_go = max(0.5, t_go)
    #     # <<< 修改结束 >>>
    #
    #     # # 如果 Rdot >= 0 (远离)，t_go 设为一个较小正数防止错误，或者保持导引
    #     # t_go = max(0.1, - R / (Rdot - 1e-6))
    #     # print("t_go", t_go)
    #
    #     # 7. --- 最优导引律 (OGL) 核心公式 ---
    #     # 移植自 AirCombatEnv3_radar.py
    #     if target_pos_equiv is not None:  # 只有锁定状态下才机动
    #         denom = self.g * (3 * self.lambda1_ + t_go ** 3)
    #
    #         # <<< 最佳修改方案 >>>
    #         # 使用 "Effective Velocity" (有效导引速度)
    #         # 逻辑：如果接近率太低（比如大离轴角追尾），就用导弹自身速度的一半来代入计算。
    #         # 这样能保证在任何角度下，导弹都有足够的增益去转弯。
    #         effective_V = max(abs(Rdot), 0.5 * missile_speed)
    #         # 代入 effective_V
    #         ny = (self.K1 * t_go ** 3 * effective_V * theta_L_dot) / denom + np.cos(theta)
    #         nz = (self.K2 * t_go ** 3 * effective_V * phi_L_dot) / denom
    #
    #         # # 纵向过载 (含重力补偿: + cos(theta))
    #         # # 注意: abs(Rdot) 确保在接近时增益为正
    #         # ny = (self.K1 * t_go ** 3 * abs(Rdot) * theta_L_dot) / denom + np.cos(theta)
    #         #
    #         # # 侧向过载
    #         # nz = (self.K2 * t_go ** 3 * abs(Rdot) * phi_L_dot) / denom
    #     else:
    #         # 失锁状态：保持重力补偿，侧向不机动
    #         ny = np.cos(theta)
    #         nz = 0.0
    #
    #     # print("ny, nz", ny, nz)
    #
    #     # 8. 执行动力学积分 (复用内部逻辑)
    #     self.state = self._missile_dynamics_integration(self.state, dt, ny, nz)

    def _missile_dynamics_integration(self, state, dt, ny, nz):
        """
        内部私有方法：给定过载 ny, nz，执行物理积分
        (结合了 AIM-9X 的气动模型)
        """
        V, theta, psi_c, x_pos, y_pos, z_pos = state

        # --- 过载限幅 ---
        n_total_cmd = np.sqrt(ny ** 2 + nz ** 2)
        if n_total_cmd > self.max_g_load:
            scale = self.max_g_load / n_total_cmd
            ny *= scale
            nz *= scale

        # --- 气动阻力计算 (AIM-9X 模型) ---
        H = y_pos
        Temper = 15.0
        T_H = 273 + Temper - 0.6 * H / 100
        # 简单大气模型保护
        if H < 44300:
            P_H = (1 - H / 44300) ** 5.256
        else:
            P_H = 0
        rho = 1.293 * P_H * (273 / (T_H + 0.01))
        Ma = V / 340.0

        # 阻力系数函数 (内嵌)
        def get_Cx_AIM9X(m):
            if m <= 0.9:
                return 0.20
            elif m <= 1.2:
                return 0.20 + (0.38 - 0.20) * (m - 0.9) / 0.3
            elif m <= 4.0:
                return 0.38 * np.exp(-0.35 * (m - 1.2)) + 0.15
            else:
                return 0.15

        Cx = get_Cx_AIM9X(Ma)
        S = np.pi * (self.diameter ** 2) / 4
        q = 0.5 * rho * V ** 2
        F_drag = Cx * q * S

        # --- 动力学方程 ---
        # v_dot = (推力 - 阻力 - 重力沿速度方向分量) / m
        # 假设无推力(滑翔阶段)
        v_dot = -F_drag / self.mass - self.g * np.sin(theta)

        # 采用 File 2 的更新策略：更稳定的角度更新
        # theta_dot * V = a_n - g*cos(theta) -> a_n = ny * g
        # 所以: theta_next = theta + (ny*g - g*cos(theta))/V * dt
        # 化简后即为下面代码:
        V_next = V + v_dot * dt + 1e-8  # 防止V=0

        theta_next = theta + ((ny - np.cos(theta)) * self.g / V_next) * dt

        # psi_dot * V * cos(theta) = nz * g
        cos_theta_safe = np.cos(theta) if abs(np.cos(theta)) > 0.01 else 0.01
        psi_c_next = psi_c + (nz * self.g / (V_next * cos_theta_safe)) * dt

        # --- 限制与归一化 ---
        theta_next = np.clip(theta_next, -np.deg2rad(89), np.deg2rad(89))
        if psi_c_next > np.pi: psi_c_next -= 2 * np.pi
        if psi_c_next < -np.pi: psi_c_next += 2 * np.pi

        # --- 位置更新 ---
        # 使用更新后的速度矢量进行位置积分
        vx = V_next * np.cos(theta_next) * np.cos(psi_c_next)
        vy = V_next * np.sin(theta_next)
        vz = V_next * np.cos(theta_next) * np.sin(psi_c_next)

        pos_next = np.array([x_pos, y_pos, z_pos]) + np.array([vx, vy, vz]) * dt

        return np.array([V_next, theta_next, psi_c_next, *pos_next])

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

        def get_Cx_AIM9X(Ma: float) -> float:
            if Ma <= 0.9:
                return 0.20
            elif Ma <= 1.2:
                return 0.20 + (0.38 - 0.20) * (Ma - 0.9) / 0.3
            elif Ma <= 4.0:
                return 0.38 * np.exp(-0.35 * (Ma - 1.2)) + 0.15
            else:
                return 0.15

        # # AIM-9X 阻力系数模型 更大
        # def get_Cx_AIM9X(Ma: float) -> float:
        #     """
        #     根据马赫数 Ma 返回导弹阻力系数 Cd
        #     分段拟合：
        #       1) Ma <= 0.8 : 常数
        #       2) 0.8 < Ma <= 1.1 : 线性
        #       3) 1.1 < Ma <= 4.0 : 三次多项式
        #     """
        #     Ma1, Ma2, Ma3 = 0.8, 1.1, 4.0
        #
        #     if Ma <= Ma1:
        #         return 0.5
        #     elif Ma <= Ma2:
        #         # 线性段 (0.8,0.5) -> (1.1,0.78)
        #         return 0.9333 * Ma - 0.2466
        #     elif Ma <= Ma3:
        #         # 三次多项式段
        #         return (-0.0004537 * Ma ** 3
        #                 + 0.0290835 * Ma ** 2
        #                 - 0.2867967 * Ma
        #                 + 1.0608893)
        #     else:
        #         # 超出拟合范围可选择外推或固定
        #         return (-0.0004537 * Ma ** 3
        #                 + 0.0290835 * Ma ** 2
        #                 - 0.2867967 * Ma
        #                 + 1.0608893)

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