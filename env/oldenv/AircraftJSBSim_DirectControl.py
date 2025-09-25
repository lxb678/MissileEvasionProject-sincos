# 文件: AircraftJSBSim_DirectControl.py

import jsbsim
import numpy as np

# --- (新增) 定义单位转换常量 ---
M_TO_FT = 3.28084
MPS_TO_KTS = 1.94384
RAD_TO_DEG = 180.0 / np.pi


class Aircraft:
    """
    一个使用JSBSim作为飞行动力学模型的Aircraft类。
    <<< 适配器模式 >>>：此类负责处理所有与JSBSim的单位转换，
    对外部环境暴露统一的单位制（米，米/秒，弧度）。
    """

    def __init__(self, dt: float, initial_state: np.ndarray):
        """
        初始化 JSBSim FDM。

        Args:
            dt (float): 仿真时间步长 (秒)。
            initial_state (np.ndarray): 来自环境的初始状态向量，使用标准单位：
                [x(北,m), y(天,m), z(东,m), Vt(m/s), theta(rad), psi(rad), phi(rad), p_real(rad/s)]
        """
        """
                初始化 JSBSim FDM。
                <<< 修正版 v2: 保证正确的 JSBSim 初始化顺序 >>>
                """
        # --- 1. 创建 FDM 实例并加载模型 ---
        # 直接创建，不做重定向
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)  # 只关闭后续调试信息

        # 确保飞机模型文件存在。'f16' 是 jsbsim 库自带的。
        # 如果您使用自定义模型，请确保路径正确。
        if not self.fdm.load_model('f16'):
            raise RuntimeError(f"未能加载 JSBSim 飞机模型: f16")

        self.fdm.set_dt(dt)
        self.physical_dt = dt

        # --- 2. 准备从环境传入的初始状态 ---
        self.state = np.array(initial_state, dtype=float)

        # 解包并进行单位转换
        _x, y_m, _z, vt_mps, theta_rad, psi_rad, phi_rad, _p = initial_state
        h_ft = y_m * M_TO_FT
        vc_kts = vt_mps * MPS_TO_KTS
        theta_deg = theta_rad * RAD_TO_DEG
        psi_deg = psi_rad * RAD_TO_DEG
        phi_deg = phi_rad * RAD_TO_DEG

        # --- 3. 设置所有【初始条件 (ic)】属性 ---
        self.fdm['ic/h-sl-ft'] = h_ft
        self.fdm['ic/vc-kts'] = vc_kts
        self.fdm['ic/theta-deg'] = theta_deg
        self.fdm['ic/psi-true-deg'] = psi_deg
        self.fdm['ic/phi-deg'] = phi_deg

        # --- 4. 【关键】调用 run_ic() 来构建飞机内部模型 ---
        # 这一步会根据上面的 'ic/' 属性创建发动机、气动面等所有组件。
        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim initial conditions failed. 检查初始值是否在合理范围内。")

        # --- 5. 【关键】在 run_ic() 成功后，才能操作飞机组件，如启动发动机 ---
        # 此时，发动机模型已经存在，可以安全调用。
        self.fdm['propulsion/engine[0]/set-running'] = 1


        # --- 6. 同步我们自己的状态向量 ---
        self._update_state_from_jsbsim()

    def _update_state_from_jsbsim(self):
        """
        从 JSBSim 提取数据，转换为标准单位，并填充 self.state。
        """
        # --- 位置更新 ---
        # 我们继续使用速度积分的方式来更新NUE位置，因为这与环境的逻辑一致
        vel_nue = self.get_velocity_vector_from_jsbsim()  # 使用一个专用函数获取JSBSim的原始速度
        self.state[0] += vel_nue[0] * self.physical_dt
        self.state[1] += vel_nue[1] * self.physical_dt
        self.state[2] += vel_nue[2] * self.physical_dt

        # --- (核心更改) 状态更新与单位转换 ---
        # 从JSBSim获取值（节, 英尺, 弧度），然后转换为（米/秒, 米, 弧度）
        self.state[3] = self.fdm['velocities/vt-fps'] * 0.3048  # Vt (m/s)
        self.state[4] = self.fdm['attitude/theta-rad']  # theta (rad)
        self.state[5] = self.fdm['attitude/psi-rad']  # psi (rad)
        self.state[6] = self.fdm['attitude/phi-rad']  # phi (rad)
        self.state[7] = self.fdm['velocities/p-rad_sec']  # p_real (rad/s)

        # (可选) 我们可以覆盖积分的位置，直接使用JSBSim的高度
        # self.state[1] = self.fdm['position/h-sl-ft'] / M_TO_FT

    def update(self, action: list):
        """
        根据AI的底层控制面指令，更新飞机状态。

        Args:
            action (list): [油门, 升降舵, 副翼, 方向舵]。值在 [-1, 1] 或 [0, 1] 范围。
                           这个方法不需要单位转换，因为指令是归一化的。
        """
        throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd = action

        # 直接将归一化的动作指令发送给 JSBSim
        self.fdm['fcs/throttle-cmd-norm'] = np.clip(throttle_cmd, 0.0, 1.0)
        self.fdm['fcs/elevator-cmd-norm'] = np.clip(elevator_cmd, -1.0, 1.0)
        self.fdm['fcs/aileron-cmd-norm'] = np.clip(aileron_cmd, -1.0, 1.0)
        self.fdm['fcs/rudder-cmd-norm'] = np.clip(rudder_cmd, -1.0, 1.0)

        # 运行一步 JSBSim 仿真
        self.fdm.run()

        # 从 JSBSim 的新状态更新我们自己的标准单位状态向量
        self._update_state_from_jsbsim()

    # --- 属性访问器 (对环境暴露标准单位) ---
    @property
    def state_vector(self) -> np.ndarray:
        """返回完整的、使用标准单位的状态向量"""
        return self.state

    @property
    def pos(self) -> np.ndarray:
        """返回飞机的位置向量 [x, y, z] (北, 天, 东)，单位：米"""
        return self.state[:3]

    @property
    def velocity(self) -> float:
        """返回飞机的总速度大小 (Vt)，单位：米/秒"""
        return self.state[3]

    @property
    def roll_rate_rad_s(self) -> float:
        """返回飞机的实际滚转角速度 (p_real)，单位：弧度/秒"""
        return self.state[7]

    @property
    def attitude_rad(self) -> tuple:
        """返回飞机的姿态（theta, psi, phi），单位：弧度"""
        return self.state[4], self.state[5], self.state[6]

    @property
    def nz(self) -> float:
        """返回飞行员感受到的法向G过载 (无单位)"""
        return self.fdm["accelerations/nz-pilot-g's"]

    # --- (新增) 辅助方法 ---
    def get_velocity_vector(self) -> np.ndarray:
        """
        计算并返回飞机在世界坐标系(NUE)下的速度矢量，单位：米/秒。
        这个方法现在基于我们自己维护的、单位正确的 state 向量。
        """
        Vt, theta, psi = self.state[3], self.state[4], self.state[5]
        vx = Vt * np.cos(theta) * np.cos(psi)  # 北向速度 (m/s)
        vy = Vt * np.sin(theta)  # 天向速度 (m/s)
        vz = Vt * np.cos(theta) * np.sin(psi)  # 东向速度 (m/s)
        return np.array([vx, vy, vz])

    def get_velocity_vector_from_jsbsim(self) -> np.ndarray:
        """
        一个内部辅助函数，直接从JSBSim获取速度矢量并转换为NUE(m/s)。
        主要用于 _update_state_from_jsbsim 中的位置积分。
        """
        # JSBSim 的机体速度分量 (前, 右, 下), 单位：英尺/秒
        u_fps = self.fdm['velocities/u-fps']
        v_fps = self.fdm['velocities/v-fps']
        w_fps = self.fdm['velocities/w-fps']

        # 转换为 m/s
        u, v, w = u_fps * 0.3048, v_fps * 0.3048, w_fps * 0.3048

        # 获取姿态角 (弧度)
        phi, theta, psi = self.fdm['attitude/phi-rad'], self.fdm['attitude/theta-rad'], self.fdm['attitude/psi-rad']

        # 从机体坐标系 (FRD) 到惯性系 (NED) 的旋转矩阵
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_phi, s_phi = np.cos(phi), np.sin(phi)

        R_b_to_i = np.array([
            [c_th * c_psi, s_phi * s_th * c_psi - c_phi * s_psi, c_phi * s_th * c_psi + s_phi * s_psi],
            [c_th * s_psi, s_phi * s_th * s_psi + c_phi * c_psi, c_phi * s_th * s_psi - s_phi * c_psi],
            [-s_th, s_phi * c_th, c_phi * c_th]
        ])

        v_ned = R_b_to_i @ np.array([u, v, w])  # (北, 东, 地) in m/s

        # 从 NED (北-东-地) 转换到 NUE (北-天-东)
        v_nue = np.array([v_ned[0], -v_ned[2], v_ned[1]])
        return v_nue