# 文件: AircraftJSBSim_DirectControl.py
import jsbsim
import numpy as np

# --- 单位转换常量 ---
M_TO_FT  = 3.28084
MPS_TO_KTS = 1.94384
RAD_TO_DEG = 180.0 / np.pi
FT_TO_M   = 0.3048
FPS_TO_MPS = 0.3048

class Aircraft:
    """
    使用 JSBSim 作为飞行动力学模型的适配器类。
    向外暴露统一的标准单位 (米、米/秒、弧度)。
    """
    def __init__(self, dt: float, initial_state: np.ndarray):
        self.dt = dt
        self.state = np.zeros(8, dtype=float)  # [x,y,z,Vt,theta,psi,phi,p_real]

        # --- 1. 创建并初始化 FDM 实例 ---
        self.fdm = jsbsim.FGFDMExec(None)   # 只创建一次
        self.fdm.set_debug_level(0)         # 关闭后续调试输出

        if not self.fdm.load_model('f16'):
            raise RuntimeError("未能加载 JSBSim 飞机模型: f16")

        self.fdm.set_dt(dt)
        # 初始化一次
        self.reset(initial_state)

    # ------------------------------------------------------------------
    def reset(self, initial_state: np.ndarray):
        """
        重新加载初始条件，而不重新创建 FDM。
        可以在环境的 reset() 中调用。
        """
        self.state[:] = np.array(initial_state, dtype=float)
        x_m, y_m, z_m, vt_mps, theta_rad, psi_rad, phi_rad, p_rad_s = initial_state

        # 位姿/高度（保持你原来的写法）
        self.fdm['ic/h-sl-ft'] = y_m * M_TO_FT
        self.fdm['ic/theta-deg'] = theta_rad * RAD_TO_DEG
        self.fdm['ic/psi-true-deg'] = psi_rad * RAD_TO_DEG
        self.fdm['ic/phi-deg'] = phi_rad * RAD_TO_DEG

        # ---- 关键：选择正确的 airspeed IC ----
        # 如果 initial_state[3] 表示 真空速 (TAS, m/s) -> 写 ic/vt-kts
        self.fdm['ic/vt-kts'] = vt_mps * MPS_TO_KTS

        # 如果 initial_state[3] 表示 仪表速度/校准空速 (CAS) -> 写 ic/vc-kts
        # self.fdm['ic/vc-kts'] = vt_mps * MPS_TO_KTS

        # ---- 清除/固定风场（排除风的影响） ----
        self.fdm['atmosphere/wind-north-fps'] = 0.0
        self.fdm['atmosphere/wind-east-fps'] = 0.0
        self.fdm['atmosphere/wind-down-fps'] = 0.0

        # 运行初始化（run_ic 可能会调整内部状态）
        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() 失败，请检查初始条件。")

        # 启动发动机（注意：set-running 不一定会自动给出推力，详见下面说明）
        self.fdm['propulsion/engine[0]/set-running'] = -1

        # 读取并打印 JSBSim 的真实速度（供调试）
        vt_fps = float(self.fdm['velocities/vt-fps'])  # true airspeed (ft/s)
        # print(f"After run_ic: vt = {vt_fps * FPS_TO_MPS:.2f} m/s  (vt-fps={vt_fps:.2f} fps)")

        return self.state

    # ------------------------------------------------------------------
    def update(self, action: list):
        """
        根据 AI 动作更新飞机状态。
        action: [throttle, elevator, aileron, rudder]
        """
        throttle, elevator, aileron, rudder = action
        self.fdm['fcs/throttle-cmd-norm'] = np.clip(throttle, 0.0, 1.0)
        self.fdm['fcs/elevator-cmd-norm'] = np.clip(elevator, -1.0, 1.0)
        self.fdm['fcs/aileron-cmd-norm']  = np.clip(aileron, -1.0, 1.0)
        self.fdm['fcs/rudder-cmd-norm']   = np.clip(rudder, -1.0, 1.0)

        # 运行一步 JSBSim
        self.fdm.run()

        # 更新状态向量
        self._update_state_from_jsbsim()

    # ------------------------------------------------------------------
    def _update_state_from_jsbsim(self):
        """
        从 JSBSim 提取数据，更新标准单位状态向量。
        """
        vel_nue = self._get_velocity_vector_from_jsbsim()
        self.state[0:3] += vel_nue * self.dt

        self.state[3] = self.fdm['velocities/vt-fps'] * FPS_TO_MPS
        self.state[4] = self.fdm['attitude/theta-rad']
        self.state[5] = self.fdm['attitude/psi-rad']
        self.state[6] = self.fdm['attitude/phi-rad']
        self.state[7] = self.fdm['velocities/p-rad_sec']

    # ------------------------------------------------------------------
    def _get_velocity_vector_from_jsbsim(self):
        """
        从JSBSim直接获取速度向量(世界坐标NUE)，单位 m/s。
        """
        u = self.fdm['velocities/u-fps'] * FPS_TO_MPS
        v = self.fdm['velocities/v-fps'] * FPS_TO_MPS
        w = self.fdm['velocities/w-fps'] * FPS_TO_MPS
        phi = self.fdm['attitude/phi-rad']
        theta = self.fdm['attitude/theta-rad']
        psi = self.fdm['attitude/psi-rad']

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R_b2i = np.array([
            [cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth,    sphi*cth,                  cphi*cth]
        ])
        v_ned = R_b2i @ np.array([u, v, w])
        # NED → NUE
        return np.array([v_ned[0], -v_ned[2], v_ned[1]])

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

    def get_normal_g_force(self) -> float:
        """
        直接从JSBSim获取法向过载(nz)。
        注意JSBSim的符号约定：拉杆时nz为负。
        """
        # 属性名称可能是 'accelerations/nz-pilot-g' 或 'accelerations/nz-g'
        # 请根据您的JSBSim配置确认
        return self.fdm.get_property_value('accelerations/Nz')

    def get_total_g_force(self):
        """
        获取飞机的总过载 (Total G-Force) 以及各轴分量。
        基于飞行员位置的归一化加速度 (accelerations/n-pilot-*-norm)。

        Returns:
            total_g (float): 总过载大小 (G)
            components (tuple): (g_x, g_y, g_z) 各轴分量
        """
        # 获取三个轴向过载 (单位: G)
        # JSBSim 属性: accelerations/n-pilot-x-norm, y-norm, z-norm
        g_x = self.fdm.get_property_value('accelerations/n-pilot-x-norm')
        g_y = self.fdm.get_property_value('accelerations/n-pilot-y-norm')
        g_z = self.fdm.get_property_value('accelerations/n-pilot-z-norm')

        # 计算总过载 (使用文件头部导入的 numpy)
        total_g = np.sqrt(g_x ** 2 + g_y ** 2 + g_z ** 2)

        return total_g , (g_x, g_y, g_z)

    def beta_deg(self):
        '''
        计算并返回飞机的侧滑角 (beta)，单位：弧度。
        侧滑角定义为：beta = arcsin(v / Vt)
        其中 v 是机体侧向速度分量，Vt 是总速度。
        '''
        # 获取弧度值
        return self.fdm.get_property_value('aero/beta-rad')

