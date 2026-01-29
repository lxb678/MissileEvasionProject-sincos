# 文件: Vec_missile_evasion_environment_jsbsim.py

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- 导入原有模块 ---
from .AircraftJSBSim_DirectControl import Aircraft
from .missile2 import Missile
from .decoys import FlareManager
from .reward_system import RewardCalculator
from .tacview_interface import TacviewInterface

# --- 导入 PID 控制器 ---
# 假设您的 PID 文件名为 F16PIDController2.py，且在同一目录下
from .F16PIDController2 import F16PIDController, sub_of_degree

# --- 导入配置常量 ---
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.PPOMLP混合架构.Hybrid_PPO_混合架构雅可比修正优势归一化 import CONTINUOUS_DIM, \
    DISCRETE_DIMS, \
    DISCRETE_ACTION_MAP

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv(gym.Env):
    """
    (混合控制版) 强化学习空战环境。
    - RL Agent: 仅负责红外干扰弹投放决策 (离散动作)。
    - PID Controller: 负责飞机机动控制 (连续动作)，执行预设的规避策略。
    """

    def __init__(self, tacview_enabled=False, dt=0.02):
        super().__init__()

        # --- 1. 定义动作空间 (仅保留离散动作) ---
        # 移除 continuous_actions，只保留干扰弹投放相关的离散维度
        self.action_space = spaces.Dict({
            "discrete_actions": spaces.MultiDiscrete(
                np.array([
                    2,  # flare_trigger: 0或1 (是否开启投放程序)
                    DISCRETE_DIMS['salvo_size'],  # 投放数量选项
                    DISCRETE_DIMS['num_groups'],  # 投放组数选项
                    DISCRETE_DIMS['inter_interval'],  # 组间隔选项
                ])
            )
        })

        # --- 2. 定义观测空间 ---
        # 维度计算 (单导弹): 导弹特征(4) + 飞机自身特征(7) = 11
        # <<< 核心修改 >>>
        # 维度计算:
        # 导弹特征(5): [o_dis_min_norm, o_dis_max_norm, o_beta_sin, o_beta_cos, o_theta_L_norm]
        # 飞机特征(6): [o_av_norm, o_h_norm, o_ae_norm, o_am_sin, o_am_cos, o_ir_norm]
        # 总计: 11
        observation_dim = 11 #10 #11
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )

        # --- 3. 仿真步长参数 ---
        self.dt_normal = dt  # 大步长
        self.dt_small = dt  # 小步长
        self.dt_flare = dt  # 投放时的步长
        self.R_switch = 500  # 切换精细步长的距离阈值
        self.dt_dec = 0.2  # 决策间隔 (RL动作更新频率)
        self.physical_dt = dt  # 物理仿真基准步长

        # --- 4. 导引头与环境参数 ---
        self.t_end = 60.0
        self.R_kill = 12.0
        self.D_max = 30000.0
        self.Angle_IR_rad = np.deg2rad(90)
        self.omega_max_rad_s = np.deg2rad(90.0)
        self.T_max = 60.0

        # --- 5. 核心组件实例化 ---
        # 初始化飞机 (初始状态将在 reset 中覆盖)
        initial_state_dummy = np.array([0, 1000, 0, 200, 0.1, 0.0, 0.0, 0.0])
        self.aircraft = Aircraft(dt=self.dt_small, initial_state=initial_state_dummy)
        self.missile = None
        self.flare_manager = FlareManager(flare_per_group=1)
        self.reward_calculator = RewardCalculator()

        # <<< 初始化 PID 控制器 >>>
        self.pid_controller = F16PIDController()

        # Tacview 接口
        self.tacview_enabled = tacview_enabled
        self.tacview = TacviewInterface() if tacview_enabled else None

        # --- 6. 预设机动目标参数 (供 PID 使用) ---
        self.target_height = 6000.0  # 目标保持高度 (m)
        self.target_speed = 340.0  # 目标保持速度 (m/s)
        # 目标航向将在 step 中动态计算 (例如切向机动)

        # [新增] 初始化机动类型变量
        self.maneuver_type = 'beaming'

        # --- 状态变量 ---
        self.t_now = 0.0
        self.tacview_global_time = 0.0
        self.done = False
        self.success = False
        self.miss_distance = None
        self.o_ir = 30
        self.N_infrared = 30
        self.episode_count = 0

        # 历史记录
        self.aircraft_history = []
        self.missile_history = []
        self.time_history = []

        # 缓存变量
        self.prev_aircraft_state = None
        self.prev_missile_state = None
        self.prev_R_rel = None
        self.prev_theta_L, self.prev_phi_L = None, None
        self.missile_exploded = False

    def reset(self, seed=None, options=None) -> tuple:
        """
        重置环境到初始状态
        """
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
        super().reset(seed=seed)

        # --- 1. Tacview 清理 ---
        if self.tacview_enabled and self.episode_count > 0:
            self.tacview.end_of_episode(
                t_now=self.tacview_global_time,
                flare_count=len(self.flare_manager.flares),
                missile_exploded=self.missile_exploded
            )
            self.tacview.tacview_final_frame_sent = False

        # --- 2. 状态重置 ---
        self.t_now = 0.0
        self.done = False
        self.success = False
        self.miss_distance = None
        self.missile_exploded = False
        self.episode_count += 1
        self.o_ir = self.N_infrared
        self.flare_manager.reset()
        self.reward_calculator.reset()

        # 重置 PID 控制器内部状态 (如果有)
        # self.pid_controller = F16PIDController() # 可选：完全重置PID对象

        # --- [修改] 3. 随机选择机动策略 & 场景随机化 ---

        # A. 随机选择本回合的机动策略
        maneuver_options = ['turn_left', 'turn_right', 'dive', 'climb']
        self.maneuver_type = np.random.choice(maneuver_options)

        # B. 随机生成飞机初始高度 (不再依赖机动类型，统一随机范围)
        # 例如在 4000m 到 9000m 之间随机
        y_t = np.random.uniform(4000, 8000)

        # C. 根据机动类型设定 PID 的【目标高度】
        if self.maneuver_type == 'dive':
            # 俯冲：目标设为低空
            self.target_height = 2000.0
        elif self.maneuver_type == 'climb':
            # 爬升：目标设为高空
            self.target_height = 10000.0
        else:
            # 左转/右转：目标高度保持当前的初始高度 (即保持平飞转弯)
            self.target_height = y_t

        # print(f">>> 机动: {self.maneuver_type}, 初始高度: {y_t:.1f}, 目标高度: {self.target_height:.1f}")

        # --- 3. 场景随机化 ---
        # y_t = np.random.uniform(5000, 10000)
        R_dist = np.random.uniform(9000, 11000)
        # 导弹初始方位
        theta1 = np.deg2rad(np.random.uniform(-180, 180))

        # 计算导弹位置
        x_m = R_dist * (-np.cos(theta1))
        z_m = R_dist * np.sin(theta1)
        y_m = y_t + np.random.uniform(-1000, 1000)

        # 飞机初始状态
        aircraft_pitch = np.deg2rad(np.random.uniform(-30, 30))  # 初始平飞或微俯仰
        aircraft_vel = np.random.uniform(300, 400)
        self.target_speed = aircraft_vel

        # 飞机初始航向：为了增加难度，可以设为迎头或随机
        aircraft_heading = np.deg2rad(np.random.uniform(-180, 180))

        initial_aircraft_state = np.array([
            0, y_t, 0,  # pos (m)
            aircraft_vel,  # Vel (m/s)
            aircraft_pitch,  # theta (rad)
            aircraft_heading,  # psi (rad)
            0,  # phi (rad)
            0.0  # p_real (rad/s)
        ])
        self.aircraft.reset(initial_state=initial_aircraft_state)

        # 计算导弹理想指向
        R_vec = self.aircraft.pos - np.array([x_m, y_m, z_m])
        R_mag = np.linalg.norm(R_vec)
        theta_L = np.arcsin(R_vec[1] / R_mag)
        phi_L = np.arctan2(R_vec[2], R_vec[0])

        # 添加导弹发射误差
        initial_heading_error_deg = 30.0  # 稍微减小误差，保证威胁
        error_rad = np.deg2rad(initial_heading_error_deg)
        delta_theta = np.random.uniform(-error_rad, error_rad)
        delta_phi = np.random.uniform(-error_rad, error_rad)

        missile_vel = np.random.uniform(800, 900)
        initial_missile_state = np.array([
            missile_vel,
            theta_L + delta_theta,
            phi_L + delta_phi,
            x_m, y_m, z_m
        ])
        self.missile = Missile(initial_missile_state)

        # --- 4. 历史记录初始化 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.prev_missile_state = self.missile.state_vector.copy()
        self.prev_R_rel = R_mag
        self.prev_theta_L, self.prev_phi_L = None, None
        self.aircraft_history = []
        self.missile_history = []
        self.time_history = []

        # --- 5. 发送 Tacview ---
        if self.tacview_enabled:
            self.tacview.tacview_final_frame_sent = False
            self.tacview.stream_frame(0.0, self.aircraft, self.missile, [])

        return self._get_observation(), {}

    def step(self, action: dict) -> tuple:
        """
        RL Step:
        1. 接收干扰弹投放指令 (discrete_actions)。
        2. PID 控制器根据当前态势自动计算并执行飞行机动 (continuous control)。
        3. 推进物理仿真。
        """
        # --- 1. 解析 RL 离散动作 ---
        discrete_cmds_indices = action["discrete_actions"]
        trigger_cmd_idx = discrete_cmds_indices[0]
        salvo_size_idx = discrete_cmds_indices[1]
        num_groups_idx = discrete_cmds_indices[2]
        inter_interval_idx = discrete_cmds_indices[3]

        # --- 2. 干扰弹投放逻辑 ---
        # 如果触发，则更新 FlareManager 的计划表
        release_flare_program = (trigger_cmd_idx == 1)
        if release_flare_program:
            # 假设 intra_interval 固定为 0.04s (与 PPO 定义一致)
            intra_interval = 0.04
            self._execute_flare_program(salvo_size_idx, intra_interval, num_groups_idx, inter_interval_idx)

        # --- 3. 准备物理循环 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.prev_missile_state = self.missile.state_vector.copy()
        self.prev_R_rel = np.linalg.norm(self.aircraft.pos - self.missile.pos)

        # 确定循环步长 (简单处理，统一用 dt_dec 拆分为 physical_dt)
        num_steps = int(round(self.dt_dec / self.physical_dt))

        for i in range(num_steps):
            # ================= PID 飞行控制接管核心 =================
            # 1. 获取 PID 控制器需要的 14维状态向量
            pid_input_obs = self._get_pid_input_vector()

            # 2. 计算 PID 输出
            # output: [aileron, elevator, rudder, throttle] (顺序根据 F16PIDController 决定)
            # 注意：flight_output 内部已包含姿态保持和机动逻辑
            pid_output = self.pid_controller.flight_output(pid_input_obs, dt=self.physical_dt)

            # 3. 映射到 Aircraft 类需要的控制顺序
            # F16PIDController 返回: [aileron(0), elevator(1), rudder(2), throttle(3)]
            # Aircraft.update 需求:  [throttle, elevator, aileron, rudder]
            aileron_cmd = pid_output[0]
            elevator_cmd = pid_output[1]
            rudder_cmd = pid_output[2]
            throttle_cmd = pid_output[3]

            aircraft_action = [throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd]
            # ======================================================

            # 执行物理更新
            self.run_one_step(self.physical_dt, aircraft_action)

            if self.done:
                break

        # --- 4. 计算奖励 ---
        reward = 0.0
        if self.done:
            self._calculate_final_miss_distance()
            if self.miss_distance > self.R_kill:
                self.success = True
            reward += self.reward_calculator.get_sparse_reward(self.miss_distance, self.R_kill)
        else:
            reward += self.reward_calculator.calculate_dense_reward(
                self.aircraft, self.missile, self.o_ir, self.N_infrared, action
            )

        # --- 5. 终止状态处理 ---
        terminated = self.done
        truncated = self.t_now >= self.t_end
        if truncated:
            terminated = True
            self.success = True

        observation = self._get_observation()
        info = {}
        if terminated or truncated:
            info["success"] = self.success

        return observation, reward, terminated, truncated, info

    def _get_pid_input_vector(self) -> np.ndarray:
        """
        构建 PID 观测向量。
        【纯机动版】移除对导弹位置的依赖，仅基于飞机自身状态执行纯粹的机动动作。
        """
        # --- 1. 获取基础状态 ---
        ac_state = self.aircraft.state_vector
        current_heading_rad = ac_state[5]  # 当前航向

        # (已移除) 不再计算导弹方位 azimuth_to_missile

        # --- 2. 根据机动类型计算 Target Heading ---

        # 初始化目标航向
        target_heading_rad = current_heading_rad

        if self.maneuver_type == 'turn_left':
            # === 左转逻辑 ===
            # 策略：每一帧都将目标设为当前航向的左侧 90度
            # 效果：PID 会检测到巨大的左侧航向误差，从而持续向左压最大坡度进行盘旋
            target_heading_rad = current_heading_rad - np.deg2rad(90)

        elif self.maneuver_type == 'turn_right':
            # === 右转逻辑 ===
            # 策略：每一帧都将目标设为当前航向的右侧 90度
            # 效果：持续向右压最大坡度盘旋
            target_heading_rad = current_heading_rad + np.deg2rad(90)

        elif self.maneuver_type == 'dive':
            # === 俯冲逻辑 ===
            # 航向：保持当前航向不变 (Target = Current)，即改平机翼，直线俯冲
            # 高度：在 reset 中已设为 1000m
            target_heading_rad = current_heading_rad

        elif self.maneuver_type == 'climb':
            # === 爬升逻辑 ===
            # 航向：保持当前航向不变，直线爬升
            # 高度：在 reset 中已设为 12000m
            target_heading_rad = current_heading_rad

        else:
            # 默认保持当前航向平飞
            target_heading_rad = current_heading_rad

        # 规范化目标航向到 [-pi, pi] 之间
        target_heading_rad = (target_heading_rad + np.pi) % (2 * np.pi) - np.pi

        # --- 3. 获取气动数据 (保持不变) ---
        alpha_rad = self.aircraft.fdm['aero/alpha-rad']
        beta_rad = self.aircraft.fdm['aero/beta-rad']
        q_rad_s = self.aircraft.fdm['velocities/q-rad_sec']
        r_rad_s = self.aircraft.fdm['velocities/r-rad_sec']

        v_n = self.aircraft.fdm['velocities/v-north-fps'] * 0.3048
        v_e = self.aircraft.fdm['velocities/v-east-fps'] * 0.3048
        v_d = self.aircraft.fdm['velocities/v-down-fps'] * 0.3048
        v_h = np.sqrt(v_n ** 2 + v_e ** 2)
        gamma_rad = np.arctan2(-v_d, v_h)
        course_rad = np.arctan2(v_e, v_n)

        # --- 4. 组装 OBS (保持不变) ---
        obs = np.zeros(14)
        obs[0] = self.target_height / 5000.0  # 目标高度(由reset设定)

        # 计算航向误差
        delta_heading = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(current_heading_rad))
        obs[1] = np.deg2rad(delta_heading)

        # obs[2] = self.target_speed / 340.0
        # ==================== [核心修改] ====================
        # 不要除以 340.0！传给 PID 真实值，让它对速度误差更敏感
        obs[2] = self.target_speed
        obs[3] = ac_state[4]  # Theta
        # obs[4] = ac_state[3] / 340.0  # Speed
        # 不要除以 340.0！传给 PID 真实值
        obs[4] = ac_state[3]
        # ====================================================
        obs[5] = ac_state[6]  # Phi
        obs[6] = alpha_rad
        obs[7] = beta_rad
        obs[8] = ac_state[7]  # p
        obs[9] = q_rad_s
        obs[10] = r_rad_s
        obs[11] = gamma_rad

        delta_course = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(course_rad))
        obs[12] = np.deg2rad(delta_course)

        obs[13] = ac_state[1] / 5000.0  # Current Height Norm

        return obs

    # def _get_pid_input_vector(self) -> np.ndarray:
    #     """
    #     构建 F16PIDController 所需的 14 维观测向量。
    #     根据 self.maneuver_type 决定飞行策略。
    #     """
    #     # --- 1. 获取基础状态 (不变) ---
    #     ac_state = self.aircraft.state_vector
    #     current_heading_rad = ac_state[5]  # 当前航向
    #
    #     # 计算相对导弹的方位 (用于基础的生存规避)
    #     rel_pos = self.missile.pos - self.aircraft.pos
    #     azimuth_to_missile = np.arctan2(rel_pos[2], rel_pos[0])
    #
    #     # --- [修改] 2. 根据机动类型计算 Target Heading 和 Target Height ---
    #
    #     # 初始化目标航向变量
    #     target_heading_rad = current_heading_rad
    #
    #     if self.maneuver_type == 'turn_left':
    #         # === 左转逻辑 ===
    #         # 策略：持续将目标航向设定为当前航向的左侧 60度
    #         # PID 控制器会为了追赶这个“永远在左边”的目标而持续压坡度左转
    #         target_heading_rad = current_heading_rad - np.deg2rad(60)
    #         # target_height 保持在 reset 中设定的值
    #
    #     elif self.maneuver_type == 'turn_right':
    #         # === 右转逻辑 ===
    #         # 策略：持续将目标航向设定为当前航向的右侧 60度
    #         target_heading_rad = current_heading_rad + np.deg2rad(60)
    #
    #     elif self.maneuver_type == 'dive':
    #         # === 俯冲逻辑 ===
    #         # 高度控制：target_height 已经在 reset 中设为 1000.0 (低空)
    #         # 航向控制：俯冲时保持切向 (Beaming) 以维持对导弹的几何优势
    #         # (也可以选择直线俯冲，但切向俯冲生存率更高)
    #         target_heading_rad = azimuth_to_missile + np.pi / 2
    #
    #     elif self.maneuver_type == 'climb':
    #         # === 爬升逻辑 ===
    #         # 高度控制：target_height 已经在 reset 中设为 12000.0 (高空)
    #         # 航向控制：同样保持切向
    #         target_heading_rad = azimuth_to_missile + np.pi / 2
    #
    #     else:
    #         # === 默认/保底逻辑 (切向机动) ===
    #         target_heading_rad = azimuth_to_missile + np.pi / 2
    #
    #     # 规范化目标航向到 [-pi, pi] 之间，防止角度跳变
    #     target_heading_rad = (target_heading_rad + np.pi) % (2 * np.pi) - np.pi
    #
    #     # --- 3. 获取气动数据 (不变) ---
    #     alpha_rad = self.aircraft.fdm['aero/alpha-rad']
    #     beta_rad = self.aircraft.fdm['aero/beta-rad']
    #     q_rad_s = self.aircraft.fdm['velocities/q-rad_sec']
    #     r_rad_s = self.aircraft.fdm['velocities/r-rad_sec']
    #
    #     v_n = self.aircraft.fdm['velocities/v-north-fps'] * 0.3048
    #     v_e = self.aircraft.fdm['velocities/v-east-fps'] * 0.3048
    #     v_d = self.aircraft.fdm['velocities/v-down-fps'] * 0.3048
    #     v_h = np.sqrt(v_n ** 2 + v_e ** 2)
    #     gamma_rad = np.arctan2(-v_d, v_h)
    #     course_rad = np.arctan2(v_e, v_n)
    #
    #     # --- 4. 组装 OBS (不变) ---
    #     obs = np.zeros(14)
    #     obs[0] = self.target_height / 5000.0  # 目标高度
    #
    #     delta_heading = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(current_heading_rad))
    #     obs[1] = np.deg2rad(delta_heading)  # 航向误差
    #
    #     obs[2] = self.target_speed / 340.0
    #     obs[3] = ac_state[4]  # Theta
    #     obs[4] = ac_state[3] / 340.0  # Speed
    #     obs[5] = ac_state[6]  # Phi
    #     obs[6] = alpha_rad
    #     obs[7] = beta_rad
    #     obs[8] = ac_state[7]  # p
    #     obs[9] = q_rad_s
    #     obs[10] = r_rad_s
    #     obs[11] = gamma_rad
    #
    #     delta_course = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(course_rad))
    #     obs[12] = np.deg2rad(delta_course)
    #
    #     obs[13] = ac_state[1] / 5000.0  # Current Height Norm
    #
    #     return obs

    # def _get_pid_input_vector(self) -> np.ndarray:
    #     """
    #     构建 F16PIDController 所需的 14 维观测向量。
    #     在此处定义“预设机动”逻辑（例如：切向规避/置尾机动）。
    #     """
    #     # --- 1. 定义机动策略：切向机动 (Beaming) ---
    #     # 目标：保持飞机航向与导弹方位成 90 度，最大化视线角速度
    #
    #     # 计算从飞机指向导弹的矢量 (NUE坐标系)
    #     # R_vm = Pos_missile - Pos_aircraft
    #     rel_pos = self.missile.pos - self.aircraft.pos
    #
    #     # 计算导弹相对于飞机的方位角 (Azimuth)
    #     # atan2(z, x) -> 0是北, pi/2是东
    #     azimuth_to_missile = np.arctan2(rel_pos[2], rel_pos[0])
    #
    #     # 目标航向 = 导弹方位 + 90度 (顺时针切向) 或 -90度
    #     # 这里为了简单，始终尝试右转切向。更高级的逻辑可以判断哪边转弯更快。
    #     target_heading_rad = azimuth_to_missile + np.pi / 2
    #
    #     # 规范化目标航向到 [-pi, pi]
    #     target_heading_rad = (target_heading_rad + np.pi) % (2 * np.pi) - np.pi
    #
    #     # --- 2. 获取当前状态数据 ---
    #     # 飞机状态: [x, y, z, Vt, theta, psi, phi, p]
    #     ac_state = self.aircraft.state_vector
    #
    #     # JSBSim 获取特定气动数据 (alpha, beta, rates)
    #     # 注意：需要确保 Aircraft 类暴露了 fdm 属性
    #     alpha_rad = self.aircraft.fdm['aero/alpha-rad']
    #     beta_rad = self.aircraft.fdm['aero/beta-rad']
    #     q_rad_s = self.aircraft.fdm['velocities/q-rad_sec']
    #     r_rad_s = self.aircraft.fdm['velocities/r-rad_sec']
    #
    #     # 速度分量计算 Gamma (爬升角) 和 Course Angle (航迹角)
    #     v_n = self.aircraft.fdm['velocities/v-north-fps'] * 0.3048
    #     v_e = self.aircraft.fdm['velocities/v-east-fps'] * 0.3048
    #     v_d = self.aircraft.fdm['velocities/v-down-fps'] * 0.3048
    #     v_h = np.sqrt(v_n ** 2 + v_e ** 2)  # 水平速度
    #
    #     gamma_rad = np.arctan2(-v_d, v_h)  # 爬升角 (正为爬升)
    #     course_rad = np.arctan2(v_e, v_n)  # 航迹角
    #
    #     # --- 3. 组装 PID 输入向量 (参考 F16PIDController 的 obs_jsbsim) ---
    #     obs = np.zeros(14)
    #
    #     # [0] Target Pitch (rad) or Target Height (Normalized)
    #     # F16PIDController 中根据输入大小判断是姿态控制还是高度控制
    #     # 这里我们使用高度控制模式
    #     obs[0] = self.target_height / 5000.0
    #
    #     # [1] Delta Heading (rad) = Target - Current
    #     delta_heading = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(ac_state[5]))
    #     obs[1] = np.deg2rad(delta_heading)
    #
    #     # [2] Target Speed (Normalized by 340 m/s)
    #     obs[2] = self.target_speed / 340.0
    #
    #     # [3] Current Pitch (rad)
    #     obs[3] = ac_state[4]
    #
    #     # [4] Current Speed (Normalized)
    #     obs[4] = ac_state[3] / 340.0
    #
    #     # [5] Current Roll (rad)
    #     obs[5] = ac_state[6]
    #
    #     # [6] Alpha (rad)
    #     obs[6] = alpha_rad
    #
    #     # [7] Beta (rad)
    #     obs[7] = beta_rad
    #
    #     # [8] p (rad/s)
    #     obs[8] = ac_state[7]
    #
    #     # [9] q (rad/s)
    #     obs[9] = q_rad_s
    #
    #     # [10] r (rad/s)
    #     obs[10] = r_rad_s
    #
    #     # [11] Gamma (rad)
    #     obs[11] = gamma_rad
    #
    #     # [12] Delta Course (rad)
    #     delta_course = sub_of_degree(np.rad2deg(target_heading_rad), np.rad2deg(course_rad))
    #     obs[12] = np.deg2rad(delta_course)
    #
    #     # [13] Current Height (Normalized by 5000m)
    #     obs[13] = ac_state[1] / 5000.0
    #
    #     return obs

    def run_one_step(self, dt, aircraft_action):
        """
        执行一步物理更新
        """
        # --- 1. 更新诱饵弹 ---
        self.flare_manager.update(self.t_now, dt, self.aircraft)

        # --- 2. 计算导引头目标 ---
        target_pos_equiv = self._calculate_equivalent_target()

        # --- 3. <<< 核心修改：调用导弹 OGL 更新 >>> ---
        # 获取飞机的物理状态
        aircraft_pos = self.aircraft.pos
        aircraft_vel = self.aircraft.get_velocity_vector()

        # 直接调用导弹的更新方法
        # 如果 target_pos_equiv 为 None (失锁)，导弹内部会自动处理惯性飞行
        self.missile.update_OGL(dt, aircraft_pos, aircraft_vel, target_pos_equiv)

        # 获取更新后的状态用于记录
        missile_state_next = self.missile.state_vector

        # # --- 3. 计算导弹动态 ---
        # if target_pos_equiv is not None:
        #     current_theta_L, current_phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
        #         target_pos_equiv, (self.prev_theta_L, self.prev_phi_L), dt)
        # else:
        #     current_theta_L = self.prev_theta_L
        #     current_phi_L = self.prev_phi_L
        #     theta_L_dot, phi_L_dot = 0.0, 0.0
        #
        # missile_state_next = self.missile._missile_dynamics(self.missile.state_vector, dt, theta_L_dot, phi_L_dot)

        # --- 4. 更新时间与历史 ---
        self.t_now += dt
        self.tacview_global_time += dt

        # 飞机更新 (JSBSim)
        self.aircraft.update(aircraft_action)

        self.aircraft_history.append(self.aircraft.state_vector.copy())
        self.missile_history.append(missile_state_next.copy())
        self.time_history.append(self.t_now)

        # # --- 5. 更新导弹状态 ---
        # self.missile.state = missile_state_next
        #
        # # 更新 LOS 历史
        # if target_pos_equiv is not None:
        #     self.prev_theta_L = current_theta_L
        #     self.prev_phi_L = current_phi_L

        # --- 6. 检查引信与终止条件 ---
        self._check_fuze_condition(dt)
        self._check_termination_conditions(dt)

        # --- 7. Tacview ---
        if self.tacview_enabled and not self.missile_exploded:
            self.tacview.stream_frame(self.tacview_global_time, self.aircraft, self.missile, self.flare_manager.flares)

    def _execute_flare_program(self, salvo_idx, intra_interval, num_groups_idx, inter_idx):
        """
        执行投放程序，逻辑保持不变
        """
        future_schedule = [t for t in self.flare_manager.schedule if t >= self.t_now]
        refunded_flares = len(future_schedule)
        current_available_flares = self.o_ir + refunded_flares

        ideal_program = {
            'salvo_size': DISCRETE_ACTION_MAP['salvo_size'][salvo_idx],
            'num_groups': DISCRETE_ACTION_MAP['num_groups'][num_groups_idx],
            'inter_interval': DISCRETE_ACTION_MAP['inter_interval'][inter_idx]
        }

        ideal_release_times = []
        if ideal_program['num_groups'] > 0 and ideal_program['salvo_size'] > 0:
            for group_idx in range(ideal_program['num_groups']):
                for salvo_idx in range(ideal_program['salvo_size']):
                    group_start_time = self.t_now + group_idx * ideal_program['inter_interval']
                    release_time = group_start_time + salvo_idx * intra_interval
                    ideal_release_times.append(release_time)

        num_to_release = min(len(ideal_release_times), current_available_flares)
        final_release_times = ideal_release_times[:num_to_release]

        past_schedule = [t for t in self.flare_manager.schedule if t < self.t_now]
        self.flare_manager.schedule = past_schedule + final_release_times
        self.flare_manager.schedule.sort()
        self.o_ir = current_available_flares - num_to_release

    # --- 下面的辅助方法保持原样，无需修改逻辑 ---

    def _calculate_los_rate(self, target_pos: np.ndarray, prev_los_angles: tuple, dt: float) -> tuple:
        prev_theta_L, prev_phi_L = prev_los_angles
        R_vec = target_pos - self.missile.pos
        R_mag = np.linalg.norm(R_vec)

        if R_mag < 1e-6:
            if prev_theta_L is not None:
                return prev_theta_L, prev_phi_L, 0.0, 0.0
            else:
                return 0.0, 0.0, 0.0, 0.0

        theta_L = np.arcsin(np.clip(R_vec[1] / R_mag, -1.0, 1.0))
        phi_L = np.arctan2(R_vec[2], R_vec[0])

        if prev_theta_L is None or prev_phi_L is None:
            theta_L_dot = 0.0
            phi_L_dot = 0.0
        else:
            theta_L_dot = (theta_L - prev_theta_L) / dt
            dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
            phi_L_dot = dphi / dt

        return theta_L, phi_L, theta_L_dot, phi_L_dot

    # def _check_fuze_condition(self, dt):
    #     if len(self.missile_history) < 2: return
    #     T2 = self.aircraft_history[-1][:3]
    #     T1 = self.aircraft_history[-2][:3]
    #     M2 = self.missile_history[-1][3:6]
    #     M1 = self.missile_history[-2][3:6]
    #     R_rel_now = np.linalg.norm(T2 - M2)
    #
    #     if R_rel_now < self.R_kill:
    #         self._trigger_explosion(R_rel_now, T2, M2)
    #         return
    #
    #     R_rel_prev = np.linalg.norm(T1 - M1)
    #     if min(R_rel_prev, R_rel_now) > 100: return
    #
    #     A = T1 - M1
    #     B = (T2 - T1) - (M2 - M1)
    #     A_dot_B = np.dot(A, B)
    #     B_dot_B = np.dot(B, B)
    #     tau_min = -A_dot_B / B_dot_B if B_dot_B > 1e-6 else 0
    #     tau_check = np.clip(tau_min, 0.0, 1.0)
    #
    #     min_dist = np.linalg.norm(A + tau_check * B)
    #     if min_dist < self.R_kill:
    #         # 计算精确爆炸位置
    #         ac_pos_exp = T1 + tau_check * (T2 - T1)
    #         ms_pos_exp = M1 + tau_check * (M2 - M1)
    #         # 修正 Tacview 时间
    #         t_exp_global = self.tacview_global_time - dt + tau_check * dt
    #
    #         self.done = True
    #         self.success = False
    #         self.miss_distance = min_dist
    #         self.missile_exploded = True
    #
    #         if self.tacview_enabled and not self.tacview.tacview_final_frame_sent:
    #             self.tacview.stream_explosion(t_exp_global, ac_pos_exp, ms_pos_exp)

    def _check_fuze_condition(self, dt):
        if len(self.missile_history) < 2: return
        T2 = self.aircraft_history[-1][:3]
        T1 = self.aircraft_history[-2][:3]
        M2 = self.missile_history[-1][3:6]
        M1 = self.missile_history[-2][3:6]
        R_rel_now = np.linalg.norm(T2 - M2)

        # 1. 当前时刻直接命中
        if R_rel_now < self.R_kill:
            self._trigger_explosion(R_rel_now, T2, M2)
            return

        # 2. 区间插值判定
        R_rel_prev = np.linalg.norm(T1 - M1)
        if min(R_rel_prev, R_rel_now) > 100: return

        A = T1 - M1
        B = (T2 - T1) - (M2 - M1)
        B_dot_B = np.dot(B, B)
        # 避免除零
        if B_dot_B < 1e-6:
            tau_min = 0.0
        else:
            A_dot_B = np.dot(A, B)
            tau_min = -A_dot_B / B_dot_B

        tau_check = np.clip(tau_min, 0.0, 1.0)

        min_dist = np.linalg.norm(A + tau_check * B)

        if min_dist < self.R_kill:
            # 计算精确爆炸位置
            ac_pos_exp = T1 + tau_check * (T2 - T1)
            ms_pos_exp = M1 + tau_check * (M2 - M1)
            # 修正 Tacview 时间
            t_exp_global = self.tacview_global_time - dt + tau_check * dt

            print(f">>> [步间引信] 判定命中！插值最小距离: {min_dist:.2f} m")  # <--- 新增这行

            self.done = True
            self.success = False
            self.miss_distance = min_dist
            self.missile_exploded = True

            if self.tacview_enabled and not self.tacview.tacview_final_frame_sent:
                self.tacview.stream_explosion(t_exp_global, ac_pos_exp, ms_pos_exp)
                self.tacview.tacview_final_frame_sent = True

    # def _trigger_explosion(self, dist, ac_pos, ms_pos):
    #     self.done = True
    #     self.success = False
    #     self.miss_distance = dist
    #     self.missile_exploded = True
    #     if self.tacview_enabled and not self.tacview.tacview_final_frame_sent:
    #         self.tacview.stream_explosion(self.tacview_global_time, ac_pos, ms_pos)

    def _trigger_explosion(self, dist, ac_pos, ms_pos):
        print(f">>> [引信触发] 判定命中！距离: {dist:.2f} m")  # <--- 新增这行
        self.done = True
        self.success = False
        self.miss_distance = dist
        self.missile_exploded = True
        if self.tacview_enabled and not self.tacview.tacview_final_frame_sent:
            self.tacview.stream_explosion(self.tacview_global_time, ac_pos, ms_pos)
            self.tacview.tacview_final_frame_sent = True  # 确保标记已发送

    def _get_observation(self) -> np.ndarray:
        # 1. 导弹观测
        R_vec = self.aircraft.pos - self.missile.pos
        R_rel = np.linalg.norm(R_vec)
        o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, self.missile.state_vector)
        Ry_rel = self.missile.pos[1] - self.aircraft.pos[1]
        o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))

        # <<< 模糊距离逻辑 >>>
        # 1. 生成随机模糊参数
        d_mohu = np.random.randint(500, 1000)
        d_min_offset = np.random.randint(100, 400)
        d_max_offset = d_mohu - d_min_offset

        # 2. 计算区间
        o_dis_min_val = max(0.0, R_rel - d_min_offset)
        o_dis_max_val = R_rel + d_max_offset

        # 3. 归一化 (参考最大距离 10000m)
        max_obs_dist = 10000.0
        o_dis_min_norm = 2 * (o_dis_min_val / max_obs_dist) - 1.0
        o_dis_max_norm = 2 * (o_dis_max_val / max_obs_dist) - 1.0

        missile_obs = [
            o_dis_min_norm,
            o_dis_max_norm,
            np.sin(o_beta_rad),
            np.cos(o_beta_rad),
            o_theta_L_rel_rad / (np.pi / 2)
        ]

        # # 归一化
        # o_dis_norm = 2 * (int(R_rel / 1000.0) / 10.0) - 1.0
        # missile_obs = [
        #     o_dis_norm,
        #     np.sin(o_beta_rad), np.cos(o_beta_rad),
        #     o_theta_L_rel_rad / (np.pi / 2)
        # ]

        # 2. 飞机观测
        aircraft_pitch_rad = self.aircraft.state_vector[4]
        aircraft_bank_rad = self.aircraft.state_vector[6]

        aircraft_obs = [
            2 * ((self.aircraft.velocity - 150) / 250) - 1,  # Vt
            2 * ((self.aircraft.pos[1] - 500) / 11500) - 1,  # Alt
            aircraft_pitch_rad / (np.pi / 2),  # Theta
            np.sin(aircraft_bank_rad), np.cos(aircraft_bank_rad),  # Phi
            2 * (self.o_ir / self.N_infrared) - 1.0,  # Flares
            # self.aircraft.roll_rate_rad_s / (4.0 * np.pi / 3.0)  # p
        ]

        return np.array(missile_obs + aircraft_obs, dtype=np.float32)

    def _compute_relative_beta2(self, x_target, x_missile):
        psi_t = x_target[5]
        cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)
        R_vec_beta = x_missile[3:6] - x_target[0:3]
        R_proj_world = np.array([R_vec_beta[0], R_vec_beta[2]])
        x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
        z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
        threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
        return threat_angle_rad + 2 * np.pi if threat_angle_rad < 0 else threat_angle_rad

    def _check_termination_conditions(self, dt):
        if self.done: return

        current_R_rel = np.linalg.norm(self.aircraft.pos - self.missile.pos)
        range_rate = (current_R_rel - self.prev_R_rel) / dt if self.prev_R_rel else 0.0

        # 检查锁定
        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            self.missile.state_vector, self.aircraft.state_vector,
            self.missile.get_velocity_vector(), self.aircraft.get_velocity_vector(), self.t_now
        )

        # 1. 坠毁判断
        if self.aircraft.pos[1] <= 500 or self.aircraft.pos[1] >= 12000 or self.aircraft.velocity <= 150:
            print(f">>> 飞机状态异常终止 (H={self.aircraft.pos[1]:.1f}, V={self.aircraft.velocity:.1f})")
            self.done, self.success = True, False
            return

        # 2. 逃逸判断
        # a) 物理逃逸 (导弹更慢且距离拉开)
        if self.missile.velocity < self.aircraft.velocity and range_rate > 5.0:
            self.escape_timer += dt
        else:
            self.escape_timer = 0.0

        # b) 信息逃逸 (丢失锁定且距离拉开)
        if (not lock_aircraft) and range_rate > 0:
            self.lost_and_separating_duration += dt
        else:
            self.lost_and_separating_duration = 0.0

        if self.escape_timer >= 2.0 or self.lost_and_separating_duration >= 2.0:
            self.done, self.success = True, True
            return

        # 3. 超时
        if self.t_now >= self.t_end:
            self.done, self.success = True, True
            return

    def _check_seeker_lock(self, x_missile, x_target, V_missile_vec, V_target_vec, t_now):
        R_vec = np.array(x_target[0:3]) - np.array(x_missile[3:6])
        R_mag = np.linalg.norm(R_vec)
        if R_mag > self.D_max: return False, False, True, True, True

        cos_Angle = np.dot(R_vec, V_missile_vec) / (R_mag * np.linalg.norm(V_missile_vec) + 1e-6)
        if cos_Angle < np.cos(self.Angle_IR_rad): return False, True, False, True, True

        delta_V = V_missile_vec - V_target_vec
        omega_R = np.linalg.norm(np.cross(R_vec, delta_V)) / (R_mag ** 2 + 1e-6)
        if omega_R > self.omega_max_rad_s: return False, True, True, False, True

        if t_now > self.T_max: return False, True, True, True, False

        return True, True, True, True, True

    def _infrared_intensity_model(self, beta_rad: float) -> float:
        beta_deg = np.array([0, 40, 90, 140, 180])
        intensity_vals = np.array([3800, 5000, 2500, 2000, 800])
        return max(np.interp(beta_rad, np.deg2rad(beta_deg), intensity_vals), 0.0)

    def _compute_relative_beta_for_ir(self, x_target, x_missile) -> float:
        R_vec = x_target[0:3] - x_missile[3:6]
        R_proj = np.array([R_vec[0], 0.0, R_vec[2]])
        if np.linalg.norm(R_proj) < 1e-6: return 0.0
        psi_t = x_target[5]
        V_body = np.array([np.cos(psi_t), 0.0, np.sin(psi_t)])
        cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
        return np.arccos(np.clip(cos_beta, -1.0, 1.0))

    # def _calculate_final_miss_distance(self):
    #     if self.miss_distance is not None: return
    #     if not self.success and (self.aircraft.pos[1] <= 500 or self.aircraft.velocity <= 150):
    #         self.miss_distance = 0.0
    #         return
    #
    #     if not self.aircraft_history:
    #         self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)
    #         return
    #
    #     Xt = np.array(self.aircraft_history)
    #     Yt = np.array(self.missile_history)
    #     delta_pos = Xt[:, :3] - Yt[:, 3:6]
    #     R_all = np.linalg.norm(delta_pos, axis=1)
    #
    #     if len(R_all) > 0:
    #         self.miss_distance = np.min(R_all)
    #         self.idx_min = np.argmin(R_all)
    #     else:
    #         self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)
    #
    #     is_hit = self.miss_distance <= self.R_kill
    #     print(f">>> {'命中' if is_hit else '未命中'}，脱靶量为：{self.miss_distance:.2f} m")

    def _calculate_final_miss_distance(self):
        # 1. 如果 miss_distance 还没计算（既没有触发引信，也没有在check_termination里赋值）
        #    则在这里进行最终的轨迹遍历计算
        if self.miss_distance is None:
            # 情况A: 操作失误导致坠毁/超速/失速
            if not self.success and (self.aircraft.pos[1] <= 500 or
                                     self.aircraft.pos[1] >= 15000 or
                                     self.aircraft.velocity <= 150):
                self.miss_distance = 0.0  # 视为被击中/失败

            # 情况B: 正常结束（超时或逃逸），遍历历史寻找最近点
            elif self.aircraft_history and self.missile_history:
                Xt = np.array(self.aircraft_history)
                Yt = np.array(self.missile_history)
                # 计算每一帧的距离
                delta_pos = Xt[:, :3] - Yt[:, 3:6]
                R_all = np.linalg.norm(delta_pos, axis=1)

                if len(R_all) > 0:
                    self.miss_distance = np.min(R_all)
                    self.idx_min = np.argmin(R_all)
                else:
                    self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)
            else:
                # 没有任何历史记录的极端情况
                self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)

        # 2. 统一打印结果 (无论上面是否执行，这里都会打印)
        is_hit = self.miss_distance <= self.R_kill
        print(f">>> [回合结束] {'命中 (HIT)' if is_hit else '未命中 (MISS)'}，最终脱靶量为：{self.miss_distance:.2f} m")

    def _calculate_equivalent_target(self):
        I_total = 0.0
        numerator = np.zeros(3)

        # 飞机
        lock_ac, _, _, _, _ = self._check_seeker_lock(
            self.missile.state_vector, self.aircraft.state_vector,
            self.missile.get_velocity_vector(), self.aircraft.get_velocity_vector(), self.t_now
        )
        if lock_ac:
            beta = self._compute_relative_beta_for_ir(self.aircraft.state_vector, self.missile.state_vector)
            I_p = self._infrared_intensity_model(beta)
            I_total += I_p
            numerator += I_p * self.aircraft.pos

        # 诱饵弹
        ms_vel = self.missile.get_velocity_vector()
        ms_state = self.missile.state_vector
        for flare in self.flare_manager.flares:
            I_k = flare.get_intensity(self.t_now)
            if I_k <= 1e-3: continue

            # 简单视场检查
            R_flare = flare.pos - self.missile.pos
            R_flare_mag = np.linalg.norm(R_flare)
            if R_flare_mag > self.D_max: continue

            cos_ang = np.dot(R_flare, ms_vel) / (R_flare_mag * np.linalg.norm(ms_vel) + 1e-6)
            if cos_ang >= np.cos(self.Angle_IR_rad):
                I_total += I_k
                numerator += I_k * flare.pos

        if I_total > 1e-6:
            return numerator / I_total
        return None

    def render(self):
        # 保持原有的 matplotlib 绘图逻辑
        if not self.aircraft_history: return
        ac_traj = np.array(self.aircraft_history)[:, :3]
        ms_traj = np.array(self.missile_history)[:, 3:6]

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ms_traj[:, 2], ms_traj[:, 0], ms_traj[:, 1], 'b-', label='Missile')
        ax.plot(ac_traj[:, 2], ac_traj[:, 0], ac_traj[:, 1], 'r--', label='Aircraft')

        if hasattr(self, 'idx_min') and self.idx_min < len(ms_traj):
            ax.scatter(ms_traj[self.idx_min, 2], ms_traj[self.idx_min, 0], ms_traj[self.idx_min, 1],
                       color='m', s=20, label=f'Min Dist: {self.miss_distance:.2f}m')

        for i, flare in enumerate(self.flare_manager.flares):
            if not flare.history: continue
            ft = np.array(flare.history)[:, :3]
            ax.plot(ft[:, 2], ft[:, 0], ft[:, 1], color='orange', linewidth=1, label='Flare' if i == 0 else "")

        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Up')
        ax.legend()
        ax.grid(True)
        plt.show()