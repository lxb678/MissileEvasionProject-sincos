# 文件: environment.py

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- (中文) 从我们创建的模块中导入所有需要的类 ---
from .AircraftJSBSim_DirectControl import Aircraft
from .missile2 import Missile
from .decoys import FlareManager
from .reward_system实体 import RewardCalculator
from .tacview_interface import TacviewInterface

# <<< 新增 >>> 从PPO代码文件导入动作空间定义，确保一致性
# 注意：这里的路径可能需要根据您的项目结构调整
from Interference_code.PPO_model.PPO_evasion_fuza.PPOMLP混合架构.Hybrid_PPO_混合架构 import CONTINUOUS_DIM, \
    DISCRETE_DIMS, \
    DISCRETE_ACTION_MAP

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv(gym.Env):
    """
    <<< 多导弹更改 >>>
    强化学习的多导弹空战环境主类。
    负责协调仿真流程、管理对象、计算奖励和提供RL接口(reset, step)。
    此版本处理两个可能非同步来袭的导弹。
    """

    def __init__(self, tacview_enabled=False, dt=0.02, num_missiles=2):
        super().__init__()

        # <<< 多导弹更改 >>> 场景中的导弹数量
        self.num_missiles = num_missiles

        # <<< 多导弹更改 >>> 定义动作空间 (Action Space) - 保持不变，智能体只有一个
        self.action_space = spaces.Dict({
            "continuous_actions": spaces.Box(
                low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(CONTINUOUS_DIM,)
            ),
            "discrete_actions": spaces.MultiDiscrete(
                np.array([
                    2,
                    DISCRETE_DIMS['salvo_size'],
                    DISCRETE_DIMS['num_groups'],
                    DISCRETE_DIMS['inter_interval'],
                ])
            )
        })

        # <<< 多导弹更改 >>> 定义观测空间 (Observation Space)
        # 每个导弹有3个相对观测量: [o_dis, o_beta, o_theta_L]
        # 飞机自身有6个观测量: [o_av, o_h, o_ae, o_am, o_ir, o_q]
        # 总维度 = 3 * num_missiles + 6
        # observation_dim = 3 * self.num_missiles + 6
        # <<< 多导弹更改 >>> 定义观测空间 (Observation Space) - 【核心修改】
        # 每个导弹: o_dis(1) + o_beta(sin,cos)(2) + o_theta_L(sin,cos)(2) = 5
        # 飞机自身: o_av(1) + o_h(1) + o_ae(sin,cos)(2) + o_am(sin,cos)(2) + o_ir(1) + o_q(1) = 8
        # 总维度 = 5 * num_missiles + 8
        # observation_dim = 5 * self.num_missiles + 8

        # # <<< 俯仰角修改 >>> 定义观测空间 (Observation Space) - 【核心修改】
        # # 每个导弹: o_dis(1) + o_beta(sin,cos)(2) + o_theta_L_norm(1) = 4
        # # 飞机自身: o_av(1) + o_h(1) + o_ae_norm(1) + o_am(sin,cos)(2) + o_ir(1) + o_q(1) = 7
        # # 总维度 = 4 * num_missiles + 7
        # observation_dim = 4 * self.num_missiles + 7
        #
        # self.observation_space = spaces.Box(
        #     low=-np.inf, # sin/cos 的值域是[-1, 1]，但为保持一致性，可以仍用-inf, inf
        #     high=np.inf,
        #     shape=(observation_dim,),
        #     dtype=np.float32  #ai推荐使用32
        # )

        # <<< 核心修改 >>> 定义观测空间 (Observation Space)
        # 每个导弹:
        #   1. o_dis_min_norm (模糊距离下界)
        #   2. o_dis_max_norm (模糊距离上界)
        #   3. o_beta_sin
        #   4. o_beta_cos
        #   5. o_theta_L_norm
        #   共 5 维
        # # 飞机自身: 7 维 (保持不变: av, h, ae, am_sin, am_cos, ir, q)
        # # 总维度 = 5 * num_missiles + 7
        # observation_dim = 5 * self.num_missiles + 7

        # 飞机自身: 6 维 (去掉 q: av, h, ae, am_sin, am_cos, ir)
        # 总维度 = 5 * num_missiles + 6
        observation_dim = 5 * self.num_missiles + 6  # <--- 修改这里

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )

        # --- 仿真参数 ---
        self.t_end = 60.0
        self.R_kill = 12.0
        self.dt_normal = dt
        self.dt_small = dt
        self.dt_flare = dt
        self.R_switch = 500
        self.dt_dec = 0.2
        self.D_max = 30000.0
        self.Angle_IR_rad = np.deg2rad(90)
        self.omega_max_rad_s = np.deg2rad(90.0)#12.0 #np.deg2rad(100.0) #12.0
        self.T_max = 60.0

        # --- 核心组件实例化 ---
        initial_state = np.array([0, 1000, 0, 200, 0.1, 0.0, 0.0, 0.0])
        self.aircraft = Aircraft(dt=self.dt_small, initial_state=initial_state)

        # <<< 多导弹更改 >>> 将单个导弹替换为列表
        self.missiles = []

        self.flare_manager = FlareManager(flare_per_group=1)
        self.reward_calculator = RewardCalculator()
        self.tacview_enabled = tacview_enabled
        self.tacview = TacviewInterface() if tacview_enabled else None

        # --- 状态变量 ---
        self.t_now = 0.0
        self.tacview_global_time = 0.0
        self.done = False
        self.success = False

        # <<< 多导弹更改 >>> 将单个脱靶量替换为列表
        self.miss_distances = []

        self.o_ir = 24 #30
        self.N_infrared = 24 #30
        self.prev_aircraft_state = None

        # <<< 多导弹更改 >>> 替换为列表
        self.prev_missile_states = []
        self.prev_R_rels = []
        self.prev_theta_Ls, self.prev_phi_Ls = [], []
        self.last_valid_theta_dots, self.last_valid_phi_dots = [], []

        # <<< 多导弹更改 >>> 历史轨迹也使用列表
        self.aircraft_history = []
        self.missile_histories = []
        self.time_history = []

        # <<< 多导弹更改 >>> 爆炸标志位现在需要跟踪任何一个导弹是否爆炸
        self.any_missile_exploded = False
        self.episode_count = 0
        self.physical_dt = dt

        # <<< 新增：在这里初始化标志位 >>>
        self._final_miss_dist_printed = False

    def reset(self, seed=None, options=None) -> tuple:
        # ---------------------------------------------------------
        # 必须要有这几行！否则 Run...py 里的 seed 传进来也控制不了 step 中的随机数
        # ---------------------------------------------------------
        # <<< 添加这两行 >>>
        if seed is not None:
            np.random.seed(seed)
            # 如果你也用了 random 库，最好也加一行
            import random
            random.seed(seed)
        super().reset(seed=seed)

        # <<< 确保这行代码在这里，用于重置标志位 >>>
        self._final_miss_dist_printed = False

        # --- 1. (Tacview) 清理上一回合的对象 ---
        if self.tacview_enabled and self.episode_count > 0:
            self.tacview.end_of_episode(
                t_now=self.tacview_global_time,
                flare_count=len(self.flare_manager.flares),
                missile_count=self.num_missiles,  # <<< 核心更改 >>> 传递导弹数量
                any_missile_exploded=self.any_missile_exploded  # 使用新的标志位
            )
            self.tacview.tacview_final_frame_sent = False

        # --- 2. 重置所有状态变量 ---
        self.t_now = 0.0
        self.done = False
        self.success = False
        self.miss_distances = []  # 清空列表
        self.any_missile_exploded = False
        self.episode_count += 1
        self.o_ir = self.N_infrared

        # --- 3. 重置组件 ---
        self.flare_manager.reset()
        self.reward_calculator.reset()

        # --- 4. <<< 多导弹更改 >>> 随机化初始场景并创建多个导弹 ---
        # (飞机初始状态)
        y_t = np.random.uniform(4000, 8000)
        aircraft_pitch = np.deg2rad(np.random.uniform(-30, 30))
        # aircraft_pitch = 0
        aircraft_vel = np.random.uniform(300, 400)
        initial_aircraft_state = np.array([0, y_t, 0, aircraft_vel, aircraft_pitch, 0, 0, 0.0])
        self.aircraft.reset(initial_state=initial_aircraft_state)

        # print("initial_aircraft_state", initial_aircraft_state)

        # <<< 新增：清理上一回合可能留下的属性 >>>
        for m in self.missiles:
            if hasattr(m, 'final_miss_distance'):
                del m.final_miss_distance
            # <<< 新增这行 >>>
            if hasattr(m, 'caused_hit'):
                del m.caused_hit

        # (导弹初始状态 - 循环创建)
        self.missiles.clear()
        self.prev_missile_states = []
        self.prev_R_rels = []
        self.prev_theta_Ls, self.prev_phi_Ls = [], []
        self.last_valid_theta_dots, self.last_valid_phi_dots = [], []
        self.time_step_counter = 0  # <<< 新增：全局时间步计数器

        # 为第一个导弹设置一个基准来袭方向
        base_theta1 = np.deg2rad(np.random.uniform(-180, 180))

        for i in range(self.num_missiles):
            # 随机化每个导弹的参数
            R_dist = np.random.uniform(9000, 11000)

            # 让第二个导弹从一个稍微不同的角度来袭
            # theta1_offset = np.deg2rad(np.random.uniform(-30, 30) * np.random.choice([-1, 1])) if i > 0 else 0
            theta1_offset = np.deg2rad(np.random.uniform(-180, 180) * np.random.choice([-1, 1])) if i > 0 else 0
            theta1 = base_theta1 + theta1_offset

            x_m = R_dist * (-np.cos(theta1))
            z_m = R_dist * np.sin(theta1)
            y_m = y_t + np.random.uniform(-1000, 1000)
            missile_vel = np.random.uniform(800, 900)

            # 计算初始视线角
            R_vec = self.aircraft.pos - np.array([x_m, y_m, z_m])
            R_mag = np.linalg.norm(R_vec)
            theta_L = np.arcsin(R_vec[1] / R_mag)
            phi_L = np.arctan2(R_vec[2], R_vec[0])

            # --- <<< 新增修改开始：添加初始指向误差 >>> ---
            # 设定最大发射误差角度 (例如 20度)
            # 这一步模拟了导弹发射时的不确定性，或者由于挂架/发射方式导致的初始离轴
            initial_heading_error_deg = 30.0 #60.0
            error_rad = np.deg2rad(initial_heading_error_deg)

            # 在俯仰 (theta) 和 偏航 (phi) 上分别增加随机扰动
            # 导弹初始速度方向 = 理想视线方向 + 随机误差
            delta_theta = np.random.uniform(-error_rad, error_rad)
            delta_phi = np.random.uniform(-error_rad, error_rad)

            missile_init_theta = theta_L + delta_theta
            missile_init_phi = phi_L + delta_phi
            # --- <<< 新增修改结束 >>> ---

            initial_missile_state = np.array([missile_vel, missile_init_theta, missile_init_phi, x_m, y_m, z_m])
            missile = Missile(initial_missile_state)

            # <<< 多导弹更改 >>> 为导弹附加额外状态
            missile.id = i
            # 第一个导弹立即发射，后续导弹延迟发射
            # missile.launch_time = 0.0 if i == 0 else np.random.uniform(2.0, 8.0)
            missile.launch_time = 0.0 if i == 0 else np.random.uniform(0.0, 10.0)
            missile.start_step_index = -1  # <<< 新增：记录开始的时间步
            missile.terminated = False  # 标志位，表示该导弹是否已命中或被规避

            # 为该导弹初始化独立的计时器
            missile.escape_timer = 0.0
            missile.lost_and_separating_duration = 0.0

            self.missiles.append(missile)

            # 初始化每个导弹的历史状态
            self.prev_missile_states.append(missile.state_vector.copy())
            self.prev_R_rels.append(R_mag)
            self.prev_theta_Ls.append(None)
            self.prev_phi_Ls.append(None)
            self.last_valid_theta_dots.append(0)
            self.last_valid_phi_dots.append(0)

        # --- 5. 初始化历史状态 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.aircraft_history = []
        self.missile_histories = [[] for _ in range(self.num_missiles)]
        self.time_history = []

        # 6. (Tacview) 发送初始帧
        if self.tacview_enabled:
            self.tacview.tacview_final_frame_sent = False
            # <<< 多导弹更改 >>> 传递导弹列表
            self.tacview.stream_frame(
                t_global=self.tacview_global_time,  # 全局时间
                t_episode=self.t_now,  # 回合内时间
                aircraft=self.aircraft,
                missiles=self.missiles,  # 传递完整的导弹列表
                flares=self.flare_manager.flares
            )

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: dict) -> tuple:
        # --- 1. 解析动作 (不变) ---
        continuous_cmds = action["continuous_actions"]
        discrete_cmds_indices = action["discrete_actions"]
        throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd = continuous_cmds
        trigger_cmd_idx, salvo_size_idx, num_groups_idx, inter_interval_idx = discrete_cmds_indices
        intra_interval = 0.04

        release_flare_program = (trigger_cmd_idx == 1)
        if release_flare_program:
            self._execute_flare_program(salvo_size_idx, intra_interval, num_groups_idx, inter_interval_idx)

        # --- 2. 保存决策步开始前的状态 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        for i in range(self.num_missiles):
            self.prev_missile_states[i] = self.missiles[i].state_vector.copy()
            self.prev_R_rels[i] = np.linalg.norm(self.aircraft.pos - self.missiles[i].pos)

        # --- 3. 物理仿真循环 (逻辑基本不变，但内部的run_one_step已修改) ---
        # 步长选择基于最近的*活动*导弹
        min_dist = float('inf')
        for i, missile in enumerate(self.missiles):
            if not missile.terminated and self.t_now >= missile.launch_time:
                min_dist = min(min_dist, self.prev_R_rels[i])

        if min_dist < self.R_switch:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_small)), self.dt_small
        elif self.flare_manager.schedule:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_flare)), self.dt_flare
        else:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_normal)), self.dt_normal

        aircraft_action = [throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd]
        for _ in range(num_steps):
            self.run_one_step(step_dt, aircraft_action)
            if self.done:
                break

        # --- 3. 判定回合最终状态 ---
        # 此时, self.done 和 self.success 已被核心逻辑函数正确设置
        terminated = self.done
        truncated = (self.t_now >= self.t_end) and not self.done

        if truncated:
            terminated = True
            # 如果是因为超时而截断, 说明没有发生任何失败事件, 故为成功
            self.success = True

        # --- 4. 根据最终状态计算奖励 ---
        reward = 0.0
        if terminated:  # 只要回合结束 (无论何种原因)
            # 为日志记录计算最终脱靶量  # (1) 必须先计算最终脱靶量，供奖励系统使用
            self._calculate_final_miss_distance()  # 这行甚至可以不调用了，如果只为了奖励的话

            # 根据已经判定好的 self.success 状态, 计算稀疏奖励
            # (2) <<< 核心修改 >>> 调用新的稀疏奖励接口
            # 将具体的数值计算逻辑移交给 reward_calculator
            # if self.success:
            #     reward += self.reward_calculator.W
            # else:
            #     reward += self.reward_calculator.U
            # <<< 核心修改：直接传入导弹列表 >>>
            reward += self.reward_calculator.get_sparse_reward(
                success=self.success,
                missiles=self.missiles  # 传入对象列表
            )

            # # (可选) 打印调试信息，确认逻辑生效
            # if not self.success:
            #     print(f"DEBUG: 任务失败。 -> 稀疏奖励: {reward}")
        else:  # 如果回合仍在继续
            # 计算密集奖励
            # <<< 关键调用修改 >>>
            # 现在，我们将完整的 `action` 字典（可能包含 "attention_weights"）
            # 直接传递给奖励计算器。
            reward += self.reward_calculator.calculate_dense_reward(
                self.aircraft, self.missiles, self.o_ir, self.N_infrared, action, self.t_now
            )

        # --- 5. 准备并返回结果 ---
        observation = self._get_observation()
        info = {}
        if terminated:
            # 报告由核心逻辑函数最终确定的 self.success
            info["success"] = self.success

        return observation, reward, terminated, truncated, info

    def run_one_step(self, dt, aircraft_action):
        """ <<< 多导弹更改 (V2 - 独立导引头版) >>> """
        # --- 1. 更新诱饵弹和飞机 (只需一次) ---
        self.flare_manager.update(self.t_now, dt, self.aircraft)
        self.aircraft.update(aircraft_action)

        # --- 2. <<< 核心修改：移除全局的目标计算 >>> ---
        # target_pos_equiv = self._calculate_equivalent_target() # <--- 删除这一行

        # 获取飞机状态用于传递给导弹
        aircraft_pos = self.aircraft.pos
        aircraft_vel = self.aircraft.get_velocity_vector()

        # --- 3. <<< 多导弹更改 >>> 遍历并更新每个导弹 ---
        for i, missile in enumerate(self.missiles):
            if self.t_now < missile.launch_time or missile.terminated:
                continue

            # 记录导弹首次活动的全局步索引（使用 run_one_step 的计数）
            if missile.start_step_index == -1:
                missile.start_step_index = self.time_step_counter

            # <<< 核心修改：为当前导弹独立计算其制导目标 >>># A. 为当前导弹独立计算其制导目标 (红外质心)
            target_pos_equiv = self._calculate_equivalent_target_for_missile(missile)

            # B. <<< 核心修改 >>> 调用导弹的 OGL 更新
            # 不再在环境里手动计算 theta_dot/phi_dot，全部交给 missile.update_OGL 处理
            # 如果 target_pos_equiv 为 None，导弹内部应处理失锁/惯性飞行逻辑
            missile.update_OGL(dt, aircraft_pos, aircraft_vel, target_pos_equiv)

            # C. 追加导弹历史（每个导弹每个全局步一条）
            self.missile_histories[i].append(missile.state_vector.copy())

            # # 计算该导弹的视线角速率
            # if target_pos_equiv is not None:
            #     # 导引头锁定目标，正常计算LOS rate
            #     current_theta_L, current_phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
            #         target_pos_equiv, (self.prev_theta_Ls[i], self.prev_phi_Ls[i]), dt, missile)
            #
            #     # 更新历史状态，以备下一帧使用
            #     self.prev_theta_Ls[i] = current_theta_L
            #     self.prev_phi_Ls[i] = current_phi_L
            #     # 存储有效的角速率，但现在我们不再在失锁时使用它
            #     # 它的作用主要是为了在下一帧重新锁定时能平滑计算
            #     self.last_valid_theta_dots[i] = theta_L_dot
            #     self.last_valid_phi_dots[i] = phi_L_dot
            # else:
            #     # 导引头失锁，模拟“过载指令归零”
            #     # 将视线角速率强制设为0，这将导致法向过载为0
            #     theta_L_dot, phi_L_dot = 0.0, 0.0
            #
            #     # 此时 current_theta_L 和 current_phi_L 未定义，
            #     # 但由于我们不再更新 prev_theta_Ls, prev_phi_Ls，所以没关系。
            #     # 导弹将沿直线飞行，直到重新锁定。
            #
            #     # # 导引头失锁，导弹将继续沿惯性飞行
            #     # # 使用上一步的角速率（通常为0或很小），模拟无制导指令
            #     # current_theta_L = self.prev_theta_Ls[i]
            #     # current_phi_L = self.prev_phi_Ls[i]
            #     # theta_L_dot, phi_L_dot = self.last_valid_theta_dots[i], self.last_valid_phi_dots[i]
            #
            # missile_state_next = missile._missile_dynamics(missile.state_vector, dt, theta_L_dot, phi_L_dot)
            # missile.state = missile_state_next
            #
            # # # 更新该导弹的历史状态
            # # if target_pos_equiv is not None:
            # #     self.prev_theta_Ls[i] = current_theta_L
            # #     self.prev_phi_Ls[i] = current_phi_L
            # #     # 存储有效的角速率，以备失锁时使用
            # #     self.last_valid_theta_dots[i] = theta_L_dot
            # #     self.last_valid_phi_dots[i] = phi_L_dot
            # # 追加导弹历史（每个导弹每个全局步一条）
            # self.missile_histories[i].append(missile.state_vector.copy())

        # ---- 把计数器的自增移到这里（每调用一次 run_one_step 增 1） ----
        self.time_step_counter += 1

        # --- 3. 更新时间和通用历史记录 ---
        self.t_now += dt
        self.tacview_global_time += dt
        self.aircraft_history.append(self.aircraft.state_vector.copy())
        self.time_history.append(self.t_now)

        # --- 4. <<< 多导弹更改 >>> 检查所有导弹的引信和终止条件 ---
        for i in range(self.num_missiles):
            if not self.missiles[i].terminated and self.t_now >= self.missiles[i].launch_time:
                # 检查引信，如果命中，会直接设置 self.done=True
                self._check_fuze_for_missile(dt, i)
                if self.done: break  # 任何一个命中则立即停止检查

        if not self.done:
            self._update_episode_status(dt)  # 检查其他终止条件（如逃逸）

        # --- 5. Tacview 更新 ---
        if self.tacview_enabled and not self.any_missile_exploded:
            # 传递导弹列表
            self.tacview.stream_frame(
                t_global=self.tacview_global_time,  # 全局时间
                t_episode=self.t_now,  # 回合内时间
                aircraft=self.aircraft,
                missiles=self.missiles,  # 传递完整的导弹列表
                flares=self.flare_manager.flares
            )

    def _get_observation(self) -> np.ndarray:
        """ <<< 核心修改: 统一归一化至[-1, 1]，并使用与单导弹版相同的模糊距离和物理边界 >>>
        组装观测向量。为每个导弹（无论是否激活）生成观测值。
        - 激活导弹: 使用精确的相对几何关系并进行归一化。
        - 非激活导弹: 提供一个标准的“无威胁”观测值，其归一化值也落在[-1, 1]区间。
        """
        """ 
                <<< 核心修改: 统一归一化至[-1, 1]，增加距离模糊化 >>>
                组装观测向量。为每个导弹生成 5 个观测值 + 飞机 7 个观测值。
                """
        obs_parts = []

        # --- 1. 遍历每个导弹，计算其相对观测值 ---
        for missile in self.missiles:
            if missile.terminated or self.t_now < missile.launch_time:
                # # --- 非激活导弹的观测值 (统一到 [-1, 1] 区间) ---
                # o_dis_norm = 1.0  # 距离最远 -> 1.0
                # o_beta_sin, o_beta_cos = 0.0, 1.0  # beta=0度 (正前方)
                # o_theta_L_rel_norm = 0.0  # 相对俯仰角为0 (水平) -> 0.0
                #
                # obs_parts.extend([o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_rel_norm])
                # continue

                # --- 非激活导弹的观测值 (统一到 [-1, 1] 区间) ---
                # 距离最远 -> min=1.0, max=1.0
                # beta=0度 -> sin=0.0, cos=1.0
                # theta_L=0 -> 0.0
                obs_parts.extend([1.0, 1.0, 0.0, 1.0, 0.0])
                continue

            # --- 以下代码只对激活导弹执行 ---
            R_vec = self.aircraft.pos - missile.pos
            R_rel = np.linalg.norm(R_vec)

            o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, missile.state_vector)
            Ry_rel = missile.pos[1] - self.aircraft.pos[1]
            o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))

            # --- 归一化 (与单导弹版逻辑完全一致) ---
            # # 1. 距离模糊化与归一化
            # quantized_distance_km = int(R_rel / 1000.0)
            # max_quantized_dist = 10.0
            # o_dis_norm = 2 * (quantized_distance_km / max_quantized_dist) - 1.0

            # --- <<< 核心修改：模糊距离逻辑 >>> ---
            # 1. 生成随机模糊参数 (单位: 米)
            d_mohu = np.random.randint(500, 1000)  # 总的不确定范围
            d_min_offset = np.random.randint(100, 400)  # 向下偏差量
            d_max_offset = d_mohu - d_min_offset  # 向上偏差量

            # 2. 计算模糊后的物理距离区间
            o_dis_min_val = max(0.0, R_rel - d_min_offset)
            o_dis_max_val = R_rel + d_max_offset

            # 3. 归一化到 [-1, 1] (参考最大距离 10000 米)
            max_obs_dist = 10000.0
            o_dis_min_norm = 2 * (o_dis_min_val / max_obs_dist) - 1.0
            o_dis_max_norm = 2 * (o_dis_max_val / max_obs_dist) - 1.0

            # 2. 方位角 beta 使用 sin/cos
            o_beta_sin = np.sin(o_beta_rad)
            o_beta_cos = np.cos(o_beta_rad)

            # 3. 相对俯仰角 theta_L 使用线性归一化 [-1, 1]
            o_theta_L_rel_norm = o_theta_L_rel_rad / (np.pi / 2)
            # 添加 5 个特征
            obs_parts.extend([o_dis_min_norm, o_dis_max_norm, o_beta_sin, o_beta_cos, o_theta_L_rel_norm])
            # obs_parts.extend([o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_rel_norm])

        # --- 2. 获取飞机自身状态 (与单导弹版逻辑完全一致) ---
        aircraft_pitch_rad = self.aircraft.state_vector[4]  # 俯仰角 theta
        aircraft_bank_rad = self.aircraft.state_vector[6]  # 滚转角 phi

        # 1. 速度归一化 (使用物理边界)
        o_av_norm = 2 * ((self.aircraft.velocity - 150) / 250) - 1

        # 2. 高度归一化 (使用物理边界)
        o_h_norm = 2 * ((self.aircraft.pos[1] - 500) / 11500) - 1

        # 3. 飞机俯仰角归一化
        o_ae_norm = aircraft_pitch_rad / (np.pi / 2)

        # 4. 滚转角使用 sin/cos 表示
        o_am_sin = np.sin(aircraft_bank_rad)
        o_am_cos = np.cos(aircraft_bank_rad)

        # 5. 剩余诱饵弹数量归一化
        o_ir_norm = 2 * (self.o_ir / self.N_infrared) - 1.0

        # # 6. 滚转速率归一化
        # o_q_norm = self.aircraft.roll_rate_rad_s / (4.0 * np.pi / 3.0)

        # aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_sin, o_am_cos, o_ir_norm, o_q_norm]

        # 列表中去掉了 o_q_norm
        aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_sin, o_am_cos, o_ir_norm]  # <--- 修改这里

        # --- 3. 拼接成最终的观测向量 ---
        observation = np.array(obs_parts + aircraft_obs)
        return observation.astype(np.float32)

    # def _get_observation(self) -> np.ndarray:
    #     """ <<< 多导弹更改 (V3 - sin/cos 角度表示法) >>> """
    #     obs_parts = []
    #
    #     # --- 1. 遍历每个导弹，计算其相对观测值 ---
    #     for missile in self.missiles:
    #         if missile.terminated or self.t_now < missile.launch_time:
    #             # 无效导弹提供标准“无威胁”观测值
    #             o_dis_norm = 1.0
    #             o_beta_sin, o_beta_cos = 0.0, 1.0  # beta=0度 (正前方) -> sin(0)=0, cos(0)=1
    #             o_theta_L_sin, o_theta_L_cos = 0.0, 1.0  # theta_L=0度 (水平) -> sin(0)=0, cos(0)=1
    #             obs_parts.extend([o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_sin, o_theta_L_cos])
    #             continue
    #
    #         # --- 以下代码只对有效导弹执行 ---
    #         R_vec = self.aircraft.pos - missile.pos
    #         R_rel = np.linalg.norm(R_vec)
    #
    #         # 计算原始弧度角
    #         o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, missile.state_vector)
    #         Ry_rel = missile.pos[1] - self.aircraft.pos[1]
    #         o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))
    #
    #         # --- 更改前 (Before) ---
    #         o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10
    #         # o_beta_norm = o_beta_rad / (2 * np.pi)
    #         # o_theta_L_rel_norm = (o_theta_L_rel_rad + np.pi / 2) / np.pi
    #         # obs_parts.extend([o_dis_norm, o_beta_norm, o_theta_L_rel_norm])
    #
    #         # --- 更改后 (After) ---
    #         # o_dis_norm = np.clip(R_rel / 30000.0, 0, 1)  # 距离用更平滑的归一化
    #         o_beta_sin = np.sin(o_beta_rad)
    #         o_beta_cos = np.cos(o_beta_rad)
    #         # o_theta_L_rel_rad 的范围是 [-pi/2, pi/2]，sin/cos 转换依然有效且有益
    #         o_theta_L_sin = np.sin(o_theta_L_rel_rad)
    #         o_theta_L_cos = np.cos(o_theta_L_rel_rad)
    #         obs_parts.extend([o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_sin, o_theta_L_cos])
    #
    #     # --- 2. 获取飞机自身状态 ---
    #     aircraft_pitch_rad = self.aircraft.state_vector[4]  # 俯仰角 theta
    #     aircraft_bank_rad = self.aircraft.state_vector[6]  # 滚转角 phi
    #
    #     # --- 更改后 (After) ---
    #     o_av_norm = (self.aircraft.velocity - 100) / 300
    #     o_h_norm = (self.aircraft.pos[1] - 1000) / 14000
    #     o_ae_sin = np.sin(aircraft_pitch_rad)  # 俯仰角
    #     o_ae_cos = np.cos(aircraft_pitch_rad)
    #     o_am_sin = np.sin(aircraft_bank_rad)  # 滚转角
    #     o_am_cos = np.cos(aircraft_bank_rad)
    #     o_ir_norm = self.o_ir / self.N_infrared
    #     # 滚转速率 o_q 不是角度，是角速度，不需要也不能用sin/cos，保持线性归一化
    #     o_q_norm = (self.aircraft.roll_rate_rad_s + 4.0 * np.pi / 3.0) / (8.0 * np.pi / 3.0)
    #
    #     aircraft_obs = [o_av_norm, o_h_norm, o_ae_sin, o_ae_cos, o_am_sin, o_am_cos, o_ir_norm, o_q_norm]
    #
    #     # --- 3. 拼接成最终的观测向量 ---
    #     observation = np.array(obs_parts + aircraft_obs)
    #     return observation.astype(np.float32)

    # def _get_observation(self) -> np.ndarray:
    #     """ <<< 多导弹更改 (V2 - 修正版) >>> 组装观测向量，为无效导弹设置无意义值。"""
    #     obs_parts = []
    #
    #     # --- 1. 遍历每个导弹，计算其相对观测值 ---
    #     for missile in self.missiles:
    #         # <<< 核心修改 >>> 检查导弹是否有效（已发射且未终结）
    #         if missile.terminated or self.t_now < missile.launch_time:
    #             # 如果导弹无效，提供一个标准的“无威胁”观测值
    #             o_dis_norm = 1.0  # 归一化距离为1 (最远)
    #             o_beta_norm = 0.0  # 归一化方位角为0 (正前方)
    #             o_theta_L_rel_norm = 0.5  # 归一化俯仰角为0.5 (水平)
    #             obs_parts.extend([o_dis_norm, o_beta_norm, o_theta_L_rel_norm])
    #             continue  # 处理下一个导弹
    #
    #         # --- 以下代码只对有效导弹执行 ---
    #         R_vec = self.aircraft.pos - missile.pos
    #         R_rel = np.linalg.norm(R_vec)
    #
    #         # 使用内部函数计算 beta 和 theta_L
    #         o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, missile.state_vector)
    #         Ry_rel = missile.pos[1] - self.aircraft.pos[1]
    #         o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))
    #
    #         # 归一化
    #         # o_dis_norm = np.clip(R_rel / 30000.0, 0, 1) # 修正：使用更平滑的归一化
    #         o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10  # 保持原来的离散化归一化
    #         o_beta_norm = o_beta_rad / (2 * np.pi)
    #         o_theta_L_rel_norm = (o_theta_L_rel_rad + np.pi / 2) / np.pi  # 修正：确保范围是 [0, 1]
    #
    #         obs_parts.extend([o_dis_norm, o_beta_norm, o_theta_L_rel_norm])
    #
    #     # --- 2. 获取飞机自身状态 (逻辑不变) ---
    #     o_av_norm = (self.aircraft.velocity - 100) / 300
    #     o_h_norm = (self.aircraft.pos[1] - 1000) / 14000
    #     o_ae_norm = (self.aircraft.state_vector[4] + np.pi / 2) / np.pi  # 修正：确保范围是 [0, 1]
    #     o_am_norm = (self.aircraft.state_vector[6] + np.pi) / (2 * np.pi)  # 修正：确保范围是 [0, 1]
    #     o_q_norm = (self.aircraft.roll_rate_rad_s + 4.0 * np.pi / 3.0) / (8.0 * np.pi / 3.0)  # 修正：确保范围是 [0, 1]
    #     o_ir_norm = self.o_ir / self.N_infrared
    #
    #     aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_norm, o_ir_norm, o_q_norm]
    #
    #     # --- 3. 拼接成最终的观测向量 ---
    #     observation = np.array(obs_parts + aircraft_obs)
    #     return observation.astype(np.float32)

    def _execute_flare_program(self, salvo_idx, intra_idx, num_groups_idx, inter_idx):
        """
        (V6 - 终极修正版：彻底清空，精确记账)
        """
        # --- 步骤 1: 计算当前总可用弹药 ---
        future_schedule = [t for t in self.flare_manager.schedule if t >= self.t_now]
        refunded_flares = len(future_schedule)
        current_available_flares = self.o_ir + refunded_flares

        # --- 步骤 2: 将智能体请求转换为理想的投放时间戳列表 ---
        ideal_program = {
            'salvo_size': DISCRETE_ACTION_MAP['salvo_size'][salvo_idx],
            # 'intra_interval': intra_idx, #DISCRETE_ACTION_MAP['intra_interval'][intra_idx],
            'num_groups': DISCRETE_ACTION_MAP['num_groups'][num_groups_idx],
            'inter_interval': DISCRETE_ACTION_MAP['inter_interval'][inter_idx]
        }
        # print(intra_idx)

        ideal_release_times = []
        if ideal_program['num_groups'] > 0 and ideal_program['salvo_size'] > 0:
            for group_idx in range(ideal_program['num_groups']):
                for salvo_idx in range(ideal_program['salvo_size']):
                    group_start_time = self.t_now + group_idx * ideal_program['inter_interval']
                    release_time = group_start_time + salvo_idx * intra_idx #ideal_program['intra_interval']
                    ideal_release_times.append(release_time)

        # --- 步骤 3: 根据可用弹药，截取实际要执行的计划 ---
        num_to_release = min(len(ideal_release_times), current_available_flares)
        final_release_times = ideal_release_times[:num_to_release]

        # --- 步骤 4: 更新最终的投放计划和库存 (核心修正) ---

        # a. 彻底清空未来的旧计划，只保留已经过去的
        past_schedule = [t for t in self.flare_manager.schedule if t < self.t_now]

        # b. 将新计算出的最终计划与过去的历史合并
        self.flare_manager.schedule = past_schedule + final_release_times

        # c. 排序以保证时间顺序
        self.flare_manager.schedule.sort()

        # d. 用最直接的方式更新最终库存
        self.o_ir = current_available_flares - num_to_release

        # # --- 调试日志 ---
        # if num_to_release > 0:
        #     if num_to_release < len(ideal_release_times):
        #         print(
        #             f"[{self.t_now:.2f}s] 弹药不足！请求: {len(ideal_release_times)}, 可用: {current_available_flares}. "
        #             f"实际安排: {num_to_release} 发。")
        #     else:
        #         print(f"[{self.t_now:.2f}s] 已安排投放 {num_to_release} 发诱饵弹。")
        #
        # print(f"[{self.t_now:.2f}s] 更新后剩余诱饵弹 (o_ir): {self.o_ir}")


    def _check_fuze_for_missile(self, dt, missile_index):
        """ <<< 多导弹更改 (V2 - 修正版) >>> 检查指定索引的导弹的引信条件，并正确处理 Tacview 调用。"""
        if len(self.missile_histories[missile_index]) < 2:
            return

        missile_history_single = self.missile_histories[missile_index]
        T2 = self.aircraft_history[-1][:3]
        T1 = self.aircraft_history[-2][:3]
        M2 = missile_history_single[-1][3:6]
        M1 = missile_history_single[-2][3:6]

        R_rel_now = np.linalg.norm(T2 - M2)

        # --- 变量初始化 ---
        is_hit = False
        min_dist_final = float('inf')
        tau_check = 1.0  # 默认值，对应当前时刻

        # --- 判断路径 1: 当前时刻直接命中 ---
        if R_rel_now < self.R_kill:
            is_hit = True
            min_dist_final = R_rel_now
            tau_check = 1.0  # 命中发生在区间的终点 (当前时刻)
            # print(f">>> 导弹 {missile_index + 1} 引信引爆！当前时刻直接命中，距离 = {min_dist_final:.2f} m")

            # <<< 确保这行代码在这里，用于重置标志位 >>>
            self._final_miss_dist_printed = True

            # --- 判断路径 2: 区间内插值命中 (仅在未直接命中时检查) ---
        else:
            A = T1 - M1
            B = (T2 - T1) - (M2 - M1)
            B_dot_B = np.dot(B, B)

            if B_dot_B < 1e-6:
                tau_min = 0.0
            else:
                tau_min = -np.dot(A, B) / B_dot_B

            # 使用临时变量存储插值的 tau
            tau_interpolated = np.clip(tau_min, 0.0, 1.0)
            min_dist_in_interval = np.linalg.norm(A + tau_interpolated * B)

            if min_dist_in_interval < self.R_kill:
                is_hit = True
                min_dist_final = min_dist_in_interval
                tau_check = tau_interpolated  # 使用计算出的插值系数
                # print(f">>> 导弹 {missile_index + 1} 引信引爆！区间内最小脱靶量 = {min_dist_final:.2f} m")

        # --- 统一处理命中事件 ---
        if is_hit:
            self.done = True
            self.success = False
            # <<< 核心修改：不再修改 self.miss_distances >>>
            # # 确保只添加一次脱靶量 (删除这两行)
            # if not self.missiles[missile_index].terminated:
            #     self.miss_distances.append(min_dist_final)

            # 存储临时的、最精确的命中距离，以备最终计算时使用
            # 我们可以将它直接附加到导弹对象上
            self.missiles[missile_index].final_miss_distance = min_dist_final

            self.any_missile_exploded = True
            self.missiles[missile_index].terminated = True

            # <<< 核心修改：标记这枚导弹是导致坠机的“凶手” >>>
            self.missiles[missile_index].caused_hit = True

            # --- Tacview 爆炸事件处理 (现在所有变量都已定义) ---
            if self.tacview_enabled and self.tacview.is_connected and not self.tacview.tacview_final_frame_sent:
                t1_global = self.tacview_global_time - dt
                explosion_time_global = t1_global + tau_check * dt

                # 使用 tau_check 进行线性插值得到精确位置
                aircraft_pos_at_explosion = T1 + tau_check * (T2 - T1)
                missile_pos_at_explosion = M1 + tau_check * (M2 - M1)

                exploding_missile_id_str = f"{self.tacview.missile_base_id + missile_index + 1}"

                self.tacview.stream_explosion(
                    t_explosion=explosion_time_global,
                    aircraft_pos=aircraft_pos_at_explosion,
                    missile_pos=missile_pos_at_explosion,
                    missile_id_str=exploding_missile_id_str
                )

    def _update_episode_status(self, dt):
        """ <<< 多导弹更改 (V2 - 恢复计时器逻辑) >>>
        检查所有终止条件，为每个导弹独立维护逃逸计时器。
        """
        if self.done: return

        # 1. [全局失败条件] 检查飞机坠毁等立即失败条件
        if self.aircraft.pos[1] <= 500 or self.aircraft.pos[1] >= 12000 or self.aircraft.velocity <= 150:
            print(f">>> 飞机状态异常！仿真终止。")
            self.done, self.success = True, False
            # 在这里设置打印标志位可能不是最佳位置，但为了保持逻辑一致性先加上
            if not self._final_miss_dist_printed:
                self._final_miss_dist_printed = True
            return

        # 2. [独立规避/追踪] 遍历每个未终结的导弹，更新其追踪状态和计时器
        active_missiles_count = 0
        for i, missile in enumerate(self.missiles):
            # 跳过已终结或未发射的导弹
            if missile.terminated or self.t_now < missile.launch_time:
                continue

            # 如果执行到这里，说明至少还有一个导弹是活动的
            active_missiles_count += 1

            # --- a) 获取判断所需的状态 ---
            aircraft_velocity = self.aircraft.velocity
            missile_velocity = missile.velocity
            current_R_rel = np.linalg.norm(self.aircraft.pos - missile.pos)
            range_rate = (current_R_rel - self.prev_R_rels[i]) / dt if dt > 1e-6 else 0.0

            # --- b) 更新“物理逃逸”计时器 (missile.escape_timer) ---
            cond_missile_slower = missile_velocity < aircraft_velocity
            cond_separating = range_rate > 5.0  # 使用一个小的正阈值
            if cond_missile_slower and cond_separating:
                missile.escape_timer += dt
            else:
                missile.escape_timer = 0.0  # 只要条件不满足，计时器就清零

            # --- c) 更新“信息逃逸”计时器 (missile.lost_and_separating_duration) ---
            lock, _, _, _, _ = self._check_seeker_lock(
                missile.state_vector, self.aircraft.state_vector,
                missile.get_velocity_vector(), self.aircraft.get_velocity_vector(), self.t_now
            )
            cond_lost_lock = not lock
            cond_separating_simple = range_rate > 0
            if cond_lost_lock and cond_separating_simple:
                missile.lost_and_separating_duration += dt
            else:
                missile.lost_and_separating_duration = 0.0  # 条件不满足则清零

            # --- d) 检查该导弹是否被成功规避 ---
            # 定义持续时间要求 (可以从 __init__ 中作为参数传入)
            ESCAPE_DURATION_REQ = 2.0 #3.0 #2.0#1.0
            LOST_DURATION_REQ = 2.0 #3.0 2.0#1.0

            if missile.escape_timer >= ESCAPE_DURATION_REQ:
                # print(f">>> 导弹 {i + 1} 被成功规避 (物理逃逸)! (持续 {missile.escape_timer:.1f}s)")
                missile.terminated = True
                continue  # 继续检查下一个导弹

            if missile.lost_and_separating_duration >= LOST_DURATION_REQ:
                # print(f">>> 导弹 {i + 1} 被成功规避 (丢失目标)! (持续 {missile.lost_and_separating_duration:.1f}s)")
                missile.terminated = True
                continue  # 继续检查下一个导弹

        # 3. [全局终止条件] 在检查完所有导弹后，判断整个回合是否结束

        # a) 如果已经没有活动的导弹了 (所有导弹都被规避)
        if active_missiles_count > 0:
            # 检查是否所有活动的导弹都已经被终结
            all_terminated = all(m.terminated for m in self.missiles if self.t_now >= m.launch_time)
            if all_terminated:
                # print(">>> 所有导弹均已被规避，回合成功结束。")
                self.done = True
                self.success = True  # 因为没有被命中
                return

        # b) 检查仿真是否超时
        if self.t_now >= self.t_end:
            print(f">>> 仿真达到{self.t_end}s, 判定为成功逃离!")
            self.done = True
            # 超时成功的前提是，没有被任何一个导弹命中
            if not self.any_missile_exploded:
                self.success = True
            else:
                self.success = False  # 即使超时，如果之前被命中了，仍然是失败
            return

    def _calculate_final_miss_distance(self):
        """ <<< 多导弹更改 (V9 - 修复逻辑漏洞版) >>> """

        # 防止重复打印
        # if getattr(self, '_final_miss_dist_printed', False):
        #     return

        final_distances = []
        final_statuses = []  # 新增：用于暂存状态字符串

        for i, missile in enumerate(self.missiles):
            # --- 情况 1: 已有确切命中记录 ---
            # (通常由 _check_fuze_for_missile 设置)
            if hasattr(missile, 'final_miss_distance'):
                final_distances.append(missile.final_miss_distance)
                final_statuses.append(f"命中 (引信触发)，距离 {missile.final_miss_distance:.2f} m")
                continue

            # --- 情况 2: 导弹尚未发射 ---
            if self.t_now < missile.launch_time:
                dist = np.linalg.norm(self.aircraft.pos - missile.pos)
                final_distances.append(dist)
                final_statuses.append(f"未发射，当前距离 {dist:.2f} m")
                continue

            # --- 情况 3: 需要从历史轨迹计算最小距离 ---
            start_index = missile.start_step_index
            missile_history_i = self.missile_histories[i]

            # 安全检查：如果导弹刚发射还没产生历史数据
            if not missile_history_i:
                dist = np.linalg.norm(self.aircraft.pos - missile.pos)
                final_distances.append(dist)
                final_statuses.append(f"刚发射 (无轨迹)，距离 {dist:.2f} m")
                continue

            num_missile_steps = len(missile_history_i)
            # 切片对齐
            aircraft_history_aligned = self.aircraft_history[start_index: start_index + num_missile_steps]

            # 长度校验
            if len(aircraft_history_aligned) != num_missile_steps:
                print(f"!!! 轨迹对齐警告: M{i + 1} len={num_missile_steps}, Plane len={len(aircraft_history_aligned)}")
                # 紧急回退：使用当前瞬时距离
                min_dist = np.linalg.norm(self.aircraft.pos - missile.pos)
            else:
                Xt_aligned = np.array(aircraft_history_aligned)
                Yt = np.array(missile_history_i)
                # 计算全过程距离
                delta_pos = Xt_aligned[:, :3] - Yt[:, 3:6]
                R_all = np.linalg.norm(delta_pos, axis=1)
                min_dist = np.min(R_all) if len(R_all) > 0 else np.linalg.norm(self.aircraft.pos - missile.pos)

            final_distances.append(min_dist)

            # --- 核心修正：判定状态 ---
            if missile.terminated:
                # 导弹已经处于“终止”状态（可能是燃料耗尽、丢失目标超时、或者被判定逃逸）
                # 这种情况下，min_dist 确实代表了“最终脱靶量”
                if min_dist <= self.R_kill:
                    # 理论上不应进这里，因为命中会走情况1，但也防一手
                    final_statuses.append(f"命中 (补算)，距离 {min_dist:.2f} m")
                else:
                    final_statuses.append(f"已规避/脱靶，最小距离 {min_dist:.2f} m")
            else:
                # 导弹 terminated 为 False，说明仿真结束时它还在追！
                # 这不是脱靶，而是“中断”
                final_statuses.append(f"仿真中断 (导弹仍活跃)，当前距离 {min_dist:.2f} m")

        # 更新成员变量
        self.miss_distances = final_distances

        # --- 打印日志 ---
        print("\n--- 最终交战结果 ---")
        for i, status in enumerate(final_statuses):
            print(f">>> 导弹 {i + 1}: {status}")
        print("--------------------")

        # 设置标志位
        self._final_miss_dist_printed = True

    # def _calculate_final_miss_distance(self):
    #     """ <<< 多导弹更改 (V8 - 绝对时间对齐版) >>> """
    #     if getattr(self, '_final_miss_dist_printed', False):
    #         return
    #
    #     final_distances = []
    #     for i, missile in enumerate(self.missiles):
    #         if hasattr(missile, 'final_miss_distance'):
    #             final_distances.append(missile.final_miss_distance)
    #             continue
    #
    #         # 如果导弹从未发射，历史为空
    #         if missile.start_step_index == -1:
    #             dist = np.linalg.norm(self.aircraft.pos - missile.pos)
    #             final_distances.append(dist)
    #             continue
    #
    #         # --- 核心修正：基于绝对索引进行切片 ---
    #         start_index = missile.start_step_index
    #         missile_history_i = self.missile_histories[i]
    #         num_missile_steps = len(missile_history_i)
    #
    #         # 从飞机历史中，切出从导弹开始活动到仿真结束的片段
    #         aircraft_history_aligned = self.aircraft_history[start_index: start_index + num_missile_steps]
    #
    #         # --- 健壮性检查 ---
    #         if len(aircraft_history_aligned) != num_missile_steps:
    #             print(
    #                 f"!!! 严重错误: 导弹 {i + 1} 轨迹与飞机轨迹无法对齐! 飞机段: {len(aircraft_history_aligned)}, 导弹段: {num_missile_steps}")
    #             # 使用最后时刻的距离作为紧急备用
    #             dist = np.linalg.norm(self.aircraft.pos - missile.pos)
    #             final_distances.append(dist)
    #             continue
    #
    #         Xt_aligned = np.array(aircraft_history_aligned)
    #         Yt = np.array(missile_history_i)
    #         delta_pos = Xt_aligned[:, :3] - Yt[:, 3:6]
    #         R_all = np.linalg.norm(delta_pos, axis=1)
    #         min_dist = np.min(R_all) if len(R_all) > 0 else np.linalg.norm(self.aircraft.pos - missile.pos)
    #         final_distances.append(min_dist)
    #
    #     self.miss_distances = final_distances
    #
    #     # --- 步骤 3: 打印最终结果 (使用之前完善的逻辑) ---
    #     print("\n--- 最终交战结果 ---")
    #     THREAT_DISTANCE_THRESHOLD = 5000.0
    #
    #     for i, md in enumerate(self.miss_distances):
    #         missile = self.missiles[i]
    #         final_status = ""
    #
    #         if self.t_now < missile.launch_time:
    #             final_status = f"未发射 (初始距离: {md:.2f} m)"
    #         else:
    #             ever_was_threat = md < THREAT_DISTANCE_THRESHOLD
    #             is_hit = md <= self.R_kill
    #
    #             if is_hit:
    #                 final_status = f"命中，脱靶量为：{md:.2f} m"
    #             elif ever_was_threat:
    #                 final_status = f"未命中，脱靶量为：{md:.2f} m"
    #             else:
    #                 final_status = f"交战中止 (最小距离: {md:.2f} m)"
    #
    #         print(f">>> 导弹 {i + 1}: {final_status}")
    #
    #     self._final_miss_dist_printed = True
    #     print("--------------------")

    def render(self, view_init_elev=20, view_init_azim=-150):
        """ <<< 多导弹更改 >>> 渲染所有导弹的轨迹。"""
        if not self.aircraft_history:
            print("警告：历史轨迹数据为空，无法进行可视化。")
            return

        aircraft_traj = np.array(self.aircraft_history)[:, :3]
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(aircraft_traj[:, 2], aircraft_traj[:, 0], aircraft_traj[:, 1], 'r--', label='目标轨迹')
        ax.scatter(aircraft_traj[0, 2], aircraft_traj[0, 0], aircraft_traj[0, 1], color='orange', s=50,
                   label='飞机起点')

        # <<< 多导弹更改 >>> 循环绘制每个导弹的轨迹
        colors = ['b', 'c', 'm', 'y']
        for i, missile_history in enumerate(self.missile_histories):
            if not missile_history: continue

            missile_traj = np.array(missile_history)[:, 3:6]
            color = colors[i % len(colors)]

            ax.plot(missile_traj[:, 2], missile_traj[:, 0], missile_traj[:, 1], f'{color}-', label=f'导弹 {i + 1} 轨迹')
            ax.scatter(missile_traj[0, 2], missile_traj[0, 0], missile_traj[0, 1], color='g', s=50,
                       label=f'导弹 {i + 1} 起点' if i == 0 else "")

        # ... (省略绘制最近点和诱饵弹的代码，逻辑类似) ...

        ax.set_xlabel('东 (Z) / m')
        ax.set_ylabel('北 (X) / m')
        ax.set_zlabel('天 (Y) / m')
        ax.legend()
        ax.set_title('多导弹空战对抗三维轨迹')
        ax.grid(True)
        plt.show()

    # ============================================================================
    # 以下是未发生重大逻辑变化的辅助方法，但可能需要传入额外参数
    # ============================================================================

    def _calculate_los_rate(self, target_pos: np.ndarray, prev_los_angles: tuple, dt: float, missile: Missile) -> tuple:
        """ <<< 多导弹更改 >>> 需要传入具体是哪个导弹。"""
        prev_theta_L, prev_phi_L = prev_los_angles
        R_vec = target_pos - missile.pos  # 使用传入导弹的位置
        R_mag = np.linalg.norm(R_vec)
        if R_mag < 1e-6:
            return (prev_theta_L or 0.0), (prev_phi_L or 0.0), 0.0, 0.0

        theta_L = np.arcsin(np.clip(R_vec[1] / R_mag, -1.0, 1.0))
        phi_L = np.arctan2(R_vec[2], R_vec[0])

        if prev_theta_L is None or prev_phi_L is None:
            theta_L_dot, phi_L_dot = 0.0, 0.0
        else:
            theta_L_dot = (theta_L - prev_theta_L) / dt
            dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
            phi_L_dot = dphi / dt
        return theta_L, phi_L, theta_L_dot, phi_L_dot

    # _compute_relative_beta2 签名不变，因为它是计算两个对象间的关系
    def _compute_relative_beta2(self, x_target, x_missile):
        psi_t = x_target[5]
        cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)
        R_vec_beta = x_missile[3:6] - x_target[0:3]
        R_proj_world = np.array([R_vec_beta[0], R_vec_beta[2]])
        x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
        z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
        threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
        return threat_angle_rad + 2 * np.pi if threat_angle_rad < 0 else threat_angle_rad

    def _calculate_equivalent_target_for_missile(self, missile: Missile):
        """
        (V2 - 独立导引头版)
        为【单个特定】导弹计算其导引头看到的等效红外质心。
        """
        I_total = 0.0
        numerator = np.zeros(3)
        aircraft_vel_vec = self.aircraft.get_velocity_vector()

        # <<< 核心修改：使用传入的 missile 对象的状态 >>>
        missile_vel_vec = missile.get_velocity_vector()

        # --- 1. 检查飞机是否在该导弹的视场内 ---
        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            missile.state_vector, self.aircraft.state_vector,
            missile_vel_vec, aircraft_vel_vec, self.t_now)

        if lock_aircraft:
            # <<< 核心修改：使用传入的 missile 对象的视角计算红外强度 >>>
            beta_p = self._compute_relative_beta_for_ir(self.aircraft.state_vector, missile.state_vector)
            I_p = self._infrared_intensity_model(beta_p)
            I_total += I_p
            numerator += I_p * self.aircraft.pos

        # --- 2. 遍历诱饵弹，检查是否在该导弹的视场内 ---
        for flare in self.flare_manager.flares:
            if flare.get_intensity(self.t_now) <= 1e-3: continue
            flare_state_dummy = np.concatenate((flare.pos, [0, 0, 0, 0]))

            # <<< 核心修改：使用传入的 missile 对象的视角检查诱饵弹 >>>
            lock_flare, _, _, _, _ = self._check_seeker_lock(
                missile.state_vector, flare_state_dummy,
                missile_vel_vec, flare.vel, self.t_now)

            if lock_flare:
                I_k = flare.get_intensity(self.t_now)
                I_total += I_k
                numerator += I_k * flare.pos

        # --- 3. 计算最终质心或返回 None ---
        if I_total > 1e-6:
            return numerator / I_total
        else:
            # 如果这个导弹的导引头什么都看不见，返回 None 表示失锁
            return None

    # 其他辅助函数 (_check_seeker_lock, _infrared_intensity_model, _compute_relative_beta_for_ir, _execute_flare_program)
    # 它们的内部逻辑不需要改变，因为它们计算的是两个特定对象之间的关系，或者只与飞机和诱饵弹相关。
    # 在调用它们时，需要传入正确的对象（例如，在循环中传入 `missile.state_vector`）。
    # 为保持代码简洁，这里不再重复粘贴它们。

    def _check_seeker_lock(self, x_missile, x_target, V_missile_vec, V_target_vec, t_now):
        """
        纯粹的锁定逻辑计算，精确复现您原代码。
        注意：这里的参数都是完整的状态向量或速度矢量。
        """
        # (中文) 从 __init__ 中获取导引头参数

        R_vec = np.array(x_target[0:3]) - np.array(x_missile[3:6])
        R_mag = np.linalg.norm(R_vec)

        Flag_D = R_mag <= self.D_max
        if not Flag_D: return False, False, True, True, True

        norm_product = R_mag * np.linalg.norm(V_missile_vec) + 1e-6
        cos_Angle = np.dot(R_vec, V_missile_vec) / norm_product

        Flag_A = (cos_Angle >= np.cos(self.Angle_IR_rad))  # (修正) 视场角通常指全角，判断用半角
        if not Flag_A: return False, True, False, True, True

        delta_V = V_missile_vec - V_target_vec
        # (修正) omega_R 的分母应该是 R_mag，而不是 R_vec
        omega_R = np.linalg.norm(np.cross(R_vec, delta_V)) / (R_mag ** 2 + 1e-6)
        Flag_omega = omega_R <= self.omega_max_rad_s
        if not Flag_omega: return False, True, True, False, True

        Flag_T = t_now <= self.T_max
        if not Flag_T: return False, True, True, True, False

        return True, True, True, True, True

    def _infrared_intensity_model(self, beta_rad: float) -> float:
        """飞机的红外辐射强度模型。"""
        beta_deg = np.array([0, 40, 90, 140, 180])
        intensity_vals = np.array([3800, 5000, 2500, 2000, 800])
        beta_rad_points = np.deg2rad(beta_deg)

        # (中文) 使用线性插值，更简单且稳定
        intensity = np.interp(beta_rad, beta_rad_points, intensity_vals)
        return max(intensity, 0.0)

    def _compute_relative_beta_for_ir(self, x_target, x_missile) -> float:
        """
        计算用于红外模型的无符号角度 [0, pi]。
        精确复现您原来的 compute_relative_beta 逻辑。
        """
        # 导弹指向飞机的矢量，并投影到水平面
        R_vec = x_target[0:3] - x_missile[3:6]
        R_proj = np.array([R_vec[0], 0.0, R_vec[2]])

        if np.linalg.norm(R_proj) < 1e-6: return 0.0

        # 飞机机头在水平面上的朝向单位向量
        psi_t = x_target[5]
        V_body = np.array([np.cos(psi_t), 0.0, np.sin(psi_t)])

        cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))

        return beta