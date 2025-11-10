# 文件: environment.py

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- (中文) 从我们创建的模块中导入所有需要的类 ---
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile
from .decoys import FlareManager
from .reward_system import RewardCalculator
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
        observation_dim = 3 * self.num_missiles + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float64
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
        self.omega_max_rad_s = 12.0
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

        self.o_ir = 30
        self.N_infrared = 30
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
        y_t = np.random.uniform(5000, 10000)
        aircraft_pitch = np.deg2rad(np.random.uniform(-30, 30))
        aircraft_vel = np.random.uniform(200, 400)
        initial_aircraft_state = np.array([0, y_t, 0, aircraft_vel, aircraft_pitch, 0, 0, 0.0])
        self.aircraft.reset(initial_state=initial_aircraft_state)

        # (导弹初始状态 - 循环创建)
        self.missiles.clear()
        self.prev_missile_states = []
        self.prev_R_rels = []
        self.prev_theta_Ls, self.prev_phi_Ls = [], []
        self.last_valid_theta_dots, self.last_valid_phi_dots = [], []

        # 为第一个导弹设置一个基准来袭方向
        base_theta1 = np.deg2rad(np.random.uniform(-180, 180))

        for i in range(self.num_missiles):
            # 随机化每个导弹的参数
            R_dist = np.random.uniform(9000, 11000)

            # 让第二个导弹从一个稍微不同的角度来袭
            theta1_offset = np.deg2rad(np.random.uniform(-30, 30) * np.random.choice([-1, 1])) if i > 0 else 0
            theta1 = base_theta1 + theta1_offset

            x_m = R_dist * (-np.cos(theta1))
            z_m = R_dist * np.sin(theta1)
            y_m = y_t + np.random.uniform(-2000, 2000)
            missile_vel = np.random.uniform(700, 900)

            # 计算初始视线角
            R_vec = self.aircraft.pos - np.array([x_m, y_m, z_m])
            R_mag = np.linalg.norm(R_vec)
            theta_L = np.arcsin(R_vec[1] / R_mag)
            phi_L = np.arctan2(R_vec[2], R_vec[0])

            initial_missile_state = np.array([missile_vel, theta_L, phi_L, x_m, y_m, z_m])
            missile = Missile(initial_missile_state)

            # <<< 多导弹更改 >>> 为导弹附加额外状态
            missile.id = i
            # 第一个导弹立即发射，后续导弹延迟发射
            missile.launch_time = 0.0 if i == 0 else np.random.uniform(2.0, 8.0)
            missile.terminated = False  # 标志位，表示该导弹是否已命中或被规避

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
            self.tacview.stream_frame(0.0, self.aircraft, self.missiles, [])

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
            # 为日志记录计算最终脱靶量
            self._calculate_final_miss_distance()

            # 根据已经判定好的 self.success 状态, 计算稀疏奖励
            if self.success:
                reward += self.reward_calculator.W
            else:
                reward += self.reward_calculator.U
        else:  # 如果回合仍在继续
            # 计算密集奖励
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
        """ <<< 多导弹更改 >>> 执行单个物理时间步的更新，处理多个导弹。"""
        # --- 1. 更新诱饵弹和飞机 (只需一次) ---
        self.flare_manager.update(self.t_now, dt, self.aircraft)
        self.aircraft.update(aircraft_action)

        # --- 2. 计算等效目标 (只需一次) ---
        target_pos_equiv = self._calculate_equivalent_target()

        # --- 3. <<< 多导弹更改 >>> 遍历并更新每个导弹 ---
        for i, missile in enumerate(self.missiles):
            # 如果导弹还未发射或已终结，则跳过
            if self.t_now < missile.launch_time or missile.terminated:
                continue

            # 计算该导弹的视线角速率
            if target_pos_equiv is not None:
                current_theta_L, current_phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
                    target_pos_equiv, (self.prev_theta_Ls[i], self.prev_phi_Ls[i]), dt, missile)
            else:
                current_theta_L = self.prev_theta_Ls[i]
                current_phi_L = self.prev_phi_Ls[i]
                theta_L_dot, phi_L_dot = self.last_valid_theta_dots[i], self.last_valid_phi_dots[i]

            missile_state_next = missile._missile_dynamics(missile.state_vector, dt, theta_L_dot, phi_L_dot)
            missile.state = missile_state_next

            # 更新该导弹的历史状态
            if target_pos_equiv is not None:
                self.prev_theta_Ls[i] = current_theta_L
                self.prev_phi_Ls[i] = current_phi_L

            # <<< 多导弹更改 >>> 将历史记录追加到对应的子列表中
            self.missile_histories[i].append(missile.state_vector.copy())

        # --- 4. 更新时间和通用历史记录 ---
        self.t_now += dt
        self.tacview_global_time += dt
        self.aircraft_history.append(self.aircraft.state_vector.copy())
        self.time_history.append(self.t_now)

        # --- 5. <<< 多导弹更改 >>> 检查所有导弹的引信和终止条件 ---
        for i in range(self.num_missiles):
            if not self.missiles[i].terminated and self.t_now >= self.missiles[i].launch_time:
                # 检查引信，如果命中，会直接设置 self.done=True
                self._check_fuze_for_missile(dt, i)
                if self.done: break  # 任何一个命中则立即停止检查

        if not self.done:
            self._update_episode_status(dt)  # 检查其他终止条件（如逃逸）

        # --- 6. Tacview 更新 ---
        if self.tacview_enabled and not self.any_missile_exploded:
            # 传递导弹列表
            self.tacview.stream_frame(self.tacview_global_time, self.aircraft, self.missiles,
                                      self.flare_manager.flares)

    def _get_observation(self) -> np.ndarray:
        """ <<< 多导弹更改 (V2 - 修正版) >>> 组装观测向量，为无效导弹设置无意义值。"""
        obs_parts = []

        # --- 1. 遍历每个导弹，计算其相对观测值 ---
        for missile in self.missiles:
            # <<< 核心修改 >>> 检查导弹是否有效（已发射且未终结）
            if missile.terminated or self.t_now < missile.launch_time:
                # 如果导弹无效，提供一个标准的“无威胁”观测值
                o_dis_norm = 1.0  # 归一化距离为1 (最远)
                o_beta_norm = 0.0  # 归一化方位角为0 (正前方)
                o_theta_L_rel_norm = 0.5  # 归一化俯仰角为0.5 (水平)
                obs_parts.extend([o_dis_norm, o_beta_norm, o_theta_L_rel_norm])
                continue  # 处理下一个导弹

            # --- 以下代码只对有效导弹执行 ---
            R_vec = self.aircraft.pos - missile.pos
            R_rel = np.linalg.norm(R_vec)

            # 使用内部函数计算 beta 和 theta_L
            o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, missile.state_vector)
            Ry_rel = missile.pos[1] - self.aircraft.pos[1]
            o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))

            # 归一化
            # o_dis_norm = np.clip(R_rel / 30000.0, 0, 1) # 修正：使用更平滑的归一化
            o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10  # 保持原来的离散化归一化
            o_beta_norm = o_beta_rad / (2 * np.pi)
            o_theta_L_rel_norm = (o_theta_L_rel_rad + np.pi / 2) / np.pi  # 修正：确保范围是 [0, 1]

            obs_parts.extend([o_dis_norm, o_beta_norm, o_theta_L_rel_norm])

        # --- 2. 获取飞机自身状态 (逻辑不变) ---
        o_av_norm = (self.aircraft.velocity - 100) / 300
        o_h_norm = (self.aircraft.pos[1] - 1000) / 14000
        o_ae_norm = (self.aircraft.state_vector[4] + np.pi / 2) / np.pi  # 修正：确保范围是 [0, 1]
        o_am_norm = (self.aircraft.state_vector[6] + np.pi) / (2 * np.pi)  # 修正：确保范围是 [0, 1]
        o_q_norm = (self.aircraft.roll_rate_rad_s + 4.0 * np.pi / 3.0) / (8.0 * np.pi / 3.0)  # 修正：确保范围是 [0, 1]
        o_ir_norm = self.o_ir / self.N_infrared

        aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_norm, o_ir_norm, o_q_norm]

        # --- 3. 拼接成最终的观测向量 ---
        observation = np.array(obs_parts + aircraft_obs)
        return observation.astype(np.float32)

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
            print(f">>> 导弹 {missile_index + 1} 引信引爆！当前时刻直接命中，距离 = {min_dist_final:.2f} m")

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
                print(f">>> 导弹 {missile_index + 1} 引信引爆！区间内最小脱靶量 = {min_dist_final:.2f} m")

        # --- 统一处理命中事件 ---
        if is_hit:
            self.done = True
            self.success = False
            # 确保只添加一次脱靶量
            if not self.missiles[missile_index].terminated:
                self.miss_distances.append(min_dist_final)

            self.any_missile_exploded = True
            self.missiles[missile_index].terminated = True

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
        """ <<< 多导弹更改 >>> 检查所有可能导致回合结束或某个导弹终结的条件。 """
        if self.done: return

        # 1. 检查飞机坠毁等立即失败条件
        if self.aircraft.pos[1] <= 500 or self.aircraft.pos[1] >= 15000 or self.aircraft.velocity <= 150:
            print(f">>> 飞机状态异常！仿真终止。")
            # <<< 确保这行代码在这里，用于重置标志位 >>>
            self._final_miss_dist_printed = True
            self.done, self.success = True, False
            return

        # 2. 遍历每个未终结的导弹，检查其是否被规避
        active_missiles_remaining = False
        for i, missile in enumerate(self.missiles):
            if missile.terminated or self.t_now < missile.launch_time:
                continue

            active_missiles_remaining = True
            current_R_rel = np.linalg.norm(self.aircraft.pos - missile.pos)
            range_rate = (current_R_rel - self.prev_R_rels[i]) / dt

            # a) 检查物理逃逸
            if missile.velocity < self.aircraft.velocity and range_rate > 5.0:
                # (简化逻辑) 如果连续逃逸超过2秒，则认为此导弹被规避
                # 在真实实现中，每个导弹需要有自己的逃逸计时器
                print(f">>> 导弹 {i + 1} 被成功规避 (物理逃逸)!")
                missile.terminated = True
                continue  # 检查下一个导弹

            # b) 检查信息逃逸 (简化版)
            lock, _, _, _, _ = self._check_seeker_lock(
                missile.state_vector, self.aircraft.state_vector,
                missile.get_velocity_vector(), self.aircraft.get_velocity_vector(), self.t_now
            )
            if not lock and range_rate > 0:
                print(f">>> 导弹 {i + 1} 被成功规避 (丢失目标)!")
                missile.terminated = True
                continue

        # 3. 检查全局终止条件
        if not active_missiles_remaining:
            print(">>> 所有导弹均已处理，回合结束。")
            self.done = True
            # 此时 self.success 默认为 True, 因为如果被命中，done 早就为 True 了
            if not self.any_missile_exploded:
                self.success = True
            return

        if self.t_now >= self.t_end:
            print(f">>> 仿真达到{self.t_end}s, 判定为成功逃离!")
            self.done = True
            if not self.any_missile_exploded:
                self.success = True
            return

    def _calculate_final_miss_distance(self):
        """ <<< 多导弹更改 (V3 - 再次修正) >>> 为每个导弹计算最终脱靶量，处理引信和非同步轨迹。"""

        # --- 步骤 1: 检查是否已有因引信引爆记录的脱靶量 ---
        # 如果有，我们保留这些精确值，只为其他导弹计算
        # 创建一个字典来存储已计算的脱靶量，以导弹索引为键
        final_distances = {}
        if self.any_missile_exploded and self.miss_distances:
            # 找到是哪个导弹爆炸了
            # 假设第一个爆炸的导弹的脱靶量被记录在 self.miss_distances[0]
            # 并且该导弹的 terminated 标志位已被设为 True
            exploded_missile_found = False
            for i, missile in enumerate(self.missiles):
                if missile.terminated and not exploded_missile_found:
                    final_distances[i] = self.miss_distances[0]
                    exploded_missile_found = True
                    # 注意：这里假设只有一个导弹会引爆并结束回合
                    # 如果允许多个导弹在同一step引爆，逻辑需要更复杂
                    break

        # --- 步骤 2: 为所有尚未计算脱靶量的导弹进行计算 ---
        for i in range(self.num_missiles):
            # 如果该导弹的脱靶量已经有了（来自引信），则跳过
            if i in final_distances:
                continue

            missile_history_i = self.missile_histories[i]
            num_missile_steps = len(missile_history_i)

            if num_missile_steps == 0:
                dist = np.linalg.norm(self.aircraft.pos - self.missiles[i].pos)
                final_distances[i] = dist
                continue

            aircraft_history_aligned = self.aircraft_history[-num_missile_steps:]

            if len(aircraft_history_aligned) != num_missile_steps:
                print(f"警告: 导弹 {i + 1} 轨迹与飞机轨迹无法对齐!")
                dist = np.linalg.norm(self.aircraft.pos - self.missiles[i].pos)
                final_distances[i] = dist
                continue

            Xt_aligned = np.array(aircraft_history_aligned)
            Yt = np.array(missile_history_i)

            delta_pos = Xt_aligned[:, :3] - Yt[:, 3:6]
            R_all = np.linalg.norm(delta_pos, axis=1)

            min_dist = np.min(R_all) if len(R_all) > 0 else np.linalg.norm(self.aircraft.pos - self.missiles[i].pos)
            final_distances[i] = min_dist

        # --- 步骤 3: 将最终结果按导弹顺序整理到 self.miss_distances 列表中 ---
        self.miss_distances = [final_distances[i] for i in sorted(final_distances.keys())]

        # --- 步骤 4: 打印最终结果 ---
        # (可选) 检查是否已经打印过，避免重复输出
        if not getattr(self, '_final_miss_dist_printed', False):
            for i, md in enumerate(self.miss_distances):
                is_hit = md <= self.R_kill
                print(f">>> (最终计算) 导弹 {i + 1}: {'命中' if is_hit else '未命中'}，脱靶量为：{md:.2f} m")
            self._final_miss_dist_printed = True

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

    # _calculate_equivalent_target 的逻辑完全不变，它对所有导弹都是通用的
    def _calculate_equivalent_target(self):
        I_total = 0.0
        numerator = np.zeros(3)
        aircraft_vel_vec = self.aircraft.get_velocity_vector()

        # 假设所有导弹共享一个导引头模型，我们用第一个导弹来检查
        # 这是一个简化，在真实世界中每个导弹导引头独立
        missile_vel_vec = self.missiles[0].get_velocity_vector()

        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            self.missiles[0].state_vector, self.aircraft.state_vector,
            missile_vel_vec, aircraft_vel_vec, self.t_now)

        if lock_aircraft:
            beta_p = self._compute_relative_beta_for_ir(self.aircraft.state_vector, self.missiles[0].state_vector)
            I_p = self._infrared_intensity_model(beta_p)
            I_total += I_p
            numerator += I_p * self.aircraft.pos

        for flare in self.flare_manager.flares:
            if flare.get_intensity(self.t_now) <= 1e-3: continue
            flare_state_dummy = np.concatenate((flare.pos, [0, 0, 0, 0]))
            lock_flare, _, _, _, _ = self._check_seeker_lock(
                self.missiles[0].state_vector, flare_state_dummy,
                missile_vel_vec, flare.vel, self.t_now)
            if lock_flare:
                I_k = flare.get_intensity(self.t_now)
                I_total += I_k
                numerator += I_k * flare.pos

        if I_total > 1e-6:
            return numerator / I_total
        else:
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