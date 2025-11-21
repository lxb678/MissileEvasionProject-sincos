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
from Interference_code.PPO_model.PPO_evasion_fuza.PPOMLP混合架构.Hybrid_PPO_混合架构 import CONTINUOUS_DIM, DISCRETE_DIMS, \
    DISCRETE_ACTION_MAP

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv(gym.Env):
    """
    强化学习的空战环境主类。
    负责协调仿真流程、管理对象、计算奖励和提供RL接口(reset, step)。
    (Gymnasium兼容版) 强化学习的空战环境主类。
    """

    def __init__(self, tacview_enabled=False, dt=0.02):
        # <<< GYMNASIUM CHANGE >>> 必须调用父类的 __init__
        super().__init__()
        # <<< GYMNASIUM CHANGE >>> 定义动作空间和观测空间
        # --- 1. <<< 更改 >>> 定义动作空间 (Action Space) ---
        # 使用 spaces.Dict 来定义结构化的混合动作空间
        self.action_space = spaces.Dict({
            # 连续机动动作
            "continuous_actions": spaces.Box(
                low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),  # throttle, elevator, aileron, rudder
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(CONTINUOUS_DIM,)
            ),
            # 离散投放决策
            "discrete_actions": spaces.MultiDiscrete(
                # MultiDiscrete 接受一个数组，每个元素代表该维度有多少个选项
                # 例如 [2, 3, 3] 表示第一个维度有2个选项(0,1)，后两个各有3个选项(0,1,2)
                np.array([
                    2,  # flare_trigger: 0或1
                    DISCRETE_DIMS['salvo_size'],  # e.g., 3 options
                    # DISCRETE_DIMS['intra_interval'],  # e.g., 3 options
                    DISCRETE_DIMS['num_groups'],  # e.g., 3 options
                    DISCRETE_DIMS['inter_interval'],  # e.g., 3 options
                ])
            )
        })

        # # --- 2. 定义观测空间 (Observation Space) ---
        # # 9个观测值，根据 _get_observation() 的输出
        # # [o_dis, o_beta, o_theta_L, o_av, o_h, o_ae, o_am, o_ir, o_q]
        # # 最好使用真实的归一化范围，但为了简单起见，可以先用一个较宽松的范围
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(9,),
        #     dtype=np.float64
        # )
        # # --- 修改后的代码 (After) ---
        # # 新维度 = 1个导弹(5) + 飞机自身(8) = 13
        # observation_dim = 13

        # --- 修改后的代码 (After) ---
        # 维度计算 (单导弹): 导弹特征(4) + 飞机自身特征(7) = 11
        observation_dim = 11
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32  # 推荐使用 float32
        )
        # <<< 修改结束 >>>

        # --- 仿真参数 ---
        # self.dt = 0.1  # 这是外部决策步长
        self.t_end = 60.0
        self.R_kill = 12.0

        # --- (中文) 新增：补全缺失的内部循环参数 ---
        # 这些参数决定了在 step 方法内部的物理仿真是如何运行的
        self.dt_normal = dt #0.02  # 大步长 (当距离远时)
        self.dt_small = dt #0.02  # 小步长 (当距离近时)
        self.dt_flare = dt #0.02  # 投放诱饵弹时的步长
        self.R_switch = 500  # (米) 切换大小步长的距离阈值

        # (中文) 从您的主脚本中看到您还引用了 dt_dec，这里也为您补上
        # 它似乎与 self.dt 含义相同，代表决策步长
        self.dt_dec = 0.2

        self.D_max = 30000.0  # 导引头最大搜索范围 (m)
        self.Angle_IR_rad = np.deg2rad(90)  # 导引头最大视场角度 (弧度)
        self.omega_max_rad_s = np.deg2rad(90.0)#12.0  # 导引头最大角速度 (弧度/秒)
        self.T_max = 60.0  # 导引头最大搜索时间 (秒)

        # --- 核心组件实例化 ---
        initial_state = np.array([0, 1000, 0, 200, 0.1, 0.0, 0.0, 0.0])
        self.aircraft = Aircraft(dt=self.dt_small, initial_state=initial_state)
        self.missile = None
        # <<< 更改 >>> FlareManager现在需要处理更复杂的逻辑，但初始化不变
        # flare_per_group 在新逻辑中不再直接使用，但可以保留
        self.flare_manager = FlareManager(flare_per_group=1)
        self.reward_calculator = RewardCalculator()

        self.tacview_enabled = tacview_enabled
        self.tacview = TacviewInterface() if tacview_enabled else None

        # --- 状态变量 ---
        self.t_now = 0.0
        self.tacview_global_time = 0.0  # (中文) 新增: 跨回合的、连续的Tacview时间
        self.done = False
        self.success = False
        self.miss_distance = None
        self.o_ir = 30  # 初始诱饵弹数量
        self.N_infrared = 30

        # 用于存储历史状态，以计算变化率 (如速度变化、距离变化)
        self.prev_aircraft_state = None
        self.prev_missile_state = None
        self.prev_R_rel = None
        self.prev_theta_L, self.prev_phi_L = None, None

        # 用于存储历史轨迹的列表 ---
        self.aircraft_history = []
        self.missile_history = []
        self.time_history = []

        # 用于Tacview的最终事件
        self.missile_exploded = False
        self.episode_count = 0
        # <<< 新增 >>> 将仿真步长 dt 存储为成员变量，以便传递给飞机
        # JSBSim需要知道物理仿真的时间步长
        self.physical_dt = dt #0.02  # 假设您的 dt_normal/small/flare 都是0.1
        # 如果它们不同，需要在 run_one_step 中传递正确的 dt

    def reset(self, seed=None, options=None) -> tuple:
        """
        重置环境到初始状态，为新的一回合做准备。
        (Gymnasium兼容版) 重置环境。
        """
        # <<< GYMNASIUM CHANGE >>> 处理可选的 seed 参数
        # 如果需要基于seed的随机化，可以在这里使用
        super().reset(seed=seed)

        # --- 1. (Tacview) 清理上一回合的对象 ---
        if self.tacview_enabled and self.episode_count > 0:
            # (中文) 使用全局时间来清理，确保时间戳是递增的
            self.tacview.end_of_episode(
                t_now=self.tacview_global_time,
                flare_count=len(self.flare_manager.flares),
                missile_exploded=self.missile_exploded
            )
            # --- (中文) 关键补充：重置标志位 ---
            self.tacview.tacview_final_frame_sent = False

        # --- 2. 重置所有状态变量 ---
        self.t_now = 0.0
        self.done = False
        self.success = False
        self.miss_distance = None
        self.missile_exploded = False
        self.episode_count += 1
        self.o_ir = self.N_infrared

        # --- 3. 重置所有组件 ---
        self.flare_manager.reset()
        self.reward_calculator.reset()

        # --- 4. 随机化初始场景 ---
        y_t = np.random.uniform(5000, 10000)
        R_dist = np.random.uniform(9000, 11000)
        theta1 = np.deg2rad(np.random.uniform(-180, 180))

        # 计算导弹初始位置，使其朝向飞机
        x_m = R_dist * (-np.cos(theta1))
        z_m = R_dist * np.sin(theta1)
        y_m = y_t + np.random.uniform(-2000, 2000)

        # 5.
        aircraft_pitch = np.deg2rad(np.random.uniform(-30, 30))

        # 6.
        aircraft_vel = np.random.uniform(300, 400)
        # aircraft_vel = 400
        # 7.
        missile_vel = np.random.uniform(800, 900)

        # <<< 核心更改 >>> 像以前一样创建初始状态向量
        initial_aircraft_state = np.array([
            0, y_t, 0,  # pos (m)
            aircraft_vel,  # Vel (m/s)
            aircraft_pitch,  # theta (rad)
            np.deg2rad(0),  # psi (rad)
            0,  # phi (rad)
            0.0  # p_real (rad/s)
        ])
        # print(initial_aircraft_state)
        # <<< 核心更改 >>> Aircraft类的实例化变得简单
        # 只需要传递 physical_dt 和标准单位的 initial_state
        self.aircraft.reset(initial_state=initial_aircraft_state)

        # 计算初始视线角，让导弹直接对准飞机
        R_vec = self.aircraft.pos - np.array([x_m, y_m, z_m])
        R_mag = np.linalg.norm(R_vec)
        theta_L = np.arcsin(R_vec[1] / R_mag) # 理想俯仰角
        phi_L = np.arctan2(R_vec[2], R_vec[0]) # 理想偏航角

        # --- <<< 新增修改开始：添加初始指向误差 >>> ---
        # 设定最大发射误差角度 (例如 20度)
        # 这一步模拟了导弹发射时的不确定性，或者由于挂架/发射方式导致的初始离轴
        initial_heading_error_deg = 60.0
        error_rad = np.deg2rad(initial_heading_error_deg)

        # 在俯仰 (theta) 和 偏航 (phi) 上分别增加随机扰动
        # 导弹初始速度方向 = 理想视线方向 + 随机误差
        delta_theta = np.random.uniform(-error_rad, error_rad)
        delta_phi = np.random.uniform(-error_rad, error_rad)

        missile_init_theta = theta_L + delta_theta
        missile_init_phi = phi_L + delta_phi
        # --- <<< 新增修改结束 >>> ---

        initial_missile_state = np.array([
            missile_vel,  # Vel
            # theta_L,  # theta
            # phi_L,  # psi
            missile_init_theta,  # theta (含误差)
            missile_init_phi,  # psi (含误差)
            x_m, y_m, z_m  # pos
        ])
        self.missile = Missile(initial_missile_state)
        # print(initial_missile_state)

        # --- 5. 初始化历史状态 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.prev_missile_state = self.missile.state_vector.copy()
        self.prev_R_rel = R_mag
        self.prev_theta_L, self.prev_phi_L = None, None
        self.last_valid_theta_dot = 0
        self.last_valid_phi_dot = 0

        # --- 在 reset 时清空历史记录 ---
        self.aircraft_history = []
        self.missile_history = []
        self.time_history = []

        # print(self.aircraft.pos)

        # 6. (Tacview) 发送初始帧
        if self.tacview_enabled:
            # 重置Tacview内部状态
            self.tacview.tacview_final_frame_sent = False
            self.tacview.stream_frame(0.0, self.aircraft, self.missile, [])

        # <<< GYMNASIUM CHANGE >>> 返回 observation 和一个空的 info 字典
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: dict) -> tuple:
        """
        执行一个决策时间步。
        内部会根据情况，以不同的小步长多次调用 run_one_step。
        (Gymnasium兼容版) 执行一个决策时间步。
        """
        # --- 1. <<< 更改 >>> 解析结构化的动作字典 ---
        continuous_cmds = action["continuous_actions"]
        discrete_cmds_indices = action["discrete_actions"]

        throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd = continuous_cmds

        trigger_cmd_idx = discrete_cmds_indices[0]
        salvo_size_idx = discrete_cmds_indices[1]
        # intra_interval_idx = discrete_cmds_indices[2]
        intra_interval = 0.04
        num_groups_idx = discrete_cmds_indices[2]
        inter_interval_idx = discrete_cmds_indices[3]

        # --- 2. <<< 新增 >>> 执行投放程序逻辑 ---
        release_flare_program = (trigger_cmd_idx == 1)
        if release_flare_program:
            self._execute_flare_program(salvo_size_idx, intra_interval, num_groups_idx, inter_interval_idx)

        # --- 3. 保存决策步开始前的状态，用于奖励计算 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.prev_missile_state = self.missile.state_vector.copy()
        self.prev_R_rel = np.linalg.norm(self.aircraft.pos - self.missile.pos)

        # --- 4. 物理仿真循环 (逻辑不变) ---
        R_rel_start = self.prev_R_rel
        if R_rel_start < self.R_switch:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_small)), self.dt_small
        elif self.flare_manager.schedule:  # 检查是否有待投放的计划
            num_steps, step_dt = int(round(self.dt_dec / self.dt_flare)), self.dt_flare
        else:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_normal)), self.dt_normal

        # 执行内部物理循环
        for i in range(num_steps):
            # 飞机机动指令在整个决策步内保持不变
            aircraft_action = [throttle_cmd, elevator_cmd, aileron_cmd, rudder_cmd]

            # <<< 更改 >>> release_flare 参数现在不再直接使用，投放由 FlareManager 的 schedule 控制
            # 但 run_one_step 中仍需检查 schedule
            self.run_one_step(step_dt, aircraft_action)

            if self.done:
                break

        # --- 3. (循环后) 检查其他所有终止条件 ---# --- 3. (简化) 终止条件检查已经移入 run_one_step ---
        # self._check_termination_conditions(step_dt)  # 检查坠地、逃逸等

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

        # <<< GYMNASIUM CHANGE >>> 区分 terminated 和 truncated
        # 在我们的场景中，所有 self.done=True 的情况都是终局
        terminated = self.done

        # 检查是否因为超时而结束，如果是，则 truncated 为 True
        truncated = self.t_now >= self.t_end
        if truncated:
            terminated = True  # 通常超时也算 terminated
            self.success = True

        observation = self._get_observation()
        info = {}
        # 只有当回合真的结束时，才把最终信息放入 info 字典
        if terminated or truncated:
            # self.success 的值是在 _check_termination_conditions 中设置的
            info["success"] = self.success
        # return self._get_observation(), reward, self.done, {}, {}
        return observation, reward, terminated, truncated, info

    def run_one_step(self, dt, aircraft_action):
        """
        (修正版) 执行单个物理时间步的更新。
        投放计划的制定已移至 step 方法，这里只负责执行。
        """
        # --- 1. 投放逻辑 ---
        # <<< 核心修正：删除下面这一行错误的代码 >>>
        # self.flare_manager.release_flare_group(self.t_now)  # <--- 删除此行

        # <<< 逻辑修正：现在只更新诱饵弹状态，并根据 schedule 创建新的 >>>
        # FlareManager 的 update 方法内部会检查 self.schedule 列表，
        # 如果当前时间 t 匹配了计划中的某个时间点，它会自动创建新的 Flare 实例。
        self.flare_manager.update(self.t_now, dt, self.aircraft)  # 注意：这里传入的是 t 时刻的 aircraft 对象

        # c) 导引头：根据【当前】的飞机和导弹状态，计算导引目标
        target_pos_equiv = self._calculate_equivalent_target()
        # d) 导弹：根据导引目标，计算下一时刻的状态
        if target_pos_equiv is not None:
            current_theta_L, current_phi_L, theta_L_dot, phi_L_dot = self._calculate_los_rate(
                target_pos_equiv, (self.prev_theta_L, self.prev_phi_L), dt)
            # self.prev_theta_L, self.prev_phi_L = new_theta_L, new_phi_L
            # self.last_valid_theta_dot, self.last_valid_phi_dot = theta_L_dot, phi_L_dot
        else:
            current_theta_L = self.prev_theta_L
            current_phi_L = self.prev_phi_L
            # theta_L_dot, phi_L_dot = self.last_valid_theta_dot, self.last_valid_phi_dot
            theta_L_dot, phi_L_dot = 0.0,0.0

        missile_state_next = self.missile._missile_dynamics(self.missile.state_vector, dt, theta_L_dot, phi_L_dot)

        # --- 3. 更新时间并记录历史 ---
        self.t_now += dt
        self.tacview_global_time += dt
        # a) 飞机：直接调用 update，它内部会运行 JSBSim
        self.aircraft.update(aircraft_action)  # JSBSim的dt在初始化时已设置
        self.aircraft_history.append(self.aircraft.state_vector.copy())
        self.missile_history.append(missile_state_next.copy())
        self.time_history.append(self.t_now)

        # --- 4. 检查高精度引信 ---
        self._check_fuze_condition(dt)
        # if self.done: return  # 如果引信引爆，立即返回

        # --- 2. (新时序) 【同步】更新所有对象的状态 ---
        # self.aircraft.state = aircraft_state_next
        self.missile.state = missile_state_next

        # 3. 【最后】，才更新用于下一轮计算的【历史状态】
        if target_pos_equiv is not None:
            self.prev_theta_L = current_theta_L
            self.prev_phi_L = current_phi_L
            # self.last_valid_theta_dot = theta_L_dot
            # self.last_valid_phi_dot = phi_L_dot

        # --- 5. (核心修正) 在这里检查所有其他的终止条件 ---
        self._check_termination_conditions(dt)  # 传入正确的物理步长 dt

        # --- 6. (可选) Tacview 更新 ---
        if self.tacview_enabled:
            if not self.missile_exploded:
                self.tacview.stream_frame(self.tacview_global_time, self.aircraft, self.missile,
                                          self.flare_manager.flares)

    # # <<< 新增 >>> 辅助方法，用于执行复杂的投放程序
    # def _execute_flare_program(self, salvo_idx, intra_idx, num_groups_idx, inter_idx):
    #     """
    #     将离散动作索引转换为物理参数，并安排投放计划。(追加模式）
    #     """
    #     # 1. 使用 DISCRETE_ACTION_MAP 将索引转换为实际值
    #     program = {
    #         'salvo_size': DISCRETE_ACTION_MAP['salvo_size'][salvo_idx],
    #         'intra_interval': DISCRETE_ACTION_MAP['intra_interval'][intra_idx],
    #         'num_groups': DISCRETE_ACTION_MAP['num_groups'][num_groups_idx],
    #         'inter_interval': DISCRETE_ACTION_MAP['inter_interval'][inter_idx]
    #     }
    #
    #     # 2. 计算这个程序总共需要多少发诱饵弹
    #     total_flares_needed = program['salvo_size'] * program['num_groups']
    #
    #     # 3. 检查是否有足够的诱饵弹
    #     if self.o_ir >= total_flares_needed:
    #         # 4. 如果足够，则更新剩余数量并安排计划
    #         self.o_ir -= total_flares_needed
    #         self.flare_manager.schedule_program(self.t_now, program)
    #         print(f"[{self.t_now:.2f}s] 已安排投放程序: {program}, 剩余诱饵弹: {self.o_ir}")
    #     else:
    #         # 如果不够，则不执行任何操作
    #         # print(f"[{self.t_now:.2f}s] 诱饵弹不足，无法执行投放程序。需要 {total_flares_needed}, 剩余 {self.o_ir}")
    #         pass

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

    def _calculate_los_rate(self, target_pos: np.ndarray, prev_los_angles: tuple, dt: float) -> tuple:
        """
        根据目标位置，计算当前的视线角(LOS)和视线角速率。
        这个方法的逻辑精确地从您原始代码中提取而来。

        Args:
            target_pos (np.ndarray): 导引头锁定的目标位置 [x, y, z]。
            prev_los_angles (tuple): 上一个时间步的视线角 (theta_L, phi_L)。
            dt (float): 时间步长。

        Returns:
            tuple: (theta_L, phi_L, theta_L_dot, phi_L_dot)
                   当前视线角和角速率。
        """
        # --- 1. 解包历史数据 ---
        prev_theta_L, prev_phi_L = prev_los_angles

        # --- 2. 计算当前视线角 ---
        # a) 计算导弹到目标的相对位置矢量
        R_vec = target_pos - self.missile.pos
        R_mag = np.linalg.norm(R_vec)

        # 增加一个极小值，防止距离为0时除零
        if R_mag < 1e-6:
            # 如果距离极近，无法计算有效角度，返回上一次的值
            if prev_theta_L is not None:
                return prev_theta_L, prev_phi_L, 0.0, 0.0
            else:
                return 0.0, 0.0, 0.0, 0.0

        # b) 计算俯仰角 (theta_L) 和 偏航角 (phi_L)
        theta_L = np.arcsin(np.clip(R_vec[1] / R_mag, -1.0, 1.0))
        phi_L = np.arctan2(R_vec[2], R_vec[0])

        # --- 3. 计算视线角速率 ---
        if prev_theta_L is None or prev_phi_L is None:
            # 如果是第一步，没有历史数据，角速率为0
            theta_L_dot = 0.0
            phi_L_dot = 0.0
        else:
            # a) 计算俯仰角速率
            theta_L_dot = (theta_L - prev_theta_L) / dt

            # b) 计算偏航角速率 (使用 atan2 处理角度环绕问题)
            dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
            phi_L_dot = dphi / dt

        return theta_L, phi_L, theta_L_dot, phi_L_dot

    def _check_fuze_condition(self, dt):
        """
        (V6 - 优化版) 检查近距离引信引爆条件。
        优先检查当前时刻是否直接命中，如果没有，再通过求解区间最小值来捕捉“步间命中”。
        """
        # 至少需要两个历史点
        if len(self.missile_history) < 2:
            return

        # --- 1. 获取运动矢量和当前距离 ---
        T2 = self.aircraft_history[-1][:3]  # 当前飞机位置
        T1 = self.aircraft_history[-2][:3]  # 上一步飞机位置
        M2 = self.missile_history[-1][3:6]  # 当前导弹位置
        M1 = self.missile_history[-2][3:6]  # 上一步导弹位置

        R_rel_now = np.linalg.norm(T2 - M2)

        # --- 2. (新增) 最高优先级检查：当前时刻是否直接命中 ---
        if R_rel_now < self.R_kill:
            print(f">>> 引信引爆！当前时刻直接命中，距离 = {R_rel_now:.2f} m")
            self.done = True
            self.success = False
            self.miss_distance = R_rel_now
            self.missile_exploded = True

            # 处理 Tacview 爆炸事件
            if self.tacview_enabled and self.tacview.is_connected and not self.tacview.tacview_final_frame_sent:
                self.tacview.stream_explosion(
                    t_explosion=self.tacview_global_time,
                    aircraft_pos=T2,
                    missile_pos=M2
                )
                self.tacview.tacview_final_frame_sent = True

            # 命中后，直接返回，不再执行后续计算
            return

        # --- 3. 如果未直接命中，再进行插值判断的准备 ---
        R_rel_prev = np.linalg.norm(T1 - M1)

        # 效率优化：如果两端点距离都还很远，可以安全地跳过复杂的插值计算
        if min(R_rel_prev, R_rel_now) > 100:  # 100m 是一个安全的启动阈值
            return

        # --- 4. 求解区间内的真实最小距离 (逻辑与V5完全相同) ---
        # 令 A = T1 - M1 (初始相对位置矢量)A：代表在时间步开始时，从导弹指向飞机的矢量。它的长度就是 R_rel_prev。
        # 令 B = (T2 - T1) - (M2 - M1) (这个时间步内的相对速度矢量)B：代表在这个时间步 dt 内，飞机相对于导弹移动的矢量。
        A = T1 - M1
        B = (T2 - T1) - (M2 - M1)
        # 距离平方 D^2(tau) = ||A + tau*B||^2 = (A+tauB)·(A+tauB)
        # D^2(tau) = (B·B)tau^2 + 2(A·B)tau + (A·A)
        # 这里，我们引入了一个叫tau(τ)的变量，它就像一个进度条，从0（时间步开始）变化到1（时间步结束）。 A + tau * B 就是在进度条为tau时，从导弹指向飞机的相对位置矢量。
        # 通过点积运算，我们把距离的平方 D²(tau) 展开成了一个标准的一元二次方程y = ax² + bx + c的形式：a = B_dot_B  b = 2 * A_dot_B c = A_dot_A
        A_dot_A = np.dot(A, A)
        A_dot_B = np.dot(A, B)
        B_dot_B = np.dot(B, B)
        # 抛物线的顶点在 tau_min = - (A·B) / (B·B) 对于抛物线 y = ax² + bx + c，它的顶点（最低点或最高点）在 x = -b / (2a)。
        # 代入我们的变量，顶点就在 tau = - (2 * A_dot_B) / (2 * B_dot_B)，化简后就是 -A_dot_B / B_dot_B。
        # 这个 tau_min 就是数学上让距离达到最小的那个“理想时刻”。
        if B_dot_B < 1e-6:
            tau_min = 0
        else:
            tau_min = -A_dot_B / B_dot_B
        # 确定在 [0, 1] 区间内，取最小值的实际tau值  这是最关键的一步！我们计算出的理想时刻 tau_min 可能不在我们的时间步内（比如算出来是-0.5，代表最近点其实在上一帧之前就发生了）。
        # 所以我们需要判断：
        # 如果 tau_min 小于0，说明在我们的 [0, 1] 区间内，抛物线是单调递增的，所以最小值在起点 tau=0。
        # 如果 tau_min 大于1，说明在我们的 [0, 1] 区间内，抛物线是单调递减的，所以最小值在终点 tau=1。
        # 如果 tau_min 正好在0和1之间，太棒了！最小值就在抛物线的顶点，即 tau = tau_min。
        # tau_check 就是我们最终确定的、在我们关心的时间段内，能让距离最小的那个时刻（进度条位置）。
        if tau_min <= 0:
            tau_check = 0.0 # 最小值在区间起点
        elif tau_min >= 1:
            tau_check = 1.0 # 最小值在区间终点
        else:
            tau_check = tau_min # 最小值在区间内部
        # 计算该点的真实距离  最后，我们把这个最优点 tau_check 代入我们的相对位置公式 A + tau*B，就得到了那个时刻的相对位置矢量。
        # 取它的长度（范数），就得到了我们梦寐以求的**“区间内的真实最小距离”**！
        min_dist_vec_in_interval = A + tau_check * B
        min_dist_in_interval = np.linalg.norm(min_dist_vec_in_interval)

        # --- 5. 检查“步间命中” ---
        if min_dist_in_interval < self.R_kill:
            print(
                f">>> 引信引爆！区间内最小脱靶量 = {min_dist_in_interval:.2f} m (在 {R_rel_prev:.2f}m 和 {R_rel_now:.2f}m 之间)")
            self.done = True
            self.success = False
            self.miss_distance = min_dist_in_interval
            self.missile_exploded = True

            # 处理 Tacview 爆炸事件
            if self.tacview_enabled and self.tacview.is_connected and not self.tacview.tacview_final_frame_sent:
                t1_global = self.tacview_global_time - dt
                explosion_time_global = t1_global + tau_check * dt
                aircraft_pos_at_explosion = T1 + tau_check * (T2 - T1)
                missile_pos_at_explosion = M1 + tau_check * (M2 - M1)
                self.tacview.stream_explosion(
                    t_explosion=explosion_time_global,
                    aircraft_pos=aircraft_pos_at_explosion,
                    missile_pos=missile_pos_at_explosion
                )
                self.tacview.tacview_final_frame_sent = True

    # def _check_fuze_condition(self, dt):
    #     """
    #     (V5 - 数学完备版) 检查近距离引信引爆条件。
    #     本方法通过求解两个时间步之间距离函数的精确最小值来判断引爆，
    #     能够正确处理所有情况，包括在距离仍在减小时，其中间过程已满足引爆条件。
    #     """
    #     # 至少需要两个历史点
    #     if len(self.missile_history) < 2:
    #         return
    #
    #     # --- 1. 获取运动矢量 ---
    #     T2 = self.aircraft_history[-1][:3]  # 当前飞机位置
    #     T1 = self.aircraft_history[-2][:3]  # 上一步飞机位置
    #     M2 = self.missile_history[-1][3:6]  # 当前导弹位置
    #     M1 = self.missile_history[-2][3:6]  # 上一步导弹位置
    #
    #     R_rel_now = np.linalg.norm(T2 - M2)
    #     R_rel_prev = np.linalg.norm(T1 - M1)
    #
    #     # --- 2. 效率优化：只有在足够近时才进行精确计算 ---
    #     # 如果两端点距离都远大于杀伤半径，可以安全地跳过计算
    #     if min(R_rel_prev, R_rel_now) > 100:  # 100m 是一个安全的启动阈值
    #         return
    #
    #     # --- 3. 求解区间内的真实最小距离 ---
    #     # 设 t 从 t1 到 t_now，用插值系数 tau 从 0 到 1 表示
    #     # 相对位置矢量 R(tau) = (T1-M1) + tau * ((T2-T1)-(M2-M1))
    #     # 令 A = T1-M1 (初始相对位置)
    #     # 令 B = (T2-T1)-(M2-M1) (相对速度 * dt)
    #     A = T1 - M1
    #     B = (T2 - T1) - (M2 - M1)
    #
    #     # 距离平方 D^2(tau) = ||A + tau*B||^2 = (A+tauB)·(A+tauB)
    #     # D^2(tau) = (B·B)tau^2 + 2(A·B)tau + (A·A)
    #     # 这是一个关于tau的二次函数，我们要找它在 [0, 1] 上的最小值
    #     A_dot_A = np.dot(A, A)
    #     A_dot_B = np.dot(A, B)
    #     B_dot_B = np.dot(B, B)
    #
    #     # 抛物线的顶点在 tau_min = - (A·B) / (B·B)
    #     if B_dot_B < 1e-6:  # 避免除以零（相对速度为零）
    #         tau_min = 0  # 距离不变，最小值在任意点
    #     else:
    #         tau_min = -A_dot_B / B_dot_B
    #
    #     # 确定在 [0, 1] 区间内，取最小值的实际tau值
    #     if tau_min <= 0:
    #         tau_check = 0.0  # 最小值在区间起点
    #     elif tau_min >= 1:
    #         tau_check = 1.0  # 最小值在区间终点
    #     else:
    #         tau_check = tau_min  # 最小值在区间内部
    #
    #     # 计算该点的真实距离
    #     min_dist_vec_in_interval = A + tau_check * B
    #     min_dist_in_interval = np.linalg.norm(min_dist_vec_in_interval)
    #
    #     # --- 4. 检查引信是否触发 ---
    #     if min_dist_in_interval < self.R_kill:
    #         print(
    #             f">>> 引信引爆！区间内最小脱靶量 = {min_dist_in_interval:.2f} m (在 {R_rel_prev:.2f}m 和 {R_rel_now:.2f}m 之间)")
    #         self.done = True
    #         self.success = False
    #         self.miss_distance = min_dist_in_interval
    #         self.missile_exploded = True
    #
    #         # --- 5. 处理 Tacview 爆炸事件 ---
    #         if self.tacview_enabled and self.tacview.is_connected and not self.tacview.tacview_final_frame_sent:
    #             # tau_check 就是用于插值的精确系数
    #             t1_global = self.tacview_global_time - dt
    #             explosion_time_global = t1_global + tau_check * dt
    #
    #             aircraft_pos_at_explosion = T1 + tau_check * (T2 - T1)
    #             missile_pos_at_explosion = M1 + tau_check * (M2 - M1)
    #
    #             self.tacview.stream_explosion(
    #                 t_explosion=explosion_time_global,
    #                 aircraft_pos=aircraft_pos_at_explosion,
    #                 missile_pos=missile_pos_at_explosion
    #             )
    #             self.tacview.tacview_final_frame_sent = True

    # def _check_fuze_condition(self, dt):
    #     """
    #    检查近距离引信引爆条件，并在触发时处理包括 Tacview 在内的所有相关事件。
    #    """
    #     # 如果历史记录太短，无法比较
    #     if len(self.aircraft_history) < 2:
    #         return
    #
    #     # 获取当前和上一步的状态
    #     T2 = self.aircraft_history[-1][:3]  # 当前飞机位置
    #     T1 = self.aircraft_history[-2][:3]  # 上一步飞机位置
    #     M2 = self.missile_history[-1][3:6]  # 当前导弹位置
    #     M1 = self.missile_history[-2][3:6]  # 上一步导弹位置
    #
    #     R_rel_now = np.linalg.norm(T2 - M2)
    #
    #     # 只有在非常近的距离才进行精确的插值计算
    #     if R_rel_now < 500:
    #         M1_M2 = M2 - M1
    #         T1_T2 = T2 - T1
    #         M1_T1 = T1 - M1
    #
    #         denominator = np.linalg.norm(T1_T2 - M1_M2) ** 2
    #         if denominator < 1e-6:
    #             # 相对速度为零，t_star未定义或不稳定
    #             interpolated_miss_distance = np.linalg.norm(M1_T1)
    #             t_star = self.t_now - dt  # 假设发生在步长开始时
    #         else:
    #             t1 = self.t_now - dt
    #             t_star = t1 - (np.dot(M1_T1, T1_T2 - M1_M2) * dt) / denominator
    #
    #             print(f"--- 引信检查 --- t*: {t_star:.4f}, 当前时间: {self.t_now:.4f}")
    #
    #             # 确保插值时刻在有效范围内
    #             if t1 <= t_star <= self.t_now:
    #                 interpolated_miss_distance = np.linalg.norm(M1_T1 + (t_star - t1) * (T1_T2 - M1_M2) / dt)
    #                 print(f"插值脱靶量计算: {interpolated_miss_distance:.2f} m")
    #                 # else:
    #                 #     interpolated_miss_distance = R_rel_now
    #                 #     t_star = self.t_now  # 如果不在区间内，则认为最近点就是当前时刻
    #
    #                 # --- 2. 检查引信是否触发 ---
    #                 if interpolated_miss_distance < self.R_kill:
    #                     print(f">>> 引信引爆！插值脱靶量 = {interpolated_miss_distance:.2f} m")
    #                     self.done = True
    #                     self.success = False
    #                     self.miss_distance = interpolated_miss_distance  # 保存精确的脱靶量
    #                     self.missile_exploded = True
    #
    #                     # --- (中文) 核心补全：处理 Tacview 爆炸事件 ---
    #                     if self.tacview_enabled and self.tacview.is_connected:
    #                         # a) 计算爆炸发生的精确时刻 (相对于整个仿真的时间)
    #                         # (中文) 计算精确的全局爆炸时间
    #                         t1_global = self.tacview_global_time - dt
    #                         explosion_time_global = t1_global + (t_star - (self.t_now - dt))
    #
    #                         # b) 计算爆炸时刻，飞机和导弹的插值位置
    #                         tau = (t_star - (self.t_now - dt)) / dt if dt != 0 else 0.0
    #                         tau = np.clip(tau, 0.0, 1.0)  # 确保 tau 在 [0, 1] 之间
    #
    #                         aircraft_pos_at_explosion = T1 + tau * T1_T2
    #                         missile_pos_at_explosion = M1 + tau * M1_M2
    #
    #                         # c) 调用 Tacview 接口发送爆炸帧
    #                         self.tacview.stream_explosion(
    #                             t_explosion=explosion_time_global,
    #                             aircraft_pos=aircraft_pos_at_explosion,
    #                             missile_pos=missile_pos_at_explosion
    #                         )
    #
    #                         # d) (重要补全) 手动设置 Tacview 接口内部的标志位，阻止后续帧发送
    #                         self.tacview.tacview_final_frame_sent = True

    # --- (中文) 私有辅助方法 ---

    def _get_observation(self) -> np.ndarray:
        """
        <<< 核心修改: 去除clip，允许归一化值超出[-1, 1]，距离模糊化 >>>
        组装观测向量。
        - 距离被量化为公里级整数，然后归一化，模拟模糊感知。
        - 允许归一化后的值超出[-1, 1]，让网络感知超调量。
        - (注意) np.arcsin内部的clip被保留，以防止数学错误。
        """
        # --- 1. 计算导弹相关的观测值 ---
        R_vec = self.aircraft.pos - self.missile.pos
        R_rel = np.linalg.norm(R_vec)

        o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, self.missile.state_vector)
        Ry_rel = self.missile.pos[1] - self.aircraft.pos[1]

        # 为了数值稳定性，此处的clip必须保留，防止arcsin的输入超出定义域
        o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))

        # --- 归一化 ---
        # --- 导弹观测 ---

        # 1. 距离模糊化与归一化 (无 clip)
        quantized_distance_km = int(R_rel / 1000.0)
        max_quantized_dist = 10.0
        # 如果实际距离超过10km，o_dis_norm 将会大于 1.0
        o_dis_norm = 2 * (quantized_distance_km / max_quantized_dist) - 1.0

        # 2. 角度使用 sin/cos 表示，天然在 [-1, 1]
        o_beta_sin = np.sin(o_beta_rad)
        o_beta_cos = np.cos(o_beta_rad)

        # 3. 相对俯仰角归一化到 [-1, 1]
        o_theta_L_rel_norm = o_theta_L_rel_rad / (np.pi / 2)

        missile_obs = [o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_rel_norm]

        # --- 2. 获取飞机自身状态 ---
        aircraft_pitch_rad = self.aircraft.state_vector[4]  # 俯仰角 theta
        aircraft_bank_rad = self.aircraft.state_vector[6]  # 滚转角 phi

        # --- 飞机观测 ---
        # 1. 速度归一化 (无 clip)
        # 如果速度超出[150, 400]范围，归一化值会超出[-1, 1]
        o_av_norm = 2 * ((self.aircraft.velocity - 150) / 250) - 1

        # 2. 高度归一化 (无 clip)
        # 如果高度超出[500, 15000]范围，归一化值会超出[-1, 1]
        o_h_norm = 2 * ((self.aircraft.pos[1] - 500) / 14500) - 1

        # 3. 飞机俯仰角归一化
        o_ae_norm = aircraft_pitch_rad / (np.pi / 2)

        # 4. 滚转角使用 sin/cos 表示
        o_am_sin = np.sin(aircraft_bank_rad)
        o_am_cos = np.cos(aircraft_bank_rad)

        # 5. 剩余诱饵弹数量归一化
        o_ir_norm = 2 * (self.o_ir / self.N_infrared) - 1.0

        # 6. 滚转速率归一化
        o_q_norm = self.aircraft.roll_rate_rad_s / (4.0 * np.pi / 3.0)

        aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_sin, o_am_cos, o_ir_norm, o_q_norm]

        # --- 拼接 ---
        observation = np.array(missile_obs + aircraft_obs)
        return observation.astype(np.float32)

    # def _get_observation(self) -> np.ndarray:
    #     """
    #     <<< 核心修改 (V3 - sin/cos 角度表示法) >>>
    #     根据当前状态组装并归一化观测向量。
    #     """
    #     # --- 1. 计算导弹相关的观测值 ---
    #     R_vec = self.aircraft.pos - self.missile.pos
    #     R_rel = np.linalg.norm(R_vec)
    #
    #     # 计算原始弧度角
    #     o_beta_rad = self._compute_relative_beta2(self.aircraft.state_vector, self.missile.state_vector)
    #     Ry_rel = self.missile.pos[1] - self.aircraft.pos[1]
    #     o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))
    #
    #     # --- 核心逻辑修改：从线性归一化到 sin/cos 编码 ---
    #     # 您当前的代码 (Before)
    #     o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10
    #     # o_beta_norm = o_beta_rad / (2 * np.pi)
    #     # o_theta_L_rel_norm = (o_theta_L_rel_rad - (-np.pi / 2)) / np.pi
    #     # missile_obs = [o_dis_norm, o_beta_norm, o_theta_L_rel_norm]
    #
    #     # 修改后的代码 (After)
    #     # o_dis_norm = np.clip(R_rel / 30000.0, 0, 1)  # 距离使用更平滑的归一化
    #     o_beta_sin = np.sin(o_beta_rad)
    #     o_beta_cos = np.cos(o_beta_rad)
    #     o_theta_L_sin = np.sin(o_theta_L_rel_rad)
    #     o_theta_L_cos = np.cos(o_theta_L_rel_rad)
    #     missile_obs = [o_dis_norm, o_beta_sin, o_beta_cos, o_theta_L_sin, o_theta_L_cos]
    #
    #     # --- 2. 获取飞机自身状态 (也需要修改) ---
    #     aircraft_pitch_rad = self.aircraft.state_vector[4]  # 俯仰角 theta
    #     aircraft_bank_rad = self.aircraft.state_vector[6]  # 滚转角 phi
    #
    #     # 您当前的代码 (Before)
    #     # o_av_norm = (self.aircraft.velocity - 100) / 300
    #     # o_h_norm = (self.aircraft.pos[1] - 1000) / 14000
    #     # o_ae_norm = (self.aircraft.state_vector[4] - (-np.pi / 2)) / np.pi
    #     # o_am_norm = (self.aircraft.state_vector[6] - (-np.pi)) / (2 * np.pi)
    #     # o_q_norm = (self.aircraft.roll_rate_rad_s - (-4.0 * np.pi / 3.0)) / (8.0 * np.pi / 3.0)
    #     # o_ir_norm = self.o_ir / self.N_infrared
    #     # aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_norm, o_ir_norm, o_q_norm]
    #
    #     # 修改后的代码 (After)
    #     o_av_norm = (self.aircraft.velocity - 100) / 300
    #     o_h_norm = (self.aircraft.pos[1] - 1000) / 14000
    #     o_ae_sin = np.sin(aircraft_pitch_rad)  # 俯仰角
    #     o_ae_cos = np.cos(aircraft_pitch_rad)
    #     o_am_sin = np.sin(aircraft_bank_rad)  # 滚转角
    #     o_am_cos = np.cos(aircraft_bank_rad)
    #     o_ir_norm = self.o_ir / self.N_infrared
    #     # 滚转速率 o_q 不是角度，是角速度，保持线性归一化
    #     o_q_norm = (self.aircraft.roll_rate_rad_s + 4.0 * np.pi / 3.0) / (8.0 * np.pi / 3.0)
    #     aircraft_obs = [o_av_norm, o_h_norm, o_ae_sin, o_ae_cos, o_am_sin, o_am_cos, o_ir_norm, o_q_norm]
    #
    #     # --- 3. 拼接成最终的观测向量 ---
    #     observation = np.array(missile_obs + aircraft_obs)
    #     return observation.astype(np.float32)

    # 附注：您环境中的 _get_observation 中有一个 compute_relative_beta2 的内部定义。
    # 为了保持代码整洁，我已将其替换为 self._compute_relative_beta2 的调用，
    # 假设 _compute_relative_beta2 已经作为类的方法存在（就像多导弹版本中那样）。
    # 如果没有，请将 compute_relative_beta2 改为 self._compute_relative_beta2 并作为类的方法。
    def _compute_relative_beta2(self, x_target, x_missile):
        # 这个方法的实现保持不变
        psi_t = x_target[5]
        cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)
        R_vec_beta = x_missile[3:6] - x_target[0:3]
        R_proj_world = np.array([R_vec_beta[0], R_vec_beta[2]])
        x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
        z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
        threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
        return threat_angle_rad + 2 * np.pi if threat_angle_rad < 0 else threat_angle_rad

    # def _get_observation(self) -> np.ndarray:
    #     """
    #     根据当前状态组装并归一化观测向量。
    #     所有逻辑精确地从您最新的 AirCombatEnv 文件中提取。
    #     """
    #     # --- 1. 计算基础相对几何关系 ---
    #
    #     # a) 计算相对距离 R_rel
    #     R_vec = self.aircraft.pos - self.missile.pos
    #     R_rel = np.linalg.norm(R_vec)
    #
    #     # b) 计算相对方位角 (beta, 水平面)
    #     #    这个函数现在应该在 RewardCalculator 或一个独立的 kinematics 模块中
    #     #    为了自洽，我们暂时在这里重新定义它，但最好的做法是放在外面
    #     def compute_relative_beta2(x_target, x_missile):
    #         """
    #         (v3 - 修正版) 计算导弹相对于飞机机头的方位角 (0-2pi)。
    #         该版本使用坐标系旋转和 arctan2，比旧的 arccos+cross_product 方法更健壮。
    #         """
    #         # 1. 获取飞机的偏航角 (从北向东为正)
    #         psi_t = x_target[5]
    #         # 2. 计算从世界坐标系 (NUE) 旋转到以飞机机头为前方的参考系所需要的旋转矩阵
    #         # 这是一个绕 y 轴 (天轴) 的二维旋转
    #         cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)
    #         # 旋转矩阵
    #         # R = [[cos, -sin],
    #         #      [sin,  cos]]
    #         # 我们只关心水平面 xz (北-东)
    #         # 3. 计算飞机指向导弹的相对位置矢量，并投影到水平面
    #         R_vec_beta = x_missile[3:6] - x_target[0:3]  # 水平面上的 (x, z) 分量
    #         R_proj_world = np.array([R_vec_beta[0], R_vec_beta[2]])
    #         # 4. 将这个相对位置矢量旋转到以飞机为参考的坐标系下
    #         # [x_rel_body] = [cos, -sin] * [x_rel_world]
    #         # [z_rel_body]   [sin,  cos]   [z_rel_world]
    #         x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
    #         z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
    #         # 5. 使用 arctan2 直接计算角度
    #         # 在这个新的参考系下：
    #         # x_rel_body > 0 表示在前半球, < 0 表示在后半球
    #         # z_rel_body > 0 表示在右半侧(东), < 0 表示在左半侧(西)
    #         # arctan2(z, x) 会给出从 x 轴正方向 (机头) 到该向量的角度
    #         threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
    #         return threat_angle_rad + 2 * np.pi if threat_angle_rad < 0 else threat_angle_rad
    #
    #     o_beta_rad = compute_relative_beta2(self.aircraft.state_vector, self.missile.state_vector)
    #
    #     # c) 计算相对俯仰角 (theta_L_rel, 垂直面)
    #     #    导弹相对于飞机的俯仰角 (飞机坐标系下)
    #     #    Ry_rel < 0 表示导弹在飞机下方
    #     Ry_rel = self.missile.pos[1] - self.aircraft.pos[1]
    #     o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))
    #
    #     # --- 2. 获取飞机自身状态 ---
    #     o_av = self.aircraft.velocity  # 飞机速度
    #     o_h = self.aircraft.pos[1]  # 飞机高度
    #     o_ae_rad = self.aircraft.state_vector[4]  # 飞机俯仰角 (theta)
    #     o_am_rad = self.aircraft.state_vector[6]  # 飞机滚转角 (phi)
    #     o_q_rad_s = self.aircraft.roll_rate_rad_s  # 飞机滚转角速度 (p_real)
    #
    #     # o_ir (剩余诱饵弹数量) 由环境直接管理
    #
    #     # --- 3. 对所有观测值进行归一化 ---
    #
    #     # a) 距离归一化
    #     o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10
    #
    #     # b) 角度归一化
    #     o_beta_norm = o_beta_rad / (2 * np.pi)
    #     o_theta_L_rel_norm = (o_theta_L_rel_rad - (-np.pi / 2)) / np.pi
    #
    #     o_di_norm = np.array([o_beta_norm, o_theta_L_rel_norm])
    #
    #     # c) 飞机状态归一化
    #     o_av_norm = (o_av - 100) / 300
    #     o_h_norm = (o_h - 1000) / 14000
    #     o_ae_norm = (o_ae_rad - (-np.pi / 2)) / np.pi
    #     o_am_norm = (o_am_rad - (-np.pi)) / (2 * np.pi)
    #
    #     # (中文) 滚转角速度归一化 (与您的代码一致)
    #     # 范围是 [-240, 240] deg/s -> [-4pi/3, 4pi/3] rad/s
    #     o_q_norm = (o_q_rad_s - (-4.0 * np.pi / 3.0)) / (8.0 * np.pi / 3.0)
    #
    #     # d) 诱饵弹数量归一化
    #     o_ir_norm = self.o_ir / self.N_infrared
    #
    #     # --- 4. 拼接成最终的观测向量 ---
    #     observation = np.concatenate((
    #         np.array([o_dis_norm]),
    #         o_di_norm,
    #         np.array([o_av_norm]),
    #         np.array([o_h_norm]),
    #         np.array([o_ae_norm]),
    #         np.array([o_am_norm]),
    #         np.array([o_ir_norm]),
    #         np.array([o_q_norm])
    #     ))
    #
    #     return observation.astype(np.float32)

    def _check_termination_conditions(self, dt):
        """
        检查所有可能导致回合结束的条件。
        所有逻辑精确地从您最新的 AirCombatEnv 文件中整合而来。
        """
        # 如果在本轮 step 的其他地方（如引信引爆）已经判定结束，则直接返回
        if self.done:
            return

        # --- 1. 获取判断所需的所有状态 ---
        aircraft_velocity = self.aircraft.velocity
        missile_velocity = self.missile.velocity
        current_R_rel = np.linalg.norm(self.aircraft.pos - self.missile.pos)

        # 计算距离变化率
        if self.prev_R_rel is None:
            range_rate = 0.0
        else:
            range_rate = (current_R_rel - self.prev_R_rel) / dt

        # 检查导弹锁定状态 (需要一个独立的 check_seeker_lock 函数)
        # 为了自洽，我们暂时在这里实现一个简化的版本
        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            self.missile.state_vector, self.aircraft.state_vector,
            self.missile.get_velocity_vector(), self.aircraft.get_velocity_vector(), self.t_now
        )

        # --- 2. 按优先级检查终止条件 ---

        # [优先级最高: 飞机的直接失败条件]
        if self.aircraft.pos[1] <= 500:
            print(f">>> 飞机坠地！仿真终止。")
            self.done, self.success = True, False
            return  # 一旦确定，立即返回

        if self.aircraft.pos[1] >= 15000:
            print(f">>> 飞机超出升限！仿真终止。")
            self.done, self.success = True, False
            return

        if self.aircraft.velocity <= 150:
            print(f">>> 飞机速度过低！仿真终止。")
            self.done, self.success = True, False
            return

        # [优先级次之: 飞机的成功规避条件]

        # a) 判断“物理逃逸” (导弹更慢且正在拉开距离)
        # (中文) 从 __init__ 中获取超参数
        # self.ESCAPE_DURATION_REQ, self.MIN_SEPARATION_RATE_FOR_ESCAPE
        cond_missile_slower = missile_velocity < aircraft_velocity
        cond_separating = range_rate > 5.0  # 使用您代码中的 MIN_SEPARATION_RATE_FOR_ESCAPE
        if cond_missile_slower and cond_separating:
            self.escape_timer += dt
        else:
            self.escape_timer = 0.0

        # b) 判断“信息逃逸” (丢失目标且正在拉开距离)
        # (中文) 从 __init__ 中获取超参数
        # self.lost_and_separating_duration
        cond_lost_lock = not lock_aircraft
        cond_separating_simple = range_rate > 0
        if cond_lost_lock and cond_separating_simple:
            self.lost_and_separating_duration += dt
        else:
            self.lost_and_separating_duration = 0.0

        # c) 检查是否触发成功
        if self.escape_timer >= 1.0:#2.0:  # 使用您代码中的 ESCAPE_DURATION_REQ
            # print(f">>> 成功逃逸(物理)！(导弹更慢且距离拉大已持续 {self.escape_timer:.1f}s)，仿真提前终止!")
            self.done, self.success = True, True
            return

        elif self.lost_and_separating_duration >= 1.0:#2.0:
            # print(f">>> 成功逃逸(信息)！(持续丢失目标且距离拉大已持续 {self.lost_and_separating_duration:.1f}s)，仿真提前终止!")
            self.done, self.success = True, True
            return

        # [优先级最低: 仿真超时]
        if self.t_now >= self.t_end:
            print(f">>> 仿真达到{self.t_end}s, 判定为成功逃离!")
            self.done, self.success = True, True
            return

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

    def _calculate_final_miss_distance(self):
        """
        在回合结束时，通过遍历历史轨迹，统一计算最终脱靶量。
        该逻辑精确复现了您原始代码中的 evaluate_miss 方法。
        """
        # --- 1. 防止重复计算 ---
        if self.miss_distance is not None:
            return

        # --- 2. 处理因操作失误导致的直接失败 ---
        if not self.success and (self.aircraft.pos[1] <= 500 or
                                 self.aircraft.pos[1] >= 15000 or
                                 self.aircraft.velocity <= 150):
            self.miss_distance = 0.0  # 操作失误，视为最差结果（直接命中）
            # (可选) 可以为调试设置 idx_min
            if hasattr(self, 'aircraft_history') and len(self.aircraft_history) > 0:
                self.idx_min = len(self.aircraft_history) - 1
            return

        # --- 3. 正常计算历史最小距离 ---

        # a) 检查历史轨迹是否为空
        if not self.aircraft_history or not self.missile_history:
            # 如果没有历史记录（例如第一步就结束），则使用当前距离
            self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)
            self.idx_min = 0
            return

        # b) 将历史列表转换为高效的 Numpy 数组
        Xt = np.array(self.aircraft_history)
        Yt = np.array(self.missile_history)

        # c) 计算每个时间点的相对距离
        #    使用 Numpy 的矢量化操作，比 for 循环快得多
        delta_pos = Xt[:, :3] - Yt[:, 3:6]
        R_all = np.linalg.norm(delta_pos, axis=1)

        # d) 找到最小值及其索引
        if len(R_all) > 0:
            self.miss_distance = np.min(R_all)
            self.idx_min = np.argmin(R_all)
        else:  # 再次检查以防万一
            self.miss_distance = np.linalg.norm(self.aircraft.pos - self.missile.pos)
            self.idx_min = 0

        # (可选) 打印最终结果，与您旧代码一致
        is_hit = self.miss_distance <= self.R_kill
        print(f">>> {'命中' if is_hit else '未命中'}，脱靶量为：{self.miss_distance:.2f} m")

    def _calculate_equivalent_target(self):
        """
        计算导引头看到的等效红外质心。
        所有逻辑精确地从您最新的 AirCombatEnv 文件中提取和整合。
        """
        # --- 1. 初始化 ---
        I_total = 0.0
        # numerator 是计算质心的分子部分：Σ(I_k * pos_k)
        numerator = np.zeros(3)

        # --- 2. 检查飞机是否在导引头视场内 ---
        aircraft_vel_vec = self.aircraft.get_velocity_vector()
        missile_vel_vec = self.missile.get_velocity_vector()

        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            self.missile.state_vector,
            self.aircraft.state_vector,
            missile_vel_vec,
            aircraft_vel_vec,
            self.t_now
        )
        # print(f"飞机是否被锁定：{lock_aircraft}")

        # --- 3. 如果飞机被锁定，计算其贡献 ---
        if lock_aircraft:
            # a) 计算飞机的红外辐射强度
            #    (需要方位角计算，这个函数最好也放在 kinematics.py 中)
            beta_p = self._compute_relative_beta_for_ir(self.aircraft.state_vector, self.missile.state_vector)
            I_p = self._infrared_intensity_model(beta_p)

            # b) 累加到总强度和分子中
            I_total += I_p
            numerator += I_p * self.aircraft.pos

        # --- 4. 遍历所有诱饵弹，检查是否在视场内并计算贡献 ---
        visible_flares = []
        for flare in self.flare_manager.flares:
            # a) 跳过已经烧完的诱饵弹
            if flare.get_intensity(self.t_now) <= 1e-3:
                continue

            # b) 检查诱饵弹是否在导引头视场内
            #    我们需要为诱饵弹创建一个临时的“状态向量”以适配检查函数
            flare_state_dummy = np.concatenate((flare.pos, [0, 0, 0, 0]))  # 姿态等不重要

            lock_flare, _, _, _, _ = self._check_seeker_lock(
                self.missile.state_vector,
                flare_state_dummy,  # 使用dummy state
                missile_vel_vec,
                flare.vel,  # 使用诱饵弹自己的速度矢量
                self.t_now
            )

            if lock_flare:
                # c) 如果被锁定，计算其贡献
                I_k = flare.get_intensity(self.t_now)
                I_total += I_k
                numerator += I_k * flare.pos

        # --- 5. 计算最终的等效质心位置 ---
        if I_total > 1e-6:
            # 如果总红外强度大于0，计算质心
            target_pos_equiv = numerator / I_total
            return target_pos_equiv
        else:
            # 如果导引头什么都看不见 (目标和诱饵弹都不在视场内)
            # 导弹会继续沿当前方向飞行。我们可以返回一个“无限远”的目标，
            # 或者更简单地，返回飞机的位置，让导引律自然失效。
            # 为了稳定，我们返回飞机位置。
            # (中文) 返回 None，作为一个清晰的“失锁”信号
            return None

    def render(self, view_init_elev=20, view_init_azim=-150):
        """
        (v2 - 修正视觉样式) 使用 matplotlib 进行三维可视化。
        - 诱饵弹轨迹改为实线。
        - 最近点标签和大小按要求修改。
        """
        # --- 1. 检查是否有数据可供渲染 ---
        if not self.aircraft_history or not self.missile_history:
            print("警告：历史轨迹数据为空，无法进行可视化。")
            return

        # --- 2. 将历史列表转换为高效的 Numpy 数组 ---
        aircraft_traj = np.array(self.aircraft_history)[:, :3]
        missile_traj = np.array(self.missile_history)[:, 3:6]

        # --- 3. 创建 3D 绘图窗口 ---
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')

        # --- 4. 绘制轨迹 ---
        ax.plot(missile_traj[:, 2], missile_traj[:, 0], missile_traj[:, 1], 'b-', label='导弹轨迹')
        ax.plot(aircraft_traj[:, 2], aircraft_traj[:, 0], aircraft_traj[:, 1], 'r--', label='目标轨迹')

        # --- 5. 绘制特殊点 ---
        ax.scatter(missile_traj[0, 2], missile_traj[0, 0], missile_traj[0, 1], color='g', s=50, label='导弹起点')
        ax.scatter(aircraft_traj[0, 2], aircraft_traj[0, 0], aircraft_traj[0, 1], color='orange', s=50,
                   label='飞机起点')

        # c) (核心修改) 绘制最近点
        if hasattr(self, 'idx_min') and self.idx_min < len(missile_traj):
            ax.scatter(missile_traj[self.idx_min, 2], missile_traj[self.idx_min, 0], missile_traj[self.idx_min, 1],
                       color='m',
                       marker='o',  # 使用圆形标记，与您原始的 scatter 默认值一致
                       s=20,  # (修改) 将大小从 150 减小到 20，使其更小
                       label=f'远点 ({self.miss_distance:.2f} m)')  # (修改) 标签改为“远点”

        # --- 6. (核心修改) 绘制诱饵弹轨迹 ---
        for i, flare in enumerate(self.flare_manager.flares):
            if not flare.history: continue

            flare_traj = np.array(flare.history)[:, :3]
            if flare_traj.shape[0] > 0:
                ax.plot(flare_traj[:, 2], flare_traj[:, 0], flare_traj[:, 1],
                        color='orange',
                        linewidth=1.0,  # (修改) 线宽设为 1.0
                        linestyle='-',  # (修改) 线型改为实线 '-'
                        label='红外诱饵弹' if i == 0 else "")

        # --- 7. 设置图表样式 ---
        ax.set_xlabel('东 (Z) / m')
        ax.set_ylabel('北 (X) / m')
        ax.set_zlabel('天 (Y) / m')

        # (可选) 设置坐标轴范围以保持一致的视图
        # all_points = np.vstack((aircraft_traj, missile_traj))
        # x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        # y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        # z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])
        # x_center, y_center, z_center = np.mean(all_points, axis=0)
        # max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1)  # 避免max_range为0
        #
        # ax.set_xlim(z_center - max_range / 2, z_center + max_range / 2)
        # ax.set_ylim(x_center - max_range / 2, x_center + max_range / 2)
        # ax.set_zlim(y_center - max_range / 2, y_center + max_range / 2)

        ax.legend()
        ax.set_title('空战对抗三维轨迹')
        ax.grid(True)

        # ax.view_init(elev=view_init_elev, azim=view_init_azim)

        plt.show()
