# 文件: environment.py

import numpy as np
import math
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
from Interference_code.PPO_model.PPO_evasion_fuza.Hybrid_PPO_jsbsim import CONTINUOUS_DIM, DISCRETE_DIMS, DISCRETE_ACTION_MAP

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv(gym.Env):
    """
    强化学习的空战环境主类。
    负责协调仿真流程、管理对象、计算奖励和提供RL接口(reset, step)。
    (Gymnasium兼容版) 强化学习的空战环境主类。
    """

    def __init__(self, tacview_enabled=False):
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
                    DISCRETE_DIMS['intra_interval'],  # e.g., 3 options
                    DISCRETE_DIMS['num_groups'],  # e.g., 3 options
                    DISCRETE_DIMS['inter_interval'],  # e.g., 3 options
                ])
            )
        })

        # --- 2. 定义观测空间 (Observation Space) ---
        # 9个观测值，根据 _get_observation() 的输出
        # [o_dis, o_beta, o_theta_L, o_av, o_h, o_ae, o_am, o_ir, o_q]
        # 最好使用真实的归一化范围，但为了简单起见，可以先用一个较宽松的范围
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float64
        )

        # --- 仿真参数 ---
        # self.dt = 0.1  # 这是外部决策步长
        self.t_end = 60.0
        self.R_kill = 12.0

        # --- (中文) 新增：补全缺失的内部循环参数 ---
        # 这些参数决定了在 step 方法内部的物理仿真是如何运行的
        self.dt_normal = 0.02  # 大步长 (当距离远时)
        self.dt_small = 0.02 # 小步长 (当距离近时)
        self.dt_flare = 0.02  # 投放诱饵弹时的步长
        self.R_switch = 500  # (米) 切换大小步长的距离阈值

        # (中文) 从您的主脚本中看到您还引用了 dt_dec，这里也为您补上
        # 它似乎与 self.dt 含义相同，代表决策步长
        self.dt_dec = 0.2

        self.D_max = 30000.0  # 导引头最大搜索范围 (m)
        self.Angle_IR_rad = np.deg2rad(90)  # 导引头最大视场角度 (弧度)
        self.omega_max_rad_s = 12.0  # 导引头最大角速度 (弧度/秒)
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
        self.o_ir = 24  # 初始诱饵弹数量
        self.N_infrared = 24

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
        self.physical_dt = 0.02  # 假设您的 dt_normal/small/flare 都是0.1
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
        aircraft_vel = np.random.uniform(200, 400)
        # aircraft_vel = 400
        # 7.
        missile_vel = np.random.uniform(700, 900)

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
        theta_L = np.arcsin(R_vec[1] / R_mag)
        phi_L = np.arctan2(R_vec[2], R_vec[0])

        initial_missile_state = np.array([
            missile_vel,  # Vel
            theta_L,  # theta
            phi_L,  # psi
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
        intra_interval_idx = discrete_cmds_indices[2]
        num_groups_idx = discrete_cmds_indices[3]
        inter_interval_idx = discrete_cmds_indices[4]

        # --- 2. <<< 新增 >>> 执行投放程序逻辑 ---
        release_flare_program = (trigger_cmd_idx == 1)
        if release_flare_program:
            self._execute_flare_program(salvo_size_idx, intra_interval_idx, num_groups_idx, inter_interval_idx)

        # --- 3. 保存决策步开始前的状态，用于奖励计算 ---
        self.prev_aircraft_state = self.aircraft.state_vector.copy()
        self.prev_missile_state = self.missile.state_vector.copy()
        self.prev_R_rel = np.linalg.norm(self.aircraft.pos - self.missile.pos)

        # --- 4. 物理仿真循环 (逻辑不变) ---
        R_rel_start = self.prev_R_rel
        if R_rel_start < self.R_switch:
            num_steps, step_dt = int(round(self.dt_dec / self.dt_small)), self.dt_small
        elif self.flare_manager.schedule: # 检查是否有待投放的计划
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
        self.flare_manager.update(self.t_now, dt, self.aircraft) # 注意：这里传入的是 t 时刻的 aircraft 对象

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
            theta_L_dot, phi_L_dot = self.last_valid_theta_dot, self.last_valid_phi_dot

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
                self.tacview.stream_frame(self.tacview_global_time, self.aircraft, self.missile, self.flare_manager.flares)

    # <<< 新增 >>> 辅助方法，用于执行复杂的投放程序
    def _execute_flare_program(self, salvo_idx, intra_idx, num_groups_idx, inter_idx):
        """
        将离散动作索引转换为物理参数，并安排投放计划。
        """
        # 1. 使用 DISCRETE_ACTION_MAP 将索引转换为实际值
        program = {
            'salvo_size': DISCRETE_ACTION_MAP['salvo_size'][salvo_idx],
            'intra_interval': DISCRETE_ACTION_MAP['intra_interval'][intra_idx],
            'num_groups': DISCRETE_ACTION_MAP['num_groups'][num_groups_idx],
            'inter_interval': DISCRETE_ACTION_MAP['inter_interval'][inter_idx]
        }

        # 2. 计算这个程序总共需要多少发诱饵弹
        total_flares_needed = program['salvo_size'] * program['num_groups']

        # 3. 检查是否有足够的诱饵弹
        if self.o_ir >= total_flares_needed:
            # 4. 如果足够，则更新剩余数量并安排计划
            self.o_ir -= total_flares_needed
            self.flare_manager.schedule_program(self.t_now, program)
            print(f"[{self.t_now:.2f}s] 已安排投放程序: {program}, 剩余诱饵弹: {self.o_ir}")
        else:
            # 如果不够，则不执行任何操作
            # print(f"[{self.t_now:.2f}s] 诱饵弹不足，无法执行投放程序。需要 {total_flares_needed}, 剩余 {self.o_ir}")
            pass

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
       检查近距离引信引爆条件，并在触发时处理包括 Tacview 在内的所有相关事件。
       """
        # 如果历史记录太短，无法比较
        if len(self.aircraft_history) < 2:
            return

        # 获取当前和上一步的状态
        T2 = self.aircraft_history[-1][:3]  # 当前飞机位置
        T1 = self.aircraft_history[-2][:3]  # 上一步飞机位置
        M2 = self.missile_history[-1][3:6]  # 当前导弹位置
        M1 = self.missile_history[-2][3:6]  # 上一步导弹位置

        R_rel_now = np.linalg.norm(T2 - M2)

        # 只有在非常近的距离才进行精确的插值计算
        if R_rel_now < 500:
            M1_M2 = M2 - M1
            T1_T2 = T2 - T1
            M1_T1 = T1 - M1

            denominator = np.linalg.norm(T1_T2 - M1_M2) ** 2
            if denominator < 1e-6:
                # 相对速度为零，t_star未定义或不稳定
                interpolated_miss_distance = np.linalg.norm(M1_T1)
                t_star = self.t_now - dt  # 假设发生在步长开始时
            else:
                t1 = self.t_now - dt
                t_star = t1 - (np.dot(M1_T1, T1_T2 - M1_M2) * dt) / denominator

                # 确保插值时刻在有效范围内
                if t1 <= t_star <= self.t_now:
                    interpolated_miss_distance = np.linalg.norm(M1_T1 + (t_star - t1) * (T1_T2 - M1_M2) / dt)
                # else:
                #     interpolated_miss_distance = R_rel_now
                #     t_star = self.t_now  # 如果不在区间内，则认为最近点就是当前时刻

                    # --- 2. 检查引信是否触发 ---
                    if interpolated_miss_distance < self.R_kill:
                        print(f">>> 引信引爆！插值脱靶量 = {interpolated_miss_distance:.2f} m")
                        self.done = True
                        self.success = False
                        self.miss_distance = interpolated_miss_distance  # 保存精确的脱靶量
                        self.missile_exploded = True

                        # --- (中文) 核心补全：处理 Tacview 爆炸事件 ---
                        if self.tacview_enabled and self.tacview.is_connected:
                            # a) 计算爆炸发生的精确时刻 (相对于整个仿真的时间)
                            # (中文) 计算精确的全局爆炸时间
                            t1_global = self.tacview_global_time - dt
                            explosion_time_global = t1_global + (t_star - (self.t_now - dt))

                            # b) 计算爆炸时刻，飞机和导弹的插值位置
                            tau = (t_star - (self.t_now - dt)) / dt if dt != 0 else 0.0
                            tau = np.clip(tau, 0.0, 1.0)  # 确保 tau 在 [0, 1] 之间

                            aircraft_pos_at_explosion = T1 + tau * T1_T2
                            missile_pos_at_explosion = M1 + tau * M1_M2

                            # c) 调用 Tacview 接口发送爆炸帧
                            self.tacview.stream_explosion(
                                t_explosion=explosion_time_global,
                                aircraft_pos=aircraft_pos_at_explosion,
                                missile_pos=missile_pos_at_explosion
                            )

                            # d) (重要补全) 手动设置 Tacview 接口内部的标志位，阻止后续帧发送
                            self.tacview.tacview_final_frame_sent = True

    # --- (中文) 私有辅助方法 ---

    def _get_observation(self) -> np.ndarray:
        """
        根据当前状态组装并归一化观测向量。
        所有逻辑精确地从您最新的 AirCombatEnv 文件中提取。
        """
        # --- 1. 计算基础相对几何关系 ---

        # a) 计算相对距离 R_rel
        R_vec = self.aircraft.pos - self.missile.pos
        R_rel = np.linalg.norm(R_vec)

        # b) 计算相对方位角 (beta, 水平面)
        #    这个函数现在应该在 RewardCalculator 或一个独立的 kinematics 模块中
        #    为了自洽，我们暂时在这里重新定义它，但最好的做法是放在外面
        def compute_relative_beta2(x_target, x_missile):
            """
            (v3 - 修正版) 计算导弹相对于飞机机头的方位角 (0-2pi)。
            该版本使用坐标系旋转和 arctan2，比旧的 arccos+cross_product 方法更健壮。
            """
            # 1. 获取飞机的偏航角 (从北向东为正)
            psi_t = x_target[5]
            # 2. 计算从世界坐标系 (NUE) 旋转到以飞机机头为前方的参考系所需要的旋转矩阵
            # 这是一个绕 y 轴 (天轴) 的二维旋转
            cos_psi, sin_psi = np.cos(-psi_t), np.sin(-psi_t)
            # 旋转矩阵
            # R = [[cos, -sin],
            #      [sin,  cos]]
            # 我们只关心水平面 xz (北-东)
            # 3. 计算飞机指向导弹的相对位置矢量，并投影到水平面
            R_vec_beta = x_missile[3:6] - x_target[0:3]# 水平面上的 (x, z) 分量
            R_proj_world = np.array([R_vec_beta[0], R_vec_beta[2]])
            # 4. 将这个相对位置矢量旋转到以飞机为参考的坐标系下
            # [x_rel_body] = [cos, -sin] * [x_rel_world]
            # [z_rel_body]   [sin,  cos]   [z_rel_world]
            x_rel_body = cos_psi * R_proj_world[0] - sin_psi * R_proj_world[1]
            z_rel_body = sin_psi * R_proj_world[0] + cos_psi * R_proj_world[1]
            # 5. 使用 arctan2 直接计算角度
            # 在这个新的参考系下：
            # x_rel_body > 0 表示在前半球, < 0 表示在后半球
            # z_rel_body > 0 表示在右半侧(东), < 0 表示在左半侧(西)
            # arctan2(z, x) 会给出从 x 轴正方向 (机头) 到该向量的角度
            threat_angle_rad = np.arctan2(z_rel_body, x_rel_body)
            return threat_angle_rad + 2 * np.pi if threat_angle_rad < 0 else threat_angle_rad

        o_beta_rad = compute_relative_beta2(self.aircraft.state_vector, self.missile.state_vector)

        # c) 计算相对俯仰角 (theta_L_rel, 垂直面)
        #    导弹相对于飞机的俯仰角 (飞机坐标系下)
        #    Ry_rel < 0 表示导弹在飞机下方
        Ry_rel = self.missile.pos[1] - self.aircraft.pos[1]
        o_theta_L_rel_rad = np.arcsin(np.clip(Ry_rel / (R_rel + 1e-6), -1.0, 1.0))

        # --- 2. 获取飞机自身状态 ---
        o_av = self.aircraft.velocity  # 飞机速度
        o_h = self.aircraft.pos[1]  # 飞机高度
        o_ae_rad = self.aircraft.state_vector[4]  # 飞机俯仰角 (theta)
        o_am_rad = self.aircraft.state_vector[6]  # 飞机滚转角 (phi)
        o_q_rad_s = self.aircraft.roll_rate_rad_s  # 飞机滚转角速度 (p_real)

        # o_ir (剩余诱饵弹数量) 由环境直接管理

        # --- 3. 对所有观测值进行归一化 ---

        # a) 距离归一化
        o_dis_norm = np.clip(int(R_rel / 1000.0), 0, 10) / 10

        # b) 角度归一化
        o_beta_norm = o_beta_rad / (2 * np.pi)
        o_theta_L_rel_norm = (o_theta_L_rel_rad - (-np.pi / 2)) / np.pi

        o_di_norm = np.array([o_beta_norm, o_theta_L_rel_norm])

        # c) 飞机状态归一化
        o_av_norm = (o_av - 100) / 300
        o_h_norm = (o_h - 1000) / 14000
        o_ae_norm = (o_ae_rad - (-np.pi / 2)) / np.pi
        o_am_norm = (o_am_rad - (-np.pi)) / (2 * np.pi)

        # (中文) 滚转角速度归一化 (与您的代码一致)
        # 范围是 [-240, 240] deg/s -> [-4pi/3, 4pi/3] rad/s
        o_q_norm = (o_q_rad_s - (-4.0 * np.pi / 3.0)) / (8.0 * np.pi / 3.0)

        # d) 诱饵弹数量归一化
        o_ir_norm = self.o_ir / self.N_infrared

        # --- 4. 拼接成最终的观测向量 ---
        observation = np.concatenate((
            np.array([o_dis_norm]),
            o_di_norm,
            np.array([o_av_norm]),
            np.array([o_h_norm]),
            np.array([o_ae_norm]),
            np.array([o_am_norm]),
            np.array([o_ir_norm]),
            np.array([o_q_norm])
        ))

        return observation.astype(np.float32)

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
        if self.aircraft.pos[1] <= 100:
            print(f">>> 飞机坠地！仿真终止。")
            self.done, self.success = True, False
            return  # 一旦确定，立即返回

        if self.aircraft.pos[1] >= 15000:
            print(f">>> 飞机超出升限！仿真终止。")
            self.done, self.success = True, False
            return

        if self.aircraft.velocity <= 110:
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
        if self.escape_timer >= 2.0:  # 使用您代码中的 ESCAPE_DURATION_REQ
            print(f">>> 成功逃逸(物理)！(导弹更慢且距离拉大已持续 {self.escape_timer:.1f}s)，仿真提前终止!")
            self.done, self.success = True, True
            return

        elif self.lost_and_separating_duration >= 2.0:
            print(
                f">>> 成功逃逸(信息)！(持续丢失目标且距离拉大已持续 {self.lost_and_separating_duration:.1f}s)，仿真提前终止!")
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
        if not self.success and (self.aircraft.pos[1] <= 100 or
                                 self.aircraft.pos[1] >= 15000 or
                                 self.aircraft.velocity <= 110):
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