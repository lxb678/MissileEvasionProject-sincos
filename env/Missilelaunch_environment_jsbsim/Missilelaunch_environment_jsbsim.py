import numpy as np
import time
import math
import matplotlib
from matplotlib import pyplot as plt

# --- (中文) 从我们创建的模块中导入所有需要的类 ---
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile
from .decoys import FlareManager
from .reward_system import RewardCalculator
from .tacview_interface import TacviewInterface
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv:
    """
    强化学习的空战环境主类。
    负责协调仿真流程、管理对象、计算奖励和提供RL接口(reset, step)。
    """

    def __init__(self, tacview_enabled=False):
        # --- 仿真参数 ---
        self.dt_dec = 0.2  # 决策步长
        self.dt_normal = 0.02  # 物理仿真步长
        self.t_end = 120.0  # 最大仿真时间
        self.R_kill = 12.0  # 导弹引信杀伤半径
        self.max_disengagement_range = 40000.0  # 最大脱离距离
        # --- 弹药库 ---
        self.initial_missiles = 2
        self.initial_flares = 24
        # --- <<< 从规避环境中引入的导引头参数 >>> ---
        self.D_max = 30000.0  # 导引头最大搜索范围 (m)
        self.Angle_IR_rad = np.deg2rad(90)  # 导引头最大视场角度 (弧度)
        self.omega_max_rad_s = 12.0  # 导引头最大角速度 (弧度/秒)
        self.T_max = 60.0  # 导引头最大搜索时间 (秒)
        # --- 核心组件实例化 ---
        # 使用 None 初始化，在 reset 中具体设置
        self.red_aircraft = None
        self.blue_aircraft = None
        self.red_missiles = []
        self.blue_missiles = []
        self.red_flare_manager = FlareManager(flare_per_group=1)
        self.blue_flare_manager = FlareManager(flare_per_group=1)
        # --- 状态变量 ---
        self.t_now = 0.0
        self.episode_count = 0
        self.dones = {"__all__": False}
        self.red_alive = True
        self.blue_alive = True

        # --- 核心组件实例化 ---
        initial_state = np.array([0, 1000, 0, 200, 0.1, 0.0, 0.0, 0.0])
        self.aircraft = Aircraft(dt=self.dt_normal, initial_state=initial_state)
        self.missile = None
        self.flare_manager = FlareManager(flare_per_group=1)
        self.reward_calculator = RewardCalculator()

        # 用于存储历史状态，以计算变化率 (如速度变化、距离变化)
        self.prev_aircraft_state = None
        self.prev_missile_state = None
        self.prev_R_rel = None
        self.prev_theta_L, self.prev_phi_L = None, None

        # 历史记录 (用于可视化和计算)
        self.history = {
            'time': [],
            'red_aircraft': [],
            'blue_aircraft': [],
            'red_missiles': {},  # key: missile_id, value: trajectory
            'blue_missiles': {}
        }
        self.tacview_enabled = tacview_enabled
        self.tacview_global_time = 0.0
        self.tacview = TacviewInterface() if tacview_enabled else None

    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态，为新的一回合做准备。
        """
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

        # --- 重置状态变量 ---
        self.t_now = 0.0
        self.dones = {"__all__": False}
        self.red_alive = True
        self.blue_alive = True

        self.red_missiles.clear()
        self.blue_missiles.clear()
        self.red_flare_manager.reset()
        self.blue_flare_manager.reset()

        for key in self.history:
            if isinstance(self.history[key], list):
                self.history[key].clear()
            else:
                self.history[key] = {}

        # --- 3. 重置所有组件 ---
        self.flare_manager.reset()
        self.reward_calculator.reset()

        # --- 2. 设置初始对抗场景 (例如: 30km 迎头对冲) ---
        altitude = 8000.0
        initial_speed = 250.0  # m/s
        separation = 30000.0

        # 红方初始状态 (向东飞行)
        red_initial_state = np.array([
            0, altitude, -separation / 2,  # pos (x, y, z)
            initial_speed,  # Vel (m/s)
            0.0,  # theta (pitch)
            np.deg2rad(90),  # psi (heading, east)
            0.0,  # phi (roll)
            0.0  # roll rate
        ])
        if self.red_aircraft is None:
            self.red_aircraft = Aircraft(dt=self.dt_normal, initial_state=red_initial_state, name="F-16", color="Red")
        else:
            self.red_aircraft.reset(initial_state=red_initial_state)
            # (可选) 如果 reset 后希望属性保持一致，也可以在这里设置
            self.red_aircraft.name = "F-16"
            self.red_aircraft.color = "Red"

        self.red_aircraft.missile_ammo = self.initial_missiles
        self.red_aircraft.flare_ammo = self.initial_flares

        # 蓝方初始状态 (向西飞行)
        blue_initial_state = np.array([
            0, altitude, separation / 2,  # pos (x, y, z)
            initial_speed,  # Vel (m/s)
            0.0,  # theta (pitch)
            np.deg2rad(-90),  # psi (heading, west)
            0.0,  # phi (roll)
            0.0  # roll rate
        ])
        if self.blue_aircraft is None:
            self.blue_aircraft = Aircraft(dt=self.dt_normal, initial_state=blue_initial_state,name="F-16", color="Blue")
        else:
            self.blue_aircraft.reset(initial_state=blue_initial_state)
            self.blue_aircraft.name = "F-16"
            self.blue_aircraft.color = "Blue"

        self.blue_aircraft.missile_ammo = self.initial_missiles
        self.blue_aircraft.flare_ammo = self.initial_flares

        if self.tacview_enabled:
            self._stream_tacview_frame()

        return self._get_observations()

    def step(self, actions: dict) -> tuple:
        """
        执行一个多智能体决策步。
        :param actions: 字典, e.g., {'red_agent': red_action, 'blue_agent': blue_action}
                        每个 action 是 [油门, 升降舵, 副翼, 方向舵, 投放诱饵, 发射导弹]
        """
        red_action = actions.get('red_agent', [0.5, 0, 0, 0, 0, 0])
        blue_action = actions.get('blue_agent', [0.5, 0, 0, 0, 0, 0])

        num_steps = int(round(self.dt_dec / self.dt_normal))

        # 在决策步开始时计算一次性事件 (发射)
        # 只有在第一步且存活时才能发射
        if self.red_alive and red_action[5] > 0.5:
            self._fire_missile('red')
        if self.blue_alive and blue_action[5] > 0.5:
            self._fire_missile('blue')

        # 物理仿真内循环
        for _ in range(num_steps):
            if self.dones["__all__"]:
                break

            # --- 1. 更新飞机 ---
            if self.red_alive:
                self.red_aircraft.update(red_action[:4])
            if self.blue_alive:
                self.blue_aircraft.update(blue_action[:4])

            # --- 2. 更新诱饵弹 ---
            if self.red_alive and red_action[4] > 0.5 and self.red_aircraft.flare_ammo > 0:
                self.red_flare_manager.release_flare_group(self.t_now)
                self.red_aircraft.flare_ammo -= self.red_flare_manager.flare_per_group
            self.red_flare_manager.update(self.t_now, self.dt_normal, self.red_aircraft)

            if self.blue_alive and blue_action[4] > 0.5 and self.blue_aircraft.flare_ammo > 0:
                self.blue_flare_manager.release_flare_group(self.t_now)
                self.blue_aircraft.flare_ammo -= self.blue_flare_manager.flare_per_group
            self.blue_flare_manager.update(self.t_now, self.dt_normal, self.blue_aircraft)

            # --- 3. 更新导弹 ---
            self._update_all_missiles(self.dt_normal)

            # --- 4. 检查命中与碰撞 ---
            self._check_hits_and_crashes()

            # --- 5. 更新时间和历史 ---
            self.t_now += self.dt_normal
            self.tacview_global_time += self.dt_normal
            self._update_history()

            # --- 6. 检查通用终止条件 ---
            self._check_termination_conditions()

            # --- 7. 更新 Tacview ---
            if self.tacview_enabled:
                self._stream_tacview_frame()

        # --- 8. 计算奖励 ---
        rewards = self._calculate_rewards()

        # --- 9. 获取新观测 ---
        observations = self._get_observations()

        # 在 MARL 中, info 字典也应该是 agent-keyed
        infos = {'red_agent': {}, 'blue_agent': {}}

        return observations, rewards, self.dones, infos

    # ==========================================================================
    # --- 核心逻辑方法 (部分从规避环境中移植和改造) ---
    # ==========================================================================
    def _fire_missile(self, shooter: str):
        """处理导弹发射逻辑"""
        if shooter == 'red':
            shooter_obj, target_obj, missile_list = self.red_aircraft, self.blue_aircraft, self.red_missiles
        elif shooter == 'blue':
            shooter_obj, target_obj, missile_list = self.blue_aircraft, self.red_aircraft, self.blue_missiles
        else:
            return

        if shooter_obj.missile_ammo <= 0: return
        shooter_obj.missile_ammo -= 1

        # 计算导弹初始状态
        # R_vec = target_obj.pos - shooter_obj.pos
        # theta_L = np.arcsin(np.clip(R_vec[1] / np.linalg.norm(R_vec), -1.0, 1.0))
        # phi_L = np.arctan2(R_vec[2], R_vec[0])
        # missile_vel = shooter_obj.velocity
        #
        # initial_missile_state = np.array([
        #     missile_vel, theta_L, phi_L,
        #     shooter_obj.pos[0], shooter_obj.pos[1], shooter_obj.pos[2]
        # ])
        # --- <<< 核心修改：完全继承载机的速度矢量 >>> ---

        # 1. 继承速度大小
        missile_vel = shooter_obj.velocity

        # 2. 继承飞行方向 (俯仰角 theta 和 偏航角 psi)
        #    我们从飞机的状态向量中直接获取
        #    state_vector: [x,y,z, Vt, theta, psi, phi, p_real]
        missile_theta = shooter_obj.state_vector[4]  # 飞机的俯仰角
        missile_psi = shooter_obj.state_vector[5]  # 飞机的偏航角

        # 3. 继承位置
        missile_pos = shooter_obj.pos

        initial_missile_state = np.array([
            missile_vel,
            missile_theta,
            missile_psi,
            *missile_pos
        ])

        # new_missile = Missile(initial_missile_state)
        # new_missile.target = target_obj  # 绑定目标
        # new_missile.id = f"{shooter[:1].upper()}M{len(missile_list) + 1}_{self.episode_count}"
        # new_missile.name = f"{shooter_obj.name}-Missile-{len(missile_list) + 1}"
        # new_missile.color = shooter_obj.color
        # --- 核心修改部分 ---
        new_missile = Missile(initial_missile_state)

        # 绑定目标
        new_missile.target = target_obj

        # 设置Tacview属性
        new_missile.name = "AIM-9X"  # <<< 将名称硬编码为 AIM-9X
        new_missile.color = shooter_obj.color  # 导弹颜色继承自发射方 (Red 或 Blue)

        # 设置唯一ID
        # (确保ID在同一场仿真中是唯一的)
        unique_part = f"{shooter[:1].upper()}M{len(missile_list) + 1}"
        new_missile.id = f"{unique_part}_{self.episode_count}_{time.time()}"
        # --- 结束修改 ---
        missile_list.append(new_missile)
        print(f">>> {shooter.capitalize()} 方发射了一枚导弹！剩余 {shooter_obj.missile_ammo} 枚。")

    def _update_all_missiles(self, dt):
        """更新所有在飞导弹的状态"""
        # 更新红方导弹
        for m in self.red_missiles:
            if not m.is_active: continue
            # 红方导弹的目标是蓝方飞机和其诱饵弹
            target_pos = self._calculate_equivalent_target(m, self.blue_aircraft, self.blue_flare_manager)
            m.update(dt, target_pos)

        # 更新蓝方导弹
        for m in self.blue_missiles:
            if not m.is_active: continue
            # 蓝方导弹的目标是红方飞机和其诱饵弹
            target_pos = self._calculate_equivalent_target(m, self.red_aircraft, self.red_flare_manager)
            m.update(dt, target_pos)

    def _check_hits_and_crashes(self):
        """
        (V4) 检查命中/坠毁。修正了在正常交战中因投放诱饵弹而提前结束的bug。
        """
        # --- 1. 检查导弹命中和飞机坠毁事件 ---
        # (这部分逻辑与 V3 完全相同，负责更新 self.red_alive 和 self.blue_alive 的状态)
        if self.blue_alive:
            for m in self.red_missiles:
                if not m.is_active: continue
                if np.linalg.norm(m.pos - self.blue_aircraft.pos) < self.R_kill:
                    print(f"\n>>> 蓝方飞机被红方导弹 {m.name} 击中！")
                    self.blue_alive = False
                    m.is_active = False
                    if self.tacview_enabled:
                        self.tacview.stream_explosion(self.tacview_global_time, self.blue_aircraft.pos, m.pos)
                    break

        if self.red_alive:
            for m in self.blue_missiles:
                if not m.is_active: continue
                if np.linalg.norm(m.pos - self.red_aircraft.pos) < self.R_kill:
                    print(f"\n>>> 红方飞机被蓝方导弹 {m.name} 击中！")
                    self.red_alive = False
                    m.is_active = False
                    if self.tacview_enabled:
                        self.tacview.stream_explosion(self.tacview_global_time, self.red_aircraft.pos, m.pos)
                    break

        if self.red_alive and (self.red_aircraft.pos[1] < 100 or self.red_aircraft.velocity < 110):
            print("\n>>> 红方飞机坠毁/失速！")
            self.red_alive = False
        if self.blue_alive and (self.blue_aircraft.pos[1] < 100 or self.blue_aircraft.velocity < 110):
            print("\n>>> 蓝方飞机坠毁/失速！")
            self.blue_alive = False

        # --- 2. <<< 核心修改：仅在有伤亡时才判断是否结束回合 >>> ---
        # 只有当 red_alive 或 blue_alive 在上面的代码块中被设为 False 后，才进入这部分逻辑
        # --- 3. <<< 核心修改：V6 结束判断逻辑 >>> ---
        # 首先，我们需要知道当前的物理步长 dt，因为 _check_threat_escaped 需要它
        dt = self.dt_normal  # 假设我们总是用 dt_normal，这是一个合理的简化

        if not self.red_alive or not self.blue_alive:
            if not self.red_alive and not self.blue_alive:
                print(">>> 双方均已损失，仿真结束。")
                self.dones["__all__"] = True
                return

            # a) 红方存活，蓝方被击落
            if self.red_alive and not self.blue_alive:
                # 检查是否所有对红方的威胁都已被规避
                all_threats_to_red_escaped = True
                for m in self.blue_missiles:
                    if m.is_active:
                        # 如果这枚导弹还没有被规避，那么威胁就依然存在
                        if not self._check_threat_escaped(m, self.red_aircraft, dt):
                            all_threats_to_red_escaped = False
                            break  # 只要有一个威胁存在，就没必要再检查了

                if all_threats_to_red_escaped:
                    print(">>> 蓝方已被击落，且所有来袭导弹均已规避，红方胜利！仿真结束。")
                    self.dones["__all__"] = True
                else:
                    print(">>> 蓝方已被击落，但其导弹威胁仍在，红方需继续规避...")

            # b) 蓝方存活，红方被击落
            if self.blue_alive and not self.red_alive:
                all_threats_to_blue_escaped = True
                for m in self.red_missiles:
                    if m.is_active:
                        if not self._check_threat_escaped(m, self.blue_aircraft, dt):
                            all_threats_to_blue_escaped = False
                            break

                if all_threats_to_blue_escaped:
                    print(">>> 红方已被击落，且所有来袭导弹均已规避，蓝方胜利！仿真结束。")
                    self.dones["__all__"] = True
                else:
                    print(">>> 红方已被击落，但其导弹威胁仍在，蓝方需继续规避...")

    def _check_threat_escaped(self, missile: Missile, target_aircraft: Aircraft, dt: float) -> bool:
        """
        (移植自规避环境) 检查单个威胁（导弹）是否已经被目标飞机成功规避。
        这个函数会更新导弹内部的计时器。
        :param missile: 需要检查的导弹对象。
        :param target_aircraft: 导弹的目标飞机对象。
        :param dt: 物理仿真步长。
        :return: 如果该导弹已被规避，则返回 True，否则返回 False。
        """
        # --- 1. 获取判断所需的所有状态 ---
        aircraft_velocity = target_aircraft.velocity
        missile_velocity = missile.velocity
        current_dist = np.linalg.norm(target_aircraft.pos - missile.pos)

        # 计算距离变化率 (range_rate)
        if missile.prev_dist_to_target is None:
            range_rate = 0.0
        else:
            range_rate = (current_dist - missile.prev_dist_to_target) / dt
        missile.prev_dist_to_target = current_dist  # 更新历史距离

        # 检查导弹是否能锁定飞机
        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            missile, target_aircraft,
            missile.get_velocity_vector(), target_aircraft.get_velocity_vector(), self.t_now
        )

        # --- 2. 判断“物理逃逸” ---
        cond_missile_slower = missile_velocity < aircraft_velocity
        cond_separating = range_rate > 5.0  # 正在显著拉开距离
        if cond_missile_slower and cond_separating:
            missile.escape_timer += dt
        else:
            missile.escape_timer = 0.0  # 重置计时器

        if missile.escape_timer >= 2.0:
            print(f">>> 导弹 {missile.name} 已被 {target_aircraft.name} [物理规避]！")
            return True  # 判定为已规避

        # --- 3. 判断“信息逃逸” ---
        cond_lost_lock = not lock_aircraft
        cond_separating_simple = range_rate > 0  # 只要在拉开距离
        if cond_lost_lock and cond_separating_simple:
            missile.lost_and_separating_duration += dt
        else:
            missile.lost_and_separating_duration = 0.0  # 重置计时器

        if missile.lost_and_separating_duration >= 2.0:
            print(f">>> 导弹 {missile.name} 已被 {target_aircraft.name} [信息规避]！")
            return True  # 判定为已规避

        # 如果以上条件都不满足，则认为威胁依然存在
        return False

    def _check_termination_conditions(self):
        """检查通用的回合结束条件"""
        if self.dones["__all__"]: return

        if self.t_now >= self.t_end:
            print(">>> 仿真时间到，回合结束。")
            self.dones["__all__"] = True

        if self.red_alive and self.blue_alive:
            dist = np.linalg.norm(self.red_aircraft.pos - self.blue_aircraft.pos)
            if dist > self.max_disengagement_range:
                print(">>> 双方距离过远，判定脱离接触。")
                self.dones["__all__"] = True

    # ==========================================================================
    # --- <<< 以下是 从 Vec_missile_evasion_environment_jsbsim.py 移植的辅助方法 >>> ---
    # ==========================================================================

    def _calculate_equivalent_target(self, missile, primary_target, flare_manager):
        """
        计算导引头看到的等效红外质心 (移植并改造)
        :param missile: 正在计算的导弹对象
        :param primary_target: 导弹的主要目标 (飞机对象)
        :param flare_manager: 目标方的诱饵弹管理器
        :return: 导引头应该瞄准的位置向量 [x, y, z]，如果失锁则返回 None
        """
        I_total = 0.0
        numerator = np.zeros(3)
        missile_vel_vec = missile.get_velocity_vector()

        # 1. 检查飞机是否在导引头视场内
        lock_aircraft, _, _, _, _ = self._check_seeker_lock(
            missile, primary_target,
            missile_vel_vec, primary_target.get_velocity_vector(), self.t_now
        )


        # 2. 如果飞机被锁定，计算其贡献
        if lock_aircraft:
            beta_p = self._compute_relative_beta_for_ir(primary_target.state_vector, missile.state_vector)
            I_p = self._infrared_intensity_model(beta_p)
            I_total += I_p
            numerator += I_p * primary_target.pos

        # 3. 遍历所有诱饵弹，检查是否在视场内并计算贡献
        for flare in flare_manager.flares:
            if flare.get_intensity(self.t_now) <= 1e-3: continue

            flare_state_dummy = np.concatenate((flare.pos, [0, 0, 0, 0]))
            # <<< 确保传递的是对象实例 missile 和 flare >>>
            # 注意：target_obj 是 flare，它没有 .state_vector，但我们的新版 _check_seeker_lock 能处理
            lock_flare, _, _, _, _ = self._check_seeker_lock(
                missile, flare,
                missile_vel_vec, flare.vel, self.t_now
            )

            if lock_flare:
                I_k = flare.get_intensity(self.t_now)
                I_total += I_k
                numerator += I_k * flare.pos

        # 4. 计算最终的等效质心位置
        if I_total > 1e-6:
            return numerator / I_total
        else:
            return None  # 返回 None 表示失锁

    # def _check_seeker_lock(self, x_missile, x_target, V_missile_vec, V_target_vec, t_now):
    #     """ 纯粹的锁定逻辑计算 (直接移植) """
    #     R_vec = x_target[:3] - x_missile[3:6]
    #     R_mag = np.linalg.norm(R_vec)
    #
    #     if not (R_mag <= self.D_max): return False, False, True, True, True
    #
    #     norm_product = R_mag * np.linalg.norm(V_missile_vec) + 1e-6
    #     cos_Angle = np.dot(R_vec, V_missile_vec) / norm_product
    #     if not (cos_Angle >= np.cos(self.Angle_IR_rad)): return False, True, False, True, True
    #
    #     delta_V = V_missile_vec - V_target_vec
    #     omega_R = np.linalg.norm(np.cross(R_vec, delta_V)) / (R_mag ** 2 + 1e-6)
    #     if not (omega_R <= self.omega_max_rad_s): return False, True, True, False, True
    #
    #     if not (t_now <= self.T_max): return False, True, True, True, False
    #
    #     return True, True, True, True, True
    def _check_seeker_lock(self, missile_obj, target_obj, V_missile_vec, V_target_vec, t_now):
        """
        纯粹的锁定逻辑计算。
        V2版修正了时间判断逻辑，使用导弹自身的飞行时间。
        并且要求输入为对象实例，而非状态向量，以便访问 .flight_time 等属性。
        """

        x_missile = missile_obj.state_vector
        # 目标可能是飞机或诱饵弹的dummy object
        x_target = target_obj.state_vector if hasattr(target_obj, 'state_vector') else np.concatenate(
            (target_obj.pos, [0, 0, 0, 0]))

        R_vec = x_target[:3] - x_missile[3:6]
        R_mag = np.linalg.norm(R_vec)

        # 调试信息
        missile_name = missile_obj.name
        print_debug = False
        if missile_obj.flight_time < 2.1 and missile_obj.flight_time > 0:
            print_debug = True

        # 1. 距离检查
        if not (R_mag <= self.D_max):
            if print_debug: print(f"[{missile_name}] 失锁: 超出最大距离 ({R_mag:.0f}m)")
            return False, False, True, True, True

        # 2. 视场角检查
        norm_product = R_mag * np.linalg.norm(V_missile_vec) + 1e-6
        if norm_product < 1e-6: return False, True, False, True, True
        cos_Angle = np.dot(R_vec, V_missile_vec) / norm_product
        angle_deg = np.rad2deg(np.arccos(np.clip(cos_Angle, -1, 1)))
        fov_deg = np.rad2deg(self.Angle_IR_rad)

        if not (angle_deg <= fov_deg):
            if print_debug: print(
                f"[{missile_name}] 失锁: 超出视场角 (Angle={angle_deg:.1f}° > FOV_HALF={fov_deg:.1f}°)")
            return False, True, False, True, True

        # 3. 视线角速率检查
        delta_V = V_missile_vec - V_target_vec
        omega_R_rad_s = np.linalg.norm(np.cross(R_vec, delta_V)) / (R_mag ** 2 + 1e-6)
        if not (omega_R_rad_s <= self.omega_max_rad_s):
            if print_debug: print(
                f"[{missile_name}] 失锁: 超出角速度 (Omega={omega_R_rad_s:.2f} > Max={self.omega_max_rad_s:.2f} rad/s)")
            return False, True, True, False, True

        # 4. <<< 核心修正：使用导弹自身的飞行时间 >>>
        if not (missile_obj.flight_time <= self.T_max):
            if print_debug: print(
                f"[{missile_name}] 失锁: 超出导引头工作寿命 (FlightTime={missile_obj.flight_time:.1f}s > T_max={self.T_max:.1f}s)")
            return False, True, True, True, False

        # 所有检查都通过
        if print_debug: print(f"[{missile_name}] 锁定成功 (Angle={angle_deg:.1f}°, R={R_mag:.0f}m)")
        return True, True, True, True, True

    def _infrared_intensity_model(self, beta_rad: float) -> float:
        """飞机的红外辐射强度模型 (直接移植)"""
        beta_deg = np.array([0, 40, 90, 140, 180])
        intensity_vals = np.array([3800, 5000, 2500, 2000, 800])
        beta_rad_points = np.deg2rad(beta_deg)
        return max(np.interp(beta_rad, beta_rad_points, intensity_vals), 0.0)

    def _compute_relative_beta_for_ir(self, x_target, x_missile) -> float:
        """计算用于红外模型的无符号角度 [0, pi] (直接移植)"""
        R_vec = x_target[0:3] - x_missile[3:6]
        R_proj = np.array([R_vec[0], 0.0, R_vec[2]])
        if np.linalg.norm(R_proj) < 1e-6: return 0.0

        psi_t = x_target[5]
        V_body = np.array([np.cos(psi_t), 0.0, np.sin(psi_t)])

        cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
        return np.arccos(np.clip(cos_beta, -1.0, 1.0))

    # ==========================================================================
    # --- 以下是标准的多智能体辅助方法 (来自原始 Missilelaunch 文件) ---
    # ==========================================================================

    def _calculate_rewards(self) -> dict:
        """计算每个智能体的奖励"""
        rewards = {'red_agent': 0.0, 'blue_agent': 0.0}
        # 生存奖励
        if self.red_alive: rewards['red_agent'] += 0.1
        if self.blue_alive: rewards['blue_agent'] += 0.1
        # 结束时的稀疏奖励
        if self.dones["__all__"]:
            if self.red_alive and not self.blue_alive:  # 红方胜利
                rewards['red_agent'] += 100
                rewards['blue_agent'] -= 100
            elif not self.red_alive and self.blue_alive:  # 蓝方胜利
                rewards['red_agent'] -= 100
                rewards['blue_agent'] += 100
        return rewards

    def _get_observations(self) -> dict:
        """为每个智能体生成观测向量"""
        obs_red = self._get_agent_observation('red') if self.red_alive else np.zeros(10)
        obs_blue = self._get_agent_observation('blue') if self.blue_alive else np.zeros(10)
        return {'red_agent': obs_red, 'blue_agent': obs_blue}

    def _get_agent_observation(self, agent_side: str) -> np.ndarray:
        """为单个智能体生成观测"""
        if agent_side == 'red':
            agent_ac, opponent_ac, incoming_missiles = self.red_aircraft, self.blue_aircraft, self.blue_missiles
        else:
            agent_ac, opponent_ac, incoming_missiles = self.blue_aircraft, self.red_aircraft, self.red_missiles

        # 1. 自身状态
        o_vel = agent_ac.velocity / 400.0
        o_alt = agent_ac.pos[1] / 15000.0
        o_pitch = agent_ac.state_vector[4] / (np.pi / 2)
        o_roll = agent_ac.state_vector[6] / np.pi
        o_missiles = agent_ac.missile_ammo / self.initial_missiles
        o_flares = agent_ac.flare_ammo / self.initial_flares

        # 2. 对手信息
        rel_vec = opponent_ac.pos - agent_ac.pos
        rel_dist = np.linalg.norm(rel_vec)
        o_target_dist = np.clip(rel_dist / self.max_disengagement_range, 0, 1)
        rel_bearing = (np.arctan2(rel_vec[2], rel_vec[0]) - agent_ac.state_vector[5] + np.pi) % (2 * np.pi) - np.pi
        o_target_bearing = rel_bearing / np.pi
        rel_elev = (np.arcsin(np.clip(rel_vec[1] / (rel_dist + 1e-6), -1, 1)) - agent_ac.state_vector[
            4] + np.pi) % (2 * np.pi) - np.pi
        o_target_elev = rel_elev / np.pi

        # 3. 威胁信息 (最危险的来袭导弹)
        threat_dist = 1.0
        active_threats = [m for m in incoming_missiles if m.is_active]
        if active_threats:
            closest_threat_dist = min([np.linalg.norm(m.pos - agent_ac.pos) for m in active_threats])
            threat_dist = np.clip(closest_threat_dist / self.max_disengagement_range, 0, 1)
        o_threat_dist = threat_dist

        return np.array([
            o_vel, o_alt, o_pitch, o_roll, o_missiles, o_flares,
            o_target_dist, o_target_bearing, o_target_elev, o_threat_dist
        ]).astype(np.float32)

    def _update_history(self):
        """记录当前帧的数据"""
        self.history['time'].append(self.t_now)
        if self.red_alive: self.history['red_aircraft'].append(self.red_aircraft.state_vector.copy())
        if self.blue_alive: self.history['blue_aircraft'].append(self.blue_aircraft.state_vector.copy())

        for m in self.red_missiles:
            if m.id not in self.history['red_missiles']: self.history['red_missiles'][m.id] = []
            self.history['red_missiles'][m.id].append(m.state_vector.copy())

        for m in self.blue_missiles:
            if m.id not in self.history['blue_missiles']: self.history['blue_missiles'][m.id] = []
            self.history['blue_missiles'][m.id].append(m.state_vector.copy())

    def _stream_tacview_frame(self):
        """发送当前世界状态到Tacview"""
        if not self.tacview_enabled or not self.tacview.is_connected or self.tacview.tacview_final_frame_sent:
            return

        # --- 修正部分 ---
        # 直接根据 red_alive 和 blue_alive 标志来构建列表，更简单且不会出错
        all_aircraft = []
        if self.red_alive and self.red_aircraft:
            all_aircraft.append(self.red_aircraft)
        if self.blue_alive and self.blue_aircraft:
            all_aircraft.append(self.blue_aircraft)
        # --- 修正结束 ---

        all_missiles = [m for m in self.red_missiles if m.is_active] + [m for m in self.blue_missiles if m.is_active]
        all_flares = self.red_flare_manager.flares + self.blue_flare_manager.flares
        self.tacview.stream_multi_object_frame(self.tacview_global_time, all_aircraft, all_missiles, all_flares)

    def render(self, view_init_elev=20, view_init_azim=-150):
        """使用matplotlib进行3D可视化"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制飞机轨迹
        if self.history['red_aircraft']:
            red_traj = np.array(self.history['red_aircraft'])
            ax.plot(red_traj[:, 2], red_traj[:, 0], red_traj[:, 1], 'r-', label='红方飞机')
            ax.scatter(red_traj[-1, 2], red_traj[-1, 0], red_traj[-1, 1], color='red', s=50, marker='>')

        if self.history['blue_aircraft']:
            blue_traj = np.array(self.history['blue_aircraft'])
            ax.plot(blue_traj[:, 2], blue_traj[:, 0], blue_traj[:, 1], 'b--', label='蓝方飞机')
            ax.scatter(blue_traj[-1, 2], blue_traj[-1, 0], blue_traj[-1, 1], color='blue', s=50, marker='<')

        # 绘制导弹轨迹
        for missile_id, traj_list in self.history['red_missiles'].items():
            traj = np.array(traj_list)
            ax.plot(traj[:, 5], traj[:, 3], traj[:, 4], 'm:', label=f'红方导弹')

        for missile_id, traj_list in self.history['blue_missiles'].items():
            traj = np.array(traj_list)
            ax.plot(traj[:, 5], traj[:, 3], traj[:, 4], 'c:', label=f'蓝方导弹')

        ax.set_xlabel('东 (Z) / m')
        ax.set_ylabel('北 (X) / m')
        ax.set_zlabel('天 (Y) / m')
        ax.legend()
        ax.set_title('红蓝空战对抗三维轨迹')
        ax.view_init(elev=view_init_elev, azim=view_init_azim)
        plt.show()