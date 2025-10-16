# 文件: decoys.py

import numpy as np

# (中文) 导入 Aircraft 类，以便在 FlareManager 中进行类型提示和访问飞机状态
# 您需要确保 aircraft.py 和这个文件在同一个目录下或可被Python解释器找到
from .AircraftJSBSim_DirectControl import Aircraft


class Flare:
    """
    封装了单个红外诱饵弹的所有状态、物理参数和行为。
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray, release_time: float,
                 m0=0.5, m_dot=0.01, rho=1.225, c=0.5, s=0.01,
                 I_max=10000, t1=0.5, t2=3.5, t3=5.0):
        """
        初始化单个诱饵弹。
        """
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.release_time = release_time

        # 物理参数
        self.m0 = m0
        self.m_dot = m_dot
        self.rho = rho  # (注意) 固定的空气密度，简化模型
        self.c = c
        self.s = s
        self.g_vec = np.array([0, -9.81, 0])  # 重力矢量 (NUE坐标系)

        # 红外辐射参数
        self.I_max = I_max
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        # 历史轨迹记录
        self.history = [np.array([*self.pos, release_time])]

    def update(self, t: float, dt: float):
        """
        更新诱饵弹在一个时间步 dt 后的位置与速度。
        """
        t_rel = t - self.release_time
        # 如果诱饵弹还未燃烧或已烧完，则不更新物理状态
        if t_rel < 0 or t_rel > self.t3:
            return

        v_mag = np.linalg.norm(self.vel)
        if v_mag < 1e-6:  # 避免除零
            return

        # (中文) 您的代码中，m_t, f, a 的计算依赖于固定的rho，这里保持一致
        # 更真实的模型会让 rho 随高度变化
        m_t = max(self.m0 - self.m_dot * t_rel, 0.01)
        f_drag = 0.5 * self.rho * v_mag ** 2 * self.c * self.s
        a_drag_vec = - (f_drag / m_t) * (self.vel / v_mag)

        total_accel = a_drag_vec + self.g_vec

        self.pos += self.vel * dt + 0.5 * total_accel * dt ** 2
        self.vel += total_accel * dt

        self.history.append(np.array([*self.pos, t]))

    def get_intensity(self, t: float) -> float:
        """
        根据当前时间 t，计算诱饵弹的红外辐射强度。
        """
        t_rel = t - self.release_time
        if t_rel < 0: return 0.0

        if t_rel <= self.t1:
            return self.I_max * t_rel / self.t1
        elif t_rel <= self.t2:
            return self.I_max
        elif t_rel <= self.t3:
            return (self.t3 - t_rel) * self.I_max / (self.t3 - self.t2)
        else:
            return 0.0


class FlareManager:
    """
    管理所有诱饵弹的投放计划、创建和更新。
    """

    def __init__(self, flare_per_group=1, interval=0.1, release_speed=50):
        self.flare_per_group = flare_per_group
        self.interval = interval
        self.release_speed = release_speed

        self.flares: list[Flare] = []
        self.schedule: list[tuple] = []

    def reset(self):
        """清空所有诱饵弹和投放计划，为新回合做准备。"""
        self.flares = []
        self.schedule = []

    def release_flare_group(self, t_start: float):
        """
        制定一个投放计划，从 t_start 开始，按间隔投放一组诱饵弹。
        """
        for i in range(self.flare_per_group):
            release_time = t_start + i * self.interval
            self.schedule.append(release_time)

    # <<< 新增 >>> 用于处理复杂投放计划的新方法
    def schedule_program(self, t_start, program):
        """
        根据一个投放程序，将未来的所有投放事件安排到时间表中。

        Args:
            t_start (float): 投放程序的开始时间。
            program (dict): 包含投放参数的字典, 例如:
                {
                    'salvo_size': 4,      # 每组4发
                    'intra_interval': 0.1,  # 组内间隔0.1秒
                    'num_groups': 2,      # 共2组
                    'inter_interval': 1.0   # 组间隔1.0秒
                }
        """
        salvo_size = program['salvo_size']
        intra_interval = program['intra_interval']
        num_groups = program['num_groups']
        inter_interval = program['inter_interval']

        # 遍历每一组
        for group_idx in range(num_groups):
            # 计算当前组的开始时间
            group_start_time = t_start + group_idx * inter_interval

            # 遍历组内的每一发
            for salvo_idx in range(salvo_size):
                # 计算每一发的精确投放时间
                release_time = group_start_time + salvo_idx * intra_interval

                # 将这个时间点加入到投放时间表中
                # self.schedule 是一个存储未来投放时间的列表或队列
                # 假设它是一个简单的列表
                if release_time not in self.schedule:
                    self.schedule.append(release_time)

        # 对时间表进行排序，确保按时间顺序执行
        self.schedule.sort()

    def update(self, t: float, dt: float, aircraft: Aircraft):
        """
        (V2 - 稳健的时间戳匹配)
        在每个时间步被调用：
        1. 检查计划列表，创建并释放新的诱饵弹。
        2. 更新所有在空中的诱饵弹的物理状态。
        """
        # --- 1. 创建并释放新的诱饵弹 ---

        # <<< 核心修正：使用稳健的时间区间判断 >>>
        # a. 定义当前物理时间步所覆盖的时间区间
        t_start_of_step = t
        t_end_of_step = t + dt

        # b. 找出所有落在这个时间区间内的计划时间点
        #    我们使用 t_start_of_step <= sch_time < t_end_of_step
        #    左闭右开区间确保每个时间点只被触发一次。
        newly_scheduled_times = [sch_time for sch_time in self.schedule if t_start_of_step <= sch_time < t_end_of_step]

        if newly_scheduled_times:
            # 获取飞机当前状态 (只在需要释放时获取一次)
            aircraft_pos = aircraft.pos
            aircraft_vel_vec = aircraft.get_velocity_vector()
            phi, theta, psi = aircraft.attitude_rad  # 确保顺序正确: roll, pitch, yaw

            # --- 计算释放方向 ---
            # 机体坐标系下的“后下方”方向向量 (x前, y上, z右)
            v_b = np.array([-1.0, -1.0, 0.0])
            v_b /= np.linalg.norm(v_b)

            # (中文) 使用更标准的 ZYX 欧拉角旋转顺序 (Yaw-Pitch-Roll)
            # 这通常比分别应用旋转矩阵更稳健
            cr, sr = np.cos(phi), np.sin(phi)
            cp, sp = np.cos(theta), np.sin(theta)
            cy, sy = np.cos(psi), np.sin(psi)

            # 从机体到世界的旋转矩阵 R_b_to_n
            R_b_to_n = np.array([
                [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                [-sp, sr * cp, cr * cp]
            ])

            release_dir_inertial = R_b_to_n @ v_b

            # 合成诱饵弹初始速度
            v_rel = self.release_speed * release_dir_inertial
            initial_flare_velocity = aircraft_vel_vec + v_rel

            # 为每个计划好的时间点创建 Flare 对象
            for release_time in newly_scheduled_times:
                # 使用 release_time 作为创建 Flare 的精确时间戳
                new_flare = Flare(aircraft_pos, initial_flare_velocity, release_time)
                self.flares.append(new_flare)

                # <<< 调试日志：确认实际释放 >>>
                # print(
                #     f"*** [{t:.4f}s] 成功匹配并释放了计划在 {release_time:.4f}s 的诱饵弹！已释放总数: {len(self.flares)} ***")

            # 从计划列表中移除已处理的项
            self.schedule = [sch_time for sch_time in self.schedule if sch_time not in newly_scheduled_times]

        # --- 2. 更新所有在空中的诱饵弹 ---
        # 更新时，应该传入当前步的结束时间 t_end_of_step
        for flare in self.flares:
            flare.update(t_end_of_step, dt)
