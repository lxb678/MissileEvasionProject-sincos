import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AirCombatEnv:
    def __init__(self):
        self.g = 9.81
        self.N = 4.7  # 导引律参数
        self.ny_max = 30  # 导弹最大过载
        self.nz_max = 30  # 导弹最大过载
        self.R_kill = 12  # 导弹杀伤半径
        self.dt_dec = 0.6 # 决策步长
        self.dt_normal = 0.3 # 大步长
        self.dt_small = 0.1  # 小步长
        self.dt_flare = 0.1  # 诱饵弹步长
        self.dt = self.dt_normal
        self.R_switch = 500  # 步长切换的相对距离阈值
        self.delta = 50  # 滞回带，避免频繁切换
        self.t_end = 60  # 仿真结束时间
        self.n_ty_max = 2.5  # 飞机过载

        self.flare_manager = FlareManager()
        # self.flare = Flare()

        self.D_max1 = 30000  # 导引头最大距离
        self.Angle_IR = np.deg2rad(90)  # 导引头角度
        self.omega_max = 12  # 0.2秒内完成180度偏转，太夸张了吧！！！！！# 导引头角速度
        self.T_max = 60  # 导引头工作时间

        self.D_bin = 1000
        self.D_max = 10000
        self.N_infrared = 24

        self.t = []
        self.Xt = []
        self.Y = []
        self.seeker_log = []
        self.prev_theta_L = None
        self.prev_phi_L = None
        self.last_valid_theta_dot = 0
        self.last_valid_phi_dot = 0
        # 干扰回报奖励参数
        #稀疏回报1奖励参数
        self.W = 100  # 成功奖励基准
        self.lambda_ = 3  # 成功后按脱靶量递增奖励
        self.U = -100  # 失败固定惩罚
        # 稀疏回报2奖励参数
        self.R_com = 50  # 复合干扰奖励值
        # 密集回报1奖励参数
        self.c1 = 1.0  # 基础奖励
        self.alpha1 = 0.2  # 对数增长系数
        # 密集回报2奖励参数
        self.alpha2 = 2.0  # 奖励幅度
        self.beta = 3.0  # 增益系数
        self.k = 2.0  # 控制系数
        # 密集回报3参数
        self.alpha4 = 10.0  # 导弹失锁奖励值
        self.alpha5 = -5.0 # 激光无效惩罚值
        # 密集回报4参数
        self.alphaR = 2.0  # 惩罚幅度
        self.k1 = 3.0  # 资源紧张增益

    def reset(self):
        self.t_now = 0
        self.done = False
        self.reward = 0

        # self.vT_est = np.array([0, 0, 0])

        self.prev_theta_L = None
        self.prev_phi_L = None
        self.last_valid_theta_dot = 0
        self.last_valid_phi_dot = 0
        self.t = []
        self.Xt = []
        self.Y = []
        self.seeker_log = []
        # self.flare.history = []
        self.flare_manager.flares = []
        # ********************************************************************
        self.prev_R_rel = None
        self.lost_and_separating_duration = 0.0
        # 稀疏回报函数2奖励参数
        self.history_flare = []
        self.history_laser = []
        # 密集回报2奖励参数
        self.target_pos_equiv = None
        # self.prev_target_pos_equiv = None

        self.success = False

        # self.x_missile_now = np.array([800, 0, 0, 0, 5000, 0])  # 导弹状态 [V, theta, psi_c, x, y, z]
        # self.x_target_now = np.array([9000, 1000, 1000, 300, 0, 0])  # 飞机状态[x, y, z, V, theta, psi]
        # self.target_mode = 1

        # self.target_mode = np.random.randint(1, 5)  # 1=左转，2=右转，3=爬升，4=俯冲
        "初始化更加随机"
        y_t = np.random.uniform(5000, 10000)
        R = np.random.uniform(9000, 11000)
        theta1 = np.deg2rad(np.random.uniform(-180, 180))
        x_m = R * (-np.cos(theta1))  # 导弹初始x坐标
        z_m = R * np.sin(theta1)  # 导弹初始z坐标
        y_m = y_t + np.random.uniform(-2000, 2000)

        x_target_theta = np.deg2rad(np.random.uniform(-30, 30))  # 设定飞机初始俯仰角
        x_target_phi = np.deg2rad(np.random.uniform(-180, 180))
        # x_target_theta = np.deg2rad(np.random.uniform(0))  # 设定飞机初始俯仰角
        # print("初始飞机俯仰角：{}，飞机机动方式：{}".format(np.rad2deg(x_target_theta), self.target_mode))
        # print("初始飞机俯仰角：{}".format(np.rad2deg(x_target_theta)))
        # self.x_target_now = np.array([0,y_t, 0, np.random.uniform(200, 510), x_target_theta,
        #                               x_target_phi])  # 飞机状态[x, y, z, V, theta, psi]
        # self.x_missile_now = np.array([np.random.uniform(640, 850), 0, 0, x_m, y_m,
        #                                z_m])  # 导弹状态 [V, theta, psi_c, x, y, z]
        self.x_target_now = np.array([0, y_t, 0, np.random.uniform(200, 400), x_target_theta,
                                      x_target_phi])  # 飞机状态[x, y, z, V, theta, psi]
        self.x_missile_now = np.array([np.random.uniform(700, 900), 0, 0, x_m, y_m,
                                       z_m])  # 导弹状态 [V, theta, psi_c, x, y, z]
        # self.x_missile_now[3] = -6000
        # self.x_missile_now[4]=6000
        # self.x_missile_now[5] = 3000
        # self.control_input = self.generate_control_input(self.x_target_now, self.n_ty_max, self.target_mode)
        Rx = self.x_target_now[0] - self.x_missile_now[3]
        Ry = self.x_target_now[1] - self.x_missile_now[4]
        Rz = self.x_target_now[2] - self.x_missile_now[5]
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        self.theta_L = np.arcsin(Ry / R)
        self.phi_L = np.arctan2(Rz, Rx)

        # x_missile_theta = np.random.uniform(0, self.theta_L*2)  # 导弹初始倾角 :随机产生，处于 0度 和 弹目倾角 之间
        # x_missile_psi_c = np.random.uniform(0, -self.phi_L*2)  # 导弹初始偏角 :随机产生，处于 0度 和 弹目偏角 之间
        x_missile_theta = np.random.uniform(0, self.theta_L * 2)  # 导弹初始倾角 :随机产生，处于 0度 和 弹目倾角 之间
        x_missile_psi_c = np.random.uniform(-self.phi_L + np.deg2rad(80), -self.phi_L - np.deg2rad(80))
        self.x_missile_now[1] = x_missile_theta
        self.x_missile_now[2] = x_missile_psi_c

        "尾后态势下初始化"
        # R = np.random.uniform(6000, 8000)
        # theta1 = np.deg2rad(np.random.uniform(-30, 30))
        # x_m = R * (-np.cos(theta1))  # 导弹初始x坐标
        # z_m = R * np.sin(theta1)  # 导弹初始z坐标
        #
        # x_target_theta = np.deg2rad(np.random.uniform(-30, 30))  # 设定飞机初始俯仰角
        # # print("初始飞机俯仰角：{}，飞机机动方式：{}".format(np.rad2deg(x_target_theta), self.target_mode))
        # # print("初始飞机俯仰角：{}".format(np.rad2deg(x_target_theta)))
        # self.x_target_now = np.array([0, np.random.uniform(3000, 5000), 0, np.random.uniform(200, 400), x_target_theta,
        #                               0])  # 飞机状态[x, y, z, V, theta, psi]
        # self.x_missile_now = np.array([np.random.uniform(700, 900), 0, 0, x_m, np.random.uniform(3000, 5000),
        #                                z_m])  # 导弹状态 [V, theta, psi_c, x, y, z]
        # # self.control_input = self.generate_control_input(self.x_target_now, self.n_ty_max, self.target_mode)
        # Rx = self.x_target_now[0] - self.x_missile_now[3]
        # Ry = self.x_target_now[1] - self.x_missile_now[4]
        # Rz = self.x_target_now[2] - self.x_missile_now[5]
        # R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        # self.theta_L = np.arcsin(Ry / R)
        # self.phi_L = np.arctan2(Rz, Rx)
        #
        # x_missile_theta = np.random.uniform(0, self.theta_L)  # 导弹初始倾角 :随机产生，处于 0度 和 弹目倾角 之间
        # x_missile_psi_c = np.random.uniform(0, -self.phi_L)  # 导弹初始偏角 :随机产生，处于 0度 和 弹目偏角 之间
        # self.x_missile_now[1] = x_missile_theta
        # self.x_missile_now[2] = x_missile_psi_c
        "尾后态势下初始化"

        # self.x_missile_now[2] = np.deg2rad(80)
        # print("初始弹目倾角：{}，初始导弹倾角：{}".format(np.rad2deg(self.theta_L),np.rad2deg(x_missile_theta)))
        # print("初始弹目偏角：{}，初始导弹偏角：{}".format(np.rad2deg(self.phi_L), np.rad2deg(x_missile_psi_c)))
        # self.x_missile_now[4] = 6000
        # print("self.x_missile_now",self.x_missile_now)
        # print("self.x_target_now",self.x_target_now)
        o_dis = np.clip(int(R / self.D_bin), 0, self.D_max / self.D_bin)  # 机弹距离状态输入值
        o_di = np.array([self.compute_relative_beta2(self.x_target_now, self.x_missile_now), -self.theta_L])  # 机弹相对方位：水平和倾斜

        # print("初始机弹相对方位：水平：{}，倾斜：{}".format(np.rad2deg(o_di[0]), np.rad2deg(o_di[1])))
        if o_di[0] < np.pi and np.abs(Ry) <= 1000:
            self.target_mode = 1
        elif o_di[0] >= np.pi and np.abs(Ry) <= 1000:
            self.target_mode = 2
        elif Ry > 1000:
            self.target_mode = 3
        elif Ry < -1000:
            self.target_mode = 4
        # self.target_mode = np.random.randint(1, 5)  # 1=左转，2=右转，3=爬升，4=俯冲

        # self.x_missile_now[0] = 800
        # self.x_missile_now[1] = 0
        # self.x_missile_now[2] = -1.221
        # self.x_missile_now[3] = -1897
        # self.x_missile_now[4] = 9995
        # self.x_missile_now[5] = -7583
        # self.x_target_now[0] = 0
        # self.x_target_now[1] = 8000
        # self.x_target_now[2] = 0
        # self.x_target_now[3] = 223
        # self.x_target_now[4] = 0
        # self.x_target_now[5] = -1.702
        # R = 12689
        # angle_1 = 4.469
        # angle_2 = 0.250
        # self.target_mode = 1
        # self.N_infrared = 24
        # self.x_missile_now = np.array(self.x_missile_now)
        # self.x_target_now = np.array(self.x_target_now)
        # o_dis = np.clip(int(R / self.D_bin), 0, self.D_max / self.D_bin)  # 机弹距离状态输入值
        # o_di = np.array(
        #     [angle_1, angle_2])  # 机弹相对方位：水平和倾斜



        self.control_input = self.generate_control_input(self.x_target_now, self.n_ty_max, self.target_mode)

        o_av = self.x_target_now[3]  # 飞机速度
        # o_ap = self.x_target_now.take([0, 2]) #飞机位置
        o_h = self.x_target_now[1]  # 飞机高度
        o_ae = self.x_target_now[4]  # 飞机俯仰角
        o_am = self.target_mode  # 飞机机动动作
        self.o_ir = self.N_infrared  # 红外诱饵弹数量

        # 归一化
        o_dis = np.array([o_dis / 10])
        o_di = np.array([o_di[0] / (2 * np.pi), (o_di[1] - (- np.pi / 2)) / np.pi])
        o_av = np.array([(o_av - 200) / 310])
        o_h = np.array([(o_h - 1000) / 7000])
        o_ae = np.array([(o_ae - (- np.pi / 2)) / np.pi])
        o_am = np.array([(o_am - 1) / 3])
        o_ir = np.array([self.o_ir / self.N_infrared])

        self.observation = np.concatenate((o_dis, o_di, o_av, o_h, o_ae, o_am, o_ir))

        return self.observation


    def step(self, action):
        release_flare = action[0]  #  是否释放红外诱饵弹
        open_laser = action[1]  # 是否开启激光定向干扰
        self.reward = 0
        # release_flare = 0  #  是否释放红外诱饵
        # open_laser = 0  # 是否开启激光定向干扰
        # 在每个 step 中记录行为（建议在 run_one_step 或 step 中执行一次）
        self.history_flare.append(release_flare)
        self.history_laser.append(open_laser)

        self.interference_success = False  # 激光定向干扰是否成功
        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        if R_rel < self.R_switch:
            for _ in range(int(round(self.dt / self.dt_small))):
                self.run_one_step(self.dt_small, release_flare, open_laser)
                release_flare = 0
                if self.done:
                    break

        else:
            if release_flare or self.flare_manager.schedule:
                # print("self.t_now", self.t_now)
                # print('self.flare_manager.schedule', self.flare_manager.schedule)
                for _ in range(int(round(self.dt / self.dt_flare))):
                    # print("int(self.dt / self.dt_flare", int(round(self.dt / self.dt_flare)))
                    self.run_one_step(self.dt_flare, release_flare, open_laser)
                    release_flare = 0
                    if self.done:
                        break
            else:
                self.run_one_step(self.dt, release_flare, open_laser)
        # print("self.x_missile_now", self.x_missile_now)
        # print("self.x_target_now", self.x_target_now)
        ##todo:输入状态

        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        o_dis = np.clip(int(R_rel / self.D_bin), 0, self.D_max / self.D_bin)  # 机弹距离状态输入值

        Ry_rel = self.x_missile_now[4] - self.x_target_now[1]
        self.theta_L_rel = np.arcsin(Ry_rel / R_rel)  # 导弹相对于飞机，导弹在飞机上方为正

        o_di = np.array([self.compute_relative_beta2(self.x_target_now, self.x_missile_now), self.theta_L_rel])  # 机弹相对方位：水平和倾斜
        # print("机弹相对方位：水平和倾斜", np.rad2deg(o_di[0]), np.rad2deg(o_di[1]))
        o_av = self.x_target_now[3]  # 飞机速度
        # o_ap = self.x_target_now.take([0, 2]) #飞机位置
        o_h = self.x_target_now[1]  # 飞机高度
        o_ae = self.x_target_now[4]  # 飞机俯仰角
        o_am = self.target_mode  # 飞机机动动作

        # 归一化
        o_dis = np.array([o_dis / 10])
        o_di = np.array([o_di[0] / (2 * np.pi), (o_di[1] - (- np.pi / 2)) / np.pi])
        o_av = np.array([(o_av - 200) / 310])
        o_h = np.array([(o_h - 1000) / 7000])
        o_ae = np.array([(o_ae - (- np.pi / 2)) / np.pi])
        o_am = np.array([(o_am - 1) / 3])
        o_ir = np.array([self.o_ir / self.N_infrared])

        self.observation = np.concatenate((o_dis, o_di, o_av, o_h, o_ae, o_am, o_ir))
        # observation = np.concatenate((self.x_missile_now, self.x_target_now))

        if self.t_now >= self.t_end and not self.done:
            print(f">>> 仿真达到60s,成功逃离!")
            self.done = True
            self.Xt = np.array(self.Xt)
            self.Y = np.array(self.Y)
            self.t = np.array(self.t)
            self.miss_distance, is_hit, R_all, t_minR, self.idx_min = self.evaluate_miss(self.t, self.Y, self.Xt, self.R_kill)
            # # self.reward += 100

        # == 奖励计算 ==
        if self.done:
            if self.miss_distance > self.R_kill:
                self.success = True
            #miss_distance, is_hit, R_all, t_minR, self.idx_min = self.evaluate_miss(self.t, self.Y, self.Xt,self.R_kill)
            self.reward += self.compute_sparse_reward_1(self.miss_distance)
            self.reward += self.compute_sparse_reward_2()

            self.reward += self.compute_dense_reward_3(open_laser)
            reward_4 = 0
        else:
            self.reward += self.compute_dense_reward_1()
            self.reward += self.compute_dense_reward_2()
            self.reward += self.compute_dense_reward_3(open_laser)
            # self.reward3 = self.compute_dense_reward_3(open_laser)
            # print("self.compute_dense_reward_3()", self.compute_dense_reward_3(open_laser))
            # self.reward += self.compute_dense_reward_4(action[0])
            # self.reward += self.compute_dense_reward_4(action[0])
            reward_4 = self.compute_dense_reward_4(action[0]) * 1.5 / 5.0
            # reward_4 = 0
            self.reward /= 5.0

            # if self.compute_dense_reward_4(action[0]) != 0:
            #     reward_4=self.compute_dense_reward_4(action[0])
            # else:
            #     reward_4 =0




        return self.observation, self.reward, self.done, reward_4, 1

    # == 回报函数 ==
    def compute_sparse_reward_1(self, miss_distance):
        """干扰成功与失败稀疏回报函数，要提前算一下脱靶量"""
        if miss_distance > self.R_kill:
            #self.W + self.lambda_ * (miss_distance - self.R_kill)
            return self.W
        else:
            return self.U

    def compute_sparse_reward_2(self):
        """复合干扰奖励：只要在整段过程中使用过红外诱饵和激光各一次，就给予奖励"""
        if any(self.history_flare) and any(self.history_laser):
            return self.R_com
        return 0

    def compute_dense_reward_1(self):
        """生存时间密集奖励"""
        return self.c1 + self.alpha1 * np.log(1 + self.t_now)

    def compute_dense_reward_2(self):
        """等效辐射中心偏移距离奖励"""
        if self.target_pos_equiv is None:
            return 0.0
        # 计算扰动误差 Δd_t（等效红外质心位置与真实目标位置的偏移）
        delta_d = np.linalg.norm(self.target_pos_equiv - self.x_target_now[:3])

        # 奖励函数
        return self.alpha2 * (1 - np.exp(-self.beta * ((delta_d / self.R_kill) ** self.k)))


    def compute_dense_reward_3(self, open_laser):
        """激光定向干扰导致导引头失锁，给予奖励，若无效给予惩罚"""
        if self.interference_success:
            return self.alpha4
        elif open_laser == 1 and not self.interference_success:
            return self.alpha5
        else:
            return 0


    def compute_dense_reward_4(self, release_flare):
        """干扰资源使用过量惩罚函数"""
        if release_flare == 1:
            fraction_remaining = self.o_ir / self.N_infrared
            penalty = -self.alphaR * (1 + self.k1 * (1 - fraction_remaining))
            return penalty
        else:
            return 0.0

    def render(self):
        # ========================= 可视化 =========================
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.Y[:, 3], self.Y[:, 5], self.Y[:, 4], 'b-', label='导弹轨迹')
        ax.plot(self.Xt[:, 0], self.Xt[:, 2], self.Xt[:, 1], 'r--', label='目标轨迹')
        ax.scatter(self.Y[0, 3], self.Y[0, 5], self.Y[0, 4], color='g', label='导弹起点')
        ax.scatter(self.Y[self.idx_min, 3], self.Y[self.idx_min, 5], self.Y[self.idx_min, 4], color='m', label='最近点')

        for i, flare in enumerate(self.flare_manager.flares):
            traj = np.array(flare.history)
            if traj.shape[0] > 0:
                ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color='orange', linewidth=1,
                        label='红外诱饵弹' if i == 0 else "")

        ax.set_xlabel('X / m')
        ax.set_ylabel('Z / m')
        ax.set_zlabel('Y / m')
        ax.invert_yaxis()
        ax.legend()
        ax.set_title('导引头模型 + 红外诱饵弹建模仿真')
        ax.grid()
        ax.view_init(elev=20, azim=-150)
        plt.show()

    def run_one_step(self, dt, release_flare, open_laser):

        def rk4_aircraft(x):
            k1 = self.aircraft_dynamics(self.t_now, x, self.control_input, self.g)
            k2 = self.aircraft_dynamics(self.t_now + dt / 2, x + dt / 2 * k1, self.control_input, self.g)
            k3 = self.aircraft_dynamics(self.t_now + dt / 2, x + dt / 2 * k2, self.control_input, self.g)
            k4 = self.aircraft_dynamics(self.t_now + dt, x + dt * k3, self.control_input, self.g)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.x_target_next = rk4_aircraft(self.x_target_now)
        # print(np.rad2deg(self.x_target_next[4]))
        # if self.x_target_next[4] >= np.deg2rad(60):
        #     self.x_target_next[4] = np.deg2rad(60)
        # elif self.x_target_next[4] <= np.deg2rad(-60):
        #     self.x_target_next[4] = np.deg2rad(-60)

        if release_flare == 1:
            if self.o_ir > 0:
                self.o_ir -= 6  # 红外诱饵弹数量
                self.flare_manager.release_flare_group(self.t_now)

        self.flare_manager.update(self.t_now, dt, self.x_target_now)

        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        if open_laser == 1:
            self.interference_success = self.laser_interference(R_vec, R_rel,
                                                           self.compute_velocity_vector(self.x_missile_now))

        lock_aircraft,  Flag_D, Flag_A, Flag_omega, Flag_T = self.check_seeker_lock(self.x_missile_now, self.x_target_now,
                                                   self.compute_velocity_vector(self.x_missile_now),
                                                   self.compute_velocity_vector_from_target(self.x_target_now),
                                                   self.t_now, self.D_max1, self.Angle_IR, self.omega_max, self.T_max)
        # print("lock_aircraft", lock_aircraft, Flag_D, Flag_A, Flag_omega, Flag_T)
        if lock_aircraft and not self.interference_success:
            beta_p = self.compute_relative_beta(self.x_target_now, self.x_missile_now)
            I_p = self.infrared_intensity_model(beta_p)
            pos_p = self.x_target_now[:3]
        else:
            I_p = 0.0
            pos_p = np.zeros(3)

        visible_flares = []
        for flare in self.flare_manager.flares:
            if flare.get_intensity(self.t_now) <= 1e-3:
                continue
            lock_flare, *_ = self.check_seeker_lock(self.x_missile_now,
                                                    np.concatenate((flare.pos, [0, 0, 0])),
                                                    self.compute_velocity_vector(self.x_missile_now),
                                                    flare.vel,
                                                    self.t_now, self.D_max1, self.Angle_IR, self.omega_max, self.T_max)
            if lock_flare:
                visible_flares.append(flare)

        I_total = I_p
        numerator = I_p * pos_p
        for flare in visible_flares:
            I_k = flare.get_intensity(self.t_now)
            numerator += I_k * flare.pos
            I_total += I_k

        if I_total > 0:
            # self.prev_target_pos_equiv = self.target_pos_equiv

            self.target_pos_equiv = numerator / I_total
            Rx = self.target_pos_equiv[0] - self.x_missile_now[3]
            Ry = self.target_pos_equiv[1] - self.x_missile_now[4]
            Rz = self.target_pos_equiv[2] - self.x_missile_now[5]
            R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
            self.theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
            self.phi_L = np.arctan2(Rz, Rx)
            theta_L_dot = 0.0 if self.prev_theta_L is None else (self.theta_L - self.prev_theta_L) / dt
            dphi = np.arctan2(np.sin(self.phi_L - self.prev_phi_L),
                              np.cos(self.phi_L - self.prev_phi_L)) if self.prev_phi_L else 0.0
            phi_L_dot = dphi / dt
        else:
            self.theta_L = self.prev_theta_L if self.prev_theta_L is not None else 0.0
            self.phi_L = self.prev_phi_L if self.prev_phi_L is not None else 0.0
            theta_L_dot = self.last_valid_theta_dot
            phi_L_dot = self.last_valid_phi_dot

        def update_missile(x, theta_L_dot, phi_L_dot, N, ny_max, nz_max):
            V, theta, psi_c, x_pos, y_pos, z_pos = x
            g = 9.81
            m = 60  # 导弹质量
            # print("V",V)

            # # === LOS单位向量 ===
            # if self.target_pos_equiv  is None or self.prev_target_pos_equiv is None:
            #     aT_perp_body = np.array([0, 0, 0])
            # else:
            #     r_rel = self.target_pos_equiv - self.x_missile_now[3:6]
            #     r_hat = r_rel / (np.linalg.norm(r_rel) + 1e-6)
            #     vT_last_est = self.vT_est
            #     self.vT_est = (self.target_pos_equiv - self.prev_target_pos_equiv) / dt
            #     aT = (self.vT_est - vT_last_est) / dt
            #
            #     # === aT 垂直LOS的分量 ===
            #     aT_parallel = np.dot(aT, r_hat) * r_hat
            #     aT_perp = aT - aT_parallel
            #
            #     # === 转到体轴系 ===
            #     from scipy.spatial.transform import Rotation as R
            #     def euler_to_quat_fixed(theta, psi):
            #         # 明确使用 Z（偏航）-Y（俯仰）顺序
            #         r = R.from_euler('zyx', [theta, psi, 0.0])  # ψ, θ, φ
            #         return r.as_quat()  # 返回 [x, y, z, w]
            #     def quat_to_rotation_matrix_fixed(q):
            #         # 输入 [x, y, z, w]
            #         r = R.from_quat(q)
            #         return r.as_matrix()
            #     q = euler_to_quat_fixed(theta, psi_c)
            #     R = quat_to_rotation_matrix_fixed(q)
            #     R_ib = R.T
            #     aT_perp_body = R_ib @ aT_perp
            #     aT_perp_y = aT_perp_body[1]
            #     aT_perp_z = -aT_perp_body[2]  # 注意z轴方向

            # xt,yt,zt,vt,theta_t,psi_t = self.x_target_now  # 飞机状态[x, y, z, V, theta, psi]
            #
            # Rx = xt - x_pos
            # Ry = yt - y_pos
            # Rz = zt - z_pos
            # R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
            #
            # Vmx = V * np.cos(theta) * np.cos(psi_c)
            # Vmy = V * np.sin(theta)
            # Vmz = -V * np.cos(theta) * np.sin(psi_c)
            # Vxt = vt * np.cos(theta_t) * np.cos(psi_t)
            # Vyt = vt * np.sin(theta_t)
            # Vzt = -vt * np.cos(theta_t) * np.sin(psi_t)
            # #
            # Vr_x = Vxt - Vmx
            # Vr_y = Vyt - Vmy
            # Vr_z = Vzt - Vmz
            #
            # Rdot = (Rx * Vr_x + Ry * Vr_y + Rz * Vr_z) / R
            # theta_L_dot = (Vr_y * np.sqrt(Rx ** 2 + Rz ** 2) - Ry * (Rx * Vr_x + Rz * Vr_z) / np.sqrt(Rx ** 2 + Rz ** 2)) / (Rx ** 2 + Ry ** 2 + Rz ** 2)
            # phi_L_dot = (Vr_z * Rx - Rz * Vr_x) / (Rx ** 2 + Rz ** 2)
            # ny =   self.N * np.abs(Rdot) * theta_L_dot / g + np.cos(theta)
            # nz =   -self.N * np.abs(Rdot) * phi_L_dot / g

            # === 1. 过载计算 ===
            ny = self.N * V * theta_L_dot / g + np.cos(theta)
            nz = -self.N * V * phi_L_dot / g
            # print("theta_L_dot / g",theta_L_dot / g,phi_L_dot)
            # print("V , theta_L_dot",V , theta_L_dot)
            # print("v",V)
            # print("ny,nz", ny, nz)


            n_total_cmd = np.sqrt(ny ** 2 + nz ** 2)
            n_max = 50
            if n_total_cmd > n_max:
                scale = n_max / n_total_cmd
                ny *= scale
                nz *= scale
            # print("ny,nz22222222", ny, nz)
            # ny = np.clip(ny_cmd, -self.ny_max, self.ny_max)
            # nz = np.clip(nz_cmd, -self.nz_max, self.nz_max)

            # print("n_total_cmd",n_total_cmd)


            # === 2. 当前速度矢量 ===
            # vx = V * np.cos(theta) * np.cos(psi_c)
            # vy = V * np.sin(theta)
            # vz = -V * np.cos(theta) * np.sin(psi_c)
            # v_now = np.array([vx, vy, vz])

            # === 3. 空气阻力计算 ===
            H = y_pos
            Temper = 15.0
            T_H = 273 + Temper - 0.6 * H / 100
            P_H = (1 - H / 44300) ** 5.256
            rho = 1.293 * P_H * (273 / T_H)
            Ma = V / 340

            def get_Cx(Ma):
                if Ma <= 1.0:
                    return 0.45
                elif Ma <= 1.3:
                    return 0.45 + (0.65 - 0.45) * (Ma - 1.0) / 0.3
                elif Ma <= 4.0:
                    return 0.65 * np.exp(-0.5 * (Ma - 1.3))
                else:
                    return 0.3
            def get_Cx1(Ma):
                if Ma <= 1.0:
                    return 0.5
                elif Ma <= 1.3:
                    return 0.5 + (0.8 - 0.5) * (Ma - 1.0) / 0.3
                elif Ma <= 4.0:
                    return 0.8 * np.exp(-0.25 * (Ma - 1.3))
                else:
                    return 0.3

            Cx = get_Cx1(Ma)
            # Cx = 0
            S = np.pi * (0.127) ** 2 / 4
            q = 0.5 * rho * V ** 2
            F_drag = Cx * q * S  # 阻力大小
            # a_drag = -F_drag / m * v_now / (np.linalg.norm(v_now) + 1e-6)

            # === 4. 过载加速度 ===
            # 导弹体轴方向：x-前, y-上, z-右
            # a_overload_body = np.array([0.0, ny * g, -nz * g])  # z轴为右侧方向
            # 转到惯性系
            # def rotation_matrix(theta, psi):
            #     """
            #     生成从机体坐标系 -> 惯性坐标系的旋转矩阵。
            #     假设滚转角 φ=0，仅考虑俯仰 θ 和偏航 ψ。
            #     """
            #     ct = np.cos(theta)
            #     st = np.sin(theta)
            #     cp = np.cos(psi)
            #     sp = np.sin(psi)
            #
            #     # 方向余弦矩阵（3x3）
            #     # 将机体坐标系中向量 R·v 变换为惯性坐标系下向量
            #     # R = np.array([
            #     #     [ct * cp, -sp, st * cp],
            #     #     [st, 0, -ct],
            #     #     [-ct * sp, -cp, -st * sp]
            #     # ])
            #     R = np.array([
            #         [ct * cp, -st*cp, sp],
            #         [st, ct, 0],
            #         [-ct * sp, st*sp, cp]
            #     ])
            #     return R
            #
            # R = rotation_matrix(theta, psi_c)
            # a_overload = R @ a_overload_body


            # def quat_to_rotation_matrix(q):
            #     """
            #     输入：
            #         q : 四元数 [q0, q1, q2, q3]，满足单位四元数约束
            #     输出：
            #         R : 3x3 旋转矩阵（机体系 -> 惯性系）
            #     """
            #     q0, q1, q2, q3 = q
            #
            #     R = np.array([
            #         [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            #         [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            #         [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
            #     ])
            #     return R
            #
            # def euler_to_quat(theta, psi):
            #     """
            #     仅考虑俯仰角 theta 和偏航角 psi（假设 φ = 0）
            #     """
            #     phi = 0.0
            #     cy = np.cos(psi * 0.5)
            #     sy = np.sin(psi * 0.5)
            #     cp = np.cos(theta * 0.5)
            #     sp = np.sin(theta * 0.5)
            #     cr = np.cos(phi * 0.5)
            #     sr = np.sin(phi * 0.5)
            #
            #     q0 = cr * cp * cy + sr * sp * sy
            #     q1 = sr * cp * cy - cr * sp * sy
            #     q2 = cr * sp * cy + sr * cp * sy
            #     q3 = cr * cp * sy - sr * sp * cy
            #     return np.array([q0, q1, q2, q3])
            #
            # from scipy.spatial.transform import Rotation as R
            # def euler_to_quat_fixed(theta, psi):
            #     # 明确使用 Z（偏航）-Y（俯仰）顺序
            #     r = R.from_euler('zyx', [theta, psi, 0.0])  # ψ, θ, φ
            #     return r.as_quat()  # 返回 [x, y, z, w]
            #
            # def quat_to_rotation_matrix_fixed(q):
            #     # 输入 [x, y, z, w]
            #     r = R.from_quat(q)
            #     return r.as_matrix()
            #
            # # q = euler_to_quat(psi_c,theta)
            # # R = quat_to_rotation_matrix(q)
            # q = euler_to_quat_fixed(theta, psi_c)
            # R = quat_to_rotation_matrix_fixed(q)
            # a_overload = R @ a_overload_body
            #
            # # === 5. 重力加速度 ===
            # # a_gravity = np.array([0.0, -g, 0.0])  # 惯性系 y 为天方向
            # a_gravity = np.array([0.0, -g, 0.0])
            # # === 6. 合加速度 ===
            # a_total = a_overload + a_gravity + a_drag
            # # print("a_total",a_total[1])
            #
            # # === 7. 速度更新 ===
            # v_dot = -F_drag / m - g * np.sin(theta)
            # V_next_value = V + v_dot * dt + 1e-8
            # v_next_dir = v_now + a_total * dt
            # v_next = V_next_value * v_next_dir/np.linalg.norm(v_next_dir)
            # V_next = np.linalg.norm(v_next)
            #
            # # === 8. 姿态更新 ===
            # theta_next = np.arcsin(np.clip(v_next[1] / V_next, -1, 1))
            # psi_c_next = np.arctan2(-v_next[2], v_next[0])  # 注意符号，与模型一致

            # === 新方法,速度不受步长影响 ===
            v_dot = -F_drag / m - g * np.sin(theta)
            V_next = V + v_dot * dt + 1e-8
            # print("V_next",V_next)
            theta_next = theta + ((ny - np.cos(theta)) * g / V_next) * dt
            psi_c_next = psi_c + nz * g / V_next / np.cos(theta) * dt

            theta_next = np.clip(theta_next, -np.deg2rad(89), np.deg2rad(89))
            if psi_c_next > np.pi:
                psi_c_next -= 2 * np.pi
            if psi_c_next < -np.pi:
                psi_c_next += 2 * np.pi
            v_next = V_next * np.array(
                [np.cos(theta_next) * np.cos(psi_c_next), np.sin(theta_next), -np.cos(theta_next) * np.sin(psi_c_next)])
            # === 新方法,速度不受步长影响 结束===

            # === 9. 位置更新 ===
            pos_now = np.array([x_pos, y_pos, z_pos])
            pos_next = pos_now + v_next * dt

            return np.array([V_next, theta_next, psi_c_next, *pos_next])

        self.x_missile_next = update_missile(self.x_missile_now,theta_L_dot, phi_L_dot, self.N, self.ny_max,self.nz_max)
        if self.x_missile_next[1] >= np.deg2rad(89):
            self.x_missile_next[1] = np.deg2rad(89)
        elif self.x_missile_next[1] <= np.deg2rad(-89):
            self.x_missile_next[1] = np.deg2rad(-89)

        # def rk4_missile(x):
        #     k1 = self.missile_dynamics_given_rate(x, theta_L_dot, phi_L_dot, self.N, self.ny_max, self.nz_max)
        #     k2 = self.missile_dynamics_given_rate(x + dt / 2 * k1, theta_L_dot, phi_L_dot, self.N, self.ny_max,
        #                                           self.nz_max)
        #     k3 = self.missile_dynamics_given_rate(x + dt / 2 * k2, theta_L_dot, phi_L_dot, self.N, self.ny_max,
        #                                           self.nz_max)
        #     k4 = self.missile_dynamics_given_rate(x + dt * k3, theta_L_dot, phi_L_dot, self.N, self.ny_max,
        #                                           self.nz_max)
        #     return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        #
        # self.x_missile_next = rk4_missile(self.x_missile_now)


        # # 状态更新
        self.t_now += dt
        self.t.append(self.t_now)
        self.Xt.append(self.x_target_next)
        self.Y.append(self.x_missile_next)

        R_vec_now = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel_now = np.linalg.norm(R_vec_now)
        R_vec_next = self.x_target_next[0:3] - self.x_missile_next[3:6]
        R_rel_next = np.linalg.norm(R_vec_next)

        if R_rel < 500:
            M1 = self.x_missile_now[3:6]
            T1 = self.x_target_now[:3]
            M2 = self.x_missile_next[3:6]
            T2 = self.x_target_next[:3]
            M1_M2 = M2 - M1
            T1_T2 = T2 - T1
            M1_T1 = T1 - M1
            t1 = self.t_now - dt
            t2 = self.t_now
            denominator = np.linalg.norm(T1_T2 - M1_M2) ** 2
            if denominator == 0:
                t_star = t1  # 如果分母为零，直接使用 t1
            else:
                t_star = t1 - (M1_T1.dot(T1_T2 - M1_M2) * dt) / denominator

            # if R_rel < R_rel_next and R_rel < 200:
            if t1 <= t_star <= t2:
                # self.miss_distance = self.calculate_miss_distance(self.x_missile_now, self.x_target_now,
                # self.x_missile_next, self.x_target_next, self.t_now - dt,self.t_now)
                self.miss_distance = np.linalg.norm(M1_T1 + (t_star - t1) * (T1_T2 - M1_M2) / dt)
                if self.miss_distance < self.R_kill:
                    self.done = True
                    print(f">>> 引信引爆！脱靶量 = {self.miss_distance:.2f} m")
                    self.Xt = np.array(self.Xt)
                    self.Y = np.array(self.Y)
                    self.t = np.array(self.t)
                    self.idx_min = len(self.t) - 2


        self.x_target_now = self.x_target_next
        self.x_missile_now = self.x_missile_next
        self.prev_theta_L = self.theta_L
        self.prev_phi_L = self.phi_L



        # ACT_F, self.R_mt_now, self.V_rel_now = self.check_fuze_trigger(self.x_missile_now, self.x_target_now,
        #                                                                self.R_kill, 150)
        # if ACT_F:
        #     self.done = True
        #     self.miss_distance, is_hit, R_all, t_minR, self.idx_min = self.evaluate_miss(self.t, self.Y, self.Xt,
        #                                                                             self.R_kill)
        #     print(f">>> 引信引爆！R = {self.R_mt_now:.2f} m, V_rel = {self.V_rel_now:.2f} m/s")
        #     self.Xt = np.array(self.Xt)
        #     self.Y = np.array(self.Y)
        #     self.t = np.array(self.t)

        # 飞机逃脱判据******************************************************************
        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        lock_aircraft, *_ = self.check_seeker_lock(
            self.x_missile_now, self.x_target_now,
            self.compute_velocity_vector(self.x_missile_now),
            self.compute_velocity_vector_from_target(self.x_target_now),
            self.t_now, self.D_max1, self.Angle_IR, self.omega_max, self.T_max
        )
        if (not lock_aircraft) and \
                (self.prev_R_rel is not None and R_rel > self.prev_R_rel):
            self.lost_and_separating_duration += dt
        else:
            self.lost_and_separating_duration = 0.0

        self.prev_R_rel = R_rel

        if self.lost_and_separating_duration >= 2.0:
            print(f">>> 持续丢失目标 + 距离增大超过2秒，仿真提前终止！m")
            self.done = True
            self.Xt = np.array(self.Xt)
            self.Y = np.array(self.Y)
            self.t = np.array(self.t)
            self.miss_distance, is_hit, R_all, t_minR, self.idx_min = self.evaluate_miss(self.t, self.Y, self.Xt,
                                                                                    self.R_kill)




        # ========================= 飞机动力学模型 =========================

    def aircraft_dynamics(self, t, x, control_input, g):
        xt, yt, zt, Vt, theta, psi = x
        n_tx, n_ty, mu_t = control_input(x)
        dxt = Vt * np.cos(theta) * np.cos(psi)
        dyt = Vt * np.sin(theta)
        dzt = -Vt * np.cos(theta) * np.sin(psi)
        dVt = g * (n_tx - np.sin(theta))
        dtheta = (g / Vt) * (n_ty * np.cos(mu_t) - np.cos(theta))
        dpsi = -(n_ty * g * np.sin(mu_t)) / (Vt * np.cos(theta))
        return np.array([dxt, dyt, dzt, dVt, dtheta, dpsi])

        # ========================= 飞机控制输入 =========================

    def generate_control_input(self, x0, n_ty_max, mode):
        if mode == 1:
            return lambda x: (np.sin(x[4]), n_ty_max, -np.arccos(np.cos(x[4]) / n_ty_max))
        elif mode == 2:
            return lambda x: (np.sin(x[4]), n_ty_max, np.arccos(np.cos(x[4]) / n_ty_max))
        elif mode == 3:
            return lambda x: (np.sin(x[4]), (np.cos(x[4]) + n_ty_max) / np.cos(0), 0)
        elif mode == 4:
            return lambda x: (np.sin(x[4]), (np.cos(x[4]) - n_ty_max) / np.cos(0), 0)
        else:
            raise ValueError('Invalid maneuver mode')
        # ========================= 飞机红外辐射强度模型 =========================

    def infrared_intensity_model(self, beta):
        """
        输入：beta（弧度），导弹相对飞机机体的水平角
        输出：对应红外辐射强度
        """
        # 关键角度点（单位：度），0°表示导弹正对飞机尾部，180°为正前方
        beta_deg = np.array([0, 40, 90, 140, 180])
        intensity_vals = np.array([3800, 5000, 2500, 2000, 800])
        beta_rad = np.deg2rad(beta_deg)

        # 三次样条插值模型
        spline_model = CubicSpline(beta_rad, intensity_vals, bc_type='natural')

        return np.maximum(spline_model(beta), 0.0)

        # 相对角度计算（

    def compute_relative_beta(self, x_target, x_missile):
        """
        输入：
            x_target: 飞机状态向量 [x, y, z, V, theta, psi]
            x_missile: 导弹状态向量 [V, theta, psi, x, y, z]
        输出：
            beta: 目标视线相对于飞机机头水平方向的夹角（弧度）
        """
        # 弹目相对矢量（R）并投影到水平面（xz）
        R_vec = x_target[0:3] - x_missile[3:6]
        R_proj = np.array([R_vec[0], 0.0, R_vec[2]])

        # 飞机水平方向单位向量（注意坐标系方向 x前 y上 z右）
        psi_t = x_target[5]  # 飞机偏航角
        V_body = np.array([np.cos(psi_t), 0.0, -np.sin(psi_t)])  # 机头朝向单位向量

        # 计算夹角 β（视线与机头水平夹角）
        cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)

        return beta

    def compute_relative_beta2(self, x_target, x_missile):
        """
        输入：
            x_target: 飞机状态向量 [x, y, z, V, theta, psi]
            x_missile: 导弹状态向量 [V, theta, psi, x, y, z]
        输出：
            beta: 目标视线相对于飞机机头水平方向的夹角（弧度）
        """
        # 飞机与导弹相对矢量（R）并投影到水平面（xz）  方向：飞机指向导弹
        R_vec = x_missile[3:6] - x_target[0:3]
        R_proj = np.array([R_vec[0], 0.0, R_vec[2]])

        # 飞机水平方向单位向量（注意坐标系方向 x前 y上 z右）
        psi_t = x_target[5]  # 飞机偏航角
        V_body = np.array([np.cos(psi_t), 0.0, -np.sin(psi_t)])  # 机头朝向单位向量

        # 计算夹角 β（视线与机头水平夹角）
        cos_beta = np.dot(V_body, R_proj) / (np.linalg.norm(V_body) * np.linalg.norm(R_proj))
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        # 计算向量的叉积
        cross_product = np.cross(R_proj, V_body)
        reference_axis = (0, 1, 0)
        # 检查叉积与参考轴的方向
        if np.dot(cross_product, reference_axis) < 0:
            beta = 2 * np.pi - beta

        return beta

        # ========================= 导弹动力学模型 =========================

    def missile_dynamics_given_rate(self, y, theta_L_dot, phi_L_dot, N, ny_max, nz_max):
        V, theta, psi_c, x, y_pos, z = y
        g = 9.81

        # === 导引律过载控制 ===
        ny_cmd = N * V * theta_L_dot / g
        nz_cmd = N * V * phi_L_dot / g
        ny = np.clip(ny_cmd, -ny_max, ny_max)
        nz = np.clip(nz_cmd, -nz_max, nz_max)

        # === 飞行高度 ===
        H = y_pos  # y坐标为高度（天方向）

        # === 空气温度、密度、音速 ===
        Temper = 15.0  # 海平面温度（单位°C）
        T_H = 273 + Temper - 0.6 * H / 100  # 高度相关温度 (K)
        P_H = (1 - H / 44300) ** 5.256  # 气压随高度变化（相对比值）
        rho = 1.293 * P_H * (273 / T_H)  # 空气密度 ρ

        # === 马赫数 Ma、音速 ===
        Ma = V / 340

        # === 阻力系数（分段） ===
        def get_Cx(Ma):
            if Ma <= 1.0:
                return 0.45
            elif Ma <= 1.3:
                return 0.45 + (0.65 - 0.45) * (Ma - 1.0) / (1.3 - 1.0)
            elif Ma <= 4.0:
                return 0.65 * np.exp(-0.5 * (Ma - 1.3))
            else:
                return 0.3

        Cx = get_Cx(Ma)

        # === 动压、空气阻力 ===
        S = np.pi * (0.127) ** 2 / 4  # 假设导弹直径 0.15 m，对应迎风面积
        q = 0.5 * rho * V ** 2
        X = Cx * q * S  # 空气阻力

        # === 导弹动力学微分方程 ===
        m = 86  # 导弹质量（单位 kg，可根据实际更改）
        dV = (-X - m * g * np.sin(theta)) / m
        dtheta = (ny - np.cos(theta)) * g / V
        dpsi_c = -nz * g / (V * np.cos(theta))
        dx = V * np.cos(theta) * np.cos(psi_c)
        dy = V * np.sin(theta)
        dz = -V * np.cos(theta) * np.sin(psi_c)

        return np.array([dV, dtheta, dpsi_c, dx, dy, dz])
        # ========================= 视线角 =========================

    def compute_los_angles(self, x_target, x_missile, prev_theta_L, prev_phi_L, dt):
        Rx = x_target[0] - x_missile[3]
        Ry = x_target[1] - x_missile[4]
        Rz = x_target[2] - x_missile[5]
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        theta_L = np.arcsin(Ry / R)
        phi_L = np.arctan2(Rz, Rx)
        if prev_theta_L is None:
            theta_L_dot = 0.0
        else:
            theta_L_dot = (theta_L - prev_theta_L) / dt
        if prev_phi_L is None:
            phi_L_dot = 0.0
        else:
            dphi = np.arctan2(np.sin(phi_L - prev_phi_L), np.cos(phi_L - prev_phi_L))
            phi_L_dot = dphi / dt
        return theta_L, phi_L, theta_L_dot, phi_L_dot

        # ========================= 飞机速度向量 =========================

    def compute_velocity_vector_from_target(self, x):
        V, theta, psi = x[3], x[4], x[5]
        vx = V * np.cos(theta) * np.cos(psi)
        vy = V * np.sin(theta)
        vz = -V * np.cos(theta) * np.sin(psi)
        return np.array([vx, vy, vz])

    def compute_velocity_vector(self, x):
        V, theta, psi = x[0], x[1], x[2]
        vx = V * np.cos(theta) * np.cos(psi)
        vy = V * np.sin(theta)
        vz = -V * np.cos(theta) * np.sin(psi)
        return np.array([vx, vy, vz])

        # ========================= 锁定判据 =========================

    def check_seeker_lock(self, x_missile, x_target, V_missile_vec, V_target_vec, t_now,
                          D_max, Angle_IR, omega_max, T_max):
        R_vec = np.array(x_target[0:3]) - np.array(x_missile[3:6])
        R_mag = np.linalg.norm(R_vec)
        Flag_D = R_mag <= D_max
        cos_Angle = np.dot(R_vec, V_missile_vec) / (np.linalg.norm(R_vec) * np.linalg.norm(V_missile_vec))
        cos_Angle = np.clip(cos_Angle, -1.0, 1.0)
        Angle = np.arccos(cos_Angle)
        Flag_A = (cos_Angle >= np.cos(Angle_IR)) and (cos_Angle > 0)
        # print("cos_Angle",cos_Angle)
        delta_V = V_missile_vec - V_target_vec
        omega_R = np.linalg.norm(delta_V) * np.sin(Angle) / np.linalg.norm(R_vec)
        Flag_omega = omega_R <= omega_max
        Flag_T = t_now <= T_max
        IR_seeker_lock = Flag_D and Flag_A and Flag_omega and Flag_T
        return IR_seeker_lock, Flag_D, Flag_A, Flag_omega, Flag_T

        # ========================= 脱靶评估 =========================

    def evaluate_miss(self, t, Y, Xt, R_kill):
        R_all = []
        for i in range(len(t)):
            xt, yt, zt = Xt[i][:3]
            xm, ym, zm = Y[i][3:6]
            R = np.linalg.norm([xt - xm, yt - ym, zt - zm])
            R_all.append(R)
        R_all = np.array(R_all)
        miss_distance = np.min(R_all)
        idx_min = np.argmin(R_all)
        t_minR = t[idx_min]
        is_hit = (miss_distance <= R_kill)
        print(f">>> {'命中' if is_hit else '未命中'}，脱靶量为：{miss_distance:.2f} m")
        return miss_distance, is_hit, R_all, t_minR, idx_min

    def calculate_miss_distance(self, x_missile_now, x_target_now, x_missile_next, x_target_next, t_now, t_next):
        """
        计算导弹脱靶量。

        参数:
        M1: 时刻 t1 的导弹位置向量
        T1: 时刻 t1 的目标位置向量
        M2: 时刻 t2 的导弹位置向量
        T2: 时刻 t2 的目标位置向量
        Vm: 导弹速度
        Vt: 目标速度
        t1: 时刻 t1
        t2: 时刻 t2

        返回:
        d_miss: 导弹脱靶量
        """
        M1 = x_missile_now[3:6]
        T1 = x_target_now[:3]
        M2 = x_missile_next[3:6]
        T2 = x_target_next[:3]
        t1 = t_now
        t2 = t_next
        # 计算仿真步长
        delta_t = t2 - t1
        # 计算向量
        M1_M2 = M2 - M1
        T1_T2 = T2 - T1
        M1_T1 = T1 - M1
        # 计算弹目距离最小的时刻 t*
        denominator = np.linalg.norm(T1_T2 - M1_M2) ** 2
        if denominator == 0:
            t_star = t1  # 如果分母为零，直接使用 t1
        else:
            t_star = t1 - ((M1_T1.dot(T1_T2) - M1_T1.dot(M1_M2)) * delta_t) / denominator

        # 计算导弹脱靶量
        d_miss = np.linalg.norm(M1_T1 + (t_star - t1) * (T1_T2 - M1_M2) / delta_t)

        return d_miss

        # ================= 引信起爆判断 =================

    def check_fuze_trigger(self, x_missile, x_target, R_cons, V_fuse):
        pos_m = x_missile[3:6]
        pos_t = x_target[:3]
        R_mt = np.linalg.norm(pos_m - pos_t)
        Cons_D = R_mt <= R_cons

        V_m = self.compute_velocity_vector(x_missile)
        V_t = self.compute_velocity_vector_from_target(x_target)
        V_rel = np.linalg.norm(V_m - V_t)
        Cons_V = V_rel >= V_fuse

        ACT_F = Cons_D and Cons_V
        return ACT_F, R_mt, V_rel

        # ========================= 激光定向干扰 =========================

    def laser_interference(self, R_vec, R_rel, V_missile_vec):
        """
        激光定向干扰计算
        :return: 饱和光斑的半径 (mm), 光斑面积 (mm^2), 目标在探测器上的面积 (mm^2), 是否成功干扰导弹 (bool)
        """
        # 参数设定
        H = 5  # 目标高度 (m)
        W = 16  # 目标宽度 (m)
        R = R_rel / 1000  # 目标与热像仪的距离 (km)
        d0 = 12  # 探测器像元尺寸 (mm)
        d = 27  # mm
        f = 57  # 焦距 (mm)
        P0 = 4000  # 激光辐照能量 (W)
        theta = 1  # 束散角 (mrad)
        D0 = 100  # 光学镜头通光口径 (mm)
        a = 43.6  # 衍射光阑半径 (mm)
        tau0 = 0.8  # 大气透过率
        tau1 = 0.8  # 光学镜头透过率
        Ith = 5*10**-2  # 刚饱和时的激光能量 (W/m^2)
        lambda_ = 10.6  # 激光波长 (um)

        # 计算激光干扰入射角
        alpha = np.arccos(
            np.dot(R_vec, V_missile_vec) / (np.linalg.norm(R_vec) * np.linalg.norm(V_missile_vec)))  # 激光干扰入射角 (rad)
        # print("激光入射角度", np.rad2deg(alpha))
        if alpha > self.Angle_IR:
            # print("激光入射角度过大")
            return False
        else:
            # 计算目标成像尺寸
            nH = (H * f) / (R * 1000 * d0)  # 目标像在探测器纵向上所占像素数
            nW = (W * f) / (R * 1000 * d0)  # 目标像在探测器横向上所占像素数

            # 计算激光打在探测器表面焦面功率密度
            I0 = (4 * tau0 * tau1 * P0 * (D0 / d0) ** 2 * np.cos(alpha)) / (np.pi * R ** 2 * theta ** 2)
            # print(f"激光打在探测器表面焦面功率密度: {I0} W/m^2")

            # 计算饱和光斑的半径
            if I0 < Ith:
                r_laser = 0  # 光斑较小，对干扰效果影响不大
            else:
                r_laser = (lambda_ * 0.001 * d) / (2 * np.pi * a) * np.cbrt(4 * I0 / Ith)
            # print(f"饱和光斑的半径: {r_laser} mm")

            # 计算光斑面积和目标在探测器上的面积
            spot_area = np.pi * r_laser ** 2  # 光斑面积
            target_area = np.pi * d0 ** 2 * nH * nW  # 目标在探测器上的面积

            # print(f"光斑面积: {spot_area} mm^2")
            # print(f"目标在探测器上的面积: {target_area} mm^2")

            # 判断是否成功干扰导弹
            if spot_area >= target_area:
                #print("成功干扰")
                return True  # 成功干扰导弹导引头
            else:
                #print("失败")
                return False  # 没有成功干扰导弹导引头


# ========================= 红外诱饵弹类 =========================
class Flare:
    def __init__(self, position, velocity, release_time, m0=0.5, m_dot=0.01,
                 rho=1.225, c=0.5, s=0.01, I_max=6000, t1=1.5, t2=3.5, t3=5.0):
        self.pos = np.array(position, dtype=float)  #释放位置（飞机当前位置）
        self.vel = np.array(velocity, dtype=float)  #初速度（飞机速度 + 后下方扰动）
        self.release_time = release_time  #释放时刻
        self.m0 = m0  #初始质量
        self.m_dot = m_dot  #单位时间内质量损失速率（质量随时间线性下降）
        self.rho = rho  #空气密度
        self.c = c  #阻力系数
        self.s = s  #迎风面积
        self.I_max = I_max  #最大红外辐射强度
        self.t1 = t1  #
        self.t2 = t2  #
        self.t3 = t3  #
        self.history = [self.pos.copy()]  #存储轨迹

    def update(self, t, dt):  #更新位置与速度
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return
        v_mag = np.linalg.norm(self.vel)
        m_t = max(self.m0 - self.m_dot * t_rel, 0.01)
        f = 0.5 * self.rho * v_mag ** 2 * self.c * self.s
        a = f / m_t
        a_vec = -a * (self.vel / v_mag)
        self.pos += self.vel * dt + 0.5 * a_vec * dt ** 2
        self.vel += a_vec * dt
        self.history.append(self.pos.copy())

    # def get_intensity(self, t):  #根据当前时间 t 和释放时刻 t_release，计算诱饵弹的红外辐射强度
    #     t_rel = t - self.release_time
    #     if t_rel < 0 or t_rel > self.t3:
    #         return 0.0
    #     if t_rel <= self.t1:
    #         return self.I_max * t_rel / self.t1
    #     elif t_rel <= self.t2:
    #         return self.I_max
    #     elif t_rel <= self.t3:
    #         return (self.t3 - t_rel) * self.I_max / (self.t3 - self.t2)
    #     else:
    #         return 0.0

    def get_intensity(self, t):
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return 0.0
        else:
            # τ 控制衰减速度，越小越快
            tau = 1.5
            return self.I_max * np.exp(-t_rel / tau)

class FlareManager:
    def __init__(self, flare_per_group=6, interval=0.1, release_speed=50):
        self.flare_per_group = flare_per_group  #每一组包含6
        self.interval = interval  #每隔0.1秒释放一次
        self.release_speed = release_speed  # 相对释放速度
        self.flares = []  #  当前存在的所有诱饵弹
        self.schedule = []  #计划投放清单（等待释放）

    def release_flare_group(self, t_start):
        for i in range(self.flare_per_group):
            t_i = t_start + i * self.interval
            self.schedule.append((t_i, None, None))

    def update(self, t, dt, aircraft_state):
        # === 判断当前有哪些诱饵弹需要释放 ===
        newly_released = [(ti, i) for i, (ti, _, _) in enumerate(self.schedule) if abs(t - ti) < dt / 2]

        for _, idx in newly_released:
            ti, _, _ = self.schedule[idx]

            env = AirCombatEnv()
            # --- 获取飞机当前位置与姿态 ---
            pos_now = aircraft_state[:3].copy()
            vel_now = env.compute_velocity_vector_from_target(aircraft_state)
            theta = aircraft_state[4]  # 俯仰角
            psi = aircraft_state[5]  # 偏航角

            # --- 机体坐标系下的“后下方”方向向量 ---
            v_b = np.array([-1.0, -1.0, 0.0])  # 后+下，在你定义的机体坐标：x前，y上，z右
            v_b /= np.linalg.norm(v_b)

            # --- 构造方向余弦矩阵，将v_b从机体系变换到惯性系（北-天-东） ---
            R_pitch = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            R_yaw = np.array([
                [np.cos(psi), 0, np.sin(psi)],
                [0, 1, 0],
                [-np.sin(psi), 0, np.cos(psi)]
            ])
            R_bn = R_yaw @ R_pitch  # 总旋转矩阵
            back_down = R_bn @ v_b  # 释放方向在惯性系下的表达

            # --- 合成诱饵弹初始速度 ---
            v_rel = self.release_speed * back_down
            v0 = vel_now + v_rel

            # --- 创建诱饵弹对象并加入当前诱饵弹列表 ---
            self.flares.append(Flare(pos_now, v0, ti))

        # === 清理已释放完的计划条目 ===
        self.schedule = [(ti, pos, vel) for (ti, pos, vel) in self.schedule if ti > t]

        # === 更新所有在空中的诱饵弹轨迹 ===
        for flare in self.flares:
            flare.update(t, dt)
