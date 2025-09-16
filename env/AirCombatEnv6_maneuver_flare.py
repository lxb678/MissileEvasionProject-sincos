import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import socket
import time
import keyboard  # 导入键盘控制库


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================== Tacview 接口类 ==================
class Tacview(object):
    def __init__(self):
        try:
            host = "localhost"
            port = 42674
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {host}, 端口: {port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(5)
            self.client_socket, address = server_socket.accept()
            print(f"Tacview 连接成功: {address}")
            handshake = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake.encode())
            self.client_socket.recv(1024)
            header = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2020-04-01T00:00:00Z\n"
            self.client_socket.send(header.encode())
        except Exception as e:
            print(f"Tacview连接失败: {e}")
            self.client_socket = None

    def send(self, data: str):
        if self.client_socket:
            try:
                self.client_socket.send(data.encode())
            except Exception as e:
                print(f"Tacview 发送失败: {e}")
                self.client_socket = None

class AirCombatEnv:
    def __init__(self,tacview_enabled=False):
        self.g = 9.81
        self.N = 5  # 导引律参数
        self.ny_max = 30  # 导弹最大过载
        self.nz_max = 30  # 导弹最大过载
        self.R_kill = 12  # 导弹杀伤半径
        self.dt_dec = 0.1 # 决策步长
        self.dt_normal = 0.1 # 大步长
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

        # Tacview 集成
        self.tacview_enabled = tacview_enabled
        self.tacview = None
        self.lon0 = 116.4074  # 参考经度 (北京)
        self.lat0 = 39.9042  # 参考纬度 (北京)
        if self.tacview_enabled:
            self.tacview = Tacview()

        # 状态标志（在 __init__ 中）
        self.episode_count = 0
        self.aircraft_id = "101"  # 固定 ID
        self.missile_id = "201"  # 固定 ID
        self.flare_base_id = 300  # 诱饵弹的基础 ID
        self.explosion_id = "901"  # 为爆炸特效指定一个固定ID

        # Tacview 状态
        self.missile_exploded = False
        self.missile_explosion_time = None
        self.missile_explosion_pos = None
        self.target_explosion_pos = None
        self.tacview_final_frame_sent = False  # 新增：标记是否已发送最终帧
        self.t_now = 0
        self.tacview_now = 0

        # --- NEW METHOD ---
    # def _send_tacview_data(self):
    #     """格式化并发送当前帧的所有对象数据到Tacview"""
    #     if not self.tacview_enabled or not self.tacview:
    #         return
    #
    #     data_str = f"#{self.t_now:.2f}\n"
    #
    #     # 1. 飞机数据 (Object ID 101)
    #     x_t, y_t, z_t, _, theta_t, psi_t, phi_t = self.x_target_now
    #     lon_t = self.lon0 + z_t / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #     lat_t = self.lat0 + x_t / 110574.0
    #     alt_t = y_t
    #     roll_t, pitch_t, yaw_t = np.rad2deg([phi_t, theta_t, psi_t])
    #     # 1. 飞机数据 (使用固定 ID "101")
    #     data_str += (f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f}|{roll_t:.2f}|{pitch_t:.2f}|{yaw_t:.2f},"
    #                  f"Name=F-16,Color=Red,Type=Aircraft\n")
    #
    #     # 2. 导弹数据 (Object ID 201)
    #     _, theta_m, psi_m, x_m, y_m, z_m = self.x_missile_now
    #     lon_m = self.lon0 + z_m / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #     lat_m = self.lat0 + x_m / 110574.0
    #     alt_m = y_m
    #     roll_m, pitch_m, yaw_m = 0.0, np.rad2deg(theta_m), np.rad2deg(psi_m)
    #     # 2. 导弹数据 (使用固定 ID "201")
    #     data_str += (f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}|{roll_m:.2f}|{pitch_m:.2f}|{yaw_m:.2f},"
    #                  f"Name=AIM-9X,Color=Blue,Type=Missile\n")
    #
    #     # 3. 诱饵弹数据 (Object IDs 301, 302, ...)
    #     for i, flare in enumerate(self.flare_manager.flares):
    #         x_f, y_f, z_f = flare.pos
    #         lon_f = self.lon0 + z_f / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #         lat_f = self.lat0 + x_f / 110574.0
    #         alt_f = y_f
    #         flare_id = f"3{i + 1:02d}"  # ID 例如 301, 302, ...
    #         data_str += (f"{flare_id},T={lon_f:.8f}|{lat_f:.8f}|{alt_f:.1f},"
    #                      f"Name=Flare,Color=Orange,Type=Decoy+Flare\n")
    #
    #     # 4. 导弹爆炸事件 (Event ID 901)
    #     if getattr(self, "missile_exploded", False):
    #         x_e, y_e, z_e = self.missile_explosion_pos
    #         lon_e = self.lon0 + z_e / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #         lat_e = self.lat0 + x_e / 110574.0
    #         alt_e = y_e
    #         blast_radius = 12.0  # 爆炸半径（米）
    #
    #         # === 爆炸事件行 ===
    #         data_str += (
    #             f"901,T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
    #             f"Event=Explosion,Type=Misc+Explosion,Color=Yellow,"
    #             f"Info=\"t*={self.missile_explosion_time:.3f}s, "
    #             f"Radius={blast_radius:.1f}m\"\n"
    #         )
    #
    #         # === 爆炸时刻的飞机位置 (Object ID 101) ===
    #         x_t, y_t, z_t = self.target_explosion_pos
    #         lon_t = self.lon0 + z_t / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #         lat_t = self.lat0 + x_t / 110574.0
    #         alt_t = y_t
    #         roll_t, pitch_t, yaw_t = np.rad2deg([phi_t, theta_t, psi_t])
    #         data_str += (
    #             f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f},"
    #             f"Name=F-16,Color=Red,Type=Aircraft\n"
    #         )
    #
    #         # === 爆炸时刻的导弹位置 (Object ID 201) ===
    #         x_m, y_m, z_m = self.missile_explosion_pos
    #         lon_m = self.lon0 + z_m / (111320.0 * np.cos(np.deg2rad(self.lat0)))
    #         lat_m = self.lat0 + x_m / 110574.0
    #         alt_m = y_m
    #         roll_m, pitch_m, yaw_m = 0.0, np.rad2deg(theta_m), np.rad2deg(psi_m)
    #         data_str += (
    #             f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f},"
    #             f"Name=AIM-9X,Color=Blue,Type=Missile\n"
    #         )
    #
    #         data_str += f"{self.aircraft_id},Remove=true\n"
    #         data_str += f"{self.missile_id},Remove=true\n"
    #         # 移除所有诱饵弹
    #         for i in range(50):
    #             self.tacview.send(f"3{i + 1:02d},Remove=true\n")
    #         self.missile_exploded = False
    #     self.tacview.send(data_str)
    # --- NEW METHOD END ---

    def _send_tacview_data(self):
        if not self.tacview_enabled or not self.tacview:
            return

        # 如果最终帧已发送，则不再发送任何数据
        if self.tacview_final_frame_sent:
            return

        data_str = ""

        # --- 爆炸帧 ---
        if self.missile_exploded:
            self.tacview_final_frame_sent = True
            data_str += f"#{self.missile_explosion_time:.2f}\n"

            # 飞机（爆炸时刻）
            x_t, y_t, z_t = self.target_explosion_pos
            lon_t = self.lon0 + z_t / (111320.0 * np.cos(np.deg2rad(self.lat0)))
            lat_t = self.lat0 + x_t / 110574.0
            alt_t = y_t
            data_str += f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f},Name=F-16,Color=Red,Type=Aircraft\n"

            # 导弹
            x_m, y_m, z_m = self.missile_explosion_pos
            lon_m = self.lon0 + z_m / (111320.0 * np.cos(np.deg2rad(self.lat0)))
            lat_m = self.lat0 + x_m / 110574.0
            alt_m = y_m
            data_str += f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f},Name=AIM-9X,Color=Blue,Type=Missile\n"

            # 爆炸特效
            lon_e, lat_e, alt_e = lon_m, lat_m, alt_m
            data_str += (
                f"{self.explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
                f"Event=Explosion,Type=Misc+Explosion,Color=Yellow,"
                f"Info=\"t*={self.missile_explosion_time:.3f}s,Radius=12m\"\n"
            )

        # --- 常规帧 ---
        else:
            data_str += f"#{self.tacview_now:.2f}\n"

            # 飞机
            x_t, y_t, z_t, _, theta_t, psi_t, phi_t = self.x_target_now
            lon_t = self.lon0 + z_t / (111320.0 * np.cos(np.deg2rad(self.lat0)))
            lat_t = self.lat0 + x_t / 110574.0
            alt_t = y_t
            roll_t, pitch_t, yaw_t = np.rad2deg([phi_t, theta_t, psi_t])
            data_str += (
                f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f}|{roll_t:.2f}|{pitch_t:.2f}|{yaw_t:.2f},"
                f"Name=F-16,Color=Red,Type=Aircraft\n"
            )

            # 导弹
            _, theta_m, psi_m, x_m, y_m, z_m = self.x_missile_now
            lon_m = self.lon0 + z_m / (111320.0 * np.cos(np.deg2rad(self.lat0)))
            lat_m = self.lat0 + x_m / 110574.0
            alt_m = y_m
            pitch_m, yaw_m = np.rad2deg(theta_m), np.rad2deg(psi_m)
            data_str += (
                f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}|0.00|{pitch_m:.2f}|{yaw_m:.2f},"
                f"Name=AIM-9X,Color=Blue,Type=Missile\n"
            )

            # 诱饵
            for i, flare in enumerate(self.flare_manager.flares):
                x_f, y_f, z_f = flare.pos
                lon_f = self.lon0 + z_f / (111320.0 * np.cos(np.deg2rad(self.lat0)))
                lat_f = self.lat0 + x_f / 110574.0
                alt_f = y_f
                flare_id = f"{self.flare_base_id + i + 1}"
                data_str += (
                    f"{flare_id},T={lon_f:.8f}|{lat_f:.8f}|{alt_f:.1f},"
                    f"Name=Flare,Color=Orange,Type=Decoy+Flare\n"
                )

        if data_str:
            self.tacview.send(data_str)

    def reset(self):
        """
        开始新一回合仿真。
        使用 Event=LeftArea 来优雅地终止上一回合的对象。
        """
        cleanup_time = self.tacview_now
        remove_str = f"#{cleanup_time:.2f}\n"

        # 用 LeftArea 事件标明是“离开区域/清理”，然后真正移除（前缀 '-'）
        remove_str += f"0,Event=LeftArea|{self.aircraft_id}|\n"
        remove_str += f"-{self.aircraft_id}\n"

        # 导弹：如果想标为被击中/爆炸可先发 Destroyed/Timeout 事件，否则也用 LeftArea
        # （如果上一回合导弹已爆炸，可能先发送 Destroyed/Timeout 会更语义化）
        if getattr(self, "missile_exploded", False):
            remove_str += f"0,Event=Destroyed|{self.missile_id}|\n"
        remove_str += f"-{self.missile_id}\n"

        # 移除上一回合中所有实际存在的诱饵弹（同样用 '-' 前缀）
        for i, _ in enumerate(self.flare_manager.flares):
            flare_id = f"{self.flare_base_id + i + 1}"
            remove_str += f"-{flare_id}\n"

        # 移除爆炸特效（无论是否发生过爆炸，都尝试移除）
        remove_str += f"-{self.explosion_id}\n"

        self.tacview.send(remove_str)

        # --- 在 Tacview 清理完成后，再重置所有内部状态 ---

        # 回合计数 +1
        self.episode_count += 1

        # 时间与奖励清零
        self.t_now = 0
        self.done = False
        self.reward = 0.0

        # 状态标志位复位
        self.missile_exploded = False
        self.tacview_final_frame_sent = False
        self.missile_explosion_reported = False
        self.missile_explosion_time = None
        self.missile_explosion_pos = None
        self.target_explosion_pos = None

        # 诱饵清空 (现在在这里清空是安全的)
        self.flare_manager.flares = []


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

        "初始化更加随机"
        y_t = np.random.uniform(5000, 10000)
        R = np.random.uniform(9000, 11000)
        theta1 = np.deg2rad(np.random.uniform(-180, 180))
        x_m = R * (-np.cos(theta1))  # 导弹初始x坐标
        z_m = R * np.sin(theta1)  # 导弹初始z坐标
        y_m = y_t + np.random.uniform(-2000, 2000)

        x_target_theta = np.deg2rad(np.random.uniform(-30, 30))  # 设定飞机初始俯仰角
        x_target_phi = np.deg2rad(0)
        self.x_target_now = np.array([0, y_t, 0, np.random.uniform(200, 400), x_target_theta,
                                      x_target_phi, 0])  # 飞机状态[x, y, z, V, theta, psi, phi]
        self.x_missile_now = np.array([np.random.uniform(700, 900), 0, 0, x_m, y_m,
                                       z_m])  # 导弹状态 [V, theta, psi_c, x, y, z]
        Rx = self.x_target_now[0] - self.x_missile_now[3]
        Ry = self.x_target_now[1] - self.x_missile_now[4]
        Rz = self.x_target_now[2] - self.x_missile_now[5]
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        self.theta_L = np.arcsin(Ry / R)
        self.phi_L = np.arctan2(Rz, Rx)
        # print(self.phi_L, np.rad2deg(self.phi_L))

        # x_missile_theta = np.random.uniform(0, self.theta_L * 2)  # 导弹初始倾角 :随机产生，处于 0度 和 弹目倾角 之间
        # x_missile_psi_c = np.random.uniform(self.phi_L + np.deg2rad(0), self.phi_L - np.deg2rad(0))
        # print(x_missile_psi_c, np.rad2deg(x_missile_psi_c))
        # self.x_missile_now[1] = x_missile_theta
        # self.x_missile_now[2] = x_missile_psi_c

        # 确保导弹初始方向精确指向目标
        self.x_missile_now[1] = self.theta_L  # 导弹俯仰角 = 弹目视线俯仰角
        self.x_missile_now[2] = self.phi_L  # 导弹偏航角 = 弹目视线偏航角


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

        # --- MODIFICATION START ---
        # 开始时，发送数据到Tacview
        self._send_tacview_data()
        # --- MODIFICATION END ---

        return self.observation


    def step(self, action):
        nx = action[0]  # 飞机水平速度
        nz = action[1]  # 飞机垂直速度
        phi2 = action[2]  # 飞机偏航角
        release_flare = action[3]  #  是否释放红外诱饵弹
        # open_laser = action[1]  # 是否开启激光定向干扰
        self.reward = 0
        # release_flare = 0  #  是否释放红外诱饵
        # open_laser = 0  # 是否开启激光定向干扰
        # 在每个 step 中记录行为（建议在 run_one_step 或 step 中执行一次）
        self.history_flare.append(release_flare)
        # self.history_laser.append(open_laser)

        self.interference_success = False  # 激光定向干扰是否成功
        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

        if R_rel < self.R_switch:
            for _ in range(int(round(self.dt / self.dt_small))):
                # self.run_one_step(self.dt_small, release_flare, open_laser)
                self.run_one_step(self.dt_small, nx, nz, phi2, release_flare)
                release_flare = 0
                if self.done:
                    break

        else:
            if release_flare or self.flare_manager.schedule:
                # print("self.t_now", self.t_now)
                # print('self.flare_manager.schedule', self.flare_manager.schedule)
                for _ in range(int(round(self.dt / self.dt_flare))):
                    # print("int(self.dt / self.dt_flare", int(round(self.dt / self.dt_flare)))
                    # self.run_one_step(self.dt_flare, release_flare, open_laser)
                    self.run_one_step(self.dt_flare, nx, nz, phi2, release_flare)
                    release_flare = 0
                    if self.done:
                        break
            else:
                # self.run_one_step(self.dt, release_flare, open_laser)
                self.run_one_step(self.dt, nx, nz, phi2, release_flare)
        # print("self.x_missile_now", self.x_missile_now)
        # print("self.x_target_now", self.x_target_now)

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

            # self.reward += self.compute_dense_reward_3(open_laser)   #只有红外时关闭
            reward_4 = 0
        else:
            self.reward += self.compute_dense_reward_1()
            self.reward += self.compute_dense_reward_2()
            # self.reward += self.compute_dense_reward_3(open_laser)   #只有红外时关闭
            # self.reward3 = self.compute_dense_reward_3(open_laser)
            # print("self.compute_dense_reward_3()", self.compute_dense_reward_3(open_laser))
            # self.reward += self.compute_dense_reward_4(action[0])
            # self.reward += self.compute_dense_reward_4(action[0])
            reward_4 = self.compute_dense_reward_4(action[0]) * 1.0 / 5.0 # 红外干扰资源使用过量惩罚函数
            # reward_4 = 0
            self.reward /= 5.0

            # if self.compute_dense_reward_4(action[0]) != 0:
            #     reward_4=self.compute_dense_reward_4(action[0])
            # else:
            #     reward_4 =0




        return self.observation, self.reward, self.done, reward_4, 1

        # return self.observation, self.reward, self.done,0, 1

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

    # def render(self):
    #     # ========================= 可视化 =========================
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(self.Y[:, 3], self.Y[:, 5], self.Y[:, 4], 'b-', label='导弹轨迹')
    #     ax.plot(self.Xt[:, 0], self.Xt[:, 2], self.Xt[:, 1], 'r--', label='目标轨迹')
    #     ax.scatter(self.Y[0, 3], self.Y[0, 5], self.Y[0, 4], color='g', label='导弹起点')
    #     ax.scatter(self.Y[self.idx_min, 3], self.Y[self.idx_min, 5], self.Y[self.idx_min, 4], color='m', label='最近点')
    #
    #     for i, flare in enumerate(self.flare_manager.flares):
    #         traj = np.array(flare.history)
    #         if traj.shape[0] > 0:
    #             ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color='orange', linewidth=1,
    #                     label='红外诱饵弹' if i == 0 else "")
    #
    #     ax.set_xlabel('X / m')
    #     ax.set_ylabel('Z / m')
    #     ax.set_zlabel('Y / m')
    #     ax.invert_yaxis()
    #     ax.legend()
    #     ax.set_title('导引头模型 + 红外诱饵弹建模仿真')
    #     ax.grid()
    #     ax.view_init(elev=20, azim=-150)
    #     plt.show()
    def render(self):
        # ========================= 可视化 =========================

        # <<<--- 关键修复：在这里将列表转换为Numpy数组 ---<<<
        # 确保无论仿真如何结束，这里的数据都是正确的数组格式
        self.Xt = np.array(self.Xt)
        self.Y = np.array(self.Y)

        # <<<--- 增加健壮性检查：如果轨迹为空，则不绘图 ---<<<
        if len(self.Xt) == 0 or len(self.Y) == 0:
            print("警告：轨迹数据为空，无法进行可视化。")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 现在这两行代码可以安全地执行了
        ax.plot(self.Y[:, 3], self.Y[:, 5], self.Y[:, 4], 'b-', label='导弹轨迹')
        ax.plot(self.Xt[:, 0], self.Xt[:, 2], self.Xt[:, 1], 'r--', label='目标轨迹')

        ax.scatter(self.Y[0, 3], self.Y[0, 5], self.Y[0, 4], color='g', label='导弹起点')

        # 只有在仿真有结果时（idx_min被赋值），才绘制最近点
        if hasattr(self, 'idx_min') and self.idx_min < len(self.Y):
            ax.scatter(self.Y[self.idx_min, 3], self.Y[self.idx_min, 5], self.Y[self.idx_min, 4], color='m',
                       label='最近点')

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

    def export_to_acmi(self, filename="simulation.acmi", lat0=60.0, lon0=120.0):
        """
        导出 Tacview 2.1 ACMI 文件 (含诱饵)
        坐标系: 北-天-东
        飞机: F-16
        导弹: AIM-9X
        诱饵: Flare (Name=Flare1, Type=Decoy+Flare)
        """
        import math
        from collections import defaultdict

        with open(filename, "w", encoding="utf-8") as f:
            # === 文件头 ===
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

            if len(self.Xt) == 0 or len(self.Y) == 0:
                print("⚠️ 轨迹为空，未能导出 ACMI")
                return

            # 初始参考点 (以飞机初始点为原点)
            x0, y0, z0 = self.Xt[0][:3]

            # === 构建每个时间点的对象字典 ===
            objects_by_time = defaultdict(list)

            # 飞机 & 导弹
            for k, t_now in enumerate(self.t):
                # 飞机 A0100
                xn, yu, ze = self.Xt[k][:3]
                lat = lat0 + (xn - x0) / 111000.0
                lon = lon0 + (ze - z0) / (111000.0 * math.cos(math.radians(lat0)))
                alt = yu
                objects_by_time[t_now].append(("A0100", "F16", lon, lat, alt, "Red"))

                # 导弹 A0102
                xm, ym, zm = self.Y[k][3:6]
                lat = lat0 + (xm - x0) / 111000.0
                lon = lon0 + (zm - z0) / (111000.0 * math.cos(math.radians(lat0)))
                alt = ym
                objects_by_time[t_now].append(("A0102", "AIM-9X", lon, lat, alt, "Blue"))

            # 诱饵弹
            for i, flare in enumerate(self.flare_manager.flares):
                times_written = set()  # 同一 flare 记录过的时间点
                for pos in flare.history:
                    if len(pos) < 4:
                        continue
                    xn, yu, ze, t_rel = pos[:4]
                    if t_rel in times_written:
                        continue  # 已写过这个时间点
                    times_written.add(t_rel)

                    lat = lat0 + (xn - x0) / 111000.0
                    lon = lon0 + (ze - z0) / (111000.0 * math.cos(math.radians(lat0)))
                    alt = yu
                    flare_id = f"A01{10 + i}"
                    objects_by_time[t_rel].append(
                        (flare_id, f"Flare{i + 1}", lon, lat, alt, "Orange", "Decoy+Flare")
                    )

            # === 按时间排序写入 ACMI 文件 ===
            for t_now in sorted(objects_by_time.keys()):
                f.write(f"#{t_now:.2f}\n")
                for obj in objects_by_time[t_now]:
                    if len(obj) == 6:
                        obj_id, name, lon, lat, alt, color = obj
                        f.write(f"{obj_id},T={lon}|{lat}|{alt},Name={name},Color={color}\n")
                    else:
                        obj_id, name, lon, lat, alt, color, type_str = obj
                        f.write(f"{obj_id},T={lon}|{lat}|{alt},Name={name},Type={type_str},Color={color}\n")

        print(f"✅ ACMI 文件已导出: {filename}")

    def run_one_step(self, dt, nx, nz, phi2, release_flare):

        self.x_target_next = self.aircraft_dynamics2(self.x_target_now, dt, nx, nz, phi2)

        if release_flare == 1:
            if self.o_ir > 0:
                self.o_ir -= 1  # 红外诱饵弹数量
                self.flare_manager.release_flare_group(self.t_now)

        self.flare_manager.update(self.t_now, dt, self.x_target_now)

        R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
        R_rel = np.linalg.norm(R_vec)

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

            # === 1. 过载计算 ===
            ny = self.N * V * theta_L_dot / g + np.cos(theta)
            nz = self.N * V * phi_L_dot / g

            n_total_cmd = np.sqrt(ny ** 2 + nz ** 2)
            n_max = 50
            if n_total_cmd > n_max:
                scale = n_max / n_total_cmd
                ny *= scale
                nz *= scale

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

            def get_Cx_AIM9X(Ma: float) -> float:
                """
                近似计算 AIM-9X 导弹的阻力系数 (Cx) 随马赫数变化
                数据基于典型空空导弹经验模型
                """

                if Ma <= 0.9:
                    # 亚音速平稳阻力
                    return 0.20

                elif Ma <= 1.2:
                    # 跨声速阻力线性上升到峰值 0.38
                    return 0.20 + (0.38 - 0.20) * (Ma - 0.9) / 0.3

                elif Ma <= 4.0:
                    # 超声速指数衰减到 0.18
                    return 0.38 * np.exp(-0.35 * (Ma - 1.2)) + 0.15

                else:
                    # 高超声速趋稳
                    return 0.15

            Cx = get_Cx_AIM9X(Ma)
            # Cx = 0
            S = np.pi * (0.127) ** 2 / 4
            q = 0.5 * rho * V ** 2
            F_drag = Cx * q * S  # 阻力大小
            # a_drag = -F_drag / m * v_now / (np.linalg.norm(v_now) + 1e-6)

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
                [np.cos(theta_next) * np.cos(psi_c_next), np.sin(theta_next), np.cos(theta_next) * np.sin(psi_c_next)])
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

        # # 状态更新
        self.t_now += dt
        self.tacview_now += dt
        self.t.append(self.t_now)
        self.Xt.append(self.x_target_next.copy())
        self.Y.append(self.x_missile_next.copy())

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

                    # 计算爆炸在插值时刻的位置（对导弹和目标分别插值）
                    # 导弹位置在 t_star:
                    tau = (t_star - t1) / dt if dt != 0 else 0.0
                    missile_pos_at_tstar = M1 + tau * (M1_M2)
                    target_pos_at_tstar = T1 + tau * (T1_T2)
                    # 记录爆炸信息（以导弹位置为爆点）
                    self.missile_exploded = True
                    self.missile_explosion_time = self.tacview_now - t2 + t_star
                    self.missile_explosion_pos = missile_pos_at_tstar.copy()
                    self.target_explosion_pos = target_pos_at_tstar.copy()

                    # 记录脱靶量
                    print(f">>> 引信引爆！脱靶量 = {self.miss_distance:.2f} m")
                    self.Xt = np.array(self.Xt)
                    self.Y = np.array(self.Y)
                    self.t = np.array(self.t)
                    self.idx_min = len(self.t) - 2

                    # 注意：此处不马上把 missile_explosion_reported 设为 True，
                    # 我们希望 _send_tacview_data 在下一次被调用时生成 Event 并移除对象，
                    # 但也可以直接调用一次 _send_tacview_data() 以立即上报（见下方）。

                    # 立即上报一次 Tacview 事件（可选）
                    # 我们把爆炸时间与位置写入事件文本，便于在 Tacview 中定位
                    # _send_tacview_data 内会检测 self.missile_exploded 和 missile_explosion_reported
                    # 并把 Event/移除行发送出去 (参见你已有的实现)
                    self._send_tacview_data()



        self.x_target_now = self.x_target_next
        self.x_missile_now = self.x_missile_next
        self.prev_theta_L = self.theta_L
        self.prev_phi_L = self.phi_L

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
        # print(self.t_now,self.missile_exploded)
        # --- MODIFICATION START ---
        # 在每一步物理更新后，发送数据到Tacview
        if self.missile_exploded == False:
            self._send_tacview_data()
        # --- MODIFICATION END ---

        # ========================= 飞机动力学模型 =========================

    def aircraft_dynamics2(self, state, dt, nx, nz, phi2):
        # ================== 坐标系定义 ==================
        # 惯性坐标系 (Inertial Frame):
        #   - 外部状态表示: 北-天-东 (North-Up-East, NUE) -> 用于 state 向量和绘图
        #   - 内部物理计算: 北-东-地 (North-East-Down, NED) -> 用于核心物理引擎
        # 机体坐标系 (Body Frame): 前-右-下 (Forward-Right-Down, FRD)
        # =========================================================

        # ================== 四元数与旋转矩阵函数 (保持不变) ==================
        # 这些函数依然基于 NED/FRD 的欧拉角定义 (phi-roll, theta-pitch, psi-yaw)
        def normalize(q):
            norm = np.linalg.norm(q)
            if norm < 1e-9: return np.array([1.0, 0.0, 0.0, 0.0])
            return q / norm

        def euler_to_quaternion(phi, theta, psi):
            cy = np.cos(psi * 0.5)
            sy = np.sin(psi * 0.5)
            cp = np.cos(theta * 0.5)
            sp = np.sin(theta * 0.5)
            cr = np.cos(phi * 0.5)
            sr = np.sin(phi * 0.5)
            q0 = cr * cp * cy + sr * sp * sy
            q1 = sr * cp * cy - cr * sp * sy
            q2 = cr * sp * cy + sr * cp * sy
            q3 = cr * cp * sy - sr * sp * cy
            return normalize(np.array([q0, q1, q2, q3]))

        def quaternion_to_rotation_matrix(q):
            q0, q1, q2, q3 = q
            # 从机体系(FRD)到惯性系(NED)的旋转矩阵 R_frd_to_ned
            return np.array([
                [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)]
            ])

        def quaternion_to_continuous_euler(q):
            R = quaternion_to_rotation_matrix(q)
            sin_theta = -R[2, 0]
            theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
            cos_theta = np.cos(theta)
            if np.abs(cos_theta) < 1e-6:
                phi = 0.0
                psi = np.arctan2(-R[0, 1], R[1, 1])
            else:
                psi = np.arctan2(R[1, 0], R[0, 0])
                phi = np.arctan2(R[2, 1], R[2, 2])
            return phi, theta, psi

        g = 9.81
        m = 1.0

        # MODIFIED: 状态向量定义更新为 北-天-东 (NUE)
        # 状态向量: [x(北), y(上), z(东), Vt, theta(俯仰), psi(偏航), phi(滚转)]
        # 初始高度 5000 米，在 NUE 中 y 坐标为 5000

        # ================== 主循环 (状态NUE, 物理NED/FRD) ==================

        # === 解包状态向量 (NUE 格式) ===
        x_nue, y_nue, z_nue, Vt, theta, psi, phi = state

        # === 坐标转换: 从 NUE (状态) 到 NED (物理) ===
        # pos_ned = [North, East, Down]
        # x_ned = x_nue
        # y_ned = z_nue
        # z_ned = -y_nue
        pos_ned = np.array([x_nue, z_nue, -y_nue])

        # === 转换 (欧拉角到旋转矩阵) ===
        q_current = euler_to_quaternion(phi, theta, psi)
        R_frd_to_ned = quaternion_to_rotation_matrix(q_current)
        R_ned_to_frd = R_frd_to_ned.T

        # --- 核心物理计算 ---
        # 1. 获取控制指令
        # --- 滚转控制 ---
        phi_cmd = phi2
        nx_cmd = nx
        nz_cmd = nz

        # --- 将过载指令转换为力 ---
        thrust_minus_drag = nx_cmd * m * g
        lift = nz_cmd * m * g

        # 对升力进行限制，模拟飞机结构极限
        # lift = np.clip(lift, -3.0 * m * g, 9.0 * m * g)

        # 2. MODIFIED: 在机体系(FRD: 前-右-下)中定义气动力
        # 升力 (lift) 产生向上的力, 在 FRD 的 Z 轴 (向下) 是负方向
        F_aero_frd = np.array([thrust_minus_drag, 0, -lift])
        V_vec_frd = np.array([Vt, 0, 0])  # 速度矢量在机体系中总是指向前方

        # 3. 转换到惯性系 (NED)
        F_aero_ned = R_frd_to_ned @ F_aero_frd
        V_vec_ned = R_frd_to_ned @ V_vec_frd

        # 4. MODIFIED: 在 NED 中加入重力
        # 重力指向地心, 即 NED 的 Z 轴正方向
        F_gravity_ned = np.array([0, 0, m * g])
        F_total_ned = F_aero_ned + F_gravity_ned

        # 5. 计算航迹角速度
        if Vt < 1e-3:
            omega_ned = np.array([0.0, 0.0, 0.0])
        else:
            # F_total_ned 中垂直于 V_vec_ned 的分力导致了速度方向的改变
            F_perp_ned = F_total_ned - np.dot(F_total_ned, V_vec_ned / Vt) * (V_vec_ned / Vt)
            # a_perp = F_perp / m, R = V^2 / a_perp, omega = V / R = a_perp / V
            omega_ned = np.cross(V_vec_ned, F_total_ned) / (m * Vt * Vt + 1e-8)

        # 6. 转换回机体系 (FRD)
        omega_frd = R_ned_to_frd @ omega_ned

        # 7. 混合控制与角速度最终确定
        # 滚转率 p (绕 X 轴) 由直接控制决定
        tau_p = 0.1
        p_body_cmd = (1 / tau_p) * (phi_cmd - phi)
        max_roll_rate = np.deg2rad(240)
        p_body = np.clip(p_body_cmd, -max_roll_rate, max_roll_rate)

        # MODIFIED: 俯仰率 q (绕 Y 轴) 和 偏航率 r (绕 Z 轴) 由动力学决定
        q_body = omega_frd[1]  # Pitch rate is omega_y
        r_body = omega_frd[2]  # Yaw rate is omega_z

        # --- 动力学与运动学积分 (仍然在 NED/FRD 框架下) ---
        p, q, r = p_body, q_body, r_body
        dq_dt = 0.5 * np.array([
            -q_current[1] * p - q_current[2] * q - q_current[3] * r,
            q_current[0] * p + q_current[2] * r - q_current[3] * q,
            q_current[0] * q - q_current[1] * r + q_current[3] * p,
            q_current[0] * r + q_current[1] * q - q_current[2] * p
        ])
        q_new = normalize(q_current + dt * dq_dt)
        phi_new, theta_new, psi_new = quaternion_to_continuous_euler(q_new)

        acceleration_ned = F_total_ned / m
        V_unit_vec_ned = V_vec_ned / Vt if Vt > 1e-3 else np.array([1., 0., 0.])
        Vt_new = Vt + dt * np.dot(acceleration_ned, V_unit_vec_ned)

        # 计算 NED 坐标下的位移
        d_pos_ned = (R_frd_to_ned @ np.array([Vt, 0, 0])) * dt
        pos_ned_new = pos_ned + d_pos_ned

        # === 坐标转换: 从 NED (物理) 回到 NUE (状态) ===
        # x_nue = x_ned
        # y_nue = -z_ned
        # z_nue = y_ned
        x_nue_new = pos_ned_new[0]
        y_nue_new = -pos_ned_new[2]
        z_nue_new = pos_ned_new[1]

        # === 更新整个 state 向量 (NUE 格式) ===
        state2 = np.zeros_like(state)
        state2[0] = x_nue_new
        state2[1] = y_nue_new
        state2[2] = z_nue_new
        state2[3] = Vt_new
        state2[4:] = [theta_new, psi_new, phi_new]

        return state2
    def aircraft_dynamics(self, t, x, control_input, g):
        xt, yt, zt, Vt, theta, psi = x
        n_tx, n_ty, mu_t = control_input(x)
        dxt = Vt * np.cos(theta) * np.cos(psi)
        dyt = Vt * np.sin(theta)
        dzt = Vt * np.cos(theta) * np.sin(psi)
        dVt = g * (n_tx - np.sin(theta))
        dtheta = (g / Vt) * (n_ty * np.cos(mu_t) - np.cos(theta))
        dpsi = (n_ty * g * np.sin(mu_t)) / (Vt * np.cos(theta))
        # print(np.array([dxt, dyt, dzt, dVt, dtheta, dpsi]))
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

    # def compute_velocity_vector_from_target(self, x):
    @staticmethod
    def compute_velocity_vector_from_target(x):
        V, theta, psi = x[3], x[4], x[5]
        vx = V * np.cos(theta) * np.cos(psi)
        vy = V * np.sin(theta)
        vz = V * np.cos(theta) * np.sin(psi)
        return np.array([vx, vy, vz])

    def compute_velocity_vector(self, x):
        V, theta, psi = x[0], x[1], x[2]
        vx = V * np.cos(theta) * np.cos(psi)
        vy = V * np.sin(theta)
        vz = V * np.cos(theta) * np.sin(psi)
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


# ========================= 红外诱饵弹类 =========================
class Flare:
    def __init__(self, position, velocity, release_time, m0=0.5, m_dot=0.01,
                 rho=1.225, c=0.5, s=0.01, I_max=10000, t1=0.5, t2=3.5, t3=5.0):
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
        # history 存 numpy array [x, y, z, t]
        self.history = [np.array([self.pos[0], self.pos[1], self.pos[2], release_time])]
    def update(self, t, dt):  #更新位置与速度
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return
        v_mag = np.linalg.norm(self.vel)
        m_t = max(self.m0 - self.m_dot * t_rel, 0.01)
        f = 0.5 * self.rho * v_mag ** 2 * self.c * self.s
        a = f / m_t
        a_vec = -a * (self.vel / v_mag)

        g_vec = np.array([0, -9.81, 0])  # y 轴向下
        a_vec += g_vec

        self.pos += self.vel * dt + 0.5 * a_vec * dt ** 2
        self.vel += a_vec * dt
        # 存储 numpy array
        self.history.append(np.array([self.pos[0], self.pos[1], self.pos[2], t]))

    def get_intensity(self, t):  #根据当前时间 t 和释放时刻 t_release，计算诱饵弹的红外辐射强度
        t_rel = t - self.release_time
        if t_rel < 0 or t_rel > self.t3:
            return 0.0
        if t_rel <= self.t1:
            return self.I_max * t_rel / self.t1
        elif t_rel <= self.t2:
            return self.I_max
        elif t_rel <= self.t3:
            return (self.t3 - t_rel) * self.I_max / (self.t3 - self.t2)
        else:
            return 0.0

    # def get_intensity(self, t):
    #     t_rel = t - self.release_time
    #     if t_rel < 0 or t_rel > self.t3:
    #         return 0.0
    #     else:
    #         # τ 控制衰减速度，越小越快
    #         tau = 1.5
    #         return self.I_max * np.exp(-t_rel / tau)

class FlareManager:
    def __init__(self, flare_per_group=1, interval=0.1, release_speed=50):
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

            # env = AirCombatEnv()
            # --- 获取飞机当前位置与姿态 ---
            pos_now = aircraft_state[:3].copy()
            # vel_now = env.compute_velocity_vector_from_target(aircraft_state)
            vel_now = AirCombatEnv.compute_velocity_vector_from_target(aircraft_state)
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


# ========================= 主执行循环 (键盘控制) =========================
if __name__ == '__main__':
    # --- 1. 初始化环境，启用Tacview ---
    # 将 tacview_enabled 设置为 True
    env = AirCombatEnv(tacview_enabled=True)
    obs = env.reset()
    done = False

    print("\n--- 飞机控制 ---")
    print("W: 拉杆 (增大法向过载)")
    print("S: 推杆 (减小/负法向过载)")
    print("A: 向左滚转")
    print("D: 向右滚转")
    print("Shift: 加速 (增大切向过载)")
    print("Ctrl: 减速 (减小切向过载)")
    print("f: 投放红外诱饵弹")
    print("Q: 退出仿真")
    print("------------------")

    # --- 2. 主循环 ---
    while not done:
        # --- 2.1. 检测键盘输入并生成控制指令 ---

        # 默认指令: 1g平飞，保持速度，不滚转
        nx = 0.0  # 切向过载
        nz = 1.0  # 法向过载
        phi_cmd = env.x_target_now[6]  # 保持当前滚转角
        release_flare = 0

        # 检测按键
        if keyboard.is_pressed('w'): nz = 9.0
        if keyboard.is_pressed('s'): nz = -2.0
        if keyboard.is_pressed('a'): phi_cmd -= np.deg2rad(60)   # 按住时持续滚转
        if keyboard.is_pressed('d'): phi_cmd += np.deg2rad(60)   # 按住时持续滚转
        if keyboard.is_pressed('shift'): nx = 2.0
        if keyboard.is_pressed('ctrl'): nx = -1.0
        if keyboard.is_pressed('f'): release_flare = 1
        if keyboard.is_pressed('q'):
            print("用户退出仿真。")
            break

        # 将指令打包为action
        action = [nx, nz, phi_cmd, release_flare]

        # --- 2.2. 执行仿真步 ---
        obs, reward, done, _, _ = env.step(action)

        # 短暂休眠以匹配决策步长，并降低CPU占用率
        # time.sleep(env.dt_small)
        time.sleep(0.1)
    # --- 3. 仿真结束处理 ---
    print("\n仿真结束!")
    if env.success:
        print(f"飞机成功规避！最终脱靶量: {env.miss_distance:.2f} m")
    else:
        print(f"飞机被击落！最终脱靶量: {env.miss_distance:.2f} m")

    # 显示matplotlib轨迹图
    # env.render()
