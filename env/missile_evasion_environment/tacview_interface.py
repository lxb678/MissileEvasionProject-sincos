# 文件: tacview_interface.py

import socket
import numpy as np
from typing import List

# (中文) 导入我们之前创建的类，以便进行类型提示
from .aircraft import Aircraft
from .missile import Missile
from .decoys import Flare


class TacviewInterface:
    """
    封装了与 Tacview 实时遥测功能交互的所有逻辑。
    负责连接、格式化数据帧、发送数据。
    """

    def __init__(self, host="localhost", port=42674, lon0=-155.400, lat0=18.800):
        """
        初始化接口并尝试连接到 Tacview。
        """
        self.host = host
        self.port = port
        self.lon0 = lon0  # 参考经度
        self.lat0 = lat0  # 参考纬度
        self.is_connected = False
        self.client_socket = None

        # --- 对象ID管理 ---
        self.aircraft_id = "101"
        self.missile_id = "201"
        self.flare_base_id = 300
        self.explosion_id = "901"

        # (中文) 新增：这个标志位仍然很重要，由主环境控制
        self.tacview_final_frame_sent = False

        try:
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {self.host}, 端口: {self.port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.client_socket, address = server_socket.accept()
            print(f"Tacview 连接成功: {address}")

            # 发送握手协议
            handshake = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake.encode())
            self.client_socket.recv(1024)

            # 发送文件头
            header = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2020-04-01T00:00:00Z\n"
            self.client_socket.send(header.encode())

            self.is_connected = True
        except Exception as e:
            print(f"Tacview连接失败: {e}")
            self.is_connected = False

    def _send(self, data: str):
        """私有的发送方法，包含错误处理。"""
        if not self.is_connected or not self.client_socket:
            return
        try:
            self.client_socket.send(data.encode())
        except Exception as e:
            print(f"Tacview 发送失败: {e}")
            self.is_connected = False
            self.client_socket = None

    def _pos_to_lon_lat_alt(self, pos: np.ndarray) -> tuple:
        """将 NUE 坐标转换为经纬高。"""
        x_nue, y_nue, z_nue = pos
        lon = self.lon0 + z_nue / (111320.0 * np.cos(np.deg2rad(self.lat0)))
        lat = self.lat0 + x_nue / 110574.0
        alt = y_nue
        return lon, lat, alt

    def stream_frame(self, t_now: float, aircraft: Aircraft, missile: Missile, flares: List[Flare]):
        """
        发送一个常规的、包含所有活动对象状态的遥测帧。
        """
        # (中文) 增加检查，如果最终帧已发送，则不再发送
        if self.tacview_final_frame_sent:
            return
        data_str = f"#{t_now:.2f}\n"

        # 1. 飞机数据
        lon_t, lat_t, alt_t = self._pos_to_lon_lat_alt(aircraft.pos)
        theta_t, psi_t, phi_t = aircraft.attitude_rad
        roll_t, pitch_t, yaw_t = np.rad2deg([phi_t, theta_t, psi_t])
        data_str += (
            f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f}|{roll_t:.2f}|{pitch_t:.2f}|{yaw_t:.2f},"
            f"Name=F-16,Color=Red,Type=Aircraft\n"
        )

        # 2. 导弹数据
        lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(missile.pos)
        _, theta_m, psi_m = missile.state_vector[0:3]  # (V, theta, psi)
        pitch_m, yaw_m = np.rad2deg(theta_m), np.rad2deg(psi_m)
        data_str += (
            f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}|0.00|{pitch_m:.2f}|{yaw_m:.2f},"
            f"Name=AIM-9X,Color=Blue,Type=Missile\n"
        )

        # 3. 诱饵弹数据
        for i, flare in enumerate(flares):
            lon_f, lat_f, alt_f = self._pos_to_lon_lat_alt(flare.pos)
            flare_id = f"{self.flare_base_id + i + 1}"  # 为每个诱饵弹生成唯一ID
            data_str += (
                f"{flare_id},T={lon_f:.8f}|{lat_f:.8f}|{alt_f:.1f},"
                f"Name=Flare,Color=Orange,Type=Decoy+Flare\n"
            )

        self._send(data_str)

    def stream_explosion(self, t_explosion: float, aircraft_pos: np.ndarray, missile_pos: np.ndarray):
        """
        发送一个描述爆炸事件的最终帧。
        """
        data_str = f"#{t_explosion:.2f}\n"

        # 1. 飞机 (在爆炸瞬间的位置)
        lon_t, lat_t, alt_t = self._pos_to_lon_lat_alt(aircraft_pos)
        data_str += f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f},Name=F-16,Color=Red,Type=Aircraft\n"

        # 2. 导弹 (在爆炸瞬间的位置)
        lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(missile_pos)
        data_str += f"{self.missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f},Name=AIM-9X,Color=Blue,Type=Missile\n"

        # 3. 爆炸特效
        lon_e, lat_e, alt_e = lon_m, lat_m, alt_m
        data_str += (
            f"{self.explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
            f"Event=Explosion,Type=Misc+Explosion,Color=Yellow,"
            f"Info=\"t*={t_explosion:.3f}s,Radius=12m\"\n"
        )

        self._send(data_str)

        # (中文) 核心：发送完爆炸帧后，设置标志位
        self.tacview_final_frame_sent = True

    def end_of_episode(self, t_now: float, flare_count: int, missile_exploded: bool):
        """
        发送清理指令，以在 Tacview 中移除上一回合的对象。
        """
        data_str = f"#{t_now:.2f}\n"

        # 移除飞机
        data_str += f"-{self.aircraft_id}\n"

        # 移除导弹 (如果上一回合已爆炸，可以先发送一个Destroyed事件)
        if missile_exploded:
            data_str += f"0,Event=Destroyed|{self.missile_id}|\n"
        data_str += f"-{self.missile_id}\n"

        # 移除所有诱饵弹
        for i in range(flare_count):
            flare_id = f"{self.flare_base_id + i + 1}"
            data_str += f"-{flare_id}\n"

        # 移除爆炸特效
        data_str += f"-{self.explosion_id}\n"

        self._send(data_str)