# 文件: tacview_interface.py

import socket
import numpy as np
from typing import List

# (中文) 导入我们之前创建的类，以便进行类型提示
from .AircraftJSBSim_DirectControl import Aircraft
from .missile import Missile
from .decoys import Flare


class TacviewInterface:
    """
    <<< 多导弹更改 >>>
    封装了与 Tacview 实时遥测功能交互的所有逻辑。
    此版本已适配多目标（多导弹）场景。
    """

    def __init__(self, host="localhost", port=42674, lon0=-155.400, lat0=18.800):
        """
        初始化接口并尝试连接到 Tacview。
        """
        self.host = host
        self.port = port
        self.lon0 = lon0
        self.lat0 = lat0
        self.is_connected = False
        self.client_socket = None

        # --- <<< 多导弹更改 >>> 对象ID管理 ---
        self.aircraft_id = "101"
        # 将单个 missile_id 改为 missile_base_id，用于动态生成ID
        self.missile_base_id = 200
        self.flare_base_id = 300
        self.explosion_id = "901"

        self.tacview_final_frame_sent = False

        try:
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {self.host}, 端口: {self.port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # <<< 核心修改：添加此行以允许地址重用 >>>
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

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

    def stream_frame(self, t_global: float, t_episode: float, aircraft: Aircraft, missiles: List[Missile],
                     flares: List[Flare]):
        """
        <<< 多导弹更改 (V4 - 时间域修正版) >>>
        发送遥测帧。使用全局时间作为时间戳，使用回合内时间来判断逻辑。

        Args:
            t_global (float): 全局的、连续的Tacview时间，用于ACMI文件的时间戳。
            t_episode (float): 当前回合内的相对时间 (从0开始)，用于逻辑判断。
            aircraft (Aircraft): 飞机对象。
            missiles (List[Missile]): 包含所有导弹的列表 (不过滤)。
            flares (List[Flare]): 诱饵弹列表。
        """
        if self.tacview_final_frame_sent:
            return

        # 使用全局时间作为ACMI文件的时间戳
        data_str = f"#{t_global:.2f}\n"

        # 1. 飞机数据 (逻辑不变)
        lon_t, lat_t, alt_t = self._pos_to_lon_lat_alt(aircraft.pos)
        theta_t, psi_t, phi_t = aircraft.attitude_rad
        roll_t, pitch_t, yaw_t = np.rad2deg([phi_t, theta_t, psi_t])
        data_str += (
            f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f}|{roll_t:.2f}|{pitch_t:.2f}|{yaw_t:.2f},"
            f"Name=F-16,Color=Red,Type=Aircraft\n"
        )

        # 2. <<< 核心修改 >>> 导弹数据 - 使用 t_episode 进行逻辑判断
        for i, missile in enumerate(missiles):
            # 使用回合内的相对时间来判断导弹是否应该显示
            if t_episode >= missile.launch_time:
                missile_id = f"{self.missile_base_id + i + 1}"

                lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(missile.pos)
                _, theta_m, psi_m = missile.state_vector[0:3]
                pitch_m, yaw_m = np.rad2deg(theta_m), np.rad2deg(psi_m)

                data_str += (
                    f"{missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}|0.00|{pitch_m:.2f}|{yaw_m:.2f},"
                    f"Name=AIM-9X-{i + 1},Color=Blue,Type=Missile\n"
                )

        # 3. 诱饵弹数据 (逻辑不变)
        for i, flare in enumerate(flares):
            lon_f, lat_f, alt_f = self._pos_to_lon_lat_alt(flare.pos)
            flare_id = f"{self.flare_base_id + i + 1}"
            data_str += (
                f"{flare_id},T={lon_f:.8f}|{lat_f:.8f}|{alt_f:.1f},"
                f"Name=Flare,Color=Orange,Type=Decoy+Flare\n"
            )

        self._send(data_str)

    def stream_explosion(self, t_explosion: float, aircraft_pos: np.ndarray, missile_pos: np.ndarray,
                         missile_id_str: str):
        """
        <<< 多导弹更改 >>>
        发送一个描述爆炸事件的最终帧。
        新增参数 `missile_id_str` 来指定是哪个导弹爆炸了。
        """
        data_str = f"#{t_explosion:.2f}\n"

        # 1. 飞机 (在爆炸瞬间的位置)
        lon_t, lat_t, alt_t = self._pos_to_lon_lat_alt(aircraft_pos)
        data_str += f"{self.aircraft_id},T={lon_t:.8f}|{lat_t:.8f}|{alt_t:.1f},Name=F-16,Color=Red,Type=Aircraft\n"

        # 2. <<< 多导弹更改 >>> 爆炸的导弹 (使用传入的ID)
        lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(missile_pos)
        data_str += f"{missile_id_str},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f},Name=AIM-9X,Color=Blue,Type=Missile\n"

        # 3. 爆炸特效
        lon_e, lat_e, alt_e = lon_m, lat_m, alt_m
        data_str += (
            f"{self.explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
            f"Event=Explosion,Type=Misc+Explosion,Color=Yellow,"
            f"Info=\"t*={t_explosion:.3f}s,Radius=12m\"\n"
        )

        self._send(data_str)
        self.tacview_final_frame_sent = True

    def end_of_episode(self, t_now: float, flare_count: int, missile_count: int, any_missile_exploded: bool):
        """
        <<< 多导弹更改 >>>
        发送清理指令，以在 Tacview 中移除上一回合的所有对象。
        新增参数 `missile_count`，用于移除所有导弹。
        """
        data_str = f"#{t_now:.2f}\n"

        # 移除飞机 (逻辑不变)
        data_str += f"-{self.aircraft_id}\n"

        # <<< 多导弹更改 >>> 移除所有导弹 - 循环处理
        # 简化处理：不再发送 'Destroyed' 事件，直接移除对象
        for i in range(missile_count):
            missile_id = f"{self.missile_base_id + i + 1}"
            data_str += f"-{missile_id}\n"

        # 移除所有诱饵弹 (逻辑不变)
        for i in range(flare_count):
            flare_id = f"{self.flare_base_id + i + 1}"
            data_str += f"-{flare_id}\n"

        # 移除爆炸特效 (逻辑不变)
        data_str += f"-{self.explosion_id}\n"

        self._send(data_str)