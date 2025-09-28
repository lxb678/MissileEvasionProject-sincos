# 文件: tacview_interface.py (最终修正版)
# 描述: 结合了您文件的稳定结构和多智能体环境所需的功能。
#       - 提供了环境所需的 stream_multi_object_frame 方法。
#       - 提供了健壮的 end_of_episode 方法，不会因额外参数而崩溃。
#       - 实现了动态ID管理，以支持多架飞机和导弹。

import socket
import numpy as np
from typing import List, Dict


class TacviewInterface:
    """
    封装了与 Tacview 实时遥测功能交互的所有逻辑。
    此版本为多智能体环境的最终修正版。
    """

    def __init__(self, host="localhost", port=42674, lon0=-155.400, lat0=18.800):
        self.host = host
        self.port = port
        self.lon0 = lon0
        self.lat0 = lat0
        self.is_connected = False
        self.client_socket = None
        self.tacview_final_frame_sent = False

        # 动态ID管理系统，以支持多个对象
        self.object_ids: Dict[int, str] = {}
        self.next_id_counter = 100
        self.next_event_id_counter = 10000

        try:
            print(f"请打开 Tacview 高级版 → 记录 → 实时遥测，IP: {self.host}, 端口: {self.port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.client_socket, address = server_socket.accept()
            print(f"Tacview 连接成功: {address}")

            handshake = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake.encode())
            self.client_socket.recv(1024)

            header = "FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2020-04-01T00:00:00Z\n"
            self.client_socket.send(header.encode())
            self.is_connected = True
        except Exception as e:
            print(f"Tacview连接失败: {e}")
            self.is_connected = False

    def _get_or_create_id(self, obj: object) -> str:
        """为持久化对象获取或创建唯一ID"""
        py_id = id(obj)
        if py_id not in self.object_ids:
            self.object_ids[py_id] = str(self.next_id_counter)
            self.next_id_counter += 1
        return self.object_ids[py_id]

    def _get_next_event_id(self) -> str:
        """为一次性事件（如爆炸）生成唯一ID"""
        event_id = str(self.next_event_id_counter)
        self.next_event_id_counter += 1
        return event_id

    def _send(self, data: str):
        if not self.is_connected or not self.client_socket: return
        try:
            self.client_socket.send(data.encode())
        except Exception:
            self.is_connected = False
            self.client_socket = None

    def _pos_to_lon_lat_alt(self, pos: np.ndarray) -> tuple:
        if pos is None: return self.lon0, self.lat0, 0
        x_nue, y_nue, z_nue = pos
        lon = self.lon0 + np.rad2deg(z_nue / 6378137.0 / np.cos(np.deg2rad(self.lat0)))
        lat = self.lat0 + np.rad2deg(x_nue / 6371000.0)
        alt = y_nue
        return lon, lat, alt

    # --- 关键修改 1: 提供环境正在调用的正确方法名 ---
    def stream_multi_object_frame(self, t_now: float, aircraft_list: List, missile_list: List, flare_list: List):
        """
        发送一个包含所有活动对象列表的遥测帧。
        这个方法名匹配多智能体环境的需求。
        """
        if self.tacview_final_frame_sent: return
        data_str = f"#{t_now:.2f}\n"

        # 处理飞机列表
        for ac in aircraft_list:
            ac_id = self._get_or_create_id(ac)
            lon, lat, alt = self._pos_to_lon_lat_alt(ac.pos)
            if hasattr(ac, 'attitude_rad'):
                theta, psi, phi = ac.attitude_rad
                roll, pitch, yaw = np.rad2deg([phi, theta, psi])
            else:
                roll, pitch, yaw = 0, 0, 0
            name = getattr(ac, 'name', 'Aircraft')
            color = getattr(ac, 'color', 'Green')
            data_str += f"{ac_id},T={lon:.8f}|{lat:.8f}|{alt:.1f}|{roll:.2f}|{pitch:.2f}|{yaw:.2f},Name={name},Color={color},Type=Air+FixedWing\n"

        # 处理导弹列表
        for m in missile_list:
            m_id = self._get_or_create_id(m)
            lon, lat, alt = self._pos_to_lon_lat_alt(m.pos)
            _, theta, psi = m.state_vector[0:3]
            pitch, yaw = np.rad2deg(theta), np.rad2deg(psi)
            name = getattr(m, 'name', 'Missile')
            color = getattr(m, 'color', 'Orange')
            data_str += f"{m_id},T={lon:.8f}|{lat:.8f}|{alt:.1f}|0.00|{pitch:.2f}|{yaw:.2f},Name={name},Color={color},Type=Weapon+Missile\n"

        # 处理诱饵弹列表
        for flare in flare_list:
            if flare.get_intensity(t_now) > 1e-3:
                f_id = self._get_or_create_id(flare)
                lon, lat, alt = self._pos_to_lon_lat_alt(flare.pos)
                data_str += f"{f_id},T={lon:.8f}|{lat:.8f}|{alt:.1f},Name=Flare,Color=Yellow,Type=Decoy+Flare\n"

        self._send(data_str)

    def stream_explosion(self, t_explosion: float, aircraft_pos: np.ndarray, missile_pos: np.ndarray):
        """发送一个自我清理的爆炸效果。"""
        explosion_id = self._get_next_event_id()
        lon_e, lat_e, alt_e = self._pos_to_lon_lat_alt(missile_pos)
        self._send(
            f"#{t_explosion:.2f}\n{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},Type=Misc+Explosion,Name=Hit\n")
        self._send(f"#{t_explosion + 2.0:.2f}\n-{explosion_id}\n")

    # --- 关键修改 2: 提供环境正在调用的正确方法名和健壮的签名 ---
    def end_of_episode(self, t_now: float, **kwargs):
        """
        在回合结束时清理所有持久化对象。
        使用 **kwargs 接收并忽略所有额外的、可能导致崩溃的参数。
        """
        if not self.object_ids: return
        print(f"Tacview: Cleaning up {len(self.object_ids)} persistent objects...")

        data_str = f"#{t_now:.2f}\n"
        for tacview_id in self.object_ids.values():
            data_str += f"-{tacview_id}\n"

        self._send(data_str)
        self.object_ids.clear()
        self.next_id_counter = 100
        self.tacview_final_frame_sent = False