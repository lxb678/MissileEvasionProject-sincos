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

        # # 动态ID管理系统，以支持多个对象
        # self.object_ids: Dict[int, str] = {}
        self.next_id_counter = 100
        self.next_event_id_counter = 10000

        # 动态ID管理系统
        self.object_ids: Dict[int, str] = {}
        # --- <<< 核心修正 1: 分开管理持久对象和一次性事件 >>> ---
        # self.next_id_counter 用于飞机、导弹等

        # self.next_event_id_counter 用于爆炸等，这些ID需要在回合结束时被清理

        # 新增一个列表来存储本回合创建的所有事件ID
        self.episode_event_ids: List[str] = []

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
        """为一次性事件（如爆炸）生成唯一ID，并记录下来以便清理"""
        event_id = str(self.next_event_id_counter)
        self.next_event_id_counter += 1
        # 关键：将生成的事件ID【添加】到列表中
        self.episode_event_ids.append(event_id)
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
    def stream_multi_object_frame(self, global_t: float, episode_t: float,
                                  aircraft_list: List, missile_list: List, flare_list: List):
        """
        发送一个包含所有活动对象列表的遥测帧。

        Args:
            global_t (float): 用于Tacview时间轴的全局连续时间。
            episode_t (float): 用于计算物理属性（如诱饵弹强度）的当前回合时间。
        """
        if self.tacview_final_frame_sent: return
        # 使用 global_t 作为Tacview的时间戳
        data_str = f"#{global_t:.2f}\n"

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
            # --- <<< 核心修正：使用 episode_t 来计算强度 >>> ---
            if flare.get_intensity(episode_t) > 1e-3:
                f_id = self._get_or_create_id(flare)
                lon, lat, alt = self._pos_to_lon_lat_alt(flare.pos)
                data_str += f"{f_id},T={lon:.8f}|{lat:.8f}|{alt:.1f},Name=Flare,Color=Yellow,Type=Decoy+Flare\n"

        self._send(data_str)

    # def stream_explosion(self, t_explosion: float, aircraft_pos: np.ndarray, missile_pos: np.ndarray):
    #     """发送一个自我清理的爆炸效果。"""
    #     explosion_id = self._get_next_event_id()
    #     lon_e, lat_e, alt_e = self._pos_to_lon_lat_alt(missile_pos)
    #     self._send(
    #         f"#{t_explosion:.2f}\n{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},Type=Misc+Explosion,Name=Hit\n")
    #     self._send(f"#{t_explosion + 2.0:.2f}\n-{explosion_id}\n")
    # --- <<< 核心修正：增强 stream_explosion 方法 >>> ---
    # def stream_explosion(self, t_explosion: float,
    #                      aircraft_pos: np.ndarray = None,
    #                      missile_pos: np.ndarray = None,
    #                      is_hit: bool = True,
    #                     destroy_object: object = None): # <<< 新增参数
    #     """
    #     发送一个自我清理的爆炸效果。
    #     (V2: 修改了签名以接受 missile_pos，并能处理其为None的情况)
    #
    #     Args:
    #         t_explosion (float): 爆炸发生的全局时间。
    #         aircraft_pos (np.ndarray, optional): 如果是命中，可以提供飞机位置。
    #         missile_pos (np.ndarray, optional): 爆炸发生的中心位置 (通常是导弹位置)。
    #         is_hit (bool): 标志是真实命中还是导弹自毁。
    #     """
    #     """
    #     发送一个自我清理的爆炸效果，并可选择立即销毁一个关联的对象。
    #
    #     Args:
    #         ...
    #         destroy_object (object, optional): 需要在爆炸后立即从Tacview中移除的对象 (例如导弹)。
    #     """
    #     # --- 确定爆炸中心位置 ---
    #     # 优先级：优先使用 missile_pos，如果未提供，则使用 aircraft_pos
    #     if missile_pos is not None:
    #         explosion_center_pos = missile_pos
    #     elif aircraft_pos is not None:
    #         explosion_center_pos = aircraft_pos
    #     else:
    #         # 如果两个位置都未提供，则无法创建爆炸，直接返回
    #         print("[警告][Tacview] stream_explosion被调用，但未提供任何位置信息。")
    #         return
    #
    #     explosion_id = self._get_next_event_id()
    #     lon_e, lat_e, alt_e = self._pos_to_lon_lat_alt(explosion_center_pos)
    #
    #     # 根据事件类型选择不同的名称和效果 (逻辑不变)
    #     if is_hit:
    #         explosion_name = "Hit"
    #         explosion_type = "Misc+Explosion"
    #     else:
    #         explosion_name = "Missile_End"
    #         explosion_type = "Misc+Explosion"
    #
    #     data_str = f"#{t_explosion:.2f}\n"
    #
    #     # 1. 创建爆炸对象
    #     data_str += (f"{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
    #                  f"Type={explosion_type},Name={explosion_name}\n")
    #
    #     # --- <<< 核心修正：立即删除关联对象 >>> ---
    #     # 2. 如果指定了要销毁的对象，则查找其ID并发送删除指令
    #     # if destroy_object is not None:
    #     #     py_id = id(destroy_object)
    #     #     if py_id in self.object_ids:
    #     #         object_to_destroy_id = self.object_ids[py_id]
    #     #         data_str += f"-{object_to_destroy_id}\n"
    #     #         # 从我们的ID映射中也移除它，防止未来复用
    #     #         del self.object_ids[py_id]
    #     self._send(data_str)
    #
    #     # # 发送创建爆炸对象的帧
    #     # self._send(
    #     #     f"#{t_explosion:.2f}\n"
    #     #     f"{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
    #     #     f"Type={explosion_type},Name={explosion_name}\n"
    #     # )
    #     # 发送2秒后自动删除爆炸对象的帧
    #     # self._send(f"#{t_explosion + 2.0:.2f}\n-{explosion_id}\n")

    # # --- <<< 核心修正：最终的、统一的 stream_explosion 方法 >>> ---
    # def stream_explosion(self, t_explosion: float,
    #                      explosion_pos: np.ndarray,
    #                      is_hit: bool,
    #                      hit_object: object = None,
    #                      destroy_object: object = None,
    #                      hit_object_pos: np.ndarray = None):
    #     """
    #     发送一个统一的爆炸效果。
    #     - 如果是命中 (is_hit=True)，会创建大爆炸，并在爆炸前发送位置修正帧。
    #     - 如果是自毁 (is_hit=False)，会创建小爆炸。
    #     - 总是会销毁 destroy_object (通常是导弹)。
    #
    #     Args:
    #         t_explosion (float): 爆炸发生的精确插值时间。
    #         explosion_pos (np.ndarray): 爆炸发生的中心位置 (导弹的精确位置)。
    #         is_hit (bool): 标志是真实命中还是导弹自毁。
    #         hit_object (object, optional): 【仅命中时提供】被命中的对象 (飞机)。
    #         destroy_object (object, optional): 需要立即销毁的对象 (导弹)。
    #         hit_object_pos (np.ndarray, optional): 【仅命中时提供】被命中对象在碰撞时的精确位置。
    #     """
    #     # --- 步骤 1: (仅命中时) 发送位置修正帧 ---
    #     if is_hit and hit_object is not None and destroy_object is not None and hit_object_pos is not None:
    #         correction_str = f"#{t_explosion:.3f}\n"
    #         missile_id = self._get_or_create_id(destroy_object)
    #         aircraft_id = self._get_or_create_id(hit_object)
    #         lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(explosion_pos)
    #         lon_a, lat_a, alt_a = self._pos_to_lon_lat_alt(hit_object_pos)
    #         correction_str += f"{missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}\n"
    #         correction_str += f"{aircraft_id},T={lon_a:.8f}|{lat_a:.8f}|{alt_a:.1f}\n"
    #         self._send(correction_str)
    #
    #     # --- 步骤 2: 创建爆炸特效并销毁对象 ---
    #     explosion_id = self._get_next_event_id()
    #     lon_e, lat_e, alt_e = self._pos_to_lon_lat_alt(explosion_pos)
    #
    #     if is_hit:
    #         explosion_name = "Hit"
    #         explosion_type = "Misc+Explosion"
    #     else:
    #         explosion_name = "Missile_End"
    #         explosion_type = "Misc+Explosion"
    #
    #     data_str = f"#{t_explosion:.3f}\n"
    #     data_str += (f"{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
    #                  f"Type={explosion_type},Name={explosion_name}\n")
    #
    #     # # 销毁导弹 (总是执行)
    #     # if destroy_object is not None:
    #     #     py_id = id(destroy_object)
    #     #     if py_id in self.object_ids:
    #     #         object_id = self.object_ids[py_id];
    #     #         data_str += f"-{object_id}\n";
    #     #         del self.object_ids[py_id]
    #     #
    #     # # 销毁被命中的飞机 (仅命中时执行)
    #     # if is_hit and hit_object is not None:
    #     #     py_id = id(hit_object)
    #     #     if py_id in self.object_ids:
    #     #         object_id = self.object_ids[py_id];
    #     #         data_str += f"-{object_id}\n";
    #     #         del self.object_ids[py_id]
    #
    #     self._send(data_str)

    # --- <<< 核心修正：最终的、功能完备的 stream_explosion 方法 >>> ---
    def stream_explosion(self, t_explosion: float,
                         explosion_pos: np.ndarray,
                         is_hit: bool,
                         hit_object: object = None,
                         destroy_object: object = None,
                         hit_object_pos: np.ndarray = None):
        """
        发送一个统一的爆炸效果，并在命中时发送精确的位置修正帧。
        所有创建的对象都将在回合结束时被统一清理。
        """
        # --- 步骤 1: (仅命中时) 发送精确位置修正帧 ---
        if is_hit and hit_object is not None and destroy_object is not None and hit_object_pos is not None:
            correction_str = f"#{t_explosion:.3f}\n"  # 使用高精度时间戳
            missile_id = self._get_or_create_id(destroy_object)
            aircraft_id = self._get_or_create_id(hit_object)
            lon_m, lat_m, alt_m = self._pos_to_lon_lat_alt(explosion_pos)
            lon_a, lat_a, alt_a = self._pos_to_lon_lat_alt(hit_object_pos)
            correction_str += f"{missile_id},T={lon_m:.8f}|{lat_m:.8f}|{alt_m:.1f}\n"
            correction_str += f"{aircraft_id},T={lon_a:.8f}|{lat_a:.8f}|{alt_a:.1f}\n"
            self._send(correction_str)

        # --- 步骤 2: 创建持久化的爆炸特效 ---
        # --- 步骤 2: 创建持久化的爆炸特效 ---
        # !!! 关键：为爆炸事件调用【专属的】事件ID生成器 !!!
        explosion_id = self._get_next_event_id()
        # explosion_dummy_object = f"explosion_{t_explosion}_{np.random.rand()}"
        # explosion_id = self._get_or_create_id(explosion_dummy_object)
        lon_e, lat_e, alt_e = self._pos_to_lon_lat_alt(explosion_pos)

        explosion_name = "Hit" if is_hit else "Missile_End"
        explosion_type = "Misc+Explosion"

        data_str = f"#{t_explosion:.3f}\n"  # 使用高精度时间戳
        data_str += (f"{explosion_id},T={lon_e:.8f}|{lat_e:.8f}|{alt_e:.1f},"
                     f"Type={explosion_type},Name={explosion_name}\n")
        self._send(data_str)


    # --- 关键修改 2: 提供环境正在调用的正确方法名和健壮的签名 ---
    def end_of_episode(self, t_now: float, **kwargs):
        """
        在回合结束时清理所有持久化对象【和】所有本回合创建的事件对象。
        """
        # 准备一个用于发送的指令字符串
        data_str = f"#{t_now:.2f}\n"
        something_to_clean = False

        # 1. 清理所有持久化对象 (飞机、剩余的导弹等)
        if self.object_ids:
            # print(f"[DEBUG][Tacview] Cleaning up {len(self.object_ids)} persistent objects...")
            for tacview_id in self.object_ids.values():
                data_str += f"-{tacview_id}\n"
            something_to_clean = True

        # 2. 清理本回合创建的所有事件对象 (爆炸特效)
        if self.episode_event_ids:
            # print(f"[DEBUG][Tacview] Cleaning up {len(self.episode_event_ids)} event objects (explosions)...")
            for event_id in self.episode_event_ids:
                data_str += f"-{event_id}\n"
            something_to_clean = True

        # 3. 如果有任何需要清理的对象，则发送指令
        if something_to_clean:
            self._send(data_str)

        # 4. 重置状态，为下一回合做准备
        self.object_ids.clear()
        self.episode_event_ids.clear()  # <<< 关键：清空事件ID列表

        # (ID计数器保持不清空，以保证全局唯一性)
        # self.next_persistent_id_counter = 100

        self.tacview_final_frame_sent = False