# 文件名: Run_AFSIM_Bridge.py
import random, socket, time, os, sys
import numpy as np
import torch
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.Config import *
from Interference_code.PPO_model.PPO_evasion_fuza_单一干扰.PPOMLP混合架构.Hybrid_PPO_混合架构雅可比修正优势归一化 import \
    PPO_discrete
from Interference_code.env.missile_evasion_environment_jsbsim_fuza_单一干扰.Vec_missile_evasion_environment_jsbsim import \
    AirCombatEnv


class LocalAfsimInterface:
    def __init__(self, ip="127.0.0.1", send_port=8888, recv_port=9999, lon0=-155.400, lat0=18.800):
        self.ip, self.send_port, self.recv_port = ip, send_port, recv_port
        self.lon0, self.lat0 = lon0, lat0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SIO_UDP_CONNRESET'): self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        self.sock.bind(("0.0.0.0", self.recv_port))
        self.sock.settimeout(10.0)  # 步进同步模式下，超时可以设长一点

    def handshake(self):
        print(f"[Python] 等待 AFSIM 握手...")
        while True:
            try:
                self.sock.sendto(b"PING", (self.ip, self.send_port))
                data, _ = self.sock.recvfrom(1024)
                if "READY" in data.decode():
                    print("[Python] 握手成功! AFSIM 已同步。")
                    return True
            except socket.timeout:
                pass

    def _nue_to_lla(self, pos_nue):
        x, y, z = pos_nue
        lat = self.lat0 + x / 110574.0
        lon = self.lon0 + z / (111320.0 * np.cos(np.deg2rad(self.lat0)))
        return lon, lat, y

    def send_and_wait(self, env_instance, ep_time, global_offset, flare_released):
        abs_t = global_offset + ep_time

        # --- 1. 提取 F16 数据 ---
        ac = env_instance.aircraft
        lon_a, lat_a, alt_a = self._nue_to_lla(ac.pos)
        pitch_rad, yaw_rad, roll_rad = ac.attitude_rad

        # 封装 F16 字符串
        # 顺序: Name,Lon,Lat,Alt,Yaw,Pitch,Roll,Spd,Time,Fire
        data_f16 = (f"F16,{lon_a:.6f},{lat_a:.6f},{alt_a:.1f},"
                    f"{np.rad2deg(yaw_rad):.2f},{np.rad2deg(pitch_rad):.2f},{np.rad2deg(roll_rad):.2f},"
                    f"{ac.velocity:.1f},{abs_t:.3f},{1 if flare_released else 0}")

        # --- 2. 提取 AIM9X 数据 ---
        msl = env_instance.missile
        lon_m, lat_m, alt_m = self._nue_to_lla(msl.pos)
        v_m, pit_m, yaw_m = msl.state_vector[0:3]
        data_msl = (f"AIM9X,{lon_m:.6f},{lat_m:.6f},{alt_m:.1f},"
                    f"{np.rad2deg(yaw_m):.2f},{np.rad2deg(pit_m):.2f},0.00,"
                    f"{v_m:.1f},{abs_t:.3f},0")

        # --- 3. 【打印发送信息】 ---
        print(f"\n[PY -> SEND] T_abs:{abs_t:.3f}")
        print(
            f"   F16   -> LLA:({lat_a:.4f},{lon_a:.4f},{alt_a:.1f}) HPR:({np.rad2deg(yaw_rad):.1f},{np.rad2deg(pitch_rad):.1f},{np.rad2deg(roll_rad):.1f}) V:{ac.velocity:.1f}")
        print(
            f"   AIM9X -> LLA:({lat_m:.4f},{lon_m:.4f},{alt_m:.1f}) HPR:({np.rad2deg(yaw_m):.1f},{np.rad2deg(pit_m):.1f},0.0) V:{v_m:.1f}")

        # --- 4. 发送并同步 ---
        full_msg = f"{data_f16}|{data_msl}"
        self.sock.sendto(full_msg.encode(), (self.ip, self.send_port))

        try:
            # 等待 AFSIM ACK
            data, _ = self.sock.recvfrom(1024)
            print(f"[PY <- RECV] AFSIM ACK: {data.decode()}")
        except socket.timeout:
            print("[PY] ERROR: AFSIM Timeout!")


if __name__ == "__main__":
    env = AirCombatEnv(tacview_enabled=False, dt=0.02)
    model_path = r"D:\code\规避导弹项目sincos\save\save_evade_fuza_单一干扰\PPO_2026-01-05_18-55-08"
    agent = PPO_discrete(load_able=True, model_dir_path=model_path)
    agent.prep_eval_rl()

    bridge = LocalAfsimInterface()
    bridge.handshake()

    global_clock = 0.0
    for i_episode in range(1):
        obs, info = env.reset(seed=AGENTPARA.RANDOM_SEED + i_episode)
        done, ep_start = False, global_clock
        print(f"\n>>>> 回合 {i_episode + 1} 开始 (AFSIM 起始时间: {ep_start:.2f}s) <<<<")

        # --- 【关键修复 1】: 发送 T=0 的初始状态包，对齐 AFSIM 起点 ---
        # 此时没有 action_indices，flare_released 设为 False
        bridge.send_and_wait(env, 0.0, ep_start, False)

        while not done:
            with torch.no_grad():
                action_indices, _, _ = agent.choose_action(obs, deterministic=True)

            # env.step 会让 env.t_now 增加 (如从 0 变到 0.02)
            obs, reward, terminated, truncated, info = env.step({"discrete_actions": action_indices.astype(int)})
            done = terminated or truncated

            # 发送更新后的状态
            bridge.send_and_wait(env, env.t_now, ep_start, (action_indices[0] == 1))

        # 【注意】建议 global_clock 严格累加，不要加 5.0，或者确保 AFSIM 也能跳跃
        global_clock += env.t_now + 0.2