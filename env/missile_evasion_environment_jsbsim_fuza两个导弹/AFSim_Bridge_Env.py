import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import math
import time

# 导入你原有的配置，保持维度一致
from Interference_code.PPO_model.PPO_evasion_fuza两个导弹.PPOMLP混合架构.Hybrid_PPO_ATTMLP注意力GRU注意力后yakebi修正优势归一化 import \
    CONTINUOUS_DIM, DISCRETE_DIMS


class AFSimBridgeEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=9000, num_missiles=2):
        super().__init__()

        self.num_missiles = num_missiles
        self.host = host
        self.port = port

        # --- 1. 动作空间 (保持不变) ---
        self.action_space = spaces.Dict({
            "continuous_actions": spaces.Box(
                low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),  # Throttle, Ele, Ail, Rud
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(CONTINUOUS_DIM,)
            ),
            "discrete_actions": spaces.MultiDiscrete(
                np.array([2, DISCRETE_DIMS['salvo_size'], DISCRETE_DIMS['num_groups'], DISCRETE_DIMS['inter_interval']])
            )
        })

        # --- 2. 观测空间 (保持不变) ---
        # 你的定义: 4 * num_missiles + 7
        observation_dim = 4 * self.num_missiles + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        # --- 3. 连接状态 ---
        self.sock = None
        self.t_now = 0.0
        self.obs_cache = None

        # 物理常数 (用于计算 beta 等)
        self.N_infrared = 30
        self.o_ir = 30

    def connect(self):
        """建立 TCP 连接"""
        if self.sock: self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 降低延迟
        print(f"--- 正在连接 AFSim ({self.host}:{self.port}) ...")
        try:
            self.sock.connect((self.host, self.port))
            print("--- AFSim 连接成功 ---")
        except ConnectionRefusedError:
            raise Exception("无法连接到 AFSim！请确保 AFSim 已启动并正在监听端口。")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sock is None: self.connect()

        # 发送重置指令
        cmd = {
            "command": "RESET",
            "seed": int(seed) if seed else 0
        }
        self._send_json(cmd)

        # 接收初始状态
        init_state = self._recv_json()
        self.t_now = 0.0
        self.o_ir = self.N_infrared

        obs = self._parse_state_to_obs(init_state)
        return obs, {}

    def step(self, action_dict):
        # 1. 解析动作
        cont = action_dict["continuous_actions"]
        # PPO 输出是 [-1, 1]，需要转换成 AFSim 需要的物理角度 (弧度)
        # 假设 F-16 物理限制：升降舵25度，副翼21.5度，方向舵30度
        deg2rad = math.pi / 180.0

        throttle = float(np.clip(cont[0], 0.0, 1.0))
        elevator_rad = float(cont[1]) * 25.0 * deg2rad
        aileron_rad = float(cont[2]) * 21.5 * deg2rad
        rudder_rad = float(cont[3]) * 30.0 * deg2rad

        flare_trig = int(action_dict["discrete_actions"][0])

        # 2. 构造发送包
        cmd = {
            "command": "STEP",
            "dt": 0.04,  # 你的仿真步长
            "controls": {
                "throttle": throttle,
                "elevator": elevator_rad,
                "aileron": aileron_rad,
                "rudder": rudder_rad
            },
            "flare": (flare_trig == 1)
        }

        # 3. 发送并接收
        self._send_json(cmd)
        state_data = self._recv_json()

        # 4. 处理返回数据
        self.t_now = state_data.get("time", self.t_now + 0.04)
        obs = self._parse_state_to_obs(state_data)

        # 简单奖励计算 (因为不能调用你原来的 RewardCalculator)
        # 这里只是一个占位，你需要根据 state_data['missiles'] 里的距离来计算
        reward = self._simple_reward(state_data)

        done = state_data.get("game_over", False)
        # 增加超时判定
        truncated = (self.t_now > 60.0)

        info = {"success": state_data.get("blue_survived", False)}

        return obs, reward, done, truncated, info

    def _parse_state_to_obs(self, data):
        """将 AFSim 数据转换为 PPO 观测向量"""
        # 数据结构假设:
        # data['f16'] = {'pos': [x,y,z], 'vel': [vx,vy,vz], 'att': [pitch, roll, yaw], 'omega': [p,q,r]}
        # data['missiles'] = list of dicts

        ac = data['f16']
        ac_pos = np.array(ac['pos'])  # 必须是 NUE (北天东) 坐标系，需要在 AFSim 端转换好
        ac_vel_vec = np.array(ac['vel'])
        ac_vel_mag = np.linalg.norm(ac_vel_vec)

        obs_parts = []

        # 处理导弹
        active_missiles = data.get('missiles', [])
        for i in range(self.num_missiles):
            if i < len(active_missiles):
                m_data = active_missiles[i]
                m_pos = np.array(m_data['pos'])

                # 计算相对几何 (复用你原来的逻辑)
                R_vec = ac_pos - m_pos
                R_rel = np.linalg.norm(R_vec)

                # 重新实现 _compute_relative_beta2
                # 需要导弹速度矢量
                m_vel = np.array(m_data['vel'])
                # 这里简化处理，你需要确保 AFSim 传回了足够的数据计算 Aspect Angle
                # ... (此处省略复杂的几何计算，直接用归一化距离占位，你需要填入真实逻辑)

                # 示例：简单的归一化填充
                o_dis_norm = np.clip(R_rel / 30000.0 * 2 - 1, -1, 1)
                obs_parts.extend([o_dis_norm, 0.0, 1.0, 0.0])  # 占位: dis, beta_sin, beta_cos, theta
            else:
                # 无效导弹
                obs_parts.extend([1.0, 0.0, 1.0, 0.0])

        # 处理飞机自身
        # 速度归一化
        o_av_norm = 2 * ((ac_vel_mag - 150) / 250) - 1
        # 高度归一化
        o_h_norm = 2 * ((ac_pos[1] - 500) / 14500) - 1
        # 姿态
        pitch = ac['att'][0]
        roll = ac['att'][1]
        o_ae_norm = pitch / (math.pi / 2)
        o_am_sin = math.sin(roll)
        o_am_cos = math.cos(roll)
        o_ir_norm = 2 * (self.o_ir / self.N_infrared) - 1
        o_q_norm = ac['omega'][0]  # roll rate

        aircraft_obs = [o_av_norm, o_h_norm, o_ae_norm, o_am_sin, o_am_cos, o_ir_norm, o_q_norm]

        return np.array(obs_parts + aircraft_obs, dtype=np.float32)

    def _simple_reward(self, data):
        # 简易奖励：存活奖励
        if data.get('game_over', False):
            if data.get('blue_survived', False):
                return 10.0
            else:
                return -10.0
        return 0.1

    def _send_json(self, data):
        msg = json.dumps(data).encode('utf-8')
        # 4字节长度头 + 内容
        self.sock.sendall(len(msg).to_bytes(4, byteorder='big'))
        self.sock.sendall(msg)

    def _recv_json(self):
        head = self._recv_all(4)
        length = int.from_bytes(head, byteorder='big')
        body = self._recv_all(length)
        return json.loads(body.decode('utf-8'))

    def _recv_all(self, n):
        data = b''
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet: raise ConnectionError("Connection closed")
            data += packet
        return data