import socket
import json
import struct
import math
import afsim  # 这是 AFSim 内置的库，只有在 AFSim 运行时才存在

# 配置
HOST = '127.0.0.1'
PORT = 9000
DT = 0.04  # 仿真步长，必须与 RL 环境一致

# 获取平台对象引用
platform_name = "F16_Blue"

# 建立 TCP 服务器
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"[AFSim Server] Listening on {HOST}:{PORT}...")
print("[AFSim Server] Waiting for RL Client...")

# 阻塞等待 RL 代码连接
conn, addr = server.accept()
print(f"[AFSim Server] Connected by {addr}")

# 辅助函数：经纬度转局部直角坐标 (简单平坦地球近似，用于 RL 观测)
# 注意：更严谨的做法是使用 pyproj，但 AFSim 内部可能没有安装第三方库
# 这里以 (22.0, 121.0) 为原点
ORIGIN_LAT = 22.0
ORIGIN_LON = 121.0
R_EARTH = 6371000.0


def geo_to_xyz(lat, lon, alt):
    dlat = math.radians(lat - ORIGIN_LAT)
    dlon = math.radians(lon - ORIGIN_LON)
    x = dlat * R_EARTH  # North (米)
    z = dlon * R_EARTH * math.cos(math.radians(ORIGIN_LAT))  # East (米)
    y = alt  # Up (米)
    # 转换为 NUE: [North, Up, East]
    return [x, y, z]


def get_state_dict(platform):
    # 获取 AFSim 原始数据
    lat, lon, alt = platform.get_position_lla()
    vn, ve, vd = platform.get_velocity_ned()  # North, East, Down
    # 姿态 (Roll, Pitch, Yaw) - 需确认 AFSim API 返回的是弧度还是度
    # 假设 get_attitude 返回 (psi, theta, phi) 弧度
    psi, theta, phi = platform.get_attitude()
    p, q, r = platform.get_body_angular_velocity()

    # 转换坐标系
    pos_nue = geo_to_xyz(lat, lon, alt)
    vel_nue = [vn, -vd, ve]  # NED -> NUE

    # 获取导弹信息 (简化逻辑：遍历场景中所有红方导弹)
    missiles = []
    # 注意：afsim.get_all_platforms() 的具体 API 需查阅文档
    # 这里假设可以通过名字过滤
    all_plats = afsim.get_all_platforms()
    for p in all_plats:
        if p.get_side() == "Red" and "Missile" in p.get_type():
            m_lat, m_lon, m_alt = p.get_position_lla()
            m_vn, m_ve, m_vd = p.get_velocity_ned()
            missiles.append({
                "pos_nue": geo_to_xyz(m_lat, m_lon, m_alt),
                "vel_nue": [m_vn, -m_vd, m_ve]
            })

    return {
        "time": afsim.get_sim_time(),
        "game_over": False,  # 你可以添加逻辑判断是否被击中
        "blue_survived": True,
        "f16": {
            "pos_nue": pos_nue,
            "vel_nue": vel_nue,
            "attitude_rad": [theta, phi, psi],  # Pitch, Roll, Yaw
            "omega": [p, q, r]
        },
        "missiles": missiles
    }


def send_json(data):
    msg = json.dumps(data).encode('utf-8')
    # 4字节大端长度头
    conn.sendall(struct.pack('>I', len(msg)) + msg)


def recv_json():
    # 接收 4 字节长度
    header = b''
    while len(header) < 4:
        packet = conn.recv(4 - len(header))
        if not packet: return None
        header += packet
    length = struct.unpack('>I', header)[0]

    # 接收内容
    body = b''
    while len(body) < length:
        packet = conn.recv(length - len(body))
        if not packet: return None
        body += packet
    return json.loads(body)


# --- 主循环 ---
try:
    blue_jet = afsim.get_platform(platform_name)

    while True:
        # 1. 接收 RL 指令
        cmd = recv_json()
        if not cmd: break

        if cmd['command'] == "RESET":
            # AFSim 重置逻辑 (通常需要重新加载场景或重置所有实体状态)
            # 简单做法：将飞机移回原点
            afsim.set_sim_time(0.0)
            blue_jet.set_position_lla(22.0, 121.0, 5000.0)
            blue_jet.set_velocity_ned(250.0, 0.0, 0.0)
            blue_jet.set_attitude(0, 0, 0)

            state = get_state_dict(blue_jet)
            send_json(state)

        elif cmd['command'] == "STEP":
            controls = cmd['controls']

            # 【关键】开启 Override 模式，忽略自动驾驶
            # API 可能名为 set_controller_override 或 disable_component("WS_Air_Vehicle_Brain")
            # 此处为示意，具体参考 AFSim Python API 文档
            blue_jet.set_input("throttle", controls['throttle'])
            blue_jet.set_input("elevator", controls['elevator'])  # 弧度
            blue_jet.set_input("aileron", controls['aileron'])  # 弧度
            blue_jet.set_input("rudder", controls['rudder'])  # 弧度

            if cmd['flare']:
                blue_jet.process_event("RELEASE_FLARE")

            # 步进仿真
            afsim.step_simulation(DT)

            state = get_state_dict(blue_jet)
            send_json(state)

except Exception as e:
    print(f"[AFSim Server Error] {e}")
finally:
    conn.close()
    server.close()