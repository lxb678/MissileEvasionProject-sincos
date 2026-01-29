# AFSim_Server_Script.py
# 在 AFSim 环境中运行此脚本

import socket
import json
import struct
import afsim  # 假设这是 AFSim 的内部 API

# 1. 设置服务器
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9000))
server.listen(1)
print("Waiting for RL Agent connection...")
conn, addr = server.accept()
print(f"Connected to {addr}")

# 获取平台引用
blue_jet = afsim.get_platform("F16_Blue")

while True:
    # 2. 接收指令
    try:
        header = conn.recv(4)
        if not header: break
        length = struct.unpack('>I', header)[0]
        data = json.loads(conn.recv(length))
    except:
        break

    if data['command'] == "RESET":
        afsim.reset_simulation()  # 重置仿真
        # 重新设置初始位置等
        out_state = get_state(blue_jet)
        send_json(conn, out_state)

    elif data['command'] == "STEP":
        controls = data['controls']

        # --- 3. 施加 6DOF 控制 (核心) ---
        # 覆盖自动驾驶仪
        blue_jet.set_controller_override(True)

        # 设置舵面偏转 (AFSim API 可能会有所不同，需查阅文档)
        # 这里的 SetInput 对应 WS_Aerodynamics_6DOF 的通道
        blue_jet.set_input("elevator", controls['elevator'])  # 弧度
        blue_jet.set_input("aileron", controls['aileron'])  # 弧度
        blue_jet.set_input("rudder", controls['rudder'])  # 弧度
        blue_jet.set_input("throttle", controls['throttle'])  # 0-1

        if data['flare']:
            blue_jet.release_flare()

        # 4. 步进仿真
        afsim.step(data['dt'])

        # 5. 获取并返回状态
        out_state = get_state(blue_jet)
        send_json(conn, out_state)


def get_state(platform):
    # 获取数据并转换坐标系 (LatLon -> NUE Meter)
    # 假设 afsim.get_position 返回 (lat, lon, alt)
    # 你需要自己写一个 GeoToLocal 的转换函数
    pos = platform.get_position_xyz()
    vel = platform.get_velocity_xyz()
    att = platform.get_attitude_euler()  # pitch, roll, yaw

    return {
        "time": afsim.get_sim_time(),
        "game_over": False,  # 根据逻辑判断
        "f16": {
            "pos": pos,
            "vel": vel,
            "att": att,
            "omega": platform.get_angular_velocity()
        },
        "missiles": []  # 遍历并填充导弹列表
    }


def send_json(conn, data):
    msg = json.dumps(data).encode('utf-8')
    conn.sendall(struct.pack('>I', len(msg)) + msg)