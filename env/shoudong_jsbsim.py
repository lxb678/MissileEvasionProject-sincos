from missile_evasion_environment_jsbsim.missile_evasion_environment_jsbsim import AirCombatEnv
import keyboard
import time
import numpy as np

# ========================= 主执行循环 (键盘控制 JSBSim) =========================
if __name__ == '__main__':
    # --- 1. 初始化环境，启用Tacview ---
    env = AirCombatEnv(tacview_enabled=True)
    obs = env.reset()
    done = False

    print("\n--- 飞机控制 (JSBSim Direct Control) ---")
    print("  油门控制:")
    print("    Shift:  增加油门")
    print("    Ctrl:   减小油门")
    print("  俯仰控制 (升降舵):")
    print("    S:      拉杆 (机头向上)")
    print("    W:      推杆 (机头向下)")
    print("  滚转控制 (副翼):")
    print("    A:      向左滚转")
    print("    D:      向右滚转")
    print("  偏航控制 (方向舵, 可选):")
    print("    Z:      向左偏航")
    print("    C:      向右偏航")
    print("  其他:")
    print("    F:      投放红外诱饵弹")
    print("    Q:      退出仿真")
    print("------------------------------------------")

    # --- 2. 初始化动作变量 ---
    # 这些变量现在代表舵面指令，范围通常是 [-1, 1] 或 [0, 1]
    throttle_cmd = 0.8  # 初始油门设置在 80%
    elevator_cmd = 0.0  # 升降舵居中
    aileron_cmd = 0.0  # 副翼居中
    rudder_cmd = 0.0  # 方向舵居中

    # --- 3. 主循环 ---
    while not done:
        # --- 3.1. 检测键盘输入并更新控制指令 ---

        # --- 油门控制 ---
        if keyboard.is_pressed('shift'):
            throttle_cmd = min(1.0, throttle_cmd + 0.05)  # 逐渐增加油门
        if keyboard.is_pressed('ctrl'):
            throttle_cmd = max(0.0, throttle_cmd - 0.05)  # 逐渐减小油门

        # --- 俯仰控制 (升降舵) ---
        # 注意：在航空中，"拉杆"(stick back) 是 elevator < 0，使机头向上
        # "推杆"(stick forward) 是 elevator > 0，使机头向下
        # 这里为了符合 "W=前进/向上" 的游戏习惯，做了反向映射
        elevator_cmd = 0.0
        if keyboard.is_pressed('s'): elevator_cmd = -0.5  # 拉杆
        if keyboard.is_pressed('w'): elevator_cmd = 0.5  # 推杆

        # --- 滚转控制 (副翼) ---
        aileron_cmd = 0.0
        if keyboard.is_pressed('a'): aileron_cmd = -0.5  # 向左滚转
        if keyboard.is_pressed('d'): aileron_cmd = 0.5  # 向右滚转

        # --- 偏航控制 (方向舵) ---
        rudder_cmd = 0.0
        if keyboard.is_pressed('z'): rudder_cmd = -0.5  # 向左偏航
        if keyboard.is_pressed('c'): rudder_cmd = 0.5  # 向右偏航

        # --- 诱饵弹投放 ---
        release_flare = 1.0 if keyboard.is_pressed('f') else 0.0

        # --- 退出 ---
        if keyboard.is_pressed('q'):
            print("用户退出仿真。")
            break

        # 将所有指令打包为最终的 action 向量
        action = [
            throttle_cmd,
            elevator_cmd,
            aileron_cmd,
            rudder_cmd,
            release_flare
        ]

        # 打印当前动作，方便调试
        # print(f"Action: T:{action[0]:.2f} E:{action[1]:.2f} A:{action[2]:.2f} R:{action[3]:.2f} F:{int(action[4])}")

        # --- 3.2. 执行仿真步 ---
        obs, reward, done, _, _ = env.step(action)

        # 短暂休眠以匹配决策步长 (dt_dec)，并降低CPU占用率
        time.sleep(env.dt_dec)

    # --- 4. 仿真结束处理 ---
    print("\n仿真结束!")
    if env.success:
        print(f"飞机成功规避！最终脱靶量: {env.miss_distance:.2f} m")
    else:
        # 如果是引信引爆，也算被击落
        if env.missile_exploded:
            print(f"飞机被击落 (引信引爆)！最终脱靶量: {env.miss_distance:.2f} m")
        else:
            print(f"飞机被击落 (其他原因)！最终脱靶量: {env.miss_distance:.2f} m")

    # 显示matplotlib轨迹图
    # env.render()