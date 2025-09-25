from missile_evasion_environment.missile_evasion_environment import AirCombatEnv
import keyboard
import time
import numpy as np

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
        # phi_cmd = env.x_target_now[6]  # 保持当前滚转角
        p_cmd = 0.0  # 滚转角速度
        release_flare = 0

        # 检测按键
        if keyboard.is_pressed('w'): nz = 9.0
        if keyboard.is_pressed('s'): nz = -2.0
        if keyboard.is_pressed('a'): p_cmd -= np.deg2rad(60)   # 按住时持续滚转
        if keyboard.is_pressed('d'): p_cmd = np.deg2rad(60)   # 按住时持续滚转
        if keyboard.is_pressed('shift'): nx = 1.0
        if keyboard.is_pressed('ctrl'): nx = -1.0
        if keyboard.is_pressed('f'): release_flare = 1
        if keyboard.is_pressed('q'):
            print("用户退出仿真。")
            break

        # 将指令打包为action
        action = [nx, nz, p_cmd, release_flare]

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