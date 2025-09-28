from Missilelaunch_environment_jsbsim.Missilelaunch_environment_jsbsim import AirCombatEnv

if __name__ == '__main__':
    # 实例化环境
    env = AirCombatEnv(tacview_enabled=True)

    # 重置环境
    observations = env.reset()

    # 循环执行
    for step in range(1000):
        # 简单的规则AI
        # 红方：保持平飞，如果蓝方在前方60度内且距离小于20km，则发射导弹
        red_obs = observations['red_agent']
        red_fire_condition = (abs(red_obs[7] * 180) < 30) and (red_obs[6] * env.max_disengagement_range < 20000)
        red_action = [0.8, 0, 0, 0, 0, 1.0 if red_fire_condition else 0.0]

        # 蓝方：简单地飞向红方
        blue_obs = observations['blue_agent']
        blue_action = [0.8, 0, -blue_obs[7] * 0.5, 0, 0, 0]  # 简单的转弯逻辑

        actions = {
            'red_agent': red_action,
            'blue_agent': blue_action
        }

        # 执行一步
        observations, rewards, dones, info = env.step(actions)

        # 打印信息
        print(f"Step: {step}, Time: {env.t_now:.2f}, Rewards: {rewards}, Dones: {dones['__all__']}")

        # 如果回合结束，则退出循环
        if dones['__all__']:
            break

    # 渲染最终轨迹
    env.render()