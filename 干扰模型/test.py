def run_one_step(self, dt):
    # global t_now, x_target_now, x_missile_now, prev_theta_L, prev_phi_L
    interference_success = False

    def rk4_aircraft(x):
        k1 = self.aircraft_dynamics(self.t_now, x, self.control_input, self.g)
        k2 = self.aircraft_dynamics(self.t_now + self.dt / 2, x + self.dt / 2 * k1, self.control_input, self.g)
        k3 = self.aircraft_dynamics(self.t_now + self.dt / 2, x + self.dt / 2 * k2, self.control_input, self.g)
        k4 = self.aircraft_dynamics(self.t_now + self.dt, x + self.dt * k3, self.control_input, self.g)
        return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    x_target_next = rk4_aircraft(self.x_target_now)

    if release_flare == 1:
        self.o_ir -= 6  # 红外诱饵弹数量
        if self.o_ir >= 0:
            self.flare_manager.release_flare_group(self.t_now)

    self.flare_manager.update(self.t_now, self.dt, self.x_target_now)

    R_vec = self.x_target_now[0:3] - self.x_missile_now[3:6]
    R_rel = np.linalg.norm(R_vec)

    if open_laser == 1:
        interference_success = self.laser_interference(R_vec, R_rel, self.compute_velocity_vector(self.x_missile_now))

    lock_aircraft, *_ = self.check_seeker_lock(self.x_missile_now, self.x_target_now,
                                               self.compute_velocity_vector(self.x_missile_now),
                                               self.compute_velocity_vector_from_target(self.x_target_now),
                                               self.t_now, self.D_max, self.Angle_IR, self.omega_max, self.T_max)

    if lock_aircraft and not interference_success:
        beta_p = self.compute_relative_beta(self.x_target_now, self.x_missile_now)
        I_p = self.infrared_intensity_model(beta_p)
        pos_p = self.x_target_now[:3]
    else:
        I_p = 0.0
        pos_p = np.zeros(3)

    visible_flares = []
    for flare in self.flare_manager.flares:
        if flare.get_intensity(self.t_now) <= 1e-3:
            continue
        lock_flare, *_ = self.check_seeker_lock(self.x_missile_now,
                                                np.concatenate((flare.pos, [0, 0, 0])),
                                                self.compute_velocity_vector(self.x_missile_now),
                                                flare.vel,
                                                self.t_now, self.D_max, self.Angle_IR, self.omega_max, self.T_max)
        if lock_flare:
            visible_flares.append(flare)

    I_total = I_p
    numerator = I_p * pos_p
    for flare in visible_flares:
        I_k = flare.get_intensity(self.t_now)
        numerator += I_k * flare.pos
        I_total += I_k

    if I_total > 0:
        target_pos_equiv = numerator / I_total
        Rx = target_pos_equiv[0] - self.x_missile_now[3]
        Ry = target_pos_equiv[1] - self.x_missile_now[4]
        Rz = target_pos_equiv[2] - self.x_missile_now[5]
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        self.theta_L = np.arcsin(np.clip(Ry / R, -1.0, 1.0))
        self.phi_L = np.arctan2(Rz, Rx)
        theta_L_dot = 0.0 if self.prev_theta_L is None else (self.theta_L - self.prev_theta_L) / self.dt
        dphi = np.arctan2(np.sin(self.phi_L - self.prev_phi_L), np.cos(self.phi_L - self.prev_phi_L)) if self.prev_phi_L else 0.0
        phi_L_dot = dphi / self.dt
    else:
        self.theta_L = self.prev_theta_L if self.prev_theta_L is not None else 0.0
        self.phi_L = self.prev_phi_L if self.prev_phi_L is not None else 0.0
        theta_L_dot = self.last_valid_theta_dot
        phi_L_dot = self.last_valid_phi_dot

    def rk4_missile(x):
        k1 = self.missile_dynamics_given_rate(x, theta_L_dot, phi_L_dot, self.N, self.ny_max, self.nz_max)
        k2 = self.missile_dynamics_given_rate(x + self.dt / 2 * k1, theta_L_dot, phi_L_dot, self.N, self.ny_max, self.nz_max)
        k3 = self.missile_dynamics_given_rate(x + self.dt / 2 * k2, theta_L_dot, phi_L_dot, self.N, self.ny_max, self.nz_max)
        k4 = self.missile_dynamics_given_rate(x + self.dt * k3, theta_L_dot, phi_L_dot, self.N, self.ny_max, self.nz_max)
        return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    x_missile_next = rk4_missile(self.x_missile_now)

    # 状态更新
    self.t_now += self.dt
    self.t.append(self.t_now)
    self.Xt.append(x_target_next)
    self.Y.append(x_missile_next)
    self.x_target_now = x_target_next
    self.x_missile_now = x_missile_next
    self.prev_theta_L = self.theta_L
    self.prev_phi_L = self.phi_L

    ACT_F, R_mt_now, V_rel_now = self.check_fuze_trigger(self.x_missile_now, self.x_target_now, self.R_kill, 150)
    if ACT_F:
        print(f">>> 引信引爆！R = {R_mt_now:.2f} m, V_rel = {V_rel_now:.2f} m/s")
        return True  # 停止仿真
    return False  # 继续