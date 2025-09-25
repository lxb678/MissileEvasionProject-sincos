import numpy as np

def rho_custom(H):
    P0 = 101325
    T0 = 273 + 15
    T_H = T0 - 0.006 * H
    P_H = (1 - H / 44300) ** 5.256
    return 1.293 * P_H * (273 / T_H)

def rho_ISA(H):
    T0 = 288.15
    P0 = 101325
    L = 0.0065
    R = 287.05
    g = 9.80665
    M = 0.0289644
    R_univ = 8.31447

    T_H = T0 - L * H
    P_H = P0 * (1 - L*H/T0) ** (g*M/(R_univ*L))
    return P_H / (R * T_H)

for h in [0, 1000, 5000, 10000]:
    print(f"H={h} m: custom={rho_custom(h):.3f}, ISA={rho_ISA(h):.3f}")
