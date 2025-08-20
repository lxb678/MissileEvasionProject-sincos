import numpy as np


# def calculate_miss_distance(M1, T1, M2, T2, Vm, Vt, t1, t2):
#     """
#     计算导弹脱靶量。
#
#     参数:
#     M1: 时刻 t1 的导弹位置向量
#     T1: 时刻 t1 的目标位置向量
#     M2: 时刻 t2 的导弹位置向量
#     T2: 时刻 t2 的目标位置向量
#     Vm: 导弹速度
#     Vt: 目标速度
#     t1: 时刻 t1
#     t2: 时刻 t2
#
#     返回:
#     d_miss: 导弹脱靶量
#     """
#
#     # 计算仿真步长
#     delta_t = t2 - t1
#
#     # 计算向量
#     M1_M2 = M2 - M1
#     T1_T2 = T2 - T1
#
#     # 计算中间变量
#     MT = M1 + (t2 - t1) * (T1_T2 - M1_M2) / delta_t
#     MT_magnitude = np.sqrt((T1_T2 - M1_M2).dot(T1_T2 - M1_M2) * ((t2 - t1) ** 2) +
#                            2 * (M1_M2.dot(T1_T2) - M1_M2.dot(M1_M2)) * (t2 - t1) / delta_t +
#                            np.linalg.norm(M1_M2) ** 2)
#
#     # 计算弹目距离最小的时刻 t*
#     denominator = np.linalg.norm(T1_T2 - M1_M2) ** 2
#     if denominator == 0:
#         t_star = t1  # 如果分母为零，直接使用 t1
#     else:
#         t_star = t1 - (M1_M2.dot(T1_T2) - M1_M2.dot(M1_M2) * delta_t) / denominator
#
#     # 计算导弹脱靶量
#     d_miss = np.linalg.norm(M1 + (t_star - t1) * (T1_T2 - M1_M2) / delta_t)
#
#     return d_miss
#
#
# # 示例参数
# M1 = np.array([1.0, 2.0, 3.0])
# T1 = np.array([4.0, 5.0, 6.0])
# M2 = np.array([7.0, 8.0, 9.0])
# T2 = np.array([10.0, 11.0, 12.0])
# Vm = 100.0
# Vt = 50.0
# t1 = 0.0
# t2 = 1.0
#
# # 计算导弹脱靶量
# d_miss = calculate_miss_distance(M1, T1, M2, T2, Vm, Vt, t1, t2)
# print

# 转到惯性系
def rotation_matrix(theta, psi):
    """
    生成从机体坐标系 -> 惯性坐标系的旋转矩阵。
    假设滚转角 φ=0，仅考虑俯仰 θ 和偏航 ψ。
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(psi)
    sp = np.sin(psi)

    # 方向余弦矩阵（3x3）
    # 将机体坐标系中向量 R·v 变换为惯性坐标系下向量
    # R = np.array([
    #     [ct * cp, -sp, st * cp],
    #     [st, 0, -ct],
    #     [-ct * sp, -cp, -st * sp]
    # ])
    R = np.array([
        [ct * cp, -st * cp, sp],
        [st, ct, 0],
        [-ct * sp, st * sp, cp]
    ])
    return R




def quat_to_rotation_matrix(q):
    """
    输入：
        q : 四元数 [q0, q1, q2, q3]，满足单位四元数约束
    输出：
        R : 3x3 旋转矩阵（机体系 -> 惯性系）
    """
    q0, q1, q2, q3 = q

    R = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])
    return R


def euler_to_quat(theta, psi):
    """
    仅考虑俯仰角 theta 和偏航角 psi（假设 φ = 0）
    """
    phi = 0.0
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return np.array([q0, q1, q2, q3])


from scipy.spatial.transform import Rotation as R


def euler_to_quat_fixed(theta, psi):
    # 明确使用 Z（偏航）-Y（俯仰）顺序
    r = R.from_euler('zyx', [theta, psi, np.deg2rad(0.0)])  # ψ, θ, φ
    return r.as_quat()  # 返回 [x, y, z, w]


def quat_to_rotation_matrix_fixed(q):
    # 输入 [x, y, z, w]
    r = R.from_quat(q)
    return r.as_matrix()


# print("theta, psi_c", np.rad2deg(theta), np.rad2deg(psi_c))
# q = euler_to_quat(theta, psi_c)
q = euler_to_quat(np.deg2rad(40), np.deg2rad(0))
q = q / np.linalg.norm(q)
R = quat_to_rotation_matrix(q)
a_overload_body = np.array([0.0, 50 * 9.81, 0.0])

a_overload = R @ a_overload_body
print(a_overload)



# from scipy.spatial.transform import Rotation as R
# import numpy as np
#
# def euler_to_quat_fixed(theta_rad, psi_rad):
#     # 明确使用 Z（偏航）-Y（俯仰）顺序，输入为弧度
#     r = R.from_euler('zyx', [psi_rad, theta_rad, 0.0], degrees=False)  # ψ, θ, φ
#     return r.as_quat()  # 返回 [x, y, z, w]
#
# def quat_to_rotation_matrix_fixed(q):
#     r = R.from_quat(q)
#     return r.as_matrix()
#
# # 测试：俯仰 0°, 偏航 10°
# theta = np.deg2rad(0)
# psi = np.deg2rad(10)
# q = euler_to_quat_fixed(theta, psi)
# q = q / np.linalg.norm(q)
# R_matrix = quat_to_rotation_matrix_fixed(q)
#
# # 构造过载（机体坐标系）：前x，上y，右z
# a_overload_body = np.array([0.0, -50 * 9.81, 0.0])
# a_overload_inertial = R_matrix @ a_overload_body
#
# print("四元数 q = [x, y, z, w]:", q)
# print("惯性系过载 a_overload =", a_overload_inertial)

# R_vec = np.array([1, 0, 0])
R_vec = np.array([ 626.97201563, -786.86666182,  856.5236325 ])
V_missile_vec  = np.array([  31.99791643, -425.12227458, -234.64931454])
cos_Angle = np.dot(R_vec, V_missile_vec) / (np.linalg.norm(R_vec) * np.linalg.norm(V_missile_vec))
cos_Angle = np.clip(cos_Angle, -1.0, 1.0)
Angle = np.arccos(cos_Angle)
print("Angle", np.rad2deg(Angle))


phi_L = np.arctan2(1, 1)
print("phi_L", np.rad2deg(phi_L))



