import numpy as np

def laser_interference():
    """
    激光定向干扰计算
    :return: 饱和光斑的半径 (mm), 光斑面积 (mm^2), 目标在探测器上的面积 (mm^2), 是否成功干扰导弹 (bool)
    """
    # 参数设定
    H = 5  # 目标高度 (m)
    W = 16   # 目标宽度 (m)
    R = 7   # 目标与热像仪的距离 (km)
    d0 = 12   # 探测器像元尺寸 (mm)
    d = 27  # mm
    f = 57  # 焦距 (mm)
    P0 = 4000  # 激光辐照能量 (W)
    theta = 1  # 束散角 (mrad)
    D0 = 100  # 光学镜头通光口径 (mm)
    a = 43.6  # 衍射光阑半径 (mm)
    tau0 = 0.8  #大气透过率
    tau1 = 0.8  #光学镜头透过率
    Ith = 5 * 10**-2  # 刚饱和时的激光能量 (W/m^2)
    lambda_ = 10.6  # 激光波长 (um)
    alpha = np.deg2rad(0)  # 激光干扰入射角 (rad)
    with_infrared_decoy = True  # 是否有红外诱饵弹

    # 计算目标成像尺寸
    nH = (H * f) / (R * 1000 * d0)  #目标像在探测器纵向上所占像素数
    nW = (W * f) / (R * 1000 * d0)  #目标像在探测器横向上所占像素数

    # 计算激光打在探测器表面焦面功率密度
    I0 = (4 * tau0 * tau1 * P0 * (D0 / d0)**2 * np.cos(alpha)) / (np.pi * R**2 * theta**2)
    print(f"激光打在探测器表面焦面功率密度: {I0} W/m^2")

    # 计算饱和光斑的半径
    if I0 < Ith:
        r_laser = 0  # 光斑较小，对干扰效果影响不大
    else:
        r_laser = (lambda_ * 0.001 * d) / (2 * np.pi * a) * np.cbrt(4 * I0 / Ith)
        #r_laser = (lambda_ * 0.001 * d) / (2 * np.pi * a) * np.cbrt((16 * tau0 * tau1 * D0 ** 2 * P0 * np.cos(alpha)) / (np.pi * R ** 2 * theta ** 2 * d0 ** 2 * Ith))
        #r_laser = (lambda_ * 10**-3 * d) / (2 * np.pi * a) * np.cbrt(4 * I0 / Ith)
    print(f"饱和光斑的半径: {r_laser} mm")

    # 计算光斑面积和目标在探测器上的面积
    spot_area = np.pi * r_laser**2
    target_area = np.pi * d0**2 * nH * nW

    print(f"光斑面积: {spot_area} mm^2")
    print(f"目标在探测器上的面积: {target_area} mm^2")

    # 判断是否成功干扰导弹
    if with_infrared_decoy:
        if spot_area >= target_area:
            return True  # 成功干扰导弹导引头
        else:
            return False  # 没有成功干扰导弹导引头
    else:
        if spot_area >= target_area:
            return True  # 成功干扰导弹导引头
        else:
            return False  # 没有成功干扰导弹导引头

# 调用函数
is_success = laser_interference()
print(f"是否成功干扰导弹: {is_success}")