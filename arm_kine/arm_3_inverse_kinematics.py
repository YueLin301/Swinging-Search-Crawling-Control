import numpy as np
import torch

from arm_kine.arm_1_forward_kinematics import forward_kinematics3
from arm_kine.arm_2_jacobian import jacobian


def inverse_kinematics(n, theta, end_vol, l=0.1):
    Tforworld = forward_kinematics3(n, theta, l)

    j = jacobian(n, Tforworld)
    # print(j)
    j_inverse = np.linalg.pinv(j)

    # 注意区别于backup
    # end_vol.append(0)
    theta_vol = np.matmul(j_inverse, end_vol)
    theta_vol[n - 1] = 0

    return theta_vol, j, j_inverse


if __name__ == '__main__':
    n = 13
    theta = [0] * n
    end_vol = [-1, -1, 0, 0, 0, 0]
    print(forward_kinematics3(n, theta)[n - 1])
    theta_vol, j, j_pinverse = inverse_kinematics(n, theta, end_vol)

    dt = 0.1

    # right
    print([6 - 0.1, 0 - 0.1, 0])

    # calculation
    dtheta = dt * theta_vol
    endtheta = dtheta[0:n] + theta
    print(forward_kinematics3(n, endtheta)[n - 1])

    print(theta_vol)

    print(endtheta)
