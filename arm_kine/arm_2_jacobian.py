import torch
import numpy as np
import math

from arm_kine.arm_1_forward_kinematics import forward_kinematics3


def jacobian(n, Tforworld):
    Tforworld = Tforworld.numpy()
    k = torch.tensor([[0.], [0.], [1.]])
    jv = torch.zeros(3, n)
    jw = torch.zeros(3, n)
    k = k.numpy()
    jv = jv.numpy()
    jw = jw.numpy()
    for i in range(0, n):
        R = Tforworld[i][0:3, 0:3]
        T_i_n = np.matmul(np.linalg.inv(Tforworld[i]), Tforworld[n - 1])
        p_i_n = T_i_n[0:3, 3]
        p_i_n = p_i_n.reshape(3, 1)
        # print(p_i_n)
        z_i = np.matmul(R, k).reshape(1, 3)
        jw[:, i] = z_i
        jv[:, i] = np.cross(z_i, np.matmul(R, p_i_n).reshape(1, 3))

    return np.concatenate([jv, jw])


if __name__ == '__main__':
    n = 6
    theta = torch.zeros(6)
    theta[3] = math.pi / 2
    theta[5] = math.pi / 2
    Tforworld = forward_kinematics3(n, theta)

    j = jacobian(n, Tforworld)
    print(j)
