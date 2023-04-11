# 第2步：自运动。末端先运动过去，然后关节开始在零空间动

#  from controller import Robot, Motor, DistanceSensor
from arm_kine.arm_1_forward_kinematics import forward_kinematics3, rotationMatrixToEulerAngles
from arm_kine.arm_3_inverse_kinematics import inverse_kinematics

import math
import torch  # 用到了连接
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def cal_distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5


if __name__ == '__main__':
    plt.ion()
    fig = plt.figure(1)
    ax = Axes3D(fig)

    dt = 0.1 * 5
    n = 13

    theta = [math.pi / 6] * n
    theta[n - 1] = 0
    theta[5] = math.pi / 5
    # theta = [0]*n
    # theta[5] = math.pi /6
    Tforworld = forward_kinematics3(n, theta)
    jointpose = []
    for i in range(n):
        jointpose.append(torch.cat((Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(Tforworld[i]))).tolist())
        jointpose[i][1] += 0.9
    jointpose = np.array(jointpose)
    end = jointpose[n - 1]

    # goal_position = [0.4,0.1,0]
    # goal_position[1]+=0.9
    goal_position = [0.4, 0.9 + 0.1, 0]
    # goal_position = end[0:3]

    ax.set_xlim([-0.1, 0.4])
    ax.set_ylim([0.9, 1.04])
    ax.set_zlim([-0.25, 0.1])

    while 1:
        # 画图，画end、各个关节
        plt.ion()
        ax.scatter(end[0], end[1], end[2])
        plt.plot(jointpose[:, 0], jointpose[:, 1], jointpose[:, 2], '--*')
        plt.pause(0.0001)

        # 规划速度，速度就是直着过去，转动的速度都为0。速度大小为0.01，方向就是直接相连
        # 这么写是因为麻烦的类型问题
        end_vol = [0, 0, 0, 0, 0, 0]
        for i in range(3):
            end_vol[i] = goal_position[i] - end[i]
        # 算模长
        end_vol_magnitude = cal_distance(end_vol[0:3], [0, 0, 0])
        # end_vol = end_vol / end_vol_magnitude
        if end_vol_magnitude != 0:
            for i in range(3):
                end_vol[i] = end_vol[i] / end_vol_magnitude
                end_vol[i] = end_vol[i] * 0.01
            # print('end_vol:\t',end_vol)

        # 逆解，输出末端速度和雅可比伪逆
        theta_vol, j, j_pinverse = inverse_kinematics(n, theta, end_vol)
        # print('theta_vol:\t\t', theta_vol)
        # *********************
        I = np.eye(j_pinverse.shape[0], dtype='float32')
        # print(I.shape)
        # print(j_pinverse.shape)
        # print(np.matmul(j_pinverse, j).shape)
        # dh = np.ones((n,1))
        dh = np.random.rand(n, 1) * 0.4
        # print(dh)
        temp = np.matmul(j_pinverse, j)
        test = np.matmul((I - temp), dh).T.tolist()[0]
        # print(theta_vol)
        # print(test)
        theta_vol = theta_vol + np.matmul((I - temp), dh).T.tolist()[0]

        # *********************
        dtheta = dt * theta_vol
        # print(dtheta.shape)
        theta = theta + dtheta
        # print('theta:\n',theta)

        # 正解，经过这个变换之后，新的end去哪里了
        Tforworld = forward_kinematics3(n, theta)
        for i in range(n):
            jointpose[i] = (torch.cat((Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(Tforworld[i]))).tolist())
            jointpose[i][1] += 0.9
        # end = torch.cat((Tforworld[n - 1][0:3, 3], rotationMatrixToEulerAngles(Tforworld[n - 1])))
        # end[1] += 0.9
        end = jointpose[n - 1]
        print('end:\t', end[0:3])
