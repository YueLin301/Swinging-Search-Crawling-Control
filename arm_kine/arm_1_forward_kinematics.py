import math
import numpy as np
import torch

from roboticstoolbox import *

# 第一节是基座，第一个theta是基座，第二个theta是基座后的一个
# 一共其实要转换13次，所以有13个转换矩阵，但是只有12个theta是有用的，最后一个转换矩阵只用来延长一截，所以theta默认为0
# 输出的Tforward有13维


def forward_kinematics3(n, theta, l=0.1):
    d = torch.zeros(n)
    # a=torch.tensor([0,0,1,0,1,0,1,0,1,0,1,0,1])
    # a = a * 0.1
    # l = 0.1
    a = torch.zeros(n)
    for i in range(2, n, 2):
        a[i] = l
    alpha = torch.zeros(n)
    for i in range(1, n):
        alpha[i] = math.pi / 2 * (-1) ** (i + 1)  # (i)

    T = torch.zeros(n, 4, 4)

    Tforworld = torch.zeros(n, 4, 4)

    for i in range(0, n):
        T[i][0][0] = math.cos(theta[i])
        T[i][0][1] = -math.sin(theta[i])
        T[i][0][2] = 0
        T[i][0][3] = a[i]

        T[i][1][0] = math.sin(theta[i]) * math.cos(alpha[i])
        T[i][1][1] = math.cos(theta[i]) * math.cos(alpha[i])
        T[i][1][2] = -math.sin(alpha[i])
        T[i][1][3] = -math.sin(alpha[i]) * d[i]

        T[i][2][0] = math.sin(theta[i]) * math.sin(alpha[i])
        T[i][2][1] = math.cos(theta[i]) * math.sin(alpha[i])
        T[i][2][2] = math.cos(alpha[i])
        T[i][2][3] = math.cos(alpha[i]) * d[i]

        T[i][3][0] = 0
        T[i][3][1] = 0
        T[i][3][2] = 0
        T[i][3][3] = 1

    Tforworld[0] = T[0]
    for i in range(1, n):
        Tforworld[i] = torch.matmul(Tforworld[i - 1], T[i])

    return Tforworld


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    # print(R[0,0],R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z])


if __name__ == '__main__':
    n = 13
    theta = [0] * n
    # theta = [-0.07029916, 6.27095714, 5.91409166, 5.81058901, 6.90286021, 6.68480625,
    #               6.74297102, 7.54811843, 11.79019686, 5.27431367, 11.543118, 17.6600703,
    #          14.87505777]
    # theta.requires_grad=True
    # theta[6] = math.pi/6
    # theta[7] = math.pi/6
    # theta[0]=math.pi/6
    theta2 = theta.copy()

    theta[2] = math.pi / 6
    theta2[2 + 1] = math.pi / 6

    print(theta)
    print(theta2)

    T = forward_kinematics3(n, theta)

    print(T[n - 1])
    theta6 = rotationMatrixToEulerAngles(T[n - 1])
    # print(theta6)
    # theta6 = rotationMatrixToEulerAngles(T[n])

    # print(T.shape)
    # print(theta6)

    end_pose = torch.cat((T[n - 1][0:3, 3], theta6))
    # end_pose = torch.cat((T[n][0:3,3],theta6))
    # print(end_pose)

    # end_pose.requires_grad=True
    # torch.autograd.grad(end_pose,theta,grad_outputs=torch.ones_like(theta))
    # T.backward()
    # print(theta.grad)

    # n=13
    # a = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # a = a * 0.1
    # alpha = np.zeros(n)
    # for i in range(1, n):
    #     alpha[i] = math.pi / 2 * (-1) ** (i + 1)  # (i)
    # DHs = []
    # for i in range(len(a)):
    #     DHs.append(RevoluteDH(a=a[i],alpha=alpha[i]))
    # snake = DHRobot(DHs,name='snake')
    # # snake.teach(theta2)
    # Tsnake = snake.fkine(theta2)
    # print(Tsnake)

    n = 12
    a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    a = a * 0.1
    alpha = np.zeros(n)
    for i in range(n):
        alpha[i] = math.pi / 2 * (-1) ** (i + 1)  # (i)
    DHs = []
    for i in range(len(a)):
        DHs.append(RevoluteDH(a=a[i], alpha=alpha[i]))
    snake = DHRobot(DHs, name='snake')
    # snake.teach(theta2)
    Tsnake = snake.fkine(theta[:12])
    print(Tsnake)
