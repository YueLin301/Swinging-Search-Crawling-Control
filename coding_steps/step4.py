# 第4步：判断有没有碰撞

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import math


def generate_goal(h, r, x0, y0, z0):
    # x = random.random()*4+2
    x = random.random() * h + x0
    # 极坐标
    th = random.random() * 2 * math.pi
    r_new = random.random() * r
    y = y0 + r_new * math.cos(th)
    z = z0 + r_new * math.sin(th)

    goal_position = [x, y, z]
    # print(x.shape)

    return goal_position


def check_point_collision(x, y, z, h, r, x0, y0, z0):
    if x < x0:
        is_collision = False
    else:
        if (y - y0) ** 2 + (z - z0) ** 2 < r ** 2:
            is_collision = False
        else:
            is_collision = True

    return is_collision


def check_point_collision_incomplexpipe(x, y, z, h, r1, r2, x0, y0, z0):
    if x < x0:
        is_collision = False
    else:
        if r2 ** 2 < (y - y0) ** 2 + (z - z0) ** 2 < r1 ** 2:
            is_collision = False
        else:
            is_collision = True

    return is_collision


def sample_inline(dot1, dot2, k=10):
    dots = []
    x1, y1, z1 = dot1[0], dot1[1], dot1[2]
    x2, y2, z2 = dot2[0], dot2[1], dot2[2]
    p1 = np.array([x1,y1,z1])
    p2 = np.array([x2,y2,z2])

    lamb = 0
    while lamb < 1-1/k:
        p = lamb*p2+(1-lamb)*p1
        dots.append(p.tolist())
        lamb+=1/k
    return dots


if __name__ == '__main__':
    fig = plt.figure(1)
    ax = Axes3D(fig)

    dots = sample_inline([0,1,1],[1,0,0])
    print(dots)
    print(len(dots))
    for i in range(len(dots)):
        ax.scatter(dots[i][0],dots[i][1],dots[i][2])
    plt.show()

    # # 桶的尺寸
    # h = 4
    # r = 0.75
    # # 桶的位置
    # x0 = 2
    # y0 = 0.9
    # z0 = 0
    #
    # x, y, z = 0.9, 0.9, 0
    # print(check_point_collision(x, y, z, h, r, x0, y0, z0))
    #
    # x, y, z = 0.9, 2, 0
    # print(check_point_collision(x, y, z, h, r, x0, y0, z0))
    #
    # x, y, z = 3, 0.9, 0
    # print(check_point_collision(x, y, z, h, r, x0, y0, z0))
    #
    # x, y, z = 3, 2, 0
    # print(check_point_collision(x, y, z, h, r, x0, y0, z0))

    # =================================

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    # h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    # x = np.outer(np.ones(len(u)), h_new)
    # y = np.outer(np.cos(u)*r, np.ones(len(h_new)))
    # z = np.outer(np.sin(u)*r, np.ones(len(h_new)))
    #
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         x[i][j] += x0
    #
    # for i in range(y.shape[0]):
    #     for j in range(y.shape[1]):
    #         y[i][j] += y0
    #
    # for i in range(z.shape[0]):
    #     for j in range(z.shape[1]):
    #         z[i][j] += z0
    #
    # ax.plot_surface(x, y, z, cmap=plt.get_cmap('ocean'))
    #
    # # 生成随机目标点，看点是否生成在桶内
    # plt.ion()
    # for i in range(50):
    #     [gx,gy,gz] = generate_goal(h,r,x0,y0,z0)
    #     ax.scatter(gx, gy, gz)
    #     plt.pause(0.0001)
    # plt.ioff()
    # plt.show()
