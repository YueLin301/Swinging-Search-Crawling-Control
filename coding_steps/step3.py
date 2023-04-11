# 第3步：生成桶内的目标点

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


if __name__ == '__main__':
    # 桶的尺寸
    h = 4
    r = 0.75
    # 桶的位置，桶的轴和x平行
    x0 = 2
    y0 = 0.9
    z0 = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    x = np.outer(np.ones(len(u)), h_new)
    y = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    z = np.outer(np.sin(u) * r, np.ones(len(h_new)))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] += x0

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] += y0

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j] += z0

    ax.plot_surface(x, y, z, cmap=plt.get_cmap('ocean'), alpha=0.5)

    # 生成随机目标点，看点是否生成在桶内
    plt.ion()
    for i in range(50):
        [gx, gy, gz] = generate_goal(h, r, x0, y0, z0)
        ax.scatter(gx, gy, gz)
        plt.pause(0.0001)
    plt.ioff()
    plt.show()
