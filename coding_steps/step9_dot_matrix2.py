from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

pi = math.pi


def cal_distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5


def generate_dots_inside(h,r,x0bucket,y0bucket,z0bucket):
    fig = plt.figure(1)
    ax = Axes3D(fig)

    # ax.set_xlim([0.25, 0.85])
    # ax.set_ylim([0.9 - 0.3, 0.9 + 0.3])
    # ax.set_zlim([-0.3, 0.3])

    count = 40
    xs = np.linspace(x0bucket+0.05, x0bucket+0.6/1.4*h, count)
    ys = np.linspace(y0bucket-0.85*r, y0bucket+0.85*r, count)
    zs = np.linspace(z0bucket-0.85*r, z0bucket+0.85*r, count)
    # ys = np.linspace(y0bucket - r, y0bucket + r, count)
    # zs = np.linspace(z0bucket - r, z0bucket + r, count)
    dots = []
    for i in range(count):
        for j in range(count):
            for k in range(count):
                dots.append([xs[i], ys[j], zs[k]])
    # print(dots)

    # 桶的尺寸
    # h = 0.4
    # r = 0.075
    # 桶的位置，桶的轴和x平行
    # x0bucket = 0.25
    # y0bucket = 0.9
    # z0bucket = 0

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    xsurface = np.outer(np.ones(len(u)), h_new)
    ysurface = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    zsurface = np.outer(np.sin(u) * r, np.ones(len(h_new)))

    for i in range(xsurface.shape[0]):
        for j in range(xsurface.shape[1]):
            xsurface[i][j] += x0bucket

    for i in range(ysurface.shape[0]):
        for j in range(ysurface.shape[1]):
            ysurface[i][j] += y0bucket

    for i in range(zsurface.shape[0]):
        for j in range(zsurface.shape[1]):
            zsurface[i][j] += z0bucket

    ax.plot_surface(xsurface, ysurface, zsurface, cmap=plt.get_cmap('ocean'), alpha=0.5)

    dots_inside = []
    i = 0
    while i <= len(dots) - 1:
        dot = dots[i]
        x = dot[0]
        y = dot[1]
        z = dot[2]
        if cal_distance([y, z], [y0bucket, z0bucket]) <= 0.85*r and x0bucket <= x <= x0bucket + h:
        # if cal_distance([y, z], [y0bucket, z0bucket]) <= r and x0bucket <= x <= x0bucket + h:
            dots_inside.append(dot)
        i += 1

    return dots_inside


def generate_dots_inside_complex(h,r,x0bucket,y0bucket,z0bucket):
    fig = plt.figure(1)
    ax = Axes3D(fig)

    # ax.set_xlim([0.25, 0.85])
    # ax.set_ylim([0.9 - 0.3, 0.9 + 0.3])
    # ax.set_zlim([-0.3, 0.3])

    count = 40
    xs = np.linspace(x0bucket, x0bucket+0.8/1.4*h, count)
    ys = np.linspace(y0bucket-0.85*r, y0bucket+0.85*r, count)
    zs = np.linspace(z0bucket-0.85*r, z0bucket+0.85*r, count)
    dots = []
    for i in range(count):
        for j in range(count):
            for k in range(count):
                dots.append([xs[i], ys[j], zs[k]])
    # print(dots)

    # 桶的尺寸
    # h = 0.4
    # r = 0.075
    # 桶的位置，桶的轴和x平行
    # x0bucket = 0.25
    # y0bucket = 0.9
    # z0bucket = 0

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    xsurface = np.outer(np.ones(len(u)), h_new)
    ysurface = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    zsurface = np.outer(np.sin(u) * r, np.ones(len(h_new)))

    for i in range(xsurface.shape[0]):
        for j in range(xsurface.shape[1]):
            xsurface[i][j] += x0bucket

    for i in range(ysurface.shape[0]):
        for j in range(ysurface.shape[1]):
            ysurface[i][j] += y0bucket

    for i in range(zsurface.shape[0]):
        for j in range(zsurface.shape[1]):
            zsurface[i][j] += z0bucket

    ax.plot_surface(xsurface, ysurface, zsurface, cmap=plt.get_cmap('ocean'), alpha=0.5)

    dots_inside = []
    i = 0
    while i <= len(dots) - 1:
        dot = dots[i]
        x = dot[0]
        y = dot[1]
        z = dot[2]
        if r / 2 * 1.15<= cal_distance([y, z], [y0bucket, z0bucket]) <= 0.85*r and x0bucket <= x <= x0bucket + h:
            dots_inside.append(dot)
        i += 1

    return dots_inside


if __name__ == '__main__':
    fig = plt.figure(1)
    ax = Axes3D(fig)

    # ax.set_xlim([0.25, 0.85])
    # ax.set_ylim([0.9 - 0.3, 0.9 + 0.3])
    # ax.set_zlim([-0.3, 0.3])

    count = 40
    xs = np.linspace(0.25, 0.85, count)
    ys = np.linspace(0.9 - 0.3, 0.9 + 0.3, count)
    zs = np.linspace(-0.3, 0.3, count)
    dots = []
    for i in range(count):
        for j in range(count):
            for k in range(count):
                dots.append([xs[i], ys[j], zs[k]])
    # print(dots)

    # 桶的尺寸
    h = 0.6
    r = 0.075
    # 桶的位置，桶的轴和x平行
    x0bucket = 0.25
    y0bucket = 0.9
    z0bucket = 0

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    xsurface = np.outer(np.ones(len(u)), h_new)
    ysurface = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    zsurface = np.outer(np.sin(u) * r, np.ones(len(h_new)))

    for i in range(xsurface.shape[0]):
        for j in range(xsurface.shape[1]):
            xsurface[i][j] += x0bucket

    for i in range(ysurface.shape[0]):
        for j in range(ysurface.shape[1]):
            ysurface[i][j] += y0bucket

    for i in range(zsurface.shape[0]):
        for j in range(zsurface.shape[1]):
            zsurface[i][j] += z0bucket

    ax.plot_surface(xsurface, ysurface, zsurface, cmap=plt.get_cmap('ocean'), alpha=0.5)

    dots_inside = []
    i = 0
    while i <= len(dots) - 1:
        dot = dots[i]
        x = dot[0]
        y = dot[1]
        z = dot[2]
        if cal_distance([y, z], [y0bucket, z0bucket]) <= r and x0bucket <= x <= x0bucket + h:
            dots_inside.append(dot)
        i += 1

    # for dot in dots_inside:
    #     ax.scatter(dot[0], dot[1], dot[2], c='b')

    # print(dots_inside)
    print(dots_inside[int(len(dots_inside) / 2)])
    # print(len(dots_inside))
    # plt.show()
