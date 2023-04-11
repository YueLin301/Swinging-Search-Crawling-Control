from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import *
from crawling.crawling_v993_bugfixed_copy import *

theta_goal = \
    [0.77222561, 0.4143921, - 0.97672282, - 1.61090104, - 1.45904545, 1.23838725,
     - 0.23083603, - 0.70360938, - 0.75058545, 0.830301, 0.59480265, 1.10213931,
     0.24202847, - 0.47648631, - 0.14817738, 0.27289293]

n_forcrawling = 16
l_forcrawling = 0.3

a_forcrawling = []
alpha_forcrawling = []
for i in range(n_forcrawling):
    a_forcrawling.append(i % 2 * l_forcrawling)  # （奇数关节后跟一根杆）
    alpha_forcrawling.append(math.pi / 2 * (-1) ** (i))
# a和alpha此时都有n个
DHs_forcrawling = [RevoluteDH(a=a_forcrawling[i], alpha=alpha_forcrawling[i]) for i in range(n_forcrawling)]
snake_forcrawling = DHRobot(DHs_forcrawling, name='snake_forcrawling')

x0bucket = 1
# y0bucket = 0.9
y0bucket = 0
z0bucket = 0
h = n_forcrawling / 2 * l_forcrawling - x0bucket
r = 0.2

if __name__ == '__main__':
    Ts = snake_forcrawling.fkine_all(theta_goal)
    for i in range(len(Ts)):
        if Ts[i].t[0] < x0bucket and Ts[i + 1].t[0] > x0bucket:
            i_goal = i
            print('i_goal:', i_goal)
    solq = crawling(snake_forcrawling, theta_goal, i_goal, l_forcrawling)

    # ====================
    # ====================
    # # 最大偏差距离
    # dis_deviation, dis_list_forplot = crawling_evaluation(snake_forcrawling, theta_goal, i_goal, solq[100:], x0bucket)
    # print('dis_deviation:', dis_deviation)
    # plt.plot(dis_list_forplot)

    # ====================
    # ====================

    # # 关节角度
    # print('solq.shape:',solq.shape)
    # for i in range(i_goal+1,solq.shape[1]):
    #     plt.plot(solq[:,i],label='joint '+str(i),linewidth=2)
    #     plt.legend()
    # plt.show()

    # ====================
    # ====================
    # 目标构型图

    # fig = plt.figure(1)
    # # ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    # # ax.set_aspect('equal')
    # # set_axes_equal(ax)
    # ax.set_box_aspect((2, 0.8, 0.4))
    #
    # u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    # h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    # xsurface = np.outer(np.ones(len(u)), h_new)
    # ysurface1 = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    # zsurface1 = np.outer(np.sin(u) * r, np.ones(len(h_new)))
    #
    # for i in range(xsurface.shape[0]):
    #     for j in range(xsurface.shape[1]):
    #         xsurface[i][j] += x0bucket
    #
    # for i in range(ysurface1.shape[0]):
    #     for j in range(ysurface1.shape[1]):
    #         ysurface1[i][j] += y0bucket
    #
    # for i in range(zsurface1.shape[0]):
    #     for j in range(zsurface1.shape[1]):
    #         zsurface1[i][j] += z0bucket
    #
    # ax.plot_surface(xsurface, ysurface1, zsurface1, cmap=plt.get_cmap('gray'), alpha=0.8)
    # ts_goal = [T.t for T in Ts]
    # ts_goal = np.array(ts_goal)
    # plt.plot(ts_goal[:, 0], ts_goal[:, 1], ts_goal[:, 2], '-o', c='orange', linewidth=2)
    # plt.show()



    # ====================
    # ====================

    # 流程

    plt.ion()
    fig = plt.figure(1)
    # ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # set_axes_equal(ax)
    ax.set_box_aspect((2, 0.8, 0.8))

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    xsurface = np.outer(np.ones(len(u)), h_new)
    ysurface1 = np.outer(np.cos(u) * r, np.ones(len(h_new)))
    zsurface1 = np.outer(np.sin(u) * r, np.ones(len(h_new)))

    for i in range(xsurface.shape[0]):
        for j in range(xsurface.shape[1]):
            xsurface[i][j] += x0bucket

    for i in range(ysurface1.shape[0]):
        for j in range(ysurface1.shape[1]):
            ysurface1[i][j] += y0bucket

    for i in range(zsurface1.shape[0]):
        for j in range(zsurface1.shape[1]):
            zsurface1[i][j] += z0bucket

    ax.plot_surface(xsurface, ysurface1, zsurface1, cmap=plt.get_cmap('gray'), alpha=0.8)
    # ax.plot_surface(xsurface, ysurface2, zsurface2, cmap=plt.get_cmap('ocean'), alpha=0.5)
    plt.show()

    ts_goal = [T.t for T in Ts]
    ts_goal = np.array(ts_goal)

    for i in range(len(solq)):
        if i == len(solq) - 2:
            plt.ioff()
        if i != len(solq) - 1:
            plt.cla()
        if i == 0:
            plt.pause(5)
        Ts = snake_forcrawling.fkine_all(solq[i])
        ts = [j.t for j in Ts]
        ts = np.array(ts)
        # print(ts)
        plt.plot(ts_goal[:, 0], ts_goal[:, 1], ts_goal[:, 2], '-o', c='orange', linewidth=2)
        plt.plot(ts[:, 0], ts[:, 1], ts[:, 2], '-o', c='cornflowerblue', linewidth=2)
        ax.plot_surface(xsurface, ysurface1, zsurface1, cmap=plt.get_cmap('gray'), alpha=0.8)
        plt.pause(0.0001)
