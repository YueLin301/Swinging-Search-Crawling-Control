# 想打印耗时，就搜索'耗时'

import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox import *
from spatialmath import *
import math
import navpy
import transformations as tfm
import random

import time

pi = math.pi


def cal_distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5


def crawling(snake, theta_goal, i_goal=7,l=0.1):
    t0 = time.time()

    n = snake.n
    a = []
    alpha = []
    for i in range(n):
        a.append(i % 2 * l)
        alpha.append(math.pi / 2 * (-1) ** (i))

    # 这个为的是对齐那一步，dh参数差了pi/2，要跟第8号关节对齐，要把第8号关节角度先设成0
    theta_goal2 = theta_goal.copy()
    for i in range(i_goal + 1, len(theta_goal)):
        theta_goal2[i] = 0
    Ts_goal2 = snake.fkine_all(theta_goal2)

    # print(time.time() - t0, '：crawling创建耗时')
    t0 = time.time()
    ############################################################################################################

    # snake.teach(theta_goal)
    # snake.teach()

    ############################################################################################################
    # 第1节位置到达

    t1 = 0.5
    t = np.arange(0, t1, t1 / 100)
    T0 = snake.fkine(snake.q)
    # Ts = ctraj(T0, Ts_goal[i_goal], t)
    # Ts = ctraj(T0, Ts_goal2[i_goal + 1], t)
    # tempT = SE3(Ts_goal2[i_goal].t - Ts_goal2[i_goal + 1].t)
    # T1 = Ts_goal2[i_goal + 1] @ tempT
    T1 = SE3(np.r_[np.c_[Ts_goal2[i_goal + 1].R, Ts_goal2[i_goal].t.reshape(3, 1)], np.array([[0, 0, 0, 1]])])
    Ts = ctraj(T0, T1, t)
    # print(T0)

    solq = []
    solq.append(snake.q)
    for i in range(1, len(Ts)):
        sol = snake.ikine_LMS(Ts[i], q0=solq[-1])
        solq.append(sol.q)

    # print(time.time() - t0, '：crawling第1节位置到达耗时')
    t0 = time.time()
    ############################################################################################################
    # 第1节插入

    # 这种情况第1节后面就没东西了，不过这样的话也不需要分析，直接返回好了
    if i_goal + 3 > n:
        solq = np.array(solq)
        return solq

    # 插入始终用的是这个轨迹，只是规划关节数的不同
    # T0 = Ts_goal2[i_goal - 1]
    T0 = T1
    T1A = T0.A.copy()
    # T1A[:, 3] = Ts_goal[i_goal+2].A[:, 3]
    T1A[:, 3] = Ts_goal2[i_goal + 2].A[:, 3]
    T1 = SE3(T1A)
    Ts = ctraj(T0, T1, t)

    for i in range(len(Ts)):
        sol = snake.ikine_LMS(Ts[i], q0=solq[-1])
        solq.append(sol.q)

    # print(time.time() - t0, '：crawling第1节插入耗时')
    t0 = time.time()
    ############################################################################################################
    # 第1节插入后的动作：

    # 第2节外面对齐，并使第1节不动
    # 第2节插入

    # 第3节外面对齐，并使第1节不动
    # 第3节插入

    # ...
    # i_goal=7，则第4个杆在桶口，插入1节后还要插入2个杆
    # i_goal=9，则第5个杆子在桶口，插入1节后还要插入1个杆
    # i_goal=11，则第6个杆子在桶口，插入1节后还要插入0个杆
    # i_goal=i_goal，则第int(i_goal/2)个杆子在桶口，插入1节后还要插入int((n-1-i_goal)/2)个杆

    # --------------------------------------------------

    for i_iteration in range(1, int((n - 1 - i_goal) / 2) + 1):
        # for i_iteration in range(1, int((n - 1 - i_goal) / 2)):

        a2 = a[:-2 * i_iteration].copy()
        alpha2 = alpha[:-2 * i_iteration].copy()
        DHs2 = [RevoluteDH(a=a2[i], alpha=alpha2[i]) for i in range(len(a2))]
        snake2 = DHRobot(DHs2, name='snake2')
        snake2.q = solq[-1][:-2 * i_iteration]
        # snake2.teach(snake2.q)

        # 找到对准前的T
        Tend2 = snake2.fkine(snake2.q)
        Ttrans0 = SE3(Tend2.t)

        # 旋转矩阵转四元数，四元数可以实现姿态插补
        # 初始是对准前的姿态，目标姿态是目标杆8的姿态
        qua_0 = tfm.quaternion_from_matrix(Tend2.A)
        qua_goal = tfm.quaternion_from_matrix(Ts_goal2[i_goal + 1].A)
        # qua_goal = tfm.quaternion_from_matrix((T1@SE3(Ts_goal2[i_goal+1].t - Ts_goal2[i_goal].t)).A)

        # 为了稳住已插入关节，theta3是后面两个关节角度，要匀速变为0。所以第插入第2节和插入后续节不一样
        if -2 * i_iteration + 2 == 0:
            theta3 = solq[-1][-2 * i_iteration:]
        else:
            theta3 = solq[-1][-2 * i_iteration: -2 * i_iteration + 2]
        theta3 = np.array(theta3)

        k = 99
        for i in range(k + 1):
            sampled_qua = (1 - i / k) * qua_0 + i / k * qua_goal  # 四元数线性姿态插补
            sampled_dcm = tfm.quaternion_matrix(sampled_qua)
            sampled_se3 = Ttrans0 * SE3(sampled_dcm)
            sol = snake2.ikine_LMS(sampled_se3, q0=solq[-1][:-2 * i_iteration])
            q1 = np.r_[sol.q, theta3 - i / k * theta3]  # 后面的两个关节角度渐渐变成0
            if -2 * i_iteration + 2 != 0:
                q1 = np.r_[q1, np.array(solq[-1][-2 * i_iteration + 2:])]  # 再往后的角度保持不变
            solq.append(q1)

        # print(time.time() - t0, '：crawling第', i_iteration + 1, '节对齐耗时')
        t0 = time.time()
        # --------------------------------------------------

        remain_goal = np.array(theta_goal[i_goal + 1:i_goal + 1 + 2 * i_iteration])
        remain_zero = np.array(solq[-1][-2 * i_iteration:])

        for i in range(len(Ts)):
            sol = snake2.ikine_LMS(Ts[i], q0=solq[-1][:-2 * i_iteration])
            # q1 = np.r_[sol.q, np.array(theta_goal[i_goal + 1:i_goal + 1 + 2*i_iteration]) * (i / len(Ts))] # 不该从0开始
            q1 = np.r_[sol.q, remain_zero + (remain_goal - remain_zero) * (i / len(Ts))]  # 从当前角度开始，匀速变化到goal角度
            solq.append(q1)

        # print(time.time() - t0, '：crawling第', i_iteration + 1, '节插入耗时')
        t0 = time.time()

    solq = np.array(solq[k:])
    return solq


def sample_k_dots(dot0, dot1, k):
    dots = []
    for i in range(k):
        doti = dot0 + (dot1 - dot0) * i / k
        dots.append(doti)

    return dots


def dot_in_link(dot, t1, t2):
    is_inside = False
    if (dot - t1) @ (t2 - t1) >= 0 and (dot - t2) @ (t1 - t2) >= 0:
        is_inside = True
    return is_inside


def get_dis_of_dot_from_link(dot, t1, t2):
    # vec_link = t2 - t1
    # # vec_link_magni = (vec_link[0] ** 2 + vec_link[1] ** 2 + vec_link[2] ** 2) ** 0.5
    # vec_link_magni = 0.1 # 杆长
    # vec_link_unit = vec_link / vec_link_magni
    # vec_dis = (dot - t1) - vec_link_unit * (dot - t1)
    # vec_dis_magni = (vec_dis[0] ** 2 + vec_dis[1] ** 2 + vec_dis[2] ** 2) ** 0.5
    # dis = vec_dis_magni
    #
    # return dis
    # ---------------------
    t1 = np.array(t1)
    t2 = np.array(t2)
    dot = np.array(dot)
    x1 = t1 - dot
    x2 = t1 - t2
    # print(np.linalg.norm(p2))
    angle = x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    angle = np.arccos(angle)  # math.acos(angle)
    # print(angle)
    s = np.linalg.norm(x1)
    distance = s * np.sin(angle)
    return distance


def get_dis_of_dot_from_joint(dot, t):
    '''
    :param dot:
    :param t: 关节的translation，只有位置
    :return:点到点的距离
    '''
    vec_dis = t - dot
    vec_dis_magni = (vec_dis[0] ** 2 + vec_dis[1] ** 2 + vec_dis[2] ** 2) ** 0.5
    dis = vec_dis_magni

    return dis


def get_dis_of_dot_from_ts(dot, ts):
    '''
    :param dot:要计算的点
    :param ts:目标构型的关键点的translation，只有位置
    :return:dot距离这个构型多远
    思路是这样的：
    先给相邻的关键点作为杆，以杆为中轴，作半径无限大的桶。看dot在哪几个杆的桶内，这些杆记为link_critical。
    实际实现不是要真的画桶，而是用向量相乘作为判断依据。具体看笔记。
    如果link_critical不为空，则，计算点到这些杆的距离，也就是算点到直线的距离（因为在这个杆里面）。取最小的那个距离作为返回值
    如果link_critical一个也没有，则，计算dot到各个关键点的距离，也就是点到点的距离。取最小的那个距离作为返回值。
    '''
    link_critical_i = []
    for i in range(len(ts) - 1):
        if dot_in_link(dot, ts[i], ts[i + 1]):
            link_critical_i.append(i)
    if len(link_critical_i) != 0:
        # 计算点到这些杆到距离（就是点到直线的距离），取最小值，赋值给dis
        mindis = get_dis_of_dot_from_link(dot, ts[link_critical_i[0]], ts[link_critical_i[0] + 1])
        for i in range(len(link_critical_i) - 1):
            dis_ofi = get_dis_of_dot_from_link(dot, ts[link_critical_i[i]], ts[link_critical_i[i] + 1])
            if dis_ofi < mindis:
                mindis = dis_ofi
        # for i in range(len(link_critical_i)):   # 这个循环本来不需要的，但是考虑到数值分析，上面那个循环可能会出错，所以加个这个
        #     dis_ofi_forjoint = cal_distance(dot,ts[link_critical_i[i]])
        #     dis_ofi_forjoint2 = cal_distance(dot,ts[link_critical_i[i] + 1])
        #     if dis_ofi_forjoint < mindis:
        #         mindis = dis_ofi_forjoint
        #     if dis_ofi_forjoint2 < mindis:
        #         mindis = dis_ofi_forjoint2
        dis_returned = mindis
    else:
        # 计算点到所有ts的距离，取最小值，赋值给dis
        mindis = get_dis_of_dot_from_joint(dot, ts[0])
        for i in range(len(ts)):
            dis_ofi = get_dis_of_dot_from_joint(dot, ts[i])
            if dis_ofi < mindis:
                mindis = dis_ofi
        dis_returned = mindis

    return dis_returned


def crawling_evaluation(snake, theta_goal, i_goal, solq, x0bucket):
    n = snake.n
    Ts_goal_all = snake.fkine_all(theta_goal)
    # ts_goal = [Ts_goal_all[i].t for i in range(2, n, 2)]
    ts_goal = [Ts_goal_all[i].t for i in range(i_goal, n, 2)]
    ts_goal.append(Ts_goal_all[n].t)

    # print(ts_goal)

    # k_sampledots_aline_atatime = 1000
    k_sampledots_aline_atatime = 200
    dis_deviation = -1
    dis_list_forplot = []
    for i_q in range(len(solq)):
        # snake.teach(solq[i_q])

        # if i_q == 200:
        #     print('test')

        Ts_atatime_all = snake.fkine_all(solq[i_q])
        # ts_atatime = [Ts_atatime_all[i].t for i in range(n)]
        ts_atatime = [Ts_atatime_all[i + 1].t for i in range(i_goal, n + 1, 2) if
                      # Ts_atatime_all[i + 1].t[0] >= Ts_goal_all[i_goal].t[0]]
                      Ts_atatime_all[i + 1].t[0] >= ts_goal[0][0]]
        # Ts_atatime_all[i + 1].t[0] >= x0bucket]
        # if len(ts_atatime) - 1>0:
        #     snake.teach(solq[i_q])
        # all_sampledots_atatime = []
        maxdis_atatime = -1
        if len(ts_atatime) == 1:
            dots = sample_k_dots(Ts_atatime_all[n - 1].t, Ts_atatime_all[n].t, k_sampledots_aline_atatime)
            for dot in dots:
                dis = get_dis_of_dot_from_ts(dot, ts_goal)
                if dis > maxdis_atatime:
                    maxdis_atatime = dis
            # useful_print
            # print('maxdis_atatime:', maxdis_atatime)
            dis_list_forplot.append(maxdis_atatime)
        for i in range(len(ts_atatime) - 1):
            dots = sample_k_dots(ts_atatime[i], ts_atatime[i + 1], k_sampledots_aline_atatime)
            # if i_q == len(solq) - 1:
            #     print('haha, this print is for debug.')
            for dot in dots:
                dis = get_dis_of_dot_from_ts(dot, ts_goal)
                if dis > maxdis_atatime:
                    maxdis_atatime = dis
            # all_sampledots_atatime.append(dots)
            if i == len(ts_atatime) - 2:
                # useful_print
                # print('maxdis_atatime:', maxdis_atatime)
                dis_list_forplot.append(maxdis_atatime)
        # 此时maxdis_atatime指的是，一个时刻下所有连杆中的采样点，偏离的最大距离
        if maxdis_atatime > dis_deviation:
            dis_deviation = maxdis_atatime
            # print(i_q)

    return dis_deviation, dis_list_forplot


def get_smaller_theta(theta_goal):
    smaller_theta_goal = theta_goal.copy()
    for i in range(len(smaller_theta_goal)):
        while smaller_theta_goal[i] > pi:
            smaller_theta_goal[i] -= 2 * pi
        while smaller_theta_goal[i] < -pi:
            smaller_theta_goal[i] += 2 * pi

    i = 0
    a = 1
    while i < (len(smaller_theta_goal)):
        smaller_theta_goal[i] = smaller_theta_goal[i] * a
        smaller_theta_goal[i + 1] = smaller_theta_goal[i + 1] * a
        if abs(smaller_theta_goal[i]) > math.pi / 2 and abs(smaller_theta_goal[i + 1]) > math.pi / 2:
            if smaller_theta_goal[i] > math.pi / 2:
                smaller_theta_goal[i] = smaller_theta_goal[i] - math.pi
            if smaller_theta_goal[i] < -math.pi / 2:
                smaller_theta_goal[i] = smaller_theta_goal[i] + math.pi
            if smaller_theta_goal[i + 1] > math.pi / 2:
                smaller_theta_goal[i + 1] = math.pi - smaller_theta_goal[i + 1]
            if smaller_theta_goal[i + 1] < -math.pi / 2:
                smaller_theta_goal[i + 1] = -smaller_theta_goal[i + 1] - math.pi
            a = a * (-1)
        i = i + 2

    return smaller_theta_goal


if __name__ == '__main__':

    #     # n = 11  # 取奇数，01，23，45，67，89，1011
    #     n = 17  # 取奇数
    #     n += 1
    #     l = 0.1
    #     a = []
    #     alpha = []maxdis_atatime
    #     for i in range(n):
    #         a.append(i % 2 * l)
    #         alpha.append(math.pi / 2 * (-1) ** (i + 1))
    #     DHs = [RevoluteDH(a=a[i], alpha=alpha[i]) for i in range(n)]
    #     snake = DHRobot(DHs, name='snake')
    #
    #     # theta_goal = [0]*12
    #     # theta_goal = [-0.47992786, -0.50210473, -1.34314044, 1.37726767, 1.07720779, 0.51111189, 0.70119452, -0.19085444,
    #     #               -0.13262254, 0.9380306, -0.54126805, -0.79633556]
    #     # theta_goal = [-0.47992786, 0.3, -1.34314044, 1, 1.07720779, 0.6, 0.70119452, -0.19085444,
    #     #               -0.13262254, 0.9380306, -0.54126805, -0.79633556]
    #     theta_goal = [-0.47992786, -0.50210473, -1.34314044, 1.37726767, 1.07720779, 0.51111189, 0.70119452, -0.19085444,
    #                   -0.13262254, 0.9380306, -0.54126805, -0.79633556, 0.6, 0.70119452, -0.19085444,
    #                   -0.13262254, 0.9380306, -0.54126805]
    #
    #     # i_goal = 7
    #     i_goal = 11
    #
    #     ############################################################################################################
    #
    #     snake.teach(theta_goal)
    #
    #     ############################################################################################################
    #
    #     solq = crawling(snake, theta_goal, i_goal)
    #     # 不是要判断所有的
    #     solq_needtoevaluate = solq[200:]
    #     # print(solq_needtoevaluate)
    #     dis_deviation = crawling_evaluation(snake, theta_goal, i_goal, solq_needtoevaluate, x0bucket=0.25)
    #     print('最大偏离距离：',dis_deviation)
    #     # print(time.time() - t0)
    #
    #     # solq = [[0] * 13]
    #     # theta_goal = [0] * 13
    #     # theta_goal[1] = pi / 6
    #     # # theta_goal[1] = pi / 2
    #     # # theta_goal[3] = -pi / 2
    #     # dis_deviation = crawling_evaluation(snake, theta_goal, 8, solq, 0.25)
    #     # print(dis_deviation)
    #
    #     ############################################################################################################
    #
    #     # print(time.time()-t0)
    #     snake.plot(solq, dt=0.01).hold()
    #     # snake.plot(solq[100:], dt=0.01).hold()
    #     # snake.plot(solq[k:], dt=0.01).hold()
    #     # snake.plot(solq[-2*k:], dt=0.01).hold()

    # =====================================================================
    # n_forcrawling = 11  # 取奇数，01，23，45，67，89，1011
    # n_forcrawling = 17  # 取奇数
    # n_forcrawling = 15  # 取奇数
    n_forcrawling = 23  # 取奇数
    n_forcrawling += 1
    l_forcrawling = 0.1
    # l_forcrawling = 0.1
    a_forcrawling = []
    alpha_forcrawling = []
    for i in range(n_forcrawling):
        a_forcrawling.append(i % 2 * l_forcrawling)
        # alpha_forcrawling.append(math.pi / 2 * (-1) ** (i + 1))
        alpha_forcrawling.append(math.pi / 2 * (-1) ** (i))
    # alpha_forcrawling[0] = 0
    DHs_forcrawling = [RevoluteDH(a=a_forcrawling[i], alpha=alpha_forcrawling[i]) for i in range(n_forcrawling)]
    snake_forcrawling = DHRobot(DHs_forcrawling, name='snake_forcrawling')
    # ================================================================================
    theta_goal = [-0.1168333,  -0.13400169, -0.08564983 ,-0.12127896 , 6.52929368 , 6.87547975,
  6.6726187  , 6.70660188, 10.47451076,10.29443417,  0.72463256 , 0.54855903,
  0.57619553 , 0.11854119,  0.78551201 , 0.968389 ,   4.78664748 , 4.96500916,
  8.11939328 , 7.52287339 , 8.19849874 , 8.5699433,  21.34462525 ,15.54844523]
    theta_goal = get_smaller_theta(theta_goal)
    print(theta_goal)
    x0bucket = 0.6
    # x0bucket = 1.2
    while True:
        # snake_forcrawling.teach(theta_goal)
        Ts = snake_forcrawling.fkine_all(theta_goal)
        i_goal = n_forcrawling + 100
        for i in range(len(Ts) - 1):
            if Ts[i].t[0] < x0bucket and Ts[i + 1].t[0] > x0bucket:
                i_goal = i
                print('i_goal:', i_goal)
        if i_goal + 1 > n_forcrawling or i_goal + 3 > n_forcrawling:
            print('i_goal:', i_goal)
            print('continue')
            continue
        print(Ts[i_goal + 1].t)
        print(Ts[i_goal + 3].t)
        print(Ts[i_goal + 1].t - Ts[i_goal + 3].t + Ts[i_goal + 1].t)
        t_needtoreach_forbase = Ts[i_goal + 1].t - Ts[i_goal + 3].t + Ts[i_goal + 1].t
        print('workspace:', l_forcrawling * (i_goal + 1) / 2)
        print('distance:', cal_distance(t_needtoreach_forbase, [0, 0, 0]))
        print('distance:', cal_distance(Ts[i_goal + 1].t, [0, 0, 0]))

        # if cal_distance(t_needtoreach_forbase, [0, 0, 0]) > l_forcrawling * (i_goal + 1) / 2 - 0.5 * l_forcrawling \
        #         or i_goal < 5 \
        #         or cal_distance(Ts[i_goal + 1].t,[0, 0, 0]) > l_forcrawling * (i_goal + 1) / 2 - 0.5 * l_forcrawling:
        #     print('false')
        # else:
        #     break

        if cal_distance(t_needtoreach_forbase, [0, 0, 0]) > l_forcrawling * (i_goal + 1) / 2 - 1 * l_forcrawling \
                or i_goal < 7 \
                or cal_distance(Ts[i_goal + 1].t, [0, 0, 0]) > l_forcrawling * (i_goal + 1) / 2 - 1 * l_forcrawling:
            print('false')
        else:
            break

    snake_forcrawling.teach(theta_goal)
    solq = crawling(snake_forcrawling, theta_goal, i_goal)  # k = 99
    dis_deviation, dis_list_forplot = crawling_evaluation(snake_forcrawling, theta_goal, i_goal, solq[100:],
                                                          x0bucket=x0bucket)  # 一开始直着插入肯定不会偏离
    print('最大偏离距离：', dis_deviation)

    # snake_forcrawling.plot(solq[200:]).hold()
    snake_forcrawling.plot(solq[:]).hold()
    plt.plot(dis_list_forplot)
    # plt.plot(solq[:,0])
    plt.show()

    # ================================================================================

    # theta_goal = np.random.rand(n_forcrawling) * 3 * pi - np.array([3 * pi / 2] * n_forcrawling)
    # print(theta_goal)
    # snake_forcrawling.teach(theta_goal)
    # smaller_theta_goal = get_smaller_theta(theta_goal)
    # print(smaller_theta_goal)
    # snake_forcrawling.teach(smaller_theta_goal)
    # print('haha')
