from step4 import check_point_collision

#  from controller import Robot, Motor, DistanceSensor
from arm_kine.arm_1_forward_kinematics import forward_kinematics3, rotationMatrixToEulerAngles
from arm_kine.arm_3_inverse_kinematics import inverse_kinematics

import math
import torch#用到了连接
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def cal_distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    return sum**0.5


if __name__=='__main__':
    start = time.time()
    plt.ion()
    fig = plt.figure(1)
    ax = Axes3D(fig)

    dt = 0.1*5
    n = 13

    theta = [math.pi/6]*n
    theta[n-1] = 0
    theta[5] = math.pi / 5
    # theta = [0]*n
    # theta[5] = math.pi /6
    Tforworld = forward_kinematics3(n, theta)
    jointpose = []
    for i in range(n):
        jointpose.append(torch.cat((Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(Tforworld[i]))).tolist())
        jointpose[i][1]+=0.9
    jointpose=np.array(jointpose)
    end = jointpose[n-1]

    # goal_position = [0.4,0.1,0]
    # goal_position[1]+=0.9
    # goal_position = [0.55, 0.9-0.03, 0.05]
    goal_position = [0.53, 0.9+0.03, 0.03]
    # goal_position = end[0:3]

    ax.set_xlim([0, 0.6])
    ax.set_ylim([0.8, 1])
    ax.set_zlim([-0.1, 0.1])

    # =====================================================================
    # =====================================================================
    # 桶
    # 桶的尺寸
    h = 0.6
    r = 0.075
    # 桶的位置，桶的轴和x平行
    # x0bucket = 0.2
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

    # plt.ioff()
    ax.plot_surface(xsurface, ysurface, zsurface, cmap=plt.get_cmap('ocean'),alpha=0.5)
    # plt.pause(0.0001)
    plt.show()
    # =====================================================================
    # =====================================================================

    while 1:
        # 画图，画end、各个关节
        ax.scatter(end[0],end[1],end[2])
        plt.plot(jointpose[:,0],jointpose[:,1],jointpose[:,2],'--*')
        plt.pause(0.0001)

        # 规划速度，速度就是直着过去，转动的速度都为0。速度大小为0.01，方向就是直接相连
        # 这么写是因为麻烦的类型问题
        end_vol = [0,0,0,0,0,0]
        for i in range(3):
            end_vol[i] = goal_position[i] - end[i]
        # 算模长
        end_vol_magnitude = cal_distance(end_vol[0:3],[0,0,0])
        # end_vol = end_vol / end_vol_magnitude
        if end_vol_magnitude != 0:
            for i in range(3):
                end_vol[i] = end_vol[i] / end_vol_magnitude
                end_vol[i] = end_vol[i] *0.01
            # print('end_vol:\t',end_vol)

        #逆解，输出末端速度和雅可比伪逆
        theta_vol,j , j_pinverse = inverse_kinematics(n,theta,end_vol)
        # print('theta_vol:\t\t', theta_vol)
        # *********************
        I = np.eye(j_pinverse.shape[0],dtype='float32')
        # print(I.shape)
        # print(j_pinverse.shape)
        # print(np.matmul(j_pinverse, j).shape)
        # dh = np.ones((n,1))
        dh = np.random.rand(n, 1) * 0.5
        # print(dh)
        temp = np.matmul(j_pinverse, j)
        test = np.matmul((I-temp), dh).T.tolist()[0]
        # print(theta_vol)
        # print(test)
        theta_vol = theta_vol + np.matmul((I-temp), dh).T.tolist()[0]

        # *********************
        dtheta = dt * theta_vol
        # print(dtheta.shape)
        theta = theta + dtheta
        # print('theta:\n',theta)

        # 正解，经过这个变换之后，新的end去哪里了
        Tforworld = forward_kinematics3(n, theta)
        for i in range(n):
            jointpose[i]=(torch.cat((Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(Tforworld[i]))).tolist())
            jointpose[i][1] += 0.9
        # end = torch.cat((Tforworld[n - 1][0:3, 3], rotationMatrixToEulerAngles(Tforworld[n - 1])))
        # end[1] += 0.9
        end = jointpose[n - 1]
        # print('end:\t',end[0:3])

        # =====================================================================
        # =====================================================================
        # 输出碰撞的关节，检测刚伸入的那两个点，讨论一下（看笔记
        collide = 0
        for i in range(len(jointpose)-1):
            flag = check_point_collision(jointpose[i][0], jointpose[i][1], jointpose[i][2], h, r, x0bucket, y0bucket, z0bucket)
            if flag:
                print("joint "+str(i)+' collide')
                collide=1
            #如果当前没碰，下一个也没碰，但是这两个节点的中间的杆子碰到截口了
            elif (not check_point_collision(jointpose[i+1][0], jointpose[i+1][1], jointpose[i+1][2], h, r, x0bucket, y0bucket, z0bucket)) and (jointpose[i][0]<x0bucket and jointpose[i+1][0]>x0bucket):
                # 判断这两个节点的中间的杆子，有没有碰到截口
                # 这里直接看笔记就的推导吧，SnakeArmPlan1
                ycritical = jointpose[i][1]+(jointpose[i+1][1]-jointpose[i][1])*(x0bucket-jointpose[i][0])/(jointpose[i+1][0]-jointpose[i][0])
                zcritical = jointpose[i][2]+(jointpose[i+1][2]-jointpose[i][2])*(x0bucket-jointpose[i][0])/(jointpose[i+1][0]-jointpose[i][0])
                if (ycritical-y0bucket)**2+(zcritical-z0bucket)**2 >= r**2:
                    #这样就是碰撞了！！
                    print('>>>>>>>>')
                    print("joint "+str(i)+' collide')
                    print("joint "+str(i+1)+' collide')
                    print(ycritical,zcritical)
                    print('<<<<<<<<')
                    collide = 1
        if not collide:
            print("haha")
            # print(self.theta_ang)
            end1 = time.time()
            print(end1 - start)

        print('theta:',theta)
        print('===========================')