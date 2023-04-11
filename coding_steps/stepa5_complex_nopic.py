# 第a2步：简单管道，加上crawling和评价反馈


import time

from arm_kine.arm_1_forward_kinematics import forward_kinematics3, rotationMatrixToEulerAngles
from arm_kine.arm_3_inverse_kinematics import inverse_kinematics
from step4 import generate_goal, check_point_collision, check_point_collision_incomplexpipe, sample_inline
import numpy as np
import torch
import math
import random

from RLagent.DDPG_agent import DDPGAgent

from step1 import step1_reachgoal

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from step9_dot_matrix2 import generate_dots_inside,generate_dots_inside_complex

from crawling.crawling_v993_bugfixed import *

# =====================================================================
# n_forcrawling = 11  # 取奇数，01，23，45，67，89，1011
n_forcrawling = 17  # 取奇数
n_forcrawling += 1
l_forcrawling = 0.1
a_forcrawling = []
alpha_forcrawling = []
for i in range(n_forcrawling):
    a_forcrawling.append(i % 2 * l_forcrawling)
    # alpha_forcrawling.append(math.pi / 2 * (-1) ** (i + 1))
    alpha_forcrawling.append(math.pi / 2 * (-1) ** (i))
# alpha_forcrawling[0] = 0
DHs_forcrawling = [RevoluteDH(a=a_forcrawling[i], alpha=alpha_forcrawling[i]) for i in range(n_forcrawling)]
snake_forcrawling = DHRobot(DHs_forcrawling, name='snake_forcrawling')

# ================================================================================================
# 共15个。3个，表示目标点的位置；12个，表示12个关节的角度。第13个关节的角度没用
observationSpace = (n_forcrawling + 3,)
# observationSpace = (15,)
# 12个，表示12个dh，用来自运动
actionSpace = (n_forcrawling,)
# actionSpace = (12,)
# 0.032，和webots里的timestep一致
# world_dt = 0.032
world_dt = 0.5
world_dt_reset = 0.5


# ================================================================================================
def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax


def cal_distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5


# ================================================================================================
# ================================================================================================
class arm_robot():
    def __init__(self):
        super().__init__()
        self.test = False

        # goal_position = [0.53, 0.9 + 0.03, 0.03]
        # self.reset_new(goal_position)

    def observe(self):
        state = self.goal_position + self.theta_ang[0:-1].tolist()
        return state

    def use_message_data(self, action):
        # 这一部分要拿来自运动，最终是改变theta，算出做了这个运动后，theta去哪里了

        # print(action)
        I = np.eye(self.j_pinverse.shape[0], dtype='float32')

        # dh是action，最后加一个0
        dh = []
        for i in action:
            # dh.append(i * 0.4)
            dh.append(i)
        dh.append(0)

        # 公式
        temp = np.matmul(self.j_pinverse, self.j)
        test = np.matmul((I - temp), dh).T.tolist()[0]
        self.theta_vol = self.theta_vol + np.matmul((I - temp), dh).T.tolist()[0]
        dtheta = world_dt * self.theta_vol

        self.theta_ang = self.theta_ang + dtheta

        self.Tforworld = forward_kinematics3(self.n, self.theta_ang)
        for i in range(self.n):
            self.jointpose[i] = (
                torch.cat((self.Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(self.Tforworld[i]))).tolist())
            self.jointpose[i][1] += 0.9
        self.end = self.jointpose[self.n - 1]

        reach_limit = 500
        count = 0
        while cal_distance(self.goal_position, self.end[0:3]) > 0.01 and count < reach_limit:
            count = count + 1

            # 规划速度，速度就是直着过去，转动的速度都为0。速度大小为0.01，方向就是直接相连
            # 这么写是因为麻烦的类型问题
            self.end_vol = [0, 0, 0, 0, 0, 0]
            for i in range(3):
                self.end_vol[i] = self.goal_position[i] - self.end[i]
            # 算模长
            end_vol_magnitude = cal_distance(self.end_vol[0:3], [0, 0, 0])
            # end_vol = end_vol / end_vol_magnitude
            for i in range(3):
                self.end_vol[i] = self.end_vol[i] / end_vol_magnitude
                self.end_vol[i] = self.end_vol[i] * 0.005
            # 逆解，输出末端速度和雅可比伪逆
            self.theta_vol, self.j, self.j_pinverse = inverse_kinematics(self.n, self.theta_ang, self.end_vol)
            # print('theta_vol:\t\t', theta_vol)
            dtheta = world_dt_reset * self.theta_vol
            # print(dtheta.shape)
            self.theta_ang = self.theta_ang + dtheta
            # print('theta:\n',theta)

            # 正解，经过这个变换之后，新的end去哪里了
            self.Tforworld = forward_kinematics3(self.n, self.theta_ang)
            for i in range(self.n):
                self.jointpose[i] = (
                    torch.cat((self.Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(self.Tforworld[i]))).tolist())
                self.jointpose[i][1] += 0.9
            # end = torch.cat((Tforworld[n - 1][0:3, 3], rotationMatrixToEulerAngles(Tforworld[n - 1])))
            # end[1] += 0.9
            self.end = self.jointpose[self.n - 1]
            # print('end:\t', self.end[0:3])
            # https://blog.csdn.net/qq_20406597/article/details/103259933
            # plt.plot(self.jointpose[:, 0], self.jointpose[:, 1], self.jointpose[:, 2], '-o', c='orangered', linewidth=3)
            # plt.plot(self.jointpose[:, 0], self.jointpose[:, 1], self.jointpose[:, 2], '-o', c='cornflowerblue',linewidth=2)
            # plt.pause(0.0001)

    def reset(self, goal_position):
        # print('============reset============')
        self.goal_position = goal_position

        # 功能：重置robot，让robot的末端先到达goal_position，返回到达时的状态

        # 全是self参数：
        # 关节个数n、关节角度theta_ang、各个关节具体位置和姿态jointpose、末端位置和姿态end、目标点goal_position、各个关节的速度theta_vol、距离distance、
        # 转换矩阵Tforworld、末端速度end_vol、雅可比矩阵j、雅可比矩阵的伪逆j_pinverse

        # 关节个数，一共13个，前12个关节每两个组成一个手肘。最后一个是加上的杆子，最后一个theta不用
        self.n = n_forcrawling + 1
        # 关节角度，初始给个值，避免奇异值
        self.theta_ang = [math.pi / 6] * self.n
        self.theta_ang[self.n - 1] = 0
        self.theta_ang[5] = math.pi / 5
        # 用正解算转换矩阵，一共13个
        self.Tforworld = forward_kinematics3(self.n, self.theta_ang)
        # 算各个关节的位置，用转换矩阵乘，然后保留结果
        self.jointpose = []
        for i in range(self.n):
            self.jointpose.append(
                torch.cat((self.Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(self.Tforworld[i]))).tolist())
            self.jointpose[i][1] += 0.9
        self.jointpose = np.array(self.jointpose)
        # 末端位置。就是各个关节位置中的最后一个
        self.end = self.jointpose[self.n - 1]

        # 各个关节的速度
        self.theta_vol = [0.0] * self.n

        # 末端和目标点的距离
        self.distance = ((self.end[0] - self.goal_position[0]) ** 2 +
                         (self.end[1] - self.goal_position[1]) ** 2 +
                         (self.end[2] - self.goal_position[2]) ** 2) ** 0.5

        # ================================================
        # 重置，先让末端到目标位置
        # ================================================
        reach_limit = 3000
        count = 0
        while cal_distance(self.goal_position, self.end[0:3]) > 0.01 and count < reach_limit:
            count = count + 1

            # 规划速度，速度就是直着过去，转动的速度都为0。速度大小为0.01，方向就是直接相连
            # 这么写是因为麻烦的类型问题
            self.end_vol = [0, 0, 0, 0, 0, 0]
            for i in range(3):
                self.end_vol[i] = self.goal_position[i] - self.end[i]
            # 算模长
            end_vol_magnitude = cal_distance(self.end_vol[0:3], [0, 0, 0])
            # end_vol = end_vol / end_vol_magnitude
            for i in range(3):
                self.end_vol[i] = self.end_vol[i] / end_vol_magnitude
                self.end_vol[i] = self.end_vol[i] * 0.01
            # 逆解，输出末端速度和雅可比伪逆
            self.theta_vol, self.j, self.j_pinverse = inverse_kinematics(self.n, self.theta_ang, self.end_vol)
            # print('theta_vol:\t\t', theta_vol)
            dtheta = world_dt_reset * self.theta_vol
            # print(dtheta.shape)
            self.theta_ang = self.theta_ang + dtheta
            # print('theta:\n',theta)

            # 正解，经过这个变换之后，新的end去哪里了
            self.Tforworld = forward_kinematics3(self.n, self.theta_ang)
            for i in range(self.n):
                self.jointpose[i] = (
                    torch.cat((self.Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(self.Tforworld[i]))).tolist())
                self.jointpose[i][1] += 0.9
            # end = torch.cat((Tforworld[n - 1][0:3, 3], rotationMatrixToEulerAngles(Tforworld[n - 1])))
            # end[1] += 0.9
            self.end = self.jointpose[self.n - 1]
            # print('end:\t', self.end[0:3])
            # !!!画出一开始重置过程的地方
            # plt.plot(self.jointpose[:, 0], self.jointpose[:, 1], self.jointpose[:, 2], '--*', c='blue')
            # plt.pause(0.0001)

        # print(type(self.goal_position),type(self.theta_ang))
        observe_state = self.goal_position + self.theta_ang[0:-1].tolist()

        # plt.ioff()
        # print('###\t', self.jointpose)
        # plt.plot(self.jointpose[:, 0], self.jointpose[:, 1], self.jointpose[:, 2], '--*', c='blue')
        # TODO：怎么让规划的那个一直显示？
        # plt.show()
        print('reset finished!')

        return observe_state

    def reset_new(self, goal_position):
        self.n = 13
        self.goal_position = goal_position
        # self.theta_ang = [0.07816026, 6.61513664, 3.37798899, 3.14582907, 6.71403822, 6.64047393, 6.0516544,  6.21873271, 6.29408374, 0.119184, 3.73173063, 8.46323073, 6.72184069]
        self.theta_ang = [0.16824095, 6.47614398, 3.38223694, 3.06155742, 6.84330981, 6.51678899, 6.1491898, 6.204893,
                          6.25156927, 0.13128138, 3.6122703, 8.42081658, 6.66044573]
        self.Tforworld = forward_kinematics3(self.n, self.theta_ang)
        self.jointpose = []
        for i in range(self.n):
            self.jointpose.append(
                torch.cat((self.Tforworld[i][0:3, 3], rotationMatrixToEulerAngles(self.Tforworld[i]))).tolist())
            self.jointpose[i][1] += 0.9
        self.jointpose = np.array(self.jointpose)
        # 末端位置。就是各个关节位置中的最后一个
        self.end = self.jointpose[self.n - 1]

        # 各个关节的速度
        self.theta_vol = [0.0] * self.n

        # 末端和目标点的距离
        self.distance = ((self.end[0] - self.goal_position[0]) ** 2 +
                         (self.end[1] - self.goal_position[1]) ** 2 +
                         (self.end[2] - self.goal_position[2]) ** 2) ** 0.5

        # 走个求逆解的过场，初始化一下一些参数
        self.end_vol = [0, 0, 0, 0, 0, 0]
        for i in range(3):
            self.end_vol[i] = self.goal_position[i] - self.end[i]
        # 算模长
        end_vol_magnitude = cal_distance(self.end_vol[0:3], [0, 0, 0])
        # end_vol = end_vol / end_vol_magnitude
        for i in range(3):
            self.end_vol[i] = self.end_vol[i] / end_vol_magnitude
            self.end_vol[i] = self.end_vol[i] * 0.01
        # 逆解，输出末端速度和雅可比伪逆
        self.theta_vol, self.j, self.j_pinverse = inverse_kinematics(self.n, self.theta_ang, self.end_vol)
        # print('theta_vol:\t\t', theta_vol)
        # dtheta = world_dt_reset * self.theta_vol
        # # print(dtheta.shape)
        # self.theta_ang = self.theta_ang + dtheta

        return self.goal_position + self.theta_ang[0:-1]

    def get_reward(self, action=None):
        count = 0
        collide = False

        k = 10
        for i in range(len(self.jointpose) - 2):
            dots = sample_inline(self.jointpose[i][0:3], self.jointpose[i + 2][0:3], k)
            for j in range(len(dots)):
                dot = dots[j]
                if check_point_collision_incomplexpipe(dot[0], dot[1], dot[2], h, r1, r1 / 2, x0bucket, y0bucket,
                                                       z0bucket):
                    collide = True
                    count += 1
                    # print('ohno!\n')

        reward = -count * (1 / k)
        return reward, collide

    def get_info(self):
        return None

    def solved(self):

        return


# ================================================================================================

if __name__ == '__main__':
    dots_inside = generate_dots_inside_complex()

    # plt.ion()
    # fig = plt.figure(1)
    # ax = Axes3D(fig)
    # ax.set_xlim([0, 0.6])
    # ax.set_ylim([0.8, 1])
    # ax.set_zlim([-0.1, 0.1])

    arm_robot1 = arm_robot()
    agent = DDPGAgent(observationSpace, actionSpace, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episodeCount = 0
    # episodeLimit = 5000
    episodeLimit = 500
    solved = False
    # solved = True
    # agent.load_models()

    stepsPerEpisode = 450
    episodeScore = 0
    episodeScoreList = []
    test = False

    # =====================================================================
    # 桶
    # 桶的尺寸
    h = 0.6
    r1 = 0.075
    r2 = r1 / 2
    # 桶的位置，桶的轴和x平行
    x0bucket = 0.25
    y0bucket = 0.9
    z0bucket = 0

    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h_new = np.linspace(0, h, 20)  # 把高度1均分为20份
    xsurface = np.outer(np.ones(len(u)), h_new)
    ysurface1 = np.outer(np.cos(u) * r1, np.ones(len(h_new)))
    zsurface1 = np.outer(np.sin(u) * r1, np.ones(len(h_new)))
    ysurface2 = np.outer(np.cos(u) * r2, np.ones(len(h_new)))
    zsurface2 = np.outer(np.sin(u) * r2, np.ones(len(h_new)))

    for i in range(xsurface.shape[0]):
        for j in range(xsurface.shape[1]):
            xsurface[i][j] += x0bucket

    for i in range(ysurface1.shape[0]):
        for j in range(ysurface1.shape[1]):
            ysurface1[i][j] += y0bucket

    for i in range(zsurface1.shape[0]):
        for j in range(zsurface1.shape[1]):
            zsurface1[i][j] += z0bucket

    for i in range(ysurface2.shape[0]):
        for j in range(ysurface2.shape[1]):
            ysurface2[i][j] += y0bucket

    for i in range(zsurface2.shape[0]):
        for j in range(zsurface2.shape[1]):
            zsurface2[i][j] += z0bucket

    # ax.plot_surface(xsurface, ysurface1, zsurface1, cmap=plt.get_cmap('ocean'), alpha=0.5)
    # ax.plot_surface(xsurface, ysurface2, zsurface2, cmap=plt.get_cmap('ocean'), alpha=0.5)
    # ax.plot_surface(xsurface, ysurface1, zsurface1, cmap=plt.get_cmap('gray'), alpha=0.5)
    # ax.plot_surface(xsurface, ysurface2, zsurface2, cmap=plt.get_cmap('gray'), alpha=0.5)
    # plt.show()
    # =====================================================================

    # time_consumed = []
    # for _ in range(50):
    # t0 = time.time()

    time_consumed = []
    # dot_index = int(len(dots_inside) / 2)
    # dot_index = random.randint(0, len(dots_inside) - 1)
    # dot_index = 208
    # dot_index = 416
    dot_index = 1455
    print('dot_index:', dot_index)
    print('dot:', dots_inside[dot_index])

    # dot_index = 900
    # dot_index = 450
    # state_stored = arm_robot1.reset(dots_inside[dot_index])
    while not solved and episodeCount < episodeLimit:
        # goal_position = generate_goal(h, r, x0bucket, y0bucket, z0bucket)
        # goal_position = [0.46, 0.9 + 0.03, -0.0285]

        # state = arm_robot1.reset_new(goal_position)
        # 如果对于一个点已经走得比较好了，就换下一个点
        if len(episodeScoreList) > 10:
            if np.sum(episodeScoreList[-10:]) == max(episodeScoreList) * 10 or len(episodeScoreList) > 50:
                print('====================\n Next Dot!\n====================')
                dot_index = random.randint(0, len(dots_inside) - 1)
                print('dot_index:', dot_index)
                print('dot:', dots_inside[dot_index])

                episodeScoreList = []
        state = arm_robot1.reset(dots_inside[dot_index])
        # state_stored = arm_robot1.reset(dots_inside[dot_index])
        # state = state_stored
        episodeScore = 0

        t0 = time.time()
        # 这个循环走完，意味着走完了一个episode。一个episode对应一个goal_position的优化。找到一个不碰撞的或者超过次数限制就返回
        for step in range(stepsPerEpisode):
            selectedAction = agent.choose_action_train(state)
            selectedAction_copy = []
            for i in range(len(selectedAction)):
                # TODO：dh的范围
                # 0.8的范围接近随机了，因为奇异值，导致变化很大，和自运动没关了
                # selectedAction_copy.append(normalizeToRange(selectedAction[i], -1, 1, -0.8, 0.8, True))
                selectedAction_copy.append(normalizeToRange(selectedAction[i], -1, 1, -0.1, 0.1, True))
            selectedAction = selectedAction_copy

            arm_robot1.use_message_data(selectedAction)  # means do the action to reach the next step
            newState = arm_robot1.observe()

            # plt.plot(arm_robot1.jointpose[:, 0], arm_robot1.jointpose[:, 1], arm_robot1.jointpose[:, 2], '--*')
            # TODO：画图停留时间
            # plt.pause(0.0001)
            # plt.pause(0.05)

            reward, collided = arm_robot1.get_reward()
            done = False
            # done = True
            if not collided and cal_distance(arm_robot1.end[:3], dots_inside[dot_index]) < 0.01:
                done = True

                theta_goal = arm_robot1.theta_ang[:-1]
                print('theta_agnles:', theta_goal)
                Ts = snake_forcrawling.fkine_all(theta_goal)
                for i in range(len(Ts)):
                    if Ts[i].t[0] < x0bucket and Ts[i + 1].t[0] > x0bucket:
                        i_goal = i
                        print('i_goal:', i_goal)
                if i_goal+1>n_forcrawling or i_goal+3>n_forcrawling:
                    print('i_goal=', i_goal, ',do not need to crawl.')
                else:
                    # 保证靠近基座的要有6个以上自由度，否则就算没搜索到
                    # 要有6个以上自由度，且不能超出工作空间
                    t_needtoreach_forbase = Ts[i_goal + 1].t - Ts[i_goal + 3].t + Ts[i_goal + 1].t
                    if cal_distance(t_needtoreach_forbase, [0, 0, 0]) > l_forcrawling * (
                            i_goal + 1) / 2 - 1 * l_forcrawling \
                            or i_goal < 5 \
                            or cal_distance(Ts[i_goal + 1].t, [0, 0, 0]) > l_forcrawling * (
                            i_goal + 1) / 2 - 1 * l_forcrawling:
                        done = False
                        print('靠近基座的自由度太少了！')
                        print('workspace:', l_forcrawling * (i_goal + 1) / 2)
                        print('distance1:', cal_distance(t_needtoreach_forbase, [0, 0, 0]))
                        print('distance2:', cal_distance(Ts[i_goal + 1].t, [0, 0, 0]))
                    else:
                        print('workspace:', l_forcrawling * (i_goal + 1) / 2)
                        print('distance:', cal_distance(t_needtoreach_forbase, [0, 0, 0]))
                        print('distance:', cal_distance(Ts[i_goal + 1].t, [0, 0, 0]))
                        theta_goal = get_smaller_theta(theta_goal)
                        snake_forcrawling.teach(theta_goal)
                        solq = crawling(snake_forcrawling, theta_goal, i_goal)
                        dis_deviation,dis_list_forplot = crawling_evaluation(snake_forcrawling, theta_goal, i_goal, solq, x0bucket)
                        print('最大偏离距离：', dis_deviation)
                        # snake_forcrawling.plot(solq).hold()
                        snake_forcrawling.plot(solq)
                        reward -= dis_deviation / l_forcrawling

            agent.remember(state, selectedAction, reward, newState, int(done))
            episodeScore += reward
            agent.learn()

            if done or step == stepsPerEpisode - 1:
                episodeScoreList.append(episodeScore)
                solved = arm_robot1.solved()
                time_consumed.append(time.time() - t0)
                print('time consumed', time_consumed[-1])
                print('score:', episodeScore)
                print('step count:', step, '\n====================')
                # if step != stepsPerEpisode - 1:
                #     plt.plot(arm_robot1.jointpose[:, 0], arm_robot1.jointpose[:, 1], arm_robot1.jointpose[:, 2], '--*')
                #     plt.pause(0.5)
                break

            state = newState

        episodeCount += 1
        if episodeCount % 200 == 0:
            agent.save_models()

    print('all_time_consumed', time_consumed)
    print('Average_time_consumed:', np.mean(time_consumed))
