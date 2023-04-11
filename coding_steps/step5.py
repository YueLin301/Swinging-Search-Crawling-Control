# 第5步：强化学习开始学习dh
import time
from arm_kine.arm_1_forward_kinematics import forward_kinematics3, rotationMatrixToEulerAngles
from arm_kine.arm_3_inverse_kinematics import inverse_kinematics
from step4 import generate_goal, check_point_collision
import numpy as np
import torch
import math
# import random

from RLagent.DDPG_agent import DDPGAgent

from step1 import step1_reachgoal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================================================================
# 3个，表示目标点的位置；12个，表示12个关节的角度。第13个关节的角度没用
observationSpace = (15,)
# 12个，表示12个dh，用来自运动
actionSpace = (12,)
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

        goal_position = [0.53, 0.9 + 0.03, 0.03]
        self.reset_new(goal_position)

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
            dh.append(i * 0.4)
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

        reach_limit = 100
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

    def reset(self, goal_position):
        print('============reset============')
        self.goal_position = goal_position

        # 功能：重置robot，让robot的末端先到达goal_position，返回到达时的状态

        # 全是self参数：
        # 关节个数n、关节角度theta_ang、各个关节具体位置和姿态jointpose、末端位置和姿态end、目标点goal_position、各个关节的速度theta_vol、距离distance、
        # 转换矩阵Tforworld、末端速度end_vol、雅可比矩阵j、雅可比矩阵的伪逆j_pinverse

        # 关节个数，一共13个，前12个关节每两个组成一个手肘。最后一个是加上的杆子，最后一个theta不用
        self.n = 13
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
            print('end:\t', self.end[0:3])

        # print(type(self.goal_position),type(self.theta_ang))
        observe_state = self.goal_position + self.theta_ang[0:-1].tolist()

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
        collide = 0
        # 输出碰撞的关节，检测刚伸入的那两个点，讨论一下（看笔记
        for i in range(len(self.jointpose) - 1):
            flag = check_point_collision(self.jointpose[i][0], self.jointpose[i][1], self.jointpose[i][2], h, r,
                                         x0bucket, y0bucket, z0bucket)
            if flag:
                count = count + 1
                print("joint " + str(i) + ' collide')
                collide = 1
            # 如果当前没碰，下一个也没碰，但是这两个节点的中间的杆子碰到截口了
            elif (
            not check_point_collision(self.jointpose[i + 1][0], self.jointpose[i + 1][1], self.jointpose[i + 1][2], h,
                                      r, x0bucket, y0bucket, z0bucket)) \
                    and (self.jointpose[i][0] < x0bucket and self.jointpose[i + 1][0] > x0bucket):
                # 判断这两个节点的中间的杆子，有没有碰到截口
                # 这里直接看笔记就的推导吧，SnakeArmPlan1
                ycritical = self.jointpose[i][1] + (self.jointpose[i + 1][1] - self.jointpose[i][1]) * (
                            x0bucket - self.jointpose[i][0]) / (self.jointpose[i + 1][0] - self.jointpose[i][0])
                zcritical = self.jointpose[i][2] + (self.jointpose[i + 1][2] - self.jointpose[i][2]) * (
                            x0bucket - self.jointpose[i][0]) / (self.jointpose[i + 1][0] - self.jointpose[i][0])
                if (ycritical - y0bucket) ** 2 + (zcritical - z0bucket) ** 2 >= r ** 2:
                    # 这样就是碰撞了！！
                    count = count + 1
                    print('>>>>>>>>')
                    print("joint " + str(i) + ' collide')
                    print("joint " + str(i + 1) + ' collide')
                    print('<<<<<<<<')
                    collide = 1
        if not collide:
            print("haha")
            abc = self.theta_ang

            theta_goal = abc[0:12]
            a = dcm2angle(theta_goal)
            if a == 0:
                end = time.time()
                print(end - start)
            else:
                count = count + 1

        reward = -count
        return reward

    def get_info(self):
        return None

    def solved(self):
        # if len(self.episodeScoreList) > 100:  # Over 100 trials thus far

        #         return True
        return False


# ================================================================================================

if __name__ == '__main__':
    start = time.time()
    plt.ion()
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.set_xlim([0, 0.6])
    ax.set_ylim([0.8, 1])
    ax.set_zlim([-0.1, 0.1])

    arm_robot = arm_robot()
    agent = DDPGAgent(observationSpace, actionSpace, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    episodeCount = 0
    episodeLimit = 5000
    solved = False
    # solved = True
    # agent.load_models()

    stepsPerEpisode = 200
    # stepsPerEpisode = 500
    episodeScore = 0
    episodeScoreList = []
    test = False

    # =====================================================================
    # 桶
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

    # plt.ioff()
    ax.plot_surface(xsurface, ysurface, zsurface, cmap=plt.get_cmap('ocean'), alpha=0.5)
    # plt.pause(0.0001)
    plt.show()
    # =====================================================================

    while not solved and episodeCount < episodeLimit:
        # 随机生成目标点，目标点在桶内
        # goal_position = generate_goal(h, r, x0bucket, y0bucket, z0bucket)
        goal_position = [0.46, 0.9 + 0.03, -0.0285]

        # 我要先在reset里，走ste1，先让末端到目标点。目标点要存在state里。
        # 目标点的位置要单独存起来，在robot里共享，用self
        state = arm_robot.reset_new(goal_position)
        episodeScore = 0

        for step in range(stepsPerEpisode):
            selectedAction = agent.choose_action_train(state)

            # 这里地方是表示，身体执行大脑选择好的动作。并算好身体执行完之后，各个东西在哪
            # 这里的动作是dh，所以use_message_data要写成自运动 step2
            arm_robot.use_message_data(selectedAction)  # means do the action to reach the next step

            # newState, reward, done, info = supervisorEnv.step(selectedAction)
            # 刚刚已经算好了各个身体的情况，现在只是输出
            newState = arm_robot.observe()

            plt.plot(arm_robot.jointpose[:, 0], arm_robot.jointpose[:, 1], arm_robot.jointpose[:, 2], '--*')
            plt.pause(0.0001)

            # step4，有几个碰撞，就扣几
            reward = arm_robot.get_reward()
            done = False

            # 其实就是每个时刻都判断碰撞了几个，碰撞得越少越好
            agent.remember(state, selectedAction, reward, newState, int(done))

            episodeScore += reward
            # if abs(episodeScore)>100:
            #     print(reward)
            agent.learn()
            if done or step == stepsPerEpisode - 1:
                episodeScoreList.append(episodeScore)
                solved = arm_robot.solved()
                break

            state = newState

        if arm_robot.test:
            break
        print("=========================================================")
        print("Episode #", episodeCount, "\tscore:", episodeScore)

        episodeCount += 1
        if episodeCount % 200 == 0:
            agent.save_models()

    if not solved and not arm_robot.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    print("Press R to reset.")
    state = arm_robot.reset([0.53, 0.9 + 0.03, 0.03])
    arm_robot.test = True
    episodeScore = 0

    # while True:
    for step in range(stepsPerEpisode):
        selectedAction = agent.choose_action_test(state)
        if arm_robot.distance < 0.04:
            selectedAction = np.zeros(2)
            # state, reward, done, _ = supervisorEnv.step(selectedAction)
            arm_robot.use_message_data(selectedAction)
            break
        # state, reward, done, _ = supervisorEnv.step(selectedAction)
        arm_robot.use_message_data(selectedAction)  # means do the action to reach the next step
        newState = arm_robot.observe()
        reward = arm_robot.get_reward()

        episodeScore += reward  # Accumulate episode reward
        print('end_position:\t\t', arm_robot.endpos_pose, '\tdistance:\t', arm_robot.distance)

        # if done:
        #     print("Reward accumulated =", arm_robot.episodeScore)
        #     supervisor.episodeScore = 0
        #     state = arm_robot.reset()
