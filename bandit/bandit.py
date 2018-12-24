# -*- coding:utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

class Arm:

    def __init__(self, id, real_p):
        self.id = id
        self.real_p = real_p
        self.estimate_p = 0.0
        self.n = 0
        self.gba = 0.0

    def get_used_count(self):
        return self.n

    def get_p(self):
        return self.estimate_p

    def update(self, reward):
        self.n += 1
        self.estimate_p += (reward - self.estimate_p)/self.n

    def get_gba(self):
        return self.gba

    def update_gba(self, p, reward, alpha):
        '''
        当前 arm 是被选择的
        alpha > 0 p = 1-PI(A_t)
        否则
        alpha < 0 p = PI(A_t)
        '''
        self.gba += alpha*(reward-self.estimate_p)*p


class Bendit:

    def __init__(self, e):
        self.epsilon = e
        self.arm_count = 10
        self.arm_list = [Arm(i, np.random.randn()) for i in range(self.arm_count)]
        self.explore_count = 0
        self.exploit_count = 0
        self.select_count = 0

    def get_arm_count(self):
        return self.arm_count

    def select(self, select_method='greedy', c=2.0):
        self.select_count += 1
        '''
        选择一个arm，返回索引
        '''
        if select_method == 'greedy':
            if random.random() <= self.epsilon:
                '''
                探索
                '''
                self.explore_count += 1
                idx = random.randint(0, self.arm_count-1)
                return idx, int(idx == self.best())
            else:
                '''
                利用
                '''
                self.exploit_count += 1
                idx = np.argmax([a.estimate_p for a in self.arm_list])
                return idx, int(idx == self.best())
        elif select_method == 'ucb':
            score = list()
            for i, e in enumerate(self.arm_list):
                if e.get_used_count() == 0:
                    return i, int(i == self.best())
                score.append(e.get_p() + c*np.sqrt(np.log(self.select_count)/e.get_used_count())) 
            idx = np.argmax(score)
            return idx, int(idx == self.best())
        elif select_method == 'gba':
            gba_list = [e.get_gba() for e in self.arm_list]
            p = softmax(gba_list)
            self.p = p
            idx = np.argmax(p)
            return idx, int(idx == self.best())

    def update_gba(self, idx, reward):
        alpha = 0.1
        for i, e in enumerate(self.p):
            if i != idx:
                self.arm_list[i].update_gba(self.p[i], reward, -alpha)
            else:
                self.arm_list[i].update_gba(1-self.p[i], reward, alpha)

    def best(self):
        return np.argmax([a.real_p for a in self.arm_list])

    def act(self, arm_idx):
        reward = np.random.randn() + self.arm_list[arm_idx].real_p
        self.arm_list[arm_idx].update(reward)
        return reward

def softmax(a):
    a = np.exp(a)
    return a/np.sum(a)

def run(e, times, select_method='greedy'):
    b = Bendit(e)
    used_count_list = [0 for i in range(b.get_arm_count())]
    reward = list()
    best = list()
    for i in range(times):
        idx, is_best = b.select(select_method=select_method)
        used_count_list[idx] += 1
        r = b.act(idx)
        reward.append(r)
        if select_method == 'gba':
            b.update_gba(idx, r)
        best.append(is_best)
    return reward, best


def main():
    count = 2000
    times = 1000
    # epsilon = [0.0, 0.01, 0.1]
    # select_method = ['greedy', 'greedy', 'greedy']
    epsilon = [0.1, -1, -1]
    select_method = ['greedy', 'ucb', 'gba']
    reward_list = list()
    best_list = list()
    for i, e in enumerate(epsilon):
        a = list()
        b = list()
        for j in range(count):
            reward, best = run(e, times, select_method[i])
            a.append(reward)
            b.append(best)
        reward_list.append(a)
        best_list.append(b)

    ax = plt.subplot(2, 1, 1)
    for i, r in enumerate(reward_list):
        plt.plot(range(times), np.mean(np.asarray(r), axis=0), label='method=%s'%select_method[i])
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend(loc='upper left', frameon=False)

    ax2 = plt.subplot(2, 1, 2)
    for i, b in enumerate(best_list):
        plt.plot(range(times), np.mean(np.asarray(b), axis=0), label='method=%s'%select_method[i])
    plt.xlabel('steps')
    plt.ylabel('% optimal choice')
    plt.legend(loc='lower right', frameon=False)
    plt.legend()

    # plt.savefig('./bandit.png')
    plt.show(block=True)


if __name__ == '__main__':
    main()

