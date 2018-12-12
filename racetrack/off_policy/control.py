# -*- coding: utf-8 -*-

import numpy as np
import random


'''
action:
0  -1, -1
1  -1, 0
2  -1, 1
3  0, -1
4  0, 0
5  0, 1
6  1, -1
7  1, 0
8  1, 1

velocity(index):
1(0), 2(1), 3(2), 4(3)
'''

Q = np.random.randn(17, 32, 5, 5, 9)  # 17*32 表示位置  4*4表示当前速度  9表示动作
C = np.random.randn(17, 32, 5, 5, 9)  # 17*32 表示位置  4*4表示当前速度  9表示动作
POLICY = np.zeros([17, 32, 5, 5])


def init_policy():
    global POLICY, Q
    for x in range(17):
        for y in range(32):
            for v in range(5):
                for h in range(5):
                    POLICY[x][y][v][h] = np.argmax(Q[x][y][v][h])


def format_action(actions):
    v, h = actions
    return 3 * (v+1) + h + 1


class State:

    def __init__(self, position, velocity):
        self.x = position[0]
        self.y = position[1]
        self.vertical_velocity = velocity[0]
        self.horizontal_velocity = velocity[1]


class Episode:

    def __init__(self):
        self.states_list = []
        self.action_list = []
        self.reward_list = []

    def add_state(self, state):
        self.states_list.append(state)

    def add_action(self, action):
        self.action_list.append(action)

    def add_reward(self, reward):
        self.reward_list.append(reward)
        
    def states(self):
        return self.states_list

    def actions(self):
       return self.action_list

    def rewards(self):
        return self.reward_list


def is_forbidden(state):
    x, y = state.x, state.y
    if x > 31 or x < 0 or y > 16 or y < 0:
        return True
    if x == 0:
        return y < 3
    if x == 1:
        return y < 2
    if x == 2:
        return y < 2
    if x == 3:
        return y == 0
    if y == 0:
        return x >= 14
    if y == 1:
        return x >= 22
    if y == 2:
        return x >= 29
    if y == 9:
        return x >= 7
    if x >= 6 and x <= 31:
        return y >= 10 and y <= 16
    return False


def is_finished(state):
    return state.x < 6 and state.y == 16


def random_start():
    y = random.randint(3, 8)
    position = (31, y)
    velocity = (0, 0)
    return State(position, velocity)


def new_state(state, action):
    new_v = state.vertical_velocity + action[0]
    new_h = state.horizontal_velocity + action[1]
    new_x = state.x - new_v  # 垂直方向1往上，-1往下 需要减法
    new_y = state.y + new_h  # 水平方向则是1向右，-1往左 需要加法
    return State([new_x, new_y], [new_v, new_h])


def behavior_action(state):
    vertical_velocity = 0
    horizontal_velocity = 0
    random.seed()
    if state.vertical_velocity > 0 and state.vertical_velocity < 4:
        vertical_velocity = int(random.random() * 3) - 1   # {-1, 0, 1}
    elif state.vertical_velocity == 0:
        vertical_velocity = int(random.random() * 2)   # {0, 1}
    elif state.vertical_velocity == 4:
        vertical_velocity = int(random.random() * 2) - 1  # {-1, 0}
    else:
        raise IndexError(state.vertical_velocity)

    if state.horizontal_velocity > 0 and state.horizontal_velocity < 4:
        horizontal_velocity = int(random.random() * 3) - 1   # {-1, 0, 1}
    elif state.horizontal_velocity == 0:
        horizontal_velocity = int(random.random() * 2)   # {0, 1}
    elif state.horizontal_velocity == 4:
        horizontal_velocity = int(random.random() * 2) - 1  # {-1, 0}
    else:
        raise IndexError(state.horizontal_velocity)
    return [vertical_velocity, horizontal_velocity]


def generate_episode():
    episode = Episode()
    state = random_start()
    episode.add_state(state)
    while True:
        action = behavior_action(state)
        episode.add_action(action)
        state = new_state(state, action)
        if is_finished(state):
            episode.add_reward(0)
            break
        episode.add_reward(-1)
        if is_forbidden(state):
            state = random_start()
        episode.add_state(state)
    return episode


def run(counts):
    global POLICY, Q, C
    init_policy()
    for i in range(counts):
        episode = generate_episode()
        if (i+1) % 100 == 0:
            print('=============================', i+1)
        states = episode.states()
        actions = episode.actions()
        rewards = episode.rewards()
        l = len(states)
        g = 0.0
        w = 1.0
        for j in range(l):
            i = l - j - 1
            g = 0.9 * g + rewards[i]
            state = states[i]
            action = actions[i]
            x = state.x
            y = state.y
            vv = state.vertical_velocity
            hv = state.horizontal_velocity
            a = format_action(action)
            C[x][y][vv][hv][a] += w
            Q[x][y][vv][hv][a] += w * (g - Q[x][y][vv][hv][a]) / C[x][y][vv][hv][a]
            idx = np.argmax(Q[x][y][vv][hv])
            POLICY[x][y][vv][hv] = idx
            if idx != a:
                break
            w *= 9


def decouple_action(a):
    h = (a % 3) - 1
    v = (a - h - 1) / 3 - 1
    return v, h


def act_str(action, direction):
    if direction == 'vertical':
        if action > 0:
            return '⬆️'
        if action == 0:
            return '|'
        return '⬇️'
    if direction == 'horizontal':
        if action > 0:
            return '➡️'
        if action == 0:
            return '|'
        return '⬅️' 


def print_out():
    global POLICY
    f = open('./policy.txt', mode='w', encoding='utf-8')
    for x in range(32):
        for y in range(17):
            for v in range(5):
                for h in range(5):
                    state = State([x, y], [0, 0])
                    if is_forbidden(state):
                        continue
                    a = POLICY[y][x][v][h]
                    vv, hv = decouple_action(a)
                    vs, hs = act_str(vv, 'vertical'), act_str(hv, 'horizontal')
                    nv, nh = v + vv, h + hv
                    nx, ny = x - nv, y + nh
                    s = '%d_%d   %d_%d --> %s %s -->  %d_%d  %d_%d\n' % (x, y, v, h, vs, hs, nx, ny, nv, nh)
                    f.write(s)
                    f.flush()
    f.close()


if __name__ == "__main__":
    run(300)
    print_out()
