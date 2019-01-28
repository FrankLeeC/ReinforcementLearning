'''
MIT License

Copyright (c) 2018 Frank Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# -*- coding: utf-8 -*-

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

HEIGHT = 6
WIDTH = 9

UP_ACTION = 0
RIGHT_ACTION = 1
DOWN_ACTION = 2
LEFT_ACTION = 3

START_STATE = [2, 0]
GOAL_STATE = [0, 8]

Q = np.zeros([HEIGHT, WIDTH, 4])
MODEL = dict()

def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.legend(loc='upper right')
    plt.savefig('./maze.png')
    plt.close()


def reset():
    global Q, MODEL
    Q = np.zeros([HEIGHT, WIDTH, 4])
    MODEL = dict()
    # random.seed(47)
    # np.random.seed(79)


def is_forbidden(state):
    x, y = state
    if x < 0 or x > 5 or y < 0 or y > 8:
        return True
    if x in [0, 1, 2] and y == 7:
        return True
    if x in [1, 2, 3] and y == 2:
        return True
    if x == 4 and y == 5:
        return True
    return False

def step(state, action):
    if state == GOAL_STATE:
        return START_STATE
    x, y = state
    new_state = state
    if action == UP_ACTION:
        new_state = [x-1, y]
    elif action == RIGHT_ACTION:
        new_state = [x, y+1]
    elif action == DOWN_ACTION:
        new_state = [x+1, y]
    else:
        new_state = [x, y-1]
    if is_forbidden(new_state):
        return state
    return new_state

def get_reward(state):
    if state == GOAL_STATE:
        return 1.0
    return 0.0

def get_action(state, epsilon):
    global Q
    if np.random.binomial(1, epsilon) == 1:
        return random.randint(0, 3)
    return np.random.choice([a for a, v in enumerate(Q[state[0]][state[1]]) if v == np.max(Q[state[0]][state[1]])])

def episode(count, repeats):
    global Q, MODEL
    state = START_STATE
    epsilon = 0.1
    gamma = 0.95
    alpha = 0.1

    steps = 0
    data = []
    trajectory = dict()
    while True:
        if len(data) == count:
            break
        if state == GOAL_STATE:
            data.append(steps)
            steps = 0
            state = START_STATE
            continue
        if np.random.binomial(1, epsilon) == 1:
            action = random.randint(0, 3)
        else:
            action = np.random.choice([a for a, v in enumerate(Q[state[0]][state[1]]) if v == np.max(Q[state[0]][state[1]])])
        tmp = trajectory.get('%d_%d' % (state[0], state[1]), list())
        tmp.append(action)
        trajectory['%d_%d' % (state[0], state[1])] = tmp
        new_state = step(state, action)
        reward = get_reward(new_state)
        ma = np.random.choice([a for a, v in enumerate(Q[new_state[0]][new_state[1]]) if v == np.max(Q[new_state[0]][new_state[1]])])
        if state != GOAL_STATE:
            Q[state[0]][state[1]][action] += alpha * (reward + gamma * Q[new_state[0]][new_state[1]][ma] - Q[state[0]][state[1]][action])
            MODEL['%d_%d_%d'%(state[0], state[1], action)] = [reward, new_state]
            for _ in range(repeats):
                k = list(trajectory.keys())
                s = k[np.random.choice(len(k))]
                # s = np.random.choice(trajectory.keys())
                ss = str.split(s, '_')
                state = [int(ss[0]), int(ss[1])]
                action = np.random.choice(trajectory[s])
                reward, next_state = MODEL['%d_%d_%d'%(state[0], state[1], action)]
                ma = np.random.choice([a for a, v in enumerate(Q[new_state[0]][new_state[1]]) if v == np.max(Q[new_state[0]][new_state[1]])])
                Q[state[0]][state[1]][action] += alpha * (reward + gamma * Q[next_state[0]][next_state[1]][ma] - Q[state[0]][state[1]][action])
        state = new_state
        steps += 1
    return data

def run():
    run = 100
    data = np.zeros([3, 50])
    for _ in tqdm(range(run)):
        d = []
        repeats = [0, 5, 50]
        count = 50
        for i in range(len(repeats)):
            d.append(episode(count, repeats[i]))
            reset()
        data += np.array(d)
    image(np.arange(count) + 1, data/run, ['n=0', 'n=5', 'n=50'])


if __name__ == "__main__":
    run()
