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

# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math

HEIGHT = 6
WIDTH = 9

UP_ACTION = 0
RIGHT_ACTION = 1
DOWN_ACTION = 2
LEFT_ACTION = 3

START_STATE = [5, 3]
GOAL_STATE = [0, 8]

Q = np.zeros([HEIGHT, WIDTH, 4])
MODEL = dict()

CHANGED = False

def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative rewards')
    plt.legend(loc='upper right')
    plt.savefig('./blocking_maze.png')
    plt.close()


def reset():
    global Q, MODEL, CHANGED
    Q = np.zeros([HEIGHT, WIDTH, 4])
    MODEL = dict()
    CHANGED = False


def change():
    global CHANGED
    CHANGED = True


def is_forbidden(state):
    global CHANGED
    x, y = state
    if x < 0 or x > 5 or y < 0 or y > 8:
        return True
    if not CHANGED:
        if x == 3 and y < 8:
            return True
    else:
        if x == 3 and y > 0:
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

def episode(count, repeats, plus=False):
    global Q, MODEL
    state = START_STATE
    epsilon = 0.1
    gamma = 0.95
    alpha = 0.7
    k = 0.0001
    steps = 0
    trajectory = set()
    rewards = list()
    while True:
        if steps == 1000:
            change()
        if steps == count:
            break
        if state == GOAL_STATE:
            state = START_STATE
            continue
        if np.random.binomial(1, epsilon) == 1:
            action = random.randint(0, 3)
        else:                                 
            action = np.random.choice([a for a, v in enumerate(Q[state[0]][state[1]]) if v == np.max(Q[state[0]][state[1]])]) 
        trajectory.add('%d_%d_%d' % (state[0], state[1], action))
        new_state = step(state, action)
        reward = get_reward(new_state)
        ma = np.random.choice([a for a, v in enumerate(Q[new_state[0]][new_state[1]]) if v == np.max(Q[new_state[0]][new_state[1]])])
        rewards.append(reward)        
        Q[state[0]][state[1]][action] += alpha * (reward + gamma * Q[new_state[0]][new_state[1]][ma] - Q[state[0]][state[1]][action])
        for i in range(4):
            if i != action and ('%d_%d_%d'%(state[0], state[1], i) not in MODEL.keys()):
                trajectory.add('%d_%d_%d' % (state[0], state[1], i))
                MODEL['%d_%d_%d'%(state[0], state[1], i)] = [0.0, state, 0]
        MODEL['%d_%d_%d'%(state[0], state[1], action)] = [reward, new_state, steps]
        for _ in range(repeats):
            s = np.random.choice(list(trajectory))
            ss = str.split(s, '_', -1)
            _state = [int(ss[0]), int(ss[1])]
            _action = int(ss[2])
            r, next_state, _t = MODEL['%d_%d_%d'%(_state[0], _state[1], _action)]
            _ma = np.random.choice([a for a, v in enumerate(Q[next_state[0]][next_state[1]]) if v == np.max(Q[next_state[0]][next_state[1]])])
            upd = 0.0
            if plus:
                upd = k * math.sqrt(steps - _t)
            Q[_state[0]][_state[1]][_action] += alpha * (r + upd + gamma * Q[next_state[0]][next_state[1]][_ma] - Q[_state[0]][_state[1]][_action])
        state = new_state
        steps += 1
    return rewards

def run():
    run = 20
    count = 3000
    repeats = 20
    data = np.zeros([2, count])
    for _ in tqdm(range(run)):
        data[0] += np.add.accumulate(episode(count, repeats, False))
        reset()
        data[1] += np.add.accumulate(episode(count, repeats, True))
        reset()
    image(np.arange(count) + 1, data/run, ['dyna-q', 'dyna-q+'])


if __name__ == "__main__":
    run()
