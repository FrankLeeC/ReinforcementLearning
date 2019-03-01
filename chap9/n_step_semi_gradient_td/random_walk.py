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
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

STATE_COUNT = 1002
LEFT_ACTION = -1
RIGHT_ACTION = 1
START_STATE = 500
TRUE_VALUE = np.arange(-1001, 1003, 2) / 1001
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0.0


def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('alpha')
    plt.ylabel('avg rms')
    # plt.ylim(0.25, 0.6)
    plt.legend(loc='lower right')
    plt.savefig('./random_walk.png')
    plt.close()


def get_action(state):
    if np.random.binomial(1, 0.5) == 1:
        return LEFT_ACTION
    return RIGHT_ACTION


def step(state, action):
    r = random.randint(1, 50)
    if action == LEFT_ACTION:
        new_state = state - r
        if new_state <= 0:
            return -1, 0
        return 0, new_state
    new_state = state + r
    if new_state >= 1001:
        return 1, 1001
    return 0, new_state


W = np.zeros(20)
def reset():
    global W
    W = np.zeros(20)


def get_value(state):
    if state == 0 or state == 1001:
        return 0
    global W
    a = (state - 1) // 50
    return W[a]


def n_step(gamma, alpha, n):
    global W
    state = START_STATE
    states = [state]
    rewards = [0]
    T = float('inf')
    t = 0
    while True:
        if t < T:
            action = get_action(state)
            reward, next_state = step(state, action)
            rewards.append(reward)
            states.append(next_state)
            if next_state == 0 or next_state == 1001:
                T = t+1
        tau = t-n+1
        if tau >= 0:
            g = 0.0
            end = min(tau + n, T)
            start = tau + 1
            for j in range(end - start + 1):
                i = j + start
                g += (gamma ** (i - tau - 1)) * rewards[i]
            if tau + n < T:
                g += (gamma ** n)*get_value(states[tau+n])
            if states[tau] != 0 and states[tau] != 1001:
                W[(states[tau]-1)//50] += alpha * (g - get_value(states[tau])) * 1
        t += 1
        state = next_state
        if tau == T-1:
            break


def get_error():
    global W
    global TRUE_VALUE
    w = [0.0]
    w.extend(np.asarray(np.repeat(W, 50)))
    w.append(0.0)
    return np.sqrt(np.sum((np.asarray(w) - TRUE_VALUE) ** 2) / 1000)
    

def run(n, alpha, episodes):
    reset()
    gamma = 1.0
    err = 0.0
    for _ in range(episodes):
        n_step(gamma, alpha, n)
        err += get_error()
    return err


def main():
    count = 100
    episodes = 10
    ns = np.power(2, np.arange(0, 10))
    alpha = np.arange(0.0, 1.1, 0.1)
    errs = np.zeros([len(ns), len(alpha)])
    for _ in tqdm(range(count)):
        for i, a in enumerate(alpha):
            for j, n in enumerate(ns):
                errs[j][i] += run(n, a, episodes)
    errs /= episodes * count
    labels = ['n=%d' % n for n in ns]
    image(alpha, errs, labels)


if __name__ == "__main__":
    main()