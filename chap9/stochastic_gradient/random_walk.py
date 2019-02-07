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
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return -1
    return 1


w = np.zeros(10)
def update(state, delta):
    global w
    w[(state-1)//100] += delta
    return


def image(x, y, l, ylabel, name):
    plt.subplot(1, 1, 1)
    for i in range(len(y)):
        plt.plot(x, y[i], label=l[i])
    plt.xlabel('State')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(name)
    plt.close()

def get_value(state):
    global w
    i = (state - 1 ) // 100
    return w[i]


def step(state, action):
    r = random.randint(1,200)
    if r < 101:
        new_state = state - r
    else:
        new_state = state + r - 100
    if new_state < 1:
        return 0, -1
    elif new_state > 1000:
        return 1001, 1
    return new_state, 0


def is_end(state):
    if state == 0 or state == 1001:
        return True
    return False


def run(episodes):
    alpha = 2e-5
    gamma = 1.0
    states = np.zeros((1000, 1))
    for _ in tqdm(range(episodes)):
        reward = 0.0
        state = 500
        trajectory = [state]
        rewards = []
        while not is_end(state):
            states[state-1, 0] += 1
            action = get_action()
            new_state, reward = step(state, action)
            rewards.append(reward)
            trajectory.append(new_state)
            state = new_state
        g = 0.0
        gs = []
        for j in range(len(trajectory)-1):
            i = len(trajectory) - 1 - j - 1
            r = rewards[i]
            g = gamma * g + r
            gs.append(g)
        for i in range(len(trajectory) - 1):
            s = trajectory[i]
            d = 1
            delta = alpha * (gs[i] - get_value(s)) * d
            update(s, delta)
    return states

def main():
    true_value = (np.arange(-998, 1003, 2) / 1001.0)[0:-1]
    true_value = np.reshape(true_value, (1000, 1))
    episodes = 100000
    states = run(episodes)
    data = np.zeros((1000, 1))
    for i in range(1000):
        j = i // 100
        data[i, 0] = w[j]
    states /= np.sum(states)
    image(np.reshape(np.arange(1000) + 1, (1000, 1)), [data, true_value], ['Approximate MC value', 'True Value'], 'Value Scale', 'value.png')
    image(np.reshape(np.arange(1000) + 1, (1000, 1)), [states], ['State Distribution'], 'State Distribution', 'distribution.png')


if __name__ == "__main__":
    main()
        
        