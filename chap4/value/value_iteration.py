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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy


'''
state: {1, 2, 3, ... 99}
action: {0, 1, 2, min(s, 100-s)}

reward: 0 for all transitions except those on which gamler reaches goal, which is 1.0
discount: 1.0
'''

def get_actions(s):
    return range(min(s, 100 - s)+1)

ph = 0.4

def reward(s):
    if s < 100:
        return 0.0
    return 1.0

def step(s, a, value):
    gamma = 1.0
    return ph * (reward(s+a) + gamma * value[s+a]) + (1-ph) * (reward(s-a) + gamma * value[s-a])

def run2():
    value = np.zeros(101)  # value is the probability of reaching goal in current state
    policy = np.zeros(99)
    epsilon = 1e-9
    while True:
        m = 0.0
        for j in range(99):
            i = j + 1
            oldv = value[i]
            actions = get_actions(i)
            mv = float('-inf')
            ma = 0
            for a in actions:
                v = step(i, a, value)
                if v > mv:
                    mv = v
                    ma = a
            policy[j] = ma
            value[i] = mv
            m = max(m, abs(mv-oldv))
        if m < epsilon:
            break
    return value, policy


def run():
    value = np.random.rand(101) - 1.0  # if value is initialized arbitrarily, it must be less than 0.0, because value is the probability of reaching goal in current state
    value[0] = 0.0
    value[100] = 0.0
    policy = np.zeros(99)
    epsilon = 1e-9
    while True:
        m = 0.0
        for j in range(99):
            i = j + 1
            oldv = value[i]
            actions = get_actions(i)
            mv = float('-inf')
            ma = 0
            for a in actions:
                v = step(i, a, value)
                if v > mv:
                    mv = v
                    ma = a
            policy[j] = ma
            value[i] = mv
            m = max(m, abs(mv-oldv))
        if m < epsilon:
            break
    return value, policy

def plot(value, policy, name):
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(value[1:100])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')

    plt.subplot(2, 1, 2)
    plt.scatter(np.arange(100)[1:], policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    value, policy = run()
    plot(value, policy, 'value_iteration.png')
    value, policy = run2()
    plot(value, policy, 'value_iteration2.png')