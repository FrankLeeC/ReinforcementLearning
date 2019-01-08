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


def run3():
    value = np.random.rand(101) * 2 - 1  # [-1, 1) arbitrarily
    value[0] = 0.0
    value[100] = 0.0
    policy = np.ones(99, dtype=np.int)
    epsilon = 1e-9
    run = True
    while run:
        run = False
        while True:
            m = 0.0
            for j in range(99):
                i = j + 1
                a = policy[j]
                oldv = value[i]
                value[i] = step(i, a, value)
                m = max(m, abs(value[i]-oldv))
            if m < epsilon:
                break
        for j in range(99):
            i = j + 1
            actions = get_actions(i)
            mv = float('-inf')
            ma = 0
            ra = policy[j]
            for a in actions:
                v = step(i, a, value)
                if v >= mv:
                    mv = v
                    ma = a
            if ra != ma:
                run = True
                policy[j] = ma
    return value, policy

def run2():
    value = np.zeros(101)
    policy = np.zeros(99)
    epsilon = 1e-9
    while True:
        m = 0.0
        for j in range(99):
            i = j + 1
            oldv = value[i]
            actions = get_actions(i)
            mv = float('-inf')
            for a in actions:
                v = step(i, a, value)
                mv = max(mv, v)
            value[i] = mv
            m = max(m, abs(mv-oldv))
        if m < epsilon:
            break
    for j in range(99):
        i = j + 1
        mv = float('-inf')
        ma = 0
        for a in get_actions(i):
            v = step(i, a, value)
            if v >= mv:
                mv = v
                ma = a
        policy[j] = ma
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
            for a in actions:
                v = step(i, a, value)
                mv = max(mv, v)
            value[i] = mv
            m = max(m, abs(mv-oldv))
        if m < epsilon:
            break
    for j in range(99):
        i = j + 1
        mv = float('-inf')
        ma = 0
        for a in get_actions(i):
            v = step(i, a, value)
            if v >= mv:
                mv = v
                ma = a
        policy[j] = ma
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
    value2, policy = run2()
    plot(value, policy, 'value_iteration2.png')
    value3, policy = run3()
    plot(value, policy, 'value_iteration3.png')