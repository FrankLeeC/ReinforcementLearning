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

LEFT_ACTION = 0
RIGHT_ACTION = 1

L_STATE = 0
B_STATE = 1
A_STATE = 2
R_STATE = 3

B_ACTIONS = 10

def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        if i != 2:
            plt.plot(x, e, label=l[i])
        else:
            plt.plot(x, e, label=l[i], linestyle='dashdot')
    plt.xlabel('episode')
    plt.ylabel('left / time step')
    plt.legend(loc='upper right')
    plt.savefig('./bias.png')
    plt.close()

def step(state, action):
    if state == B_STATE:
        return L_STATE, np.random.normal(-0.1, 1.0)
    elif state == A_STATE:
        if action == LEFT_ACTION:
            return B_STATE, 0
        return R_STATE, 0
        

def double_q_learning(q1, q2):
    state = A_STATE
    gamma = 1.0
    alpha = 0.1
    epsilon = 0.1
    left = 0
    steps = 0
    while state != L_STATE and state != R_STATE:
        steps += 1
        tq = [a+b for a, b in zip(q1, q2)]
        actions = tq[state]
        if np.random.binomial(1, epsilon) == 1:  # explore
            action = np.random.choice(list(range(len(actions))))
        else:  # exploit
            action = np.random.choice([a for a, v in enumerate(actions) if v == np.max(actions)])
        if state == A_STATE and action == LEFT_ACTION:
            left += 1        
        new_state, reward = step(state, action)
        if np.random.binomial(1, 0.5) == 1:
            ma = np.random.choice([a for a, v in enumerate(q1[new_state]) if v == np.max(q1[new_state])])
            q1[state][action] += alpha * (reward + gamma * q2[new_state][ma] - q1[state][action])
        else:
            ma = np.random.choice([a for a, v in enumerate(q2[new_state]) if v == np.max(q2[new_state])])
            q2[state][action] += alpha * (reward + gamma * q1[new_state][ma] - q2[state][action])
        state = new_state
    return left

def q_learning(q):
    state = A_STATE
    gamma = 1.0
    alpha = 0.1
    epsilon = 0.1
    left = 0
    steps = 0
    while state != L_STATE and state != R_STATE:
        steps += 1
        actions = q[state]
        if np.random.binomial(1, epsilon) == 1:  # explore
            action = np.random.choice(list(range(len(actions))))
        else:  # exploit
            action = np.random.choice([a for a, v in enumerate(actions) if v == np.max(actions)])
        if state == A_STATE and action == LEFT_ACTION:
            left += 1        
        new_state, reward = step(state, action)
        new_action = np.random.choice([a for a, v in enumerate(q[new_state]) if v == np.max(q[new_state])])
        q[state][action] += alpha * (reward + gamma * q[new_state][new_action] - q[state][action])
        state = new_state
    return left

def run1():
    runs = 1000
    count = 300
    p = np.zeros(count)
    for _ in tqdm(range(runs)):
        q1 = [np.zeros([1]), np.zeros([B_ACTIONS]), np.zeros([2]), np.zeros([1])]
        q2 = [np.zeros([1]), np.zeros([B_ACTIONS]), np.zeros([2]), np.zeros([1])]
        l = list()
        for _ in range(count):
            l.append(double_q_learning(q1, q2))
        p += np.array(l)
    p /= runs
    return p

def run2():
    runs = 1000
    count = 300
    p = np.zeros(count)
    for _ in tqdm(range(runs)):
        q = [np.zeros([1]), np.zeros([B_ACTIONS]), np.zeros([2]), np.zeros([1])]
        l = list()
        for _ in range(count):
            l.append(q_learning(q))
        p += np.array(l)
    p /= runs
    return p

if __name__ == "__main__":
    p1 = run1()
    p2 = run2()
    image(np.arange(300)+1, [p1, p2, np.ones(300)/20], ['double_q_learning', 'q_learning', 'optimal'])
    
        