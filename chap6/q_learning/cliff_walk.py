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

Q = np.zeros([4, 12, 4])

START_STATE = (3, 0)
END_STATE = (3, 11)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def step(state, action):
    i, j = state
    tmp = None
    if action == UP:
        tmp = (max(0, i-1), j)
    elif action == RIGHT:
        tmp = (i, min(j+1, 11))
    elif action == DOWN:
        tmp = (min(i+1, 3), j)
    else:
        tmp = (i, max(j-1, 0))
    if tmp[0] == 3 and tmp[1] > 0 and tmp [1] < 11:
        return START_STATE, False
    return tmp, True

def get_best_action(state):
    global Q
    x, y = state
    values = Q[x][y]
    return np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])

def generate_episode():
    global Q
    state = START_STATE
    gamma = 1.0
    epsilon = 0.1
    alpha = 0.5
    while state != END_STATE:
        if np.random.binomial(1, epsilon) == 1:  # exploration
            action = random.randint(0, 3)
        else:
            action = get_best_action(state)
        new_state, b = step(state, action)
        reward = -1
        if not b:
            reward = -100
        new_action = get_best_action(new_state)
        Q[state[0]][state[1]][action] += alpha * (reward + gamma * Q[new_state[0]][new_state[1]][new_action] - Q[state[0]][state[1]][action])
        state = new_state

def action_str(a):
    if a == UP:
        return '⬆️'
    if a == RIGHT:
        return '➡️'
    if a == DOWN:
        return '⬇️'
    return '⬅️'

def output():
    global Q
    s = ''
    for i in range(4):
        for j in range(12):
            if i == 3 and j > 0 and j < 11:
                s += ' ' + 'X'
                continue
            a = action_str(np.argmax(Q[i][j]))
            s += ' ' + a
        s += '\n'
    print(s)
      
def main():
    count = 10000
    for _ in tqdm(range(count)):
        generate_episode()
    output()

if __name__ == "__main__":
    main()