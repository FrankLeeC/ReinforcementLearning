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
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

GRID = np.zeros([7, 10])

START_STATE = (3, 0)
END_STATE = (3, 7)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

Q = np.zeros([7, 10, 4])


def get_wind(state):
    _, y = state
    if y in [3, 4, 5, 8]:
        return 1
    if y in [6, 7]:
        return 2
    return 0

def step(state, action):
    x, y = state
    wind = get_wind(state)
    if action == UP:
        return (max(0, x - 1 - wind), y)
    elif action == RIGHT:
        return (max(0, x - wind), min(9, y + 1))
    elif action == DOWN:
        return (max(0, min(x + 1 - wind, 6)), y)
    else:
        return (max(0, x - wind), max(y - 1, 0))

def get_action(state):
    global Q
    x, y = state
    values = Q[x][y]
    return np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])

def generate_episode():
    global Q
    steps = 0
    state = START_STATE
    gamma = 1.0
    reward = -1
    epsilon = 0.1
    alpha = 0.5
    if np.random.binomial(1, epsilon) == 1:  # exploration
        action = random.randint(0, 3)
    else:
        action = get_action(state)
    while state != END_STATE:
        new_state = step(state, action)
        if np.random.binomial(1, epsilon) == 1:  # exploration
            new_action = random.randint(0, 3)
        else:
            new_action = get_action(new_state)
        Q[state[0]][state[1]][action] += alpha * (reward + gamma * Q[new_state[0]][new_state[1]][new_action] - Q[state[0]][state[1]][action])
        action = new_action
        state = new_state
        steps += 1
    return steps

def image(x, y):
    plt.subplot(1, 1, 1)
    plt.plot(x, y)
    plt.xlabel('time steps')
    plt.ylabel('episodes')
    plt.savefig('./windy_grid_world.png')
    plt.close()

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
    for i in range(7):
        for j in range(10):
            a = action_str(np.argmax(Q[i][j]))
            s += ' ' + a
        s += '\n'
    print(s)
            

def main():
    count = 170
    steps = []
    for _ in tqdm(range(count)):
        steps.append(generate_episode())
    steps = np.add.accumulate(steps)
    image(steps, range(1, count+1))
    output()

if __name__ == "__main__":
    main()