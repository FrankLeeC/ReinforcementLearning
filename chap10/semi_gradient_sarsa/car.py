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
import tile
import math

POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


def step(position, velocity, action):
    global POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3*position)
    new_velocity = min(max(new_velocity, VELOCITY_MIN), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(new_position, POSITION_MIN), POSITION_MAX)
    if new_position == POSITION_MIN:
        return new_position, 0.0
    return new_position, new_velocity


iht = tile.IHT(2048)
num = 8
position_scale = num / (POSITION_MAX - POSITION_MIN)
velocity_scale = num / (VELOCITY_MAX - VELOCITY_MIN)
def tile_coding(position, velocity, action):
    global iht
    p = position * position_scale
    v = velocity * velocity_scale
    return tile.tiles(iht, num, [p, v], [action])


w = np.zeros(2048)

def reset():
    global w
    w = np.zeros(2048)


def get_value(position, velocity, action):
    global w
    if position >= POSITION_MAX:
        return 0
    t = tile_coding(position, velocity, action)
    return np.sum(w[t])


def get_reward(position):
    global POSITION_MAX
    if position >= POSITION_MAX:
        return 0.0
    return -1.0


def get_action(position, velocity):
    epsilon = 0.1
    v1 = get_value(position, velocity, -1)
    v2 = get_value(position, velocity, 0)
    v3 = get_value(position, velocity, 1)
    c = [-1, 0, 1]
    a = np.argmax([v1, v2, v3])
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(c)
    return c[a]


def episode(alpha):
    global w
    gamma = 1
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0
    action = get_action(position, velocity)
    steps = 0
    while True:
        new_position, new_velocity = step(position, velocity, action)
        steps += 1
        reward = get_reward(new_position)
        v = get_value(position, velocity, action)
        t = tile_coding(position, velocity, action)
        if new_position >= POSITION_MAX:
            for i in t:
                w[i] += alpha * (reward - v)
            break
        new_action = get_action(new_position, new_velocity)        
        returns = reward + gamma * get_value(new_position, new_velocity, new_action)        
        for i in t:
            w[i] += alpha * (returns - v)
        position = new_position
        velocity = new_velocity
        action = new_action
    return steps


def run():
    count = 10
    alpha = [0.1/num, 0.2/num, 0.5/num]
    episodes = 500
    steps = np.zeros((len(alpha), episodes))
    for _ in tqdm(range(count)):
        for i, a in enumerate(alpha):
            reset()
            for j in range(episodes):
                steps[i][j] += episode(a)
    steps /= count
    image(np.arange(episodes), steps, ['alpha=0.1/8', 'alpha=0.2/8', 'alpha=0.5/8'])


def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.ylim(0, 1000)
    plt.legend(loc='upper right')
    plt.savefig('./car_step_per_episode.png')
    plt.close()


if __name__ == "__main__":
    run()