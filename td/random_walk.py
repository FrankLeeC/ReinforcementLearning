# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
import copy

TRUE_VALUES = np.array([1, 2, 3, 4, 5]) / 6
ACTIONS = [-1, 1]
REWARDS = [0, 0, 0, 0, 0, 0, 1]
VALUES = [0, 0, 0, 0, 0, 0, 0]


alpha = 0.1
gamma = 1
def run():
    state = 3
    while True:
        action = ACTIONS[np.random.binomial(1, 0.5)]
        reward = REWARDS[state + action]
        new_state = state + action
        VALUES[state] += alpha * (reward + gamma*VALUES[new_state ] - VALUES[state])   
        state = new_state    
        if state == 0 or state == 6:
            break 


def image(err):
    plt.subplot(1, 1, 1)
    plt.plot(range(100), err)
    plt.xlabel('episodes')
    plt.ylabel('error')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('./walk.png')
    plt.close()


def image2(values):
    plt.subplot(1, 1, 1)
    plt.plot(range(5), values[0], color='blue')
    plt.plot(range(5), values[1], color='red')    
    plt.plot(range(5), TRUE_VALUES, color='black')    
    plt.xlabel('state')
    plt.ylabel('value')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('./value.png')
    plt.close()


if __name__ == "__main__":
    values = []
    values.append(copy.deepcopy(VALUES[1:6]))
    err = []
    for _ in range(100):
        run()
        e = np.mean((np.array(VALUES)[1:6] - TRUE_VALUES) ** 2)
        err.append(e)
    image((err))
    values.append(copy.deepcopy(VALUES[1:6]))
    image2(values)