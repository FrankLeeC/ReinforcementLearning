# -*- coding:utf-8 -*-

import copy
import numpy as np

WIDTH = 4
HEIGHT = 3

SQUARE = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
]

UP = [-1, 0]
RIGHT = [0, 1]
DOWN = [1, 0]
LEFT = [0, -1]

COPY_SQUARE = None

def reset():
    global COPY_SQUARE
    COPY_SQUARE = [
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 0, 0]
    ]


def is_forbidden(position):
    i, j = position
    return i < 0 or i >= HEIGHT or j < 0 or j >= WIDTH or (i == j == 1)


def get_actions(position):
    i, j = position
    if i == 2:
        if j == 0:
            return [UP, LEFT, RIGHT]
        else:
            return [LEFT, DOWN, UP]
    if i == 1:
        return [UP, LEFT, RIGHT]
    if i == 0:
        return [RIGHT, UP, DOWN]


def get_reward(position):
    return 0.0


def get_value(position):
    i, j = position
    return SQUARE[i][j]


def move(current, action):
    new_position = current[0] + action[0], current[1] + action[1]
    if is_forbidden(new_position):
        return current 
    return new_position


def step():
    global COPY_SQUARE
    reset()
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if (j == 3 and i < 2) or (i == 1 and j == 1):
                continue
            position = (i, j)
            for n, a in enumerate(get_actions(position)):
                new_position = move(position, a)
                value = get_value(new_position)
                if n == 0:
                    COPY_SQUARE[i][j] += 0.8*(get_reward(position) + 0.9 * value)
                else:
                    COPY_SQUARE[i][j] += 0.1*(get_reward(position) + 0.9 * value)
                
            
def run():
    global SQUARE
    count = 0
    epsilon = 0.000001
    while True:
        step()
        count +=1
        m = np.max(np.asarray(COPY_SQUARE) - np.asarray(SQUARE))
        if m < epsilon:
            break 
        SQUARE = copy.deepcopy(COPY_SQUARE)
    print('iteration: ', count)

def main():
    run()
    for _, e in enumerate(SQUARE):
        print([float('%.2f'%each) for each in e])

if __name__ == '__main__':
    main()
