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

'''
4 x 4 grid world

|-----|-----|-----|-----|
|/////|  1  |  2  |  3  |
|-----|-----|-----|-----|
|  4  |  5  |  6  |  7  |
|-----|-----|-----|-----|
|  8  |  9  | 10  | 11  |
|-----|-----|-----|-----|
| 12  | 13  | 14  |/////|
|-----|-----|-----|-----|

R_t = -1 on all transitions

action set {UP, DOWN, RIGHT, LEFT} for all state
actions that would take the agent off the grid in fact leave the state unchanged

output VALUE: 
|-----|-----|-----|-----|
|  0  |  -1 | -2  | -3  |
|-----|-----|-----|-----|
| -1  | -2  | -3  | -2  |
|-----|-----|-----|-----|
| -2  | -3  | -2  | -1  |
|-----|-----|-----|-----|
| -3  | -2  | -1  |  0  |
|-----|-----|-----|-----|
'''

U = 'up'
R = 'right'
D = 'down'
L = 'left'

VALUE = np.random.randn(16)
VALUE[0] = 0.0
VALUE[15] = 0.0

def get_action_set(p):
    if p in [1, 2]:
        return {L: 1.0}
    elif p == 3:
        return {L: 0.5, D: 0.5}
    elif p in [4, 8]:
        return {U: 1.0}
    elif p == 5:
        return {U: 0.5, L: 0.5}
    elif p == 6:
        return {L: 0.5, D: 0.5}
    elif p in [7, 11]:
        return {D: 1.0}
    elif p in [9, 12]:
        return {U: 0.5, R: 0.5}
    elif p == 10:
        return {R: 0.5, D: 0.5}
    return {R: 1.0}  # p in [13, 14]


def step(p, a):
    if a == U:
        return p - 4
    elif a == R:
        return p + 1
    elif a == D:
        return p + 4
    return p - 1


def run(gamma=1.0, theta=0.001):
    global VALUE
    m = 0.0
    for i in range(16):
        if i in [0, 15]:
            continue
        v = VALUE[i]
        actions = get_action_set(i)
        v1 = 0.0
        for a, p in actions.items():
            v1 += p * 1.0 * (-1 + gamma * VALUE[step(i, a)])
        m = max(m, abs(v - v1))        
        VALUE[i] = v1
    return m < theta


def main():
    global VALUE
    while True:
        if run():
            break
    print(np.reshape(VALUE,(4, 4)))


if __name__ == "__main__":
    main()