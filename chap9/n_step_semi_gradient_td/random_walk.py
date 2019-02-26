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

STATE_COUNT = 1002
LEFT_ACTION = -1
RIGHT_ACTION = 1


def get_action(state):
    if np.random.binomial(1, 0.5) == 1:
        return LEFT_ACTION
    return RIGHT_ACTION


def step(state, action):
    if action == LEFT_ACTION:
        new_state = state - 50
        if new_state < 0:
            return 0
        return new_state
    new_state = state + 50
    if new_state > 1001:
        return 1001
    return new_state

W = np.zeros(20)
def get_value(state):
    '''
    state \in [1, 1000]
    '''
    global W
    a = (state - 1) // 50
    return W[a]