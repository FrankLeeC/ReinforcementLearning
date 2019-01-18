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

import random
import numpy as np
import matplotlib.pyplot as plt
import copy

def predict(value, alpha=0.1):
    s = 3
    while s != 0 and s != 6:
        a = -1 
        if np.random.uniform(0, 1) >= 0.5:
            a = 1
        ns = s + a
        r = 0.0
        value[s] += alpha*(r + 1.0 * value[ns] - value[s])
        s = ns
    return value

def run_err(alpha):
    count = 101
    true_value = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])    
    value = np.zeros(7) + 0.5
    value[6] = 1.0
    value[0] = 0.0
    errs = list()
    for _ in range(count):
        value = predict(value, alpha)
        errs.append(np.mean(np.power(value - true_value, 2)))
    return errs

def run_value():
    count = 101
    value = np.zeros(7) + 0.5
    value[6] = 1.0
    value[0] = 0.0
    values = list()
    episodes = [0, 1, 10 ,100]
    for i in range(count):
        if i in episodes:
            values.append(copy.deepcopy(value[1:6]))
        value = predict(value, 0.1)
    return values


def image_err(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('walks/episodes')
    plt.ylabel('averaged error')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('./walk_error.png')
    plt.close()

def image_value(x, y, l):
    plt.subplot(1, 1, 1)
    for i, v in enumerate(y):
        plt.plot(x, v, label=l[i])
    plt.xlabel('walks/episodes')
    plt.ylabel('value')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('./walk_value.png')
    plt.close()

def main():
    e = [run_err(0.05), run_err(0.1), run_err(0.15)]
    v = run_value()
    v.append(np.array([1/6, 2/6, 3/6, 4/6, 5/6]))
    image_err(range(101), e, ['alpha=0.05', 'alpha=0.1', 'alpha=0.15'])
    image_value(['A', 'B', 'C', 'D', 'E'], v, ['0', '1', '10', '100', 'true'])
    
if __name__ == "__main__":
    main()