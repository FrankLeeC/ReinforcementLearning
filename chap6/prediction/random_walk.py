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
from tqdm import tqdm

def mc_predict(value, alpha, batch=False):
    s = 3
    g = 0.0
    states = [s]
    while s != 0 and s != 6:
        a = -1 
        if np.random.binomial(1, 0.5) == 1.0:
            a = 1
        ns = s + a
        states.append(ns)
        r = 0.0
        if ns == 6:
            r = 1.0
        g = 1.0 * g + r
        s = ns
    if not batch:
        for s in states:
            value[s] += alpha*(g - value[s])
        return None, None
    return states, g * np.ones_like(states)

def td_predict(value, alpha, batch=False):
    s = 3
    states = list()
    while s != 0 and s != 6:
        a = -1 
        if np.random.binomial(1, 0.5) == 1.0:
            a = 1
        ns = s + a
        r = 0.0
        if not batch:
            value[s] += alpha*(r + 1.0 * value[ns] - value[s])
        else:
            states.append(ns)
        s = ns
    if not batch:
        return None
    else:
        return states, np.zeros_like(states)

def batch_mc_err(alpha):
    count = 101
    true_value = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])    
    errs = np.zeros(count)
    for _ in tqdm(range(100)):
        e = list()
        value = np.zeros(7) + 0.5
        value[6] = 1.0
        value[0] = 0.0
        states = list()
        increments = list()
        for _ in range(count):
            e.append(np.sqrt(np.sum(np.power(value - true_value, 2))/5))
            ss, incre = mc_predict(value, alpha, True)
            states.append(ss)
            increments.append(incre)
            while True:
                upd = np.zeros(7)
                for i, ss in enumerate(states):
                    incre = increments[i]
                    for j in range(len(ss)-1):
                        s = ss[j]
                        inc = incre[j]
                        upd[s] += inc - value[s]
                upd *= alpha
                if np.sum(np.abs(upd)) < 1e-3:
                    break
                value += upd 
        errs += np.asarray(e)
    return errs / 100

def mc_err(alpha):
    count = 101
    true_value = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])    
    errs = np.zeros(101)
    for _ in range(100):
        e = list()
        value = np.zeros(7) + 0.5
        value[6] = 1.0
        value[0] = 0.0
        for _ in range(count):
            e.append(np.sqrt(np.sum(np.power(value - true_value, 2))/5))
            mc_predict(value, alpha)
        errs += np.asarray(e)
    return errs / 100        

def batch_td_err(alpha):
    count = 101
    true_value = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])    
    errs = np.zeros(count)
    for _ in tqdm(range(100)):
        e = list()
        value = np.zeros(7) + 0.5
        value[6] = 1.0
        value[0] = 0.0
        states = list()
        increments = list()
        for _ in range(count):
            e.append(np.sqrt(np.sum(np.power(value - true_value, 2))/5))
            ss, incre = td_predict(value, alpha, True)
            states.append(ss)
            increments.append(incre)
            while True:
                upd = np.zeros(7)
                for i, ss in enumerate(states):
                    incre = increments[i]
                    for j in range(len(ss)-1):
                        s = ss[j]
                        inc = incre[j]
                        upd[s] += inc + value[ss[j+1]] - value[s]
                upd *= alpha
                if np.sum(np.abs(upd)) < 1e-3:
                    break
                value += upd 
        errs += np.asarray(e)
    return errs / 100
    
def td_err(alpha):
    count = 101
    true_value = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])    
    errs = np.zeros(101)
    for _ in range(100):
        e = list()
        value = np.zeros(7) + 0.5
        value[6] = 1.0
        value[0] = 0.0
        for _ in range(count):
            e.append(np.sqrt(np.sum(np.power(value - true_value, 2))/5))
            td_predict(value, alpha)
        errs += np.asarray(e)
    return errs / 100

def mc_value():
    count = 101
    value = np.zeros(7) + 0.5
    value[6] = 1.0
    value[0] = 0.0
    values = list()
    episodes = [0, 1, 10 ,100]
    for i in range(count):
        if i in episodes:
            values.append(copy.deepcopy(value[1:6]))
        mc_predict(value, 0.1)
    return values

def td_value():
    count = 101
    value = np.zeros(7) + 0.5
    value[6] = 1.0
    value[0] = 0.0
    values = list()
    episodes = [0, 1, 10 ,100]
    for i in range(count):
        if i in episodes:
            values.append(copy.deepcopy(value[1:6]))
        td_predict(value, 0.1)
    return values

def image_err(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        if i > 2:
            plt.plot(x, e, label=l[i], linestyle='dashdot')
        else:
            plt.plot(x, e, label=l[i])
    plt.xlabel('walks/episodes')
    plt.ylabel('averaged error')
    plt.legend(loc='upper right', frameon=False)
    plt.savefig('./walk_error.png')
    plt.close()

def image_batch_err(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('walks/episodes')
    plt.ylabel('averaged error')
    plt.legend(loc='upper right', frameon=False)
    plt.savefig('./batch_error.png')
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
    e = [td_err(0.05), td_err(0.1), td_err(0.15), mc_err(0.01), mc_err(0.02), mc_err(0.03)]
    v = td_value()
    v.append(np.array([1/6, 2/6, 3/6, 4/6, 5/6]))
    image_err(range(101), e, ['td_alpha=0.05', 'td_alpha=0.1', 'td_alpha=0.15', 'mc_alpha=0.01', 'mc_alpha=0.02', 'mc_alpha=0.03'])
    image_value(['A', 'B', 'C', 'D', 'E'], v, ['0', '1', '10', '100', 'true'])

    e = [batch_td_err(0.001), batch_mc_err(0.001)]
    image_batch_err(range(101), e, ['td_alpha=0.001', 'mc_alpha=0.001'])
    
if __name__ == "__main__":
    main()