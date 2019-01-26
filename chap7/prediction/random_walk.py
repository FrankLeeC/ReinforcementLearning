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

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

LEFT_ACTION = -1
RIGHT_ACTION = 1
LEFT_STATE = 0
RIGHT_STATE = 20
END_STATE = [LEFT_STATE, RIGHT_STATE]
TRUE_VALUE = np.arange(-20, 22, 2) / RIGHT_STATE
TRUE_VALUE[0] = 0.0
TRUE_VALUE[-1] = 0.0


def image(x, y, l):
    plt.subplot(1, 1, 1)
    for i, e in enumerate(y):
        plt.plot(x, e, label=l[i])
    plt.xlabel('alpha')
    plt.ylabel('avg rms')
    plt.ylim(0.25, 0.6)
    plt.legend(loc='lower right')
    plt.savefig('./random_walk.png')
    plt.close()


def get_reward(state):
    if state == LEFT_STATE:
        return -1
    if state == RIGHT_STATE:
        return 1
    return 0


def random_start():
    return 10


def is_end(state):
    return state in END_STATE


def get_action(state):
    if np.random.binomial(1, 0.5) == 1:
        return LEFT_ACTION
    return RIGHT_ACTION


def n_step_td(value, alpha, n):
    gamma = 1.0
    states, rewards = [], []
    rewards.append(0)
    state = random_start()
    states.append(state)
    T = float('inf')
    t = 0
    while True:
        if t < T:
            action = get_action(state)
            next_state = state + action
            reward = get_reward(next_state)
            states.append(next_state)
            rewards.append(reward)
            if is_end(next_state):
                T = t + 1
            state = next_state            
        tau = t - n + 1
        if tau >= 0:
            start = tau + 1
            end = min(tau+n, T)
            gs = np.zeros((end-start+1))
            for j in range(end-start+1):
                i = j + start
                gs[j] = gamma ** (i - tau - 1)
            g = np.sum(gs * (np.array(rewards)[tau+1:]))
            if tau + n < T:
                g += (gamma ** n) * value[states[tau+n]]
            value[states[tau]] += alpha * (g - value[states[tau]])
        t += 1
        if tau == T - 1:
            break


def run(n, alpha, episodes):
    # count = 100
    errs = 0.0
    # for _ in range(count):
    value = np.zeros(RIGHT_STATE+1)
    for _ in range(episodes):
        n_step_td(value, alpha, n)
        errs += np.sqrt(np.sum((value - TRUE_VALUE) ** 2) / (RIGHT_STATE - 1))
    return errs
    # return errs / (count*episodes)


def main():
    steps = np.power(2, np.arange(0, 10, 1))
    alpha = np.arange(0.0, 1.1, 0.1, dtype=np.float)
    labels = ['n=%d' % s for s in steps]
    errors = np.zeros([len(steps), len(alpha)])
    count = 100
    episodes = 10
    for _ in tqdm(range(count)):
        for i, n in enumerate(steps):
            # errs = np.zeros_like(alpha)
            for j, a in enumerate(alpha):
                # errs[j] = run(n, a)
                errors[i][j] += run(n, a, episodes)
            # errors.append(errs)
    errors /= count*episodes
    image(alpha, errors, labels)

if __name__ == "__main__":
    main()