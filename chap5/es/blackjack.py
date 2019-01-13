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
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

STICK_ACTION = 0
HIT_ACTION = 1


COUNT = np.zeros([10, 10, 2, 2])
Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action 

# 0: nousable_ace  1: usable_ace
# 0: STICK_ACTION  1: HIT_ACTION
# player's sum, dealer's sum, usable_ace, action 
POLICY = np.zeros([10, 10, 2, 2])


class State:

    def __init__(self, play_sum, dealer_showing, usable_ace):
        self.ps = play_sum
        self.ds = dealer_showing
        self.u = usable_ace
    
    def player_sum(self):
        '''
        [12, 21]
        '''
        return self.ps

    def dealer_showing(self):
        '''
        [2, 11]
        '''
        return self.ds

    def usable_ace(self):
        '''
        0: nousable_ace
        1: usable_ace
        '''
        return self.u


class Episode:

    def __init__(self):
        self.states_list = []
        self.action_list = []
        self.reward_list = []

    def add_state(self, state):
        self.states_list.append(state)

    def add_action(self, action):
        self.action_list.append(action)

    def add_reward(self, reward):
        self.reward_list.append(reward)

    def length(self):
        return len(self.states_list)
        
    def states(self):
        return self.states_list

    def action(self):
       return self.action_list

    def reward(self):
        return self.reward_list


def show_image(episode, title):
    sns.set()
    data = np.asarray(np.zeros((22, 12), dtype=float))
    for e in episode:
        data[e[0]+12][e[1]+2] = float(e[2])
    ax = sns.heatmap(data, cmap='YlGnBu')
    ax.set_xlim(2, 12)
    ax.set_ylim(12, 22)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('plar sum')
    ax.set_title(title)
    plt.savefig(title + '.png')
    plt.close()


def draw():
    global POLICY
    usable_ace_episode = []  # 1
    nousable_ace_episode = []  # 0
    for p in range(10):
        for d in range(10):
            nousable_action = 0.0
            if POLICY[p][d][0][0] == 1.0:  # nousable_ace    stick_action
                nousable_action = 1.0       # stick
            nousable_ace_episode.append([p, d, nousable_action])

            usable_action = 0.0
            if POLICY[p][d][1][0] == 1.0:
                usable_action = 1.0        # stick
            usable_ace_episode.append([p, d, usable_action])
    show_image(nousable_ace_episode, 'nousable_ace')
    show_image(usable_ace_episode, 'usable_ace')


def random_card():
    random.seed()
    c = random.randint(1, 13)
    return min(c, 10)

def init_player_state():
    '''
    initialize player state
    return sum, usable ace
    '''
    s = 0
    u = 0
    while s < 12:
        c = random_card()
        if c == 1:
            if s + 11 > 21:
                s += 1
            else:
                u = 1
                s += 11
        else:
            s += c
    return s, u

def init_dealer_state():
    '''
    initialize dealer state
    return dealer_showing, dealer usable ace
    '''
    s = random_card()
    u = 0
    if s == 1:
        s = 11
        u = 1
    return s, u

def init_start_state():
    '''
    initialize start state
    '''
    p, u = init_player_state()
    d, du = init_dealer_state()
    return State(p, d, u), du

def random_action(state):
    return random.randint(0, 1)

def policy_action(state):
    ps, d, u = state.player_sum(), state.dealer_showing(), state.usable_ace()
    q = Q[ps-12][d-2][u]
    if q[0] > q[1]:
        return 0
    elif q[0] == q[1]:
        return np.random.choice([0, 1])
    return 1


def process():
    episode = Episode()
    state, du = init_start_state()
    action = random_action(state)
    episode.add_state(state)
    episode.add_action(action)
    next_state = State(state.player_sum(), state.dealer_showing(), state.usable_ace())
    # player's turn
    while True:
        if action == STICK_ACTION:
            break
        new_sum = 0
        new_usable_ace = next_state.usable_ace()
        card = random_card()
        if card == 1:
            s = next_state.player_sum()
            new_sum = s
            if s + 11 > 21:
                new_sum += 1
            else:
                new_sum += 11
                new_usable_ace = 1
        else:
            new_sum = next_state.player_sum() + card
        if new_sum > 21:
            if new_usable_ace == 0:
                return episode, -1
            new_usable_ace = 0
            new_sum -= 10
        episode.add_reward(0)
        next_state = State(new_sum, next_state.dealer_showing(), new_usable_ace)
        action = policy_action(next_state)
        episode.add_state(next_state)
        episode.add_action(action)

    # dealder's turn
    dealer_sum = state.dealer_showing()
    while True:
        if dealer_sum >= 17:
            break
        card = random_card()
        if card == 1:
            if dealer_sum + 11 > 21:
                dealer_sum += 1
            else:
                dealer_sum += 11
                du = 1
        else:
            dealer_sum += card
        if dealer_sum > 21:
            if du == 1:
                dealer_sum -= 10
                du = 0
            else:
                return episode, 1
    if next_state.player_sum() > dealer_sum:
        return episode, 1
    elif next_state.player_sum() == dealer_sum:
        return episode, 0
    else:
        return episode, -1

def run():
    global POLICY, Q, COUNT
    for _ in tqdm(range(500000)):
        episode, reward = process()
        states = episode.states()
        actions = episode.action()
        g = 0.0
        for i in range(episode.length()):
            s = states[i]
            a = actions[i]
            g = 0.9 * g + reward
            COUNT[s.player_sum() - 12][s.dealer_showing() - 2][s.usable_ace()][a] += 1
            Q[s.player_sum() - 12][s.dealer_showing() - 2][s.usable_ace()][a] += g
    p = Q / COUNT
    for i in range(10):
        for j in range(10):
            for u in range(2):
                POLICY[i][j][u][np.argmax(p[i][j][u])] = 1.0


if __name__ == "__main__":
    run()
    draw()