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

def reset():
    Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action

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


def draw(x, y, y2, title):
    plt.plot(x, y, color='red', label='weighted importance sampling')
    plt.plot(x, y2, color='black', label='ordinary importance sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()
    plt.savefig(title + '.png')
    plt.close()


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
    return State(13, 2, 1), False

def behavior_action(state):
    return random.randint(0, 1)

def target_action(state):
    if state.player_sum() < 20:
        return HIT_ACTION
    return STICK_ACTION

def process():
    episode = Episode()
    state, du = init_start_state()
    action = behavior_action(state)
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
                episode.add_reward(-1)
                return episode
            new_usable_ace = 0
            new_sum -= 10
        episode.add_reward(0)
        next_state = State(new_sum, next_state.dealer_showing(), new_usable_ace)
        action = behavior_action(next_state)
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
                episode.add_reward(1)
                return episode
    if next_state.player_sum() > dealer_sum:
        episode.add_reward(1)
    elif next_state.player_sum() == dealer_sum:
        episode.add_reward(0)
    else:
        episode.add_reward(-1)
    return episode

def get_ratio(p, a):
    if p < 20:
        if a == HIT_ACTION:
            return 1.0 / 0.5
        return 0.0 / 0.5
    if a == HIT_ACTION:
        return 0.0 / 0.5
    return 1.0 / 0.5

def run(counts):
    global POLICY, Q, COUNT
    ratios = list()
    gs = list()
    for _ in range(counts):
        episode = process()
        states = episode.states()
        actions = episode.action()
        rewards = episode.reward()
        g = 0.0
        numerator = 1.0
        denominator = 1.0
        for j in range(episode.length()):
            i = episode.length() - j - 1
            s = states[i]
            a = actions[i]
            r = rewards[i]
            g = 0.9 * g + r
            if a == target_action(s):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
            r = get_ratio(s.player_sum(), a)
        gs.append(g)
        ratios.append(numerator/denominator)
    ratios = np.asarray(ratios)
    gs = np.asanyarray(gs)
    weighted_returns = np.add.accumulate(ratios * gs)
    ratios = np.add.accumulate(ratios)
    ordinary_sampling = weighted_returns / np.arange(1, counts+1, 1)
    weighted_sampling = np.zeros_like(ordinary_sampling)
    for i, r in enumerate(ratios):
        if r == 0.0:
            weighted_sampling[i] = 0.0
        else:
            weighted_sampling[i] = weighted_returns[i] / r
    return np.reshape(weighted_sampling, (1, counts)), np.reshape(ordinary_sampling, (1, counts))
    

if __name__ == "__main__":
    count = 100
    episodes = 10000
    weighted_variace = np.zeros((1, episodes))
    ordinary_variace = np.zeros((1, episodes))
    for i in tqdm(range(count)):
        r1, r2 = run(episodes)
        weighted_variace += np.power(r1[0]-(-0.27726), 2)
        ordinary_variace += np.power(r2[0]-(-0.27726), 2)
    weighted_variace /= 100
    ordinary_variace /= 100
    draw(np.arange(0, episodes, 1), weighted_variace[0], ordinary_variace[0], 'off_policy_importance_sampling')