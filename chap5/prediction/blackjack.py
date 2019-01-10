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

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def hit():
    return random.choice(cards)

class State:

    def __init__(self, player_sum, dealer_sum, usable_ace):
        self.p = player_sum
        self.d = dealer_sum
        self.u = usable_ace

    def player_sum(self):
        return self.p

    def dealer_sum(self):
        return self.d

    def usable_ace(self):
        return self.u


class Episode:

    def __init__(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def add_state(self, s):
        self.state_list.append(s)

    def add_action(self, a):
        self.action_list.append(a)

    def add_reward(self, r):
        self.reward_list.append(r)

    def states(self):
        return self.state_list
    
    def actions(self):
        return self.action_list

    def rewards(self):
        return self.reward_list

def init_player():
    '''
    initialize player's card    player's sum >= 12
    return player's sum, player_usable_ace
    '''
    s = 0
    usable = False
    while s < 12:
        card = hit()
        if card == 1:
            if s + 11 <= 21:
                s += 11
                usable = True
            else:
                s += 1
        else:
            s += card
    return s, usable


def generate_episode():
    episode = Episode()
    dealer = hit()
    dealer_usable_ace = False
    if dealer == 1:
        dealer = 11
        dealer_usable_ace = True
    s, player_usable = init_player()
    episode.add_state(State(s, dealer, player_usable))
    done = False
    while s < 20:
        card = hit()
        if card == 1:
            s += 1
        else:
            s += card
        if s > 21:
            if player_usable:
                player_usable = False
                s -= 10
                episode.add_reward(0)
                episode.add_state(State(s, dealer, player_usable))
            else:
                episode.add_reward(-1)
                done = True
        else:
            episode.add_reward(0)
            episode.add_state(State(s, dealer, player_usable))
    if not done:
        bust = False
        while dealer < 17:
            card = hit()
            if card == 1:
                if dealer + 11 > 21:
                    dealer += 1
                else:
                    dealer_usable_ace = True
                    dealer += 11
            else:
                dealer += card
            if dealer > 21:
                if dealer_usable_ace:
                    dealer -= 10
                    dealer_usable_ace = False
                else:
                    bust = True
        if bust:
            episode.add_reward(1)
        else:
            if dealer > s:
                episode.add_reward(-1)
            elif dealer < s:
                episode.add_reward(1)
            else:
                episode.add_reward(0)
    return episode


def show_image(episode, b, title):
    sns.set()
    data = np.asarray(np.zeros((22, 12), dtype=float))
    for k, v in episode.items():
        xy = k.split('_')
        data[int(xy[0])][int(xy[1])] = float(v)
    ax = sns.heatmap(data, cmap='YlGnBu')
    ax.set_xlim(2, 12)
    ax.set_ylim(12, 22)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('plar sum')
    ax.set_title(title)
    plt.savefig(title + '.png')
    plt.close()


state_values = {}
state_counts = {}


if __name__ == "__main__":
    count = 500000
    usable_episode = {}
    unusable_episode = {}
    for _ in range(count):
        episode = generate_episode()
        states = list(reversed(episode.states()))
        rewards = list(reversed(episode.rewards()))
        g = 0.0
        for state, reward in zip(states, rewards):
            g += 0.9 * reward
            key = '%d_%d_%d' % (state.player_sum(), state.dealer_sum(), state.usable_ace())
            n = state_counts.get(key, 0)
            old = state_values.get(key, 0.0)
            state_values[key] = old + (g - old) / (n + 1)
            state_counts[key] = n + 1
    for k, v in state_values.items():
        if '1' == k.split('_')[2]:
            usable_episode[k] = float(v)
        else:
            unusable_episode[k] = float(v)
    show_image(usable_episode, False, 'usable_ace')
    show_image(unusable_episode, True, 'unusable_ace')
            