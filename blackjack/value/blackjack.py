# -*- coding:utf-8 -*-

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cards = ['ace', 2, 3, 4, 5, 6, 7, 8, 9, 10]


def hit():
    return random.choice(cards)


def usable_ace(s):
    if s + 11 <= 21:
        return True
    return False


def prepare():
    '''
    玩家初始化牌直到 >= 12
    返回当前总和，是否拥有可用的ace
    '''
    s = 0
    usable = False
    while s < 12:
        card = hit()
        if card == 'ace':
            if usable_ace(s):
                s += 11
                usable = True
            else:
                s += 1
        else:
            s += card
    return s, usable


def generate_episode():
    dealer = hit()
    if dealer == 'ace':
        dealer = 11
    s, usable = prepare()
    states = list()
    rewards = list()

    states.append([s, dealer, usable])
    done = False
    while s < 20:
        card = hit()
        # print(card)
        if card == 'ace':
            s += 1
        else:
            s += card
        if s > 21:
            rewards.append(-1)
            done = True
        else:
            rewards.append(0)
            states.append([s, dealer, usable])
    if not done:
        bust = False
        while dealer < 17:
            card = hit()
            if card == 'ace':
                if dealer + 11 > 21:
                    dealer += 1
                else:
                    dealer += 11
            else:
                dealer += card
            if dealer > 21:
                bust = True
        # print('dealer:%d'%dealer)
        if bust:
            rewards.append(1)
        else:
            if dealer > s:
                rewards.append(-1)
            elif dealer < s:
                rewards.append(1)
            else:
                rewards.append(0)
    return states, rewards


def show_image(episode, b, title):
    sns.set()
    data = np.asarray(np.zeros((22, 12), dtype=float))
    print(np.shape(data))
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
    count = 10000
    usable_episode = {}
    unusable_episode = {}
    for _ in range(count):
        states, rewards = generate_episode()
        states = list(reversed(states))
        rewards = list(reversed(rewards))

        g = 0.0
        for state, reward in zip(states, rewards):
            g += 0.9 * reward
            key = '%d_%d_%d' % (state[0], state[1], state[2])
            n = state_counts.get(key, 0)
            old = state_values.get(key, 0.0)
            state_values[key] = old + (g - old) / (n + 1)
            state_counts[key] = n + 1
    for k, v in state_values.items():
        if '1' == k.split('_')[2]:
            usable_episode[k] = float(v)
        else:
            unusable_episode[k] = float(v)
        print('%s: %f' % (k, float(v)))
    print("state counts: %d" % len(state_values))
    show_image(usable_episode, False, 'usable_ace')
    show_image(unusable_episode, True, 'unusable_ace')
            