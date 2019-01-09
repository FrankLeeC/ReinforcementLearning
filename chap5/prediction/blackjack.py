# -*- coding:utf-8 -*-

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def hit():
    return random.choice(cards)


def usable_ace(s):
    if s + 11 <= 21:
        return True
    return False


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
    dealer_usable_ace = False
    if dealer == 1:
        dealer = 11
        dealer_usable_ace = True
    s, player_usable = init_player()
    states = list()
    rewards = list()

    states.append([s, dealer, player_usable])
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
                rewards.append(0)
                states.append([s, dealer, player_usable])
            else:
                rewards.append(-1)
                done = True
        else:
            rewards.append(0)
            states.append([s, dealer, player_usable])
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
    show_image(usable_episode, False, 'usable_ace')
    show_image(unusable_episode, True, 'unusable_ace')
            