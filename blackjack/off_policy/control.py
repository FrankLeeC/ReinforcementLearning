# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt

HIT_ACTION = 0
STICK_ACTION = 1

Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action
C = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action
POLICY = np.zeros([10, 10, 2])  # player's sum, dealer's sum, usable ace
for ps in range(10):
    for ds in range(10):
        for u in range(2):
            p = Q[ps][ds][u]
            if p[0] >= p[1]:
                POLICY[ps][ds][u] = HIT_ACTION
            else:
                POLICY[ps][ds][u] = STICK_ACTION


def behavior_policy():
    random.seed()
    r = random.random()
    if r < 0.5:
        return HIT_ACTION
    return STICK_ACTION


def target_policy(play_sum):
    if play_sum < 20:
        return HIT_ACTION
    return STICK_ACTION


def random_card():
    random.seed()
    return random.randint(1, 10)


def dealer(showing):
    sum = showing
    while sum < 17:
        card = random_card()
        if card == 1 and sum + card <= 21:
            sum += 11
        else:
            sum += card
    return sum


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
        
    def states(self):
        return self.states_list

    def action(self):
       return self.action_list

    def reward(self):
        return self.reward_list


def start_state():
    '''
    return [play's sum, dealer's showing, usable ace]
    play's sum ranges in [12, 21]
    dealer's sum ranges in [2, 11]
    usable ace  0 no  1 yes
    '''
    random.seed()
    r = random.randint(0, 199)
    if r < 100:
        return State(int(r / 10) + 12, (r % 10) + 2, 0)
    r -= 100
    return State(int(r / 10) + 12, (r % 10) + 2, 1)


def get_action(state):
    return behavior_policy()


def generate_episode():
    '''
    return episode
    '''
    global HIT_ACTION, STICK_ACTION
    episode = Episode()
    state = start_state()
    action = get_action(state)
    # usable_ace = state.usable_ace()  # 不会改变
    next_state = State(state.player_sum(), state.dealer_showing(), state.usable_ace())
    while action == HIT_ACTION:
        episode.add_state(next_state)
        episode.add_action(action)
        card = random_card()
        new_sum = next_state.player_sum() + card
        if new_sum > 21:  # burst
            episode.add_reward(-1)
            return episode
        else:
            episode.add_reward(0)
        next_state = State(new_sum, next_state.dealer_showing(), next_state.usable_ace())
        action = get_action(next_state)
    episode.add_state(next_state)
    episode.add_action(action)
    dealer_sum = dealer(state.dealer_showing())
    if next_state.player_sum() > dealer_sum:
        episode.add_reward(1)
    elif next_state.player_sum() < dealer_sum:
        episode.add_reward(-1)
    else:
        episode.add_reward(0)
    return episode


def run(count):
    global Q, C, POLICY
    for _ in range(count):
        episode = generate_episode()
        l = len(episode.states())
        g = 0.0
        w = 1.0
        states = episode.states()
        actions = episode.action()
        rewards = episode.reward()
        for j in range(l):
            i = l - j - 1
            g = 0.9 * g + rewards[i]
            ps = states[i].player_sum()
            ds = states[i].dealer_showing()
            u = states[i].usable_ace()
            a = actions[i]
            C[ps-12][ds-2][u][a] += w
            Q[ps-12][ds-2][u][a] += w*(g-Q[ps-12][ds-2][u][a])/C[ps-12][ds-2][u][a]
            p = Q[ps-12][ds-2][u]
            if p[0] >= p[1]:
                POLICY[ps-12][ds-2][u] = HIT_ACTION
            else:
                POLICY[ps-12][ds-2][u] = STICK_ACTION
            if a == POLICY[ps-12][ds-2][u]:
                w *= 1/0.5
            else:
                break


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


if __name__ == "__main__":
    episodes = 10000
    run(episodes)
    usable_list, unusable_list = [], []
    for ps in range(10):
        for ds in range(10):
            usable_list.append([ps, ds, POLICY[ps][ds][0]])
            unusable_list.append([ps, ds, POLICY[ps][ds][1]])
    show_image(usable_list, 'usable_ace')
    show_image(unusable_list, 'unusable_ace')



    
