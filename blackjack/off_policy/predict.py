# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt

STICK_ACTION = 0
HIT_ACTION = 1

Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action
C = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action


def reset():
    global Q, C
    Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action
    C = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action




def behavior_policy():
    random.seed()
    r = random.random()
    if r < 0.5:
        return STICK_ACTION
    return HIT_ACTION


def target_policy(play_sum):
    if play_sum < 20:
        return HIT_ACTION
    return STICK_ACTION


def start_state():
    '''
    play_sum: 13
    dealer_sum: 2
    usable_ace: True
    '''
    return State(13, 2, 1)


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


def ratio(state, action):
    ta = target_policy(state.player_sum())
    if ta == action:
        return 1 / 0.5
    return 0


def run(count):
    global Q, C
    returns = list()
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
            if w == 0.0:
                break
            g = 0.9 * g + rewards[i]
            ps = states[i].player_sum()
            ds = states[i].dealer_showing()
            u = states[i].usable_ace()
            a = actions[i]
            C[ps-12][ds-2][u][a] += w
            Q[ps-12][ds-2][u][a] += w*(g-Q[ps-12][ds-2][u][a])/C[ps-12][ds-2][u][a]
            w *= ratio(states[i], a)
        returns.append(Q[1][0][1][1])
    return returns


def show_image(x, y, title):
    plt.plot(x, y, label=title)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()
    plt.savefig(title + '.png')
    plt.close()

if __name__ == "__main__":
    count = 200
    episodes = 10000
    qs = np.zeros(episodes)
    for i in range(count):
        returns = run(episodes)
        returns = np.asarray(returns)
        qs += np.power(returns - (-0.27726), 2)
        reset()
        print(i+1)
    s = qs/100
    x = np.arange(0, episodes, 1)
    y = s
    show_image(x, y, 'weighted_importance_sampling')

    
