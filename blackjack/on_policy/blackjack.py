# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt

STICK_ACTION = 0
HIT_ACTION = 1


COUNT = np.zeros([10, 10, 2, 2])
Q = np.zeros([10, 10, 2, 2])  # player's sum, dealer's sum, usable ace, action 

# 第0表示nousable_ace,第1表示usable_ace
# 第0表示STICK_ACTION, 第1表示HIT_ACTION
POLICY = np.zeros([10, 10, 2, 2]) + 0.5  # initial propability. player's sum, dealer's sum, usable_ace, action 


def random_state():
    '''
    return [play's sum, dealer's showing, usable ace]
    play's sum ranges in [12, 21]
    dealer's sum ranges in [2, 11]
    usable ace  0 no  1 yes
    '''
    random.seed()
    r = random.randint(0, 199)
    if r < 100:
        return [int(r / 10) + 12, (r % 10) + 2, 0]
    r -= 100
    return [int(r / 10) + 12, (r % 10) + 2, 1]
    

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


def random_start_state():
    player_sum = random.randint(12, 21)
    dealer_showing = random.randint(2, 11)
    usable_ace = random.randint(0, 1)
    return State(player_sum, dealer_showing, usable_ace)

epsilon = 0.5
def get_action(state):
    '''
    STICK_ACTION 0
    HIT_ACTION   1
    '''
    global HIT_ACTION, STICK_ACTION
    ps = state.player_sum() - 12
    ds = state.dealer_showing() - 2
    u = state.usable_ace()
    p = POLICY[ps][ds][u]
    optimai_prop = 1 - epsilon + epsilon/2
    random.seed()
    r = random.random()
    best = True
    if r <= optimai_prop:
        best = True
    else:
        best = False
    if p[0] > p[1]:
        if best:
            return STICK_ACTION
        return HIT_ACTION
    if best:
        return HIT_ACTION
    return STICK_ACTION

def generate_episode():
    '''
    return episode
    '''
    global HIT_ACTION, STICK_ACTION
    episode = Episode()
    state = random_start_state()
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

def update_policy(player_sum, dealer_showing, usable_ace):
    global POLICY, Q
    ps = player_sum - 12
    ds = dealer_showing - 2
    if Q[ps][ds][usable_ace][0] > Q[ps][ds][usable_ace][1]:
        POLICY[ps][ds][usable_ace][0] = 1.0
        POLICY[ps][ds][usable_ace][1] = 0.0
    else:
        POLICY[ps][ds][usable_ace][1] = 1.0
        POLICY[ps][ds][usable_ace][0] = 0.0

def run():
    global COUNT, Q
    count = 10000
    for _ in range(count):
        episode = generate_episode()
        l = len(episode.states())
        g = 0.0
        states = episode.states()
        actions = episode.action()
        rewards = episode.reward()
        cache = set()
        for i in range(l):
            g = 0.9 * g + rewards[i]
            ps = states[i].player_sum()
            ds = states[i].dealer_showing()
            u = states[i].usable_ace()
            key = '%d_%d_%d' % (ps, ds, u)
            if key not in cache:
                cache.add(key)
                COUNT[ps-12][ds-2][u][actions[i]] += 1
                Q[ps-12][ds-2][u][actions[i]] += (rewards[i] - Q[ps-12][ds-2][u][actions[i]]) / COUNT[ps-12][ds-2][u][actions[i]]
                update_policy(ps, ds, u)


def show_image(episode, b, title):
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
            # print(POLICY[p][d][0][0] == 1.0)
            if POLICY[p][d][0][0] == 1.0:  # nousable_ace    stick_action
                nousable_action = 1.0       # stick
            else:
                nousable_action = 0.0       # hit
            nousable_ace_episode.append([p, d, int(nousable_action)])

            usable_action = 0.0
            if POLICY[p][d][1][0] == 1.0:
                usable_action = 1.0        # stick
            else:
                usable_action = 0.0        # hit
            usable_ace_episode.append([p, d, int(usable_action)])
    show_image(nousable_ace_episode, False, 'nousable_ace')
    show_image(usable_ace_episode, True, 'usable_ace')


def printout():
    i = 0
    for p in range(10):
        for d in range(10):
            for u in range(2):
                for a in range(2):
                    k = '%d_%d_%d_%d' % (p+12, d+2, u, a)
                    v = POLICY[p][d][u][a]
                    i += 1
                    print(k, ' = ', v) 
                    if i % 2 == 0:
                        print('-----------------')

if __name__ == "__main__":
    run()
    # printout()
    draw()

    
