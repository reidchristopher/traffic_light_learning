#!/usr/bin/env python

import pickle

states = []
actions = []
rewards = []

for i in range(500):
    with open("history/traditional_%d.pickle" % i, "rb") as file:
        new_states, new_actions, new_rewards = pickle.load(file)
        states += new_states
        actions += new_actions
        rewards += new_rewards
        
with open("combined_traditional.pickle", "wb") as file:
    pickle.dump((states, actions, rewards), file)
