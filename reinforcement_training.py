#!/usr/bin/env python

from environment import TrafficEnvironment
from reinforcement_learner import ReinforcementLearner
from tf_network import TFNetwork
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle


def main():

    num_phases = 10

    num_steps = 1800

    green_duration = 5
    yellow_duration = 4

    num_cells = 5
    deceleration_th = 0.5

    env = TrafficEnvironment(num_steps,
                             green_duration,
                             yellow_duration,
                             num_cells,
                             deceleration_th,
                             no_gui=True)

    num_episodes = 100

    hidden_layer_size = 300
    num_hidden_layers = 6

    learner = ReinforcementLearner(env.input_size, num_phases, hidden_layer_size, num_hidden_layers,
                                   discount_rate=0.9,
                                   batch_size=10)
    episodes = []
    rewards = []

    for i in range(num_episodes):

        env.run(learner)

        episodes.append(i)
        rewards.append(env.cumulative_reward)

        with open("actor_critic_results_v2.pickle", "wb") as file:
            pickle.dump(rewards, file)


    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.show()


if __name__ == "__main__":

    main()