#!/usr/bin/env python

from environment import TrafficEnvironment
from reinforcement_learner import ReinforcementLearner
from tf_network import TFNetwork
import matplotlib.pyplot as plt
import numpy as np
import pickle
import optparse

def main():

    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--from_save", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()

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

    discount_rate = 0.9
    batch_size = 10

    num_tests = 4
    starting_test = 11
    for test_num in range(starting_test, starting_test + num_tests):

        if options.from_save:
            actor = TFNetwork.from_save("test_A.h5")
            critic = TFNetwork.from_save("test_C.h5")

            learner = ReinforcementLearner.from_existing(actor, critic,
                                                         discount_rate=discount_rate,
                                                         batch_size=batch_size)

            save_file = "actor_critic_transfer_results_%d.pickle" % test_num
        else:
            learner = ReinforcementLearner(env.input_size, num_phases, hidden_layer_size, num_hidden_layers,
                                           discount_rate=discount_rate,
                                           batch_size=batch_size)

            save_file = "actor_critic_results_%d.pickle" % test_num

        episodes = []
        rewards = []

        for i in range(num_episodes):

            env.run(learner)

            episodes.append(i)
            rewards.append(env.cumulative_reward)

            with open(save_file, "wb") as file:
                pickle.dump(rewards, file)


if __name__ == "__main__":

    main()