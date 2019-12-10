#!/usr/bin/env python

import sys
from evolutionary_learner import EvolutionaryLearner
from environment import TrafficEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pickle
import optparse
from tf_network import TFNetwork

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

    num_tests = 10
    population_size = 5

    num_generations = 10

    hidden_unit_size = 300
    num_hidden_layers = 6

    mutation_fraction = 0.1
    mutation_stddev = 0.1

    for test_num in range(num_tests):
        if options.from_save:

            networks = []
            for i in range(population_size):
                networks.append(TFNetwork.from_save("evo_%d.h5" % i))

            learner = EvolutionaryLearner.from_existing(networks)

            save_file = "evolutionary_transfer_results_%d.pickle" % test_num
        else:

            learner = EvolutionaryLearner(population_size, env.input_size, num_phases, hidden_unit_size, num_hidden_layers)

            save_file = "evolutionary_results_%d.pickle" % test_num

        generations = []
        rewards = []

        for i in range(num_generations):

            learner.generate_mutations(mutation_fraction, mutation_stddev)

            scores = [None for _ in learner.networks]

            for j, network in enumerate(learner.networks):

                env.run(network)

                scores[j] = env.cumulative_reward

            learner.select(scores, 0.1)

            generations.append(i)

            for score in scores:
                rewards.append(score)

            print("Scores after selection %d" % i)
            print(scores)

            print("Average score after selection %d" % i)
            print(np.average(scores))

            with open(save_file, "wb") as file:
                pickle.dump(rewards, file)


if __name__ == "__main__":

    main()
