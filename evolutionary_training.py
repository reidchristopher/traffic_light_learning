#!/usr/bin/env python

import sys
from evolutionary_learner import EvolutionaryLearner
from environment import TrafficEnvironment
import numpy as np
import matplotlib.pyplot as plt

def main():

    num_phases = 10

    num_steps = 540

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

    population_size = 10

    num_generations = 100

    hidden_unit_size = 300
    num_hidden_layers = 6

    mutation_fraction = 0.1
    mutation_stddev = 0.1

    learner = EvolutionaryLearner(population_size, env.input_size, num_phases, hidden_unit_size, num_hidden_layers)

    generations = []
    rewards = []

    for i in range(num_generations):

        learner.generate_mutations(mutation_fraction, mutation_stddev)

        scores = [None for _ in learner.networks]

        for j, network in enumerate(learner.networks):

            env.run(network)

            scores[j] = env.REWARD

        learner.select(scores, 0.1)

        generations.append(i)
        rewards.append(np.average(scores))

        plt.clf()
        plt.plot(generations, rewards)
        plt.pause(0.01)

        print("Scores after selection %d" % i)
        print(scores)

        print("Average score after selection %d" % i)
        print(np.average(scores))

    plt.xlabel("Generation")
    plt.ylabel("Average reward")
    plt.show()


if __name__ == "__main__":

    main()
