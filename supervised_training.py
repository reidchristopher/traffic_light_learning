#!/usr/bin/env python

import sys
import pickle
import numpy as np
from tf_network import TFNetwork
from reinforcement_learner import ReinforcementLearner


def main():

    num_phases = 10

    recording_file = None
    save_file = None

    # default NN params
    hidden_layer_size = 100
    num_hidden_layers = 3

    # default training params
    num_epochs = 10
    batch_size = 100

    critic_file = None

    starting_network_file = None

    if len(sys.argv) < 3 or len(sys.argv) > 17:
        print("Usage: python supervised_training.py --recording_file <recording_file> --save_file <save_file> [options]")
        print("")
        print("Options:")
        print("--hidden_layer_size <int>\t\tSize of network hidden layers. Defaults to 100")
        print("--num_hidden_layers <int>\t\tNumber of network hidden layers. Defaults to 3")
        print("--num_epochs <int>\t\t\tNumber of iterations over all of the training data. Defaults to 10")
        print("--batch_size <int>\t\t\tBatch size for training. Defaults to 100")
        print("--starting_network <network_file>\tFile to starting network to use")
        print("--train_actor_critic <critic_file>\tIndicates to train both an actor and critic network.")
        print("\t\t\t\t\tThe actor is saved in <save_file> and the critic is saved in <critic_file>")
        return -1

    i = 1
    while i < len(sys.argv):

        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg == "--recording_file":
            recording_file = value
        elif arg == "--save_file":
            save_file = value
        elif arg == "--hidden_layer_size":
            hidden_layer_size = int(value)
        elif arg == "--num_hidden_layers":
            num_hidden_layers = int(value)
        elif arg == "--num_epochs":
            num_epochs = int(value)
        elif arg == "--batch_size":
            batch_size = int(value)
        elif arg == "--starting_network":
            starting_network_file = value
        elif arg == "--train_actor_critic":
            critic_file = value

        i += 2

    if recording_file is None:
        print("Failed to provide recording file")
        return -1

    if save_file is None:
        print("Failed to provide save file")
        return -1

    with open(recording_file, "rb") as file:
        states, actions, rewards = pickle.load(file)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    if critic_file is not None:
        actor_file = save_file

        # TODO train both actor and critic
        actor_network = None
        critic_network = None
    else:
        if starting_network_file is None:
            network = TFNetwork(input_size=len(states[0]), output_size=num_phases,
                                num_hidden_layers=num_hidden_layers, hidden_layer_size=hidden_layer_size)
        else:
            network = TFNetwork.from_save(starting_network_file)

        network.train(states, actions, num_epochs, batch_size)

        result = network.model.evaluate(states, actions, batch_size=batch_size, verbose=False)

        print("Post training accuracy: %.4f" % result[1])

        network.save(save_file)


if __name__ == "__main__":

    main()
