#!/usr/bin/env python

from tf_network import TFNetwork

class ReinforcementLearner:

    def __init__(self, input_size, output_size, hidden_layer_size, num_hidden_layers):

        self.actor = TFNetwork(input_size, output_size, hidden_layer_size, num_hidden_layers,
                               output_activation='softmax')

        self.critic = TFNetwork(input_size, output_size, hidden_layer_size, num_hidden_layers,
                                output_activation='linear')


def test():

    pass

if __name__ == "__main__":

    test()
