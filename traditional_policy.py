#!/usr/bin/env python

import numpy as np

class TraditionalPolicy:

    num_phases = 10

    def __init__(self, phase_time):

        self.phase_time = phase_time

    def get_selection(self, state):

        phases = state[-self.num_phases:]

        current_phase = np.argmax(phases)

        current_phase_time = phases[current_phase]

        if current_phase_time == self.phase_time:
            next_phase = (current_phase + 1) % self.num_phases
        else:
            next_phase = current_phase

        return next_phase
