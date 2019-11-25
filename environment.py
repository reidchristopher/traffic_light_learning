#!/usr/bin/env python

import os
import sys
import random

import optparse
import numpy as np

# Import libraries for the traffic light simulator
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

# Define traffic light phase code based on cross.add.xml
PHASE_NS_G, PHASE_NS_Y = 0, 1
PHASE_EW_G, PHASE_EW_Y = 2, 3
PHASE_NSL_G, PHASE_NSL_Y = 4, 5
PHASE_EWL_G, PHASE_EWL_Y = 6, 7
PHASE_WL_G, PHASE_WL_Y = 8, 9
PHASE_EL_G, PHASE_EL_Y = 10, 11
PHASE_SL_G, PHASE_SL_Y = 12, 13
PHASE_NL_G, PHASE_NL_Y = 14, 15
PHASE_EWA_G, PHASE_EWA_Y = 16, 17
PHASE_NSA_G, PHASE_NSA_Y = 18, 19

class TrafficEnvironment:

    num_actions = 10

    # Constructor
    def __init__(self, max_steps, green_duration, yellow_duration, cell_number, deceleration_th, no_gui):
        # Initialize global variables
        self.steps = 0
        self.max_steps = max_steps
        self.waiting_times = {}
        self.speeds = {}
        self.deceleration_th = deceleration_th
        self.accelerations = {}
        self.REWARD = 0

        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.sum_intersection_queue = 0
        self.lane_length = 50
        self.cell_number = cell_number # The number of cell
        self.cell_length = self.lane_length / self.cell_number
        self.lane_ids = {'Wi_0': 0, 'Wi_1': 1, 'Wi_2': 2,
                         'Ei_0': 3, 'Ei_1': 4, 'Ei_2': 5,
                         'Si_0': 6, 'Si_1': 7, 'Si_2': 8,
                         'Ni_0': 9, 'Ni_1': 10, 'Ni_2': 11}
        self.incoming_roads = ['Wi', 'Ei', 'Si', 'Ni']

        self.input_size = 2 * int(self.lane_ids.get('Ni_2') + 1) * self.cell_number + self.num_actions
        self.no_gui = no_gui

    # Public method
    # Method for running simulation of one episode
    def run(self, policy):

        # Select whether gui is shown or not
        if self.no_gui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # Start simulation
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        # Reset variables for RL
        self.__reset()
        action_prev = 0
        action_prev_prev = None
        total_wait_prev = 0
        state_phase_prev = np.zeros(int(self.num_actions))

        # Loop function for processing steps during one episode
        while self.steps < self.max_steps:
            # Reset state phase if the series of actions of the traffic light are different
            if action_prev_prev != action_prev:
                state_phase_prev = np.zeros(int(self.num_actions))

            # Get current state of the intersection
            state_curr, state_phase_curr = self.__get_state(action_prev, state_phase_prev)
            # Obtain waiting time and speed information
            total_wait_curr = self.__get_waiting_times()
            # Calculate reward of previous action
            reward = self.__reward(total_wait_curr, total_wait_prev, reward_type="waiting_time")

            # Select the light phase to activate, based on the current state of the intersection
            action = policy.get_selection(state_curr)

            # Conduct yellow phase action before performing the next action
            if self.steps != 0 and action_prev != action:
                self.__set_yellow_phase(action_prev)
                self.__simulate(self.yellow_duration)
            # Conduct green phase action
            self.__set_green_phase(action)
            self.__simulate(self.green_duration)

            # Set values for the next step
            state_phase_prev = state_phase_curr
            if self.steps != 0:
                action_prev_prev = action_prev
            state_prev = state_curr
            action_prev = action
            total_wait_prev = total_wait_curr
            if reward < 0:
                self.REWARD += reward

        # End simulation for one episode
        print("Total reward: {}".format(self.REWARD))
        traci.close()

    # Private method
    # Method for handling the correct number of steps to simulate
    def __simulate(self, duration):

        if (self.steps + duration) >= self.max_steps:
            # Adjust duration for avoiding more steps than the maximum number of steps
            duration = self.max_steps - self.steps
        self.steps = self.steps + duration
        while duration > 0:
            # Simulate one step in sumo
            traci.simulationStep()
            duration -= 1
            intersection_queue = self.__get_statistics()
            self.sum_intersection_queue += intersection_queue

    # Method for getting state
    def __get_state(self, action, state_phase):

        # Initialize state using input_size of the neural network
        state_position = np.zeros(int(self.lane_ids.get('Ni_2') + 1) * self.cell_number)
        state_speed = np.zeros(int(self.lane_ids.get('Ni_2') + 1) * self.cell_number)

        # Loop function for obtaining the state represented by vehicles' position
        for vehicle_id in traci.vehicle.getIDList():
            # Obtain id of each lane and position of the vehicle with offsetting the position
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_position = self.lane_length - traci.vehicle.getLanePosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            valid_car = False

            # Map vehicle's position in meters into cells
            cell_id = 0
            while True:
                if (cell_id + 1) * self.cell_length < lane_position:
                    cell_id += 1
                else:
                    lane_cell = cell_id
                    break

            # Find the lane where the car is located
            if lane_id in self.lane_ids:
                lane_group = self.lane_ids.get(lane_id)
            else:
                lane_group = -1 # Dummy value for running

            if lane_group != -1:
                vehicle_position = self.cell_number * lane_group + lane_cell
                valid_car = True

            if valid_car:
                state_position[vehicle_position] = 1
                state_speed[vehicle_position] = speed

        # Add the number to represent state of the traffic light
        state_phase[action] += 1

        state = np.hstack((state_position, state_speed, state_phase))

        return state, state_phase

    # Method for getting waiting time of every car in the incoming lanes
    def __get_waiting_times(self):

        # Loop function for obtaining the waiting time of each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)

            # Add waiting time information
            if road_id in self.incoming_roads:
                self.waiting_times[vehicle_id] = wait_time
            else:
                if vehicle_id in self.waiting_times:
                    # Remove vehicles' information if they are not in the incoming roads
                    del self.waiting_times[vehicle_id]

        total_waiting_time = sum(self.waiting_times.values())

        return total_waiting_time

    # Method for getting speed information such as mean and maximum speed
    def __get_speed_information(self):

        # Loop function for obtaining the speed information of each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)

            # Add speed information
            if road_id in self.incoming_roads:
                self.speeds[vehicle_id] = speed
            else:
                if vehicle_id in self.speeds:
                    # Remove vehicles' information if they are not in the incoming roads
                    del self.speeds[vehicle_id]

        max_speed = max(self.speeds.values())
        mean_speed = sum(self.speeds.values()) / len(self.speeds)

        return max_speed, mean_speed

    # Method for getting acceleration information
    def __get_acceleration(self):

        # Loop function for obtaining the acceleration of each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)

            # Add acceleration information
            if road_id in self.incoming_roads:
                # Add deceleration information if it is more than threshold
                if abs(acceleration) > abs(self.deceleration_th):
                    self.accelerations[vehicle_id] = acceleration
            else:
                if vehicle_id in self.accelerations:
                    # Remove vehicles' information if they are not in the incoming roads
                    del self.accelerations[vehicle_id]

        total_deceleration = sum(self.accelerations.values())

        return total_deceleration

    # Method for getting statistics of the simulation of one step
    def __get_statistics(self):

        # Get the number of halting car in each edge
        halt_W = traci.edge.getLastStepHaltingNumber("Wi")
        halt_E = traci.edge.getLastStepHaltingNumber("Ei")
        halt_S = traci.edge.getLastStepHaltingNumber("Si")
        halt_N = traci.edge.getLastStepHaltingNumber("Ni")
        queue = halt_W + halt_E + halt_S + halt_N

        return queue

    # Method for computing reward
    def __reward(self, total_wait_curr, total_wait_prev, reward_type):
        if reward_type == "waiting_time":
            reward = total_wait_prev - total_wait_curr
        return reward

    # Method for selecting the action
    def __choose_action(self, state):
        # If function for epsilon-greedy policy
        if random.random() < self.epsilon:
            # Return random action for exploration
            return random.randint(0, self.num_actions - 1)
        else:
            # Return best action given the current state (exploitation) by referencing Q (state, action) value
            return random.randint(0, self.num_actions - 1) # Here need Q (state, action) info from RL network

    # Method for setting yellow light phase
    def __set_yellow_phase(self, action_prev):
        yellow_phase = action_prev*2 + 1
        traci.trafficlight.setPhase("C", yellow_phase)

    # Method for setting green light phase
    def __set_green_phase(self, action):
        traci.trafficlight.setPhase("C", action*2) # Action is multiplied by 2 because of the existence of yellow phase

    # Method for resetting the environment
    def __reset(self):
        self.steps = 0
        self.waiting_times = {}
        self.speeds = {}
        self.accelerations = {}
        self.REWARD = 0
        self.sum_intersection_queue = 0


if __name__ == '__main__':
    from traditional_policy import TraditionalPolicy
    # Below code is example for running the simulator

    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()

    traffic_environment = TrafficEnvironment(5400, 10, 4, 5, 0.5, options.nogui)
    traffic_environment.run(TraditionalPolicy(phase_time=6))
