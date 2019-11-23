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

    # Constructor
    def __init__(self, epsilon, max_steps, num_actions, input_size, cell_number, deceleration_th):
        # Initialize global variables
        self.epsilon = epsilon
        self.steps = 0
        self.max_steps = max_steps
        self.waiting_times = {}
        self.speeds = {}
        self.deceleration_th = deceleration_th
        self.accelerations = {}
        self.num_actions = num_actions
        self.REWARD = 0

        self.lane_length = 50
        self.input_size = input_size
        self.cell_number = cell_number # The number of cell
        self.cell_length = self.lane_length / self.cell_number
        self.lane_ids = {'Wi_0': 0, 'Wi_1': 1, 'Wi_2': 2,
                         'Ei_0': 3, 'Ei_1': 4, 'Ei_2': 5,
                         'Si_0': 6, 'Si_1': 7, 'Si_2': 8,
                         'Ni_0': 10, 'Ni_1': 11, 'Ni_2': 12}
        self.incoming_roads = ['Wi', 'Ei', 'Si', 'Ni']

    # Public method
    # Method for running simulation of one episode
    def run(self, episode):

        # Select whether gui is shown or not
        options = self.__get_options()
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # Start simulation
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        # Reset variables for RL
        self.__reset()
        total_wait_prev = 0

        # Loop function for processing steps during one episode
        while self.steps < self.max_steps:
            # Get current state of the intersection
            state_curr = self.__get_state()

            # Obtain waiting time and speed information
            total_wait_curr = self.__get_waiting_times()
            max_speed_curr, mean_speed_curr = self.__get_speed_information()
            # Calculate reward of previous action
            reward = self.__reward(total_wait_curr, total_wait_prev, reward_type="waiting_time")

            # Select the light phase to activate, based on the current state of the intersection
            action = self.__choose_action(state_curr)

            # Conduct the selected action to transition next state
            if self.steps != 0 and action_prev != action:
                self.__set_yellow_phase(action_prev)
            else:
                self.__set_green_phase(action)

            state_prev = state_curr
            action_prev = action
            total_wait_prev = total_wait_curr
            if reward < 0:
                self.REWARD += reward

        # End simulation for one episode
        print("Total reward: {}".format(self.REWARD))
        traci.close()

    # Private method
    # Method for getting state
    def __get_state(self):

        # Initialize state using input_size of the neural network
        state_position = np.zeros(int(str(self.lane_ids.get('Ni_2')) + str(self.cell_number)))
        state_speed = np.zeros(int(str(self.lane_ids.get('Ni_2')) + str(self.cell_number)))

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
            lane_group = self.lane_ids.get(lane_id)

            if lane_group >= 1 and lane_group <= 7:
                vehicle_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                vehicle_position = lane_cell
                valid_car = True

            if valid_car:
                state_position[vehicle_position] = 1
                state_speed[vehicle_position] = speed

        state = np.hstack((state_position, state_speed))

        return state

    # Method for getting waiting time of every car in the incoming lanes
    def __get_waiting_times(self):

        # Loop function for obtaining the waiting time of each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            wait_time = traci.vehicle.getAccumulateWaitingTime(vehicle_id)
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
            return np.argmax() # Here need Q (state, action) info from RL network

    # Method for setting yellow light phase
    def __set_yellow_phase(self, action_prev):
        yellow_phase = action_prev + 1
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

    # Method for getting options for SUMO simulator
    def __get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()

        return options