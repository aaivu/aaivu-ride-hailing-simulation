import simpy
import logging
import numpy as np
import tensorflow as tf
from RideSimulator.Trip import Trip
import RideSimulator.reward_parameters as rp
from tf_agents.trajectories import time_step as ts


class Driver(object):
    """
    The Driver class models the attributes and the behaviour of driver agents.

    :param env: simpy environment
    :param driver_id: driver id
    :param location: initial location of the driver when created
    :param spot_id: id of the nearest driver pool
    """

    def __init__(self, env: simpy.Environment, driver_id: int, location: np.ndarray, spot_id: int):

        self.env = env
        self.id = driver_id
        self.location = location
        self.in_trip = False
        self.trip_duration = 0
        self.weekly_target = rp.base_reward_threshold
        self.reward_amount = rp.base_reward
        self.trip_reward_weight = rp.trip_reward_weights[self.id % rp.weights_length]
        self.weekly_reward_weight = rp.weekly_reward_weights[self.id % rp.weights_length]
        self.opportunity_cost_weight = rp.opportunity_cost_weights[self.id % rp.weights_length]
        self.action = env.process(self.run())
        self.weekly_trip_count = 0
        self.total_reward = 0
        self.weekly_reward_reached = False
        self.trip = None
        self.spot_id = spot_id
        self.state_history = []
        self.idle_time_begin = 0

    def run(self):
        while True:
            try:
                if self.in_trip:
                    yield self.env.timeout(self.trip_duration)
                    logging.debug(f"Driver {self.id} finished trip")
                else:
                    yield self.env.timeout(float('inf'))
            except simpy.Interrupt:
                continue

    def reset_weekly_info(self, new_target, new_reward):
        self.weekly_trip_count = 0
        self.weekly_reward_reached = False
        # self.total_reward = 0
        self.weekly_target = new_target
        self.reward_amount = new_reward

    def start_trip(self, duration: int, location: np.ndarray, trip: Trip):
        """
        When the driver starts the trip, his 'total_trip_count' and 'weekly_reward_reached' values are updated.
        If the driver has achieved the weekly goal, an additional amount of reward is added.

        :param duration: trip duration
        :param location: drop location of the trip
        :param trip: trip object
        """
        self.in_trip = True
        self.trip = trip
        self.trip_duration = duration
        self.weekly_trip_count += 1
        self.location = location
        add_reward = False
        logging.debug(f"driver {self.id} location updated  to {self.location}")

        # Record rewards and distances.
        self.total_reward += trip.get_net_reward()

        if (not self.weekly_reward_reached) and (self.weekly_trip_count >= self.weekly_target):
            self.weekly_reward_reached = True
            self.total_reward += self.reward_amount
            add_reward = True
            logging.debug(f"driver {self.id} has achieved his weekly reward")
        self.action.interrupt()

        return add_reward

    def accept_ride(self, policy, observations) -> bool:
        """
        For now the driver randomly accepts or reject the trip.
        The reinforcement learning model will be exposed to this function and the state will be passed to make the
        decision.

        :return: the decision of the driver
        """
        driver_policy = policy[self.id % len(policy)]
        # TODO - check before commit
        # previous reward shouldn't matter when decision making, so leave reward = 0?
        # observation_ts = ts.transition(np.array(observations, dtype=np.float32), reward=0.0, discount=1.0)
        # observation_ts = ts.TimeStep(tf.constant([1]), tf.constant([0.0]), tf.constant([1.0]),
        #                              tf.convert_to_tensor(np.array([observations], dtype=np.float32), dtype=tf.float32))
        # logging.debug(observation_ts)
        # action = driver_policy.action(observation_ts)
        #print("obs", observations, "act", action)

        if policy[0] == "random":
            action = tf.random.uniform([1], 0, 2, dtype=tf.int32)

        elif policy[0] == "positive":
            if observations[0] < observations[1]:
                action = tf.random.uniform([1], 1, 2, dtype=tf.int32)
            else:
                action = tf.random.uniform([1], 0, 1, dtype=tf.int32)
        elif policy[0] == "all":
            action = tf.random.uniform([1], 1, 2, dtype=tf.int32)
        else:
            if type(policy[0] == 'str'):
                action = tf.random.uniform([1], 0, 2, dtype=tf.int32)

        return action

    def get_state_info(self) -> dict:
        """
        :return: the trip count and reward status of the driver
        """
        return {
            "goal_achieved": self.weekly_reward_reached,
            "trip_count": self.weekly_trip_count
        }
