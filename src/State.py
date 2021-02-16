import numpy as np


class State(object):
    def __init__(self,
                 trip_count: int,
                 goal_achieved: bool,
                 day_time: float,
                 trip_distance: int,
                 pick_distance: int,
                 traffic: float,
                 pick_loc: np.ndarray,
                 drop_loc: np.ndarray,
                 per_km_c: float,
                 per_km_r: float):
        """
        Stores the state dictionary

        :param trip_count: number of trips the driver has completed till the current trip
        :param goal_achieved: weekly goal status
        :param day_time: time of the day
        :param trip_distance: distance between the pickup and drop location
        :param pick_distance: distance to the pickup location
        :param traffic: traffic status
        :param per_km_c: fuel and other costs per km
        :param per_km_r: reward/pay per km
        """
        self.completed_trip_count = trip_count
        self.goal_achieved = goal_achieved
        self.day_time = day_time
        self.trip_distance = trip_distance
        self.pickup_distance = pick_distance
        self.traffic_status = traffic
        self.pickup_location = pick_loc
        self.drop_location = drop_loc
        self.per_km_cost = per_km_c
        self.per_km_reward = per_km_r
