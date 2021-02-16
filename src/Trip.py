import RideSimulator.reward_parameters as rp

class Trip(object):
    """
    The Rider class handles the functionality of a trip.

    :param env: The environment the trip is created
    :type env: simpy.Environment
    :param trip_id: Unique id of the trip/rider
    :param pick_loc: Pickup location of the rider
    :param drop_loc: Drop location of the rider
    :param nearest_spots: The ids of the nearest driver pools
    """

    def __init__(self, env, trip_id: int, pick_loc: [int], drop_loc: [int], nearest_spots: list, hex_id: int,
                 additional_reward: float = 0):
        self.env = env
        self.created_time = env.now
        self.id = trip_id
        self.pickup_loc = pick_loc
        self.drop_loc = drop_loc
        self.nearest_spots = nearest_spots
        self.hex_id = hex_id
        self.driver_found = False
        self.cancelled = False
        self.driver = None
        self.driver_req = None
        self.rejected_drivers = []
        self.rejected_locations = []
        self.trip_distance = 0
        self.pick_distance = 0
        self.pickup_cost = 0
        self.total_cost = 0
        self.trip_reward = additional_reward
        self.hourly_reward = 0
        self.search_radius = 1

    def drop_driver(self, driver_id: int, driver_loc):
        """
        If a driver rejects the trip he is unassigned from the rider and the rider will start to search for a driver
        again

        :param driver_loc:
        :param driver_id: id of the driver that was assigned to the trip
        """
        self.driver_found = False
        self.driver = None
        self.rejected_drivers.append(driver_id)
        self.rejected_locations.append(driver_loc)

    def update_trip(self, per_km_cost: float, per_km_reward: float, pickup_distance: int, trip_distance: int, day_time: int):
        """
        Once the driver accepts the trip the distance to the driver is updated

        :param per_km_cost: This value can change depending on the traffic level
        :param per_km_reward: Peak and off-peak hours will reward the driver differently
        :param pickup_distance: Distance to the pickup location for the driver
        :param trip_distance: Distance between pickup and drop locations
        :param day_time: Time of day
        """
        self.trip_distance = trip_distance
        self.pick_distance = pickup_distance
        self.pickup_cost = pickup_distance * per_km_cost
        self.total_cost = (pickup_distance + trip_distance) * per_km_cost
        self.hourly_reward = rp.base_hourly_reward_weight * (rp.hourly_acceptance_rates[day_time] - rp.avg_hourly_acceptance)
        self.trip_reward = self.trip_reward + trip_distance * per_km_reward + self.hourly_reward
        #print("time of day : ", day_time, " hourly reward : ", self.hourly_reward, " total reward : ", self.trip_reward)
        #print(self.trip_reward)

    def get_net_reward(self):
        """
        :return: The net reward for the driver after completing the trip
        """
        return self.trip_reward - self.total_cost

    def get_state_info(self):
        return {
            "pick_loc": self.pickup_loc,
            "drop_loc": self.drop_loc
        }

    def update_search_radius(self):
        self.search_radius = min(5, self.search_radius + 1)
