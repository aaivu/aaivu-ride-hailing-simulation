import simpy
import itertools
import numpy as np
from RideSimulator.Driver import Driver
from RideSimulator.HexGrid import HexGrid


def get_spot_locations(width: int, height: int, interval: int) -> np.ndarray:
    """
    :param width: width of the grid
    :param height: height of the grid
    :param interval: distance between two spots
    :return: an array of all the spot locations
    """
    x_points = np.arange(0, width, interval)
    y_points = np.arange(0, height, interval)

    # If the distance to the nearest taxi spot from the corner is greater than the minimum search radius additional
    # spots are added along the edges of thr map.
    if (width - x_points[-1]) > (interval / np.sqrt(2)):
        x_points = np.append(x_points, width)

    if (height - y_points[-1]) > (interval / np.sqrt(2)):
        y_points = np.append(y_points, height)

    spots = np.array([list(i) for i in itertools.product(x_points, y_points)])
    return np.array([spots, len(y_points), len(x_points)], dtype=object)


class Grid(object):
    """
    Handles all the information and processes related to the grid. The distances between grid units can be translated to
    real world distances using the units_per_km attribute.

    Taxi spots are used to anchor drivers into locations in the map to make it easier to find the closest driver for a
    given trip.

    A hexagon overlay is used to cluster grid locations into regions where hotspots, traffic and other features are
    calculated based on the hotspots.
    """

    def __init__(self, env: simpy.Environment, width: int, height: int, interval: int, num_drivers: int,
                 hex_area: float, units_per_km: int = 1, seed: int = None):
        """

        :param env: simpy environment
        :param width: width of the grid
        :param height: height of the grid
        :param interval: distance between two spots
        :param num_drivers: number of drivers in the grid
        :param hex_area: area size of a single hex tile
        :param units_per_km: number of grid units per km
        """
        if seed is not None:
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.interval = interval
        self.hex_overlay = HexGrid(hex_area=hex_area, width=width, height=height, units_per_km=units_per_km)
        self.taxi_spots, self.spot_height, self.spot_width = get_spot_locations(width=width, height=height,
                                                                                interval=interval)
        self.driver_pools = simpy.FilterStore(env, capacity=num_drivers)

    def get_random_location(self) -> np.ndarray:
        x = np.random.randint(0, self.width)
        y = np.random.randint(0, self.height)
        return np.array([x, y])

    # Temp function to get location id until hexagons are implemented
    def get_location_id(self, location):
        grid_width = 10  # no. of cells in one axis (create 10x10 grid)
        x = np.floor((location[0] - 0) * grid_width / self.width)
        y = np.floor((location[1] - 0) * grid_width / self.height)
        return x * grid_width + y

    @staticmethod
    def get_distance(loc1: np.ndarray, loc2: np.ndarray) -> float:
        distance = np.linalg.norm(loc1 - loc2)
        return np.round(distance, 1)

    def get_spot_id(self, location):
        return int(np.round(location[0]) * self.spot_height + np.round(location[1]))

    def get_nearest_spot(self, location: np.ndarray, search_radius=1) -> list:
        """
        Find the nearest driver spot for a given location.

        Initially it'll only return the nearest spot to the driver. When search_radius = 2, the 4 taxi spots surrounding
        the rider are returned. Afterwards, with each increment to the search_radius, all taxi spots inside a square
        centered on the driver location with a side length of search_radius are returned.

        :param location: x,y coords of the location
        :param search_radius: number of breaths the search will carry out on
        :return: a list of the closest taxi spots
        """
        x_spot = location[0] / self.interval
        y_spot = location[1] / self.interval
        closet_spot = [np.round(x_spot), np.round(y_spot)]

        if search_radius == 1:
            spot_no = [self.get_spot_id(closet_spot)]

        elif search_radius == 2:
            spot_no = []
            x_points = {np.floor(x_spot), np.ceil(x_spot)}
            y_points = {np.floor(y_spot), np.ceil(y_spot)}
            spots = np.array([list(i) for i in itertools.product(x_points, y_points)])
            for spot in spots:
                spot_no.append(self.get_spot_id(spot))

        else:
            spot_no = []
            x_points = [closet_spot[0]]
            y_points = [closet_spot[1]]
            for i in range(1, search_radius - 1):
                x_points.append(max(0, closet_spot[0] - i))
                x_points.append(min(self.spot_width - 1, closet_spot[0] + i))

                y_points.append(max(0, closet_spot[1] - i))
                y_points.append(min(self.spot_height - 1, closet_spot[1] + i))

            x_points = set(x_points)
            y_points = set(y_points)

            spots = np.array([list(i) for i in itertools.product(x_points, y_points)])
            for spot in spots:
                spot_no.append(self.get_spot_id(spot))

        return spot_no

    def get_closest_drivers(self, location: np.ndarray, search_radius: int) -> list:
        """
        A more accurate closest driver search using driver distances of all the drivers in the closest taxi spots.
        Since this is more computationally expensive and the increment in accuracy does not outweigh the cost, this is
        not used at the moment.

        :param location: location the distances should be calculated from
        :param search_radius: number of breaths the search will carry out on
        :return: a list of driver ids sorted in the ascending order according to their distances to the location
        """
        spots = self.get_nearest_spot(location, search_radius=search_radius)
        driver_ids = []
        distances = []
        for driver in self.driver_pools.items:
            if driver.spot_id in spots:
                driver_ids.append(driver.id)
                distances.append(self.get_distance(location, driver.location))

        if len(driver_ids) > 0:
            _, driver_ids = zip(*sorted(zip(distances, driver_ids)))
        return driver_ids

    def assign_spot(self, driver: Driver):
        """
        Assign the driver to his nearest driver pool.

        :param driver: driver object
        """
        driver_loc = driver.location
        spot_id = self.get_nearest_spot(driver_loc)[0]
        driver.spot_id = spot_id
        self.driver_pools.put(driver)
