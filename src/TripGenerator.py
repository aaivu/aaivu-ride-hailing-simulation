import os
import simpy
import numpy as np
from pathlib import Path
from RideSimulator.Grid import Grid
from RideSimulator.Trip import Trip

lat_points, lon_points, trip_distances, trips_per_min = None, None, None, None


def read_data(directory="data", lon_file="lon_points", lat_file="lat_points", distance_file="trip_distances",
              min_file="trips_per_min"):
    global lat_points, lon_points, trip_distances, trips_per_min
    data_path = directory
    if Path(os.getcwd()).parts[-1] != "RideSimulator":
        data_path = os.path.join(Path(os.getcwd()).parent, os.path.join('RideSimulator', directory))

    print("Loading trip data...")
    lat_points = np.loadtxt(os.path.join(data_path, lat_file))
    lon_points = np.loadtxt(os.path.join(data_path, lon_file))
    trip_distances = np.loadtxt(os.path.join(data_path, distance_file))
    trips_per_min = np.loadtxt(os.path.join(data_path, min_file))
    print("Data loading complete")


class TripGenerator(object):
    def __init__(self, grid: Grid, time_unit, trips_per_week=20000, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.id = 0
        self.grid = grid
        self.width = grid.width
        self.height = grid.height
        self.granularity = 1000
        self.min_trip_distance = self.width / 100
        self.max_displacement = self.width * 0.05 * self.granularity

        self.time_unit = time_unit
        self.units_per_day = time_unit * 60 * 24
        time_slice = self.units_per_day // 24
        self.peak_times = [time_slice * 4, time_slice * 5, time_slice * 14, time_slice * 15]

        self.lat_points, self.lon_points, self.trip_distances = self.import_data()
        self.trips_per_min = self.scale_trip_count(trips_per_week)
        self.updated_hex_ids = set()

    def import_data(self):

        lat_points_copy = lat_points.copy()
        lon_points_copy = lon_points.copy()
        trip_distances_copy = trip_distances.copy()

        np.random.shuffle(lat_points_copy)
        np.random.shuffle(lon_points_copy)
        np.random.shuffle(trip_distances_copy)

        lat_points_scaled = lat_points_copy - lat_points_copy.min()
        lon_points_scaled = lon_points - lon_points_copy.min()

        lat_scale = self.height / lat_points_scaled.max()
        lon_scale = self.width / lon_points_scaled.max()
        distance_scale = self.width / trip_distances_copy.max()

        lat_points_scaled = lat_points_scaled * lat_scale
        lon_points_scaled = lon_points_scaled * lon_scale
        distances_scaled = trip_distances * distance_scale

        return lat_points_scaled, lon_points_scaled, distances_scaled

    @staticmethod
    def scale_trip_count(trips_per_week):
        scale = trips_per_min.sum() * 7 // trips_per_week
        scaled_hourly = (trips_per_min / scale).astype(int)
        dif = 0
        sampled_trips = []
        for i in range(24):
            h = i * 60
            full_trips = (trips_per_min[h:h + 60] / scale) // 1
            prob_trips = (trips_per_min[h:h + 60] / scale) % 1
            total_trips = []

            sample_trips = (np.random.rand(60) < prob_trips).astype(int)
            total_trips += (full_trips + sample_trips).tolist()

            sampled_trips += (full_trips + sample_trips).tolist()
            trip_count = int(sum(total_trips) / 100)
            dif += trip_count - scaled_hourly[i]

        return np.array(sampled_trips).astype(int)

    def get_trip_locations(self, x, y, distance):

        x = min(self.width, x + np.random.randint(0, self.max_displacement) / self.granularity)
        y = min(self.height, y + np.random.randint(0, self.max_displacement) / self.granularity)

        start_hex = self.grid.hex_overlay.get_closest_hex([x, y])
        start_hex.trip_count += 1

        theta = np.rad2deg(np.random.random() * 2 * np.pi)

        while distance < self.min_trip_distance:
            distance = distance * 1.5

        while True:
            d_x = x + distance * np.cos(theta)
            d_y = y + distance * np.sin(theta)

            x_count = 0
            y_count = 0

            while (d_x > self.width or d_x < 0) and x_count < 2:
                # print("switching x direction", d_x)
                d_x = x - distance * np.cos(theta)
                x_count += 1

            while (d_y > self.height or d_y < 0) and y_count < 2:
                # print("switching y direction", d_y)
                d_y = y - distance * np.sin(theta)
                y_count += 1

            if x_count == 2 or y_count == 2:
                # print("Reducing distance", distance)
                distance = distance / 2
            else:
                prob = np.random.random()
                if prob < 0.2:
                    return np.array([[d_x, d_y], [x, y], [start_hex.id, start_hex.additional_reward]])
                else:
                    return np.array([[x, y], [d_x, d_y], [start_hex.id, start_hex.additional_reward]])

    def create_trip(self, env: simpy.Environment, trip_id: int) -> Trip:
        """
        Creates a trip in the env with the given trip_id.
        The trip will have randomly generated pickup and drop locations, and the id of the nearest driver pool is
        assigned to the trip.
        Pickup and drop locations will not be the same.

        :param env: simpy environment
        :param trip_id: trip id
        :return: trip object
        """
        distance = self.trip_distances[trip_id]
        pick_up_loc, drop_loc, hex_data = self.get_trip_locations(self.lon_points[trip_id], self.lat_points[trip_id],
                                                                  distance)
        self.updated_hex_ids.add(hex_data[0])
        nearest_spot = self.grid.get_nearest_spot(pick_up_loc)[0]
        trip_i = Trip(env, trip_id, pick_up_loc, drop_loc, [nearest_spot], hex_data[0], hex_data[1])
        return trip_i

    def generate_trips(self, env: simpy.Environment):
        peak_time = False
        day_time = int((env.now % self.units_per_day) / self.time_unit)
        num_trips = self.trips_per_min[day_time]
        trips = []
        for _ in range(num_trips):
            trips.append(self.create_trip(env, self.id))
            self.id += 1

        return trips, peak_time
