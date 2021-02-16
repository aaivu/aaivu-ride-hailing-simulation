import time
import logging
from collections import defaultdict

import simpy
import numpy as np
from tqdm.auto import tqdm

import RideSimulator.reward_parameters as rp
from RideSimulator.Driver import Driver
from RideSimulator.Grid import Grid
from RideSimulator.State import State
from RideSimulator.Trip import Trip
from RideSimulator.TripGenerator import TripGenerator, read_data
from RideSimulator.sim_analyzer import SimAnalyzer


def create_drivers(env: simpy.Environment, num: int, map_grid: Grid) -> [Driver]:
    """
    Creates n driver objects in the env Environment.
    The drivers will have randomly generated starting locations and each driver will be assigned to the nearest driver
    pool.
    
    :param env: simpy environment
    :param num: number of drivers
    :param map_grid: grid object the drivers are located in
    :return: list of driver objects
    """
    drivers = []
    for i in range(num):
        driver_loc = map_grid.get_random_location()
        spot_id = map_grid.get_nearest_spot(driver_loc)[0]
        driver = Driver(env=env, driver_id=i, location=driver_loc, spot_id=spot_id)
        map_grid.assign_spot(driver)
        drivers.append(driver)
    return drivers


def check_trip_status(env: simpy.Environment, trip: Trip, action: simpy.Process, peak_time: bool):
    """
    If the nearest driver pool is empty, the trip will be interrupted after a fixed amount of time units and cancel the
    driver request.

    If it is a peak time the waiting time will increase to manage the demand of drivers.
    
    :param env: simpy environment
    :param trip: trip to be monitored
    :param action: driver searching process
    :param peak_time: true if it a peak time
    """
    count = 0
    msg = False
    timeout = TIMEOUT_CHECK

    if peak_time:
        timeout = timeout * 2

    while True:
        # If the trip is interrupted 5 times without a driver, the trip gets cancelled.
        if count >= 5:
            msg = True

        yield env.timeout(timeout)
        count += 1

        if not trip.driver_found and not trip.cancelled:
            logging.debug(f"Interrupting {trip.id} at {env.now}")
            action.interrupt({"cancel": msg})
        else:
            break


def release_driver(env: simpy.Environment, driver: Driver, driver_pool: simpy.FilterStore):
    """
    After a trip is completed the driver is released to the nearest driver pool

    :param env: simpy environment
    :param driver: driver object
    :param driver_pool: nearest driver pool
    """
    if env.now % TIME_UNITS_PER_MIN == 0:
        yield env.timeout(1)
    driver.in_trip = False
    driver_pool.put(driver)


def get_env_state(env: simpy.Environment, trip: Trip, driver: Driver, action: str, map_grid: Grid) -> dict:
    """
    Create the state dict of the environment of the driver and returns the (state, action) pair of the agent.

    :param env: simpy environment
    :param trip: trip assigned to the driver
    :param driver: driver agent
    :param action: action taken by the driver
    :param map_grid: grid object
    :return: (state, action) pair as a dictionary
    """
    trip_state = trip.get_state_info()
    driver_state = driver.get_state_info()

    distances = {
        "trip_distance": map_grid.get_distance(trip_state['pick_loc'], trip_state['drop_loc']),
        "pick_distance": map_grid.get_distance(driver.location, trip_state['pick_loc'])
    }

    env_state = {
        "day_time": int((env.now % UNITS_PER_DAY) / (TIME_UNITS_PER_MIN * 60)),
        "per_km_c": rp.per_km_cost,
        "per_km_r": rp.unit_reward,
        "traffic": 0
    }

    return {
        "state": State(**{**trip_state, **driver_state, **env_state, **distances}),
        "action": action
    }


def handle_trip(env: simpy.Environment, trip_i: Trip, driver_pools: simpy.FilterStore, map_grid: Grid, policy: list,
                trip_gen: TripGenerator):
    """
    The progress of the trip is handled by this function.

    Initially till a driver is found this will cycle through driver pools to find a free driver that accepts the trip.

    Then the trip details will be updated in the 'trip_i' object and the function will timeout for the amount of
    distance between the driver and the rider's pickup location.

    The function will again timeout for the duration of the trip and after completion the driver is released back to the
    driver pool.

    :param env: simpy environment
    :param trip_i: trip object to be handled
    :param driver_pools: the FilterStore where drivers without trips are located
    :param map_grid: grid object
    :param policy: the list of policies the drivers will use
    """
    while not trip_i.driver_found and not trip_i.cancelled:
        logging.debug(f"{trip_i.id} looking for driver at {env.now}")
        try:
            # Search the filterStore for drivers with the same pool_id
            # and that has not been assigned to the trip before.
            with driver_pools.get(
                    lambda d: (d.spot_id in trip_i.nearest_spots) and (d.id not in trip_i.rejected_drivers)) as req:
                trip_i.driver = yield req
                trip_i.driver_found = True
                logging.debug(f"{trip_i.id} got {trip_i.driver.id} at {env.now}")

                if trip_i.driver_found:
                    driver: Driver = trip_i.driver

                    pickup_distance = map_grid.get_distance(driver.location, trip_i.pickup_loc)
                    trip_distance = map_grid.get_distance(trip_i.pickup_loc, trip_i.drop_loc)
                    pickup_time = np.round(pickup_distance * TIME_PER_GRID_UNIT)
                    travel_time = np.round(trip_distance * TIME_PER_GRID_UNIT)

                    # current_location = map_grid.get_location_id(driver.location)
                    # destination = map_grid.get_location_id(trip_i.drop_loc)
                    destination_hex_id = trip_gen.grid.hex_overlay.get_closest_hex(
                        [trip_i.drop_loc[0], trip_i.drop_loc[1]]).id

                    state_dict = get_env_state(env, trip_i, driver, "None", map_grid)
                    env_state: State = state_dict['state']

                    # Observations array TODO - add idle time until this trip
                    observations = [env_state.pickup_distance, env_state.trip_distance,
                                    env_state.day_time,
                                    trip_i.drop_loc[0],
                                    trip_i.drop_loc[1],
                                    max(-1, (driver.weekly_target - env_state.completed_trip_count))]

                    # Check whether the driver accepts the trip.
                    action = trip_i.driver.accept_ride(policy, observations)
                    DRIVER_ACTIONS.append(
                        [trip_i.id, driver.id, trip_i.created_time, env.now, trip_i.pickup_loc, trip_i.drop_loc,
                         driver.location, pickup_time, travel_time, action[0].numpy()])
                    # time_step = tf_env.step(action)

                    opportunity_cost = rp.opportunity_cost * driver.opportunity_cost_weight * (
                            env.now - driver.idle_time_begin)

                    if action[0].numpy() == 1:
                        logging.debug(
                            f"{driver.id} traveling to pickup point, {trip_i.pickup_loc} from {driver.location}")
                        trip_i.update_trip(rp.per_km_cost, rp.unit_reward, pickup_distance, trip_distance,
                                           env_state.day_time)

                        yield env.timeout(pickup_time)

                        add_reward = driver.start_trip(travel_time, trip_i.drop_loc, trip_i)
                        logging.debug(f"Trip distance: {trip_distance}, time {env.now}")

                        # calculate driver reward
                        reward = driver.trip_reward_weight * trip_i.get_net_reward()
                        if add_reward:
                            reward += driver.reward_amount

                    else:
                        env.process(release_driver(env, driver, driver_pools))
                        trip_i.drop_driver(driver.id, driver.location)
                        reward = 0

                    # Save the (state, action) pair in the driver's history.
                    state_dict['observation'] = observations
                    state_dict['reward'] = reward
                    state_dict['action'] = action
                    driver.state_history.append(state_dict)

        # Handle the interrupt call if the trip doesn't find a driver.
        except simpy.Interrupt as interrupt:
            trip_i.update_search_radius()
            trip_i.nearest_spots = map_grid.get_nearest_spot(trip_i.pickup_loc, trip_i.search_radius)
            if interrupt.cause['cancel']:
                trip_i.cancelled = True

    if trip_i.cancelled:
        logging.debug(f"{trip_i.id} was cancelled!")
        return

    yield env.timeout(np.round(trip_distance * TIME_PER_GRID_UNIT))

    # Get the closest driver pool id and release the drive to the pool.
    closest_driver_pool_id = map_grid.get_nearest_spot(trip_i.drop_loc)[0]
    driver.spot_id = closest_driver_pool_id
    driver.idle_time_begin = env.now

    env.process(release_driver(env, driver, driver_pools))
    logging.debug(
        f"{trip_i.id} finished the trip and released driver {driver.id} at {env.now} to pool {closest_driver_pool_id}")


def check_hotspots(env: simpy.Environment, trip_generator: TripGenerator, delay: int = 5):
    """
    This process will check every 5 minutes in simulation time for hotspots on the map. It will scan HexTiles to check
    how many trips were generated in the tile within the past 5 minutes.

    In addition to the number of trips generated in the past 5 minutes, a discounted history of the tile will also be
    considered when classifying it as a hotspot. Therefore, if a tile becomes a hotspot it has a higher chance of
    staying as a hotspot even if only generates a bit lower number of trips in the next time slice.

    :param env: simpy environment
    :param trip_generator: trip generator object used to create trips for the simulation
    :param delay: how often the grid should be searched for hotspots in minutes
    :return:
    """
    hex_overlay = trip_generator.grid.hex_overlay
    # print("hex count ", len(hex_overlay.tiles.flatten()))
    discount_factor = rp.discount_factor
    threshold_count = rp.trip_threshold * (1 + discount_factor)
    hotspot_count = 0
    previous_hotspots = set()

    try:
        while True:
            yield env.timeout(delay * TIME_UNITS_PER_MIN)
            hotspots = set()

            # Ids of the hexes that trips were generated in the past time slice
            updated_hex_ids = trip_generator.updated_hex_ids

            for hex_id in updated_hex_ids:
                u_hex = hex_overlay.get_hex_from_id(hex_id=hex_id)
                u_hex.history = u_hex.trip_count + discount_factor * u_hex.history
                u_hex.trip_count = 0

                # If the discounted value of the trip count exceeds the threshold, update the additional reward given
                # for the trips that are generated in that tile.
                if u_hex.history > threshold_count:
                    u_hex.update_additional_reward()
                    hotspots.add(hex_id)
                    hotspot_count += 1

            # Hotspots become cold_spots when the number of trips generated in the HexTile falls below the threshold.
            # Therefore, the additional reward is reset for these tiles.
            cold_spots = previous_hotspots - hotspots
            for cold_spot in cold_spots:
                c_hex = hex_overlay.get_hex_from_id(hex_id=cold_spot)
                c_hex.additional_reward = 0

            # Store the hotspots of this time slice to compare with the next batch to identify colds_spots/
            previous_hotspots = hotspots
            trip_generator.updated_hex_ids = set()

    except simpy.Interrupt as interrupt:
        logging.debug(f'{interrupt.cause}, "Hotspot count:", {hotspot_count}')


def generate_trips(env: simpy.Environment, map_grid: Grid, driver_pools: simpy.FilterStore, policy: list,
                   trip_history: list, run_time: int, progress_bar=False, seed: int = None):
    """
    The number of trips generated is determined by the amount of time the simulation is run.

    Ths function generates trip processes and hands them over to the handle_trip() function.
    Another process is created for each trip to check the status of the trip and to handle the timeout and cancellation.

    :param env: simpy environment
    :param map_grid: grid object where trips are located
    :param driver_pools: the FilterStore where drivers without trips are located
    :param policy: the list of policies the drivers will use to determine whether accept or reject the trip
    :param trip_history: an empty list to stores all the trips generated
    :param run_time: how long the simulation should run
    :param progress_bar: toggle the display of the progress bar
    :param seed: seed to be used for rng
    """

    # A trip generator object handles the sampling and generation of trips using real world data.
    trip_gen = TripGenerator(grid=map_grid, time_unit=TIME_UNITS_PER_MIN, trips_per_week=TRIPS_PER_WEEK, seed=seed)
    hotspot_check = env.process(check_hotspots(env=env, trip_generator=trip_gen))

    if progress_bar:
        week_count = run_time / (UNITS_PER_DAY * 7)
        final_week_days = (run_time % (UNITS_PER_DAY * 7)) / UNITS_PER_DAY
        final_hours = (run_time % UNITS_PER_DAY) / (TIME_UNITS_PER_MIN * 60)

        print(f"The Simulation will run for {np.floor(week_count)} weeks, {np.floor(final_week_days)} days, "
              f"{final_hours} hours.")

        final_week, final_day, final_hour = (False, False, False)

        # Iterate for each week
        with tqdm(total=np.ceil(week_count), desc="Weeks", leave=False) as week_bar:
            for z in range(np.ceil(week_count).astype(int)):
                num_days = 7
                if z == np.floor(week_count):
                    final_week = True
                    num_days = np.ceil(final_week_days).astype(int)

                # Iterate for each day
                with tqdm(total=num_days, desc="Days", leave=False) as day_bar:
                    for y in range(num_days):
                        num_hours = 24
                        if y == np.floor(final_week_days) and final_week:
                            num_hours = np.ceil(final_hour).astype(int)

                        # Iterate for each hour
                        with tqdm(total=num_hours, desc="Hours", leave=False) as hour_bar:
                            for x in range(num_hours):
                                for _ in range(60):
                                    trips, peak_time = trip_gen.generate_trips(env=env)
                                    for trip_i in trips:
                                        trip_history.append(trip_i)

                                        # Creates a trip process along with a interruption call to check the driver
                                        # assignment.
                                        ride_find = env.process(
                                            handle_trip(env, trip_i, driver_pools, map_grid, policy, trip_gen))
                                        env.process(check_trip_status(env, trip_i, ride_find, peak_time))
                                    yield env.timeout(TIME_UNITS_PER_MIN)

                                hour_bar.update(1)
                        day_bar.update(1)
                week_bar.update(1)
            yield env.timeout(TIME_UNITS_PER_MIN * 5)
            hotspot_check.interrupt("Simulation Ended")

    else:
        num_minutes = int(RUN_TIME / TIME_UNITS_PER_MIN)
        for i in range(num_minutes):
            trips, peak_time = trip_gen.generate_trips(env=env)

            for trip_i in trips:
                trip_history.append(trip_i)

                # Creates a trip process along with a interruption call to check the driver assignment.
                ride_find = env.process(handle_trip(env, trip_i, driver_pools, map_grid, policy, trip_gen))
                env.process(check_trip_status(env, trip_i, ride_find, peak_time))

            yield env.timeout(TIME_UNITS_PER_MIN)

        yield env.timeout(TIME_UNITS_PER_MIN * 5)
        hotspot_check.interrupt("Simulation Ended")


def reset_week(env: simpy.Environment, driver_list: [Driver]):
    """
    After a week is completed in the simulation there are several parameters to be reset. Such as,
        - Driver trip count
        - Driver weekly reward status
        - Driver weekly goals and reward
    Ths process takes care of those parameter once the simulation completes a week.

    :param env: simpy environment
    :param driver_list: list of drivers that the parameters should be reset
    """
    week_duration = UNITS_PER_DAY * 7
    num_weeks = RUN_TIME // week_duration
    count = 0

    while count < num_weeks:
        yield env.timeout(week_duration)
        count += 1
        logging.debug(f"{'*' * 20} Updating driver goals {'*' * 20}\n")
        for driver in driver_list:
            # Based on the current number of trips completed the goals for the next week is calculated.
            next_goal, next_reward = rp.get_next_goal(driver.weekly_trip_count)
            logging.debug(
                f"Driver {driver.id} completed {driver.weekly_trip_count} trips, "
                f"New goal: {next_goal}, New reward: {next_reward}")
            driver.reset_weekly_info(next_goal, next_reward)
        logging.debug("\n")


def update_parameters(param_dict):
    global TRIPS_PER_WEEK, NUM_DAYS, DRIVER_COUNT, GRID_WIDTH, GRID_HEIGHT, HEX_AREA, RUN_TIME
    parameters = ["TRIPS_PER_WEEK", "NUM_DAYS", "DRIVER_COUNT", "GRID_WIDTH", "GRID_HEIGHT", "HEX_AREA"]
    for param in parameters:
        if param in param_dict:
            if param == "TRIPS_PER_WEEK":
                TRIPS_PER_WEEK = param_dict[param]
            if param == "NUM_DAYS":
                NUM_DAYS = param_dict[param]
            if param == "DRIVER_COUNT":
                DRIVER_COUNT = param_dict[param]
            if param == "GRID_WIDTH":
                GRID_WIDTH = param_dict[param]
            if param == "GRID_HEIGHT":
                GRID_HEIGHT = param_dict[param]
            if param == "HEX_AREA":
                HEX_AREA = param_dict[param]
    RUN_TIME = UNITS_PER_DAY * NUM_DAYS

    # print(TRIPS_PER_WEEK, NUM_DAYS, DRIVER_COUNT, GRID_WIDTH, GRID_HEIGHT, HEX_AREA, RUN_TIME)


GRID_UNITS_PER_KM = 4  # Number of grid units per km
TIME_UNITS_PER_MIN = 10  # 10 simulation time unit = 1 min  !!! DON'T CHANGE !!!
MIN_PER_KM = 0.5  # Assumption: It takes 1 min to travel 1km

TIME_PER_GRID_UNIT = MIN_PER_KM * TIME_UNITS_PER_MIN / GRID_UNITS_PER_KM
TIMEOUT_CHECK = TIME_UNITS_PER_MIN * 5
UNITS_PER_DAY = TIME_UNITS_PER_MIN * 60 * 24
NUM_DAYS = 7

TRIPS_PER_WEEK = 2000  # Average number of trips per week needed to be simulated
RUN_TIME = UNITS_PER_DAY * NUM_DAYS  # Number of days the simulation should run
DRIVER_ACTIONS = []

DRIVER_COUNT = 20
GRID_WIDTH = 60
GRID_HEIGHT = 60
HEX_AREA = 1.2 * 4

read_data()


def run_simulation(policy: list = None, debug=False, progress_bar=False, seed=None) -> dict:
    """
    This function is called with the policy list to start the simulation.

    :param seed:
    :param policy: lost of policies
    :param debug: toggle debug log
    :param progress_bar: toggle progress bar
    :return: a dictionary containing the history of all the drivers with respective to the policy they used.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    if policy is None:
        policy = ['random']
    
    if len(policy) > 1:
        print("Multiple policies given, training driver subsets")
        
    elif policy[0] == "random":
        print("Using random policy")

    elif policy[0] == "positive":
        print("Using positive reward policy")

    elif policy[0] == "all":
        print("Using accept all policy")

    else:
        if type(policy[0] == 'str'):
            print("No valid action provided, using random policy")

    driver_count = DRIVER_COUNT
    interval = 6
    grid_width = GRID_WIDTH  # 3809
    grid_height = GRID_HEIGHT  # 2622
    hex_area = HEX_AREA

    env = simpy.Environment()
    map_grid = Grid(env=env, width=grid_width, height=grid_height, interval=interval, num_drivers=driver_count,
                    hex_area=hex_area, units_per_km=GRID_UNITS_PER_KM, seed=seed)
    driver_list = create_drivers(env, driver_count, map_grid)
    driver_pools = map_grid.driver_pools
    trip_history = []

    env.process(
        generate_trips(env, map_grid, driver_pools, policy, trip_history, RUN_TIME, progress_bar=progress_bar,
                       seed=seed))
    env.process(reset_week(env, driver_list=driver_list))
    env.run(until=(RUN_TIME + TIME_UNITS_PER_MIN * 60 * 5))

    analyzer = SimAnalyzer(trips=trip_history, drivers=driver_list, actions=DRIVER_ACTIONS)

    if len(driver_pools.items) != len(driver_list):
        logging.warning(f'{"*" * 30} "DRIVERS MISSING!" {"*" * 30}')

    driver_states = defaultdict(list)
    for item in driver_pools.items:
        driver_states[item.id % len(policy)].append(item.state_history)

    print("Number of trips generated:", len(trip_history))
    return driver_states, analyzer
