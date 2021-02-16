import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from RideSimulator.Trip import Trip
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection


class SimAnalyzer(object):
    def __init__(self, trips: [Trip], drivers, actions):
        self.trips = trips
        self.total_trips = len(trips)
        self.drivers = drivers
        self.driver_actions = pd.DataFrame(
            actions,
            columns=["trip_id", "driver_id", "created_time", "assigned_time", "pickup_loc", "drop_loc", "driver_loc",
                     "pickup_time", "travel_time", "driver_action"]).set_index(["trip_id", "driver_id"])

        self.trip_data = pd.DataFrame(
            [vars(t) for t in trips]).drop(columns=["driver", "env"]).set_index("id")

    def get_completed_trips(self, driver_info=False):
        t_count = 0
        for driver in self.drivers:
            d_t_count = driver.weekly_trip_count
            t_count += d_t_count
            if driver_info:
                print(f"Driver {driver.id} completed {d_t_count} trips")

        return t_count

    def extract_trip_locations(self):
        return self.trip_data["pickup_loc"], self.trip_data["drop_loc"]

    def get_trip_data(self):
        return self.trip_data

    def cumulative_net_trip_reward(self):
        net_rewards = self.trip_data["trip_reward"] - self.trip_data["total_cost"]
        return net_rewards

    def driver_rewards(self, driver_info=False):
        d_rewards = []
        for driver in self.drivers:
            d_rewards.append([driver.total_reward, driver.weekly_reward_reached])
            # if driver_info:
            # print(
            #    f"{driver.id} obtained a reward of {driver.total_reward} and "
            #    f"{'' if driver.weekly_reward_reached else 'did not'} achieved his weekly target")
        return d_rewards

    def trip_distribution(self):
        self.trip_data["created_time"].hist(bins=24 * 7)
        plt.show()

    def completed_percentage(self, only_starved=False):
        cancelled_df = (self.trip_data[self.trip_data["cancelled"] == True]).copy()
        if only_starved:
            cancelled_df['num_rejected'] = cancelled_df['rejected_drivers'].apply(len)
            cancelled_df = cancelled_df[cancelled_df['num_rejected'] > 0]

        return (self.total_trips - len(cancelled_df)) / self.total_trips

    def plot_rejected_trips(self):
        cancelled_trips = self.trip_data[self.trip_data["cancelled"] == True]
        pick_loc = cancelled_trips['pickup_loc'].values.tolist()
        drop_loc = cancelled_trips['drop_loc'].values.tolist()

        for line in zip(pick_loc, drop_loc):
            line = np.array(line)
            plt.plot(line[:, 0], line[:, 1], color='black', zorder=1)
            plt.scatter(line[0, 0], line[0, 1], marker='o', color='green', zorder=2)
            plt.scatter(line[1, 0], line[1, 1], marker='x', color='red', zorder=2)

        if len(pick_loc) > 0:
            plt.show()

    def plot_driver_states(self, driver_id, trip_count=50, start_time=0, fig_size=(30, 2), legend_size=6,
                           xtick_size=10):
        driver_data = self.driver_actions.xs(driver_id, level=1).copy()
        driver_data.sort_values(by='assigned_time', inplace=True)

        trip_times = []
        for i in driver_data[['assigned_time', 'driver_action', 'pickup_time', 'travel_time']].values:
            if i[1] == 1:
                assign_time = i[0]
                pickup_time = assign_time + i[2]
                drop_time = pickup_time + i[3]
                points = [assign_time, pickup_time, drop_time]
                trip_times.append(points)

        states = {'idle': 1, 'pick_up': 1, 'in_trip': 1}
        color_mapping = {"idle": "C0", "pick_up": "C1", "in_trip": "C2"}
        state_mapping = {0: "pick_up", 1: 'in_trip'}

        vertices = []
        colors = []
        end_time = start_time
        for t in trip_times[:trip_count]:
            if t[2] < end_time:
                continue
            v = [
                (end_time, states['idle'] - 0.4),
                (end_time, states['idle'] + 0.4),
                (t[0], states['idle'] + 0.4),
                (t[0], states['idle'] - 0.4),
                (end_time, states['idle'] - 0.4),
            ]

            vertices.append(v)
            colors.append(color_mapping['idle'])

            for k in range(2):
                state = states[state_mapping[k]]
                v = [
                    (t[k], state - 0.4),
                    (t[k], state + 0.4),
                    (t[k + 1], state + 0.4),
                    (t[k + 1], state - 0.4),
                    (t[k], state - 0.4),
                ]

                vertices.append(v)
                colors.append(color_mapping[state_mapping[k]])

            end_time = t[2]

        bars = PolyCollection(vertices, facecolors=colors)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.add_collection(bars)
        ax.get_yaxis().set_visible(False)
        ax.autoscale()

        ax.tick_params(axis='x', which='major', labelsize=xtick_size)
        ax.tick_params(axis='x', which='minor', labelsize=xtick_size)

        idle = mpatches.Patch(color='C0', label='Idle time')
        pick_up = mpatches.Patch(color='C1', label='Pick up time')
        in_trip = mpatches.Patch(color='C2', label='Trip time')

        plt.legend(handles=[idle, pick_up, in_trip], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                   prop={'size': legend_size})
        plt.show()

    def plot_driver_movement(self, driver_id, number_of_trips, first_trip=0):
        driver_data = self.driver_actions.xs(driver_id, level=1)
        driver_data = driver_data[driver_data['driver_action'] == 1][first_trip:number_of_trips]
        pick_loc = driver_data['pickup_loc']
        drop_loc = driver_data['drop_loc']

        start_loc = driver_data[driver_data['driver_action'] == 1].iloc[0]['driver_loc']

        plt.figure(figsize=(10, 10))
        plt.scatter(start_loc[0], start_loc[1], marker="*", color='purple', s=400)
        for line in zip(pick_loc, drop_loc):
            line = np.array(line)
            plt.plot([start_loc[0], line[0][0]], [start_loc[1], line[0][1]], color='black', zorder=1)
            plt.plot(line[:, 0], line[:, 1], color='blue', zorder=1)
            plt.scatter(line[0, 0], line[0, 1], marker='o', color='green', zorder=2)
            plt.scatter(line[1, 0], line[1, 1], marker='x', color='red', zorder=2)
            start_loc = line[1]

        plt.show()
