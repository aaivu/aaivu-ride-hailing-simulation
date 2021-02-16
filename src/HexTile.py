import numpy as np
import RideSimulator.reward_parameters as rp


class HexTile(object):
    def __init__(self, hex_id: int, center_loc: np.ndarray, row_no, col_no, traffic=0, weather=0):
        self.id = hex_id
        self.row_no = row_no
        self.col_no = col_no
        self.center = center_loc
        self.traffic = traffic
        self.weather = weather
        self.trip_count = 0
        self.history = 0
        self.additional_reward = 0

    def update_additional_reward(self):
        self.additional_reward = min(3, self.history - rp.trip_threshold) / 3 * rp.hotspot_max_reward
        #self.additional_reward = 0

    def update_traffic(self):
        return

    def update_weather(self):
        return