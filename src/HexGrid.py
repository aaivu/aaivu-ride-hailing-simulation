import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from RideSimulator.HexTile import HexTile
from matplotlib.patches import RegularPolygon


class HexGrid(object):

    def __init__(self, hex_area: float, width: int, height: int, units_per_km: int = 1):
        self.area = hex_area
        self.units_per_km = units_per_km
        self.radius = self.calculate_radius(hex_area)
        self.h_shift = 1.5 * self.radius
        self.v_shift = 1.5 * (2 * np.sin(np.radians(60)) * self.radius)
        self.width = width
        self.height = height
        self.hex_id_mapping = defaultdict()
        self.tiles: np.ndarray = self.create_hex_tiles()
        self.weather: dict = dict()
        self.traffic: dict = dict()

    def get_approx_loc(self, location: np.ndarray) -> np.ndarray:
        x_num = (location[0] + self.h_shift) / (1.5 * self.radius)
        y_num = (location[1] + self.v_shift) / (self.radius * np.sin(np.radians(60)))

        return np.array(list(map(int, [np.floor(x_num / 2), np.floor(y_num)])))

    def get_closest_hex(self, location: np.ndarray) -> HexTile:
        height, width = self.tiles.shape
        x, y = self.get_approx_loc(location=location)
        candidate_hexes = []

        for i in range(x, min(x + 1, width - 1) + 1):
            for j in range(y, min(y + 1, height - 1) + 1):
                candidate_hexes.append(self.tiles[j][i])

        distance = np.inf
        tile = None
        for hex_tile in candidate_hexes:
            hex_distance = np.linalg.norm(location - hex_tile.center)
            if hex_distance < distance:
                distance = hex_distance
                tile = hex_tile

        return tile

    def get_closet_ring(self, hex_tile: HexTile):
        x, y = hex_tile.col_no, hex_tile.row_no
        height, width = self.tiles.shape
        top = min(y + 2, height - 1)
        bot = max(y - 2, 0)
        if y % 2 == 0:
            right = x
            left = max(x - 1, 0)
        else:
            right = min(x + 1, width - 1)
            left = x

        hexes = [self.tiles[top][x], self.tiles[bot][x]]

        if y == 1:
            bot = -1
        elif y == height - 2:
            top = height

        for i in range(bot + 1, top, 2):
            hexes.append(self.tiles[i][left])
            hexes.append(self.tiles[i][right])
        return hexes

    def create_hex_tiles(self) -> np.ndarray:
        lat_points = self.height
        lon_points = self.width
        num_cols = int(np.ceil(lon_points / (self.radius * 3))) + 2
        num_rows = int(np.ceil(lat_points / (np.sin(np.radians(60)) * self.radius))) + 6

        x_coords = []
        y_coords = []
        v_pos = 0
        h_pos = 0

        for row in range(num_rows):
            y_coords += [v_pos] * num_cols
            v_pos += np.sin(np.radians(60)) * self.radius

            for col in range(num_cols):
                x_coords.append(h_pos)
                h_pos += 3 * self.radius

            if row % 2 == 1:
                h_pos = 0
            else:
                h_pos = 1.5 * self.radius

        hex_tiles = []
        x_coords, y_coords = np.array(x_coords), np.array(y_coords)
        x_coords = x_coords - self.h_shift
        y_coords = y_coords - self.v_shift

        for i, c in enumerate(zip(x_coords, y_coords)):
            hex_tile = HexTile(hex_id=i, center_loc=np.array(c), row_no=i // num_cols, col_no=i % num_cols)
            hex_tiles.append(hex_tile)

        hex_tiles = np.array(hex_tiles).reshape((num_rows, num_cols))

        for z in range(num_rows):
            for y in range(num_cols):
                hex_tile = hex_tiles[z][y]
                self.hex_id_mapping[hex_tile.id] = [z, y]

        return hex_tiles

    def get_hex_from_id(self, hex_id: int) -> HexTile:
        hex_loc = self.hex_id_mapping[hex_id]
        return self.tiles[hex_loc[0]][hex_loc[1]]

    def plot_grid(self, colour_tiles: [HexTile], marker=None, show_count=False):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_aspect('equal')
        height, width = self.tiles.shape
        cols = np.array(['b'] * (height * width)).reshape(height, width)
        for tile in colour_tiles:
            cols[tile.row_no][tile.col_no] = 'k'

        cols = cols.flatten()
        for tile, c in zip(self.tiles.flatten(), cols):
            loc = tile.center
            hexagon = RegularPolygon((loc[0], loc[1]), numVertices=6, radius=self.radius, facecolor=c,
                                     orientation=np.radians(30), alpha=0.2, edgecolor='k')
            ax.add_patch(hexagon)
            if show_count:
                plt.text(loc[0], loc[1], str(tile.trip_count))
        ax.scatter(0, 0, alpha=0.1, color='k')
        if marker is not None:
            ax.scatter(marker[0], marker[1], alpha=1, color='r')

        plt.show()

    def calculate_radius(self, hex_area: float) -> float:
        return self.units_per_km * (3 ** 0.25 * (2 * hex_area / 9) ** 0.5)

    def update_hex_area(self, area: float):
        self.area = area
        self.radius = self.calculate_radius(area)
        self.tiles = self.create_hex_tiles()

    def update_traffic_details(self, traffic_details: dict):
        self.traffic = traffic_details

    def update_weather_details(self, weather_details:dict):
        self.weather = weather_details
