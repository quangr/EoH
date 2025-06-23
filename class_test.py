import numpy as np


import math # Not strictly used now, but often useful in geometry

class BasePlacementAlgorithm:
    def __init__(self, support_threshold=0.8, epsilon=1e-6):
        if not (0.0 <= support_threshold <= 1.0):
            raise ValueError("support_threshold must be between 0.0 and 1.0")
        if not epsilon > 0:
            raise ValueError("epsilon should generally be a small positive value.")

        self.support_threshold = support_threshold
        self.epsilon = epsilon

    # --- Core Geometric and Validation Primitives ---

    def _check_overlap_3d(self, item1_spatial_info: tuple, item2_spatial_info: tuple):
        """
        Checks if two 3D rectangular items' interiors overlap.
        item_spatial_info is expected to be ((x,y,z), (l,w,h)).

        Args:
            item1_spatial_info (tuple): ((x,y,z), (l,w,h)) for item 1.
            item2_spatial_info (tuple): ((x,y,z), (l,w,h)) for item 2.

        Returns:
            bool: True if item interiors overlap, False otherwise.
        """
        item1_pos, item1_dims = item1_spatial_info
        item2_pos, item2_dims = item2_spatial_info

        x1, y1, z1 = item1_pos
        l1, w1, h1 = item1_dims
        x2, y2, z2 = item2_pos
        l2, w2, h2 = item2_dims

        overlap_x = (x1 < x2 + l2) and (x1 + l1 > x2)
        overlap_y = (y1 < y2 + w2) and (y1 + w1 > y2)
        overlap_z = (z1 < z2 + h2) and (z1 + h1 > z2)

        return overlap_x and overlap_y and overlap_z

    def _is_within_truck_bounds(self, item_spatial_info: tuple, truck_dims: tuple):
        """
        Checks if an item is completely within the truck's dimensions.
        item_spatial_info is expected to be ((x,y,z), (l,w,h)).

        Args:
            item_spatial_info (tuple): ((x,y,z), (l,w,h)) of the item.
            truck_dims (tuple): (length, width, height) of the truck.

        Returns:
            bool: True if the item is within bounds (considering epsilon), False otherwise.
        """
        item_pos, item_dims = item_spatial_info
        px, py, pz = item_pos
        pl, pw, ph = item_dims
        tl, tw, th = truck_dims

        if not (px >= 0.0 - self.epsilon and px + pl <= tl + self.epsilon):
            return False
        if not (py >= 0.0 - self.epsilon and py + pw <= tw + self.epsilon):
            return False
        if not (pz >= 0.0 - self.epsilon and pz + ph <= th + self.epsilon):
            return False
        return True

    def _check_support(self, new_item_spatial_info: tuple, placed_items_spatial_info_list: list):
        """
        Checks if an item is adequately supported.
        new_item_spatial_info is ((x,y,z), (l,w,h)).
        placed_items_spatial_info_list is a list of such tuples.

        Args:
            new_item_spatial_info (tuple): ((x,y,z), (l,w,h)) of the new item.
            placed_items_spatial_info_list (list): List of ((x,y,z), (l,w,h)) for placed items.

        Returns:
            bool: True if the item is sufficiently supported, False otherwise.
        """
        new_item_pos, new_item_dims = new_item_spatial_info
        new_x, new_y, new_z = new_item_pos
        new_l, new_w, _ = new_item_dims

        if new_z < self.epsilon:
            return True

        new_item_base_area = new_l * new_w
        if new_item_base_area < self.epsilon:
            return True

        total_supported_area = 0.0
        for ex_item_pos, ex_item_dims in placed_items_spatial_info_list:
            ex, ey, ez = ex_item_pos
            el, ew, eh = ex_item_dims
            
            if abs(new_z - (ez + eh)) < self.epsilon:
                x_overlap = max(0.0, min(new_x + new_l, ex + el) - max(new_x, ex))
                y_overlap = max(0.0, min(new_y + new_w, ey + ew) - max(new_y, ey))
                total_supported_area += x_overlap * y_overlap
        
        return (total_supported_area / new_item_base_area) >= (self.support_threshold - self.epsilon)

    # --- Data Processing Primitives ---

    def _get_item_details(self, item):
        # This method remains unchanged as it processes raw item data
        l = item.get("length", 0.0)
        w = item.get("width", 0.0)
        h = item.get("height", 0.0)
        wt = item.get("weight", 0.0)
        item_id = item.get("item_id", item.get("id", None))
        vol = l * w * h
        footprint = l * w
        return {
            "id": item_id, "l": l, "w": w, "h": h, "weight": wt,
            "vol": vol, "footprint": footprint, "dims": (l, w, h),
            "original_item": item,
        }

    def _get_truck_details(self, truck_dict, truck_type_info):
        # This method's output for placed items is already in the desired format
        capacity, tr_l, tr_w, tr_h = truck_type_info
        current_weight = truck_dict.get("current_weight", 0.0)

        placed_items_spatial_info = [] # Renamed for clarity
        current_occupied_volume_sum = 0.0
        raw_occupied_volumes = truck_dict.get("occupied_volumes", [])

        for placed_data in raw_occupied_volumes:
            _item_id, x, y, z, l, w, h = placed_data[0:7]
            placed_items_spatial_info.append(((x, y, z), (l, w, h))) # This is our (pos, dims) tuple
            current_occupied_volume_sum += l * w * h

        total_truck_volume = tr_l * tr_w * tr_h
        remaining_weight_capacity = capacity - current_weight
        remaining_physical_volume = total_truck_volume - current_occupied_volume_sum

        return {
            "truck_type_index": truck_dict.get("truck_type_index"),
            "capacity": capacity, "length": tr_l, "width": tr_w, "height": tr_h,
            "dims": (tr_l, tr_w, tr_h), "total_volume": total_truck_volume,
            "current_weight": current_weight,
            "placed_items_spatial_info": placed_items_spatial_info, # Key change in name for clarity
            "raw_occupied_volumes": raw_occupied_volumes,
            "current_occupied_volume_sum": current_occupied_volume_sum,
            "remaining_weight_capacity": remaining_weight_capacity,
            "remaining_physical_volume": remaining_physical_volume,
            "original_truck_dict": truck_dict,
        }

    # --- Placement Strategy Primitives ---

    def _is_placement_valid(self, new_item_pos: tuple, new_item_details: dict, truck_details: dict):
        """
        Checks if placing an item at a given position in a truck is valid.

        Args:
            new_item_pos (tuple): The (x,y,z) position for the new item.
            new_item_details (dict): Processed details of the item to be placed.
            truck_details (dict): Processed details of the truck.

        Returns:
            bool: True if the placement is valid, False otherwise.
        """
        # Construct the standardized spatial info tuple for the new item
        new_item_spatial_info = (new_item_pos, new_item_details["dims"])

        # 1. Check if the item is within the truck's boundaries
        if not self._is_within_truck_bounds(
            new_item_spatial_info, truck_details["dims"]
        ):
            return False

        # 2. Check for overlap with already placed items in the truck.
        for placed_item_info in truck_details["placed_items_spatial_info"]:
            if self._check_overlap_3d(
                new_item_spatial_info,
                placed_item_info, # This is already an ((x,y,z),(l,w,h)) tuple
            ):
                return False # Overlap detected

        # 3. Check for sufficient support for the new item.
        if not self._check_support(
            new_item_spatial_info,
            truck_details["placed_items_spatial_info"],
        ):
            return False # Insufficient support

        return True

import numpy as np
import numpy as np

class PlacementAlgorithm(BasePlacementAlgorithm):
    def __init__(self, support_threshold=0.8, epsilon=1e-6):
        super().__init__(support_threshold, epsilon)

    def place_item(self, unplaced_items, trucks_in_use, truck_types):
        #{This algorithm prioritizes filling existing trucks by iterating through unplaced items and available trucks, finding the best valid placement based on a simple heuristic, and if no placement is found, it opens a new truck.}
        truck_index = -1
        item_index = -1
        x = 0.0
        y = 0.0
        z = 0.0
        truck_type_index = 0

        item_index = self._select_item(unplaced_items)
        if item_index == -1:
            return -1, -1, 0.0, 0.0, 0.0, 0
        
        item = unplaced_items[item_index]
        item_details = self._get_item_details(item)

        truck_index = self._find_best_truck(item_details, trucks_in_use, truck_types)
        
        if truck_index != -1:
            truck = trucks_in_use[truck_index]
            truck_details = self._get_truck_details(truck, truck_types[truck["truck_type_index"]])

            x, y, z = self._find_best_position_in_truck(item_details, truck_details)

            if x is not None and y is not None and z is not None:
                return truck_index, item_index, x, y, z, truck["truck_type_index"]
            else:
                truck_index = -1
                
        if truck_index == -1:
            truck_type_index = self._select_truck_type(item_details, truck_types)
            if truck_type_index != -1:
                return -1, item_index, 0.0, 0.0, 0.0, truck_type_index
        
        return -1, item_index, 0.0, 0.0, 0.0, truck_type_index

    def _select_item(self, unplaced_items):
        if not unplaced_items:
            return -1
        return 0

    def _find_best_truck(self, item_details, trucks_in_use, truck_types):
        best_truck_index = -1
        max_weight_utilization = -1.0
        for i, truck in enumerate(trucks_in_use):
            truck_type_index = truck["truck_type_index"]
            truck_details = self._get_truck_details(truck, truck_types[truck_type_index])
            
            if truck_details["remaining_weight_capacity"] >= item_details["weight"]:
                x, y, z = self._find_best_position_in_truck(item_details, truck_details)
                if x is not None and y is not None and z is not None:
                    weight_utilization = (truck_details["current_weight"] + item_details["weight"]) / truck_details["capacity"]
                    if weight_utilization > max_weight_utilization:
                        max_weight_utilization = weight_utilization
                        best_truck_index = i
        return best_truck_index

    def _find_best_position_in_truck(self, item_details, truck_details):
        l, w, h = item_details["dims"]
        tl, tw, th = truck_details["dims"]
        
        x_step = min(l, 200)
        y_step = min(w, 200)
        z_step = min(h, 200)
        
        best_x, best_y, best_z = None, None, None
        
        x_values = np.arange(0, tl - l + 1, x_step)
        y_values = np.arange(0, tw - w + 1, y_step)
        z_values = np.arange(0, th - h + 1, z_step)
        
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    if self._is_placement_valid((x, y, z), item_details, truck_details):
                        return x, y, z
        return best_x, best_y, best_z

    def _select_truck_type(self, item_details, truck_types):
        for i, truck_type in enumerate(truck_types):
            capacity, tr_l, tr_w, tr_h = truck_type
            if (capacity >= item_details["weight"] and
                tr_l >= item_details["l"] and
                tr_w >= item_details["w"] and
                tr_h >= item_details["h"]):
                return i
        return -1
import numpy as np

class PlacementAlgorithmIT1(PlacementAlgorithm):
    def _find_best_truck(self, item_details, trucks_in_use, truck_types):
        best_truck_index = -1
        min_waste_volume = float('inf')

        for i, truck in enumerate(trucks_in_use):
            truck_type_index = truck["truck_type_index"]
            truck_details = self._get_truck_details(truck, truck_types[truck_type_index])

            if truck_details["remaining_weight_capacity"] >= item_details["weight"]:
                x, y, z = self._find_first_valid_position(item_details, truck_details)
                if x is not None and y is not None and z is not None:
                    remaining_volume = truck_details["remaining_physical_volume"] - item_details["vol"]
                    if remaining_volume < min_waste_volume:
                        min_waste_volume = remaining_volume
                        best_truck_index = i

        return best_truck_index

    def _find_first_valid_position(self, item_details, truck_details):
        l, w, h = item_details["dims"]
        tl, tw, th = truck_details["dims"]

        # Coarser steps for initial search
        x_step = min(l, 500)
        y_step = min(w, 500)
        z_step = min(h, 500)

        x_values = np.arange(0, tl - l + 1, x_step)
        y_values = np.arange(0, tw - w + 1, y_step)
        z_values = np.arange(0, th - h + 1, z_step)

        for x in x_values:
            for y in y_values:
                for z in z_values:
                    if self._is_placement_valid((x, y, z), item_details, truck_details):
                        return x, y, z
        return None, None, None

    def _find_best_position_in_truck(self, item_details, truck_details):
        # Use _find_first_valid_position for faster initial placement
        return self._find_first_valid_position(item_details, truck_details)

import numpy as np

class PlacementAlgorithmIT2(PlacementAlgorithmIT1):
    def _find_best_truck(self, item_details, trucks_in_use, truck_types):
        best_truck_index = -1
        min_waste_volume = float('inf')

        # Sort trucks by remaining weight capacity (ascending)
        trucks_ranked = sorted(enumerate(trucks_in_use), key=lambda x: self._get_truck_details(x[1], truck_types[x[1]["truck_type_index"]])["remaining_weight_capacity"])

        for i, (truck_index, truck) in enumerate(trucks_ranked):
            truck_type_index = truck["truck_type_index"]
            truck_details = self._get_truck_details(truck, truck_types[truck_type_index])

            if truck_details["remaining_weight_capacity"] >= item_details["weight"]:
                x, y, z = self._find_first_valid_position(item_details, truck_details)
                if x is not None and y is not None and z is not None:
                    remaining_volume = truck_details["remaining_physical_volume"] - item_details["vol"]
                    if remaining_volume < min_waste_volume:
                        min_waste_volume = remaining_volume
                        best_truck_index = truck_index
                    break # Place in the first available truck
        return best_truck_index

    def _find_first_valid_position(self, item_details, truck_details):
        l, w, h = item_details["dims"]
        tl, tw, th = truck_details["dims"]

        # Precompute valid ranges for x, y, and z
        x_range = (0, tl - l)
        y_range = (0, tw - w)
        z_range = (0, th - h)

        # Initial coarse placement attempt at origin
        if self._is_placement_valid((0, 0, 0), item_details, truck_details):
            return 0, 0, 0

        # Try placing on top of existing items.
        for placed_item_pos, placed_item_dims in truck_details["placed_items_spatial_info"]:
            ex, ey, ez = placed_item_pos
            el, ew, eh = placed_item_dims
            
            #Try next to existing item on X axis, ensure we stay in bounds.
            x = ex + el
            if x >= x_range[0] and x <= x_range[1] and self._is_placement_valid((x, 0, ez + eh), item_details, truck_details):
                return x, 0, ez + eh

            y = ey + ew
            if y >= y_range[0] and y <= y_range[1] and self._is_placement_valid((0, y, ez + eh), item_details, truck_details):
                return 0, y, ez + eh
        
        return None, None, None

from packing import PackingCONST, Truck, GetData
import warnings
import types
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import copy
import time

packing_problem = PackingCONST()
getData = GetData(10)  # Pass n_truck_types
packing_problem.instance_data = getData.generate_instances(split="test")


def evaluate(self):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packing_module = types.ModuleType("packing_module")
            start_time = time.time()
            algorithm = PlacementAlgorithmIT2()
            fitness = self.greedy(algorithm.place_item)
            print("Fitness:", fitness)
            print("Execution time:", time.time() - start_time)
            return fitness
    except Exception as e:
        print("Error during evaluation:", str(e))
        return None


fitness = evaluate(packing_problem)


def visualize_truck(truck: Truck):
    """Visualizes the contents of a single truck using matplotlib."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw the truck itself as a wireframe
    ax.plot(
        [0, truck.length, truck.length, 0, 0],
        [0, 0, truck.width, truck.width, 0],
        [0, 0, 0, 0, 0],
        color="black",
    )
    ax.plot(
        [0, truck.length, truck.length, 0, 0],
        [0, 0, truck.width, truck.width, 0],
        [truck.height, truck.height, truck.height, truck.height, truck.height],
        color="black",
    )
    for i in range(2):
        ax.plot(
            [0, 0],
            [0, truck.width],
            [i * truck.height, i * truck.height],
            color="black",
        )
        ax.plot(
            [truck.length, truck.length],
            [0, truck.width],
            [i * truck.height, i * truck.height],
            color="black",
        )
        ax.plot(
            [0, truck.length],
            [0, 0],
            [i * truck.height, i * truck.height],
            color="black",
        )
        ax.plot(
            [0, truck.length],
            [truck.width, truck.width],
            [i * truck.height, i * truck.height],
            color="black",
        )

    # Draw each item as a colored box
    colors = ["r", "g", "b", "y", "c", "m"]  # Add more colors if needed
    color_index = 0

    for item in truck.items:
        # Define the vertices of the item
        vertices = [
            [item.x, item.y, item.z],
            [item.x + item.length, item.y, item.z],
            [item.x + item.length, item.y + item.width, item.z],
            [item.x, item.y + item.width, item.z],
            [item.x, item.y, item.z + item.height],
            [item.x + item.length, item.y, item.z + item.height],
            [item.x + item.length, item.y + item.width, item.z + item.height],
            [item.x, item.y + item.width, item.z + item.height],
        ]

        # Define the faces of the item
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]

        # Create a Poly3DCollection for the item and add it to the plot
        poly3d = Poly3DCollection(
            faces,
            facecolors=colors[color_index % len(colors)],
            linewidths=1,
            edgecolors="k",
            alpha=0.5,
        )
        ax.add_collection3d(poly3d)
        color_index += 1

    # Set axis labels and title
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")

    # Set axis limits
    ax.set_xlim([0, truck.length])
    ax.set_ylim([0, truck.width])
    ax.set_zlim([0, truck.height])

    # Add volume and weight ratio information to the plot
    volume_ratio = truck.used_volume / (truck.length * truck.width * truck.height)
    weight_ratio = (
        truck.used_weight / truck.max_weight
    )  # Use max_weight instead of capacity

    plt.title(f"Volume Ratio: {volume_ratio:.2f}, Weight Ratio: {weight_ratio:.2f}")

    def constrain_view(event):
        min_elev, max_elev = 20, 80
        # Get current elevation and azimuth
        current_elev = ax.elev
        current_azim = (
            ax.azim
        )  # if you want to restrict azimuth, you can do so similarly

        # Clamp the elevation to the allowed range
        constrained_elev = np.clip(current_elev, min_elev, max_elev)
        # (Optionally) clamp the azimuth. For example, to keep it between -90° and 90°:
        # constrained_azim = np.clip(current_azim, -90, 90)
        constrained_azim = current_azim  # here we leave azimuth free

        # Update the view if needed
        if constrained_elev != current_elev:
            ax.view_init(elev=constrained_elev, azim=constrained_azim)
            fig.canvas.draw_idle()

    # Connect the constrain_view callback to the mouse button release event.
    fig.canvas.mpl_connect("button_release_event", constrain_view)
    plt.show()


def visualize_plan(plan, items, truck_types):
    """Visualizes a given plan."""
    trucks = []
    # Create trucks based on the plan and truck types
    for placement in plan:
        truck_index = placement["truck_index"]
        if len(trucks) <= truck_index:
            truck_type_index = placement["truck_type_index"]
            capacity, length, width, height = truck_types[truck_type_index]
            truck = Truck(str(truck_index), length, width, height, capacity)
            truck.type_id = truck_type_index
            trucks.append(truck)

    # Populate the trucks with items based on the plan
    for placement in plan:
        truck_index = placement["truck_index"]
        item_index = placement["item_index"]
        x = placement["x"]
        y = placement["y"]
        z = placement["z"]
        item = items[item_index]
        item_copy = copy.deepcopy(item)
        item_copy.x, item_copy.y, item_copy.z = x, y, z
        trucks[truck_index].items.append(item_copy)
        trucks[truck_index].used_volume += (
            item_copy.length * item_copy.width * item_copy.height
        )
        trucks[truck_index].used_weight += item_copy.weight

    # Visualize each truck
    for truck in trucks:
        visualize_truck(truck)


if fitness is not None:
    print(f"Average container height: {fitness[0]}")

    plan_to_visualize, _ = fitness[1][0]
    items, truck_types = packing_problem.instance_data[
        0
    ]  # Get items and truck types from the first instance
    # print(f"Number of trucks used: {len(set([placement['truck_index'] for placement in plan_to_visualize]))}")
    visualize_plan(plan_to_visualize, items, truck_types)
