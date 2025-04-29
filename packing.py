import numpy as np
import types
import sys
import warnings
from typing import List, Tuple
import traceback
from tqdm import tqdm
import line_profiler


class Truck:
    """Represents a truck with its specifications."""

    def __init__(
        self,
        type_id: str,
        length: float,
        width: float,
        height: float,
        max_weight: float,
    ):
        self.type_id = type_id
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.items: List["Item"] = []  # Items loaded into this truck
        self.used_volume = 0.0
        self.used_weight = 0.0


class Item:
    """Represents an item to be transported."""

    def __init__(
        self, item_id: str, length: float, width: float, height: float, weight: float
    ):
        self.item_id = item_id
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.x = 0.0  # Coordinates within the truck
        self.y = 0.0
        self.z = 0.0
        self.placed = False  # Flag to indicate if item is placed

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def bottom_area(self) -> float:
        return self.length * self.width

    def __str__(self):
        return (
            f"Item(id={self.item_id}, l={self.length}, w={self.width}, h={self.height}, "
            f"weight={self.weight}, x={self.x}, y={self.y}, z={self.z})"
        )

    def __repr__(self):
        return self.__str__()


import glob
import json


class GetData:
    def __init__(self, n_instance):
        self.n_instance = n_instance
        self.all_items = []
        # Stores tuples: (maxLoad, length, width, height) for each truck type encountered
        self.all_truck_types_properties = []

    def generate_instances(self):
        """
        Reads data from JSON files, extracts items and truck types,
        and stores them in internal lists for later analysis.
        Returns instance-specific data as before.
        """
        files = glob.glob("data/train/*")
        instance_data = []
        processed_count = 0
        # Process up to n_instance files or all if less than n_instance
        files_to_process = files[: self.n_instance]  # Use the n_instance attribute

        if not files_to_process:
            print("No files found in data/train/. Cannot generate instances.")
            return []  # Return empty list if no files

        for file in files_to_process:
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                # Extract truck types
                truck_type_map = data.get("algorithmBaseParamDto", {}).get(
                    "truckTypeMap", {}
                )
                current_instance_truck_types = []
                for truck_id, truck in truck_type_map.items():
                    # Extract properties, using .get with default 0 in case keys are missing
                    max_load = truck.get("maxLoad", 0)
                    length = truck.get("length", 0)
                    width = truck.get("width", 0)
                    height = truck.get("height", 0)
                    truck_props = (max_load, length, width, height)
                    current_instance_truck_types.append(truck_props)
                    # Add to the aggregated list
                    self.all_truck_types_properties.append(truck_props)

                # Extract items (boxes)
                current_instance_items = []
                boxes = data.get("boxes", [])
                boxes_to_process = boxes

                for i, box in enumerate(boxes_to_process):
                    # Extract properties, using .get with default 0
                    item = Item(
                        str(i),
                        box.get("length", 0),
                        box.get("width", 0),
                        box.get("height", 0),
                        box.get("weight", 0),
                    )
                    current_instance_items.append(item)
                    # Add to the aggregated list
                    self.all_items.append(item)

                instance_data.append(
                    (current_instance_items, current_instance_truck_types)
                )
                processed_count += 1

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue  # Skip to the next file

        print(
            f"Successfully processed {processed_count} files out of {len(files_to_process)} requested."
        )
        print(
            f"Aggregated {len(self.all_items)} items and {len(self.all_truck_types_properties)} truck type properties across files."
        )
        return instance_data

    def _calculate_percentiles_string(self, data_list, property_name):
        """Helper to calculate and format percentile string for a given list."""
        if not data_list:
            return f"  - {property_name}: No data available."

        try:
            percentiles = np.percentile(data_list, [0, 25, 50, 75, 100])
            p0, p25, p50, p75, p100 = percentiles

            # Determine format: integer or float
            # Check if all values in the *original* list are integers and if the calculated percentiles are also close to integers
            # This helps decide whether to show .00 or not.
            is_integer_data = all(isinstance(x, int) for x in data_list)
            is_integer_percentiles = all(abs(p - round(p)) < 1e-9 for p in percentiles)

            fmt = "{:.0f}" if is_integer_data and is_integer_percentiles else "{:.2f}"

            return (
                f"  - {property_name}: "
                f"Min={fmt.format(p0)}, "
                f"25th={fmt.format(p25)}, "
                f"Median={fmt.format(p50)}, "
                f"75th={fmt.format(p75)}, "
                f"Max={fmt.format(p100)}. "
            )
        except Exception as e:
            return f"  - {property_name}: Error calculating percentiles - {e}"

    def generate_percentile_prompt(self):
        """
        Generates a text prompt summarizing the percentile distribution
        of item and truck properties based on aggregated data.
        Requires generate_instances() to have been called first.
        """

        if not self.all_items and not self.all_truck_types_properties:
            return "No data loaded. Please run generate_instances() first to populate data."

        prompt_parts = ["Data Distribution Summary:"]

        # --- Item Properties ---
        prompt_parts.append("\nItem Properties:")
        if not self.all_items:
            prompt_parts.append("  No item data available.")
        else:
            # Extract all values for each property
            item_lengths = [item.length for item in self.all_items]
            item_widths = [item.width for item in self.all_items]
            item_heights = [item.height for item in self.all_items]
            item_weights = [item.weight for item in self.all_items]

            # Calculate and format percentile strings using the helper
            prompt_parts.append(
                self._calculate_percentiles_string(item_lengths, "Length")
            )
            prompt_parts.append(
                self._calculate_percentiles_string(item_widths, "Width")
            )
            prompt_parts.append(
                self._calculate_percentiles_string(item_heights, "Height")
            )
            # Assuming weight might have units like kg
            prompt_parts.append(
                self._calculate_percentiles_string(item_weights, "Weight")
            )

        # --- Truck Properties ---
        prompt_parts.append("\nTruck Properties:")
        if not self.all_truck_types_properties:
            prompt_parts.append("  No truck type data available.")
        else:
            # self.all_truck_types_properties contains tuples (maxLoad, length, width, height)
            truck_maxloads = [t[0] for t in self.all_truck_types_properties]
            truck_lengths = [t[1] for t in self.all_truck_types_properties]
            truck_widths = [t[2] for t in self.all_truck_types_properties]
            truck_heights = [t[3] for t in self.all_truck_types_properties]

            # Calculate and format percentile strings using the helper
            # Assuming maxLoad might have units like kg
            prompt_parts.append(
                self._calculate_percentiles_string(truck_maxloads, "Max Load")
            )
            prompt_parts.append(
                self._calculate_percentiles_string(truck_lengths, "Length")
            )
            prompt_parts.append(
                self._calculate_percentiles_string(truck_widths, "Width")
            )
            prompt_parts.append(
                self._calculate_percentiles_string(truck_heights, "Height")
            )
        return "\n".join(prompt_parts)


class GetPrompts:
    def __init__(self, data_distribution=""):
        self.prompt_task = (
            "Given a list of items with their dimensions and weights, and a list of truck types with their dimensions and weight capacities, \
you need to place all items into the trucks without overlapping, minimizing the number of trucks used. \
The task can be solved step-by-step. In each step, you select an item, a truck, and a position within the truck for the item. \
Design a novel algorithm to select the next item, truck and its placement to ensure the constraints are satisfied."
            + """
        Constraints:
        1. **No Overlap:** Items cannot overlap in the truck.  The (x, y, z) coordinates represent the bottom-left-front corner of an item, and you must ensure that no two items occupy the same space.
        2. **Truck Dimensions:** Items must fit completely within the chosen truck's length, width, and height.
        3. **Weight Capacity:** The total weight of items in a truck cannot exceed the truck's maximum weight capacity.
        4. **Support:**  An item must be supported by either the floor of the truck or by other items.  An item is considered supported if at least 80% of its bottom surface area is directly above the top surface of other items or the truck floor. The z-coordinate represents the height.
        5. **Complete Placement:** All items must be placed in a truck.
        6. **Minimize Trucks:** The primary goal is to minimize the number of trucks used, while still satisfying all other constraints.  A lower number of trucks used results in a better solution.
        7. **Return Values:** The `place_item` function must return valid integer values for `truck_index`, `item_index`, `x`, `y`, `z`, and `truck_type_index`.  Invalid return values will result in an immediate failure.
        8. **New Truck Selection:**  If `truck_index` is -1, indicating a new truck is needed, `truck_type_index` must be a valid index within the `truck_types` list.
        9. **Existing Truck Selection:** If `truck_index` is not -1, it must be a valid index within the `trucks_in_use` list.
        10. **Item Selection:** The `item_index` must be a valid index within the `unplaced_items` list.
        11. **Coordinate System**: The coordinate system's origin (0,0,0) is at the bottom-left-front corner of the truck.  'x' increases along the length, 'y' along the width, and 'z' along the height.
        """
            + "\n"
            + data_distribution
        )
        self.prompt_func_name = "place_item"
        self.prompt_func_inputs = ["unplaced_items", "trucks_in_use", "truck_types"]
        self.prompt_func_outputs = [
            "truck_index",
            "item_index",
            "x",
            "y",
            "z",
            "truck_type_index",
        ]  # Added truck_type_index
        self.prompt_inout_inf = """
        'unplaced_items' is a list of dictionaries, each representing an unplaced item.  Each dictionary has the following keys:
            - 'item_id': str, a unique identifier for the item.
            - 'length': float, the length of the item.
            - 'width': float, the width of the item.
            - 'height': float, the height of the item.
            - 'weight': float, the weight of the item.

        'trucks_in_use' is a list of dictionaries, each representing a truck that is currently in use.  Each dictionary has the following keys:
            - 'truck_type_index': int, the index of the truck type in the 'truck_types' list.
            - 'occupied_volumes': list of tuples.  Each tuple represents an item placed in the truck and contains:
                - item_id (str)
                - x (float):  x-coordinate of the item's bottom-left-front corner.
                - y (float):  y-coordinate of the item's bottom-left-front corner.
                - z (float):  z-coordinate of the item's bottom-left-front corner.
                - length (float)
                - width (float)
                - height (float)
            - 'current_weight': float, the total weight of all items currently in the truck.

        'truck_types' is a list of tuples.  Each tuple represents a truck type and contains:
            - capacity (float): The maximum weight capacity of the truck.
            - length (float): The length of the truck.
            - width (float): The width of the truck.
            - height (float): The height of the truck.
            
        'truck_index' is an integer representing the index of the truck in the `trucks_in_use` list to place the item in.  If -1, a new truck should be used.

        'item_index' is an integer representing the index of the item in the `unplaced_items` list to be placed.
        'x', 'y', and 'z' are floats representing the coordinates of the bottom-left-front corner of the item within the truck.
        'truck_type_index' is an integer representing the index of the truck type in the `truck_types` list to use if a new truck is needed (i.e., if `truck_index` is -1).
        """
        self.prompt_other_inf = "Use NumPy arrays where appropriate. Ensure that all return values are defined before returning them. The algorithm will be evaluated on instances with hundreds of items and a few truck types, which means that the 'occupied_volumes' in `trucks_in_use` will contain large numbers of entries. Consider implementing an efficient search for available space, weight load calculations, and placement validation to handle such large inputs efficiently."

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


class PackingCONST:
    def __init__(self):
        self.n_instance = 5
        self.running_time = 10

        getData = GetData(self.n_instance)  # Pass n_truck_types
        self.instance_data = getData.generate_instances()
        self.prompts = GetPrompts(getData.generate_percentile_prompt())

    def _check_overlap(self, item1: Item, item2: Item) -> bool:
        """Checks if two items overlap in 3D space."""
        return not (
            item1.x + item1.length <= item2.x
            or item2.x + item2.length <= item1.x
            or item1.y + item1.width <= item2.y
            or item2.y + item2.width <= item1.y
            or item1.z + item1.height <= item2.z
            or item2.z + item2.height <= item1.z
        )

    def _check_truck_constraints(self, truck: Truck, item: Item, x, y, z) -> bool:
        """Checks if adding an item to a truck violates weight or size constraints."""

        if truck.used_weight + item.weight > truck.max_weight:
            return False  # Weight constraint violated
        if (
            x + item.length > truck.length
            or y + item.width > truck.width
            or z + item.height > truck.height
        ):
            return False
        return True

    def calculate_loading_efficiency(
        self, trucks: List[Truck], item_total_volume, item_total_weight
    ):
        """Calculates the loading efficiency score."""
        if not trucks:
            return 0.0

        loading_rates = []
        for truck in trucks:
            volume_ratio = truck.used_volume / (
                truck.length * truck.width * truck.height
            )
            weight_ratio = truck.used_weight / truck.max_weight
            loading_rates.append(max(volume_ratio, weight_ratio))

        if len(trucks) == 0:  # Avoid division by zero
            return 1.0  # Or some other default value indicating worst-case scenario

        return 1 - (sum(loading_rates) / len(trucks))

    def calculate_support(self, truck: Truck, item: Item) -> float:
        """Calculates the support area for an item in the truck."""
        if item.z == 0:
            return item.bottom_area  # Fully supported by the truck floor

        supported_area = 0.0
        for other_item in truck.items:
            if other_item == item:
                continue  # Don't compare an item to itself
            # Crucial Change: Only consider support if the other item is *directly* below
            if (
                abs(other_item.z + other_item.height - item.z) < 1e-9
            ):  # Use a small epsilon for comparison
                # Check for overlap in the x-y plane (top view)
                overlap_x1 = max(item.x, other_item.x)
                overlap_y1 = max(item.y, other_item.y)
                overlap_x2 = min(item.x + item.length, other_item.x + other_item.length)
                overlap_y2 = min(item.y + item.width, other_item.y + other_item.width)

                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    supported_area += (overlap_x2 - overlap_x1) * (
                        overlap_y2 - overlap_y1
                    )

        return supported_area

    def check_support_constraint(self, truck: Truck, item: Item) -> bool:
        """Checks if an item meets the 80% support constraint."""
        supported_area = self.calculate_support(truck, item)
        return supported_area >= 0.8 * item.bottom_area

    @line_profiler.profile
    def greedy(self, place_item):
        all_plans = []  # List to store the plans for each instance data
        for items, truck_types in self.instance_data:
            trucks_in_use = []
            unplaced_items_indices = list(range(len(items)))
            item_total_volume = sum(
                [item.length * item.width * item.height for item in items]
            )
            item_total_weight = sum([item.weight for item in items])
            plan = []  # Plan for this specific instance
            pbar = tqdm(total=len(unplaced_items_indices))
            while unplaced_items_indices:
                try:
                    unplaced_items_data = []
                    for i in unplaced_items_indices:
                        item = items[i]
                        unplaced_items_data.append(
                            {
                                "item_id": item.item_id,
                                "length": item.length,
                                "width": item.width,
                                "height": item.height,
                                "weight": item.weight,
                            }
                        )

                    trucks_in_use_data = []
                    for truck in trucks_in_use:
                        occupied_volumes_data = []
                        for item in truck.items:
                            occupied_volumes_data.append(
                                (
                                    item.item_id,
                                    item.x,
                                    item.y,
                                    item.z,
                                    item.length,
                                    item.width,
                                    item.height,
                                )
                            )
                        trucks_in_use_data.append(
                            {
                                "truck_type_index": truck.type_id,  # Use type_id for consistency
                                "occupied_volumes": occupied_volumes_data,
                                "current_weight": truck.used_weight,
                            }
                        )

                    truck_index, item_index, x, y, z, truck_type_index = place_item(
                        unplaced_items_data, trucks_in_use_data, truck_types
                    )
                    # --- Input Validation ---
                    if not isinstance(truck_index, int):
                        return None
                    if not isinstance(item_index, int):
                        return None
                    if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
                        return None
                    if not isinstance(truck_type_index, int):
                        return None

                    if not 0 <= item_index < len(unplaced_items_indices):
                        return None

                    actual_item_index = unplaced_items_indices[item_index]
                    selected_item = items[actual_item_index]

                    # Handle new truck creation
                    if truck_index == -1:
                        if not 0 <= truck_type_index < len(truck_types):
                            return None
                        capacity, length, width, height = truck_types[truck_type_index]
                        new_truck = Truck(
                            str(len(trucks_in_use)), length, width, height, capacity
                        )
                        new_truck.type_id = truck_type_index
                        trucks_in_use.append(new_truck)
                        truck_index = len(trucks_in_use) - 1

                    if not 0 <= truck_index < len(trucks_in_use):
                        return None

                    selected_truck = trucks_in_use[truck_index]

                    # Check constraints
                    if not self._check_truck_constraints(
                        selected_truck, selected_item, x, y, z
                    ):
                        return None

                    # Check overlap
                    overlap = False
                    selected_item.x, selected_item.y, selected_item.z = x, y, z
                    for other_item in selected_truck.items:
                        if self._check_overlap(selected_item, other_item):
                            overlap = True
                            break
                    if overlap:
                        return None

                    # Check support constraint
                    if not self.check_support_constraint(selected_truck, selected_item):
                        return None  # Or handle the unsupported item (e.g., try a different position/truck)

                    # Update truck and item status
                    selected_truck.items.append(selected_item)
                    selected_truck.used_volume += (
                        selected_item.length
                        * selected_item.width
                        * selected_item.height
                    )
                    selected_truck.used_weight += selected_item.weight
                    unplaced_items_indices.pop(item_index)

                    # Add the placement to the plan
                    plan.append(
                        {
                            "truck_index": truck_index,
                            "item_index": actual_item_index,
                            "x": x,
                            "y": y,
                            "z": z,
                            "truck_type_index": truck_type_index,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    print("An error occurred:", str(e))
                    print(traceback.format_exc())
                    return None

            # Calculate and append the efficiency score
            efficiency_score = self.calculate_loading_efficiency(
                trucks_in_use, item_total_volume, item_total_weight
            )
            all_plans.append((plan, efficiency_score))  # Store plan and its efficiency

        # Calculate average efficiency across all instance data
        total_efficiency_scores = [score for _, score in all_plans]
        average_efficiency = np.average(total_efficiency_scores)
        return average_efficiency, all_plans

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                packing_module = types.ModuleType("packing_module")
                exec(code_string, packing_module.__dict__)
                sys.modules[packing_module.__name__] = packing_module
                fitness = self.greedy(packing_module.place_item)
                if fitness is not None:
                    fitness, all_plans = fitness
                return fitness
        except Exception as e:
            print("Error during evaluation:", str(e))
            return None
