import numpy as np
from typing import List, Dict, Tuple
import traceback
import glob
from eoh.invoker import AlgorithmInvoker
import re


class Container:
    """Represents a single, standard-sized stock container."""

    def __init__(self, container_id: str, length: float, width: float, height: float):
        self.container_id = container_id
        self.length = length
        self.width = width
        self.height = height
        self.items: List["Item"] = []
        self.used_volume = 0.0
        # Tracks if container is dedicated to set 1 (B₁) or 2 (B₂)
        self.separation_set: int = None


class Item:
    """
    Represents an individual item instance that has been packed. The item's
    dimensions (length, width, height) are set based on the chosen orientation
    from its original dimensions.
    """

    def __init__(
        self,
        item_type_id: str,
        original_length: float,
        original_width: float,
        original_height: float,
        orientation: int = 0,
        separation_set: int = None,
    ):
        self.item_type_id = item_type_id
        # Identifies if the item belongs to an incompatible set (1 or 2)
        self.separation_set = separation_set

        # Store original dimensions for reference
        self.original_length = original_length
        self.original_width = original_width
        self.original_height = original_height

        # Set current dimensions based on orientation
        self._set_orientation(orientation)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.placed = True
        self.orientation = orientation

    def _set_orientation(self, orientation: int):
        """Sets the item's length, width, and height based on the orientation code."""
        dims = (self.original_length, self.original_width, self.original_height)
        if orientation == 0:  # (L, W, H)
            self.length, self.width, self.height = dims[0], dims[1], dims[2]
        elif orientation == 1:  # (L, H, W)
            self.length, self.width, self.height = dims[0], dims[2], dims[1]
        elif orientation == 2:  # (W, L, H)
            self.length, self.width, self.height = dims[1], dims[0], dims[2]
        elif orientation == 3:  # (W, H, L)
            self.length, self.width, self.height = dims[1], dims[2], dims[0]
        elif orientation == 4:  # (H, L, W)
            self.length, self.width, self.height = dims[2], dims[0], dims[1]
        elif orientation == 5:  # (H, W, L)
            self.length, self.width, self.height = dims[2], dims[1], dims[0]
        else:
            raise ValueError(f"Invalid orientation code: {orientation}. Must be 0-5.")

    @property
    def volume(self) -> float:
        # Volume is invariant to orientation
        return self.original_length * self.original_width * self.original_height


class GetData:
    """
    Data handler for container loading problems, where each instance file defines
    its own unique container and a set of items. It correctly sorts instances
    numerically (e.g., Iva1, Iva2, ..., Iva10).
    """

    def __init__(self, n_instance):
        self.n_instance = n_instance
        # Store all unique item types found across all instances for summary stats
        self.all_item_types = []
        # Store all container types found across all instances for summary stats
        self.all_container_types = []

    def generate_instances(self, path_pattern="data/ssscsp/INSTANCES/Iva*.txt"):
        """
        Reads instance files, where each file specifies its own container and item list.
        Files are sorted in natural numerical order based on their name.

        Args:
            path_pattern (str): The glob pattern to find instance files.

        Returns:
            list: A list of tuples, where each tuple is (list_of_items, container_tuple).
                  Returns an empty list if no files are processed.
        """
        files = glob.glob(path_pattern)

        # --- Custom sorting key to sort files numerically ---
        def natural_sort_key(file_path):
            # Attempt to find one or more digits (\d+) in the filename
            match = re.search(r"(\d+)", file_path)
            if match:
                # If a number is found, return it as an integer for proper sorting
                return int(match.group(1))
            # If no number is found, return a large number to place it at the end
            return float("inf")

        # Sort files using the natural sort key
        files.sort(key=natural_sort_key)

        files_to_process = files[: self.n_instance]

        if not files_to_process:
            print(f"No files found matching the pattern: {path_pattern}")
            return []

        # Clear previous data before processing new files
        self.all_item_types = []
        self.all_container_types = []
        instance_data = []

        for file_path in files_to_process:
            try:
                with open(file_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                # First line contains the container dimensions for this specific instance
                container_dims = [int(d) for d in lines[0].split()]
                current_container_type = tuple(container_dims)
                self.all_container_types.append(current_container_type)

                # Lines from the 3rd onward contain the items
                item_lines = lines[2:]

                current_instance_item_types = []
                for i, line in enumerate(item_lines):
                    # Each item line: length width height quantity
                    parts = [int(p) for p in line.split()]
                    if len(parts) != 4:
                        continue

                    # Define separation set: 1 for the first type, 2 for the second
                    separation_set = None
                    if i == 0:
                        separation_set = 1  # Belongs to B₁
                    elif i == 1:
                        separation_set = 2  # Belongs to B₂

                    item = {
                        "item_id": str(i),
                        "length": parts[0],
                        "width": parts[1],
                        "height": parts[2],
                        "quantity": parts[3],
                        "separation_set": separation_set,
                    }
                    current_instance_item_types.append(item)
                    self.all_item_types.append(item)

                # Each instance data tuple contains its own items and its own container
                instance_data.append(
                    (current_instance_item_types, current_container_type)
                )

            except (IOError, IndexError, ValueError) as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        print(f"Successfully processed {len(instance_data)} instances.")
        return instance_data

    def _calculate_percentiles_string(self, data_list, property_name):
        """Calculates and formats percentile data for a given list of numbers."""
        if not data_list:
            return f"  - {property_name}: No data available."

        p = np.percentile(data_list, [0, 25, 50, 75, 100])
        fmt = "{:.0f}" if all(val == int(val) for val in p) else "{:.2f}"
        return f"  - {property_name}: Min={fmt.format(p[0])}, 25th={fmt.format(p[1])}, Median={fmt.format(p[2])}, 75th={fmt.format(p[3])}, Max={fmt.format(p[4])}"

    def generate_percentile_prompt(self):
        """Generates a summary string of the data distribution for items and containers."""
        if not self.all_item_types:
            return "No data loaded. Please run 'generate_instances' first."

        prompt_parts = ["Data Distribution Summary:"]

        # --- Container Percentiles ---
        prompt_parts.append("\nContainer Properties (based on all loaded instances):")
        cont_lengths = [c[0] for c in self.all_container_types]
        cont_widths = [c[1] for c in self.all_container_types]
        cont_heights = [c[2] for c in self.all_container_types]
        prompt_parts.append(self._calculate_percentiles_string(cont_lengths, "Length"))
        prompt_parts.append(self._calculate_percentiles_string(cont_widths, "Width"))
        prompt_parts.append(self._calculate_percentiles_string(cont_heights, "Height"))

        # --- Item Percentiles ---
        prompt_parts.append(
            "\nItem Properties (based on all items in loaded instances):"
        )
        item_lengths = [it["length"] for it in self.all_item_types]
        item_widths = [it["width"] for it in self.all_item_types]
        item_heights = [it["height"] for it in self.all_item_types]
        item_quantities = [it["quantity"] for it in self.all_item_types]

        prompt_parts.append(self._calculate_percentiles_string(item_lengths, "Length"))
        prompt_parts.append(self._calculate_percentiles_string(item_widths, "Width"))
        prompt_parts.append(self._calculate_percentiles_string(item_heights, "Height"))
        prompt_parts.append(
            self._calculate_percentiles_string(item_quantities, "Quantity")
        )

        return "\n".join(prompt_parts)


class GetPrompts:
    """Generates the problem description and API for the SSSCSP with item rotation."""

    def __init__(self, data_distribution=""):
        self.base_class_code = """
import numpy as np

class BaseAlgorithm:
    def __init__(self, epsilon=1e-6):
        # Initializes the base algorithm with common parameters.
        if not epsilon > 0:
            raise ValueError("epsilon should be a small positive value.")
        self.epsilon = epsilon

    def _check_overlap_3d(self, item1_pos, item1_dims, item2_pos, item2_dims):
        x1, y1, z1 = item1_pos
        l1, w1, h1 = item1_dims
        x2, y2, z2 = item2_pos
        l2, w2, h2 = item2_dims
        return (x1 < x2 + l2 and x1 + l1 > x2 and
                y1 < y2 + w2 and y1 + w1 > y2 and
                z1 < z2 + h2 and z1 + h1 > z2)

    def _get_orientations(self, item_type):
        L, W, H = item_type['length'], item_type['width'], item_type['height']
        return [
            (0, (L, W, H)), (1, (L, H, W)), (2, (W, L, H)),
            (3, (W, H, L)), (4, (H, L, W)), (5, (H, W, L))
        ]

    def _is_within_container_bounds(self, item_pos, item_dims, container_dims):
        px, py, pz = item_pos
        pl, pw, ph = item_dims
        cl, cw, ch = container_dims
        return (px >= 0.0 - self.epsilon and px + pl <= cl + self.epsilon and
                py >= 0.0 - self.epsilon and py + pw <= cw + self.epsilon and
                pz >= 0.0 - self.epsilon and pz + ph <= ch + self.epsilon)

    def _is_valid_placement(self, item_to_place_pos, item_to_place_dims, container_dims, occupied_volumes):
        '''
        Checks if placing an item at a given position is valid.

        Args:
            item_to_place_pos (tuple): The (x, y, z) position of the new item.
            item_to_place_dims (tuple): The (length, width, height) of the new item.
            container_dims (tuple): The (length, width, height) of the container.
            occupied_volumes (list): A list of dictionaries, where each dictionary
                                     represents an item already in the container.

        Returns:
            bool: True if the placement is valid, False otherwise.
        '''
        # Check if the item is within the container's boundaries.
        if not self._is_within_container_bounds(item_to_place_pos, item_to_place_dims, container_dims):
            return False

        # Check for overlaps with already placed items.
        for placed_item in occupied_volumes:
            placed_item_pos = (placed_item['x'], placed_item['y'], placed_item['z'])
            placed_item_dims = (placed_item['length'], placed_item['width'], placed_item['height'])
            if self._check_overlap_3d(item_to_place_pos, item_to_place_dims, placed_item_pos, placed_item_dims):
                return False
        
        # If no boundary violations or overlaps are found, the placement is valid.
        return True
"""
        self.prompt_task = (
            "You are to solve the **Single-Stock-Size Cutting Stock Problem (SSSCSP)**. "
            "You are given a list of item **types** (each with dimensions and a **quantity**) and the dimensions of a single standard container. "
            "Your goal is to pack all items into the minimum number of these identical containers."
            "\nDesign a novel algorithm that, at each step, selects an available item type, **chooses one of 6 possible orientations**, and finds a valid position for it in a container."
            + """
        **Primary Objective: Minimize the total number of stock containers used.**

        Constraints:
        1. **Single Container Type:** All containers are identical. No weight or support constraints apply.
        2. **Complete Placement:** All items of all types must be packed.
        3. **Item Orientation:** Items can be rotated. There are 6 possible orientations.
        4. **No Overlap:** Items cannot overlap.
        5. **Boundaries:** Items must be placed fully inside the container.
        6. **Immutable Inputs:** Your function must not modify its input arguments (unplaced_items, trucks_in_use). Treat them as read-only. The calling environment manages state.
        7. **Separation Constraint:** The first item type (`unplaced_items[0]`) and the second item type (`unplaced_items[1]`) are incompatible and cannot be placed in the same container. An item's incompatibility set (1 or 2) is indicated by the `separation_set` key.
        """
            + "\n"
            + data_distribution
        )
        self.prompt_func_name = "place_item"
        self.prompt_func_inputs = ["unplaced_items", "trucks_in_use", "truck_type"]
        self.prompt_func_outputs = [
            "truck_index",
            "item_index",
            "x",
            "y",
            "z",
            "orientation",
        ]
        self.prompt_inout_inf = """
        `unplaced_items`: A list of dictionaries, each representing an **item type**.
            - 'item_id', 'length', 'width', 'height' (these are the base dimensions)
            - 'quantity': int, the number of items of this type still needing to be packed.
            - 'separation_set': An integer (1 or 2) or None. If not None, indicates the item belongs to an incompatible set.

        `trucks_in_use`: A list of dictionaries, where each dictionary represents a container currently in use.
            - `'separation_set'`: An integer (1 or 2) or None. Indicates if this container is already dedicated to incompatible set 1 or 2.
            - `'occupied_volumes'`: A list of **dictionaries**. Each dictionary represents a placed item and has the following keys:
                - `'item_type_id'` (str): The unique identifier for the *type* of the placed item.
                - `'x'` (float): The starting position along the container's length (x-axis).
                - `'y'` (float): The starting position along the container's width (y-axis).
                - `'z'` (float): The starting position along the container's height (z-axis).
                - `'length'` (float): The dimension of the item along the x-axis **in its placed orientation**.
                - `'width'` (float): The dimension of the item along the y-axis **in its placed orientation**.
                - `'height'` (float): The dimension of the item along the z-axis **in its placed orientation**.
        `truck_type`: A **single tuple** for the standard container dimensions: `(length, width, height)`.
            
        **Return Values:**
        `truck_index`: Index of the container in `trucks_in_use`. Use -1 for a new container.
        `item_index`: Index of the item **type** in `unplaced_items` to place.
        `x`, `y`, `z`: Position for the new item. If a new container is requested (i.e., truck_index is -1), this must be the position for the item within that new container.
        `orientation`: An integer from 0 to 5, specifying the item's orientation. Let the original dimensions from `unplaced_items` be (L, W, H). The placed dimensions (length, width, height) will be:
            - 0: (L, W, H)
            - 1: (L, H, W)
            - 2: (W, L, H)
            - 3: (W, H, L)
            - 4: (H, L, W)
            - 5: (H, W, L)
        """
        self.prompt_other_inf = "The solution must not contain any comments or print statements."

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

    def get_base_class_code(self):
        return self.base_class_code


class PackingCONST:
    """Evaluator for the SSSCSP with item rotation."""

    def __init__(self):
        self.n_instance = 20
        getData = GetData(self.n_instance)
        self.instance_data = getData.generate_instances()
        self.prompts = GetPrompts(getData.generate_percentile_prompt())
        self.base_class_code = self.prompts.get_base_class_code()

    def _check_overlap(self, item1: Item, item2: Item) -> bool:
        return not (
            item1.x + item1.length <= item2.x
            or item2.x + item2.length <= item1.x
            or item1.y + item1.width <= item2.y
            or item2.y + item2.width <= item1.y
            or item1.z + item1.height <= item2.z
            or item2.z + item2.height <= item1.z
        )

    def _check_size_constraint(self, container: Container, item: Item, x, y, z) -> bool:
        return (
            x >= 0
            and y >= 0
            and z >= 0
            and x + item.length <= container.length
            and y + item.width <= container.width
            and z + item.height <= container.height
        )

    def solve_instance(
        self, place_item_algo, item_types_initial: List[Dict], container_type: Tuple[float, float, float]
    ):
        """Solves one SSSCSP instance, tracking remaining item quantities."""
        item_types = [it.copy() for it in item_types_initial]
        containers_in_use: List[Container] = []

        while any(it["quantity"] > 0 for it in item_types):
            containers_in_use_data = []
            for cont in containers_in_use:
                occupied_vols = [
                    {
                        "item_type_id": it.item_type_id,
                        "x": it.x,
                        "y": it.y,
                        "z": it.z,
                        "length": it.length,
                        "width": it.width,
                        "height": it.height,
                    }
                    for it in cont.items
                ]
                containers_in_use_data.append(
                    {
                        "occupied_volumes": occupied_vols,
                        "separation_set": cont.separation_set,
                    }
                )

            try:
                result = place_item_algo(
                    item_types, containers_in_use_data, container_type
                )
                if not isinstance(result, (list, tuple)) or len(result) != 6:
                    return (
                        -1,
                        "Algorithm must return a tuple of 6 values: (truck_index, item_index, x, y, z, orientation).",
                    )
                cont_idx, item_type_idx, x, y, z, orientation = result
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    last_frame = tb[-1]
                    return -1, f"Algorithm crashed at: '{last_frame.name}' - {e}"
                else:
                    return -1, f"Algorithm crashed: {e}"

            # --- Validation ---
            if not (
                isinstance(cont_idx, int)
                and isinstance(item_type_idx, int)
                and isinstance(orientation, int)
            ):
                return -1, "Invalid index or orientation types (must be integers)."
            if not (0 <= item_type_idx < len(item_types)):
                return -1, f"Invalid `item_index`: {item_type_idx}."
            if not (0 <= orientation <= 5):
                return (
                    -1,
                    f"Invalid `orientation`: {orientation}. Must be an integer between 0 and 5.",
                )
            if not (item_types[item_type_idx]["quantity"] > 0):
                return (
                    -1,
                    f"Invalid choice: No items with quantity > 0 for type {item_type_idx}.",
                )

            final_cont_idx = cont_idx
            if cont_idx == -1:
                new_cont = Container(str(len(containers_in_use)), *container_type)
                containers_in_use.append(new_cont)
                final_cont_idx = len(containers_in_use) - 1

            if not (0 <= final_cont_idx < len(containers_in_use)):
                return -1, f"Invalid `truck_index` {cont_idx}."

            selected_container = containers_in_use[final_cont_idx]
            selected_item_type = item_types[item_type_idx]
            
            # --- Separation Constraint Validation ---
            item_set = selected_item_type.get("separation_set")
            container_set = selected_container.separation_set
            if (item_set == 1 and container_set == 2) or \
               (item_set == 2 and container_set == 1):
                return -1, "Separation constraint violated: item types 1 and 2 cannot be in the same container."


            try:
                newly_placed_item = Item(
                    item_type_id=selected_item_type["item_id"],
                    original_length=selected_item_type["length"],
                    original_width=selected_item_type["width"],
                    original_height=selected_item_type["height"],
                    orientation=orientation,
                    separation_set=item_set
                )
            except ValueError as e:
                return -1, f"Error creating item with orientation: {e}"

            newly_placed_item.x, newly_placed_item.y, newly_placed_item.z = x, y, z

            # Check geometric constraints
            if not self._check_size_constraint(
                selected_container, newly_placed_item, x, y, z
            ):
                return -1, "Size constraint violated."
            if any(
                self._check_overlap(newly_placed_item, other)
                for other in selected_container.items
            ):
                return -1, "Overlap constraint violated."

            # Place item and update state
            if container_set is None and item_set is not None:
                selected_container.separation_set = item_set

            selected_container.items.append(newly_placed_item)
            selected_container.used_volume += newly_placed_item.volume
            item_types[item_type_idx]["quantity"] -= 1

        return len(containers_in_use), "Success"

    def evaluate(self, invoker: AlgorithmInvoker):
        """Evaluates the algorithm. Fitness is the average number of containers used (lower is better)."""
        total_containers_used = 0
        successful_instances = 0
        try:
            place_item_method = invoker.get_methods()
            for item_types, container_type in self.instance_data:
                container_count, status = self.solve_instance(
                    place_item_method, item_types, container_type
                )
                if status == "Success":
                    total_containers_used += container_count
                    successful_instances += 1
                else:
                    return None, f"Evaluation failed on an instance: {status}"

            if successful_instances == 0:
                return None, "All instances failed."

            average_containers = total_containers_used / successful_instances
            return average_containers, "Evaluation successful"

        except Exception as e:
            print(f"Error during evaluation: {e}\n{traceback.format_exc()}")
            return None, f"An error occurred during evaluation: {e}"


if __name__ == "__main__":
    print(GetData(10).generate_instances()[0])