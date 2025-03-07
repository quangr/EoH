import itertools
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class Truck:
    """Represents a truck with its specifications."""

    def __init__(self, type_id: str, length: float, width: float, height: float, max_weight: float):
        self.type_id = type_id
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.items: List['Item'] = []  # Items loaded into this truck
        self.used_volume = 0.0
        self.used_weight = 0.0


class Item:
    """Represents an item to be transported."""

    def __init__(self, item_id: str, length: float, width: float, height: float, weight: float):
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
        return (f"Item(id={self.item_id}, l={self.length}, w={self.width}, h={self.height}, "
                f"weight={self.weight}, x={self.x}, y={self.y}, z={self.z})")

    def __repr__(self):
        return self.__str__()


class Simulation:
    """Handles the core simulation logic."""

    def __init__(self):
        pass

    def check_overlap(self, item1: Item, item2: Item) -> bool:
        """Checks if two items overlap in 3D space."""
        if not (isinstance(item1, Item) and isinstance(item2, Item)):
            return False
        return not (
                item1.x + item1.length <= item2.x or
                item2.x + item2.length <= item1.x or
                item1.y + item1.width <= item2.y or
                item2.y + item2.width <= item1.y or
                item1.z + item1.height <= item2.z or
                item2.z + item2.height <= item1.z
        )

    def check_truck_constraints(self, truck: Truck, item: Item) -> bool:
        """Checks if adding an item to a truck violates weight or size constraints."""

        if truck.used_weight + item.weight > truck.max_weight:
            return False  # Weight constraint violated
        if (item.x + item.length > truck.length or
                item.y + item.width > truck.width or
                item.z + item.height > truck.height):
            return False
        return True

    def calculate_support(self, truck: Truck, item: Item) -> float:
        """Calculates the support area for an item in the truck."""
        if item.z == 0:
            return item.bottom_area  # Fully supported by the truck floor

        supported_area = 0.0
        for other_item in truck.items:
            if other_item == item:
                continue  # Don't compare an item to itself
            # Crucial Change: Only consider support if the other item is *directly* below
            if abs(other_item.z + other_item.height - item.z) < 1e-9:  # Use a small epsilon for comparison
                # Check for overlap in the x-y plane (top view)
                overlap_x1 = max(item.x, other_item.x)
                overlap_y1 = max(item.y, other_item.y)
                overlap_x2 = min(item.x + item.length, other_item.x + other_item.length)
                overlap_y2 = min(item.y + item.width, other_item.y + other_item.width)

                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    supported_area += (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)

        return supported_area

    def check_support_constraint(self, truck: Truck, item: Item) -> bool:
        """Checks if an item meets the 80% support constraint."""
        supported_area = self.calculate_support(truck, item)
        return supported_area >= 0.8 * item.bottom_area


class Solver:
    """Finds the optimal solution for the bin packing problem."""

    def __init__(self):
        self.simulation = Simulation()

    def find_placement_position(self, truck: Truck, item: Item) -> Tuple[bool, float, float, float]:
        """
        Finds a feasible placement position for the item in the truck.
        """
        # Iterate through possible z-levels first (bottom of truck, and top of existing items)
        possible_z_levels = [0.0]  # Always start with the bottom of the truck
        for other_item in truck.items:
            possible_z_levels.append(other_item.z + other_item.height)
        possible_z_levels = sorted(list(set(possible_z_levels)))  # Remove duplicates and sort

        for z in possible_z_levels:
            item.z = z
            # If it's not resting directly on another item or the floor, no need to check other positions at this level
            if not any(abs(z - (other_item.z + other_item.height)) < 1e-9 for other_item in truck.items) and z != 0.0:
                continue

            # Try placing at the "corners" formed by existing items and truck boundaries
            possible_x_positions = [0.0] + [i.x + i.length for i in truck.items]
            possible_y_positions = [0.0] + [i.y + i.width for i in truck.items]

            # Filter out x and y positions that are beyond the truck boundaries.
            possible_x_positions = [x for x in possible_x_positions if x < truck.length]
            possible_y_positions = [y for y in possible_y_positions if y < truck.width]

            for y in sorted(list(set(possible_y_positions))):
                for x in sorted(list(set(possible_x_positions))):
                    item.x = x
                    item.y = y

                    if not self.simulation.check_truck_constraints(truck, item):
                        continue

                    valid_position = True
                    for other_item in truck.items:
                        if self.simulation.check_overlap(item, other_item):
                            valid_position = False
                            break

                    if valid_position and self.simulation.check_support_constraint(truck, item):
                        return True, x, y, z

        return False, 0.0, 0.0, 0.0

    def solve_3d_bin_packing(self, truck_types: List[Truck], items: List[Item]) -> Tuple[int, List[Truck]]:
        """Solves the 3D bin packing problem."""
        trucks_used: List[Truck] = []
        items.sort(key=lambda x: x.volume, reverse=True)  # Sort by volume, largest first
        unplaced_items = items.copy()

        truck_type_combinations = itertools.cycle(
            truck_types)  # Use itertools.cycle to cycle the combinations of trucks

        while unplaced_items:
            current_truck_type = next(truck_type_combinations)
            truck = Truck(current_truck_type.type_id, current_truck_type.length,
                          current_truck_type.width, current_truck_type.height,
                          current_truck_type.max_weight)  # Create a instance of truck
            trucks_used.append(truck)
            items_to_remove = []

            for item in unplaced_items:
                placed, x, y, z = self.find_placement_position(truck, item)
                if placed:
                    item.x = x
                    item.y = y
                    item.z = z
                    item.placed = True
                    truck.items.append(item)
                    truck.used_volume += item.volume
                    truck.used_weight += item.weight
                    items_to_remove.append(item)

            for item in items_to_remove:
                unplaced_items.remove(item)
        return len(trucks_used), trucks_used


class Visualizer:
    """Visualizes the solution using matplotlib."""

    def __init__(self):
        pass

    def visualize_truck(self, truck: Truck):
        """Visualizes the contents of a single truck using matplotlib."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw the truck itself as a wireframe
        ax.plot([0, truck.length, truck.length, 0, 0],
                [0, 0, truck.width, truck.width, 0],
                [0, 0, 0, 0, 0], color='black')
        ax.plot([0, truck.length, truck.length, 0, 0],
                [0, 0, truck.width, truck.width, 0],
                [truck.height, truck.height, truck.height, truck.height, truck.height], color='black')
        for i in range(2):
            ax.plot([0, 0], [0, truck.width], [i * truck.height, i * truck.height], color='black')
            ax.plot([truck.length, truck.length], [0, truck.width], [i * truck.height, i * truck.height],
                    color='black')
            ax.plot([0, truck.length], [0, 0], [i * truck.height, i * truck.height], color='black')
            ax.plot([0, truck.length], [truck.width, truck.width], [i * truck.height, i * truck.height],
                    color='black')

        # Draw each item as a colored box
        colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Add more colors if needed
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
            poly3d = Poly3DCollection(faces, facecolors=colors[color_index % len(colors)], linewidths=1,
                                      edgecolors='k', alpha=0.5)
            ax.add_collection3d(poly3d)
            color_index += 1

        # Set axis labels and title
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_zlabel('Height')
        ax.set_title(f'Truck {truck.type_id} Contents')

        # Set axis limits
        ax.set_xlim([0, truck.length])
        ax.set_ylim([0, truck.width])
        ax.set_zlim([0, truck.height])

        def constrain_view(event):
            min_elev, max_elev = 20, 80
            # Get current elevation and azimuth
            current_elev = ax.elev
            current_azim = ax.azim  # if you want to restrict azimuth, you can do so similarly

            # Clamp the elevation to the allowed range
            constrained_elev = np.clip(current_elev, min_elev, max_elev)
            # (Optionally) clamp the azimuth. For example, to keep it between -90° and 90°:
            # constrained_azim = np.clip(current_azim, -90, 90)
            constrained_azim = current_azim  # here we leave azimuth free

            # Update the view if needed
            if (constrained_elev != current_elev):
                ax.view_init(elev=constrained_elev, azim=constrained_azim)
                fig.canvas.draw_idle()

        # Connect the constrain_view callback to the mouse button release event.
        fig.canvas.mpl_connect('button_release_event', constrain_view)
        plt.show()


def main():
    # Example Usage (with more comprehensive test data)
    truck_types = [
        Truck("A", 10, 8, 6, 2000),
        Truck("B", 12, 10, 8, 3000),
        Truck("C", 15, 10, 9, 4000),
    ]

    items = [
        Item("1", 2, 2, 2, 100),
        Item("2", 3, 2, 2, 150),
        Item("3", 4, 3, 2, 200),
        Item("4", 2, 2, 3, 120),
        Item("5", 3, 3, 2, 180),
        Item("6", 5, 4, 3, 300),
        Item("7", 2, 2, 1, 80),
        Item("8", 4, 2, 2, 160),
        Item("9", 3, 3, 3, 250),
        Item("10", 5, 5, 2, 400),
        Item("11", 2, 2, 2, 110),
        Item("12", 1, 2, 3, 90),
    ]
    # Add items to make it a bit more challenging
    items.extend([
        Item("13", 4, 4, 2, 350),
        Item("14", 3, 2, 4, 220),
        Item("15", 6, 2, 2, 280),
        Item("16", 5, 3, 1, 190)
    ])

    solver = Solver()
    visualizer = Visualizer()

    num_trucks, packed_trucks = solver.solve_3d_bin_packing(truck_types, items)

    print(f"Number of trucks used: {num_trucks}")
    for i, truck in enumerate(packed_trucks):
        print(f"Truck {i + 1} (Type: {truck.type_id}):")
        for item in truck.items:
            print(f"  - {item}")
        visualizer.visualize_truck(truck)


if __name__ == "__main__":
    main()