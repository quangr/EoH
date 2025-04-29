from packing import PackingCONST,Truck
import warnings
import types
import sys
import numpy as np
packing_problem = PackingCONST()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import copy

import numpy as np
import line_profiler
@line_profiler.profile
def place_item(unplaced_items, trucks_in_use, truck_types):
    """
    {This algorithm prioritizes placing larger, heavier items first, then iterates through existing trucks to find the best fit, considering space, weight capacity, and support, before resorting to opening a new truck if necessary.}
    """
    truck_index = -1
    item_index = -1
    x = -1.0
    y = -1.0
    z = -1.0
    truck_type_index = -1

    # Prioritize items with larger volume and weight
    item_priorities = []
    for i, item in enumerate(unplaced_items):
        item_priorities.append(item['length'] * item['width'] * item['height'] * item['weight'])
    
    sorted_item_indices = sorted(range(len(item_priorities)), key=lambda k: item_priorities[k], reverse=True)

    for item_idx in sorted_item_indices:
        item = unplaced_items[item_idx]
        item_length = item['length']
        item_width = item['width']
        item_height = item['height']
        item_weight = item['weight']

        best_truck_index = -1
        best_x = -1.0
        best_y = -1.0
        best_z = -1.0

        # Try to fit the item into existing trucks
        for t_idx, truck in enumerate(trucks_in_use):
            truck_type_index_current = truck['truck_type_index']
            truck_capacity, truck_length, truck_width, truck_height = truck_types[truck_type_index_current]

            if truck['current_weight'] + item_weight <= truck_capacity:
                # Find a suitable position within the truck
                occupied_volumes = truck['occupied_volumes']
                
                # Try placing the item at various locations within the truck
                for cur_x in np.arange(0, truck_length - item_length + 0.1, min(item_length, 0.5)):
                    for cur_y in np.arange(0, truck_width - item_width + 0.1, min(item_width, 0.5)):
                        
                        # Find the highest z where the item can be placed such that it's either on the floor or supported by other items.
                        max_z = 0.0
                        
                        # Check for support from the floor
                        
                        # Check for support from other boxes
                        for placed_item in occupied_volumes:
                            placed_item_x = placed_item[1]
                            placed_item_y = placed_item[2]
                            placed_item_z = placed_item[3]
                            placed_item_length = placed_item[4]
                            placed_item_width = placed_item[5]
                            placed_item_height = placed_item[6]
                            
                            if (cur_x >= placed_item_x - 0.001 and cur_x <= placed_item_x + placed_item_length + 0.001 - item_length and
                                cur_y >= placed_item_y - 0.001 and cur_y <= placed_item_y + placed_item_width + 0.001 - item_width):
                                max_z = max(max_z, placed_item_z + placed_item_height)

                        cur_z = max_z

                        if cur_z + item_height <= truck_height:

                            # Check for overlap with existing items
                            overlap = False
                            for placed_item in occupied_volumes:
                                placed_item_x = placed_item[1]
                                placed_item_y = placed_item[2]
                                placed_item_z = placed_item[3]
                                placed_item_length = placed_item[4]
                                placed_item_width = placed_item[5]
                                placed_item_height = placed_item[6]

                                if (cur_x < placed_item_x + placed_item_length and
                                    cur_x + item_length > placed_item_x and
                                    cur_y < placed_item_y + placed_item_width and
                                    cur_y + item_width > placed_item_y and
                                    cur_z < placed_item_z + placed_item_height and
                                    cur_z + item_height > placed_item_z):
                                    overlap = True
                                    break

                            if not overlap:
                                # Check for support
                                supported_area = 0.0
                                if cur_z == 0.0:
                                    supported_area = item_length * item_width # Supported by floor
                                else:
                                    for placed_item in occupied_volumes:
                                        placed_item_x = placed_item[1]
                                        placed_item_y = placed_item[2]
                                        placed_item_z = placed_item[3]
                                        placed_item_length = placed_item[4]
                                        placed_item_width = placed_item[5]
                                        placed_item_height = placed_item[6]

                                        x_overlap = max(0.0, min(cur_x + item_length, placed_item_x + placed_item_length) - max(cur_x, placed_item_x))
                                        y_overlap = max(0.0, min(cur_y + item_width, placed_item_y + placed_item_width) - max(cur_y, placed_item_y))

                                        if cur_z == placed_item_z + placed_item_height:
                                            supported_area += x_overlap * y_overlap

                                if supported_area >= 0.8 * item_length * item_width:
                                    # Found a valid placement
                                    best_truck_index = t_idx
                                    best_x = cur_x
                                    best_y = cur_y
                                    best_z = cur_z
                                    break
                    if best_truck_index != -1:
                        break

        if best_truck_index != -1:
            # Use the best truck found
            truck_index = best_truck_index
            item_index = item_idx
            x = best_x
            y = best_y
            z = best_z
            break

    # If no suitable truck was found, open a new truck
    if truck_index == -1:
        # Find the smallest truck that can accommodate the item
        best_truck_type_index = -1
        min_volume = float('inf')

        for tt_idx, truck_type in enumerate(truck_types):
            truck_capacity, truck_length, truck_width, truck_height = truck_type
            if (item_length <= truck_length and item_width <= truck_width and item_height <= truck_height and item_weight <= truck_capacity):
                volume = truck_length * truck_width * truck_height
                if volume < min_volume:
                    min_volume = volume
                    best_truck_type_index = tt_idx

        if best_truck_type_index != -1:
            truck_index = -1
            item_index = item_idx
            x = 0.0
            y = 0.0
            z = 0.0
            truck_type_index = best_truck_type_index
        else:
            # No truck can accommodate the item. This is an error.
            # Assign a default value so that the simulator won't break.
            # Ideally, you should handle such a case more gracefully.
            truck_index = -1
            item_index = 0
            x = 0.0
            y = 0.0
            z = 0.0
            truck_type_index = 0

    return truck_index, item_index, x, y, z, truck_type_index
def evaluate(self, code_string):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packing_module = types.ModuleType("packing_module")
            exec(code_string, packing_module.__dict__)
            sys.modules[packing_module.__name__] = packing_module
            fitness = self.greedy(place_item)
            return fitness
    except Exception as e:
        print("Error during evaluation:", str(e))
        return None
fitness = evaluate(packing_problem,"")
def visualize_truck( truck: Truck):
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

    # Set axis limits
    ax.set_xlim([0, truck.length])
    ax.set_ylim([0, truck.width])
    ax.set_zlim([0, truck.height])

    # Add volume and weight ratio information to the plot
    volume_ratio = truck.used_volume / (truck.length * truck.width * truck.height)
    weight_ratio = truck.used_weight / truck.max_weight  # Use max_weight instead of capacity

    plt.title(f'Volume Ratio: {volume_ratio:.2f}, Weight Ratio: {weight_ratio:.2f}')


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
        trucks[truck_index].used_volume += item_copy.length * item_copy.width * item_copy.height
        trucks[truck_index].used_weight += item_copy.weight

    # Visualize each truck
    for truck in trucks:
        visualize_truck(truck)

if fitness is not None:
    print(f"Average container height: {fitness[0]}")

    plan_to_visualize, _ = fitness[1][0]
    items, truck_types = packing_problem.instance_data[0]  # Get items and truck types from the first instance
    print(f"Number of trucks used: {len(set([placement['truck_index'] for placement in plan_to_visualize]))}")
    visualize_plan(plan_to_visualize, items, truck_types)