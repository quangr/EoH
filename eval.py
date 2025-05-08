from packing import PackingCONST,Truck, GetData
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
getData = GetData(50)  # Pass n_truck_types
packing_problem.instance_data = getData.generate_instances(split="test")

example_code = "import numpy as np\n\ndef place_item(unplaced_items, trucks_in_use, truck_types):\n    \"\"\"{This algorithm prioritizes placing items with the largest footprint (length * width) into existing trucks based on the tightest-fit placement strategy, opening a new truck type only when existing trucks are unsuitable, favoring trucks that minimize remaining volume.}\"\"\"\n    truck_index = -1\n    item_index = -1\n    x = -1.0\n    y = -1.0\n    z = -1.0\n    truck_type_index = -1\n\n    best_item_index = -1\n    max_footprint = -1.0\n\n    for i, item in enumerate(unplaced_items):\n        footprint = item['length'] * item['width']\n        if footprint > max_footprint:\n            max_footprint = footprint\n            best_item_index = i\n\n    if best_item_index == -1:\n        return -1, -1, -1.0, -1.0, -1.0, -1\n\n    item = unplaced_items[best_item_index]\n    item_length = item['length']\n    item_width = item['width']\n    item_height = item['height']\n    item_weight = item['weight']\n\n    best_truck_index = -1\n    best_x = -1.0\n    best_y = -1.0\n    best_z = -1.0\n    min_remaining_volume = float('inf')\n\n    for t_idx, truck in enumerate(trucks_in_use):\n        truck_type_idx = truck['truck_type_index']\n        truck_capacity, truck_length, truck_width, truck_height = truck_types[truck_type_idx]\n\n        if truck['current_weight'] + item_weight <= truck_capacity:\n            occupied_volumes = truck['occupied_volumes']\n\n            potential_placements = [[0.0, 0.0, 0.0]]\n            for placed_item in occupied_volumes:\n                potential_placements.append([placed_item[1] + placed_item[4], placed_item[2], placed_item[3]])\n                potential_placements.append([placed_item[1], placed_item[2] + placed_item[5], placed_item[3]])\n\n            for px, py, pz in potential_placements:\n                valid_position = True\n                for placed_item in occupied_volumes:\n                    if (px < placed_item[1] + placed_item[4] and\n                            px + item_length > placed_item[1] and\n                            py < placed_item[2] + placed_item[5] and\n                            py + item_width > placed_item[2] and\n                            pz < placed_item[3] + placed_item[6] and\n                            pz + item_height > placed_item[3]):\n                        valid_position = False\n                        break\n\n                if valid_position and px + item_length <= truck_length and py + item_width <= truck_width and pz + item_height <= truck_height:\n                    supported = False\n                    support_area = 0.0\n                    item_bottom_area = item_length * item_width\n\n                    if pz == 0.0:\n                        supported = True\n                    else:\n                        for other_item in occupied_volumes:\n                            if pz == other_item[3] + other_item[6]:\n                                x_overlap = max(0, min(px + item_length, other_item[1] + other_item[4]) - max(px, other_item[1]))\n                                y_overlap = max(0, min(py + item_width, other_item[2] + other_item[5]) - max(py, other_item[2]))\n                                support_area += x_overlap * y_overlap\n\n                        if support_area / item_bottom_area >= 0.8:\n                            supported = True\n\n                    if supported:\n                        occupied_volume = sum([vol[4] * vol[5] * vol[6] for vol in truck['occupied_volumes']])\n                        remaining_volume = (truck_length * truck_width * truck_height) - occupied_volume - (item_length * item_width * item_height)\n\n                        if remaining_volume < min_remaining_volume:\n                            min_remaining_volume = remaining_volume\n                            best_truck_index = t_idx\n                            best_x = px\n                            best_y = py\n                            best_z = pz\n\n    if best_truck_index != -1:\n        truck_index = best_truck_index\n        item_index = best_item_index\n        x = best_x\n        y = best_y\n        z = best_z\n        truck_type_index = trucks_in_use[best_truck_index]['truck_type_index']\n    else:\n        best_truck_type_index = -1\n        min_truck_volume = float('inf')\n        for k, truck_type in enumerate(truck_types):\n            truck_capacity, truck_length, truck_width, truck_height = truck_type\n            truck_volume = truck_length * truck_width * truck_height\n\n            if item_length <= truck_length and item_width <= truck_width and item_height <= truck_height and item_weight <= truck_capacity:\n                if truck_volume < min_truck_volume:\n                    min_truck_volume = truck_volume\n                    best_truck_type_index = k\n\n        if best_truck_type_index != -1:\n            truck_index = -1\n            item_index = best_item_index\n            x = 0.0\n            y = 0.0\n            z = 0.0\n            truck_type_index = best_truck_type_index\n        else:\n            truck_index = -1\n            item_index = -1\n            x = -1.0\n            y = -1.0\n            z = -1.0\n            truck_type_index = -1\n\n    return truck_index, item_index, x, y, z, truck_type_index"
print(example_code)
def evaluate(self, code_string):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packing_module = types.ModuleType("packing_module")
            exec(code_string, packing_module.__dict__)
            sys.modules[packing_module.__name__] = packing_module
            start_time = time.time()
            fitness = self.greedy(packing_module.place_item)
            print("Fitness:", fitness)
            print("Execution time:", time.time() - start_time)
            return fitness
    except Exception as e:
        print("Error during evaluation:", str(e))
        return None
fitness = evaluate(packing_problem,example_code)
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
    # print(f"Number of trucks used: {len(set([placement['truck_index'] for placement in plan_to_visualize]))}")
    visualize_plan(plan_to_visualize, items, truck_types)