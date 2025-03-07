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

example_code = "import numpy as np\n\ndef place_item(unplaced_items, trucks_in_use, truck_types):\n    \"\"\"{This algorithm prioritizes filling existing trucks by selecting the item that best fits the available space in a truck, considering both volume and weight, and only opens a new truck when necessary.}\"\"\"\n    truck_index = -1\n    item_index = -1\n    x = 0.0\n    y = 0.0\n    z = 0.0\n    truck_type_index = -1\n\n    best_item_index = -1\n    best_truck_index = -1\n    best_x = -1\n    best_y = -1\n    best_z = -1\n    best_fit_score = float('inf')\n\n    # Iterate through existing trucks\n    for t_idx, truck in enumerate(trucks_in_use):\n        truck_type_index_local = truck['truck_type_index']\n        truck_capacity, truck_length, truck_width, truck_height = truck_types[truck_type_index_local]\n        current_weight = truck['current_weight']\n\n        # Iterate through unplaced items\n        for i_idx, item in enumerate(unplaced_items):\n            item_length = item['length']\n            item_width = item['width']\n            item_height = item['height']\n            item_weight = item['weight']\n\n            if current_weight + item_weight > truck_capacity:\n                continue\n\n            # Calculate occupied volume\n            occupied_volume = 0\n            for placed_item in truck['occupied_volumes']:\n                occupied_volume += placed_item[4] * placed_item[5] * placed_item[6]\n            remaining_volume = truck_length * truck_width * truck_height - occupied_volume\n\n            # Find a valid placement\n            for cur_z in sorted(list({0.0} | {vol[3] + vol[6] for vol in truck['occupied_volumes']})):\n                for cur_x in sorted(list({0.0} | {vol[1] + vol[4] for vol in truck['occupied_volumes']})):\n                    for cur_y in sorted(list({0.0} | {vol[2] + vol[5] for vol in truck['occupied_volumes']})):\n\n                        if (cur_x >= 0 and cur_x + item_length <= truck_length and\n                            cur_y >= 0 and cur_y + item_width <= truck_width and\n                            cur_z >= 0 and cur_z + item_height <= truck_height):\n\n                            overlap = False\n                            for occupied_item in truck['occupied_volumes']:\n                                if (cur_x < occupied_item[1] + occupied_item[4] and cur_x + item_length > occupied_item[1] and\n                                    cur_y < occupied_item[2] + occupied_item[5] and cur_y + item_width > occupied_item[2] and\n                                    cur_z < occupied_item[3] + occupied_item[6] and cur_z + item_height > occupied_item[3]):\n                                    overlap = True\n                                    break\n\n                            if not overlap:\n                                supported = False\n                                if cur_z == 0.0:\n                                    supported = True\n                                else:\n                                    support_area = 0.0\n                                    for occupied_item in truck['occupied_volumes']:\n                                        if abs(cur_z - (occupied_item[3] + occupied_item[6])) < 0.001:\n                                            x_overlap = max(0, min(cur_x + item_length, occupied_item[1] + occupied_item[4]) - max(cur_x, occupied_item[1]))\n                                            y_overlap = max(0, min(cur_y + item_width, occupied_item[2] + occupied_item[5]) - max(cur_y, occupied_item[2]))\n\n                                            if x_overlap > 0 and y_overlap > 0:\n                                                support_area += x_overlap * y_overlap\n\n                                    if support_area >= 0.8 * item_length * item_width:\n                                        supported = True\n\n                                if supported:\n                                    # Calculate fit score (lower is better).  Prioritize tighter fits.\n                                    wasted_volume = (truck_length - (cur_x + item_length)) * truck_width * truck_height + \\\n                                                    (truck_width - (cur_y + item_width)) * truck_length * truck_height + \\\n                                                    (truck_height - (cur_z + item_height)) * truck_length * truck_width\n                                    fit_score = wasted_volume + (truck_capacity - current_weight - item_weight) \n                                    if fit_score < best_fit_score:\n                                        best_fit_score = fit_score\n                                        best_truck_index = t_idx\n                                        best_item_index = i_idx\n                                        best_x = cur_x\n                                        best_y = cur_y\n                                        best_z = cur_z\n\n    # If no suitable placement in existing trucks, open a new truck\n    if best_truck_index == -1:\n        best_truck_type_index = -1\n        best_item_index_new_truck = -1\n        for i_idx, item in enumerate(unplaced_items):\n            item_length = item['length']\n            item_width = item['width']\n            item_height = item['height']\n            item_weight = item['weight']\n            for k, truck_type in enumerate(truck_types):\n                capacity, length, width, height = truck_type\n                if capacity >= item_weight and length >= item_length and width >= item_width and height >= item_height:\n                    if best_truck_type_index == -1 or capacity < truck_types[best_truck_type_index][0]:\n                        best_truck_type_index = k\n                        best_item_index_new_truck = i_idx\n\n        if best_truck_type_index != -1:\n            truck_index = -1\n            item_index = best_item_index_new_truck\n            x = 0.0\n            y = 0.0\n            z = 0.0\n            truck_type_index = best_truck_type_index\n        else:\n            truck_index = -1\n            item_index = -1\n            x = 0.0\n            y = 0.0\n            z = 0.0\n            truck_type_index = -1\n    else:\n        truck_index = best_truck_index\n        item_index = best_item_index\n        x = best_x\n        y = best_y\n        z = best_z\n        truck_type_index = trucks_in_use[truck_index]['truck_type_index']\n        \n    return truck_index, item_index, x, y, z, truck_type_index"

def evaluate(self, code_string):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packing_module = types.ModuleType("packing_module")
            exec(code_string, packing_module.__dict__)
            sys.modules[packing_module.__name__] = packing_module
            fitness = self.greedy(packing_module)
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
    print(f"Number of trucks used: {len(set([placement['truck_index'] for placement in plan_to_visualize]))}")
    visualize_plan(plan_to_visualize, items, truck_types)