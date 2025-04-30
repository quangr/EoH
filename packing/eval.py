from packing2d import PackingCONST
import warnings
import types
import sys
import numpy as np
packing_problem = PackingCONST()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
class GetData():
    def __init__(self, n_instance, n_rectangles, container_width):
        self.n_instance = n_instance
        self.n_rectangles = n_rectangles
        self.container_width = container_width

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            # Generate rectangles with random widths and heights
            widths = np.random.randint(1, 10, self.n_rectangles)
            heights = np.random.randint(1, 10, self.n_rectangles)  # Adjust height range as needed
            rectangles = list(zip(widths, heights))
            instance_data.append((rectangles, self.container_width))
        return instance_data
example_code = "import numpy as np\n\ndef place_rectangle(unplaced_rectangles, occupied_areas, container_width):\n    \"\"\"{This algorithm prioritizes placing rectangles that maximize the area filled at the lowest available height, penalizing larger height and overlap.}\"\"\"\n    unplaced_rectangles = np.array(unplaced_rectangles)\n    occupied_areas = np.array(occupied_areas)\n\n    if len(unplaced_rectangles) == 0:\n        return -1, 0, 0\n\n    best_index = -1\n    best_x = -1\n    best_y = -1\n    best_score = -float('inf')\n\n    for index, (rect_width, rect_height) in enumerate(unplaced_rectangles):\n        for x in range(container_width - rect_width + 1):\n            y = 0\n\n            # Find the lowest possible y position\n            while True:\n                overlap = False\n                for ox, oy, owidth, oheight in occupied_areas:\n                    if (x < ox + owidth and x + rect_width > ox and\n                        y < oy + oheight and y + rect_height > oy):\n                        overlap = True\n                        break\n                if not overlap:\n                    break\n                y += 1\n\n            # Calculate area filled\n            area = rect_width * rect_height\n\n            # Calculate overlap area\n            overlap_area = 0\n            for ox, oy, owidth, oheight in occupied_areas:\n                x_overlap = max(0, min(x + rect_width, ox + owidth) - max(x, ox))\n                y_overlap = max(0, min(y + rect_height, oy + oheight) - max(y, oy))\n                overlap_area += x_overlap * y_overlap\n                \n            # Calculate overall score (prioritizing larger area, lower height, and lower overlap)\n            score = area - y * 10 - overlap_area * 100 - rect_height * 0.1\n\n            if score > best_score:\n                best_score = score\n                best_index = index\n                best_x = x\n                best_y = y\n\n    if best_index != -1:\n        index = best_index\n        x = best_x\n        y = best_y\n    else:\n        index = 0\n        x = 0\n        y = 0\n    \n    return index, x, y"
print(example_code)
container_width=10
def greedy(problem, eva):
    total_heights = []
    all_occupied_areas = []
    for rectangles, container_width in tqdm(problem.instance_data):
        occupied_areas = []
        unplaced_rectangles = list(range(len(rectangles)))  # Indices of unplaced rectangles

        while unplaced_rectangles:
            try:
                rect_index, x, y = eva.place_rectangle(
                    [rectangles[i] for i in unplaced_rectangles],
                    occupied_areas,
                    container_width
                )
                    # Check if the returned index is valid
                if not 0 <= rect_index < len(unplaced_rectangles):
                    #print("Invalid rectangle index returned.")
                    return None, None

                # Get the actual rectangle index from the list of unplaced indices
                actual_rect_index = unplaced_rectangles[rect_index]
                
                rect_width, rect_height = rectangles[actual_rect_index]
                if x + rect_width > container_width:
                    return None, None
                # Basic overlap check (you might need a more robust check)
                overlap = False
                for ox, oy, ow, oh in occupied_areas:
                    if (x < ox + ow and x + rect_width > ox and
                        y < oy + oh and y + rect_height > oy):
                        overlap = True
                        break
                if overlap:
                    #print("Overlap detected, algorithm failed.")
                    return None, None

                occupied_areas.append((x, y, rect_width, rect_height))
                unplaced_rectangles.pop(rect_index)  # Remove the placed rectangle
            except Exception as e:
                print("An error occurred:",str(e))
                return None, None


        total_height = problem.calculate_used_height(occupied_areas)
        total_heights.append(total_height)
        all_occupied_areas.append(occupied_areas) # Store for visualization

    ave_height = np.average(total_heights)
    #print("Average used height:", ave_height)
    return ave_height,all_occupied_areas

def visualize_packing( occupied_areas, container_width, container_height, instance_index):
    """Visualizes the packing of a single instance."""
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)

    # Draw the container
    container = patches.Rectangle((0, 0), container_width, container_height, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(container)

    # Draw the placed rectangles
    for x, y, w, h in occupied_areas:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='blue')
        ax.add_patch(rect)

    plt.title(f"Packing Visualization - Instance {instance_index}")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure correct aspect ratio
    plt.show()
def evaluate(problem, code_string):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packing_module = types.ModuleType("packing_module")
            exec(code_string, packing_module.__dict__)
            sys.modules[packing_module.__name__] = packing_module
            fitness, all_occupied_areas = greedy(problem,packing_module)
            return fitness, all_occupied_areas
    except Exception as e:
        print("Error during evaluation:", str(e))
        return None, None
instance_data=GetData(1000, 20, container_width).generate_instances()
packing_problem.instance_data = instance_data    
fitness, all_occupied_areas = evaluate(packing_problem,example_code)

if fitness is not None and all_occupied_areas is not None:
    print(f"Average container height: {fitness}")

    # Visualize each instance
    for i, occupied_areas in enumerate(all_occupied_areas[:2]):
        # Get the original instance data to determine container height
        rectangles, container_width = packing_problem.instance_data[i]
        container_height = packing_problem.calculate_used_height(occupied_areas)
        visualize_packing(occupied_areas, container_width, container_height, i)
else:
    print("Evaluation failed.")


