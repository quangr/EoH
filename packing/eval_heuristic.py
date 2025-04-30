from packing2d import PackingCONST
import warnings
import types
import sys
import numpy as np
packing_problem = PackingCONST()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
container_width=100

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
def implement_algorithms():
    algorithms = {}
    algorithms["BL"] = """
import numpy as np

def place_rectangle(unplaced_rectangles, occupied_areas, container_width):
    if not unplaced_rectangles:
        return -1, -1, -1

    # Sort rectangles by width in descending order
    sorted_indices = np.argsort([-w for w, h in unplaced_rectangles])
    rectangles = np.array(unplaced_rectangles)[sorted_indices]

    for rect_index in range(len(rectangles)):
        width, height = rectangles[rect_index]

        best_x = -1
        best_y = float('inf')

        # Iterate through all possible positions
        for y in range(height + max([oy + oh for _,oy,_,oh in occupied_areas]  if occupied_areas else [0])):
            for x in range(container_width - width + 1):
                valid_position = True

                # Check for overlaps
                for ox, oy, ow, oh in occupied_areas:
                    if (x < ox + ow and x + width > ox and
                        y < oy + oh and y + height > oy):
                        valid_position = False
                        break

                if valid_position:
                    # Update best position if lower
                    if y < best_y:
                        best_y = y
                        best_x = x
                    elif y == best_y and x < best_x:
                        best_x = x

        if best_x != -1 and best_y != float('inf'):
           original_index = sorted_indices[rect_index]
           return original_index, best_x, best_y

    return -1, -1, -1
"""
    algorithms["FFDH"] = """
import numpy as np

def get_levels(occupied_areas, container_width):
    levels = []
    for x, y, w, h in occupied_areas:
        level_found = False
        for i in range(len(levels)):
            level_y, level_x = levels[i]
            if y == level_y:
                levels[i] = (level_y,level_x+w)
                level_found=True
                break
        if not level_found:
            levels.append((y,x+w))
    levels = [(level_y, container_width - level_x) for level_y, level_x in levels] #convert to remaining width
    levels.sort()
    return levels


def place_rectangle(unplaced_rectangles, occupied_areas, container_width):
    if not unplaced_rectangles:
        return -1, -1, -1

    # Sort rectangles by height in descending order
    sorted_indices = np.argsort([-h for w, h in unplaced_rectangles])
    rectangles = np.array(unplaced_rectangles)[sorted_indices]
    
    
    
    for rect_index in range(len(rectangles)):
      width, height = rectangles[rect_index]
      placed = False
      levels = get_levels(occupied_areas, container_width)
      # Try to place in existing levels
      for level_index, (level_y, level_remaining_width) in enumerate(levels):
          if level_remaining_width >= width:
              # Check for overlap with existing rectangles at this level
              x = container_width - level_remaining_width
              y = level_y
              
              valid_position = True
              for ox, oy, ow, oh in occupied_areas:
                  if (x < ox + ow and x + width > ox and
                      y < oy + oh and y + height > oy):
                      valid_position = False
                      break              
              if valid_position:
                original_index = sorted_indices[rect_index]
                return original_index, x, y
                

      # If not placed, create a new level
      if not placed:
        new_level_y = 0
        if occupied_areas:
            new_level_y = max(y + h for x, y, w, h in occupied_areas)
        if width <= container_width:
          original_index = sorted_indices[rect_index]
          return original_index, 0, new_level_y  # Place at the start of the new level
    return -1,-1,-1

"""

    algorithms["NFDH"] = """
import numpy as np

def get_levels(occupied_areas, container_width):
    levels = []
    for x, y, w, h in occupied_areas:
        level_found = False
        for i in range(len(levels)):
            level_y, level_x = levels[i]
            if y == level_y:
                levels[i] = (level_y,level_x+w)
                level_found=True
                break
        if not level_found:
            levels.append((y,x+w))
    levels = [(level_y, container_width - level_x) for level_y, level_x in levels] #convert to remaining width
    levels.sort()
    return levels

def place_rectangle(unplaced_rectangles, occupied_areas, container_width):
    if not unplaced_rectangles:
        return -1, -1, -1

    # Sort by height (descending)
    sorted_indices = np.argsort([-h for w, h in unplaced_rectangles])
    rectangles = np.array(unplaced_rectangles)[sorted_indices]

    levels = get_levels(occupied_areas,container_width)
    if levels:
        current_level_y, current_level_remaining_width = levels[-1]  #Use last level
    else:
        current_level_y = 0
        current_level_remaining_width = container_width

    for rect_index in range(len(rectangles)):
        width, height = rectangles[rect_index]

        # Check if it fits on the current level
        if current_level_remaining_width >= width:
          x = container_width - current_level_remaining_width
          y = current_level_y
          valid_position = True
          for ox, oy, ow, oh in occupied_areas:
            if (x < ox + ow and x + width > ox and
                y < oy + oh and y + height > oy):
              valid_position = False
              break
          if valid_position:
            original_index = sorted_indices[rect_index]
            return original_index, x, y

        # Doesn't fit, or no current level, create a new level
        current_level_y = 0
        if occupied_areas :
            current_level_y = max(y + h for x,y,w,h in occupied_areas)
        if width <= container_width:
          current_level_remaining_width = container_width - width
          original_index = sorted_indices[rect_index]
          return original_index, 0, current_level_y
    return -1,-1,-1
"""

    algorithms["BFDH"] = """
import numpy as np

def get_levels(occupied_areas, container_width):
    levels = []
    for x, y, w, h in occupied_areas:
        level_found = False
        for i in range(len(levels)):
            level_y, level_x = levels[i]
            if y == level_y:
                levels[i] = (level_y,level_x+w)
                level_found=True
                break
        if not level_found:
            levels.append((y,x+w))
    levels = [(level_y, container_width - level_x) for level_y, level_x in levels] #convert to remaining width
    levels.sort()
    return levels

def place_rectangle(unplaced_rectangles, occupied_areas, container_width):
    if not unplaced_rectangles:
        return -1, -1, -1

    # Sort by height (descending)
    sorted_indices = np.argsort([-h for w, h in unplaced_rectangles])
    rectangles = np.array(unplaced_rectangles)[sorted_indices]
    
    for rect_index in range(len(rectangles)):
        width, height = rectangles[rect_index]
        best_level_index = -1
        min_residual_width = float('inf')
        levels = get_levels(occupied_areas,container_width)
        # Find best level
        for i, (level_y, level_remaining_width) in enumerate(levels):
            if level_remaining_width >= width:
              x = container_width - level_remaining_width
              y = level_y
              valid_position = True
              for ox, oy, ow, oh in occupied_areas:
                if (x < ox + ow and x + width > ox and
                  y < oy + oh and y + height > oy):
                  valid_position = False
                  break              
              if valid_position:
                if level_remaining_width - width < min_residual_width:
                    min_residual_width = level_remaining_width - width
                    best_level_index = i

        # Place on best level or create new level
        if best_level_index != -1:
            level_y, level_remaining_width = levels[best_level_index]
            x = container_width - level_remaining_width
            y = level_y
            original_index = sorted_indices[rect_index]
            return original_index, x, y
        else:
            new_level_y = 0
            if occupied_areas:
              new_level_y = max(oy + oh for ox, oy, ow, oh in occupied_areas)
            if width <= container_width:
              original_index = sorted_indices[rect_index]
              return original_index, 0, new_level_y
    return -1, -1, -1
"""
    return algorithms
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
algorithms = implement_algorithms()
instance_data=GetData(1000, 20, container_width).generate_instances()
packing_problem.instance_data = instance_data    
for name, code in algorithms.items():
    print(f"Evaluating {name}...")
    fitness, all_occupied_areas = evaluate(packing_problem, code)

    if fitness is not None and all_occupied_areas is not None:
        print(f"  Average container height: {fitness}")

        # Visualize each instance
        for i, occupied_areas in enumerate(all_occupied_areas[:1]):  # Limit to first 2 for brevity
            rectangles, container_width = packing_problem.instance_data[i]
            container_height = packing_problem.calculate_used_height(occupied_areas)
            visualize_packing(occupied_areas, container_width, container_height, i)
    else:
        print(f"  Evaluation of {name} failed.")