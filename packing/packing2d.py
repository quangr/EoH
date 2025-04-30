import numpy as np
import types
import sys
import warnings
from tqdm import trange

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
            widths = np.random.randint(1, self.container_width // 2, self.n_rectangles)
            heights = np.random.randint(1, 10, self.n_rectangles)  # Adjust height range as needed
            rectangles = list(zip(widths, heights))
            instance_data.append((rectangles, self.container_width))
        return instance_data

class GetPrompts():
    def __init__(self):
        self.prompt_task = "Given a set of rectangles with their widths and heights, and a container of fixed width, \
you need to place all rectangles into the container without overlapping, minimizing the total height of the container used. \
The task can be solved step-by-step by selecting a rectangle and its position in each step. \
Help me design a novel algorithm that is different from the algorithms in literature to select the next rectangle and its placement."
        self.prompt_func_name = "place_rectangle"
        self.prompt_func_inputs = ["unplaced_rectangles", "occupied_areas", "container_width"]
        self.prompt_func_outputs = ["index", "x", "y"]
        self.prompt_inout_inf = "'unplaced_rectangles' is a list of (width, height) tuples. 'occupied_areas' is a list of (x, y, width, height) tuples representing placed rectangles. 'container_width' is a scalar.  'index' is the index of the chosen rectangle in 'unplaced_rectangles'. 'x' and 'y' are the bottom-left coordinates for placement."
        self.prompt_other_inf = "All numerical values are integers.  Use NumPy arrays where appropriate."

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



class PackingCONST():
    def __init__(self) -> None:
        self.n_rectangles = 20  # Number of rectangles per instance
        self.container_width = 20
        self.n_instance = 20
        self.running_time = 10

        self.prompts = GetPrompts()

        getData = GetData(self.n_instance, self.n_rectangles, self.container_width)
        self.instance_data = getData.generate_instances()


    def calculate_used_height(self, occupied_areas):
        if not occupied_areas:
            return 0
        max_y = 0
        for x, y, w, h in occupied_areas:
            max_y = max(max_y, y + h)
        return max_y

    #@func_set_timeout(5)
    def greedy(self, eva):
        total_heights = []
        for rectangles, container_width in self.instance_data:
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
                        return None

                    # Get the actual rectangle index from the list of unplaced indices
                    actual_rect_index = unplaced_rectangles[rect_index]
                    
                    rect_width, rect_height = rectangles[actual_rect_index]
                    if x + rect_width > container_width:
                        return None
                    # Basic overlap check (you might need a more robust check)
                    overlap = False
                    for ox, oy, ow, oh in occupied_areas:
                        if (x < ox + ow and x + rect_width > ox and
                            y < oy + oh and y + rect_height > oy):
                            overlap = True
                            break
                    if overlap:
                        #print("Overlap detected, algorithm failed.")
                        return None

                    occupied_areas.append((x, y, rect_width, rect_height))
                    unplaced_rectangles.pop(rect_index)  # Remove the placed rectangle
                except Exception as e:
                    print("An error occurred:",str(e))
                    return None


            total_height = self.calculate_used_height(occupied_areas)
            total_heights.append(total_height)

        ave_height = np.average(total_heights)
        #print("Average used height:", ave_height)
        return ave_height



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
            #print("Error during evaluation:", str(e))
            return None