from packing import PackingCONST, GetData
import warnings
import types
import sys
import numpy as np
import time
from func_timeout import func_timeout, FunctionTimedOut


def get_init_prompt(base_class_code):
    return (
    """Given a list of items with their dimensions and weights, and a list of truck types with their dimensions and weight capacities, you need to place all items into the trucks without overlapping, minimizing the number of trucks used. The task can be solved step-by-step. In each step, you select an item, a truck, and a position within the truck for the item. Design a novel algorithm to select the next item, truck and its placement to ensure the constraints are satisfied.
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
        
Data Distribution Summary:

Item Properties:
  - Length: Min=6.00, 25th=570.00, Median=730.00, 75th=919.75, Max=4130.00. 
  - Width: Min=100.00, 25th=375.00, Median=480.00, 75th=715.00, Max=2250.00. 
  - Height: Min=45.00, 25th=220.00, Median=455.00, 75th=755.00, Max=2560.00. 
  - Weight: Min=0.14, 25th=8.16, Median=42.50, 75th=149.40, Max=1064.36. 

Truck Properties:
  - Max Load: Min=18000.00, 25th=18000.00, Median=23000.00, 75th=23000.00, Max=23000.00. 
  - Length: Min=5890.00, 25th=5890.00, Median=11920.00, 75th=11920.00, Max=11920.00. 
  - Width: Min=2318.00, 25th=2318.00, Median=2318.00, 75th=2318.00, Max=2318.00. 
  - Height: Min=2270.00, 25th=2270.00, Median=2270.00, 75th=2600.00, Max=2600.00. 

You will be implementing a class `Algorithm` that inherits from the following `BaseAlgorithm` class. You can use the helper methods provided in the base class.
```python
"""
    + base_class_code
    + """
```
Now, your task is to implement a class named `Algorithm` that inherits from `BaseAlgorithm`.
First, describe your novel algorithm and main steps for the `place_item` method in one sentence. This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. (e.g., `# {This describes the algorithm using specific terms.}`)
Next, implement the `Algorithm` class.
This class must contain a method named `place_item`.

**To improve modularity and allow for future targeted optimizations, break down the logic within `place_item` into several private helper methods within the `Algorithm` class (e.g., methods for selecting an item, finding valid positions, evaluating placements, selecting a new truck, etc.).**

**Do not contain definition of base class in the output.**

The `place_item` method should accept 3 input(s): 'unplaced_items', 'trucks_in_use', 'truck_types'. The method should return 6 output(s): 'truck_index', 'item_index', 'x', 'y', 'z', 'truck_type_index'.

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
        

        Use NumPy arrays where appropriate.
        Ensure that all return values are defined before returning them. 
        The algorithm in the `place_item` method and its helper methods should not contain any comments or print statements.
        The algorithm will be evaluated on instances with hundreds of items and a few truck types, which means that the 'occupied_volumes' in `trucks_in_use` will contain large numbers of entries. Consider implementing an efficient search for available space, weight load calculations, and placement validation to handle such large inputs efficiently.
        
Do not give additional explanations beyond the one-sentence algorithm description and the class code. Remove all comments before final output
"""
)

import os
import json
import requests

openrouter_api_key = os.environ.get("LLM_API_KEY")
if not openrouter_api_key:
    print("Error: OPENROUTER_API_KEY environment variable not set.")
    exit(1)


# MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"
MODEL_NAME = "google/gemini-2.0-flash-001"



def generate_with_openrouter(messages):

    url = "https://openrouter.ai/api/v1/chat/completions"
    

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    full_response_content = None
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        response.raise_for_status()

        response_data = response.json()

        if (
            response_data
            and "choices" in response_data
            and len(response_data["choices"]) > 0
            and "message" in response_data["choices"][0]
            and "content" in response_data["choices"][0]["message"]
        ):
            full_response_content = response_data["choices"][0]["message"]["content"]
            print(full_response_content)
        else:
            print("Warning: Could not find expected content in the response structure.")

    except Exception as e:
        print(f"An error occurred during the API call: {e}")

    return full_response_content



packing_problem = PackingCONST()
getData = GetData(5)  # Pass n_truck_types
packing_problem.instance_data = getData.generate_instances(split="test")

class AlgorithmInvoker:
    def __init__(self, base_code_str, definitions_to_exec_list, class_to_instantiate_name):
        self.base_code_str = base_code_str
        self.definitions_to_exec_list = definitions_to_exec_list
        self.class_to_instantiate_name = class_to_instantiate_name

    def get_methods(self):
            module_name_suffix = "".join(filter(str.isalnum, self.class_to_instantiate_name))
            module_name = f"packing_module_{module_name_suffix}_{str(time.time_ns())}"
            packing_module = types.ModuleType(module_name)
            sys.modules[module_name] = packing_module
            if self.base_code_str and self.base_code_str.strip():
                exec(self.base_code_str, packing_module.__dict__)
            for i, code_def in enumerate(self.definitions_to_exec_list):
                try:
                    exec(code_def, packing_module.__dict__)
                except Exception as e_exec:
                    error_msg = (
                        f"Error executing definition for class '{self.class_to_instantiate_name}' or its ancestor. "
                        f"Problem in segment {i+1} of definitions. Error: {type(e_exec).__name__}: {str(e_exec)}\n"
                    )
                    return (None, error_msg, packing_module)
            if not hasattr(packing_module, self.class_to_instantiate_name):
                error_msg = f"Class '{self.class_to_instantiate_name}' not found in module after executing definitions."
                return (None, error_msg, packing_module)
            AlgorithmClass = getattr(packing_module, self.class_to_instantiate_name)
            algorithm = AlgorithmClass()
            return algorithm.place_item

def evaluate(
    problem_instance, base_code_str, definitions_to_exec_list, class_to_instantiate_name
):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create a unique module name to avoid conflicts if run multiple times
            module_name_suffix = "".join(filter(str.isalnum, class_to_instantiate_name))
            module_name = f"packing_module_{module_name_suffix}_{str(time.time_ns())}"

            packing_module = types.ModuleType(module_name)
            sys.modules[module_name] = packing_module  # Register the module

            # Execute base code first, always
            if base_code_str and base_code_str.strip():
                exec(base_code_str, packing_module.__dict__)

            # Execute all class definitions provided, in order
            for i, code_def in enumerate(definitions_to_exec_list):
                try:
                    exec(code_def, packing_module.__dict__)
                except Exception as e_exec:
                    error_msg = (
                        f"Error executing definition for class '{class_to_instantiate_name}' or its ancestor. "
                        f"Problem in segment {i+1} of definitions. Error: {type(e_exec).__name__}: {str(e_exec)}\n"
                    )
                    # print(f"Debug: {error_msg}") # Debug
                    return (None, error_msg, packing_module)

            # Instantiate the target class
            if not hasattr(packing_module, class_to_instantiate_name):
                error_msg = f"Class '{class_to_instantiate_name}' not found in module after executing definitions."
                return (None, error_msg, packing_module)

            AlgorithmClass = getattr(packing_module, class_to_instantiate_name)

            try:
                algorithm = AlgorithmClass()  # Potential __init__ error here
            except Exception as e_init:
                error_msg = f"Error during instantiation of '{class_to_instantiate_name}': {type(e_init).__name__}: {str(e_init)}"
                return (None, error_msg, packing_module)

            # Call greedy method from the problem_instance
            fitness_val_greedy, error_msg_greedy = problem_instance.greedy(
                algorithm.place_item
            )

            # Unregister module after use to prevent pollution if many iterations
            # del sys.modules[module_name] # Be cautious if module objects are stored elsewhere

            return (fitness_val_greedy, error_msg_greedy, packing_module)

    except Exception as e_eval_setup:
        # Catch-all for other unexpected errors during evaluation setup
        error_msg = f"Broader evaluation setup error for '{class_to_instantiate_name}': {type(e_eval_setup).__name__}: {str(e_eval_setup)}"
        # print(f"Debug: {error_msg}") # Debug
        return (None, error_msg, None)


packing_problem = PackingCONST()
getData = GetData(5)
packing_problem.instance_data = getData.generate_instances(split="test")
base_class_code = packing_problem.base_class_code
all_infos = []

code_file_path = "flash_offspring.json"
with open(code_file_path, "r") as file:
    all_codes = json.load(file)

for code_dict in all_codes:
    crappy_code = code_dict["code"]
    code_info = []
    class_namees = ["Algorithm", "AlgorithmIT1", "AlgorithmIT2", "AlgorithmIT3"]
    max_attempts = len(class_namees)
    llm_conversation_history = [
        {"role": "user", "content": get_init_prompt(base_class_code)},
        {"role": "assistant", "content": crappy_code},
    ]

    successful_parent_code_definitions = []

    llm_generated_code_fix = None  # Will hold the code from the LLM for IT1, IT2 etc.
    final_successful_code_block = None
    final_fitness_achieved = None
    import re
    def extract_python_code(markdown_string):
        if not isinstance(markdown_string, str):
            return None
        # Regex to find code block ```python ... ```
        match = re.search(r"```python\n(.*?)\n```", markdown_string, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: if the string starts with "class " or "def " and has no markdown fences
        if "```" not in markdown_string and (
            markdown_string.strip().startswith("class ") or \
            markdown_string.strip().startswith("def ") or \
            markdown_string.strip().startswith("import ")
            ):
            return markdown_string.strip()
        return None


    for attempt_idx in range(max_attempts):
        current_class_to_instantiate = class_namees[attempt_idx]
        print(
            f"\n--- Attempt {attempt_idx + 1}/{max_attempts}: Testing class '{current_class_to_instantiate}' ---"
        )

        if attempt_idx == 0:
            code_for_current_class = crappy_code
        else:
            if llm_generated_code_fix is None:
                print(
                    "Error: LLM did not provide code in the previous iteration. Stopping."
                )
                break
            code_for_current_class = llm_generated_code_fix

        definitions_for_eval = successful_parent_code_definitions + [code_for_current_class]

        print(
            f"Code for current class ({current_class_to_instantiate}) being evaluated:\n{code_for_current_class[:500]}...\n"
        )
        try:
            fitness_val, error_msg_eval, _ = func_timeout(10, evaluate, args=(
                packing_problem,
                base_class_code,
                definitions_for_eval,
                current_class_to_instantiate,
            ))
        except FunctionTimedOut:
            fitness_val = None
            error_msg_eval = f"Evaluation timed out for class '{current_class_to_instantiate}'."

        info={
                "fitness": fitness_val,
        }        
        if fitness_val is None:
            info["error"] = error_msg_eval
        code_info.append(info)

        if (
            fitness_val is not None
        ):  # Success (error_msg_eval would be None or a warning from greedy)
            print(
                f"SUCCESS: Evaluation successful for {current_class_to_instantiate}! Fitness: {fitness_val}"
            )
            final_successful_code_block = code_for_current_class
            final_fitness_achieved = (fitness_val, error_msg_eval)
            break  # Exit loop on success
        else:  # Failure
            print(
                f"FAILURE: Evaluation failed for {current_class_to_instantiate}. Error: {error_msg_eval}"
            )

            if attempt_idx == max_attempts - 1:
                print("Max attempts reached. Could not fix the code.")
                break

            # Prepare for the next iteration: Ask LLM to fix `code_for_current_class`
            # The class to be generated by LLM is `class_namees[attempt_idx + 1]`
            # It should inherit from `current_class_to_instantiate` (which is `class_namees[attempt_idx]`)

            next_llm_class_name = class_namees[attempt_idx + 1]
            parent_class_for_llm_fix = current_class_to_instantiate

            fix_prompt = f"""The Python code for class `{parent_class_for_llm_fix}` has an issue.
    When this class was used in the packing algorithm, it resulted in the following error:
    {error_msg_eval}

    Instructions:
    1. Analyze the error message and the provided code for `{parent_class_for_llm_fix}`.
    2. Create a new Python class named `{next_llm_class_name}`.
    3. This new class `{next_llm_class_name}` MUST inherit from class `{parent_class_for_llm_fix}`.
    4. In the new class (`{next_llm_class_name}`), override only the specific method(s) from `{parent_class_for_llm_fix}` that are causing the error or are directly related to fixing it. Do NOT rewrite the entire `{parent_class_for_llm_fix}` class, only provide the overridden methods or necessary additions in `{next_llm_class_name}`.
    5. The primary objective is to resolve the specified error so the code can run without this error.
    6. Ensure your fixed code is robust and adheres to the original problem's requirements if the error points to a deviation.
    7. Your response should ONLY be the Python code for the new `{next_llm_class_name}` class. Do not include any explanatory text, markdown formatting for the code block itself, or anything other than the class definition.
    """

            current_fix_request_message = {"role": "user", "content": fix_prompt}
            messages_to_send_to_llm = llm_conversation_history + [
                current_fix_request_message
            ]

            print(
                f"\nSending request to OpenRouter for class {next_llm_class_name} (attempt {attempt_idx + 2})..."
            )
            # print(f"LLM History Preview (last user message): {messages_to_send_to_llm[-1]['content'][:200]}...")

            llm_generated_code_fix_raw = generate_with_openrouter(messages_to_send_to_llm)
            llm_generated_code_fix = extract_python_code(llm_generated_code_fix_raw)

            if llm_generated_code_fix:
                llm_conversation_history.append(current_fix_request_message)
                llm_conversation_history.append(
                    {"role": "assistant", "content": llm_generated_code_fix}
                )
                successful_parent_code_definitions.append(code_for_current_class)
            else:
                print(
                    "LLM did not return any response (e.g., API error or empty content). Stopping."
                )
                break
    else:  # Loop finished (either by break on success, break on error, or max_attempts)
        if final_successful_code_block:
            print(f"\n--- Iteration Finished: SUCCESS ---")
        else:
            print(f"\n--- Iteration Finished: FAILED ---")
    all_infos.append(code_info)
dump_results={
    "model_name": MODEL_NAME,
    "all_infos": all_infos,
}
import time
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_file = f"./results/{MODEL_NAME}_{timestamp}.json"
save_dir= os.path.dirname(output_file)
os.makedirs(save_dir, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(dump_results, f, indent=4)
