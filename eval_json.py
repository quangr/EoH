from packing import PackingCONST, GetData
import warnings
import types
import sys
import numpy as np
import time
from func_timeout import func_timeout, FunctionTimedOut


import os
import json






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
getData = GetData(50)  # Pass n_truck_types
packing_problem.instance_data = getData.generate_instances(split="test")
base_class_code = packing_problem.base_class_code
all_infos = []

code_file_path = "output/50_2/results/pops/population_generation_5.json"
with open(code_file_path, "r") as file:
    all_codes = json.load(file)

for code_dict in all_codes:
    crappy_code = code_dict["code"]
    class_namees = ["Algorithm", "AlgorithmIT1", "AlgorithmIT2", "AlgorithmIT3"]

    llm_generated_code_fix = None  # Will hold the code from the LLM for IT1, IT2 etc.
    final_successful_code_block = None
    final_fitness_achieved = None


    current_class_to_instantiate = code_dict["class_name"]
    code_for_current_class = llm_generated_code_fix

    definitions_for_eval = [code_dict["code"]]
    start_time = time.time()
    fitness_val, error_msg_eval, _ = evaluate(packing_problem,
        base_class_code,
        definitions_for_eval,
        current_class_to_instantiate,
    )
    print(f"Evaluation time for {current_class_to_instantiate}: {time.time() - start_time:.2f} seconds")
    print(fitness_val)