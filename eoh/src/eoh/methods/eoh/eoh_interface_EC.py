import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
from func_timeout import func_timeout, FunctionTimedOut
import traceback
import functools # Recommended practice for decorators
import random
from eoh.invoker import AlgorithmInvoker



import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
from func_timeout import func_timeout, FunctionTimedOut
import traceback
import functools # Recommended practice for decorators
import random

DEFAULT_CLASS_NAME_HIERARCHY = ["Algorithm", "AlgorithmFIX1", "AlgorithmFIX2", "AlgorithmFIX3"]

def timed_code_evaluation(
    interface_eval, invoker: AlgorithmInvoker, timeout
):
    try:
        fitness_val, error_msg_eval = func_timeout(
            timeout,
            interface_eval.evaluate,
            args=(
                invoker,
            ),
        )
    except FunctionTimedOut:
        fitness_val = None
        error_msg_eval = f"Evaluation timed out for class '{invoker.class_to_instantiate_name}'."
    except Exception as e:
        fitness_val = None
        error_msg_eval = f"An unexpected error occurred during timed evaluation: {type(e).__name__}: {str(e)}"
    return fitness_val, error_msg_eval

def extract_python_code_from_llm_response(markdown_string):
    if not isinstance(markdown_string, str):
        return None
    match = re.search(r"```python\n(.*?)\n```", markdown_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "```" not in markdown_string and (
        markdown_string.strip().startswith("class ") or
        markdown_string.strip().startswith("def ") or
        markdown_string.strip().startswith("import ")
        ):
        return markdown_string.strip()
    return None

def process_code_submission_with_llm_fixing(
    initial_prompt,
    initial_code,
    base_class_code_str,
    interface_eval,
    call_llm_api,
    timeout
):
    llm_conversation_history = [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": initial_code},
    ]
    successful_parent_code_definitions = []
    final_successful_code_block = None
    class_name_hierarchy = DEFAULT_CLASS_NAME_HIERARCHY 
    max_attempts = len(class_name_hierarchy)
    code_for_current_attempt = initial_code
    final_fitness_achieved = None
    for attempt_idx in range(max_attempts):
        current_class_to_instantiate = class_name_hierarchy[attempt_idx]
        definitions_for_eval = successful_parent_code_definitions + [code_for_current_attempt]

        fitness_val, error_msg_eval = timed_code_evaluation(interface_eval,AlgorithmInvoker(base_class_code_str,definitions_for_eval,current_class_to_instantiate,interface_eval.prompts.prompt_func_name ), timeout)
        attempt_info = {"class_name": current_class_to_instantiate, "fitness": fitness_val}
        if fitness_val is None:
            attempt_info["error"] = error_msg_eval
            print(f"Attempt {attempt_idx + 1}: {error_msg_eval}")
        if fitness_val is not None:
            print(f"SUCCESS: Evaluation successful for {current_class_to_instantiate}! Fitness: {fitness_val}")
            final_successful_code_block = "\n".join(definitions_for_eval)
            final_fitness_achieved = fitness_val
            break
        else:
            if attempt_idx == max_attempts - 1:
                print("Max attempts reached. Could not fix the code for this submission.")
                break

            next_llm_class_name = class_name_hierarchy[attempt_idx + 1]
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
            messages_to_send_to_llm = llm_conversation_history + [current_fix_request_message]

            llm_generated_code_fix_raw = call_llm_api(messages_to_send_to_llm)
            llm_generated_code_fix = extract_python_code_from_llm_response(llm_generated_code_fix_raw)

            if llm_generated_code_fix:
                llm_conversation_history.append(current_fix_request_message)
                llm_conversation_history.append({"role": "assistant", "content": llm_generated_code_fix_raw})
                successful_parent_code_definitions.append(code_for_current_attempt)
                code_for_current_attempt = llm_generated_code_fix
            else:
                print("LLM did not return valid code. Stopping iterative fixing for this submission.")
                break
    
    status_message = "SUCCESS" if final_successful_code_block else "FAILED"
    print(f"\n--- Processing of code submission Finished: {status_message} ---")
    return current_class_to_instantiate, final_successful_code_block, final_fitness_achieved



def _standalone_get_alg(pop, operator, select, m, evol):
    """
    Standalone version of _get_alg. Generates code and algorithm name.

    Args:
        pop (list): Current population.
        operator (str): Evolutionary operator name ('i1', 'e1', etc.).
        select (object): Selection strategy object with parent_selection method.
        m (int): Number of parents needed for operators like 'e1', 'e2'.
        evol (object): Evolution object with methods like i1, e1, m1, etc.

    Returns:
        tuple: (parents, offspring_dict)
               parents: List of selected parents (or None).
               offspring_dict: Dictionary containing 'code' and 'algorithm'.
    """
    offspring = {
        'algorithm': None,
        'code': None,
        'prompt': None
        # These will be filled later
        # 'objective': None,
        # 'other_inf': None
    }
    parents = None # Initialize parents to None

    try:
        if operator == "i1":
            parents = None # Explicitly None for initialization
            offspring['prompt'], offspring['code'], offspring['algorithm'] = evol.i1()
        elif operator == "e1":
            parents = select.parent_selection(pop, m)
            if not parents: raise ValueError(f"Parent selection failed for e1 (m={m}, pop_size={len(pop)})")
            offspring['prompt'], offspring['code'], offspring['algorithm']  = evol.e1(parents)
        elif operator == "e2":
            parents = select.parent_selection(pop, m)
            if not parents: raise ValueError(f"Parent selection failed for e2 (m={m}, pop_size={len(pop)})")
            offspring['prompt'], offspring['code'], offspring['algorithm']  = evol.e2(parents)
        elif operator == "m1":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m1 (pop_size={len(pop)})")
            offspring['prompt'], offspring['code'], offspring['algorithm']  = evol.m1(parents[0])
        elif operator == "m2":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m2 (pop_size={len(pop)})")
            offspring['prompt'], offspring['code'], offspring['algorithm']  = evol.m2(parents[0])
        elif operator == "m3":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m3 (pop_size={len(pop)})")
            offspring['prompt'], offspring['code'], offspring['algorithm']  = evol.m3(parents[0])
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")
            # Return empty offspring if operator is invalid
            return parents, {'algorithm': None, 'code': None}

    except Exception as e:
        print(f"Error during _standalone_get_alg with operator {operator}: {e}")
        traceback.print_exc()
        # Return empty offspring on error
        return parents, {'algorithm': None, 'code': None}

    return parents, offspring



def get_offspring_standalone(pop, operator, select, m, evol, interface_eval, use_numba, debug, timeout):
    """
    Standalone function to generate and evaluate one offspring, designed for parallel execution.

    Args:
        pop (list): Current population.
        operator (str): Evolutionary operator name.
        select (object): Selection strategy object.
        m (int): Number of parents for crossover/evolution.
        evol (object): Evolution object.
        interface_eval (object): Evaluation object with an 'evaluate' method.
        use_numba (bool): Whether to apply Numba decorator.
        debug (bool): Enable debug printing.
        timeout (float): Maximum time allowed for the entire process (generation + evaluation).

    Returns:
        tuple: (parents, offspring_dict)
               parents: List of parents used (or None).
               offspring_dict: Dictionary for the evaluated offspring (including 'objective')
                               or a default dict with None values on failure/timeout.
    """
    p = None
    final_offspring = {
        'algorithm': None, 'code': None, 'objective': None, 'other_inf': None
    }
    try:
        # --- Generation Part (Attempt 1) ---
        # Call the generation function once initially
        p, offspring_data = _standalone_get_alg(pop, operator, select, m, evol)

        # Check if initial generation produced valid data/code
        if offspring_data is None or offspring_data.get('code') is None:
            if debug: print(f"Initial generation failed for operator {operator}.")
            # If generation fails, trigger the exception handling like get_offspring
            raise ValueError("Initial generation failed to produce valid code.")

        generated_code = offspring_data['code'] # The original code from the first attempt

        code_to_evaluate = generated_code

        processed_code = code_to_evaluate # Default if Numba is off or fails
        if use_numba and code_to_evaluate:
            try:
                # Apply Numba decorator if requested
                pattern = r"def\s+(\w+)\s*\(.*\):"
                match = re.search(pattern, code_to_evaluate) # Search in the code that might be Numba'd
                if match:
                    function_name = match.group(1)
                    processed_code = add_numba_decorator(program=code_to_evaluate, function_name=function_name)
                else:
                    if debug: print("Warning: No function definition found for Numba decoration. Using original code.")
                    # processed_code remains code_to_evaluate
            except Exception as numba_e:
                 print(f"Error applying Numba: {numba_e}")
                 # If Numba application fails, trigger exception handling
                 # get_offspring's broad except would catch this.
                 raise # Re-raise the Numba error to be caught below

        class_name,generated_code,fitness = process_code_submission_with_llm_fixing(
            initial_prompt=offspring_data['prompt'],
            initial_code=processed_code,
            base_class_code_str=interface_eval.base_class_code,
            interface_eval=interface_eval,
            call_llm_api=evol.interface_llm.interface_llm.call_llm_api,
            timeout=timeout
        )

        # --- Success Case ---
        if fitness is not None: # Evaluation succeeded (could be 0)
            final_offspring['algorithm'] = offspring_data['algorithm']
            final_offspring['code'] = generated_code # Store original code
            final_offspring['objective'] = fitness
            final_offspring['class_name'] =class_name
            # Potentially store processed_code in 'other_inf' if needed
            # final_offspring['other_inf'] = {'processed_code': processed_code}
            if debug: print(f"Successfully generated and evaluated offspring (Obj: {fitness}).")
        return p, final_offspring # Success! Exit the loop and function.


    except FunctionTimedOut:
        if debug: print(f"get_offspring_standalone timed out after {timeout} seconds.")
        return p, final_offspring # Return default empty offspring on timeout
    except Exception as e:
        print(f"Unexpected error in get_offspring_standalone: {e}")
        traceback.print_exc()
        return p, final_offspring # Return default empty offspring on other errors



class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        
        n_create = 2
        
        population = []

        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1')
            for p in pop:
                population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def get_offspring(self, pop, operator):

        try:
            p, offspring = self._get_alg(pop, operator)
            
            if self.use_numba:
                
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])

                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                    
                if n_retry > 1:
                    break
                
                
            fitness = self.interface_eval.evaluate(code)
            offspring['objective'] = np.round(fitness, 5)
                # fitness = self.interface_eval.evaluate(code)

        except Exception as e:

            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Round the objective values
        return p, offspring
    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    def long_task(self, i, max_sleep):
        """
        A task function that simulates work and potential delays.
        """
        sleep_time = random.uniform(0.1, max_sleep)
        print(f"Task {i}: Starting, will sleep for {sleep_time:.2f} seconds.")
        # time.sleep will now be interrupted by the decorator if it exceeds the timeout
        time.sleep(sleep_time)
        print(f"Task {i}: Finished.")
        return f"Result from task {i}"
        
    def get_algorithm(self, pop, operator):
        results = []
        print(f"Generating {self.pop_size} offspring using operator '{operator}' with timeout {self.timeout}s each...")
        # Prepare arguments for the delayed calls
        args_list = []
        for _ in range(self.pop_size):
            args_list.append(
                {
                    'pop': pop,
                    'operator': operator,
                    'select': self.select,
                    'm': self.m,
                    'evol': self.evol,
                    'interface_eval': self.interface_eval,
                    'use_numba': self.use_numba,
                    'debug': self.debug,
                    'timeout': self.timeout # Pass the per-job timeout
                }
            )
        print("\n--- Results ---")
        for i, result in enumerate(results):
            print(f"Job {i}: {result}")
        try:
            results = Parallel(n_jobs=self.n_p)( # , prefer="threads"
                delayed(get_offspring_standalone)(**args) for args in args_list
            )
        except Exception as e:
            print(f"Error: {e}")
            print("Parallel time out .")
            

        # print(f"Generating {self.pop_size} offspring using operator '{operator}' with timeout {self.timeout}s each...")
        # # Prepare arguments for the delayed calls
        # args_list = []
        # for _ in range(self.pop_size):
        #     args_list.append(
        #     {
        #         'pop': pop,
        #         'operator': operator,
        #         'select': self.select,
        #         'm': self.m,
        #         'evol': self.evol,
        #         'interface_eval': self.interface_eval,
        #         'use_numba': self.use_numba,
        #         'debug': self.debug,
        #         'timeout': self.timeout # Pass the per-job timeout
        #     }
        #     )
        # results = []
        # for i, args in enumerate(args_list):
        #     result = get_offspring_standalone(**args)
        #     results.append(result)
        #     if self.debug:
        #         print(f"Job {i}: {result}")
        time.sleep(2)


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        # # save out_off to json file
        # with open("./offspring.json", "w") as f:
        #     import json
        #     json.dump(out_off, f, indent=4) 

        return out_p, out_off
    # def get_algorithm(self,pop,operator, pop_size, n_p):
        
    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
