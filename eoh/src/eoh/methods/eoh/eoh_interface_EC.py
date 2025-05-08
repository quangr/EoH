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


# Define the decorator
def timeout_decorator(timeout_seconds):
    """
    A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (float): The maximum time in seconds the decorated function
                                 is allowed to run before timing out.
    """
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Timeout seconds must be a positive number.")

    def decorator(func):
        # Use functools.wraps to preserve the original function's name, docstring, etc.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This wrapper function is what gets called when the decorated func is invoked
            try:
                # Call the original function (func) with the specified timeout
                # Pass along all arguments (*args, **kwargs)
                result = func_timeout(timeout_seconds, func, args=args, kwargs=kwargs)
                return result
            except FunctionTimedOut:
                # Handle the timeout case
                print(f"Task timed out after {timeout_seconds} seconds (args={args}, kwargs={kwargs})")
                # Return None or some other indicator for timeout
                return None # Matching the original task_with_timeout function's behavior
            except Exception as e:
                # Handle any other exceptions raised by the original function
                print(f"Task failed with an exception (args={args}, kwargs={kwargs}): {e}")
                traceback.print_exc()
                # Return an error indicator - match original behavior
                return f"Error: {e}"

        # The decorator returns the wrapper function
        return wrapper

    # The decorator factory returns the actual decorator
    return decorator
job_timeout = 1.5  # Timeout for each individual job in seconds
max_task_duration = 2.5 # Max sleep time for tasks

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


# Define the decorator
def timeout_decorator(timeout_seconds):
    """
    A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (float): The maximum time in seconds the decorated function
                                 is allowed to run before timing out.
    """
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("Timeout seconds must be a positive number.")

    def decorator(func):
        # Use functools.wraps to preserve the original function's name, docstring, etc.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This wrapper function is what gets called when the decorated func is invoked
            try:
                # Call the original function (func) with the specified timeout
                # Pass along all arguments (*args, **kwargs)
                result = func_timeout(timeout_seconds, func, args=args, kwargs=kwargs)
                return result
            except FunctionTimedOut:
                # Handle the timeout case
                print(f"Task timed out after {timeout_seconds} seconds (args={args}, kwargs={kwargs})")
                # Return None or some other indicator for timeout
                return None # Matching the original task_with_timeout function's behavior
            except Exception as e:
                # Handle any other exceptions raised by the original function
                print(f"Task failed with an exception (args={args}, kwargs={kwargs}): {e}")
                traceback.print_exc()
                # Return an error indicator - match original behavior
                return f"Error: {e}"

        # The decorator returns the wrapper function
        return wrapper

    # The decorator factory returns the actual decorator
    return decorator
job_timeout = 1.5  # Timeout for each individual job in seconds
max_task_duration = 2.5 # Max sleep time for tasks

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
        # These will be filled later
        # 'objective': None,
        # 'other_inf': None
    }
    parents = None # Initialize parents to None

    try:
        if operator == "i1":
            parents = None # Explicitly None for initialization
            offspring['code'], offspring['algorithm'] = evol.i1()
        elif operator == "e1":
            parents = select.parent_selection(pop, m)
            if not parents: raise ValueError(f"Parent selection failed for e1 (m={m}, pop_size={len(pop)})")
            offspring['code'], offspring['algorithm'] = evol.e1(parents)
        elif operator == "e2":
            parents = select.parent_selection(pop, m)
            if not parents: raise ValueError(f"Parent selection failed for e2 (m={m}, pop_size={len(pop)})")
            offspring['code'], offspring['algorithm'] = evol.e2(parents)
        elif operator == "m1":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m1 (pop_size={len(pop)})")
            offspring['code'], offspring['algorithm'] = evol.m1(parents[0])
        elif operator == "m2":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m2 (pop_size={len(pop)})")
            offspring['code'], offspring['algorithm'] = evol.m2(parents[0])
        elif operator == "m3":
            parents = select.parent_selection(pop, 1)
            if not parents: raise ValueError(f"Parent selection failed for m3 (pop_size={len(pop)})")
            offspring['code'], offspring['algorithm'] = evol.m3(parents[0])
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


def _check_duplicate_standalone(population, code):
    """Standalone check for duplicate code in the population."""
    if code is None: # Cannot check duplicates if code is None
        return False
    for ind in population:
        if code == ind.get('code'): # Use .get for safety
            return True
    return False

def _evaluate_offspring_timed(code, interface_eval, timeout):
    """Helper function to evaluate code with a specific timeout."""
    if code is None:
        print("Cannot evaluate None code.")
        return None # Or raise an error, depending on desired behavior

    try:
        # Use func_timeout directly here
        fitness = func_timeout(timeout, interface_eval.evaluate, args=(code,))
        return np.round(fitness, 5)
    except FunctionTimedOut:
        print(f"Evaluation timed out after {timeout} seconds")
        return None # Indicate timeout
    except Exception as e:
        print(f"Evaluation failed\nError: {e}")
        # traceback.print_exc() # Optional: print full traceback for eval errors
        return None # Indicate failure

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

        # --- Duplicate Check & Retry Part (Mimicking get_offspring's logic) ---
        is_duplicate = _check_duplicate_standalone(pop, generated_code)

        # get_offspring has a loop that runs IF duplicate is found, and breaks after 1 retry.
        # We can replicate this logic structure.
        n_retry = 0 # Counter for duplicate retries

        # Loop runs if currently a duplicate AND we haven't retried yet (max 1 retry)
        while is_duplicate and n_retry < 1: # This loop runs at most once
             if debug: print(f"Duplicated code found (Attempt {n_retry+1}). Retrying generation once...")

             # Attempt generation again (this is the single duplicate retry)
             p_retry, offspring_data_retry = _standalone_get_alg(pop, operator, select, m, evol)

             if offspring_data_retry is not None and offspring_data_retry.get('code') is not None:
                 # Update the main variables with the retry result regardless of its duplicate status
                 # This matches how get_offspring updates 'offspring' and 'p' in its retry loop
                 offspring_data = offspring_data_retry
                 p = p_retry
                 generated_code = offspring_data['code'] # Update the code reference to the new code
                 is_duplicate = _check_duplicate_standalone(pop, generated_code) # Check duplicate status of the new code
                 if debug: print(f"Retry {n_retry+1}: Generated new code. Is it duplicate? {is_duplicate}")
             else:
                # If the retry generation itself failed
                if debug: print(f"Retry {n_retry+1} generation failed to produce valid code.")
                # Trigger exception handling if the retry generation fails
                raise ValueError("Retry generation failed.")

             n_retry += 1 # Increment retry counter. Loop condition check will now be `n_retry < 1` (1 < 1 or 2 < 1) which will be false, ending the loop.


        # After the duplicate retry block:
        # - If the first attempt was not a duplicate, generated_code is the original code.
        # - If the first attempt was a duplicate, generated_code is the code from the single retry.
        # - is_duplicate holds the duplicate status of the *final* generated_code.
        # Note: Similar to get_offspring, we proceed even if the final code (from retry) is still a duplicate.
        # The responsibility to handle this might be outside this function (e.g., in the main GA loop).


        # --- Numba Application Part (Applied to the final generated code) ---
        # This code will be used for evaluation
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


            # --- Evaluation Part ---
        remaining_timeout = timeout
        if remaining_timeout <= 0:
                print("Timeout exceeded before evaluation.")
                raise FunctionTimedOut("Overall timeout exceeded before evaluation.")

        fitness = _evaluate_offspring_timed(processed_code, interface_eval, remaining_timeout)

        # --- Success Case ---
        if fitness is not None: # Evaluation succeeded (could be 0)
            final_offspring['algorithm'] = offspring_data['algorithm']
            final_offspring['code'] = generated_code # Store original code
            final_offspring['objective'] = fitness
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
    

    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

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
            
        time.sleep(2)


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
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
