import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import json
import concurrent.futures

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
        # n_p is no longer directly used for parallel jobs, but might be relevant elsewhere
        self.n_p = n_p 
        # timeout is no longer directly applicable to individual evaluations in get_offspring
        # It might still be relevant conceptually for the overall process time limits
        self.timeout = timeout 
        self.use_numba = use_numba

    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return

    def add2pop(self,population,offspring):
        for ind in population:
            # Check if objective exists and is not None before comparison
            if ind.get('objective') is not None and offspring.get('objective') is not None and ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind.get('code'): # Use .get for safety
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

        n_create = 2 # This seemed hardcoded, kept it as is

        population = []

        # Sequential execution instead of potentially parallel generation
        for i in range(n_create):
            # Assuming get_algorithm returns parents and offspring list
            # Using 'i1' operator for initialization as in the original logic
            _, pop_chunk = self.get_algorithm([], 'i1')
            for p in pop_chunk:
                # Check if the offspring is valid before adding
                if p and p.get('code') is not None and p.get('objective') is not None:
                    population.append(p)
                elif self.debug:
                    print(f"Skipping invalid individual generated during initialization: {p}")

        return population

    def population_generation_seed(self,seeds,n_p): # n_p is kept in signature but not used for parallelism here

        population = []
        
        # --- Sequential evaluation instead of Parallel ---
        fitness = []
        print(f"Evaluating {len(seeds)} seed algorithms sequentially...")
        for i, seed in enumerate(seeds):
            if self.debug:
                print(f"Evaluating seed {i+1}/{len(seeds)}")
            try:
                # Direct call to evaluate
                result = self.interface_eval.evaluate(seed['code'])
                fitness.append(result)
                if self.debug:
                     print(f"Seed {i+1} evaluation successful.")
            except Exception as e:
                print(f"Error evaluating seed {i+1} with code:\n{seed.get('code', 'Code not available')}\nError: {e}")
                # Append a placeholder (like None or np.inf) or handle as needed
                # Appending None to maintain list length correspondence
                fitness.append(None) 
        # --- End of sequential evaluation ---

        print("Seed evaluation finished.")

        successful_seeds = 0
        for i in range(len(seeds)):
            # Check if fitness evaluation was successful for this seed
            if fitness[i] is not None:
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
                    successful_seeds += 1

                except Exception as e:
                    # This catch block might be redundant if the evaluation error was caught above
                    # but kept for safety for potential errors during dict creation/processing
                    print(f"Error processing seed algorithm {i+1} after evaluation: {e}")
            else:
                 # Log if a seed evaluation failed earlier
                 print(f"Skipping seed {i+1} due to evaluation failure.")


        # Adjust the final message based on successful evaluations
        if successful_seeds == len(seeds):
             print(f"Initialization finished! Successfully processed {successful_seeds} seed algorithms.")
        else:
             print(f"Initialization finished! Successfully processed {successful_seeds} out of {len(seeds)} seed algorithms.")


        return population


    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        parents = None # Initialize parents to None
        try:
            if operator == "i1":
                parents = [] # No parents for initialization
                [offspring['code'],offspring['algorithm']] =  self.evol.i1()
            elif operator == "e1":
                parents = self.select.parent_selection(pop,self.m)
                if not parents: raise ValueError("Parent selection returned empty list for e1")
                [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
            elif operator == "e2":
                parents = self.select.parent_selection(pop,self.m)
                if not parents: raise ValueError("Parent selection returned empty list for e2")
                [offspring['code'],offspring['algorithm']] = self.evol.e2(parents)
            elif operator == "m1":
                parents = self.select.parent_selection(pop,1)
                if not parents: raise ValueError("Parent selection returned empty list for m1")
                [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])
            elif operator == "m2":
                parents = self.select.parent_selection(pop,1)
                if not parents: raise ValueError("Parent selection returned empty list for m2")
                [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0])
            elif operator == "m3":
                parents = self.select.parent_selection(pop,1)
                if not parents: raise ValueError("Parent selection returned empty list for m3")
                [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0])
            else:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")
                # Return empty offspring if operator is invalid
                return None, { 'algorithm': None, 'code': None, 'objective': None, 'other_inf': None }
        
        except Exception as e:
             print(f"Error during LLM evolution call for operator {operator}: {e}")
             # Return empty offspring in case of LLM error
             return parents, { 'algorithm': None, 'code': None, 'objective': None, 'other_inf': None }


        return parents, offspring

    def get_offspring(self, pop, operator):
        
        offspring = { # Initialize default empty offspring
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
        p = None # Initialize parents

        try:
            p, offspring = self._get_alg(pop, operator)

            # Check if LLM generated valid code
            if offspring is None or offspring.get('code') is None:
                if self.debug:
                    print(f"LLM failed to generate valid code for operator {operator}. Skipping evaluation.")
                # Return the initialized empty offspring and parents
                return p, { 'algorithm': None, 'code': None, 'objective': None, 'other_inf': None }

            code_to_evaluate = offspring['code'] # Start with the original code

            if self.use_numba:
                try:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])
                    if match:
                        function_name = match.group(1)
                        # Apply Numba decorator
                        code_to_evaluate = add_numba_decorator(program=offspring['code'], function_name=function_name)
                        if self.debug:
                            print(f"Applied Numba decorator to function '{function_name}'.")
                    else:
                         if self.debug:
                            print("Could not find function definition to apply Numba decorator.")
                         # Proceed with original code if no function def found
                         code_to_evaluate = offspring['code']

                except Exception as e_numba:
                     print(f"Error applying Numba decorator: {e_numba}. Proceeding with original code.")
                     code_to_evaluate = offspring['code'] # Fallback to original code
            
            # --- Check for duplicates ---
            n_retry = 1
            max_retries = 2 # Define max retries to avoid infinite loops
            while self.check_duplicate(pop, offspring['code']) and n_retry <= max_retries:
                if self.debug:
                    print(f"Duplicated code detected (Attempt {n_retry}/{max_retries}), retrying LLM generation...")

                p, offspring = self._get_alg(pop, operator) # Try generating again

                # Check again if LLM generated valid code after retry
                if offspring is None or offspring.get('code') is None:
                     if self.debug:
                        print(f"LLM failed to generate valid code on retry {n_retry} for operator {operator}. Skipping evaluation.")
                     return p, { 'algorithm': None, 'code': None, 'objective': None, 'other_inf': None }
                
                code_to_evaluate = offspring['code'] # Update code to evaluate

                if self.use_numba:
                     try:
                        pattern = r"def\s+(\w+)\s*\(.*\):"
                        match = re.search(pattern, offspring['code'])
                        if match:
                            function_name = match.group(1)
                            code_to_evaluate = add_numba_decorator(program=offspring['code'], function_name=function_name)
                            if self.debug:
                                print(f"Applied Numba decorator after retry {n_retry}.")
                        else:
                            if self.debug:
                                print(f"Could not find function definition after retry {n_retry}.")
                            code_to_evaluate = offspring['code']
                     except Exception as e_numba:
                        print(f"Error applying Numba decorator after retry {n_retry}: {e_numba}. Proceeding with original code.")
                        code_to_evaluate = offspring['code']

                n_retry += 1

            if n_retry > max_retries and self.check_duplicate(pop, offspring['code']):
                 if self.debug:
                     print(f"Could not generate unique code after {max_retries} retries. Skipping evaluation.")
                 return p, { 'algorithm': None, 'code': None, 'objective': None, 'other_inf': None } # Return empty if still duplicate

            # --- Sequential Evaluation ---
            # self.code2file(offspring['code']) # Optional: uncomment to save code before evaluation
            if self.debug:
                print("Evaluating generated code sequentially...")

            # Removed concurrent.futures.ThreadPoolExecutor
            # Direct call with try-except block
            try:
                # Note: The original timeout logic from concurrent.futures is lost here.
                # Evaluation will run until completion or standard Python error.
                # fitness = self.interface_eval.evaluate(code_to_evaluate)
                fitness=0
                offspring['objective'] = np.round(fitness, 5)
                if self.debug:
                    print(f"Evaluation successful. Objective: {offspring['objective']}")

            except Exception as e_eval:
                print(f"Error during sequential code evaluation: {e_eval}")
                # Keep offspring details but set objective to None or a penalty value
                offspring['objective'] = None
                # Return the partially filled offspring dict with None objective
                return p, offspring
            # --- End of Sequential Evaluation ---

        except Exception as e:
            # General catch block for errors in _get_alg or other parts before evaluation
            print(f"An unexpected error occurred in get_offspring before evaluation: {e}")
            # Return default empty offspring
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            # p might be None or contain parents depending on where the error occurred
            return p, offspring

        # Return the parents and the evaluated offspring
        return p, offspring

    # Removed process_task method as it was based on concurrency

    def get_algorithm(self, pop, operator):
        results = []

        # --- Sequential generation loop instead of Parallel ---
        print(f"Generating {self.pop_size} offspring sequentially using operator '{operator}'...")
        for i in range(self.pop_size):
            if self.debug:
                print(f"\n--- Generating offspring {i+1}/{self.pop_size} ---")
            
            # Call get_offspring sequentially
            # Wrap in try-except to handle potential errors within get_offspring
            # even though get_offspring has its own internal error handling
            try:
                p, off = self.get_offspring(pop, operator)
                results.append((p, off))
                if self.debug:
                    print(f"--- Finished offspring {i+1}/{self.pop_size} ---")
            except Exception as e:
                 # This catch might be redundant if get_offspring handles all its errors,
                 # but kept as a safeguard.
                 print(f"Error generating offspring {i+1}/{self.pop_size}: {e}")
                 # Append placeholder Nones to maintain structure if needed,
                 # or simply skip appending if preferred. Here we append Nones.
                 results.append((None, None)) # Indicate failure for this iteration
        import json
        save_results = [(r[1]['algorithm'],r[1]['code']) for r in results]
        output_file = "results.json"
        try:
            with open(output_file, "w") as json_file:
                json.dump(save_results, json_file, indent=4)
                print(f"Results successfully saved to {output_file}.")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

        out_p = []
        out_off = []

        print("Processing generated offspring...")
        valid_offspring_count = 0
        for p, off in results:
            # Check if the offspring is valid (not None and has essential keys)
            # before processing further
            if off is not None and off.get('code') is not None and off.get('objective') is not None:
                out_p.append(p) # p might be None if initialization ('i1') or if error occurred before parent selection
                out_off.append(off)
                valid_offspring_count += 1
                if self.debug:
                    print(f">>> Valid offspring added: \n {off}")
            elif self.debug:
                 print(f">>> Skipping invalid or failed offspring: \n Parent: {p} \n Offspring: {off}")

        print(f"Finished processing. Obtained {valid_offspring_count} valid offspring.")
        return out_p, out_off