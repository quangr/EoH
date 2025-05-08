import tyro
import multiprocessing
from typing import List, Optional
import sys
import dataclasses

# Use @dataclasses.dataclass decorator
@dataclasses.dataclass
class TyroParas:
    """
    Configuration parameters for the system.

    Parameters can be set via command-line arguments (e.g., --method eoh --ec-pop-size 10)
    or loaded from configuration files (tyro supports YAML, JSON, etc.).
    """

    #####################
    ### General settings  ###
    #####################
    method: str = 'eoh' # Method name (e.g., 'eoh', 'ael', 'ls', 'sa')
    problem: str = 'tsp_construct' # Problem name (e.g., 'tsp_construct', 'bp_online')
    selection: Optional[str] = None # Selection strategy (defaults based on method if None)
    management: Optional[str] = None # Population management strategy (defaults based on method if None)

    #####################
    ###  EC settings  ###
    #####################
    ec_pop_size: int = 5  # Number of algorithms in each population
    ec_n_pop: int = 5 # Number of populations
    ec_operators: Optional[List[str]] = None # Evolution operators (defaults based on method if None)
    ec_m: int = 2  # Number of parents for 'e1' and 'e2' operators
    ec_operator_weights: Optional[List[float]] = None  # Weights (probabilities) for operators (defaults to [1.0] if None)

    #####################
    ### LLM settings  ###
    #####################
    llm_use_local: bool = False  # Whether to use a local LLM model
    llm_local_url: Optional[str] = None  # URL for local LLM server
    llm_api_endpoint: Optional[str] = None # API endpoint for remote LLM
    llm_api_key: Optional[str] = None  # API key for remote LLM
    llm_model: Optional[str] = None  # Model type for remote LLM

    #####################
    ###  Exp settings  ###
    #####################
    exp_debug_mode: bool = False  # Enable debug mode
    exp_output_path: str = "./"  # Default folder for outputs
    exp_use_seed: bool = False # Use predefined seeds
    exp_seed_path: str = "./seeds/seeds.json" # Path to seed file
    exp_use_continue: bool = False # Continue from a previous run
    exp_continue_id: int = 0 # ID for continuation
    exp_continue_path: str = "./results/pops/population_generation_0.json" # Path to continuation file
    exp_n_proc: int = 1 # Number of processes for parallelism (-1 for all available CPUs)

    #####################
    ###  Evaluation settings  ###
    #####################
    eva_timeout: int = 120 # Timeout in seconds for evaluation
    eva_numba_decorator: bool = False # Use numba decorator for evaluation

    _problem_instance: object = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Called after the dataclass is initialized. Applies dependency logic
        and adjusts parameters based on other parameter values.
        """
        print("--- Applying post-initialization logic ---")
        # Call the internal setup methods in the desired order
        self._set_parallel()
        self._set_ec()
        self._set_evaluation() # This method should create _problem_instance
        print("--- Post-initialization complete ---")


    # --- Keep _set_parallel, _set_ec methods as before ---
    def _set_parallel(self):
        # ... (logic as before) ...
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1:
            print(f"> Setting number of processes to all available CPU cores: {num_processes}.")
            self.exp_n_proc = num_processes
        elif self.exp_n_proc <= 0 or self.exp_n_proc > num_processes:
             print(f"> Warning: Specified number of processes ({self.exp_n_proc}) is invalid or exceeds available ({num_processes}). Setting to {num_processes}.")
             self.exp_n_proc = num_processes

    def _set_ec(self):
        # ... (logic as before) ...
        if self.management is None:
            if self.method in ['ael','eoh']: self.management = 'pop_greedy'
            elif self.method == 'ls': self.management = 'ls_greedy'
            elif self.method == 'sa': self.management = 'ls_sa'
        if self.selection is None: self.selection = 'prob_rank'
        if self.ec_operators is None:
            if self.method == 'eoh': self.ec_operators  = ['e1','e2','m1','m2']
            elif self.method == 'ael': self.ec_operators  = ['crossover','mutation']
            elif self.method in ['ls','sa']: self.ec_operators  = ['m1']
        if self.ec_operators is not None:
            if self.ec_operator_weights is None:
                self.ec_operator_weights = [1.0 for _ in range(len(self.ec_operators))]
            elif len(self.ec_operators) != len(self.ec_operator_weights):
                print(f"> Warning! Lengths of ec_operator_weights ({len(self.ec_operator_weights)}) and ec_operator ({len(self.ec_operators)}) should be the same. Resetting weights.")
                self.ec_operator_weights = [1.0 for _ in range(len(self.ec_operators))]
        if self.method in ['ls','sa']:
            if self.ec_pop_size > 1:
                self.ec_pop_size = 1
                print(f"> Method '{self.method}' is single-point based, setting pop size to 1.")
            if self.exp_n_proc != 1:
                 print(f"> Method '{self.method}' is single-point based, setting number of processes to 1.")
                 self.exp_n_proc = 1


    def _set_evaluation(self):
        """Adjusts evaluation settings and creates problem instance based on the problem type string."""
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
            print("> Problem 'bp_online' detected, setting eva_timeout=20 and eva_numba_decorator=True.")
            # Add logic here if you need a specific object instance for bp_online
            # e.g., self._problem_instance = BinPackingProblem(...)
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 20
            print("> Problem 'tsp_construct' detected, setting eva_timeout=20.")
            # Add logic here if you need a specific object instance for tsp_construct
            # e.g., self._problem_instance = TSPProblem(...)
        # Add case for 'packing_const' problem
        elif self.problem == 'packing_const':
            print("> Problem 'packing_const' detected.")
            try:
                 # Import and instantiate the specific problem class
                 # Make sure PackingCONST is importable in the environment this runs in
                 from packing import PackingCONST # Example import
                 self._problem_instance = PackingCONST()
                 print("> Created PackingCONST instance and stored in _problem_instance.")
            except ImportError:
                 print("> Warning: Could not import PackingCONST. Problem instance will not be available.")
                 self._problem_instance = None # Ensure it's None if import fails
            except Exception as e:
                 print(f"> Warning: Error creating PackingCONST instance: {e}")
                 self._problem_instance = None # Ensure it's None if instantiation fails


    # --- Keep __repr__ method as before ---
    def __repr__(self) -> str:
       # ... (formatted string representation) ...
       data = dataclasses.asdict(self)
       output = "--- Parsed and Finalized Configuration ---\n"
       # Manually structure for clarity, similar to the original __repr__
       output += f"General:\n" + \
              f"  method: {self.method!r}\n" + \
              f"  problem: {self.problem!r}\n" + \
              f"  selection: {self.selection!r}\n" + \
              f"  management: {self.management!r}\n" + \
              f"EC:\n" + \
              f"  ec_pop_size: {self.ec_pop_size}\n" + \
              f"  ec_n_pop: {self.ec_n_pop}\n" + \
              f"  ec_operators: {self.ec_operators!r}\n" + \
              f"  ec_m: {self.ec_m}\n" + \
              f"  ec_operator_weights: {self.ec_operator_weights!r}\n" + \
              f"LLM:\n" + \
              f"  llm_use_local: {self.llm_use_local}\n" + \
              f"  llm_local_url: {self.llm_local_url!r}\n" + \
              f"  llm_api_endpoint: {self.llm_api_endpoint!r}\n" + \
              f"  llm_api_key: {'***' if self.llm_api_key else 'None'!r}\n" + \
              f"  llm_model: {self.llm_model!r}\n" + \
              f"Experiment:\n" + \
              f"  exp_debug_mode: {self.exp_debug_mode}\n" + \
              f"  exp_output_path: {self.exp_output_path!r}\n" + \
              f"  exp_use_seed: {self.exp_use_seed}\n" + \
              f"  exp_seed_path: {self.exp_seed_path!r}\n" + \
              f"  exp_use_continue: {self.exp_use_continue}\n" + \
              f"  exp_continue_id: {self.exp_continue_id}\n" + \
              f"  exp_continue_path: {self.exp_continue_path!r}\n" + \
              f"  exp_n_proc: {self.exp_n_proc}\n" + \
              f"Evaluation:\n" + \
              f"  eva_timeout: {self.eva_timeout}\n" + \
              f"  eva_numba_decorator: {self.eva_numba_decorator}\n" + \
              f"Problem Instance Available: {hasattr(self, '_problem_instance') and self._problem_instance is not None}\n" + \
              f"----------------------------------------"
       return output
