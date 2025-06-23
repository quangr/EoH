import os # Import the os module to access environment variables
from eoh import eoh
from eoh.utils.getParas import Paras
from packing import PackingCONST
# Parameter initilization #
from eoh.utils.tyroParas import TyroParas
import tyro

# This block makes the script runnable via command line using tyro
if __name__ == "__main__":
    # --- Get API Key Securely ---
    # Read the API key from an environment variable
    # We use .get() to avoid errors if the variable isn't set,
    # returning None instead.
    # You might want to add an error check here to ensure the key is present.
    llm_api_key_value = os.environ.get("LLM_API_KEY")

    # Optional: Add a check to ensure the key was found
    if not llm_api_key_value:
        print("Error: LLM_API_KEY environment variable not set.")
    # ----------------------------

    programmatic_defaults = TyroParas(
        method="eoh",
        problem="packing_const",
        llm_api_endpoint="openrouter.ai",
        llm_api_key=llm_api_key_value, # Use the value from the environment variable        
        llm_model="google/gemini-2.0-flash-001",
        # llm_model="google/gemini-2.5-flash-preview-05-20",
        # llm_model="google/gemini-2.0-flash-lite-001",
        ec_pop_size=5, # number of samples in each population
        ec_n_pop=5,  # number of populations
        exp_n_proc=20,  # multi-core parallel
        exp_debug_mode=False,
        # Any other parameters you want to set as a default baseline
    )
    params = tyro.cli(TyroParas, default=programmatic_defaults)

    evolution = eoh.EVOL(params)

    evolution.run()