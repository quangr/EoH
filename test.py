from eoh import eoh
from eoh.utils.getParas import Paras
from packing_position import PackingCONST
# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = PackingCONST(),
                llm_api_endpoint = "generativelanguage.googleapis.com", # set your LLM endpoint
                llm_api_key = "AIzaSyB6Blg32ziFNh-SNmZQvJhuIP1Ho55wQ-g",   # set your LLM key
                llm_model = "gemini-2.0-flash-001",
                ec_pop_size = 10, # number of samples in each population
                ec_n_pop = 5,  # number of populations
                exp_n_proc = 10,  # multi-core parallel
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()