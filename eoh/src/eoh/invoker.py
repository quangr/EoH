
import sys
import time
import types
class AlgorithmInvoker:
    def __init__(self, base_code_str, definitions_to_exec_list, class_to_instantiate_name, prompt_func_name):
        self.base_code_str = base_code_str
        self.definitions_to_exec_list = definitions_to_exec_list
        self.class_to_instantiate_name = class_to_instantiate_name
        self.prompt_func_name = prompt_func_name

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
        return getattr(algorithm, self.prompt_func_name)
