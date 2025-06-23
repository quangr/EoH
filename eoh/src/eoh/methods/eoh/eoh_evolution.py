import re
import time
from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompts = prompts
        self.method_name = self.prompts.get_func_name()
        self.method_inputs = self.prompts.get_func_inputs()
        self.method_outputs = self.prompts.get_func_outputs()

        if len(self.method_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.method_inputs)
        else:
            self.joined_inputs = "'" + self.method_inputs[0] + "'"

        if len(self.method_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.method_outputs)
        else:
            self.joined_outputs = "'" + self.method_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)


    def _get_common_class_instructions(self):
        return (
            "You will be implementing a class `Algorithm` that inherits from the following `BaseAlgorithm` class. "
            "You can use the helper methods provided in the base class.\n"
            "```python\n"
            + self.prompts.get_base_class_code() + "\n" # Access base_class_code here
            "```\n\n"
            "Now, your task is to implement a class named `Algorithm` that inherits from `BaseAlgorithm`.\n"
        )

    def _get_common_method_instructions(self):
        return (
            "This class must contain a method named `" + self.method_name + "`.\n\n"
            "**To improve modularity and allow for future targeted optimizations, break down the logic within `" + self.method_name + "` "
            "into several private helper methods within the `Algorithm` class (e.g., methods for selecting an item, "
            "finding valid positions, evaluating placements, selecting a new truck, etc.).**\n\n"
            "**Do not contain definition of base class in the output.**\n\n"
            "The `" + self.method_name + "` method should accept " + str(len(self.method_inputs)) + " input(s): "
            + self.joined_inputs + ". The method should return " + str(len(self.method_outputs)) + " output(s): "
            + self.joined_outputs + ".\n"
            + self.prompts.get_inout_inf() + "\n"
            + self.prompts.get_other_inf() + "\n"
            "Do not give additional explanations beyond the one-sentence algorithm description and the class code. Remove all comments before final output"
        )

    def get_prompt_i1(self):
        prompt_content = (
            self.prompts.get_task() + "\n\n"
            + self._get_common_class_instructions() +
            "First, describe your novel algorithm and main steps for the `" + self.method_name + "` method in one sentence. "
            "This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. "
            "(e.g., `# {This describes the algorithm using specific terms.}`)\n"
            "Next, implement the `Algorithm` class.\n"
            + self._get_common_method_instructions()
        )
        return prompt_content

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv += f"No.{i+1} algorithm description for `{self.method_name}` and the corresponding `{indivs[i]['class_name']}` class code are: \n{indivs[i]['algorithm']}\nCode:\n```python\n{indivs[i]['code']}\n```\n\n"

        prompt_content = (
            self.prompts.get_task() + "\n\n"
            + self._get_common_class_instructions() +
            "I have " + str(len(indivs)) + f" existing algorithms (for the `{self.method_name}` method) with their class codes as follows: \n"
            + prompt_indiv +
            f"Please help me create a new `Algorithm` class whose `{self.method_name}` method implements a novel algorithm that has a totally different form from the given ones. \n"
            f"First, describe your new algorithm for the `{self.method_name}` method in one sentence. "
            "This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. "
            "(e.g., `# {This describes the algorithm using specific terms.}`)\n"
            "Next, implement the new `Algorithm` class.\n"
            + self._get_common_method_instructions()
        )
        return prompt_content
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv += f"No.{i+1} algorithm description for `{self.method_name}` and the corresponding `{indivs[i]['class_name']}` class code are: \n: {indivs[i]['algorithm']}\nCode:\n```python\n{indivs[i]['code']}\n```\n\n"

        prompt_content = (
            self.prompts.get_task() + "\n\n"
            + self._get_common_class_instructions() +
            "I have " + str(len(indivs)) + f" existing algorithms (for the `{self.method_name}` method) with their class codes as follows: \n"
            + prompt_indiv +
            f"Please help me create a new `Algorithm` class whose `{self.method_name}` method implements a novel algorithm that has a totally different form from the given ones but can be motivated from them. \n"
            f"Firstly, identify the common backbone idea in the `{self.method_name}` methods of the provided algorithms. "
            f"Secondly, based on the backbone idea describe your new algorithm for `{self.method_name}` in one sentence. "
            "This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. "
            "(e.g., `# {This describes the algorithm using specific terms.}`)\n"
            "Thirdly, implement the new `Algorithm` class.\n"
            + self._get_common_method_instructions()
        )
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        prompt_content = (
            self.prompts.get_task() + "\n\n"
            + self._get_common_class_instructions() +
            f"I have one `{indiv1['class_name']}` class with its `{self.method_name}` algorithm description and code as follows.\n"
            f"Algorithm description for `{self.method_name}`: {indiv1['algorithm']}\n"
            f"Code of `{indiv1['class_name']}` class:\n```python\n{indiv1['code']}\n```\n\n"
            f"Please assist me in creating a new `Algorithm` class whose `{self.method_name}` method implements an algorithm that has a different form but can be a modified version of the algorithm provided. \n"
            f"First, describe your new algorithm for the `{self.method_name}` method in one sentence. "
            "This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. "
            "(e.g., `# {This describes the algorithm using specific terms.}`)\n"
            "Next, implement the new `Algorithm` class.\n"
            + self._get_common_method_instructions()
        )
        return prompt_content
    
    def get_prompt_m2(self,indiv1):
        prompt_content = (
            self.prompts.get_task() + "\n\n"
            + self._get_common_class_instructions() +
            f"I have one `{indiv1['class_name']}` class with its `{self.method_name}` algorithm description and code as follows.\n"
            f"Algorithm description for `{self.method_name}`: {indiv1['algorithm']}\n"
            f"Code of `{indiv1['class_name']}` class:\n```python\n{indiv1['code']}\n```\n\n"
            f"Please analyze the provided `{indiv1['class_name']}` class, particularly its `{self.method_name}` method and any helper methods it uses for scoring or decision-making. "
            "Identify key numerical parameters, constants, or weighting factors within this logic that influence its placement decisions (these act like a 'score function').\n"
            f"Then, assist me in creating a new `Algorithm` class by modifying these identified parameters or the way they are used to achieve a different placement behavior. \n"
            f"First, describe your new algorithm for the `{self.method_name}` method, highlighting the changes in parameter settings or scoring, in one sentence. "
            "This description must be a single Python comment line, starting with '# ', and the sentence itself must be enclosed in curly braces. "
            "(e.g., `# {This describes the algorithm using specific terms.}`)\n"
            "Next, implement the new `Algorithm` class with these modifications.\n"
            + self._get_common_method_instructions()
        )
        return prompt_content
    
    def get_prompt_m3(self,indiv1):
        prompt_content = (
            self.prompts.get_task() + "\n\n"
            f"You are working with a `Algorithm` class that inherits from `BaseAlgorithm` (definition provided below for context, but do not include it in your output).\n"
            "```python\n"
            + self.prompts.get_base_class_code() + "\n" # Access base_class_code here
            "```\n\n"
            f"Consider the following `{indiv1['class_name']}` implementation:\n"
            f"Algorithm description for its `{self.method_name}` method: {indiv1['algorithm']}\n"
            f"Code of `{indiv1['class_name']}` class:\n```python\n{indiv1['code']}\n```\n\n"
            "Your task is to revise this class to a `Algorithm` class:\n"
            f"1. Identify main components in the `{self.method_name}` method and its helper methods.\n"
            "2. Analyze if any components might be overfit to specific data patterns (e.g., from the 'Data Distribution Summary' in the task description if available, or general overfitting).\n"
            "3. Simplify these components to enhance generalization to potential out-of-distribution instances, while ensuring all constraints are still met.\n\n"
            "First, describe the main simplifications or changes you made to improve generalization in one sentence. "
            "This description must be inside a brace {}.\n"
            "Next, provide the revised new `Algorithm` class code. "
            f"Keep the class name `Algorithm`, its inheritance from `BaseAlgorithm`, and the `{self.method_name}` method signature (`{self.method_name}` taking "
            + self.joined_inputs + " and returning " + self.joined_outputs + ") unchanged.\n\n"
            "**Do not contain definition of base class in the output.**\n\n"
            + self.prompts.get_inout_inf() + "\n"
            + self.prompts.get_other_inf() + "\n"
            "Do not give additional explanations beyond the one-sentence description and the revised code."
        )
        return prompt_content



    def _get_alg(self,prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        algorithm_desc_list = re.findall(r"\{(.*?)\}", response, re.DOTALL)
        
        extracted_code_str = ""
        # Prioritize finding code within markdown blocks
        code_blocks = re.findall(r"```python\n([\s\S]*?)\n```", response)
        if code_blocks:
            for block in code_blocks:
                # Make sure it's the Algorithm and not the BaseAlgorithm
                if "class Algorithm(BaseAlgorithm):" in block and "class BaseAlgorithm:" not in block:
                    extracted_code_str = block.strip()
                    break
        
        if not extracted_code_str:
            # Fallback if no markdown block or class not in it
            # Search for the class definition, ensuring it's not just the base class
            # This regex tries to find "class Algorithm(BaseAlgorithm):"
            # and then captures everything until a likely end of the class or end of text.
            # It avoids capturing if "class BaseAlgorithm:" immediately follows,
            # which might happen if the LLM includes both.
            pattern = r"class Algorithm\(BaseAlgorithm\):(?!\s*class BaseAlgorithm:)[\s\S]*?(?=\n\n\n|\Z|^\s*class\s+[A-Z]|\n\s*#\s*---|\n\s*\"\"\"Additional explanation)"
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                extracted_code_str = match.group(0).strip()
            else: # Simpler, more greedy fallback if the above is too restrictive
                match = re.search(r"class Algorithm\(BaseAlgorithm\):[\s\S]*", response)
                if match:
                    class_code_candidate = match.group(0)
                    # Attempt to remove common trailing explanations more robustly
                    # Find the end of the Algorithm class more carefully
                    # It might end with the last indented line, or before a new, unindented comment/text
                    
                    # Try to find where the class definition likely ends based on indentation or common stop patterns
                    end_match = re.search(r"(\n\s*\n\s*[^#\s]|\n# End of class|class BaseAlgorithm:)", class_code_candidate)
                    if end_match:
                        extracted_code_str = class_code_candidate[:end_match.start()].strip()
                    else:
                        # If no clear end, use heuristics for common explanation starters
                        stop_phrases = [
                            "\n\nDo not give additional explanations.", "\n\nNote:", "\n\nExplanation:",
                            "\n\nThis class", "\n\nThe `Algorithm`", "\n\n```" # End of a code block
                        ]
                        min_pos = len(class_code_candidate)
                        for phrase in stop_phrases:
                            pos = class_code_candidate.rfind(phrase)
                            if pos != -1:
                                # Ensure the phrase is not part of a docstring or comment within the class
                                # This is a heuristic, might need refinement
                                if class_code_candidate.count('\n', 0, pos) > class_code_candidate.count('\n', 0, class_code_candidate.rfind("class Algorithm")):
                                    min_pos = min(min_pos, pos)
                        extracted_code_str = class_code_candidate[:min_pos].strip()


        code_list = [extracted_code_str] if extracted_code_str and "class Algorithm(BaseAlgorithm):" in extracted_code_str else []


        n_retry = 0 
        max_retries = 3
        while (len(algorithm_desc_list) == 0 or not code_list or not code_list[0].strip()) and n_retry < max_retries:
            n_retry += 1
            if self.debug_mode:
                print(f"Error: algorithm description or code not identified (attempt {n_retry}/{max_retries}), retrying ... ")
            
            time.sleep(1) 
            response = self.interface_llm.get_response(prompt_content) 

            algorithm_desc_list = re.findall(r"\{(.*?)\}", response, re.DOTALL)
            
            extracted_code_str = ""
            code_blocks = re.findall(r"```python\n([\s\S]*?)\n```", response)
            if code_blocks:
                for block in code_blocks:
                    if "class Algorithm(BaseAlgorithm):" in block and "class BaseAlgorithm:" not in block:
                        extracted_code_str = block.strip()
                        break
            
            if not extracted_code_str:
                pattern = r"class Algorithm\(BaseAlgorithm\):(?!\s*class BaseAlgorithm:)[\s\S]*?(?=\n\n\n|\Z|^\s*class\s+[A-Z]|\n\s*#\s*---|\n\s*\"\"\"Additional explanation)"
                match = re.search(pattern, response, re.MULTILINE)
                if match:
                    extracted_code_str = match.group(0).strip()
                else:
                    match = re.search(r"class Algorithm\(BaseAlgorithm\):[\s\S]*", response)
                    if match:
                        class_code_candidate = match.group(0)
                        end_match = re.search(r"(\n\s*\n\s*[^#\s]|\n# End of class|class BaseAlgorithm:)", class_code_candidate)
                        if end_match:
                             extracted_code_str = class_code_candidate[:end_match.start()].strip()
                        else:
                            stop_phrases = ["\n\nDo not give additional explanations.", "\n\nNote:", "\n\nExplanation:", "\n\nThis class", "\n\nThe `Algorithm`", "\n\n```"]
                            min_pos = len(class_code_candidate)
                            for phrase in stop_phrases:
                                pos = class_code_candidate.rfind(phrase)
                                if pos != -1:
                                    if class_code_candidate.count('\n', 0, pos) > class_code_candidate.count('\n', 0, class_code_candidate.rfind("class Algorithm")):
                                        min_pos = min(min_pos, pos)
                            extracted_code_str = class_code_candidate[:min_pos].strip()

            code_list = [extracted_code_str] if extracted_code_str and "class Algorithm(BaseAlgorithm):" in extracted_code_str else []
            
        if len(algorithm_desc_list) == 0 or not code_list or not code_list[0].strip():
            final_algorithm_desc = "Error: Algorithm description not found." if not algorithm_desc_list else algorithm_desc_list[0].strip()
            final_code_content = "Error: Code content not found or invalid."
            
            if self.debug_mode:
                print("Failed to extract algorithm description or code after retries.")
                print("LLM Response was:\n", response)
                print(f"Algorithm found: {bool(algorithm_desc_list)}, Code found: {bool(code_list and code_list[0].strip())}")
                if code_list and code_list[0].strip():
                    print("Problematic code snippet:", code_list[0][:500]) # Print start of problematic code

            # Provide specific placeholders based on what's missing
            if not algorithm_desc_list and (code_list and code_list[0].strip()):
                final_algorithm_desc = "{Description missing, but code was found.}"
                final_code_content = code_list[0].strip() # Use the found code
            elif algorithm_desc_list and (not code_list or not code_list[0].strip()):
                final_code_content = f"# Code missing for algorithm: {algorithm_desc_list[0].strip()}\nclass Algorithm(BaseAlgorithm):\n    pass"
                final_algorithm_desc = algorithm_desc_list[0].strip() # Use the found description
            elif not algorithm_desc_list and (not code_list or not code_list[0].strip()):
                 raise ValueError("Failed to extract both algorithm description and `Algorithm` class code from LLM response after retries.")

            return [final_code_content, final_algorithm_desc]


        final_algorithm_desc = algorithm_desc_list[0].strip()
        final_code_content = code_list[0].strip() 

        # Final check to ensure the extracted code is indeed the Algorithm
        if "class Algorithm(BaseAlgorithm):" not in final_code_content:
            if self.debug_mode:
                print("Extracted code does not seem to be the Algorithm class.")
                print("Extracted code snippet:", final_code_content[:500])
            # This case should ideally be caught by the loop's condition or the error handling above
            # but as a last resort:
            raise ValueError("Extracted code is not the `Algorithm` class. LLM response format issue.")

        return [final_code_content, final_algorithm_desc]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
    
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return prompt_content, code_all, algorithm
