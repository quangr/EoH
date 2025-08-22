import inspect
import textwrap

def get_standalone_class_source(target_class):
    """
    Attempts to generate a string representing the source code of a class
    as if all its Python-defined methods (including inherited ones not overridden)
    were defined directly within it, without the inheritance clause.

    Limitations:
    - Does not robustly handle reconstruction of class variables from parents.
    - Relies on inspect.getsource, so source files must be available.
    - Formatting and comments (outside methods) from parent classes are lost.
    - Order of methods might differ from original definition order across classes.
    - Won't work for built-ins or C extensions.
    """
    if not inspect.isclass(target_class):
        raise TypeError("Input must be a class.")

    output_parts = []

    # 1. Class definition line (no inheritance)
    output_parts.append(f"class {target_class.__name__}:")

    # 2. Class docstring (from the target class itself)
    class_doc = inspect.getdoc(target_class)
    has_content = bool(class_doc)

    if class_doc:
        # Format the docstring properly, indenting it
        doc_lines = class_doc.strip().splitlines()
        if not doc_lines: # Empty docstring
            pass
        elif len(doc_lines) == 1:
            formatted_doc = f'    """{doc_lines[0]}"""'
        else:
            formatted_doc = '    """{}\n'.format(doc_lines[0])
            for line in doc_lines[1:-1]:
                formatted_doc += f"    {line}\n"
            if len(doc_lines) > 1 : # Ensure last line is quoted if multi-line
                 formatted_doc += f'    {doc_lines[-1]}"""'
            else: # Single line doc was already handled
                 formatted_doc = f'    """{doc_lines[0]}"""' # Redundant but safe
        output_parts.append(formatted_doc)

    # 3. Methods
    # Collect all unique Python methods, using getattr to resolve overrides
    method_sources = {}  # name -> dedented source_string
    
    # Iterate through MRO to try and get a somewhat sensible order,
    # but mainly to find all methods. getattr will resolve overrides.
    # We process names to avoid duplicates if a method appears in multiple getmembers lists.
    processed_method_names = set()

    # We want to capture methods in an order that makes sense, often __init__ first.
    # A simple sort order: dunder methods, then regular methods.
    member_tuples = []
    for name, member_type in inspect.getmembers(target_class):
        member_tuples.append((name, member_type))

    # Sort for somewhat consistent order (dunders first, then alphabetical)
    member_tuples.sort(key=lambda item: (not (item[0].startswith("__") and item[0].endswith("__")), item[0]))

    for name, _ in member_tuples:
        if name in processed_method_names:
            continue
        processed_method_names.add(name)

        try:
            # Get the actual member object as it would be resolved on the class
            member = getattr(target_class, name)

            # We only care about Python functions/methods
            # (not, e.g., built-in methods of object unless overridden)
            if not (inspect.isfunction(member) or inspect.ismethod(member)):
                continue
            
            # Try to get the source code. This will get the source from where
            # it was *actually* defined (could be parent or child).
            source = inspect.getsource(member)
            method_sources[name] = textwrap.dedent(source)
            has_content = True

        except (TypeError, OSError):
            # Cannot get source (e.g., built-in, C extension, defined in REPL)
            # Or member is not a function/method (e.g. a slot wrapper that is callable)
            continue
        except AttributeError: # Should not occur if name is from getmembers
            continue
            
    # Add method sources, indented
    if method_sources:
        if class_doc: # Add a newline after docstring if there are methods
            output_parts.append("")

        # Sort method names again for consistent output
        sorted_names = sorted(method_sources.keys(), key=lambda k: (not (k.startswith("__") and k.endswith("__")), k))

        for i, name in enumerate(sorted_names):
            source = method_sources[name]
            # Indent the dedented source. Strip to remove existing surrounding newlines.
            indented_source = textwrap.indent(source.strip(), "    ")
            output_parts.append(indented_source)
            if i < len(sorted_names) - 1: # Add a blank line between methods
                output_parts.append("")

    # 4. Handle 'pass' if class body is effectively empty
    if not has_content:
        # If there was no docstring and no methods were added
        if len(output_parts) == 1: # Only "class Name:" is present
            output_parts.append("    pass")
        # If there was a docstring, it counts as content, 'pass' is not strictly needed.
        # If docstring formatting added a blank line, remove it if no methods follow.
        elif class_doc and not method_sources and output_parts[-1] == "":
            output_parts.pop()


    return "\n".join(output_parts)

# --- Example Usage ---
class GrandParent:
    """A grand parent class."""
    type = "mammal"
    def __init__(self, name):
        self.name = name
        self.age = 0

    def get_age(self):
        """Returns the age."""
        return self.age

    def common_method(self):
        return "From GrandParent"

class Parent(GrandParent):
    """A parent class."""
    def __init__(self, name, parent_attribute): # Overrides GrandParent.__init__
        super().__init__(name)
        self.parent_attribute = parent_attribute
        self.sound = "Parent Sound"

    def speak(self):
        """Parent speaks."""
        return f"{self.name} says {self.sound}"

    # get_age is inherited
    # common_method is inherited

class Child(Parent):
    """A child class, with its own methods and overrides."""
    # type is inherited from GrandParent
    # get_age is inherited from GrandParent

    def __init__(self, name, parent_attribute, child_feature): # Overrides Parent.__init__
        super().__init__(name, parent_attribute)
        self.child_feature = child_feature
        self.sound = "Child Sound" # Overrides Parent.sound

    def speak(self): # Overrides Parent.speak
        """Child speaks louder!"""
        return f"{self.name} the child shouts {self.sound.upper()}!"

    def play(self):
        """Child plays."""
        return f"{self.name} is playing with {self.child_feature}."

    def common_method(self): # Overrides GrandParent.common_method
        return "From Child"

# --- Test the function ---
print("--- Standalone source for Child ---")
child_standalone_source = get_standalone_class_source(Child)
print(child_standalone_source)
print("\n" + "="*50 + "\n")

print("--- Standalone source for Parent ---")
parent_standalone_source = get_standalone_class_source(Parent)
print(parent_standalone_source)
print("\n" + "="*50 + "\n")

class EmptyNoDoc:
    pass

print("--- Standalone source for EmptyNoDoc ---")
empty_standalone_source = get_standalone_class_source(EmptyNoDoc)
print(empty_standalone_source)
print("\n" + "="*50 + "\n")

class DocOnly:
    """Just a docstring."""

print("--- Standalone source for DocOnly ---")
doc_only_standalone_source = get_standalone_class_source(DocOnly)
print(doc_only_standalone_source)
print("\n" + "="*50 + "\n")

class InitOnly(object): # Explicitly inherit from object
    def __init__(self):
        self.x = 1

print("--- Standalone source for InitOnly ---")
init_only_source = get_standalone_class_source(InitOnly)
print(init_only_source)