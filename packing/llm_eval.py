# Keep imports and visualize_packing function as they are

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
from google import genai # Make sure you have the google-generativeai package installed
import os
import time # Import time for potential rate limiting


# --- visualize_packing function remains the same ---
def visualize_packing(
    occupied_areas,
    container_width,
    container_height,
    instance_index,
    placements,
    save_dir="plots",
):
    """Visualizes the packing of a single instance with step numbers and mistake highlighting."""
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height) # Use calculated or estimated height
    container = patches.Rectangle(
        (0, 0),
        container_width,
        container_height, # Use calculated or estimated height
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(container)

    max_y_coord = 0 # Keep track of the max y reached for setting ylim

    for step_number, x, y, w, h, mistake in placements:
        # Use dummy values if placement failed early
        x = x if x is not None else 0
        y = y if y is not None else 0
        w = w if w is not None else 1
        h = h if h is not None else 1

        rect_color = "blue"
        edge_color = "r" if mistake else "black"
        line_width = 2 if mistake else 1
        alpha = 1.0 # Make mistakes clearly visible

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=line_width,
            edgecolor=edge_color,
            facecolor=rect_color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Add step number in the center of the rectangle
        center_x = x + w / 2
        center_y = y + h / 2
        ax.text(
            center_x,
            center_y,
            str(step_number),
            color="white",
            ha="center",
            va="center",
            fontsize=8,
        )
        max_y_coord = max(max_y_coord, y + h)

    # Adjust ylim based on actual placements, ensure non-zero height
    final_container_height = max(max_y_coord, container_height, 1) # Ensure at least 1
    ax.set_ylim(0, final_container_height)
    # Re-adjust container patch if height changed significantly
    container.set_height(final_container_height)


    plt.title(f"Packing Visualization - Instance {instance_index}")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.gca().set_aspect("equal", adjustable="box")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"instance_{instance_index}.png"))
    plt.close(fig)

# --- Modified llm_place_rectangle ---
def llm_place_rectangle(
    rectangles_available,
    occupied_areas,
    container_width,
    chat,
    previous_attempt=None,
    error_message=None,
    max_retries=2 # Allow retries within this function if needed (e.g., for JSON errors)
):
    """
    Uses Gemini API to decide where to place the next rectangle,
    handling feedback on constraint violations.
    """
    prompt = (
        "You are an expert in 2D bin packing. Given the following packing problem state, "
        "provide a JSON decision for the next rectangle placement. **Return only valid JSON.**\n\n"
        f"Available Rectangles (list of (width, height) tuples):\n{rectangles_available}\n\n"
        f"Already Placed Rectangles (list of (x, y, width, height) tuples):\n{occupied_areas}\n\n"
        f"Container width: {container_width}\n\n"
    )

    if error_message:
        prompt += (
            f"**Feedback on your previous attempt:** Your suggestion {previous_attempt} was invalid because: **{error_message}**. "
            "Please analyze the available space and rectangles again and provide a *different* and *valid* placement.\n\n"
        )
    else:
         prompt += "Choose the *index* of the *next available rectangle* to place and the (x, y) coordinates for its bottom-left corner.\n\n"


    prompt += (
        "**Constraints:**\n"
        "1. The chosen rectangle must be from the 'Available Rectangles' list.\n"
        "2. The placement (x, y) must be such that the entire rectangle (width, height) fits within the container width (0 <= x and x + width <= container_width).\n"
        "3. The placed rectangle must *not* overlap with any 'Already Placed Rectangles'.\n"
        "4. Place rectangles starting from y=0 and try to keep the total height minimized (though not strictly required for this step).\n\n"
        "**Output Format:** Return a single JSON object with the keys:\n"
        "  'rect_index': an integer index from the 'Available Rectangles' list (0-based).\n"
        "  'x': the integer x coordinate for the bottom-left corner.\n"
        "  'y': the integer y coordinate for the bottom-left corner.\n"
        "**Do not include any explanations, comments, or markdown formatting like ```json ... ```. Just the raw JSON.**"
    )

    for attempt in range(max_retries):
        try:
            # print(f"--- LLM Prompt (Attempt {attempt+1}) ---") # Debugging
            # print(prompt) # Debugging
            response = chat.send_message(prompt)
            result_text = response.text.strip()
            # print(f"--- LLM Response ---") # Debugging
            # print(result_text) # Debugging

            # Strict JSON parsing attempt
            try:
                # Find the start and end of the JSON object
                json_start = result_text.find('{')
                json_end = result_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    result_text = result_text[json_start:json_end+1]
                else:
                    raise ValueError("Could not find JSON object delimiters.")

                decision = json.loads(result_text)
                # Basic check for required keys
                if not all(k in decision for k in ['rect_index', 'x', 'y']):
                     raise ValueError("Missing required keys in JSON response.")
                # Check types
                if not isinstance(decision['rect_index'], int) or \
                   not isinstance(decision['x'], (int, float)) or \
                   not isinstance(decision['y'], (int, float)):
                    raise ValueError("Incorrect data types for keys in JSON response.")
                # Convert x, y to int if they are floats
                decision['x'] = int(decision['x'])
                decision['y'] = int(decision['y'])

                return decision # Success

            except (json.JSONDecodeError, ValueError) as json_e:
                print(f"Warning: LLM response parsing failed (Attempt {attempt+1}/{max_retries}): {json_e}. Response: '{result_text}'")
                if attempt < max_retries - 1:
                     # Provide feedback about the JSON format error for the next retry
                     prompt += f"\n\n**Correction:** Your previous response was not valid JSON or missed required keys/types. Please ensure you return *only* the JSON object described."
                     time.sleep(1) # Avoid hitting rate limits quickly
                else:
                    print("Error: LLM failed to provide valid JSON after multiple attempts.")
                    return {"rect_index": -1, "x": -1, "y": -1, "error": "Invalid JSON response"}


        except Exception as e:
            print(f"Error calling Gemini API (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # Wait longer after API errors
            else:
                 print("Error: LLM API call failed after multiple attempts.")
                 return {"rect_index": -1, "x": -1, "y": -1, "error": "API call failed"}

    # Fallback if max_retries reached without success
    return {"rect_index": -1, "x": -1, "y": -1, "error": "Max retries exceeded"}


# --- GetData class remains the same ---
class GetData:
    def __init__(self, n_instance, n_rectangles, container_width):
        self.n_instance = n_instance
        self.n_rectangles = n_rectangles
        self.container_width = container_width

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for i in range(self.n_instance):
             # Ensure unique rectangles slightly to avoid identical dimensions confusing LLM
            rects = set()
            while len(rects) < self.n_rectangles:
                w = np.random.randint(1, 10)
                h = np.random.randint(1, 10)
                rects.add((w,h))
            rectangles = list(rects)
            np.random.shuffle(rectangles) # Shuffle the order
            instance_data.append((rectangles, self.container_width))
        return instance_data

    def calculate_used_height(self, occupied_areas):
        if not occupied_areas:
            return 0
        # occupied_areas is now list of tuples (x, y, w, h)
        return max(y + h for x, y, w, h in occupied_areas) if occupied_areas else 0

# --- Modified greedy function ---
def greedy(problem, client):
    total_heights = []
    all_final_occupied_areas = []
    max_placement_retries = 5 # Max attempts for the LLM per rectangle placement decision

    # Loop through each instance
    for instance_index, (rectangles_orig, container_width) in enumerate(
        tqdm(problem.instance_data, desc="Processing Instances")
    ):
        print(f"\n--- Instance {instance_index} ---")
        occupied_areas = [] # Stores tuples (x, y, w, h)
        # Keep track of available rectangles with their original indices
        # List of tuples: (original_index, (width, height))
        available_rectangles_with_orig_idx = list(enumerate(rectangles_orig))

        # Estimate container height for visualization purposes if needed early
        # We calculate the actual height at the end based on placements
        est_container_height = sum(h for _, (_, h) in available_rectangles_with_orig_idx) / container_width * 1.5 # Heuristic factor

        # Initialize chat *per instance* to maintain context for that instance
        chat = client.chats.create(
            model="gemini-2.0-flash-lite"
        )
        step = 1
        placements = [] # Stores tuples (step, x, y, w, h, mistake_flag) for visualization
        instance_failed = False # Flag if LLM fails irrecoverably for this instance

        while available_rectangles_with_orig_idx and not instance_failed:
            print(f"Step {step}: {len(available_rectangles_with_orig_idx)} rectangles remaining.")
            current_available_rects = [r[1] for r in available_rectangles_with_orig_idx] # Just (w, h) tuples for the prompt

            feedback_error_message = None
            last_attempt_details = None
            llm_decision_valid = False

            for attempt in range(max_placement_retries):
                print(f"  Placement attempt {attempt + 1}/{max_placement_retries}")
                # Get LLM decision with potential feedback
                decision = llm_place_rectangle(
                    current_available_rects,
                    occupied_areas,
                    container_width,
                    chat,
                    previous_attempt=last_attempt_details,
                    error_message=feedback_error_message,
                )

                rect_idx_llm = decision.get("rect_index")
                x = decision.get("x")
                y = decision.get("y")
                llm_error = decision.get("error") # Check for errors from llm_place_rectangle

                # --- Validation ---
                validation_error = None
                placed_rect_details = None # Store (w, h) here if validation starts ok

                if llm_error:
                    validation_error = f"LLM internal error: {llm_error}"
                elif rect_idx_llm is None or x is None or y is None:
                    validation_error = "LLM decision is incomplete (missing keys)."
                elif not isinstance(rect_idx_llm, int) or not (0 <= rect_idx_llm < len(current_available_rects)):
                    validation_error = (f"Invalid 'rect_index' {rect_idx_llm}. "
                                        f"Index must be between 0 and {len(current_available_rects) - 1}.")
                elif not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
                    validation_error = f"Invalid coordinates (x={x}, y={y}). Must be non-negative integers."
                else:
                    # Index is valid relative to the *current* available list
                    original_list_index, (rect_width, rect_height) = available_rectangles_with_orig_idx[rect_idx_llm]
                    placed_rect_details = (rect_width, rect_height)

                    # 1. Check container bounds
                    if x + rect_width > container_width:
                        validation_error = (f"Rectangle (index={rect_idx_llm}, size={rect_width}x{rect_height}) at x={x},y={y} "
                                            f"exceeds container width ({container_width}).")
                    # We don't check height bound strictly, as container height is dynamic

                    # 2. Check for overlaps
                    else:
                        for ox, oy, ow, oh in occupied_areas:
                            # Basic AABB overlap check
                            if (x < ox + ow and x + rect_width > ox and
                                y < oy + oh and y + rect_height > oy):
                                validation_error = (f"Placement of rectangle (index={rect_idx_llm}, size={rect_width}x{rect_height}, position=({x},{y})) "
                                                    f"overlaps with existing rectangle at ({ox},{oy}) with size ({ow},{oh}).")
                                break # Found overlap

                # --- Process Validation Result ---
                if validation_error:
                    print(f"  Attempt {attempt + 1} failed validation: {validation_error}")
                    feedback_error_message = validation_error
                    # Record the LLM's suggestion for the feedback message
                    last_attempt_details = f"(rect_index={rect_idx_llm}, x={x}, y={y})"
                    if attempt == max_placement_retries - 1:
                        print(f"Instance {instance_index} failed: LLM could not provide valid placement after {max_placement_retries} attempts for step {step}.")
                        instance_failed = True
                        # Record the final failed attempt for visualization
                        placements.append(
                            (step, x, y,
                             placed_rect_details[0] if placed_rect_details else None,
                             placed_rect_details[1] if placed_rect_details else None,
                             True) # Mark as mistake
                        )
                    # Go to the next attempt in the inner loop
                else:
                    # VALID PLACEMENT!
                    print(f"  Attempt {attempt + 1} successful: Placing rectangle {original_list_index} "
                          f"({rect_width}x{rect_height}) at ({x},{y}).")
                    occupied_areas.append((x, y, rect_width, rect_height))
                    placements.append((step, x, y, rect_width, rect_height, False)) # Not a mistake

                    # Remove placed rectangle from available list
                    available_rectangles_with_orig_idx.pop(rect_idx_llm)
                    llm_decision_valid = True
                    break # Exit the inner retry loop, move to next step

            # End of retry loop for one placement
            if not llm_decision_valid and not instance_failed:
                 # This case should ideally be caught by the retry limit failure
                 print(f"Error: Exited retry loop unexpectedly for step {step}. Marking instance as failed.")
                 instance_failed = True

            if instance_failed:
                break # Exit the outer while loop (stop processing this instance)

            step += 1
        # End of while loop for an instance

        # --- Instance Finalization & Visualization ---
        final_height = problem.calculate_used_height(occupied_areas) # Use the method from the problem object
        if not instance_failed:
            print(f"Instance {instance_index} completed successfully. Final height: {final_height}")
            total_heights.append(final_height)
            all_final_occupied_areas.append(occupied_areas)
        else:
             print(f"Instance {instance_index} visualization includes the failed step.")
             # Use estimated height or max y from placements for visualization if failed
             max_y_vis = 0
             for _, px, py, pw, ph, _ in placements:
                 if py is not None and ph is not None:
                    max_y_vis = max(max_y_vis, py + ph)
             final_height = max(est_container_height, max_y_vis, 1) # Ensure non-zero height


        visualize_packing(
            occupied_areas,
            container_width,
            final_height, # Use calculated actual height or estimated height if failed
            instance_index,
            placements, # Pass the history including any final mistake
            save_dir="plots",
        )

    # --- Overall Results ---
    if total_heights: # Only average successful instances
       ave_height = np.average(total_heights)
       print(f"\nAverage height over {len(total_heights)} successful instances: {ave_height}")
    else:
       ave_height = float('inf') # Or None, or handle as error
       print("\nNo instances were completed successfully.")

    return ave_height, all_final_occupied_areas


# --- evaluate function remains similar ---
def evaluate(problem, client):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Directly call greedy which now uses the LLM decision function with feedback
            fitness, final_occupied_areas = greedy(problem, client)
            return fitness, final_occupied_areas
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None, None


# --- Main execution ---
container_width = 10
n_instances = 10 # Keep small for testing feedback
n_rectangles_per_instance = 20 # Keep small for testing feedback

print("Generating instances...")
instance_data_generator = GetData(
    n_instances, n_rectangles_per_instance, container_width
)
instance_data = instance_data_generator.generate_instances()

# Dummy class to hold instance data and calculation method
class PackingProblem:
     def __init__(self, instance_data):
          self.instance_data = instance_data
          # Assign the method directly
          self.calculate_used_height = instance_data_generator.calculate_used_height

     # Make calculate_used_height callable with self
     def calculate_used_height(self, occupied_areas):
         return GetData.calculate_used_height(self, occupied_areas)


packing_problem = PackingProblem(instance_data)

print("Configuring Gemini client...")
client = genai.Client(api_key="AIzaSyB6Blg32ziFNh-SNmZQvJhuIP1Ho55wQ-g")

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

print("Starting evaluation...")
fitness, all_occupied_areas = evaluate(packing_problem, client)

if fitness is not None:
    print(f"\nFinal Result: Average container height (successful instances): {fitness}")
    print(f"Plots saved to 'plots' directory")
else:
    print("\nEvaluation failed or no instances were successful.")