from packing2d import PackingCONST  # Still not directly used, but kept for consistency
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
import os
from google import genai  # Ensure google-generativeai is installed


def visualize_packing(
    occupied_areas,
    container_width,
    container_height,
    instance_index,
    placements,
    save_dir="plots",
):
    """Visualizes the packing, handling potential None values in placements."""
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    container = patches.Rectangle(
        (0, 0),
        container_width,
        container_height,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(container)

    for step_number, x, y, w, h, mistake in placements:
        if x is None or y is None or w is None or h is None:  # Handle None values
            continue  # Skip this placement if any coordinate is None

        rect_color = "blue"
        edge_color = "r" if mistake else "black"
        alpha = 1.0 if not mistake else 0.5

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1,
            edgecolor=edge_color,
            facecolor=rect_color,
            alpha=alpha,
        )
        ax.add_patch(rect)

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

    plt.title(f"Packing Visualization - Instance {instance_index}")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.gca().set_aspect("equal", adjustable="box")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"instance_{instance_index}.png"))
    plt.close(fig)



def llm_solve_instance(rectangles, container_width, chat):
    """
    Requests the LLM to solve the entire packing problem instance at once.
    """
    prompt = (
        "Solve the following 2D packing problem.  Provide a JSON array of placement decisions.  Do *not* provide any explanation, return *only* valid JSON.\n\n"
        f"Rectangles (each tuple is (width, height)): {rectangles}\n"
        f"Container width: {container_width}\n\n"
        "Return a JSON array where each element is a dictionary with the keys:\n"
        "  'rect_index': an integer representing the index (in the provided rectangles list) of the rectangle to place,\n"
        "  'x': the x coordinate of the placement,\n"
        "  'y': the y coordinate of the placement.\n"
        "The rectangles must be placed within the container and must not overlap.  Minimize the total height used."
    )

    try:
        response = chat.send_message(prompt)
        result_text = response.text.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        decisions = json.loads(result_text)
        return decisions

    except Exception as e:
        print("Error calling Gemini API or parsing JSON:", e)
        return []  # Return an empty list to indicate failure


class GetData:
    def __init__(self, n_instance, n_rectangles, container_width):
        self.n_instance = n_instance
        self.n_rectangles = n_rectangles
        self.container_width = container_width

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            rectangles = np.stack(
                (
                    np.random.randint(1, 10, self.n_rectangles),
                    np.random.randint(1, 10, self.n_rectangles),
                ),
                axis=1,
            ).tolist()
            instance_data.append((rectangles, self.container_width))
        return instance_data

    def calculate_used_height(self, occupied_areas):
        if not occupied_areas:
            return 0
        return max(y + h for _, y, _, h in occupied_areas)


def validate_solution(rectangles, container_width, decisions):
    """Validates the LLM's solution for overlaps and container bounds."""
    occupied_areas = []
    placements = []
    mistakes = 0

    # Sort decisions by rect_index to ensure correct order
    decisions.sort(key=lambda x: x.get('rect_index', float('inf')))

    for step, decision in enumerate(decisions):
        rect_index = decision.get("rect_index")
        x = decision.get("x")
        y = decision.get("y")
        mistake = False

        if rect_index is None or x is None or y is None:
            print(f"Step {step+1}: Incomplete decision.")
            mistake = True
            mistakes += 1
            placements.append((step + 1, x, y, None, None, mistake)) #Append None for width and height
            continue

        if not 0 <= rect_index < len(rectangles):
            print(f"Step {step+1}: Invalid rectangle index.")
            mistake = True
            mistakes += 1
            placements.append((step + 1, x, y, None, None, mistake))  # Append None for w and h
            continue

        rect_width, rect_height = rectangles[rect_index]

        if x + rect_width > container_width:
            print(f"Step {step+1}: Rectangle exceeds container width.")
            mistake = True
            mistakes += 1

        overlap = False
        for ox, oy, ow, oh in occupied_areas:
            if (
                x < ox + ow
                and x + rect_width > ox
                and y < oy + oh
                and y + rect_height > oy
            ):
                overlap = True
                break
        if overlap:
            print(f"Step {step+1}: Overlap detected.")
            mistake = True
            mistakes += 1

        placements.append((step + 1, x, y, rect_width, rect_height, mistake))
        if not mistake:
          occupied_areas.append((x, y, rect_width, rect_height))


    return occupied_areas, placements, mistakes


def solve_and_evaluate(problem, client):
    """Solves each instance, validates the solution, visualizes, and calculates the average height."""
    total_heights = []
    all_occupied_areas = []

    for instance_index, (rectangles, container_width) in enumerate(problem.instance_data):
        chat = client.chats.create(model="gemini-2.0-flash")
        decisions = llm_solve_instance(rectangles, container_width, chat)

        if not decisions:  # If LLM call failed
            print(f"Failed to get a solution for instance {instance_index}")
            continue  # Skip to the next instance

        occupied_areas, placements, mistakes = validate_solution(
            rectangles, container_width, decisions
        )
        #Calculate container_height
        container_height_estimate = sum(h for _,h in rectangles) / container_width

        if mistakes > 0:
            print(f"Instance {instance_index}: Solution has {mistakes} mistakes.")
             # Visualize current state with mistakes
            used_height_in_mistake=0
            if placements:
                used_height_in_mistake = max((y + h) for _, x, y, w, h, _ in placements if x is not None and y is not None and w is not None and h is not None)

            visualize_packing(
                    occupied_areas,
                    container_width,
                    max(container_height_estimate, used_height_in_mistake),  # Use estimated height, and actual used height
                    instance_index,
                    placements,
                    save_dir="plots",
                )
            continue #Skip to the next instance


        total_height = problem.calculate_used_height(problem, occupied_areas)
        total_heights.append(total_height)
        all_occupied_areas.append(occupied_areas)
        visualize_packing(
            occupied_areas,
            container_width,
            total_height,
            instance_index,
            placements,
            save_dir="plots",
        )


    avg_height = np.mean(total_heights) if total_heights else None
    return avg_height, all_occupied_areas



# --- Main Execution ---
container_width = 10
instance_data = GetData(
    10, 20, container_width
).generate_instances()  # 10 instances, 20 rectangles
packing_problem = PackingCONST()
packing_problem.instance_data = instance_data
packing_problem.calculate_used_height = GetData.calculate_used_height

client = genai.Client( api_key="AIzaSyB6Blg32ziFNh-SNmZQvJhuIP1Ho55wQ-g")  # Replace with your API key

os.makedirs("plots", exist_ok=True)

fitness, all_occupied_areas = solve_and_evaluate(packing_problem, client)

if fitness is not None:
    print(f"Average container height: {fitness}")
    print("Plots saved to 'plots' directory")
else:
    print("Evaluation failed: No valid solutions were generated.")