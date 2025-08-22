import os
import numpy as np
from typing import List
from typing import Dict, List, Set, Tuple, Type
from utils import BaseAlgorithm
from utils import GetData, PatternCollector, PlanEvaluator, solve_set_partitioning_with_pulp
import pickle





class Algorithm0(BaseAlgorithm):
    """Replicates Algorithm0, using a GLOBAL search strategy."""
    def __init__(self, alpha=0.0, epsilon=1e-6, rng=None):
        super().__init__(alpha, epsilon, use_sequential_truck_search=False, rng=rng)

    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        volume_utilization = self._calculate_volume_utilization(item_dims, container_dims)
        item_priority = item.get('quantity', 1) 
        adjacency_score = self._calculate_adjacency_score(item_pos, item_dims, container_dims, placed_items)
        return (0.9 * volume_utilization + 0.05 * item_priority + 0.05 * adjacency_score)

    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        potential_positions = {(0.0, 0.0, 0.0)}
        placed_items = truck.get('placed_items', [])
        cl, cw, ch = container_dims
        l, w, h = item_dims

        for placed_item in placed_items:
            x, y, z = placed_item['x'], placed_item['y'], placed_item['z']
            length, width, height = placed_item['length'], placed_item['width'], placed_item['height']

            if x + length + l <= cl + self.epsilon: potential_positions.add((x + length, y, z))
            if y + width + w <= cw + self.epsilon: potential_positions.add((x, y + width, z))
            if z + height + h <= ch + self.epsilon: potential_positions.add((x, y, z + height))
        return list(potential_positions)

    def _calculate_volume_utilization(self, item_dims, container_dims):
        item_volume = item_dims[0] * item_dims[1] * item_dims[2]
        container_volume = container_dims[0] * container_dims[1] * container_dims[2]
        return item_volume / container_volume if container_volume > self.epsilon else 0.0

    def _calculate_adjacency_score(self, item_pos, item_dims, container_dims, placed_items):
        px, py, pz = item_pos; l, w, h = item_dims
        cl, cw, ch = container_dims
        adjacency = 0
        if abs(px) < self.epsilon: adjacency += 1
        if abs(py) < self.epsilon: adjacency += 1
        if abs(pz) < self.epsilon: adjacency += 1
        if abs((px + l) - cl) < self.epsilon: adjacency += 1
        if abs((py + w) - cw) < self.epsilon: adjacency += 1
        if abs((pz + h) - ch) < self.epsilon: adjacency += 1
        for pi in placed_items:
            x, y, z = pi['x'], pi['y'], pi['z']; length, width, height = pi['length'], pi['width'], pi['height']
            if abs((px + l) - x) < self.epsilon: adjacency += 1
            if abs((py + w) - y) < self.epsilon: adjacency += 1
            if abs((pz + h) - z) < self.epsilon: adjacency += 1
            if abs((x + length) - px) < self.epsilon: adjacency += 1
            if abs((y + width) - py) < self.epsilon: adjacency += 1
            if abs((z + height) - pz) < self.epsilon: adjacency += 1
        return adjacency


class Algorithm1(BaseAlgorithm):
    """Replicates Algorithm1, selecting the largest item and maximizing contact area."""
    def __init__(self, alpha=0.0, epsilon=1e-6):
        super().__init__(alpha, epsilon, use_sequential_truck_search=False)

    def _select_items_to_evaluate(self, unplaced_items: List[dict]) -> List[Tuple[int, dict]]:
        eligible_items = [(i, item['length'] * item['width'] * item['height'])
                          for i, item in enumerate(unplaced_items) if item.get('quantity', 0) > 0]
        if not eligible_items: return []
        best_item_index, _ = max(eligible_items, key=lambda x: x[1])
        return [(best_item_index, unplaced_items[best_item_index])]

    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        x, y, z = item_pos; il, iw, ih = item_dims
        cl, cw, ch = container_dims
        contact_area = 0.0
        if abs(x) < self.epsilon: contact_area += iw * ih
        if abs(y) < self.epsilon: contact_area += il * ih
        if abs(z) < self.epsilon: contact_area += il * iw
        if abs((x + il) - cl) < self.epsilon: contact_area += iw * ih
        if abs((y + iw) - cw) < self.epsilon: contact_area += il * ih
        if abs((z + ih) - ch) < self.epsilon: contact_area += il * iw
        for pi in placed_items:
            px, py, pz = pi['x'], pi['y'], pi['z']; pl, pw, ph = pi['length'], pi['width'], pi['height']
            if abs((x + il) - px) < self.epsilon or abs(x - (px + pl)) < self.epsilon:
                contact_area += max(0, min(y + iw, py + pw) - max(y, py)) * max(0, min(z + ih, pz + ph) - max(z, pz))
            if abs((y + iw) - py) < self.epsilon or abs(y - (py + pw)) < self.epsilon:
                contact_area += max(0, min(x + il, px + pl) - max(x, px)) * max(0, min(z + ih, pz + ph) - max(z, pz))
            if abs((z + ih) - pz) < self.epsilon or abs(z - (pz + ph)) < self.epsilon:
                contact_area += max(0, min(x + il, px + pl) - max(x, px)) * max(0, min(y + iw, py + pw) - max(y, py))
        return contact_area

    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        positions = [(0.0, 0.0, 0.0)]
        for pi in truck.get('placed_items', []):
            x, y, z = pi['x'], pi['y'], pi['z']
            l, w, h = pi['length'], pi['width'], pi['height']
            positions.extend([(x + l, y, z), (x, y + w, z), (x, y, z + h)])
        
        for pos in sorted(list(set(positions))):
            if self._is_valid_placement(pos, item_dims, container_dims, truck.get('placed_items', [])):
                return [pos]
        return []


class Algorithm2(BaseAlgorithm):
    """Replicates Algorithm2, using a SEQUENTIAL search to place largest items at the lowest height."""
    def __init__(self, alpha=0.0, epsilon=1e-6):
        super().__init__(alpha, epsilon, use_sequential_truck_search=True)

    def _select_items_to_evaluate(self, unplaced_items):
        best_item_index, max_volume = -1, -1
        for i, item_type in enumerate(unplaced_items):
            if item_type['quantity'] > 0:
                volume = item_type['length'] * item_type['width'] * item_type['height']
                if volume > max_volume:
                    max_volume, best_item_index = volume, i
        return [(best_item_index, unplaced_items[best_item_index])] if best_item_index != -1 else []

    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        final_z = item_pos[2] + item_dims[2]
        return 1e9 - final_z

    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        positions = []
        placed_items = truck.get('placed_items', [])
        
        if not placed_items:
            return [(0.0, 0.0, 0.0)]

        candidate_corners = set([(0.0, 0.0, 0.0)])
        for pi in placed_items:
            candidate_corners.add((pi['x'] + pi['length'], pi['y'], pi['z']))
            candidate_corners.add((pi['x'], pi['y'] + pi['width'], pi['z']))
            candidate_corners.add((pi['x'], pi['y'], pi['z'] + pi['height']))
        
        for pos in sorted(list(candidate_corners)):
            if self._is_valid_placement(pos, item_dims, container_dims, placed_items):
                 positions.append(pos)
        
        return positions



class AlgorithmBaseline(BaseAlgorithm):
    """Replicates Algorithm0, using a GLOBAL search strategy."""
    def __init__(self, alpha=0.0, epsilon=1e-6, rng=None):
        super().__init__(alpha, epsilon, use_sequential_truck_search=False, rng=rng)

    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        return 1


    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        potential_positions = {(0.0, 0.0, 0.0)}
        placed_items = truck.get('placed_items', [])
        cl, cw, ch = container_dims
        l, w, h = item_dims

        for placed_item in placed_items:
            x, y, z = placed_item['x'], placed_item['y'], placed_item['z']
            length, width, height = placed_item['length'], placed_item['width'], placed_item['height']

            if x + length + l <= cl + self.epsilon: potential_positions.add((x + length, y, z))
            if y + width + w <= cw + self.epsilon: potential_positions.add((x, y + width, z))
            if z + height + h <= ch + self.epsilon: potential_positions.add((x, y, z + height))
        return list(potential_positions)
    
if __name__ == "__main__":
    # --- 1. Setup ---

    DATA_PATH_PATTERN = "data/ssscsp/INSTANCES/Iva*.txt"
    # algorithms_to_test = {
    #     "test": Algorithm0(),
    # }

    # # Initialize and run the evaluator
    # # Using 2 instances for a quick test run
    # evaluator = PlanEvaluator(n_instances=20, path_pattern=DATA_PATH_PATTERN)
    # evaluator.run_evaluation(algorithms_to_test)



    # algorithm_to_randomize = AlgorithmBaseline
    # for seed in range(1, 3):
    #     for alpha in [1.0]:
    #         algorithm_parameters = {"alpha": alpha, "epsilon": 1e-6}
    #         NUM_PARALLEL_RUNS = 10000

    #         NUM_INSTANCES_PER_RUN =47
    #         # --- Execution ---
    #         collector = PatternCollector(n_instances=NUM_INSTANCES_PER_RUN, path_pattern=DATA_PATH_PATTERN, seed=seed*NUM_PARALLEL_RUNS)
            
    #         # Run the parallel collection process
    #         results = collector.run_and_collect_all(
    #             algorithm_class=algorithm_to_randomize,
    #             algorithm_params=algorithm_parameters,
    #             num_runs=NUM_PARALLEL_RUNS,
    #             max_workers=100
    #         )
    #         #save results to a pickle
    #         save_dir = f"pattern/seed{seed}/"
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         with open(f"{save_dir}/random.pkl", "wb") as f:
    #             pickle.dump(results, f)


    algorithm_to_randomize = Algorithm0
    for seed in range(3):
        for alpha in [0.05, 0.15]:
            algorithm_parameters = {"alpha": alpha, "epsilon": 1e-6}
            if alpha == 0.0:
                NUM_PARALLEL_RUNS = 1
            else:
                NUM_PARALLEL_RUNS = 500

            NUM_INSTANCES_PER_RUN =47
            # --- Execution ---
            collector = PatternCollector(n_instances=NUM_INSTANCES_PER_RUN, path_pattern=DATA_PATH_PATTERN, seed=seed*NUM_PARALLEL_RUNS)
            
            # Run the parallel collection process
            results = collector.run_and_collect_all(
                algorithm_class=algorithm_to_randomize,
                algorithm_params=algorithm_parameters,
                num_runs=NUM_PARALLEL_RUNS,
                max_workers=100
            )
            #save results to a pickle
            save_dir = f"pattern/seed{seed}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(f"{save_dir}/llm_alpha_{alpha}.pkl", "wb") as f:
                pickle.dump(results, f)

    # # We need the original instances again to get the item demands
    # data_generator = GetData(n_instance=NUM_INSTANCES_PER_RUN)
    # instances = data_generator.generate_instances(DATA_PATH_PATTERN)
    # all_collected_patterns = collector.get_all_patterns_by_instance()
    # all_truck= []
    # for instance_id, (instance_items, _) in enumerate(instances):
    #     print(f"\n--- Instance {instance_id} ---")

    #     # Get patterns collected for this specific instance
    #     patterns_for_instance = all_collected_patterns.get(instance_id)
    #     if not patterns_for_instance:
    #         print("No patterns were collected for this instance. Cannot solve.")
    #         continue

    #     # Get the total demand for each item in this instance
    #     item_demands = {item['item_id']: item['quantity'] for item in instance_items}
    #     print(f"Item Demands: {item_demands}")
    #     print(f"Number of unique patterns available: {len(patterns_for_instance)}")

    #     # Solve the Set Partitioning problem using the collected patterns
    #     solution = solve_set_partitioning_with_pulp(patterns_for_instance, item_demands)

    #     # Report the results
    #     if solution and solution['status'] == 'Optimal':
    #         print(f"\nSolver Status: {solution['status']}")
    #         print(f"Optimal Number of Trucks: {solution['total_trucks']}")
    #         all_truck.append(solution['total_trucks'])
    #         print("Optimal Pattern Usage:")
    #         for pattern_fs, count in solution['pattern_usage'].items():
    #             # Convert frozenset to a dict for readable printing
    #             pattern_dict = dict(pattern_fs)
    #             print(f"  - Use {count} time(s): {pattern_dict}")
    #     elif solution:
    #         print(f"\nSolver Status: {solution['status']}")
    #         print("Could not find an optimal solution. This likely means the collected")
    #         print("patterns are not sufficient to exactly meet the item demands.")
    #     else:
    #         print("\nAn error occurred during the solving process.")
    # print(sum(all_truck))
