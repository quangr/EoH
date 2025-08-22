import collections
from typing import List
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from typing import Dict, List, Set, Tuple, Type
from typing import List, Dict, Tuple, Optional, Any

# --- This BaseAlgorithm remains unchanged ---
from utils import GetData
import os
import pickle

from utils import GetData, PatternCollector, BasePlanAlgorithm, solve_set_partitioning_with_pulp, PlanEvaluator
class S_GRASP(BasePlanAlgorithm):
    """
    Implements the Sequential Application of GRASP (S-GRASP) heuristic.
    """
    def __init__(self, delta: float = 0.08, rng = None):
        self.delta = delta
        # Use provided numpy Generator or create a new one for reproducibility
        self.rng = rng if rng is not None else np.random.default_rng()

    def pack_all(self, items_to_pack: List[Dict], container_dims: Tuple) -> Tuple[List[Dict], List[Dict]]:
        remaining_items = self._flatten_items(items_to_pack)
        trucks_in_use = []

        while remaining_items:
            packed_items, new_remaining_items = self._pack_one_container_grasp(
                remaining_items, container_dims
            )
            if not packed_items:
                break
            trucks_in_use.append({'placed_items': packed_items})
            remaining_items = new_remaining_items

        unplaced_items_summary = self._summarize_items(remaining_items)
        return trucks_in_use, unplaced_items_summary

    def _pack_one_container_grasp(self, items: List[Dict], container_dims: Tuple) -> Tuple[List[Dict], List[Dict]]:
        placed_items = []
        L, W, H = container_dims
        residual_spaces = [(0, 0, 0, L, W, H)]
        
        current_items = deepcopy(items)

        while True:
            candidate_walls = self._generate_all_candidate_walls(current_items, residual_spaces)
            if not candidate_walls:
                break

            candidate_walls.sort(key=lambda x: x['eval_score'], reverse=True)
            
            rcl_size = max(1, int(len(candidate_walls) * self.delta))
            rcl = candidate_walls[:rcl_size]
            chosen_wall = self.rng.choice(rcl)
            
            for item in chosen_wall['items_in_wall']:
                placed_items.append(item)
            
            item_ids_to_remove = {item['unique_id'] for item in chosen_wall['items_in_wall']}
            current_items = [item for item in current_items if item['unique_id'] not in item_ids_to_remove]
            
            original_space_idx = chosen_wall['containing_space_idx']
            # Adjust index if necessary after popping
            sorted_indices = sorted([idx for idx, space in enumerate(residual_spaces) if idx != original_space_idx])
            residual_spaces = [residual_spaces[i] for i in sorted_indices]
            # Add the pre-calculated best resulting spaces
            residual_spaces.extend(chosen_wall['best_new_spaces'])

        return placed_items, current_items

    def _get_potential_partitions(self, original_space: Tuple, wall_footprint: Tuple) -> Tuple[List[Tuple], List[Tuple]]:
        ox, oy, oz, ol, ow, oh = original_space
        wx, wy, wz, wl, ww, wh = wall_footprint
        assert ox==wx and oy==wy and oz==wz, "Wall footprint must be aligned with the original space."
        # Partitioning method 1 (as in Fig 2, left)
        # This creates 3 disjoint spaces.
        spaces1 = []
        # c3: Space to the RIGHT of the wall 
        if ol - wl > 0: spaces1.append((wx + wl, wy, wz, ol - wl, ww, oh)) #C3
        # c2: Space IN FRONT of the wall 
        if ow - ww > 0: spaces1.append((wx, wy + ww, wz, ol, ow - ww, oh)) #C2
        # c1: Space ON TOP of the wall 
        if oh - wh > 0: spaces1.append((wx, wy, wz + wh, wl, ww, oh - wh)) #C1
        
        # Partitioning method 2 (as in Fig 2, right)
        # This also creates 3 disjoint spaces, but with a different cutting order.
        spaces2 = []
        # c4: Space to the RIGHT of the wall 
        if ol - wl > 0: spaces2.append((wx + wl, wy, wz, ol - wl, ow, oh)) #C4
        # c5: Space IN FRONT of the wall 
        if ow - ww > 0: spaces2.append((wx, wy + ww, wz, wl, ow - ww, oh)) #C5
        # c1: Space ON TOP of the wall 
        if oh - wh > 0: spaces2.append((wx, wy, wz + wh, wl, ww, oh - wh)) #C1

        return spaces1, spaces2

    def _calculate_rejected_volume(self, spaces: List[Tuple], remaining_items: List[Dict]) -> float:
        min_dims = self._get_min_dims_of_remaining(remaining_items)
        if not min_dims:
            return 0.0

        min_l, min_w, min_h = min_dims
        rejected_volume = 0
        for space in spaces:
            sl, sw, sh = space[3], space[4], space[5]
            # Check all 6 rotations of the smallest item
            can_fit = any([
                (min_l <= sl and min_w <= sw and min_h <= sh), (min_l <= sl and min_h <= sw and min_w <= sh),
                (min_w <= sl and min_l <= sw and min_h <= sh), (min_w <= sl and min_h <= sw and min_l <= sh),
                (min_h <= sl and min_l <= sw and min_w <= sh), (min_h <= sl and min_w <= sw and min_l <= sh)
            ])
            if not can_fit:
                rejected_volume += sl * sw * sh
        return rejected_volume

    def _generate_all_candidate_walls(self, items: List[Dict], residual_spaces: List[Tuple]) -> List[Dict]:
        candidates = []
        unique_item_types = {item['item_id']: item for item in items}.values()
        item_counts = self._count_items_by_type(items)

        # Create a temporary list of items without the ones we are building a wall with
        # to correctly calculate rejected space
        temp_remaining_items = deepcopy(items)

        for i, space in enumerate(residual_spaces):
            sx, sy, sz, sl, sw, sh = space
            original_space_volume = sl * sw * sh
            if original_space_volume < 1e-6: continue

            for item_type in unique_item_types:
                if item_counts.get(item_type['item_id'], 0) == 0: continue
                
                for orientation in self._get_orientations(item_type):
                    l, w, h = orientation
                    if l > sl or w > sw or h > sh: continue

                    n1 = min(item_counts[item_type['item_id']], int(sl // l))
                    if n1 == 0: continue
                    n2 = min(item_counts[item_type['item_id']] // n1, int(sw // w))
                    if n2 == 0: continue
                    
                    num_items_in_wall = n1 * n2
                    wall_volume = num_items_in_wall * l * w * h
                    
                    # Simulate placement to calculate eval score
                    wall_footprint = (sx, sy, sz, n1 * l, n2 * w, h)
                    
                    # Temporarily remove items used in the wall for rejected space calculation
                    items_in_wall_ids = [it['unique_id'] for it in temp_remaining_items if it['item_id'] == item_type['item_id']][:num_items_in_wall]
                    items_in_wall_ids = set(items_in_wall_ids)
                    simulated_remaining_items = [it for it in temp_remaining_items if it['unique_id'] not in items_in_wall_ids]
                    
                    partitions = self._get_potential_partitions(space, wall_footprint)
                    
                    # Choose the best partition based on the paper's first modification (Eq. 11)
                    fitness1 = self._calculate_partition_fitness(partitions[0], simulated_remaining_items)
                    fitness2 = self._calculate_partition_fitness(partitions[1], simulated_remaining_items)
                    best_new_spaces = partitions[0] if fitness1 >= fitness2 else partitions[1]

                    # Calculate rejected volume for the chosen partition
                    rejected_volume = self._calculate_rejected_volume(best_new_spaces, simulated_remaining_items)
                    
                    # Calculate final eval score based on Eq. (12)
                    eval_score = (wall_volume - rejected_volume) / original_space_volume
                    
                    # Build the list of item instances for the wall
                    wall_items = []
                    temp_items_of_type = [it for it in items if it['item_id'] == item_type['item_id']]
                    for k in range(num_items_in_wall):
                        item_instance = deepcopy(temp_items_of_type[k])
                        row, col = divmod(k, n1)
                        item_instance['position'] = (sx + col * l, sy + row * w, sz)
                        item_instance['length'], item_instance['width'], item_instance['height'] = l, w, h
                        wall_items.append(item_instance)

                    candidates.append({
                        'eval_score': eval_score,
                        'items_in_wall': wall_items,
                        'containing_space_idx': i,
                        'best_new_spaces': best_new_spaces # Store this for later
                    })
        return candidates

    def _update_residual_spaces(self, spaces: List[Tuple], placed_in_idx: int, wall_footprint: Tuple, remaining_items: List[Dict]) -> List[Tuple]:
        original_space = spaces.pop(placed_in_idx)
        ox, oy, oz, ol, ow, oh = original_space
        wx, wy, wz, wl, ww, wh = wall_footprint
        spaces1, spaces2 = [], []
        if oh - wh > 0: spaces1.append((wx, wy, wz + wh, wl, ww, oh - wh))
        if ow - ww > 0: spaces1.append((wx, wy + ww, wz, wl, ow - ww, oh))
        if ol - wl > 0: spaces1.append((wx + wl, wy, wz, ol - wl, ow, oh))
        if oh - wh > 0: spaces2.append((wx, wy, wz + wh, ol, ow, oh - wh))
        if ow - ww > 0: spaces2.append((wx, wy + ww, wz, ol, ow - ww, wh))
        if ol - wl > 0: spaces2.append((wx + wl, wy, wz, ol - wl, ow, oh))
        fitness1 = self._calculate_partition_fitness(spaces1, remaining_items)
        fitness2 = self._calculate_partition_fitness(spaces2, remaining_items)
        new_spaces = spaces1 if fitness1 >= fitness2 else spaces2
        return spaces + [s for s in new_spaces if s[3]>0 and s[4]>0 and s[5]>0]

    def _calculate_partition_fitness(self, spaces: List[Tuple], remaining_items: List[Dict]) -> float:
        if not spaces: return 0.0
        min_dims = self._get_min_dims_of_remaining(remaining_items)
        if not min_dims: return 1.0
        min_l, min_w, min_h = min_dims
        rejected_volume, useful_volumes = 0, []
        for space in spaces:
            sl, sw, sh = space[3], space[4], space[5]
            can_fit = ((min_l <= sl and min_w <= sw and min_h <= sh) or (min_l <= sl and min_h <= sw and min_w <= sh) or
                       (min_w <= sl and min_l <= sw and min_h <= sh) or (min_w <= sl and min_h <= sw and min_l <= sh) or
                       (min_h <= sl and min_l <= sw and min_w <= sh) or (min_h <= sl and min_w <= sw and min_l <= sh))
            vol = sl * sw * sh
            if can_fit: useful_volumes.append(vol)
            else: rejected_volume += vol
        largest_useful_vol = max(useful_volumes) if useful_volumes else 0
        if rejected_volume < 1e-6: return largest_useful_vol if largest_useful_vol > 0 else 1.0
        return largest_useful_vol / rejected_volume

    def _get_min_dims_of_remaining(self, items: List[Dict]) -> Optional[Tuple[int, int, int]]:
        if not items: return None
        smallest_item = min(items, key=lambda x: min(x['length'], x['width'], x['height']))
        return (smallest_item['length'], smallest_item['width'], smallest_item['height'])

    def _get_orientations(self, item: Dict) -> Set[Tuple[int, int, int]]:
        l, w, h = item['length'], item['width'], item['height']
        return {(l, w, h), (l, h, w), (w, l, h), (w, h, l), (h, l, w), (h, w, l)}

    def _flatten_items(self, items_with_qty: List[Dict]) -> List[Dict]:
        flat_list = []
        unique_counter = 0
        for item_type in items_with_qty:
            for _ in range(item_type['quantity']):
                new_item = item_type.copy(); del new_item['quantity']
                new_item['unique_id'] = f"{item_type['item_id']}_{unique_counter}"; unique_counter += 1
                flat_list.append(new_item)
        return flat_list

    def _summarize_items(self, flat_items: List[Dict]) -> List[Dict]:
        if not flat_items: return []
        summary = {}
        for item in flat_items:
            item_id = item['id']
            if item_id not in summary:
                summary[item_id] = item.copy(); summary[item_id]['quantity'] = 0
                summary[item_id].pop('unique_id', None); summary[item_id].pop('position', None)
            summary[item_id]['quantity'] += 1
        return list(summary.values())

    def _count_items_by_type(self, flat_items: List[Dict]) -> Dict[str, int]:
        counts = {}
        for item in flat_items:
            counts[item['item_id']] = counts.get(item['item_id'], 0) + 1
        return counts


if __name__ == "__main__":
    DATA_PATH_PATTERN = "data/ssscsp/INSTANCES/Iva*.txt"
    # Use a small number of instances for a quick demonstration
    # N_INSTANCES_TO_RUN = 47

    # algorithms_to_test = {
    #     "S-GRASP (delta=0.08)": S_GRASP(delta=0.0),
    # }

    # # Initialize and run the evaluator
    # # Using 2 instances for a quick test run
    # evaluator = PlanEvaluator(n_instances=20, path_pattern=DATA_PATH_PATTERN)
    # evaluator.run_evaluation(algorithms_to_test)


    algorithm_to_randomize = S_GRASP
    for seed in range(3):
        for delta in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            algorithm_parameters = {"delta": delta}
            if delta == 0:
                NUM_PARALLEL_RUNS = 1
            else:
                # Instantiate the collector
                NUM_PARALLEL_RUNS = 500

            NUM_INSTANCES_PER_RUN = 47
            # --- Execution ---
            collector = PatternCollector(n_instances=NUM_INSTANCES_PER_RUN, path_pattern=DATA_PATH_PATTERN, seed=seed*NUM_PARALLEL_RUNS)
            
            # Run the parallel collection process
            results = collector.run_and_collect_all(
                algorithm_class=algorithm_to_randomize,
                algorithm_params=algorithm_parameters,
                num_runs=NUM_PARALLEL_RUNS,
                max_workers=100
            )
            save_dir = f"pattern/seed{seed}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(f"{save_dir}/s_grasp_delta_{delta}.pkl", "wb") as f:
                pickle.dump(results, f)



    # # --- 1. Collect Patterns using the Heuristic ---
    # collector = PatternCollector(n_instances=N_INSTANCES_TO_RUN, path_pattern=DATA_PATH_PATTERN)
    
    # collected_patterns_by_instance = collector.run_and_collect(
    #     algorithm_class=S_GRASP, 
    #     algorithm_params={"delta": 0.08}, 
    #     num_runs=50,
    #     max_workers=24
    # )
    # all_truck = []
    # # --- 2. Solve the Set Partitioning Problem for each instance ---
    # if not collected_patterns_by_instance:
    #     print("No patterns were collected. Cannot solve set partitioning problem.")
    # else:
    #     print(f"\n{'='*20} SOLVING SET PARTITIONING PROBLEM {'='*20}")
    #     # We need the original instances to get the item demands
    #     data_loader = GetData(n_instance=N_INSTANCES_TO_RUN)
    #     instances = data_loader.generate_instances(DATA_PATH_PATTERN)

    #     for instance_id, patterns in collected_patterns_by_instance.items():
    #         print(f"\n--- Solving for Instance {instance_id+1} ---")
    #         print(f"  Using {len(patterns)} unique patterns found by the heuristic.")
            
    #         # Get the total demand for each item type for this instance
    #         instance_items, _ = instances[instance_id]
    #         item_demands = {item['item_id']: item['quantity'] for item in instance_items}
            
    #         # Solve the integer programming problem
    #         solution = solve_set_partitioning_with_pulp(patterns, item_demands)

    #         if not solution:
    #             print("  Solver failed to run.")
    #             continue

    #         print(f"  Solver Status: {solution['status']}")
    #         all_truck.append(solution['total_trucks'])
    #         if solution['status'] == 'Optimal':
    #             print(f"  Optimal number of trucks: {solution['total_trucks']}")
    #             print("  Pattern usage:")
    #             for i, (pattern, count) in enumerate(solution['pattern_usage'].items()):
    #                 # Make pattern printable
    #                 p_str = ", ".join([f"item_{k}:{v}" for k, v in sorted(list(pattern))])
    #                 print(f"    - Use Pattern #{i+1} ({p_str}) {count} time(s).")
    #         elif solution['status'] == 'Infeasible':
    #             print("  The problem is infeasible. This means the collected patterns are not sufficient")
    #             print("  to satisfy the exact demand for all items. Try running the collector for")
    #             print("  more iterations or with different heuristic parameters.")
    # print(sum(all_truck))
    
