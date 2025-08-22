import glob
from itertools import combinations
import os
import re
import time
import numpy as np
from typing import List
from typing import Dict, List, Set, Tuple, Type
from utils import PatternCollector, solve_set_partitioning_with_pulp, PlanEvaluator
import pickle

from abc import ABC, abstractmethod



class BaseAlgorithm(ABC):
    """
    An abstract base class for packing algorithms using a GRASP-like approach.
    """
    def __init__(self, alpha: float = 0.0, epsilon: float = 1e-6, use_sequential_truck_search: bool = False, rng = None):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        self.epsilon = epsilon
        self._use_sequential_truck_search = use_sequential_truck_search
        # Use provided numpy Generator or create a new one for reproducibility
        self.rng = rng if rng is not None else np.random.default_rng()


    def _check_overlap_3d(self, item1_pos, item1_dims, item2_pos, item2_dims):
        x1, y1, z1 = item1_pos
        l1, w1, h1 = item1_dims
        x2, y2, z2 = item2_pos
        l2, w2, h2 = item2_dims
        return (x1 < x2 + l2 and x1 + l1 > x2 and
                y1 < y2 + w2 and y1 + w1 > y2 and
                z1 < z2 + h2 and z1 + h1 > z2)

    def _get_orientations(self, item_type):
        L, W, H = item_type['length'], item_type['width'], item_type['height']
        return [
            (0, (L, W, H)), (1, (L, H, W)), (2, (W, L, H)),
            (3, (W, H, L)), (4, (H, L, W)), (5, (H, W, L))
        ]

    def _is_within_container_bounds(self, item_pos, item_dims, container_dims):
        px, py, pz = item_pos
        pl, pw, ph = item_dims
        cl, cw, ch = container_dims
        return (px >= 0.0 - self.epsilon and px + pl <= cl + self.epsilon and
                py >= 0.0 - self.epsilon and py + pw <= cw + self.epsilon and
                pz >= 0.0 - self.epsilon and pz + ph <= ch + self.epsilon)

    def _check_stability(self, item_to_place_pos, item_to_place_dims, occupied_volumes):
        px, py, pz = item_to_place_pos
        pl, pw, _ = item_to_place_dims

        # Items on the floor are always stable.
        if abs(pz) < self.epsilon:
            return True

        required_area = self.alpha * pl * pw
        # If an item has no base area, it cannot be supported, but it's trivially stable.
        if required_area < self.epsilon:
             return True

        total_support_area = 0.0
        for placed_item in occupied_volumes:
            placed_item_pos = (placed_item['x'], placed_item['y'], placed_item['z'])
            placed_item_dims = (placed_item['length'], placed_item['width'], placed_item['height'])

            # Check if the placed_item is directly underneath the new item
            if abs((placed_item_pos[2] + placed_item_dims[2]) - pz) < self.epsilon:
                # Calculate the 2D overlap area in the XY plane
                overlap_x = max(0, min(px + pl, placed_item_pos[0] + placed_item_dims[0]) - max(px, placed_item_pos[0]))
                overlap_y = max(0, min(py + pw, placed_item_pos[1] + placed_item_dims[1]) - max(py, placed_item_pos[1]))
                total_support_area += overlap_x * overlap_y
        
        return total_support_area >= required_area - self.epsilon

    def _is_valid_placement(self, item_to_place_pos, item_to_place_dims, container_dims, occupied_volumes):
        '''
        Checks if placing an item at a given position is valid.
        This includes checking container bounds, overlap with other items, and load stability.

        Args:
            item_to_place_pos (tuple): The (x, y, z) position of the new item.
            item_to_place_dims (tuple): The (length, width, height) of the new item.
            container_dims (tuple): The (length, width, height) of the container.
            occupied_volumes (list): A list of dictionaries representing placed items.

        Returns:
            bool: True if the placement is valid, False otherwise.
        '''
        if not self._is_within_container_bounds(item_to_place_pos, item_to_place_dims, container_dims):
            return False

        for placed_item in occupied_volumes:
            placed_item_pos = (placed_item['x'], placed_item['y'], placed_item['z'])
            placed_item_dims = (placed_item['length'], placed_item['width'], placed_item['height'])
            if self._check_overlap_3d(item_to_place_pos, item_to_place_dims, placed_item_pos, placed_item_dims):
                return False
        
        if not self._check_stability(item_to_place_pos, item_to_place_dims, occupied_volumes):
            return False

        return True
    def place_item(self, unplaced_items: List[Dict], trucks_in_use: List[Dict], truck_type: Tuple[float, float, float]) -> Tuple:
        best_placement = self._find_best_placement(unplaced_items, trucks_in_use, truck_type)
        if best_placement:
            return (best_placement['truck_index'], best_placement['item_index'],
                    best_placement['x'], best_placement['y'], best_placement['z'],
                    best_placement['orientation_index'])
        else:
            return -1, -1, 0.0, 0.0, 0.0, 0
            
    def _has_unplaced_items(self, items: List[Dict]) -> bool:
        return any(item.get('quantity', 0) > 0 for item in items)
        
    def pack_all(self, items_to_pack: List[Dict], container_dims: Tuple[float, float, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Packs all given items into containers.

        Returns:
            Tuple[List[Dict], List[Dict]]:
                - A list of used trucks, each a dict containing its 'placed_items'.
                - A list of unplaced items with their remaining quantities.
        """
        unplaced_items = [item.copy() for item in items_to_pack]
        for item in unplaced_items:
            item['separation_set'] = int(item.get('item_id')) if int(item.get('item_id')) in [0, 1] else None
        trucks_in_use = []
        orientations_cache = {item['item_id']: self._get_orientations(item) for item in unplaced_items}

        while self._has_unplaced_items(unplaced_items):
            result = self.place_item(unplaced_items, trucks_in_use, container_dims)
            truck_idx, item_idx, x, y, z, orient_idx = result

            if item_idx == -1:
                break

            item_type = unplaced_items[item_idx]
            item_type['quantity'] -= 1
            _, item_dims = orientations_cache[item_type['item_id']][orient_idx]
            
            placed_item_data = {
                'item_id': item_type['item_id'],
                'x': x, 'y': y, 'z': z,
                'length': item_dims[0], 'width': item_dims[1], 'height': item_dims[2],
            }
            separation_set = item_type.get('separation_set')

            if truck_idx == -1:
                trucks_in_use.append({'placed_items': [placed_item_data], 'separation_set': separation_set})
            else:
                trucks_in_use[truck_idx]['placed_items'].append(placed_item_data)
                if trucks_in_use[truck_idx].get('separation_set') is None:
                    trucks_in_use[truck_idx]['separation_set'] = separation_set

        remaining_unplaced = [item for item in unplaced_items if item.get('quantity', 0) > 0]
        return trucks_in_use, remaining_unplaced

    def _find_best_placement(self, unplaced_items: List[Dict], trucks_in_use: List[Dict], truck_type: Tuple[float, float, float]) -> Dict:
        items_to_check = self._select_items_to_evaluate(unplaced_items, truck_type)
        if not items_to_check:
            return None

        if self._use_sequential_truck_search:
            placements_in_existing = self._run_search_on_containers(items_to_check, trucks_in_use, truck_type)
            if placements_in_existing:
                return self._select_from_candidates(placements_in_existing)
            
            new_truck_container = [{'placed_items': []}]
            placements_in_new = self._run_search_on_containers(items_to_check, new_truck_container, truck_type, is_new_truck_search=True)
            return self._select_from_candidates(placements_in_new)
        else:
            containers_to_check = trucks_in_use + [{'placed_items': []}]
            is_new_truck_search_flags = [False] * len(trucks_in_use) + [True]
            all_placements = self._run_search_on_containers(items_to_check, containers_to_check, truck_type, is_new_truck_search_flags)
            return self._select_from_candidates(all_placements)

    def _run_search_on_containers(self, items_to_check, containers, truck_type, is_new_truck_search=False) -> List[Dict]:
        all_valid_placements = []
        for item_index, item in items_to_check:
            for orientation_index, (_, item_dims) in enumerate(self._get_orientations(item)):
                for i, truck in enumerate(containers):
                    is_new = is_new_truck_search if isinstance(is_new_truck_search, bool) else is_new_truck_search[i]
                    placed_items_in_truck = truck.get('placed_items', [])
                    potential_positions = self._generate_potential_positions(truck, truck_type, item_dims, item, orientation_index)
                    
                    for pos in potential_positions:
                        if self._is_valid_placement(pos, item_dims, truck_type, placed_items_in_truck):
                            score = self._calculate_placement_score(
                                item_pos=pos, item_dims=item_dims, item=item,
                                container_dims=truck_type, placed_items=placed_items_in_truck
                            )
                            final_truck_index = -1 if is_new else i
                            all_valid_placements.append({
                                'truck_index': final_truck_index, 'item_index': item_index,
                                'x': pos[0], 'y': pos[1], 'z': pos[2],
                                'orientation_index': orientation_index, 'score': score
                            })
        return all_valid_placements
        
    def _select_from_candidates(self, candidates: List[Dict]) -> Dict:
        if not candidates:
            return None

        if self.alpha == 0.0:
            best_score = max(c['score'] for c in candidates)
            best_candidates = [c for c in candidates if abs(c['score'] - best_score) <= self.epsilon]
            return best_candidates[0]
        
        scores = [c['score'] for c in candidates]
        max_score, min_score = max(scores), min(scores)

        if abs(max_score - min_score) < self.epsilon:
            return self.rng.choice(candidates)
        
        threshold = max_score - self.alpha * (max_score - min_score)
        rcl = [c for c in candidates if c['score'] >= threshold - self.epsilon]
        
        if rcl:
            return self.rng.choice(rcl)
        else:
            # Fallback to choosing from the best candidates if RCL is empty
            best_candidates = [c for c in candidates if abs(c['score'] - max_score) < self.epsilon]
            return self.rng.choice(best_candidates)

    def _select_items_to_evaluate(self, unplaced_items, truck_type):
        return [(i, item) for i, item in enumerate(unplaced_items) if item.get('quantity', 0) > 0]
    
    @abstractmethod
    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        pass

    @abstractmethod
    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        pass


    def _get_orientations(self, item_type):
        L, W, H = item_type['length'], item_type['width'], item_type['height']
        return [(0, (L, W, H)), (1, (L, H, W)), (2, (W, L, H)),
                (3, (W, H, L)), (4, (H, L, W)), (5, (H, W, L))]

class Algorithm(BaseAlgorithm):
    def __init__(self, alpha = 0, epsilon = 1e-6, rng=None):
        super().__init__(alpha, epsilon, True, rng)
    """
    This algorithm prioritizes placing items that maximize the surface area 
    they occupy on the container floor, selecting the placement that best covers 
    the floor while maintaining stability.
    """

    def _check_separation_compatibility(self, item: Dict, truck: Dict) -> bool:
        """
        Checks if an item can be placed in a given truck based on separation sets.
        """
        item_sep_set = item.get('separation_set')
        truck_sep_set = truck.get('separation_set')

        # An item without a separation requirement can go into any truck.
        if item_sep_set is None:
            return True
        
        # A truck that hasn't been assigned a separation set can accept any item.
        if truck_sep_set is None:
            return True
        
        # If both have sets, they must match.
        return item_sep_set == truck_sep_set

    def _calculate_placement_score(self, item_pos: Tuple[float, float, float], 
                                 item_dims: Tuple[float, float, float], 
                                 item: Dict, 
                                 container_dims: Tuple[float, float, float], 
                                 placed_items: List[Dict]) -> float:
        """
        Calculates a score for a potential placement, combining fill ratio,
        volume utilization, height, and stability (placement on the floor).
        """
        x, y, z = item_pos
        l, w, h = item_dims
        cl, cw, ch = container_dims

        # Original placement score components
        volume_utilization = (l * w * h) / (cl * cw * ch)
        height_ratio = (z + h) / ch if ch > 0 else 1.0
        is_bottom = 1.0 if abs(z) < self.epsilon else 0.0

        # Fill ratio component (from original item selection logic)
        item_volume = l * w * h
        truck_volume = cl * cw * ch
        occupied_volume = sum(p['length'] * p['width'] * p['height'] for p in placed_items)
        available_volume = truck_volume - occupied_volume
        
        fill_ratio = 0.0
        if available_volume > self.epsilon:
            # Cap fill ratio at 1.0 for sane scoring; we don't want to over-value
            # items that are larger than the remaining space.
            fill_ratio = min(1.0, item_volume / available_volume)

        # Weighted combination of all factors. Weights are chosen to balance objectives,
        # giving high importance to the fill ratio to mimic the original's selection criteria.
        score = (
            fill_ratio * 0.4 +             # How well the item fills the remaining space
            volume_utilization * 0.2 +     # General contribution to truck volume usage
            (1.0 - height_ratio) * 0.2 +   # Encourages lower placement
            is_bottom * 0.2                # Bonus for floor stability
        )
        return score

    def _generate_potential_positions(self, truck: Dict, 
                                    container_dims: Tuple[float, float, float], 
                                    item_dims: Tuple[float, float, float], 
                                    item: Dict, 
                                    orientation_index: int):
        """
        Generates potential placement positions based on the "bottom-left" heuristic.
        Positions are the corners of already placed items. Yields positions one by one.
        """
        # First, ensure the item is compatible with the truck's separation group.
        if not self._check_separation_compatibility(item, truck):
            return

        placed_items = truck.get('placed_items', [])

        # Generate candidate coordinates from the corners of existing items.
        # The set automatically handles duplicates. Start with the origin (0,0,0).
        x_coords = {0.0}
        y_coords = {0.0}
        z_coords = {0.0}
        for p_item in placed_items:
            x_coords.add(p_item['x'] + p_item['length'])
            y_coords.add(p_item['y'] + p_item['width'])
            z_coords.add(p_item['z'] + p_item['height'])

        # Iterate through the sorted, unique coordinates to find valid bottom-left points.
        for z in sorted(list(z_coords)):
            for y in sorted(list(y_coords)):
                for x in sorted(list(x_coords)):
                    yield (x, y, z)


if __name__ == "__main__":
    # --- 1. Setup ---

    DATA_PATH_PATTERN = "data/ssscsp/INSTANCES/Iva*.txt"
    algorithms_to_test = {
        "test": Algorithm(0.0),
    }

    # # Initialize and run the evaluator
    # # Using 2 instances for a quick test run
    # evaluator = PlanEvaluator(n_instances=20, path_pattern=DATA_PATH_PATTERN)
    # evaluator.run_evaluation(algorithms_to_test)

    algorithm_to_randomize = Algorithm
    for seed in range(0, 3):
        for alpha in [0.00,0.01,0.05,0.15]:
            algorithm_parameters = {"alpha": alpha, "epsilon": 1e-6}
            if alpha == 0.0:
                NUM_PARALLEL_RUNS = 1
            else:
                NUM_PARALLEL_RUNS = 500
            NUM_INSTANCES_PER_RUN =47
            # --- Execution ---
            collector = PatternCollector(n_instances=NUM_INSTANCES_PER_RUN, path_pattern=DATA_PATH_PATTERN, seed=seed*NUM_PARALLEL_RUNS)
            
            results = collector.run_and_collect_all(
                algorithm_class=algorithm_to_randomize,
                algorithm_params=algorithm_parameters,
                num_runs=NUM_PARALLEL_RUNS,
                max_workers=100
            )
            save_dir = f"pattern/all/seed{seed}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(f"{save_dir}/llm_alpha_{alpha}.pkl", "wb") as f:
                pickle.dump(results, f)


    # algorithm_to_randomize = Algorithm
    # for seed in range(0, 3):
    #     for alpha in [0.0,0.1]:
    #         algorithm_parameters = {"alpha": alpha, "epsilon": 1e-6}
    #         if alpha == 0.0:
    #             NUM_PARALLEL_RUNS = 1
    #         else:
    #             NUM_PARALLEL_RUNS = 500
    #         NUM_INSTANCES_PER_RUN =47
    #         # --- Execution ---
    #         collector = PatternCollector(n_instances=NUM_INSTANCES_PER_RUN, path_pattern=DATA_PATH_PATTERN, seed=seed*NUM_PARALLEL_RUNS)
            
    #         results = collector.run_and_collect_all(
    #             algorithm_class=algorithm_to_randomize,
    #             algorithm_params=algorithm_parameters,
    #             num_runs=NUM_PARALLEL_RUNS,
    #             max_workers=100
    #         )
    #         save_dir = f"pattern/sep/seed{seed}/"
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         with open(f"{save_dir}/s_grasp_alpha_{alpha}.pkl", "wb") as f:
    #             pickle.dump(results, f)
    

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
