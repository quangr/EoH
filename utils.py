import numpy as np
from typing import List
import glob
import re
import collections
import concurrent.futures
import time
from typing import Dict, List, Set, Tuple, Type
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
from itertools import combinations

class GetData:
    """
    Data handler for container loading problems, where each instance file defines
    its own unique container and a set of items. It correctly sorts instances
    numerically (e.g., Iva1, Iva2, ..., Iva10).
    """

    def __init__(self, n_instance):
        self.n_instance = n_instance
        # Store all unique item types found across all instances for summary stats
        self.all_item_types = []
        # Store all container types found across all instances for summary stats
        self.all_container_types = []

    def generate_instances(self, path_pattern="data/ssscsp/INSTANCES/Iva*.txt"):
        """
        Reads instance files, where each file specifies its own container and item list.
        Files are sorted in natural numerical order based on their name.

        Args:
            path_pattern (str): The glob pattern to find instance files.

        Returns:
            list: A list of tuples, where each tuple is (list_of_items, container_tuple).
                  Returns an empty list if no files are processed.
        """
        files = glob.glob(path_pattern)

        # --- Custom sorting key to sort files numerically ---
        def natural_sort_key(file_path):
            # Attempt to find one or more digits (\d+) in the filename
            match = re.search(r"(\d+)", file_path)
            if match:
                # If a number is found, return it as an integer for proper sorting
                return int(match.group(1))
            # If no number is found, return a large number to place it at the end
            return float("inf")

        # Sort files using the natural sort key
        files.sort(key=natural_sort_key)

        files_to_process = files[: self.n_instance]

        if not files_to_process:
            print(f"No files found matching the pattern: {path_pattern}")
            return []

        # Clear previous data before processing new files
        self.all_item_types = []
        self.all_container_types = []
        instance_data = []

        for file_path in files_to_process:
            try:
                with open(file_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                # First line contains the container dimensions for this specific instance
                container_dims = [int(d) for d in lines[0].split()]
                current_container_type = tuple(container_dims)
                self.all_container_types.append(current_container_type)

                # Lines from the 3rd onward contain the items
                item_lines = lines[2:]

                current_instance_item_types = []
                for i, line in enumerate(item_lines):
                    # Each item line: length width height quantity
                    parts = [int(p) for p in line.split()]
                    if len(parts) != 4:
                        continue

                    item = {
                        "item_id": str(i),
                        "length": parts[0],
                        "width": parts[1],
                        "height": parts[2],
                        "quantity": parts[3],
                    }
                    current_instance_item_types.append(item)
                    self.all_item_types.append(item)

                # Each instance data tuple contains its own items and its own container
                instance_data.append(
                    (current_instance_item_types, current_container_type)
                )

            except (IOError, IndexError, ValueError) as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        print(f"Successfully processed {len(instance_data)} instances.")
        return instance_data



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

            if truck_idx == -1:
                trucks_in_use.append({'placed_items': [placed_item_data]})
            else:
                trucks_in_use[truck_idx]['placed_items'].append(placed_item_data)
        
        remaining_unplaced = [item for item in unplaced_items if item.get('quantity', 0) > 0]
        return trucks_in_use, remaining_unplaced

    def _find_best_placement(self, unplaced_items: List[Dict], trucks_in_use: List[Dict], truck_type: Tuple[float, float, float]) -> Dict:
        items_to_check = self._select_items_to_evaluate(unplaced_items)
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
            best_candidates = [c for c in candidates if abs(c['score'] - best_score) < self.epsilon]
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

    def _select_items_to_evaluate(self, unplaced_items):
        return [(i, item) for i, item in enumerate(unplaced_items) if item.get('quantity', 0) > 0]
    
    @abstractmethod
    def _calculate_placement_score(self, item_pos, item_dims, item, container_dims, placed_items):
        pass

    @abstractmethod
    def _generate_potential_positions(self, truck, container_dims, item_dims, item, orientation_index):
        pass

    def _check_overlap_3d(self, item1_pos, item1_dims, item2_pos, item2_dims):
        x1, y1, z1 = item1_pos; l1, w1, h1 = item1_dims
        x2, y2, z2 = item2_pos; l2, w2, h2 = item2_dims
        return (x1 < x2 + l2 - self.epsilon and x1 + l1 > x2 + self.epsilon and
                y1 < y2 + w2 - self.epsilon and y1 + w1 > y2 + self.epsilon and
                z1 < z2 + h2 - self.epsilon and z1 + h1 > z2 + self.epsilon)

    def _get_orientations(self, item_type):
        L, W, H = item_type['length'], item_type['width'], item_type['height']
        return [(0, (L, W, H)), (1, (L, H, W)), (2, (W, L, H)),
                (3, (W, H, L)), (4, (H, L, W)), (5, (H, W, L))]

    def _is_within_container_bounds(self, item_pos, item_dims, container_dims):
        px, py, pz = item_pos; pl, pw, ph = item_dims
        cl, cw, ch = container_dims
        return (px >= -self.epsilon and px + pl <= cl + self.epsilon and
                py >= -self.epsilon and py + pw <= cw + self.epsilon and
                pz >= -self.epsilon and pz + ph <= ch + self.epsilon)

    def _is_valid_placement(self, item_to_place_pos, item_to_place_dims, container_dims, placed_items):
        if not self._is_within_container_bounds(item_to_place_pos, item_to_place_dims, container_dims): return False
        for placed_item in placed_items:
            placed_item_pos = (placed_item['x'], placed_item['y'], placed_item['z'])
            placed_item_dims = (placed_item['length'], placed_item['width'], placed_item['height'])
            if self._check_overlap_3d(item_to_place_pos, item_to_place_dims, placed_item_pos, placed_item_dims): return False
        return True

class BasePlanAlgorithm(ABC):
    @abstractmethod
    def pack_all(self, items_to_pack: List[Dict], container_dims: Tuple) -> Tuple[List[Dict], List[Dict]]:
        pass




class PlanEvaluator:
    """
    Evaluates packing algorithms that generate a complete packing plan at once,
    rather than placing items one by one. Includes solution validity checks.
    """
    def __init__(self, n_instances, path_pattern):
        self.data_generator = GetData(n_instance=n_instances)
        self.instances = self.data_generator.generate_instances(path_pattern)
        if not self.instances:
            print("Warning: No instances were loaded.")

    def run_evaluation(self, algorithms: Dict[str, 'BasePlanAlgorithm']):
        """
        Runs the evaluation for a dictionary of planning algorithms.

        Args:
            algorithms (Dict[str, 'BasePlanAlgorithm']): A dictionary where keys are
                algorithm names and values are instances of algorithms that have a
                `pack_all` method.
        """
        if not self.instances:
            print("No instances loaded to evaluate. Exiting.")
            return

        print(f"--- Starting Full-Plan Evaluation on {len(self.instances)} Instances ---\n")
        for name, algorithm in algorithms.items():
            self._evaluate_algorithm(name, algorithm)

    def _evaluate_algorithm(self, name: str, algorithm: 'BasePlanAlgorithm'):
        total_trucks = 0
        total_volume_use = 0
        total_unplaced_items = 0
        total_invalid_solutions = 0
        start_time = time.time()

        for instance_id, (instance_items, container_dims) in enumerate(self.instances):
            trucks_in_use, unplaced_items_list = algorithm.pack_all(instance_items, container_dims)
            # print(f"instance_items:{instance_items}")
            # print(f"container_dims:{container_dims}")
            # print(f"trucks_in_use:{trucks_in_use}")
            # --- NEW: Added validity check for the generated solution ---
            # if not self._check_solution_validity(name, instance_id, trucks_in_use, container_dims):
            #     total_invalid_solutions += 1
                # Continue with metric calculation, but be aware the solution is flawed.
                # Alternatively, you could skip metrics for invalid solutions.

            # The S_GRASP implementation returns trucks with a 'placed_items' key.
            # We adapt the metric calculation to handle this structure.
            metrics = self._calculate_metrics(trucks_in_use, unplaced_items_list, container_dims)
            total_trucks += metrics['trucks_used']
            total_volume_use += metrics['volume_utilization']
            total_unplaced_items += metrics['unplaced_items']

        end_time = time.time()
        duration = end_time - start_time
        num_instances = len(self.instances)

        avg_trucks = total_trucks / num_instances if num_instances > 0 else 0
        avg_utilization_percent = (total_volume_use / num_instances) * 100 if num_instances > 0 else 0
        
        self._print_summary(name, total_trucks, avg_trucks, avg_utilization_percent, total_unplaced_items, duration, total_invalid_solutions)

    def _check_solution_validity(self, algo_name: str, instance_id: int, trucks: List[Dict], container_dims: Tuple) -> bool:
        """
        Checks if the packing solution is valid (no overlaps, within bounds).

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        is_globally_valid = True
        cont_l, cont_w, cont_h = container_dims
        
        for i, truck in enumerate(trucks):
            placed_items = truck.get('placed_items', [])
            if not placed_items:
                continue

            # 1. Check for items outside container boundaries
            for item in placed_items:
                pos = item.get('position')
                if pos is None:
                    print(f"VALIDATION ERROR ({algo_name}, Instance {instance_id}): Item {item.get('unique_id')} in Truck {i} is missing 'position' key.")
                    is_globally_valid = False
                    continue

                x, y, z = pos
                l, w, h = item['length'], item['width'], item['height']
                
                # Use a small tolerance for floating point inaccuracies
                if (x + l > cont_l + 1e-6) or (y + w > cont_w + 1e-6) or (z + h > cont_h + 1e-6):
                    print(f"VALIDATION ERROR ({algo_name}, Instance {instance_id}): Item {item.get('unique_id')} in Truck {i} is out of bounds.")
                    print(f"  Item pos/dims: ({x},{y},{z}) / ({l},{w},{h}) | Container dims: ({cont_l},{cont_w},{cont_h})")
                    is_globally_valid = False

            # 2. Check for overlaps between items using itertools.combinations for efficiency
            for item1, item2 in combinations(placed_items, 2):
                pos1, pos2 = item1.get('position'), item2.get('position')
                if pos1 is None or pos2 is None: continue # Already handled above

                x1, y1, z1 = pos1
                l1, w1, h1 = item1['length'], item1['width'], item1['height']
                x2, y2, z2 = pos2
                l2, w2, h2 = item2['length'], item2['width'], item2['height']

                # Check for overlap on all three axes
                overlap_x = (x1 < x2 + l2) and (x2 < x1 + l1)
                overlap_y = (y1 < y2 + w2) and (y2 < y1 + w1)
                overlap_z = (z1 < z2 + h2) and (z2 < z1 + h1)

                if overlap_x and overlap_y and overlap_z:
                    print(f"VALIDATION ERROR ({algo_name}, Instance {instance_id}): Overlap detected in Truck {i}.")
                    print(f"  Item 1 ({item1.get('unique_id')}): pos=({x1},{y1},{z1}), dims=({l1},{w1},{h1})")
                    print(f"  Item 2 ({item2.get('unique_id')}): pos=({x2},{y2},{z2}), dims=({l2},{w2},{h2})")
                    is_globally_valid = False
                    
        return is_globally_valid


    def _calculate_metrics(self, trucks, unplaced_list, container_dims):
        trucks_used = len(trucks)
        unplaced_count = sum(item['quantity'] for item in unplaced_list)

        if trucks_used == 0:
            return {'trucks_used': 0, 'volume_utilization': 0.0, 'unplaced_items': unplaced_count}

        container_volume = container_dims[0] * container_dims[1] * container_dims[2]
        
        # --- CORRECTED: Use 'placed_items' which S_GRASP provides, not 'occupied_volumes' ---
        total_placed_volume = sum(
            item['length'] * item['width'] * item['height']
            for truck in trucks for item in truck.get('placed_items', [])
        )

        # Volume utilization is the total volume of packed items divided by the total volume of containers used.
        volume_utilization = total_placed_volume / (trucks_used * container_volume) if trucks_used > 0 else 0
        
        return {'trucks_used': trucks_used, 'volume_utilization': volume_utilization, 'unplaced_items': unplaced_count}

    def _print_summary(self, name, total_trucks, avg_trucks, avg_utilization_percent, unplaced_items, duration, invalid_solutions=0):
        print(f"--- Results for: {name} ---")
        print(f"  Total execution time: {duration:.3f} seconds")
        if invalid_solutions > 0:
            print(f"  !! Invalid solutions generated: {invalid_solutions} / {len(self.instances)} instances !!")
        print(f"  Total trucks used across all instances: {total_trucks}")
        print(f"  Average trucks per instance: {avg_trucks:.3f}")
        print(f"  Average container volume utilization: {avg_utilization_percent:.2f}%")
        print(f"  Total items left unplaced: {unplaced_items}")
        print("-" * (len(name) + 20) + "\n")





def _run_single_evaluation(
    run_id: int,
    n_instances: int,
    path_pattern: str,
    algorithm_class,
    algorithm_params: Dict,
) -> Dict[int, Set[frozenset]]:
    """
    Worker function for parallel execution. Runs one full evaluation and returns the
    collected patterns. Designed to be run in a separate process.

    Args:
        run_id (int): Identifier for this specific run.
        n_instances (int): Number of instances to generate/load.
        path_pattern (str): Path pattern for loading instance data.
        algorithm_class: The algorithm class to instantiate (e.g., S_GRASP).
        algorithm_params (Dict): Parameters for the algorithm's constructor.
    """
    data_gen = GetData(n_instance=n_instances)
    instances = data_gen.generate_instances(path_pattern)
    
    # Create a seeded numpy random number generator and pass it to the algorithm
    rng = np.random.default_rng(run_id)
    algo_params_with_rng = algorithm_params.copy()
    algo_params_with_rng['rng'] = rng
    algorithm = algorithm_class(**algo_params_with_rng)
    
    run_patterns_by_instance = collections.defaultdict(set)

    for instance_id, (instance_items, container_dims) in enumerate(instances):
        trucks_in_use, _ = algorithm.pack_all(instance_items, container_dims)
        
        for truck in trucks_in_use:
            item_counts = collections.Counter(
                item['item_id'] for item in truck.get('placed_items', [])
            )
            pattern = frozenset(item_counts.items())
            
            if pattern:
                run_patterns_by_instance[instance_id].add(pattern)
    
    print(f"Finished parallel run #{run_id}.")
    return dict(run_patterns_by_instance)


class PatternCollector:
    """
    Manages running a randomized algorithm multiple times in parallel to collect
    a diverse set of unique packing patterns from all instances across all runs.
    """
    def __init__(self, n_instances, path_pattern, seed):
        self.n_instances = n_instances
        self.path_pattern = path_pattern
        self.all_patterns_by_instance: Dict[int, Set[frozenset]] = collections.defaultdict(set)
        self.seed = seed

    def run_and_collect(self, algorithm_class, algorithm_params: Dict, num_runs=10, max_workers=None):
        """
        Runs the evaluation in parallel for `num_runs` and collects all unique patterns.

        Args:
            algorithm_class: The algorithm class to be instantiated for each run.
            algorithm_params: Parameters for the algorithm's constructor.
            num_runs: The number of parallel evaluations to execute.
            max_workers: The maximum number of processes to use. If None, it defaults
                         to the number of processors on the machine.
        """
        print(f"\n{'='*20} STARTING PARALLEL PATTERN COLLECTION {'='*20}")
        print(f"Algorithm: {algorithm_class.__name__} with params {algorithm_params}")
        print(f"Number of parallel runs: {num_runs}")
        print(f"Instances per run: {self.n_instances}")
        print(f"Max worker processes: {'Default (all cores)' if max_workers is None else max_workers}")
        print(f"{'='*68}\n")
        

        self.all_patterns_by_instance.clear()
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_evaluation,
                    i + 1 +self.seed,
                    self.n_instances,
                    self.path_pattern,
                    algorithm_class,
                    algorithm_params
                )
                for i in range(num_runs)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    run_patterns_by_instance = future.result()
                    for instance_id, patterns in run_patterns_by_instance.items():
                        self.all_patterns_by_instance[instance_id].update(patterns)
                except Exception as exc:
                    print(f"A run generated an exception: {exc}")

        duration = time.time() - start_time
        self._print_collection_summary(num_runs, duration)
        
        return self.all_patterns_by_instance

    def run_and_collect_all(self, algorithm_class, algorithm_params: Dict, num_runs=10, max_workers=None):
        """
        Runs the evaluation in parallel for `num_runs` and collects all unique patterns.

        Args:
            algorithm_class: The algorithm class to be instantiated for each run.
            algorithm_params: Parameters for the algorithm's constructor.
            num_runs: The number of parallel evaluations to execute.
            max_workers: The maximum number of processes to use. If None, it defaults
                         to the number of processors on the machine.
        """
        print(f"\n{'='*20} STARTING PARALLEL PATTERN COLLECTION {'='*20}")
        print(f"Algorithm: {algorithm_class.__name__} with params {algorithm_params}")
        print(f"Number of parallel runs: {num_runs}")
        print(f"Instances per run: {self.n_instances}")
        print(f"Max worker processes: {'Default (all cores)' if max_workers is None else max_workers}")
        print(f"{'='*68}\n")
        

        self.all_patterns_by_instance.clear()
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_evaluation,
                    i + 1 +self.seed,
                    self.n_instances,
                    self.path_pattern,
                    algorithm_class,
                    algorithm_params
                )
                for i in range(num_runs)
            ]
            results = [future.result() for future in futures]
        
        return results


    def _print_collection_summary(self, num_runs, duration):
        total_unique_patterns = sum(len(p) for p in self.all_patterns_by_instance.values())
        print(f"\n{'='*20} PATTERN COLLECTION COMPLETE {'='*20}")
        print(f"Total time for {num_runs} parallel runs: {duration:.3f} seconds")
        print(f"Collected patterns for {len(self.all_patterns_by_instance)} unique instances.")
        print(f"Total unique patterns found across all instances and runs: {total_unique_patterns}")
        print(f"{'='*61}\n")

    def get_all_patterns_by_instance(self) -> Dict[int, Set[frozenset]]:
        return dict(self.all_patterns_by_instance)

    def get_all_unique_patterns(self) -> Set[frozenset]:
        all_patterns = set()
        for patterns_for_instance in self.all_patterns_by_instance.values():
            all_patterns.update(patterns_for_instance)
        return all_patterns

import pulp

def solve_set_partitioning_with_pulp(
    all_patterns: Set[frozenset],
    item_demands: Dict[str, int]
) -> Optional[Dict]:
    """
    Solves the bin packing problem using a set partitioning formulation with PuLP.

    Args:
        all_patterns: A set of unique packing patterns generated by a heuristic.
                      Each pattern is a frozenset of (item_id, count) tuples.
        item_demands: A dictionary mapping each item_id to its total required quantity.

    Returns:
        A dictionary with the solution details ('status', 'total_trucks', 'pattern_usage')
        or None if the input is invalid.
    """
    if not all_patterns or not item_demands:
        print("Warning: Received empty patterns or demands. Cannot solve.")
        return None

    prob = pulp.LpProblem("Truck_Pattern_Selection", pulp.LpMinimize)
    pattern_list = list(all_patterns)
    pattern_indices = range(len(pattern_list))

    x = pulp.LpVariable.dicts("Pattern", pattern_indices, lowBound=0, cat='Integer')

    prob += pulp.lpSum(x[j] for j in pattern_indices), "Minimize_Total_Trucks"

    pattern_item_counts = [dict(p) for p in pattern_list]

    for item_id, demand in item_demands.items():
        constraint_expr = pulp.lpSum(
            pattern_item_counts[j].get(item_id, 0) * x[j] for j in pattern_indices
        )
        prob += constraint_expr == demand, f"Demand_Constraint_{item_id}"

    # Use a solver. PULP_CBC_CMD is the default and is included with PuLP.
    # msg=False suppresses solver output in the console.
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    solution = {
        "status": pulp.LpStatus[prob.status],
        "total_trucks": None,
        "pattern_usage": {}
    }

    if prob.status == pulp.LpStatusOptimal:
        solution["total_trucks"] = int(pulp.value(prob.objective))
        for j in pattern_indices:
            if x[j].varValue > 0.5:
                num_times_used = int(round(x[j].varValue))
                used_pattern = pattern_list[j]
                solution["pattern_usage"][used_pattern] = num_times_used
    
    return solution

