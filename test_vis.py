import json
import copy
import os
import numpy as np
import matplotlib.pyplot as plt # Used only for its colormaps

# ==============================================================================
# SECTION 1: HUMAN-DESIGNED ALGORITHM (Your provided code)
# No changes here.
# ==============================================================================
SUPPORT_RATIO = 0.8
BEYOND_BETA = 150

class TruckType:
    def __init__(self, truckTypeId, truckTypeCode, truckTypeName, length, width, height, maxLoad):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"}); self.volume = length * width * height
class Space:
    def __init__(self, x, y, z, lx, ly, lz): self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
    def __eq__(self, other): return self.x == other.x and self.y == other.y and self.z == other.z and self.lx == other.lx and self.ly == other.ly and self.lz == other.lz
    def __lt__(self, other): return (self.x, self.y, self.z) < (other.x, other.y, other.z)
class LoadingTruck:
    def __init__(self, truck_type):
        self.truck_type = truck_type; self.free_space_list = [Space(0, 0, 0, self.truck_type.length, self.truck_type.width, self.truck_type.height)]; self.discard_space_list = []; self.used_space_list = []; self.block_list = []; self.left_weight = self.truck_type.maxLoad; self.sum_used_space = 0; self.box_list = []
class Block:
    def __init__(self, stack_count_xyz, unit_size, sum_weight, box_list):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"}); self.volume = (unit_size[0] * stack_count_xyz[0] * unit_size[1] * stack_count_xyz[1] * unit_size[2] * stack_count_xyz[2])
    def __gt__(self, other): return (self.volume, self.unit_size[1]) > (other.volume, other.unit_size[1])
def get_block_list(space, size_box, left_weight, used_space_list, truck_type):
    block_list1 = []; filter_size_box = filter(lambda one: one[0] <= space.lx and one[1] <= space.ly and one[2] <= space.lz, size_box)
    for size in filter_size_box:
        box_list = size_box[size]; pre_sum_weigt = []; max_num = len(box_list)
        for i, box in enumerate(box_list):
            pre_sum_weigt.append(box["weight"] + (pre_sum_weigt[i - 1] if i > 0 else 0))
            if left_weight < pre_sum_weigt[-1]: max_num = i; break
        nx, ny, nz = int(space.lx / size[0]), int(space.ly / size[1]), int(space.lz / size[2])
        if max_num <= ny: ny, nz, nx = max_num, 1, 1
        elif max_num <= ny * nz: nz, nx = int(max_num / ny), 1
        elif max_num <= nx * ny * nz: nx = int(max_num / (ny * nz))
        if nx * ny * nz > 0: block_list1.append(Block((nx, ny, nz), size, pre_sum_weigt[nx * ny * nz - 1], box_list[: nx * ny * nz]))
    block_list2 = []; filter_size_box = filter(lambda one: not (one[2] > space.lz or space.x + one[0] > truck_type.length or space.y + one[1] > truck_type.width or (one[0] <= space.lx and one[1] <= space.ly)), size_box)
    for size in filter_size_box:
        box_list = sorted(size_box[size][:], key=lambda x: x["weight"], reverse=True)
        if (box_list and box_list[0]["weight"] <= left_weight and size[0] - space.lx < BEYOND_BETA and space.lx / size[0] > SUPPORT_RATIO and size[1] - space.ly < BEYOND_BETA and space.ly / size[1] > SUPPORT_RATIO and (min(size[0], space.lx) * min(size[1], space.ly)) / (size[0] * size[1]) > SUPPORT_RATIO):
            now_space = Space(space.x, space.y, space.z, size[0], size[1], size[2]); is_ok = True
            for used_space in used_space_list:
                if check_overlap_3d(now_space, used_space) or (used_space.x > now_space.x and check_overlap_2d(now_space, used_space)): is_ok = False; break
            if is_ok: block_list2.append(Block((1, 1, 1), size, box_list[0]["weight"], [box_list[0]]))
    return block_list1, block_list2
def check_overlap_3d(s1, s2): return not (s1.x >= s2.x + s2.lx or s1.x + s1.lx <= s2.x or s1.y >= s2.y + s2.ly or s1.y + s1.ly <= s2.y or s1.z >= s2.z + s2.lz or s1.z + s1.lz <= s2.z)
def check_overlap_2d(s1, s2): return not (s1.y >= s2.y + s2.ly or s1.y + s1.ly <= s2.y or s1.z >= s2.z + s2.lz or s1.z + s1.lz <= s2.z)
def update_space_list_by_block_space(block_space, space_list):
    updated = []; 
    for pre in space_list:
        if check_overlap_3d(block_space, pre):
            dY = block_space.y + block_space.ly - pre.y; dX = block_space.x + block_space.lx - pre.x
            if dY < min(BEYOND_BETA, block_space.ly/SUPPORT_RATIO-block_space.ly): pre.ly -= dY; pre.y = block_space.y + block_space.ly
            if dX < min(BEYOND_BETA, block_space.lx/SUPPORT_RATIO-block_space.lx): pre.lx -= dX; pre.x = block_space.x + block_space.lx
            if pre.ly <= 0 or pre.lx <= 0: continue
        if pre.x < block_space.x and check_overlap_2d(block_space, pre): pre.ly -= block_space.y + block_space.ly - pre.y; pre.y = block_space.y + block_space.ly
        if pre.ly > 0 and pre.lx > 0: updated.append(pre)
    return updated
def get_refresh_spaces(block_space, discard_space_list):
    refresh = []; 
    for s in discard_space_list:
        if check_overlap_2d(block_space, s):
            s.ly -= block_space.y + block_space.ly - s.y; s.y = block_space.y + block_space.ly
            if s.ly > 0: refresh.append(s)
    return refresh, list(filter(lambda s: s.ly > 0, discard_space_list))
def update_info(truck, size_box, space, block):
    truck.box_list.extend(block.box_list); truck.block_list.append(block); bs = Space(space.x, space.y, space.z, *tuple(x*y for x,y in zip(block.unit_size, block.stack_count_xyz))); truck.used_space_list.append(bs)
    truck.sum_used_space += bs.lx*bs.ly*bs.lz; truck.left_weight -= block.sum_weight; ids = {b["uuid"]:1 for b in block.box_list}; sizes=[block.unit_size]
    if block.unit_size[0]!=block.unit_size[1]: sizes.append((block.unit_size[1],block.unit_size[0],block.unit_size[2]))
    for s in sizes:
        if s not in size_box: continue
        size_box[s] = list(filter(lambda b: b["uuid"] not in ids, size_box[s]))
        if len(size_box[s]) == 0: del size_box[s]
    truck.free_space_list.pop(); rs, truck.discard_space_list = get_refresh_spaces(bs, truck.discard_space_list)
    truck.free_space_list = update_space_list_by_block_space(bs, truck.free_space_list); truck.discard_space_list = update_space_list_by_block_space(bs, truck.discard_space_list)
    dX,dY,dZ=space.lx-bs.lx,space.ly-bs.ly,space.lz-bs.lz
    if dX > 0: truck.free_space_list.append(Space(bs.x+bs.lx,bs.y,bs.z,dX,space.ly,space.lz))
    if dY > 0:
        s2 = Space(bs.x,bs.y+bs.ly,bs.z,min(bs.lx,space.lx),dY,space.lz); to_remove=[]
        for ps in rs:
            if (ps.x+ps.lx==s2.x and ps.y==s2.y and ps.z==s2.z and ps.ly==s2.ly and ps.lz==s2.lz): s2.x=ps.x;s2.lx+=ps.lx;to_remove.append(ps)
        for sp in to_remove:
            if sp in truck.discard_space_list: truck.discard_space_list.remove(sp)
        truck.free_space_list.append(s2)
    if dZ > 0:
        s3 = Space(bs.x,bs.y,bs.z+bs.lz,bs.lx,bs.ly,dZ)
        if bs.lx>space.lx or bs.ly>space.ly:
            overlaps = list(filter(lambda b: check_overlap_3d(s3, b), truck.used_space_list))
            if overlaps: s3.ly = min(overlaps, key=lambda x:x.y).y-s3.y
        truck.free_space_list.append(s3)
def select_block(bl1, bl2):
    b1 = max(bl1, key=lambda b: (b.volume,b.unit_size[1])) if bl1 else None; b2 = max(bl2, key=lambda b: (b.volume,b.unit_size[1])) if bl2 else None
    if b2 is None: return b1; 
    if b1 is None: return b2
    return b1 if b1.volume > b2.volume else b2
def get_size_box(box_list):
    size_box = {};
    for box in box_list:
        sizes = [(box["length"],box["width"],box["height"])]; 
        if box["length"]!=box["width"]: sizes.append((box["width"],box["length"],box["height"]))
        for s in sizes:
            if s not in size_box: size_box[s]=[]
            size_box[s].append(box)
    return size_box

# ==============================================================================
# SECTION 2: LLM-DESIGNED ALGORITHM (Your provided code, with fixes)
# ==============================================================================
class BaseAlgorithm:
    def __init__(self, support_threshold=0.8, epsilon=1e-6):
        self.support_threshold, self.epsilon = support_threshold, epsilon

    def _check_overlap_3d(self, item1_pos, item1_dims, item2_pos, item2_dims):
        x1, y1, z1 = item1_pos; l1, w1, h1 = item1_dims; x2, y2, z2 = item2_pos; l2, w2, h2 = item2_dims
        return (x1 < x2 + l2) and (x1 + l1 > x2) and (y1 < y2 + w2) and (y1 + w1 > y2) and (z1 < z2 + h2) and (z1 + h1 > z2)

    def _check_item_fits_new_truck_type(self, item_details, truck_type_info):
        _capacity, tr_l, tr_w, tr_h = truck_type_info
        empty_truck_details = {"dims": (tr_l, tr_w, tr_h), "placed_items_details": []}
        return self._is_placement_valid((0.0, 0.0, 0.0), item_details, empty_truck_details)

    def _is_within_truck_bounds(self, item_pos, item_dims, truck_dims):
        px, py, pz = item_pos; pl, pw, ph = item_dims; tl, tw, th = truck_dims
        if not (px >= 0.0 - self.epsilon and px + pl <= tl + self.epsilon): return False
        if not (py >= 0.0 - self.epsilon and py + pw <= tw + self.epsilon): return False
        if not (pz >= 0.0 - self.epsilon and pz + ph <= th + self.epsilon): return False
        return True

    def _check_support(self, new_item_pos, new_item_dims, placed_items_details_list):
        new_x, new_y, new_z = new_item_pos; new_l, new_w, _ = new_item_dims
        if new_z < self.epsilon: return True
        new_item_base_area = new_l * new_w
        if new_item_base_area < self.epsilon: return True
        total_supported_area = 0.0
        for (ex, ey, ez), (el, ew, eh) in placed_items_details_list:
            if abs(new_z - (ez + eh)) < self.epsilon:
                x_overlap = max(0.0, min(new_x + new_l, ex + el) - max(new_x, ex))
                y_overlap = max(0.0, min(new_y + new_w, ey + ew) - max(new_y, ey))
                total_supported_area += x_overlap * y_overlap
        return (total_supported_area / new_item_base_area) >= (self.support_threshold - self.epsilon)

    def _get_item_details(self, item):
        l, w, h, wt = item.get("length", 0.0), item.get("width", 0.0), item.get("height", 0.0), item.get("weight", 0.0)
        item_id = item.get("item_id", item.get("id", None))
        return {"id": item_id, "l": l, "w": w, "h": h, "weight": wt, "vol": l * w * h, "dims": (l, w, h), "original_item": item}

    def _get_truck_details(self, truck_dict, truck_type_info):
        capacity, tr_l, tr_w, tr_h = truck_type_info
        current_weight = truck_dict.get("current_weight", 0.0)
        placed_items_details_for_check, current_occupied_volume_sum = [], 0.0
        raw_occupied_volumes = truck_dict.get("occupied_volumes", [])
        for placed_data in raw_occupied_volumes:
            _item_id, x, y, z, l, w, h = placed_data[0:7]
            placed_items_details_for_check.append(((x, y, z), (l, w, h))); current_occupied_volume_sum += l * w * h
        total_truck_volume = tr_l * tr_w * tr_h
        return {"truck_type_index": truck_dict.get("truck_type_index"), "capacity": capacity, "dims": (tr_l, tr_w, tr_h), "total_volume": total_truck_volume, "current_weight": current_weight, "placed_items_details": placed_items_details_for_check, "raw_occupied_volumes": raw_occupied_volumes, "current_occupied_volume_sum": current_occupied_volume_sum, "remaining_weight_capacity": capacity - current_weight, "remaining_physical_volume": total_truck_volume - current_occupied_volume_sum, "original_truck_dict": truck_dict}

    def _is_placement_valid(self, new_item_pos, new_item_details, truck_details):
        if not self._is_within_truck_bounds(new_item_pos, new_item_details["dims"], truck_details["dims"]): return False
        for placed_item_pos, placed_item_dims in truck_details["placed_items_details"]:
            if self._check_overlap_3d(new_item_pos, new_item_details["dims"], placed_item_pos, placed_item_dims): return False
        if not self._check_support(new_item_pos, new_item_details["dims"], truck_details["placed_items_details"]): return False
        return True

class Algorithm(BaseAlgorithm):
    # {This algorithm prioritizes fitting items into existing trucks by aggressively exploring placement options that minimize wasted space, balancing volume and weight usage.}
    def __init__(self, support_threshold=0.8, epsilon=1e-6, volume_weight_tradeoff=0.6, aggressive_placement=True):
        super().__init__(support_threshold, epsilon)
        self.volume_weight_tradeoff = volume_weight_tradeoff
        self.aggressive_placement = aggressive_placement

    def place_item(self, unplaced_items, trucks_in_use, truck_types):
        item_index, truck_index, x, y, z, truck_type_index = -1, -1, 0.0, 0.0, 0.0, -1

        item_index = self._select_item(unplaced_items)
        if item_index == -1:
            return -1, -1, 0.0, 0.0, 0.0, -1

        item = unplaced_items[item_index]
        item_details = self._get_item_details(item)

        truck_index, x, y, z = self._find_placement_in_existing_trucks(item_details, trucks_in_use, truck_types)

        if truck_index != -1:
            return truck_index, item_index, x, y, z, -1

        truck_type_index = self._select_new_truck_type(item_details, truck_types)
        if truck_type_index != -1:
            return -1, item_index, 0.0, 0.0, 0.0, truck_type_index

        return -1, item_index, 0.0, 0.0, 0.0, -1

    def _select_item(self, unplaced_items):
        if not unplaced_items:
            return -1

        volumes = [item['length'] * item['width'] * item['height'] for item in unplaced_items]
        item_index = np.argmax(volumes)
        return int(item_index)

    def _find_placement_in_existing_trucks(self, item_details, trucks_in_use, truck_types):
        best_truck_index, best_x, best_y, best_z = -1, 0.0, 0.0, 0.0
        best_score = -1.0

        for truck_index, truck in enumerate(trucks_in_use):
            truck_type_index = truck['truck_type_index']
            truck_type = truck_types[truck_type_index]
            truck_details = self._get_truck_details(truck, truck_type)

            if truck_details["remaining_weight_capacity"] < item_details["weight"]:
                continue

            x, y, z, score = self._find_best_position_in_truck(item_details, truck_details)

            if x != -1 and score > best_score:
                best_truck_index, best_x, best_y, best_z = truck_index, x, y, z
                best_score = score

        if best_truck_index != -1:
            return best_truck_index, best_x, best_y, best_z
        else:
            return -1, 0.0, 0.0, 0.0

    def _find_best_position_in_truck(self, item_details, truck_details):
        tl, tw, th = truck_details["dims"]
        l, w, h = item_details["dims"]

        best_x, best_y, best_z = -1, -1, -1
        best_score = -1.0

        x_candidates, y_candidates = self._generate_candidate_positions(truck_details, item_details)

        for x in x_candidates:
            for y in y_candidates:
                z = self._calculate_support_z(x, y, item_details, truck_details)
                if z + h <= th and self._is_placement_valid((x, y, z), item_details, truck_details):
                    score = self._calculate_placement_score(x, y, z, item_details, truck_details)

                    if score > best_score:
                        best_x, best_y, best_z = x, y, z
                        best_score = score

        return best_x, best_y, best_z, best_score

    def _generate_candidate_positions(self, truck_details, item_details):
        tl, tw, th = truck_details["dims"]
        l, w, h = item_details["dims"]

        x_candidates = [0.0]
        y_candidates = [0.0]
        placed_items_details = truck_details["placed_items_details"]

        for placed_item_pos, placed_item_dims in placed_items_details:
            px, py, pz = placed_item_pos
            pl, pw, ph = placed_item_dims
            x_candidates.append(px + pl)
            y_candidates.append(px - l if px - l >= 0 else float('inf'))
            y_candidates.append(py + pw)
            y_candidates.append(py - w if py - w >= 0 else float('inf'))

        x_candidates = sorted(list(set(x for x in x_candidates if 0 <= x <= tl - l and x != float('inf'))))
        y_candidates = sorted(list(set(y for y in y_candidates if 0 <= y <= tw - w and y != float('inf'))))

        return x_candidates, y_candidates

    def _calculate_placement_score(self, x, y, z, item_details, truck_details):
        weight_utilization = item_details["weight"] / truck_details["capacity"]
        volume_utilization = item_details["vol"] / truck_details["total_volume"]
        remaining_volume_ratio = (truck_details["remaining_physical_volume"] - item_details["vol"]) / truck_details["total_volume"]
        score = (1 - self.volume_weight_tradeoff) * weight_utilization + self.volume_weight_tradeoff * (1 - remaining_volume_ratio)
        return score

    def _calculate_support_z(self, x, y, item_details, truck_details):
        l, w, h = item_details["dims"]
        max_z = 0.0

        for placed_item_pos, placed_item_dims in truck_details["placed_items_details"]:
            px, py, pz = placed_item_pos
            pl, pw, ph = placed_item_dims
            if (x < px + pl and x + l > px) and (y < py + pw and y + w > py):
                max_z = max(max_z, pz + ph)

        return max_z

    def _select_new_truck_type(self, item_details, truck_types):
        best_truck_type_index = -1
        min_waste_ratio = float('inf')
        item_volume = item_details["vol"]

        for i, truck_type in enumerate(truck_types):
            capacity, tl, tw, th = truck_type
            truck_volume = tl * tw * th

            if self._check_item_fits_new_truck_type(item_details, truck_type):
                waste_ratio = (truck_volume - item_volume) / truck_volume
                if waste_ratio < min_waste_ratio:
                    min_waste_ratio = waste_ratio
                    best_truck_type_index = i

        return best_truck_type_index
import numpy as np

class AlgorithmFIX1(Algorithm):
    def _generate_candidate_positions(self, truck_details, item_details):
        tl, tw, th = truck_details["dims"]
        l, w, h = item_details["dims"]

        x_candidates = [0.0]
        y_candidates = [0.0]
        placed_items_details = truck_details["placed_items_details"]

        for placed_item_pos, placed_item_dims in placed_items_details:
            px, py, pz = placed_item_pos
            pl, pw, ph = placed_item_dims
            x_candidates.append(px + pl)
            y_candidates.append(py + pw)

        x_candidates = sorted(list(set(x for x in x_candidates if 0 <= x <= tl - l)))
        y_candidates = sorted(list(set(y for y in y_candidates if 0 <= y <= tw - w)))

        return x_candidates, y_candidates
import numpy as np
import random

class AlgorithmFIX2(AlgorithmFIX1):
    def _find_best_position_in_truck(self, item_details, truck_details):
        tl, tw, th = truck_details["dims"]
        l, w, h = item_details["dims"]

        best_x, best_y, best_z = -1, -1, -1
        best_score = -1.0

        x_candidates, y_candidates = self._generate_candidate_positions(truck_details, item_details)
        num_x_candidates = len(x_candidates)
        num_y_candidates = len(y_candidates)

        if num_x_candidates > 10:
            x_candidates = x_candidates[:5] + x_candidates[-5:]
        if num_y_candidates > 10:
            y_candidates = y_candidates[:5] + y_candidates[-5:]

        for x in x_candidates:
            for y in y_candidates:
                z = self._calculate_support_z(x, y, item_details, truck_details)
                if z + h <= th and self._is_placement_valid((x, y, z), item_details, truck_details):
                    score = self._calculate_placement_score(x, y, z, item_details, truck_details)

                    if score > best_score:
                        best_x, best_y, best_z = x, y, z
                        best_score = score

        return best_x, best_y, best_z, best_score

# ==============================================================================
# SECTION 3: STEP-BY-STEP RUNNERS
# ==============================================================================
def run_human_packer_step_by_step(boxes, truck_types):
    history = []
    for i, box in enumerate(boxes): box["uuid"] = f"box-{i}"
    truck_type_list = sorted(truck_types, key=lambda x: x.volume, reverse=True)
    if not truck_type_list: return []
    size_box = get_size_box(copy.deepcopy(boxes))
    loading_truck = LoadingTruck(truck_type_list[0])
    history.append(copy.deepcopy(loading_truck))
    while len(loading_truck.free_space_list) > 0 and len(size_box) > 0:
        loading_truck.free_space_list.sort()
        space = loading_truck.free_space_list[-1]
        bl1, bl2 = get_block_list(space, size_box, loading_truck.left_weight, loading_truck.used_space_list, loading_truck.truck_type)
        if not bl1 and not bl2:
            loading_truck.discard_space_list.insert(0, space); loading_truck.free_space_list.pop(); continue
        block = select_block(bl1, bl2)
        update_info(loading_truck, size_box, space, block)
        history.append(copy.deepcopy(loading_truck))
    return history

def run_llm_packer_step_by_step(boxes, truck_types_info):
    history = []
    algo = AlgorithmFIX2()
    unplaced_items = copy.deepcopy(boxes)
    for i, item in enumerate(unplaced_items): item["id"] = f"item-{i}"
    trucks_in_use = []
    
    if truck_types_info:
        initial_truck_state = [{"truck_type_index": 0, "current_weight": 0.0, "occupied_volumes": []}]
        history.append(copy.deepcopy(initial_truck_state))

    while unplaced_items:
        truck_idx, item_idx, x, y, z, new_truck_type_idx = algo.place_item(unplaced_items, trucks_in_use, truck_types_info)
        if item_idx == -1: break
        item_to_place = unplaced_items.pop(item_idx)
        if new_truck_type_idx != -1:
            if trucks_in_use: break 
            new_truck = {"truck_type_index": new_truck_type_idx, "current_weight": 0.0, "occupied_volumes": []}
            trucks_in_use.append(new_truck); truck_idx = len(trucks_in_use) - 1; x,y,z = 0,0,0
        if truck_idx == 0:
            truck = trucks_in_use[truck_idx]
            item_details = algo._get_item_details(item_to_place)
            truck["current_weight"] += item_details["weight"]
            truck["occupied_volumes"].append([item_details["id"], x, y, z, *item_details["dims"]])
            history.append(copy.deepcopy(trucks_in_use))
        elif truck_idx == -1: break
        else: break
    return history

# ==============================================================================
# SECTION 4: WEB VISUALIZATION EXPORT
# ==============================================================================
def generate_comparison_visualization(human_history, llm_history, json_filename="comparison_data.json", html_filename="packing_comparison.html"):
    if not human_history or not llm_history:
        print("Cannot generate visualization: one or both histories are empty.")
        return

    first_human_state = human_history[0]
    truck_dims = {"length": first_human_state.truck_type.length, "width": first_human_state.truck_type.width, "height": first_human_state.truck_type.height}
    all_item_types = set()
    if len(human_history) > 1:
        for block in human_history[-1].block_list:
            all_item_types.add(f"item_{int(block.unit_size[0])}x{int(block.unit_size[1])}x{int(block.unit_size[2])}")
    if len(llm_history) > 1:
        for item in llm_history[-1][0]['occupied_volumes']:
            all_item_types.add(f"item_{int(item[4])}x{int(item[5])}x{int(item[6])}")
            
    colors = plt.cm.get_cmap("viridis", len(all_item_types) if all_item_types else 1)
    item_type_colors = {type_id: f'#%02x%02x%02x' % tuple(int(c*255) for c in colors(i)[:3]) for i, type_id in enumerate(sorted(list(all_item_types)))}

    def process_human_history(history):
        processed_steps = [{"step": 0, "placedItems": [], "utilization": 0}]
        for i, state in enumerate(history[1:], 1):
            step_data = {"step": i, "placedItems": []}
            for block, used_space in zip(state.block_list, state.used_space_list):
                nx, ny, nz = block.stack_count_xyz; ulx, uly, ulz = block.unit_size; ox, oy, oz = used_space.x, used_space.y, used_space.z
                type_id = f"item_{int(ulx)}x{int(uly)}x{int(ulz)}"
                for k in range(nz):
                    for j in range(ny):
                        for i_ in range(nx):
                            step_data["placedItems"].append({"x":ox+i_*ulx, "y":oy+j*uly, "z":oz+k*ulz, "lx":ulx, "ly":uly, "lz":ulz, "typeId":type_id})
            used_vol = sum(it['lx']*it['ly']*it['lz'] for it in step_data["placedItems"])
            step_data["utilization"] = (used_vol / state.truck_type.volume * 100) if state.truck_type.volume > 0 else 0
            processed_steps.append(step_data)
        return processed_steps

    def process_llm_history(history):
        processed_steps = [{"step": 0, "placedItems": [], "utilization": 0}]
        truck_vol = truck_dims['length'] * truck_dims['width'] * truck_dims['height']
        for i, state in enumerate(history[1:], 1):
            step_data = {"step": i, "placedItems": []}
            used_vol = 0
            for item in state[0]['occupied_volumes']:
                _, x, y, z, lx, ly, lz = item
                type_id = f"item_{int(lx)}x{int(ly)}x{int(lz)}"
                step_data["placedItems"].append({"x":x, "y":y, "z":z, "lx":lx, "ly":ly, "lz":lz, "typeId":type_id})
                used_vol += lx*ly*lz
            step_data["utilization"] = (used_vol / truck_vol * 100) if truck_vol > 0 else 0
            processed_steps.append(step_data)
        return processed_steps

    final_json_data = {
        "truck": truck_dims, "itemTypes": item_type_colors,
        "human": {"steps": process_human_history(human_history)}, "llm": {"steps": process_llm_history(llm_history)},
    }
    with open(json_filename, "w") as f:
        json.dump(final_json_data, f, indent=2)
    print(f"✅ Successfully exported comparison data to '{json_filename}'")
    create_comparison_html_viewer(html_filename, json_filename)

def create_comparison_html_viewer(html_filename, json_filename):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Packing Algorithm Comparison</title>
    <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif; background-color: #f0f2f5; color: #333; display: flex; flex-direction: column; height: 100vh; }}
        #header {{ padding: 10px 20px; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; z-index: 10; }}
        #controls {{ display: flex; justify-content: center; align-items: center; gap: 15px; }}
        button {{ font-size: 1em; padding: 10px 20px; cursor: pointer; border-radius: 5px; border: 1px solid #ccc; background-color: #fff; transition: background-color 0.2s; }}
        button:hover:not(:disabled) {{ background-color: #e8e8e8; }}
        button:disabled {{ background-color: #f0f0f0; color: #aaa; cursor: not-allowed; }}
        #step-counter {{ font-size: 1.2em; font-weight: 500; min-width: 150px; text-align: center; }}
        #main-content {{ display: flex; flex: 1; }}
        .view-container {{ flex: 1; display: flex; flex-direction: column; border-left: 1px solid #ddd; }}
        .view-container:first-child {{ border-left: none; }}
        .view-header {{ padding: 12px; background: #f7f7f7; text-align: center; border-bottom: 1px solid #ddd; }}
        .view-header h2 {{ margin: 0; font-size: 1.1em; }}
        .stats {{ font-size: 0.9em; color: #555; margin-top: 4px; }}
        .canvas-wrapper {{ flex: 1; position: relative; overflow: hidden; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Packing Algorithm Comparison</h1>
        <div id="controls">
            <button id="prev-btn">« Previous</button>
            <span id="step-counter">Step: 1 / 1</span>
            <button id="next-btn">Next »</button>
        </div>
    </div>
    <div id="main-content">
        <div class="view-container">
            <div class="view-header">
                <h2>Human Heuristic</h2>
                <div class="stats" id="human-stats">Items: 0, Utilization: 0.00%</div>
            </div>
            <div class="canvas-wrapper" id="human-container"></div>
        </div>
        <div class="view-container">
            <div class="view-header">
                <h2>LLM Heuristic</h2>
                <div class="stats" id="llm-stats">Items: 0, Utilization: 0.00%</div>
            </div>
            <div class="canvas-wrapper" id="llm-container"></div>
        </div>
    </div>

    <script type="importmap">{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}</script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        let comparisonData, currentStep = 0;
        const scenes = {{}};

        async function main() {{
            try {{
                const response = await fetch('{json_filename}');
                comparisonData = await response.json();
            }} catch (e) {{
                document.body.innerHTML = `Error loading {json_filename}. Ensure it exists.`;
                return;
            }}

            scenes.human = setupScene('human-container');
            scenes.llm = setupScene('llm-container');
            
            drawTruckContainers();

            document.getElementById('next-btn').addEventListener('click', () => updateStep(1));
            document.getElementById('prev-btn').addEventListener('click', () => updateStep(-1));

            renderCurrentStep();
            animate();
            window.addEventListener('resize', onWindowResize);
        }}

        function setupScene(containerId) {{
            const container = document.getElementById(containerId);
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);
            const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 20000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            const light = new THREE.DirectionalLight(0xffffff, 1);
            light.position.set(1, 1.5, 0.5).normalize();
            scene.add(light);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            const placedObjects = new THREE.Group();
            scene.add(placedObjects);

            return {{ scene, camera, renderer, controls, placedObjects, container }};
        }}

        function drawTruckContainers() {{
            const truck = comparisonData.truck;
            const geometry = new THREE.BoxGeometry(truck.length, truck.height, truck.width);
            const edges = new THREE.EdgesGeometry(geometry);
            const lineMaterial = new THREE.LineBasicMaterial({{ color: 0x333333, transparent: true, opacity: 0.5 }});
            
            Object.values(scenes).forEach(s => {{
                const line = new THREE.LineSegments(edges, lineMaterial.clone());
                line.position.set(truck.length / 2, truck.height / 2, truck.width / 2);
                s.scene.add(line);
                s.camera.position.set(truck.length * 1.4, truck.height * 1.4, truck.width * 1.4);
                s.controls.target.set(truck.length / 2, truck.height / 3, truck.width / 2);
                s.controls.update();
            }});
        }}
        
        function updateStep(delta) {{
            const maxSteps = Math.max(comparisonData.human.steps.length, comparisonData.llm.steps.length);
            const newStep = currentStep + delta;
            if (newStep >= 0 && newStep < maxSteps) {{
                currentStep = newStep;
                renderCurrentStep();
            }}
        }}

        function renderCurrentStep() {{
            renderAlgorithmStep('human', scenes.human);
            renderAlgorithmStep('llm', scenes.llm);

            const maxSteps = Math.max(comparisonData.human.steps.length, comparisonData.llm.steps.length);
            // *** BUG FIX IS HERE ***
            // Using {{...}} to escape braces for Python's f-string, so they are interpreted by JS.
            document.getElementById('step-counter').textContent = `Step: ${{currentStep + 1}} / ${{maxSteps}}`;
            document.getElementById('prev-btn').disabled = currentStep === 0;
            document.getElementById('next-btn').disabled = currentStep >= maxSteps - 1;
        }}

        function renderAlgorithmStep(algoName, scenePackage) {{
            const {{ placedObjects }} = scenePackage;
            while(placedObjects.children.length > 0) {{ placedObjects.remove(placedObjects.children[0]); }}

            const steps = comparisonData[algoName].steps;
            const stepIdx = Math.min(currentStep, steps.length - 1);
            const stepData = steps[stepIdx];

            if (!stepData) return;

            stepData.placedItems.forEach(item => {{
                const geometry = new THREE.BoxGeometry(item.lx, item.lz, item.ly);
                const color = comparisonData.itemTypes[item.typeId] || '#999';
                const material = new THREE.MeshStandardMaterial({{ color, metalness: 0.1, roughness: 0.7 }});
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(item.x + item.lx / 2, item.z + item.lz / 2, item.y + item.ly / 2);
                placedObjects.add(mesh);
                
                const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({{ color: 0x000, opacity: 0.25, transparent: true }}));
                edges.position.copy(mesh.position);
                placedObjects.add(edges);
            }});
            
            // *** BUG FIX IS HERE ***
            document.getElementById(`${{algoName}}-stats`).textContent = `Items: ${{stepData.placedItems.length}}, Utilization: ${{stepData.utilization.toFixed(2)}}%`;
        }}

        function onWindowResize() {{
            Object.values(scenes).forEach(s => {{
                s.camera.aspect = s.container.clientWidth / s.container.clientHeight;
                s.camera.updateProjectionMatrix();
                s.renderer.setSize(s.container.clientWidth, s.container.clientHeight);
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            Object.values(scenes).forEach(s => {{
                s.controls.update();
                s.renderer.render(s.scene, s.camera);
            }});
        }}

        main();
    </script>
</body>
</html>
    """
    with open(html_filename, "w") as f:
        f.write(html_content)
    print(f"✅ Successfully created comparison viewer '{html_filename}'")

# ==============================================================================
# SECTION 5: MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    DATA_PATH = "data/test/EF1596102693660"

    with open(DATA_PATH, "r") as f: data = json.load(f)

    boxes = data.get("boxes", [])
    if len(boxes) > 50:
        boxes = random.sample(boxes, 50)  # Randomly select 40 items
    else:
        boxes = boxes  # Use all items if less than 40
    truck_type_map = data.get("algorithmBaseParamDto", {}).get("truckTypeMap", {})
    
    human_truck_types = [TruckType(t["truckTypeId"], t["truckTypeCode"], t["truckTypeName"], t["length"], t["width"], t["height"], t["maxLoad"]) for t in truck_type_map.values()]
    llm_truck_types_info = [(t["maxLoad"], t["length"], t["width"], t["height"]) for t in sorted(truck_type_map.values(), key=lambda x: x["length"]*x["width"]*x["height"])]

    print("Running Human-Designed Algorithm...")
    human_history = run_human_packer_step_by_step(boxes, human_truck_types)
    print(f"  -> Captured {len(human_history)} steps.")

    print("Running LLM-Designed Algorithm...")
    llm_history = run_llm_packer_step_by_step(boxes, llm_truck_types_info)
    print(f"  -> Captured {len(llm_history)} steps.")

    generate_comparison_visualization(human_history, llm_history)
    print("\n🚀 All done! Open 'packing_comparison.html' in a web browser to view the results.")