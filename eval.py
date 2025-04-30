from block_packing import PackingCONST, TruckType, get_size_box, LoadingTruck
import uuid as UUID

import warnings
import types
import sys
import numpy as np

packing_problem = PackingCONST()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import copy
import glob
import json


def generate_instances():
    files = glob.glob("data/test/*")
    instance_data = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        # Extract truck types
        truck_type_map = data.get("algorithmBaseParamDto", {}).get("truckTypeMap", {})
        truck_types = []
        for truck_id, truck in truck_type_map.items():
            truck_types.append(
                TruckType(
                    truck["truckTypeId"],
                    truck["truckTypeCode"],
                    truck["truckTypeName"],
                    truck["length"],
                    truck["width"],
                    truck["height"],
                    truck["maxLoad"],
                )
            )
        truck_type_list = sorted(truck_types, key=lambda x: x.volume, reverse=True)
        boxes = data.get("boxes", [])
        for box in boxes:
            box["uuid"] = UUID.uuid4()
        size_box = get_size_box(boxes)
        instance_data.append((truck_type_list, size_box))  # Use size_box dict

    return instance_data


packing_problem.instance_data = generate_instances()

example_code = "import numpy as np\n\ndef select_block(block_list1, block_list2, space, left_weight):\n    \"\"\"{This algorithm prioritizes blocks that fill the largest volume fraction of the space while remaining under the weight limit, then selects the block with highest weight utilization among these volume-efficient blocks.}\"\"\"\n    best_block = None\n    best_utilization = -1\n\n    # Combine block lists for easier iteration\n    all_blocks = block_list1 + block_list2\n\n    # Calculate space volume\n    space_volume = space['lx'] * space['ly'] * space['lz']\n\n    # First pass: Find blocks that maximize volume utilization within constraints\n    candidate_blocks = []\n    max_volume_utilization = -1\n\n    for block in all_blocks:\n        # Calculate block dimensions\n        block_lx = block['stack_count_xyz'][0] * block['unit_size'][0]\n        block_ly = block['stack_count_xyz'][1] * block['unit_size'][1]\n        block_lz = block['stack_count_xyz'][2] * block['unit_size'][2]\n\n        # Check if block fits and respects weight limit\n        if block_lx <= space['lx'] and block_ly <= space['ly'] and block_lz <= space['lz'] and block['sum_weight'] <= left_weight:\n            volume_utilization = block['volume'] / space_volume\n            if volume_utilization > max_volume_utilization:\n                candidate_blocks = [block]\n                max_volume_utilization = volume_utilization\n            elif volume_utilization == max_volume_utilization:\n                candidate_blocks.append(block)\n\n    # Second pass: Select the best block from the candidates based on weight utilization\n    for block in candidate_blocks:\n        weight_utilization = block['sum_weight'] / left_weight\n        utilization = min(max_volume_utilization, weight_utilization)\n\n        if utilization > best_utilization:\n            best_utilization = utilization\n            best_block = block\n\n    return block"
print(example_code)
import numpy as np

def evaluate(self, code_string):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        packing_module = types.ModuleType("packing_module")
        exec(code_string, packing_module.__dict__)
        sys.modules[packing_module.__name__] = packing_module

        total_utilization = 0
        for truck_type, size_box in self.instance_data:  # Unpack directly
            instance_utilization = 0
            instance_truck_number = 0
            while len(size_box) > 0:
                last_loading_truck = LoadingTruck(truck_type[0])
                utilization = self.single_truck_packing_eval(
                    last_loading_truck, size_box, packing_module
                )
                switch_smaller_vehicle_result = self.switch_smaller_vehicle(truck_type, last_loading_truck, packing_module)
                if switch_smaller_vehicle_result is not None:
                    last_loading_truck, utilization = switch_smaller_vehicle_result
                instance_utilization += utilization
                instance_truck_number += 1

            total_utilization += instance_utilization / instance_truck_number
        average_utilization = total_utilization / len(self.instance_data)
        return 1 - average_utilization

fitness = evaluate(packing_problem, example_code)
print(fitness)
