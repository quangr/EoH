import numpy as np
import uuid as UUID
import types
import sys
import warnings

# --- Existing Packing Code (with modifications) ---

SUPPORT_RATIO = 0.8
BEYOND_BETA = 150


class TruckType:
    def __init__(
        self, truckTypeId, truckTypeCode, truckTypeName, length, width, height, maxLoad
    ):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.volume = length * width * height


class Space:
    def __init__(self, x, y, z, lx, ly, lz):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def __eq__(self, other):
        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.lx == other.lx
            and self.ly == other.ly
            and self.lz == other.lz
        )

    def __lt__(self, other):
        def spaces_corner_sum(space):
            return space.x + space.y + space.z  #  选空间的规则

        def space_xyz(space):
            return space.x, space.y, space.z

        def space_xyz_2(space):
            return space.x, space.y, space.z, -(space.lx * space.ly * space.lz)

        return space_xyz(self) < space_xyz(other)


class LoadingTruck:
    def __init__(self, truck_type):
        self.truck_type = truck_type
        self.free_space_list = [
            Space(
                0,
                0,
                0,
                self.truck_type.length,
                self.truck_type.width,
                self.truck_type.height,
            )
        ]
        self.discard_space_list = []  # [Space]
        self.used_space_list = []  # [Space] 对应block_list中block使用的空间
        self.block_list = []  # [Block]
        self.left_weight = self.truck_type.maxLoad
        self.visited_platform = []  # 为了求PF方便以及写入数据方便
        self.sum_used_space = 0  # 总使用体积
        self.box_list = []  # 为了换小车的


class Block:
    def __init__(self, stack_count_xyz, unit_size, sum_weight, box_list):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.volume = (
            unit_size[0]
            * stack_count_xyz[0]
            * unit_size[1]
            * stack_count_xyz[1]
            * unit_size[2]
            * stack_count_xyz[2]
        )

    def __gt__(self, other):
        return (self.volume, self.unit_size[1]) > (other.volume, other.unit_size[1])


def get_block_list(space, size_box, left_weight, used_space_list, truck_type):
    block_list1 = []
    filter_size_box = filter(
        lambda one: one[0] <= space.lx and one[1] <= space.ly and one[2] <= space.lz,
        size_box,
    )
    for size in filter_size_box:
        box_list = size_box[size]
        # 按照现在的box顺序,最多可以放下多少箱子的重量
        pre_sum_weigt = []
        for max_num, box in enumerate(box_list):
            pre_sum_weigt.append(
                box["weight"] + (pre_sum_weigt[max_num - 1] if max_num > 0 else 0)
            )
            if left_weight < pre_sum_weigt[-1]:
                break
        else:
            max_num = len(box_list)
        # 空间约束
        nx, ny, nz = (
            int(space.lx / size[0]),
            int(space.ly / size[1]),
            int(space.lz / size[2]),
        )
        if max_num <= ny:
            ny = max_num
            nz = nx = 1
        elif max_num <= ny * nz:
            nz = int(max_num / ny)
            nx = 1
        elif max_num <= nx * ny * nz:
            nx = int(max_num / (ny * nz))
        if nx * ny * nz > 0:
            block_list1.append(
                Block(
                    (nx, ny, nz),
                    size,
                    pre_sum_weigt[nx * ny * nz - 1],
                    box_list[: nx * ny * nz],
                )
            )
    block_list2 = []
    # 再次遍历所有size, 寻找超大快 #发现之前的代码的一个bug，没判断超出车厢
    # 否定条件是 lz 超过 lz, 或者超出车厢，或者完全被包住
    filter_size_box = filter(
        lambda one: not (
            one[2] > space.lz
            or space.x + one[0] > truck_type.length
            or space.y + one[1] > truck_type.width
            or (one[0] <= space.lx and one[1] <= space.ly)
        ),
        size_box,
    )
    for size in filter_size_box:
        box_list = size_box[size][:]
        box_list.sort(key=lambda x: x["weight"], reverse=True)
        # weight 和 space约束
        if (
            box_list[0]["weight"] <= left_weight
            and size[0] - space.lx < BEYOND_BETA
            and space.lx / size[0] > SUPPORT_RATIO
            and size[1] - space.ly < BEYOND_BETA
            and space.ly / size[1] > SUPPORT_RATIO
            and (min(size[0], space.lx) * min(size[1], space.ly)) / (size[0] * size[1])
            > SUPPORT_RATIO
        ):
            # 判断与其他已用空间有没有遮挡以及重叠
            now_space = Space(space.x, space.y, space.z, size[0], size[1], size[2])
            is_ok = True
            for used_space in used_space_list:
                if check_overlap_3d(now_space, used_space) or (
                    used_space.x > now_space.x
                    and check_overlap_2d(now_space, used_space)
                ):
                    is_ok = False
                    break
            if is_ok:
                block_list2.append(
                    Block((1, 1, 1), size, box_list[0]["weight"], [box_list[0]])
                )
    return block_list1, block_list2


# import numba
# @numba.jit()
def get_overlap_space(space1, space2):
    # 求两个空间的交集(可能交集不存在，此时对应轴长<=0)
    overlap_space = dict(
        x=max(space1.x, space2.x),
        y=max(space1.y, space2.y),
        z=max(space1.z, space2.z),
        lx=max(
            min(space1.x + space1.lx, space2.x + space2.lx) - max(space1.x, space2.x), 0
        ),
        ly=max(
            min(space1.y + space1.ly, space2.y + space2.ly) - max(space1.y, space2.y), 0
        ),
        lz=max(
            min(space1.z + space1.lz, space2.z + space2.lz) - max(space1.z, space2.z), 0
        ),
    )
    return overlap_space


def check_overlap_3d(space1, space2):
    return not (
        space1.x >= space2.x + space2.lx
        or space1.x + space1.lx <= space2.x
        or space1.y >= space2.y + space2.ly
        or space1.y + space1.ly <= space2.y
        or space1.z >= space2.z + space2.lz
        or space1.z + space1.lz <= space2.z
    )


def check_overlap_2d(space1, space2):
    return not (
        space1.y >= space2.y + space2.ly
        or space1.y + space1.ly <= space2.y
        or space1.z >= space2.z + space2.lz
        or space1.z + space1.lz <= space2.z
    )


def update_space_list_by_block_space(block_space, space_list):
    updated_space_list = []
    for pre_space in space_list:
        if check_overlap_3d(block_space, pre_space):
            deltaY = block_space.y + block_space.ly - pre_space.y
            deltaX = block_space.x + block_space.lx - pre_space.x
            if deltaY < min(
                BEYOND_BETA, block_space.ly / SUPPORT_RATIO - block_space.ly
            ):
                tmpLy = pre_space.ly - deltaY
                pre_space.ly = tmpLy
                pre_space.y = block_space.y + block_space.ly

            if deltaX < min(
                BEYOND_BETA, block_space.lx / SUPPORT_RATIO - block_space.lx
            ):
                tmpLx = pre_space.lx - deltaX
                pre_space.lx = tmpLx
                pre_space.x = block_space.x + block_space.lx

            if pre_space.ly <= 0 or pre_space.lx <= 0:
                continue
        if pre_space.x < block_space.x and check_overlap_2d(block_space, pre_space):
            deltaY = block_space.y + block_space.ly - pre_space.y
            tmpLy = pre_space.ly - deltaY
            pre_space.ly = tmpLy
            pre_space.y = block_space.y + block_space.ly

        if pre_space.ly > 0 and pre_space.lx > 0:
            updated_space_list.append(pre_space)
    return updated_space_list


def get_refresh_spaces(block_space, discard_space_list):
    refresh_spaces_list = []
    for space in discard_space_list:
        if check_overlap_2d(block_space, space):
            space.ly = space.ly - (block_space.y + block_space.ly - space.y)
            space.y = block_space.y + block_space.ly
            if space.ly > 0:
                refresh_spaces_list.append(space)
    discard_space_list = filter(lambda space: space.ly > 0, discard_space_list)
    return refresh_spaces_list, discard_space_list


def update_info(loading_truck, size_box, space, block):
    if (
        len(loading_truck.visited_platform) == 0
        or loading_truck.visited_platform[-1] != block.box_list[0]["platformCode"]
    ):
        loading_truck.visited_platform.append(block.box_list[0]["platformCode"])
    # 更新装载的箱子列表
    loading_truck.box_list.extend(block.box_list)
    # 车辆存上block
    loading_truck.block_list.append(block)
    # 更新车辆已用空间
    block_space = Space(
        space.x,
        space.y,
        space.z,
        *tuple(x * y for x, y in zip(block.unit_size, block.stack_count_xyz))
    )
    loading_truck.used_space_list.append(block_space)
    loading_truck.sum_used_space += block_space.lx * block_space.ly * block_space.lz
    # 更新剩余可装载重量
    loading_truck.left_weight -= block.sum_weight
    # 更新剩余箱子 # 要根据uuid更新
    tmp_box_id_dict = {box["uuid"]: True for box in block.box_list}
    sizes = [block.unit_size]
    if block.unit_size[0] != block.unit_size[1]:
        sizes.append((block.unit_size[1], block.unit_size[0], block.unit_size[2]))
    for size in sizes:
        if size not in size_box:
            continue
        size_box[size] = list(
            filter(lambda box: box["uuid"] not in tmp_box_id_dict, size_box[size])
        )
        if len(size_box[size]) == 0:
            del size_box[size]
    loading_truck.free_space_list.pop()
    # refresh_space用于后面合并空间
    refresh_spaces_list, loading_truck.discard_space_list = get_refresh_spaces(
        block_space, loading_truck.discard_space_list
    )
    # 遍历之前的判断重叠
    loading_truck.free_space_list = update_space_list_by_block_space(
        block_space, loading_truck.free_space_list
    )
    loading_truck.discard_space_list = update_space_list_by_block_space(
        block_space, loading_truck.discard_space_list
    )

    # 判断完之后把新生成的加上
    deltaX = space.lx - block_space.lx
    deltaY = space.ly - block_space.ly
    deltaZ = space.lz - block_space.lz
    if deltaX > 0:
        space1 = Space(
            block_space.x + block_space.lx,
            block_space.y,
            block_space.z,
            deltaX,
            space.ly,
            space.lz,
        )
        loading_truck.free_space_list.append(space1)
    if deltaY > 0:
        space2 = Space(
            block_space.x,
            block_space.y + block_space.ly,
            block_space.z,
            min(block_space.lx, space.lx),
            deltaY,
            space.lz,
        )
        for pre_space in refresh_spaces_list:
            if (
                pre_space.x + pre_space.lx == space2.x
                and pre_space.y == space2.y
                and pre_space.z == space2.z
                and pre_space.ly == space2.ly
                and pre_space.lz == space2.lz
            ):
                space2.x = pre_space.x
                space2.lx = space2.lx + pre_space.lx
                loading_truck.discard_space_list.remove(pre_space)
        loading_truck.free_space_list.append(space2)

    if deltaZ > 0:
        space3 = Space(
            block_space.x,
            block_space.y,
            block_space.z + block_space.lz,
            block_space.lx,
            block_space.ly,
            deltaZ,
        )
        if block_space.lx > space.lx or block_space.ly > space.ly:
            overlapRects = list(
                filter(
                    lambda boxSpace: check_overlap_3d(space3, boxSpace),
                    loading_truck.used_space_list,
                )
            )
            if len(overlapRects) > 0:
                space3.ly = min(overlapRects, key=lambda x: x.y).y - space3.y
        loading_truck.free_space_list.append(space3)


import glob
import json


def get_size_box(box_list):
    size_box = {}
    for box in box_list:
        sizes = [(box["length"], box["width"], box["height"])]
        if box["length"] != box["width"]:
            sizes.append((box["width"], box["length"], box["height"]))
        for size in sizes:
            if size not in size_box:
                size_box[size] = []
            size_box[size].append(box)
    return size_box


class GetData:
    def __init__(self, n_instance):
        self.n_instance = n_instance

    def generate_instances(self):
        files = glob.glob("data/train/*")
        instance_data = []
        for file in files[: self.n_instance]:
            with open(file, "r") as f:
                data = json.load(f)

            # Extract truck types
            truck_type_map = data.get("algorithmBaseParamDto", {}).get(
                "truckTypeMap", {}
            )
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


class GetPrompts:
    def __init__(self):
        self.prompt_task = """Select the best block to place in a given 3D space.  The goal is to maximize the loading rate which is max(volume_utilization, weight_utilization within the truck,
        considering both 'block_list1' (regular blocks) and 'block_list2' (irregular, larger blocks).
        Blocks consist of multiple boxes of the same dimensions stacked together.
        Output the selected block (from 'block_list1' or 'block_list2')."""
        self.prompt_func_name = "select_block"
        self.prompt_func_inputs = ["block_list1", "block_list2", "space", "left_weight"]
        self.prompt_func_outputs = ["block"]  # Clarified outputs
        self.prompt_inout_inf = """
        'block_list1' and 'block_list2' are lists of dictionaries.  Each dictionary in the lists represents a block and MUST have the following keys:
        - 'stack_count_xyz': a tuple (nx, ny, nz) representing how many boxes are stacked in each dimension.
        - 'unit_size': a tuple (length, width, height) representing the dimensions of a single box within the block.
        - 'sum_weight': the total weight of the block.
        - 'volume': The total volume of the block.
        - 'box_list': a list of dictionaries, each representing a box in the block (with 'uuid', 'length', 'width', 'height', 'weight', 'platformCode').

        'space' is a dictionary representing the current free space to be filled and MUST have the following keys:
        - 'x': start x coordinate of the space.
        - 'y': start y coordinate of the space.
        - 'z': start z coordinate of the space.
        - 'lx': length of the space in x dimension.
        - 'ly': length of the space in y dimension.
        - 'lz': length of the space in z dimension.

        'left_weight' is the remaining weight capacity of the truck.

        'block' is a dictionary representing the selected block and MUST from 'block_list1' or 'block_list2'.
        """
        self.prompt_other_inf = "Use NumPy arrays where appropriate."

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


def get_platform_box(box_list):
    platform_box = {}
    for box in box_list:
        if box["platformCode"] not in platform_box:
            platform_box[box["platformCode"]] = []
        # 增加一个key值,因为spuID不是唯一的
        box["uuid"] = UUID.uuid4()
        platform_box[box["platformCode"]].append(box)
    return platform_box


class PackingCONST:
    def __init__(self, n_instance=100, running_time=10):
        self.n_instance = n_instance
        self.running_time = running_time
        self.prompts = GetPrompts()

        getData = GetData(self.n_instance)  # Pass boxes here
        self.instance_data = getData.generate_instances()

    def convert_blocks_to_dicts(self, block_list):
        """Converts a list of Block objects to a list of dictionaries."""
        return [
            {
                "stack_count_xyz": block.stack_count_xyz,
                "unit_size": block.unit_size,
                "sum_weight": block.sum_weight,
                "volume": block.volume,
                "box_list": block.box_list,  # Already a list of dicts
            }
            for block in block_list
        ]

    def convert_space_to_dict(self, space):
        """Converts a Space object to a dictionary."""
        return {
            "x": space.x,
            "y": space.y,
            "z": space.z,
            "lx": space.lx,
            "ly": space.ly,
            "lz": space.lz,
        }

    def switch_smaller_vehicle(self, truck_type_list, truck, eva):
        smaller_truck_type_list = list(
            filter(
                lambda x: truck.sum_used_space <= x.volume < truck.truck_type.volume
                and x.maxLoad >= truck.truck_type.maxLoad - truck.left_weight,
                truck_type_list,
            )
        )
        smaller_truck_type_list.sort(key=lambda x: x.volume)

        route = truck.visited_platform
        platform_box = get_platform_box(truck.box_list)

        for truck_type in smaller_truck_type_list:
            loading_truck = LoadingTruck(truck_type)

            for platform in route:
                size_box = get_size_box(platform_box[platform])
                utilization = self.single_truck_packing_eval(loading_truck, size_box, eva)

                if len(size_box) > 0:
                    break
            else:
                return (
                    loading_truck,
                    utilization,
                )  # Return the new truck and its utilization

        return None

    def single_truck_packing_eval(self, loading_truck, size_box, eva):
        total_volume = loading_truck.truck_type.volume
        total_weight = loading_truck.truck_type.maxLoad
        while loading_truck.free_space_list and size_box:
            space = loading_truck.free_space_list[-1]
            block_list1, block_list2 = get_block_list(
                space,
                size_box,
                loading_truck.left_weight,
                loading_truck.used_space_list,
                loading_truck.truck_type,
            )

            # Convert Block objects to dictionaries for the LLM
            block_list1_dicts = self.convert_blocks_to_dicts(block_list1)
            block_list2_dicts = self.convert_blocks_to_dicts(block_list2)
            space_dict = self.convert_space_to_dict(
                space
            )  # Convert Space object to dictionary

            if not block_list1_dicts and not block_list2_dicts:
                loading_truck.discard_space_list.insert(0, space)
                loading_truck.free_space_list.pop()
                continue
            try:
                selected_block = eva.select_block(
                    block_list1_dicts,
                    block_list2_dicts,
                    space_dict,
                    loading_truck.left_weight,
                )  # Pass space_dict here
            except Exception as e:
                print("Error during eva.select_block:", e)
                return 0  # Indicate failure
            selected_block = Block(
                selected_block["stack_count_xyz"],
                selected_block["unit_size"],
                selected_block["sum_weight"],
                selected_block["box_list"],
            )

            update_info(loading_truck, size_box, space, selected_block)
        return max(
            loading_truck.sum_used_space / total_volume,
            1 - loading_truck.left_weight / total_weight,
        )

    def evaluate(self, code_string):
        try:
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

        except Exception as e:
            print("Error during evaluation:", str(e))
            return None
