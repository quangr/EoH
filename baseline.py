import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# packing (Your provided code -  I've kept it exactly as you provided)
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
            return space.x + space.y + space.z  # 选空间的规则

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


def select_block(block_list1, block_list2):
    """
    Selects the best block from the given lists based on volume and unit_size[1].
    Prioritizes based on (volume, unit_size[1]) tuple, similar to original __gt__.
    """
    best_block1 = (
        max(block_list1, key=lambda b: (b.volume, b.unit_size[1]))
        if block_list1
        else None
    )
    best_block2 = (
        max(block_list2, key=lambda b: (b.volume, b.unit_size[1]))
        if block_list2
        else None
    )

    if best_block2 is None:
        return best_block1
    if best_block1 is None:
        return best_block2

    if best_block1.volume > best_block2.volume:
        return best_block1
    else:
        return best_block2


def single_truck_packing(loading_truck, size_box):
    total_volume = loading_truck.truck_type.volume
    total_weight = loading_truck.truck_type.maxLoad
    # 根据6要素设计的单车厢装箱过程
    # note: 会改变loading_truck内部变量以及size_box_dict
    while len(loading_truck.free_space_list) > 0 and len(size_box) > 0:
        # 1. 选择空间 K3  argmin(x+y+z)
        # 2. 构建块 K2
        # 3. 选择块 K4
        # 4. 放置块，更新空间, 更新剩余箱子, 更新货车的相关状态 K5
        space = loading_truck.free_space_list[-1]
        # space_index, space = min(enumerate(loading_truck.free_space_list), key=lambda x: (x[1], x[0]))
        block_list1, block_list2 = get_block_list(
            space,
            size_box,
            loading_truck.left_weight,
            loading_truck.used_space_list,
            loading_truck.truck_type,
        )

        if len(block_list1) == 0 and len(block_list2) == 0:
            # loading_truck.discard_space_list.append(space) #.insert(0, chooseSpace)?
            loading_truck.discard_space_list.insert(0, space)
            loading_truck.free_space_list.pop()
            continue

        block = select_block(block_list1, block_list2)
        update_info(loading_truck, size_box, space, block)  # 下次debug从这里开始看
    return max(
        loading_truck.sum_used_space / total_volume,
        1 - loading_truck.left_weight / total_weight,
    )


# Visualization Function
def visualize_packing(loading_truck: LoadingTruck):
    """Visualizes the packing results using matplotlib."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits based on truck dimensions
    ax.set_xlim([0, loading_truck.truck_type.length])
    ax.set_ylim([0, loading_truck.truck_type.width])
    ax.set_zlim([0, loading_truck.truck_type.height])

    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    colors = ["r", "g", "b", "y", "c", "m"]  # Add more colors if needed
    color_index = 0

    # Iterate through used spaces and plot blocks
    for i, space in enumerate(loading_truck.used_space_list):
        # Define vertices of the block
        vertices = [
            [space.x, space.y, space.z],
            [space.x + space.lx, space.y, space.z],
            [space.x + space.lx, space.y + space.ly, space.z],
            [space.x, space.y + space.ly, space.z],
            [space.x, space.y, space.z + space.lz],
            [space.x + space.lx, space.y, space.z + space.lz],
            [space.x + space.lx, space.y + space.ly, space.z + space.lz],
            [space.x, space.y + space.ly, space.z + space.lz],
        ]

        # Define faces of the block
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]

        color = colors[color_index % len(colors)]
        color_index += 1

        # Plot each face
        poly3d = Poly3DCollection(
            faces, linewidths=1, edgecolors="k", alpha=0.5, facecolor=color
        )
        ax.add_collection3d(poly3d)

    # Add title and legend if necessary
    plt.title("3D Packing Visualization")
    plt.show()


# Example usage (with visualization):


import uuid as UUID


def get_platform_box(box_list):
    platform_box = {}
    for box in box_list:
        if box["platformCode"] not in platform_box:
            platform_box[box["platformCode"]] = []
        # 增加一个key值,因为spuID不是唯一的
        box["uuid"] = UUID.uuid4()
        platform_box[box["platformCode"]].append(box)
    return platform_box


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


import glob
import json


def switch_smaller_vehicle(truck_type_list, truck):
    smaller_truck_type_list = list(filter(
        lambda x: truck.sum_used_space <= x.volume < truck.truck_type.volume and 
                  x.maxLoad >= truck.truck_type.maxLoad - truck.left_weight,
        truck_type_list
    ))
    smaller_truck_type_list.sort(key=lambda x: x.volume)
    
    route = truck.visited_platform
    platform_box = get_platform_box(truck.box_list)
    
    for truck_type in smaller_truck_type_list:
        loading_truck = LoadingTruck(truck_type)
        
        for platform in route:
            size_box = get_size_box(platform_box[platform])
            utilization = single_truck_packing(loading_truck, size_box)
            
            if len(size_box) > 0:
                break
        else:
            return loading_truck, utilization  # Return the new truck and its utilization
    
    return None


files = glob.glob("data/test/*")
all_metrics = []
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

    items = []
    # Extract items (boxes)
    boxes = data.get("boxes", [])
    all_platform = [box["platformCode"] for box in boxes]
    box_list = boxes
    loading_truck_list = []
    last_loading_truck = None
    for box in box_list:
        box["uuid"] = UUID.uuid4()
    size_box = get_size_box(box_list)
    total_utilization = 0
    total_truck_number = 0
    while len(size_box) > 0:
        last_loading_truck = LoadingTruck(truck_type_list[0])
        utilization = single_truck_packing(last_loading_truck, size_box)
        switch_smaller_vehicle_result = switch_smaller_vehicle(truck_type_list, last_loading_truck)
        if switch_smaller_vehicle_result is not None:
            last_loading_truck, utilization = switch_smaller_vehicle_result
        total_utilization += utilization
        total_truck_number+=1
    average_utilization = total_utilization / total_truck_number
    all_metrics.append(1 - average_utilization)
    print(file)
print(np.mean(all_metrics))
