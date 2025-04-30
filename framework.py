from typing import List, Tuple, Dict, Callable, Any
import copy

# --- Data Structures (Keep these as they are) ---

class Truck:
    # ... (Truck class definition from previous response) ...
    def __init__(
        self,
        type_id: str,
        length: float,
        width: float,
        height: float,
        max_weight: float,
    ):
        self.type_id = type_id
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.items: List["Item"] = []  # Items loaded into this truck
        self.used_volume = 0.0
        self.used_weight = 0.0
        #added for compatibility
        self.volume = length * width * height

    def __str__(self):
        return (
            f"Truck(id={self.type_id}, l={self.length}, w={self.width}, "
            f"h={self.height}, max_weight={self.max_weight})"
        )
class Item:
    # ... (Item class definition from previous response) ...
    def __init__(
        self, item_id: str, length: float, width: float, height: float, weight: float
    ):
        self.item_id = item_id
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.x = 0.0  # Coordinates within the truck
        self.y = 0.0
        self.z = 0.0
        self.placed = False  # Flag to indicate if item is placed

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def bottom_area(self) -> float:
        return self.length * self.width

    def __str__(self):
        return (
            f"Item(id={self.item_id}, l={self.length}, w={self.width}, h={self.height}, "
            f"weight={self.weight}, x={self.x}, y={self.y}, z={self.z})"
        )

    def __repr__(self):
        return self.__str__()

class Space:
    # ... (Space class definition from previous response) ...
    def __init__(self, x: float, y: float, z: float, lx: float, ly: float, lz: float):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

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
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __str__(self):
        return f"Space(x={self.x}, y={self.y}, z={self.z}, lx={self.lx}, ly={self.ly}, lz={self.lz})"

    def __repr__(self):
        return self.__str__()

# --- Heuristic Function Types (Define these for clarity) ---

SpaceSelectionFunc = Callable[[List[Space], Truck, List[Item]], Space]
BlockGenerationFunc = Callable[[Space, List[Item], float, List[Space], Truck], List[Dict]]
BlockSelectionFunc = Callable[[List[Dict]], Dict]
UpdateInfoFunc = Callable[
    [Truck, List[Item], Space, Dict, List[Space], List[Space], List[Space]],
    Tuple[List[Item], List[Space], List[Space], List[Space]],
]

# --- The Generalized Greedy Framework ---

def greedy_packing(
    truck: Truck,
    items: List[Item],
    select_space: SpaceSelectionFunc,
    generate_blocks: BlockGenerationFunc,
    select_block: BlockSelectionFunc,
    update_info_func: UpdateInfoFunc,
) -> None:
    """
    A generalized greedy algorithm framework for 3D bin packing.

    Args:
        truck: The truck to pack.
        items: The list of items to pack.
        select_space: A function to select the next free space.
        generate_blocks: A function to generate potential blocks.
        select_block: A function to select the best block from the generated blocks.
        update_info_func: the function to update space information

    """
    free_space_list: List[Space] = [
        Space(0, 0, 0, truck.length, truck.width, truck.height)
    ]
    discard_space_list: List[Space] = []
    used_space_list: List[Space] = []
    remaining_items = copy.deepcopy(items)

    while free_space_list and remaining_items:
        # 1. Select a Space
        space = select_space(free_space_list, truck, remaining_items)
        if space is None:  # Handle cases where selection might fail
            break

        # 2. Generate Blocks
        block_list = generate_blocks(
            space,
            remaining_items,
            truck.max_weight - truck.used_weight,
            used_space_list,
            truck
        )

        # 3. Select a Block
        if not block_list:
            discard_space_list.append(space)
            free_space_list.remove(space) # Remove space if no block can be generated
            continue

        block = select_block(block_list)
        if block is None: # Handle cases where selection might fail
            discard_space_list.append(space)
            free_space_list.remove(space)
            continue

        # 4. Update Information (Place the Block)
        remaining_items, free_space_list, discard_space_list, used_space_list = update_info_func(
            truck, remaining_items, space, block, free_space_list, discard_space_list, used_space_list
        )
# --- Example Heuristic Functions (Replace these with your own) ---

def first_fit_select_space(free_space_list: List[Space], truck:Truck, items:List[Item]) -> Space:
    """Selects the first available space (no specific order)."""
    if not free_space_list:
        return None
    free_space_list.sort(reverse=True)  # Example: Sort by x, y, z (like original)
    return free_space_list[-1]

def simple_block_generation(
    space: Space, items: List[Item], left_weight: float, used_spaces:List[Space], truck:Truck
) -> List[Dict]:
    """Generates a single block if possible (very basic)."""
    # ... (Implementation similar to get_block_list, but simplified) ...
    # Group items by size, similar to the original get_block_list
    size_item_map: Dict[Tuple[float, float, float], List[Item]] = {}
    for item in items:
        size = (item.length, item.width, item.height)
        if size not in size_item_map:
            size_item_map[size] = []
        size_item_map[size].append(item)

    blocks = []
    for size, item_list in size_item_map.items():
        if size[0] <= space.lx and size[1] <= space.ly and size[2] <= space.lz:
            #Simplified to demonstrate
            nx, ny, nz = 1, 1, 1
            block_items = item_list[:nx * ny * nz]
            total_weight = sum(item.weight for item in block_items)

            if total_weight <= left_weight:
                blocks.append(
                    {
                        "stack_count": (nx, ny, nz),
                        "unit_size": size,
                        "total_weight": total_weight,
                        "items": block_items
                    }
                )
    return blocks

def largest_volume_select_block(block_list: List[Dict]) -> Dict:
    """Selects the block with the largest volume."""
    if not block_list:
        return None
    return max(
        block_list,
        key=lambda x: x["unit_size"][0]
        * x["unit_size"][1]
        * x["unit_size"][2]
        * x["stack_count"][0]
        * x["stack_count"][1]
        * x["stack_count"][2],
    )

# --- Example Usage ---
def check_overlap_3d(space1: Space, space2: Space) -> bool:
    """Checks if two 3D spaces overlap."""
    return not (
        space1.x >= space2.x + space2.lx
        or space1.x + space1.lx <= space2.x
        or space1.y >= space2.y + space2.ly
        or space1.y + space1.ly <= space2.y
        or space1.z >= space2.z + space2.lz
        or space1.z + space1.lz <= space2.z
    )


def check_overlap_2d(space1: Space, space2: Space) -> bool:
    """Checks if two 2D spaces (projected onto YZ plane) overlap."""
    return not (
        space1.y >= space2.y + space2.ly
        or space1.y + space1.ly <= space2.y
        or space1.z >= space2.z + space2.lz
        or space1.z + space1.lz <= space2.z
    )

def update_space_list_by_block_space(
    block_space: Space, space_list: List[Space]
) -> List[Space]:
    """Updates the list of free spaces after placing a block."""
    updated_space_list = []
    for pre_space in space_list:
        if check_overlap_3d(block_space, pre_space):
            deltaY = block_space.y + block_space.ly - pre_space.y
            deltaX = block_space.x + block_space.lx - pre_space.x

            if deltaY < min(BEYOND_BETA, block_space.ly / SUPPORT_RATIO - block_space.ly):
                tmpLy = pre_space.ly - deltaY
                pre_space.ly = tmpLy
                pre_space.y = block_space.y + block_space.ly

            if deltaX < min(BEYOND_BETA, block_space.lx / SUPPORT_RATIO - block_space.lx):
                tmpLx = pre_space.lx - deltaX
                pre_space.lx = tmpLx
                pre_space.x = block_space.x + block_space.lx
            #If the pre_space is completely inside, we should delete it.
            if pre_space.ly <= 0 or pre_space.lx <= 0:
                continue
        if pre_space.x < block_space.x and check_overlap_2d(block_space, pre_space):
            deltaY = block_space.y + block_space.ly - pre_space.y
            tmpLy = pre_space.ly - deltaY
            pre_space.ly = tmpLy
            pre_space.y = block_space.y + block_space.ly
        #Check if the space became invalid.
        if pre_space.ly > 0 and pre_space.lx > 0:
            updated_space_list.append(pre_space)
    return updated_space_list



def get_refresh_spaces(
    block_space: Space, discard_space_list: List[Space]
) -> Tuple[List[Space], List[Space]]:
    """Identifies spaces that can be refreshed (merged) after placing a block."""
    refresh_spaces_list = []
    new_discard_space_list = []
    for space in discard_space_list:
        if check_overlap_2d(block_space, space):
            space.ly = space.ly - (block_space.y + block_space.ly - space.y)
            space.y = block_space.y + block_space.ly
            if space.ly > 0:
                refresh_spaces_list.append(space)
            else:  # Discard space if it becomes invalid
                continue
        new_discard_space_list.append(space)  #Keep the non-overlapping ones.
    return refresh_spaces_list, new_discard_space_list



def update_info(
    truck: Truck,
    items: List[Item],
    space: Space,
    block: Dict,
    free_space_list: List[Space],
    discard_space_list: List[Space],
    used_space_list:List[Space]
) -> Tuple[List[Item], List[Space], List[Space], List[Space]]:
    """Updates truck information and space lists after placing a block."""

    # Update truck's items, used volume, and used weight
    truck.items.extend(block["items"])
    block_volume = (
        block["unit_size"][0]
        * block["stack_count"][0]
        * block["unit_size"][1]
        * block["stack_count"][1]
        * block["unit_size"][2]
        * block["stack_count"][2]
    )
    truck.used_volume += block_volume
    truck.used_weight += block["total_weight"]

    # Mark items as placed and set their coordinates
    for item in block["items"]:
        item.placed = True
        item.x = space.x
        item.y = space.y
        item.z = space.z


    # Update remaining items (remove placed items)
    block_item_ids = {item.item_id for item in block["items"]}
    updated_items = [item for item in items if item.item_id not in block_item_ids]


    # --- Update Space Lists ---
    block_space = Space(
        space.x,
        space.y,
        space.z,
        block["unit_size"][0] * block["stack_count"][0],
        block["unit_size"][1] * block["stack_count"][1],
        block["unit_size"][2] * block["stack_count"][2],
    )
    used_space_list.append(block_space)

    # Remove the used space from free_space_list
    # free_space_list.pop(space_index)
    free_space_list.remove(space) # Find the space we used

    # Get refreshable spaces and update discard_space_list
    refresh_spaces_list, discard_space_list = get_refresh_spaces(
        block_space, discard_space_list
    )

    # Update free and discarded space lists based on the placed block
    free_space_list = update_space_list_by_block_space(block_space, free_space_list)
    discard_space_list = update_space_list_by_block_space(
        block_space, discard_space_list
    )

    # --- Create new free spaces ---
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
        free_space_list.append(space1)
    if deltaY > 0:
        space2 = Space(
            block_space.x,
            block_space.y + block_space.ly,
            block_space.z,
            min(block_space.lx, space.lx),  # Use min to prevent extending beyond original space
            deltaY,
            space.lz,
        )
        # Merge with refreshable spaces if possible
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
                #discard_space_list.remove(pre_space)  # Remove merged space #Don't need, because we update it.
                break  # Only merge once
        free_space_list.append(space2)

    if deltaZ > 0:
        space3 = Space(
            block_space.x,
            block_space.y,
            block_space.z + block_space.lz,
            block_space.lx,
            block_space.ly,
            deltaZ,
        )
        # Handle overlap with already used spaces
        if block_space.lx > space.lx or block_space.ly > space.ly:
            overlapRects = list(
                filter(lambda boxSpace: check_overlap_3d(space3, boxSpace), used_space_list)
            )
            if len(overlapRects) > 0:
                space3.ly = min(overlapRects, key=lambda x: x.y).y - space3.y
        free_space_list.append(space3)


    return updated_items, free_space_list, discard_space_list, used_space_list

if __name__ == "__main__":
    # Create some sample items and a truck
    items = [
        Item("item1", 2, 2, 1, 5),
        Item("item2", 3, 1, 2, 8),
        Item("item3", 1, 1, 3, 2),
        Item("item4", 4, 2, 1, 10),
        Item("item5", 2, 3, 2, 7),
    ]
    truck = Truck("truck1", 10, 8, 5, 30)

    # Use the greedy framework with the example heuristics
    greedy_packing(
        truck,
        items,
        first_fit_select_space,
        simple_block_generation,
        largest_volume_select_block,
        update_info,
    )

    # Print results
    print(f"Truck: {truck}")
    print("Packed Items:")
    for item in truck.items:
        print(f"  {item}")
    print(f"Used Volume: {truck.used_volume}")
    print(f"Used Weight: {truck.used_weight}")
    print("-" * 20)