import math

class GeorgeAndRobinsonHeuristic:
    """
    A more faithful, though simplified, implementation of the layer-based heuristic
    described by George and Robinson (1980).
    """

    def __init__(self, container_dims):
        self.container_dims = container_dims # (length, width, height)
        self.placed_items = []
        self.open_box_types = set() # Track partially used box types

    def pack(self, items_to_pack):
        # Create a mutable copy of items with quantity
        self.items = [dict(item, quantity=item['quantity'], original_index=i) for i, item in enumerate(items_to_pack)]
        
        container_length, container_width, container_height = self.container_dims
        current_depth = 0

        while True:
            # Check if there are any items left to pack
            if not any(it['quantity'] > 0 for it in self.items):
                print("Successfully packed all items.")
                return self.placed_items, True

            # === 1. Start a new layer ===
            # Select the first box type to define the new layer's depth
            best_box_for_layer, orientation, layer_depth = self._select_box_for_new_layer(container_length - current_depth)
            
            if not best_box_for_layer:
                print("Unsuccessful packing. No box can start a new layer.")
                return self.placed_items, False
            
            print(f"\n--- Starting new layer at depth {current_depth} with depth {layer_depth} ---")

            # The "workface" is a list of 2D empty rectangles in the current layer's cross-section
            workface_spaces = [{'pos': (0, 0), 'dims': (container_width, container_height)}]

            # === 2. Fill the current layer ===
            layer_is_full = False
            while not layer_is_full:
                # Select a space on the 2D workface to fill (e.g., smallest height, then smallest width)
                workface_spaces.sort(key=lambda s: (s['pos'][1], s['pos'][0]))
                
                if not workface_spaces:
                    layer_is_full = True
                    continue

                current_space = workface_spaces.pop(0)
                
                # Find the best box to fit in this 2D space
                best_box, best_orient, best_stack = self._find_best_fit_for_space(current_space, layer_depth)
                
                if not best_box:
                    # This space can't be filled by any remaining box.
                    # A full implementation would add it to a "rejected" list.
                    # For simplicity, we'll just leave it empty.
                    if not workface_spaces: # No more spaces to try
                        layer_is_full = True
                    continue

                # Place the stack of boxes
                stack_w_count, stack_h_count = best_stack
                box_dims = self._get_oriented_dims(best_box, best_orient)
                
                # Add to placed items list
                for i in range(stack_w_count):
                    for j in range(stack_h_count):
                        pos_x = current_depth
                        pos_y = current_space['pos'][0] + i * box_dims[1]
                        pos_z = current_space['pos'][1] + j * box_dims[2]
                        self.placed_items.append({
                            'pos': (pos_x, pos_y, pos_z),
                            'dims': box_dims,
                            'type': best_box['original_index']
                        })
                
                # Update item quantity and open status
                best_box['quantity'] -= (stack_w_count * stack_h_count)
                self.open_box_types.add(best_box['original_index'])

                # Update the 2D workface by splitting the space
                new_spaces = self._split_workface_space(current_space, (stack_w_count * box_dims[1], stack_h_count * box_dims[2]))
                workface_spaces.extend(new_spaces)

            # === 3. Move to the next layer ===
            current_depth += layer_depth


    def _get_oriented_dims(self, item, orientation):
        L, W, H = item['length'], item['width'], item['height']
        # Orientation 0: (L,W,H) -> Depth=L, Width=W, Height=H
        # For simplicity, we only consider orientations where one dimension is depth
        if orientation == 0: return (L, W, H)
        if orientation == 1: return (W, L, H)
        if orientation == 2: return (H, W, L)
        # A full implementation would check all 6.

    def _get_item_rank(self, item):
        dims = sorted([item['length'], item['width'], item['height']])
        is_open = item['original_index'] in self.open_box_types
        # G&R has two rules for open boxes; we use a simple priority flag
        return (not is_open, -dims[0], -item['quantity'], -dims[2])
    
    def _select_box_for_new_layer(self, max_depth):
        """Implements the logic from Fig. 1 to find the best box to start a layer."""
        best_choice = (None, -1, 0) # box, orientation, depth
        best_rank = (True, float('inf'), float('inf'), float('inf'))

        for item in self.items:
            if item['quantity'] <= 0:
                continue
            
            rank = self._get_item_rank(item)
            
            # Simple check for the 3 main orientations
            for i, dim in enumerate([item['length'], item['width'], item['height']]):
                if dim <= max_depth:
                    # G&R Rule: "Choose the depth dimension as the longest dimension <= k"
                    # We just find the best ranked box that fits.
                    if rank < best_rank:
                         best_rank = rank
                         best_choice = (item, i, dim)
        return best_choice

    def _find_best_fit_for_space(self, space, layer_depth):
        """Finds the best box and stack to fit in a 2D space on the workface."""
        best_fit = (None, -1, (0,0)) # box, orientation, (stack_w, stack_h)
        max_fill_area = -1

        space_w, space_h = space['dims']

        for item in self.items:
            if item['quantity'] <= 0:
                continue

            for i, dim in enumerate([item['length'], item['width'], item['height']]):
                if abs(dim - layer_depth) < 1e-5: # Dimension must match layer depth
                    box_dims = self._get_oriented_dims(item, i)
                    box_w, box_h = box_dims[1], box_dims[2]

                    if box_w <= space_w and box_h <= space_h:
                        num_w = math.floor(space_w / box_w)
                        num_h = math.floor(space_h / box_h)
                        
                        num_to_place = min(item['quantity'], num_w * num_h)
                        if num_to_place == 0: continue
                        
                        # Find actual w,h count for num_to_place
                        actual_w = min(num_w, num_to_place)
                        actual_h = min(num_h, math.ceil(num_to_place / actual_w))

                        fill_area = actual_w * box_w * actual_h * box_h
                        
                        if fill_area > max_fill_area:
                            max_fill_area = fill_area
                            best_fit = (item, i, (actual_w, actual_h))
        return best_fit

    def _split_workface_space(self, space, placed_dims):
        """Splits a 2D rectangle on the workface after a block is placed."""
        new_spaces = []
        sp_pos, sp_dims = space['pos'], space['dims']
        pl_dims = placed_dims
        
        # This is the "widthwise" space in G&R
        if sp_dims[0] > pl_dims[0]:
            new_spaces.append({
                'pos': (sp_pos[0] + pl_dims[0], sp_pos[1]),
                'dims': (sp_dims[0] - pl_dims[0], sp_dims[1])
            })
        
        # This is the "heightwise" space in G&R
        if sp_dims[1] > pl_dims[1]:
            new_spaces.append({
                'pos': (sp_pos[0], sp_pos[1] + pl_dims[1]),
                'dims': (pl_dims[0], sp_dims[1] - pl_dims[1])
            })
            
        return new_spaces

# Example Usage
if __name__ == '__main__':
    from SSSCSP import PackingCONST, GetData


    getData = GetData(47)
    instance = getData.generate_instances()
    for inst in instance:
        item_types = inst[0]
        container_dimensions = inst[1]

        packer = GeorgeAndRobinsonHeuristic(container_dimensions)
        placed, success = packer.pack(item_types)

        if success:
            print(f"\nPacking successful. Total items placed: {len(placed)}")
            # for p in placed:
            #     print(f"  Placed box of type {p['type']} at {p['pos']} with dims {p['dims']}")
        else:
            print(f"\nPacking failed. Items placed: {len(placed)}")