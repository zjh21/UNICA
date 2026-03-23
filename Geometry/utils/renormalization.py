import torch
import math 

def extract_keypoints_at_y_x_ranges(frame, mask, y_coordinate, x_ranges):
    """
    Extract keypoint values at a specific y coordinate and x within specified ranges
    from a frame, excluding background pixels.
    
    Args:
        frame: tensor (3, H, W) in range [-1, 1]
        mask: boolean mask (1, H, W) or (H, W)
        y_coordinate: y coordinate to extract keypoints from (can be floating point)
        x_ranges: list of tuples [(x_min, x_max), ...] where each tuple defines an
                  inclusive range [x_min, x_max] of x coordinates to include
    
    Returns:
        List of (x, y, rgb_values) tuples for foreground pixels at the specified y
        within the specified x ranges. For floating point y coordinates, uses linear 
        interpolation between adjacent rows, only including pixels where both rows 
        are foreground to avoid blending foreground with background.
    """
    device = frame.device
    C, H, W = frame.shape
    
    # Normalize mask shape to 2D
    if len(mask.shape) == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask
    
    # Helper function to check if x is in any of the ranges
    def x_in_ranges(x):
        for x_min, x_max in x_ranges:
            if x_min <= x <= x_max:
                return True
        return False
    
    y_floor = math.floor(y_coordinate)
    y_frac = y_coordinate - y_floor
    
    # For integer or near-integer coordinates (no interpolation needed)
    if abs(y_frac) < 1e-9 or abs(y_frac - 1.0) < 1e-9:
        y_int = round(y_coordinate)
        if y_int >= H or y_int < 0:
            return []
        
        mask_y = mask_2d[y_int, :]
        
        keypoints = []
        for x in range(W):
            if x_in_ranges(x) and mask_y[x].item():
                rgb_values = frame[:, y_int, x]
                keypoints.append((x, y_coordinate, rgb_values))
        
        return keypoints
    
    # Floating point case - need interpolation
    y_ceil = y_floor + 1
    
    # Check bounds for both rows
    if y_floor < 0 or y_ceil >= H:
        return []
    
    # Interpolation weight for ceiling row
    alpha = y_frac
    
    # Get masks for both rows
    mask_floor = mask_2d[y_floor, :]
    mask_ceil = mask_2d[y_ceil, :]
    
    keypoints = []
    
    for x in range(W):
        # Check if x is in any of the specified ranges
        if x_in_ranges(x):
            # Only interpolate if BOTH pixels are foreground to avoid blending with background
            if mask_floor[x].item() and mask_ceil[x].item():
                rgb_floor = frame[:, y_floor, x]
                rgb_ceil = frame[:, y_ceil, x]
                # Linear interpolation: (1 - alpha) * floor + alpha * ceil
                rgb_values = (1 - alpha) * rgb_floor + alpha * rgb_ceil
                keypoints.append((x, y_coordinate, rgb_values))
    
    return keypoints


def calculate_keypoint_based_shift(frame3, frame4, mask, y_coordinate, x_ranges):
    """
    Calculate the average RGB shift from frame3 to frame4 using pixels at specified
    y coordinate within specified x ranges.
    
    Args:
        frame3: tensor (3, H, W) in range [-1, 1]
        frame4: tensor (3, H, W) in range [-1, 1]
        mask: boolean mask (1, H, W) or (H, W)
        y_coordinate: y coordinate to extract keypoints from (can be floating point)
        x_ranges: list of tuples [(x_min, x_max), ...] where each tuple defines an
                  inclusive range [x_min, x_max] of x coordinates to include
    
    Returns:
        torch.Tensor: RGB shift (3,)
    """
    keypoints4 = extract_keypoints_at_y_x_ranges(frame4, mask, y_coordinate, x_ranges)
    
    # Calculate average (root) values for frame4
    values4 = [kp[2] for kp in keypoints4]
    root4 = torch.stack(values4).mean(dim=0)

    shift = root4
    
    return shift


def apply_keypoint_renormalization(frame2, frame3, frame4_gen, mask, y_coordinate, x_ranges, apply_renorm=True):
    """
    Apply re-normalization to frames based on RGB shift calculated from pixels at 
    specified y coordinate within specified x ranges.
    
    Args:
        frame2, frame3, frame4_gen: tensors (3, H, W) in range [-1, 1]
        mask: boolean mask (1, H, W) or (H, W)
        y_coordinate: y coordinate to extract keypoints from (can be floating point)
        x_ranges: list of tuples [(x_min, x_max), ...] where each tuple defines an
                  inclusive range [x_min, x_max] of x coordinates to include.
                  Use [(0, x_max)] for backward compatibility with the old x_max behavior.
        apply_renorm: whether to apply renormalization (set to False to keep original motion)
    
    Returns:
        Tuple of (frame2_renorm, frame3_renorm, frame4_renorm, shift)
    """
    # Calculate shift using specified y coordinate and x ranges
    shift = calculate_keypoint_based_shift(frame3, frame4_gen, mask, y_coordinate, x_ranges)
    
    if not apply_renorm:
        # Return frames as-is without renormalization
        return frame2, frame3, frame4_gen, shift
    
    # Reshape shift for broadcasting
    shift_reshaped = shift.view(-1, 1, 1)
    
    # Ensure mask is in correct shape
    if mask.shape[0] == 1 and frame2.shape[0] == 3:
        mask_expanded = mask.repeat(3, 1, 1).float()
    else:
        mask_expanded = mask.float()
    
    # Apply shift only to foreground regions
    frame2_renorm = frame2 - shift_reshaped * mask_expanded
    frame3_renorm = frame3 - shift_reshaped * mask_expanded
    frame4_renorm = frame4_gen - shift_reshaped * mask_expanded
    
    # Ensure background remains at -1
    frame2_renorm = frame2_renorm * mask_expanded + (-1.0) * (~mask_expanded.bool()).float()
    frame3_renorm = frame3_renorm * mask_expanded + (-1.0) * (~mask_expanded.bool()).float()
    frame4_renorm = frame4_renorm * mask_expanded + (-1.0) * (~mask_expanded.bool()).float()
    
    # Clamp to valid range
    frame2_renorm = torch.clamp(frame2_renorm, -1, 1)
    frame3_renorm = torch.clamp(frame3_renorm, -1, 1)
    frame4_renorm = torch.clamp(frame4_renorm, -1, 1)
    
    return frame2_renorm, frame3_renorm, frame4_renorm, shift