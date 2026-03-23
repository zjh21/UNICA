import os

MOTION_MAPPING = {
    "0": "Idle",
    "w": "Forward",
    "a": "Left",
    "s": "Backward",
    "d": "Right",
}


def read_motion_label(label_path):
    """Read motion label from a text file and return the motion type string."""
    with open(label_path, "r") as f:
        label = f.read().strip()
    return MOTION_MAPPING.get(label)


def get_motion_label_for_group(label_root, case_name, group_numbers):
    """
    Get the motion label for a group by reading the 3rd frame's label file.

    The label file is expected at: {label_root}/{case_name}/labels/{frame:03d}.txt

    Args:
        label_root: Root directory containing label files
        case_name: Name of the case/sequence
        group_numbers: List of frame numbers in the group

    Returns:
        Motion type string (e.g. 'Forward'), or None if label not found
    """
    third_frame = group_numbers[2]
    label_path = os.path.join(
        label_root, case_name, "labels", f"{third_frame:03d}.txt"
    )

    if not os.path.exists(label_path):
        return None

    return read_motion_label(label_path)
