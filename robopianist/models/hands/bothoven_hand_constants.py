from pathlib import Path
from typing import Dict, Tuple

_HERE = Path(__file__).resolve().parent

NQ = 11  # Number of joints
NU = 11  # Number of actuators

FINGER_ARM_BODIES: Tuple[str, ...] = (
    # Important: the order of these names should not be changed.
    "finger_arm_large_1", # thumb (leftmost on right hand)
    "finger_arm_large_2",
    "finger_arm_large_3",
    "finger_arm_large_4",
    "finger_arm_large_5", # pinky (rightmost on right hand)
)

BOTHOVEN_HAND_XML = _HERE / "bothoven" / "bothoven.xml"
