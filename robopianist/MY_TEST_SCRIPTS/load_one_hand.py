from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import piano_with_one_shadow_hand
from robopianist import suite

import numpy as np

env = suite.load(
    environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed=42,
    right_hand_only=True,
    task_kwargs=dict(
        trim_silence=True,
        gravity_compensation=True,
        bothoven_reduced_action_space=True,
        change_color_on_activation=True,
    ),
)

action_spec = env.action_spec()
action_names = action_spec.name.split("\t")
up_down_idxs = [i for i,name in enumerate(action_names) if name[-2:] == "J3"]

print(f"UP DOWN IDXS:\n{up_down_idxs}")
print(f"Action Spec Length: {len(action_spec.minimum)}")
print(action_names)
print()
print(action_spec.minimum)
print()
print(action_spec.maximum)

print()

print(f"Observation spec:\n{env.observation_spec()}")

viewer.launch(env)