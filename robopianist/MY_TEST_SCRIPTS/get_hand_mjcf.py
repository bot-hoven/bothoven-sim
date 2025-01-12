from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import base as base_task

import numpy as np

def get_env():
    task = base_task.PianoTask(arena=stage.Stage(), bothoven_reduced_action_space=True)
    env = composer.Environment(
        task=task, time_limit=5.0, strip_singleton_obs_buffer_dim=True
    )
    return env

def scale_action_vector(action_vector, minimum, maximum):    
    if action_vector.shape != minimum.shape or action_vector.shape != maximum.shape:
        raise ValueError("Action vector, minimum, and maximum must have the same shape.")
    return minimum + (action_vector + 1) * 0.5 * (maximum - minimum)    

env = get_env()
model = env.physics.model
data = env.physics.named.data

# NOTE: forearm joint positions will be set to 0, because their ranges are
#       adjusted so that the initial position is 0 (see PianoTask class)
r_hand = env.task.right_hand.mjcf_model
r_hand_joints = r_hand.find_all("joint", exclude_attachments=True)
r_hand_actuators = r_hand.find_all("actuator", exclude_attachments=True)

rh_forearm_tx_idx = [i for i,a in enumerate(r_hand_actuators) if a.name == "forearm_tx"][0]
lh_forearm_tx_idx = len(r_hand_actuators) + rh_forearm_tx_idx

# joint_positions = env.physics.bind(r_hand_joints).qpos
# actuator_positions = env.physics.bind(r_hand_actuators).ctrl # i think ctrl gives the positions for position actuators?

# get indices of up/down actions
action_spec = env.action_spec()
action_names = action_spec.name.split("\t")
up_down_idxs = [i for i,name in enumerate(action_names) if name[-2:] == "J3"]
mask = np.zeros_like(action_names, dtype=bool)
mask[up_down_idxs] = True

count = 0

# initial action
# prev_action = np.random.uniform(low=action_spec.minimum,
#                              high=action_spec.maximum,
#                              size=action_spec.shape)
# prev_action[rh_forearm_tx_idx] = 0 # rh forearms
# prev_action[lh_forearm_tx_idx] = 0 # lh forearms

prev_action = np.full(action_spec.shape, 0)
prev_action[up_down_idxs] = -1.0
prev_action = scale_action_vector(prev_action, action_spec.minimum, action_spec.maximum)
prev_action[~mask] = 0.0

def random_policy(time_step):
    global count, prev_action
    count += 1
    # prev_action = np.full(action_spec.shape, 0)
    # prev_action[rh_forearm_tx_idx] = 0 # rh forearms
    # prev_action[lh_forearm_tx_idx] = 0 # lh forearms
    if count % 1 == 0:
        if prev_action[up_down_idxs][0] > 1.0: # 1.57 rad down, -0.26 rad up
            prev_action[up_down_idxs] = -1.0
        else:
            prev_action[up_down_idxs] = 1.0
        prev_action = scale_action_vector(prev_action, action_spec.minimum, action_spec.maximum)
        prev_action[~mask] = 0.0
        # prev_action = np.random.uniform(low=-1.0,
        #                      high=1.0,
        #                      size=action_spec.shape)
        # prev_action = scale_action_vector(prev_action, action_spec.minimum, action_spec.maximum)
        # prev_action[rh_forearm_tx_idx] = 0 # rh forearms
        # prev_action[lh_forearm_tx_idx] = 0 # lh forearms
        # print(prev_action)
    return prev_action

print(f"UP DOWN IDXS:\n{up_down_idxs}")
print(f"Action Spec Length: {len(action_spec.minimum)}")
print(action_names)
print()
print(action_spec.minimum)
print()
print(action_spec.maximum)

viewer.launch(env, policy=random_policy)

print("done")