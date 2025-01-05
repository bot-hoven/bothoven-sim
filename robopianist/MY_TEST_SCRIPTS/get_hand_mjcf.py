from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import base as base_task

import numpy as np

def get_env():
    task = base_task.PianoTask(arena=stage.Stage(), bothoven_reduce_action_space=True)
    env = composer.Environment(
        task=task, time_limit=5.0, strip_singleton_obs_buffer_dim=True
    )
    return env

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

joint_positions = env.physics.bind(r_hand_joints).qpos
actuator_positions = env.physics.bind(r_hand_actuators).ctrl # i think ctrl gives the positions for position actuators?

action_spec = env.action_spec()

count = 0
prev_action = np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)
# prev_action[rh_forearm_tx_idx] = 0 # rh forearms
# prev_action[lh_forearm_tx_idx] = 0 # lh forearms

def random_policy(time_step):
    global count, prev_action
    count += 1
    if count % 8 == 0:
        prev_action = np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)
        # prev_action[rh_forearm_tx_idx] = 0 # rh forearms
        # prev_action[lh_forearm_tx_idx] = 0 # lh forearms
        # print(prev_action)
    return prev_action

print(f"Action Spec Length: {len(action_spec.minimum)}")
# print(action_spec.name)
# print()
# print(action_spec.minimum)
# print()
# print(action_spec.maximum)

viewer.launch(env, policy=random_policy)

print("done")