from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import base as base_task, PianoOnlyTask
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers
from robopianist.models.arenas import stage

import numpy as np

env = suite.load(
    environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed=42,
    right_hand_only=False,
    just_piano=True,
    task_kwargs=dict(
        arena=stage.Stage()
    ),
)

action_spec = env.action_spec()
action_names = action_spec.name.split("\t")
up_down_idxs = np.array([i for i,name in enumerate(action_names) if (name[-10:])[:-2] == "solenoid"])
print(f"Action Spec Length: {len(action_spec.minimum)}")
print(action_names)
print()
print(action_spec.minimum)
print()
print(action_spec.maximum)

obs = env.observation_spec()
print("\nObservation Spec:")
for pair in obs.items():
    print(pair)
print(obs)

first_go = True
count = 0
prev_action = None

def random_policy(time_step):
    global count, prev_action, first_go

    if first_go:
        first_go = False
        return np.array([0])

    if count % 1 == 0:
        # prev_action = np.random.uniform(low=-1.0,
        #                         high=1.0,
        #                         size=action_spec.shape)
        # prev_action[up_down_idxs[prev_action[up_down_idxs] >= 0]] = 1   # Set positive values to 1
        # prev_action[up_down_idxs[prev_action[up_down_idxs] < 0]] = -1 
        # prev_action = scale_action_vector(prev_action, action_spec.minimum, action_spec.maximum)
        # prev_action[rh_forearm_tx_idx] = 0 # rh forearms
        # prev_action[lh_forearm_tx_idx] = 0 # lh forearms

        prev_action = np.zeros(action_spec.shape)
        
        print(f"Timestep {count}:")
        finger_rew = env.task._bothoven_spread_from_key(env.physics)
        print(finger_rew)
        print()
    count += 1
    arr = [0]
    return np.array(arr)

viewer.launch(env, policy=random_policy)

print("done")