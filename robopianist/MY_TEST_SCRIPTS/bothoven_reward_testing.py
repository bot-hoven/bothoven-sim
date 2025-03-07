from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import base as base_task, PianoOnlyTask
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers

import numpy as np

def scale_action_vector(action_vector, minimum, maximum):    
    if action_vector.shape != minimum.shape or action_vector.shape != maximum.shape:
        raise ValueError("Action vector, minimum, and maximum must have the same shape.")
    return minimum + (action_vector + 1) * 0.5 * (maximum - minimum)

env = suite.load(
    environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed=42,
    right_hand_only=True,
    task_kwargs=dict(
        n_steps_lookahead=0,
        trim_silence=True,
        gravity_compensation=True,
        bothoven_reduced_action_space=True,
        change_color_on_activation=True,
        use_bothoven_hand=True,
    ),
)

env = robopianist_wrappers.BothovenSolenoidObservationWrapper(env)

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

r_hand = env.task.right_hand.mjcf_model
r_hand_joints = r_hand.find_all("joint", exclude_attachments=True)
r_hand_actuators = r_hand.find_all("actuator", exclude_attachments=True)

# remove actuators
for act in r_hand_actuators:
    act.remove()
env.task.right_hand._actuators = tuple()

# remove stepper joint
# r_hand_joints[0].remove()
# env.task.right_hand._joints = tuple(r_hand_joints[1:])

joints = np.array(env.task._hand.joints)
actuators = np.array(env.task._hand.actuators)

print()
print(joints)
print(actuators)

# rh_forearm_tx_idx = [i for i,a in enumerate(r_hand_actuators) if a.name == "stepper"][0]
# lh_forearm_tx_idx = len(r_hand_actuators) + rh_forearm_tx_idx

# joint_positions = env.physics.bind(r_hand_joints).qpos

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
        spread_rew = env.task._bothoven_spread_from_key(env.physics)
        fingering_rew = env.task._bothoven_compute_fingering_reward(env.physics)
        print(f"Spread Rew: {spread_rew}")
        print(f"Fingering Rew: {fingering_rew}")
        print()
    count += 1
    arr = [0]
    return np.array(arr)

viewer.launch(env, policy=random_policy)

print("done")