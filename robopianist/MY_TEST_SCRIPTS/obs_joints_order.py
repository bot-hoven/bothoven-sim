from dm_control import composer, mjcf, viewer

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import shadow_hand
from robopianist.models.hands import shadow_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import piano_with_one_shadow_hand
from robopianist import suite
from robopianist.wrappers import PianoSoundVideoWrapper

import numpy as np

def scale_action_vector(action_vector, minimum, maximum):    
    return minimum + (action_vector + 1) * 0.5 * (maximum - minimum)

joint_actuator_map = {
    "rh_FFJ4": (0, 2),
    "rh_FFJ3": (1, 3),
    "rh_MFJ4": (4, 4),
    "rh_MFJ3": (5, 5),
    "rh_RFJ4": (8, 6),
    "rh_RFJ3": (9, 7),
    "rh_LFJ4": (12, 8),
    "rh_LFJ3": (13, 9),
    "rh_THJ4": (16, 0),
    "rh_THJ2": (18, 1)
}


env = suite.load(
    environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed=42,
    right_hand_only=True,
    task_kwargs=dict(
        n_steps_lookahead=10,
        trim_silence=True,
        gravity_compensation=True,
        bothoven_reduced_action_space=True,
        change_color_on_activation=True,
        initial_buffer_time=5.0,
    ),
)

action_spec = env.action_spec()
action_names = action_spec.name.split("\t")
up_down_idxs = [i for i,name in enumerate(action_names) if name[-2:] == "J3"]

print(f"Action Spec Length: {len(action_spec.minimum)}")
print(action_names)
# print(action_spec.minimum)
# print(action_spec.maximum)
# print(f"Observation spec:\n{env.observation_spec()}")

actuators = np.array(env.task._hand.actuators)
joints = np.array(env.task._hand.joints)

binded_joints = env.physics.bind(env.task._hand.joints)._named_index
print("Binded joints:")
print(binded_joints)
print("\n")

for joint in joints:
    print(joint)

print()

for act in actuators:
    print(act)


tip_sites = env.task._hand.fingertip_sites

def policy(timestep):
    act = np.full(actuators.shape[0] + 1, 0.0)
    return act

obs = env.observation_spec()
print(obs)

print(env.task.reward_fn.reward_fns.keys())
viewer.launch(env, policy=policy)