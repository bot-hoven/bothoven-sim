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

# record vid
# env = PianoSoundVideoWrapper(env, record_every=1, record_dir=".")

action_spec = env.action_spec()
action_names = action_spec.name.split("\t")
up_down_idxs = [i for i,name in enumerate(action_names) if name[-2:] == "J3"]

print(f"Action Spec Length: {len(action_spec.minimum)}")
# print(action_names)
# print(action_spec.minimum)
# print(action_spec.maximum)
# print(f"Observation spec:\n{env.observation_spec()}")

# timestep = env.reset()
# while not timestep.last():
#     action = np.random.uniform(low=action_spec.minimum, high=action_spec.maximum)
#     timestep = env.step(action)

actuators = np.array(env.task._hand.actuators)

for act in actuators:
    print(act)

# up-down joints
j3_idxs = [idx for idx,act in enumerate(env.task._hand.actuators[2:]) if act.name[-2:] == "J3"]
j3_idxs = np.array(j3_idxs, dtype=np.int32) + 2 # add 2 cuz we skipped thumb in iteration above

j3_min = action_spec.minimum[j3_idxs]
j3_max = action_spec.maximum[j3_idxs]

j3_act = np.full(len(j3_idxs), -1.0)
j3_act = scale_action_vector(j3_act, j3_min, j3_max)

# side-side joints
j4_idxs = [idx for idx,act in enumerate(env.task._hand.actuators[2:]) if act.name[-2:] == "J4"]
j4_idxs = np.array(j4_idxs, dtype=np.int32) + 2 # add 2 cuz we skipped thumb in iteration above

# ff_idx = j4_idxs[0]
# ff_min = action_spec.minimum[ff_idx]
# ff_max = action_spec.maximum[ff_idx]
# ff_act = scale_action_vector(-1.0, ff_min, ff_max)

j4_min = action_spec.minimum[j4_idxs]
j4_max = action_spec.maximum[j4_idxs]

j4_act = np.array([-1.0, -0.35, -0.35, -1.0])
j4_act = scale_action_vector(j4_act, j4_min, j4_max)

tip_sites = env.task._hand.fingertip_sites

def policy(timestep):
    act = np.full(actuators.shape[0] + 1, 0.0)

    j3_act = np.random.choice([-1, 1], size=len(j3_idxs))
    j3_act = scale_action_vector(j3_act, j3_min, j3_max)

    act[j3_idxs] = j3_act
    act[j4_idxs] = j4_act

    finger_rew = env.task._bothoven_finger_distance_reward(env.physics)
    print(finger_rew)
    # dist = env.task.distance_finger_to_finger(env.physics, tip_sites[2], tip_sites[3])
    # print(dist)
    return act

obs = env.observation_spec()
print(obs)

print(env.task.reward_fn.reward_fns.keys())
viewer.launch(env, policy=policy)