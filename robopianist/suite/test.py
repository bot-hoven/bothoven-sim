import numpy as np
from robopianist import suite

# Print out all available tasks.
# print(suite.ALL)

# # Print out robopianist-etude-12 task subset.
# print(suite.ETUDE_12)

# print()
# print(suite.DEBUG)

# # Load an environment from the debug subset.
env = suite.load("RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
action_spec = env.action_spec()
print(action_spec)

# Step through an episode and print out the reward, discount and observation.
timestep = env.reset()
while not timestep.last():
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape
    ).astype(action_spec.dtype)
    timestep = env.step(action)
    print(timestep.reward)
    print(timestep.discount)

    print(len(timestep.observation))

    print("Goal")
    print(timestep.observation['goal'])
    print(f"{len(timestep.observation['goal'])}\n")

    print("Fingering")
    print(timestep.observation['fingering'])
    print(f"{len(timestep.observation['fingering'])}\n")

    print("Piano")
    print(timestep.observation['piano/state'])
    print(f"{len(timestep.observation['piano/state'])}\n")

    print("Sustain")
    print(timestep.observation['piano/sustain_state'])
    print(f"{len(timestep.observation['piano/sustain_state'])}\n")

    print("RH Joints")
    print(timestep.observation['rh_shadow_hand/joints_pos'])
    print(f"{len(timestep.observation['rh_shadow_hand/joints_pos'])}\n")

    print("LH Joints")
    print(timestep.observation['lh_shadow_hand/joints_pos'])
    print(f"{len(timestep.observation['lh_shadow_hand/joints_pos'])}\n")
    # print(timestep.reward, timestep.discount, timestep.observation)
    exit(0)