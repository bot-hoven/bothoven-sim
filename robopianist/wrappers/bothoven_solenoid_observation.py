
"""A wrapper for making discrete actions appear as 1 or -1 in state observation."""


import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper

_BOTHOVEN_DISCRETE_IDXS = np.array([2, 4, 6, 8, 10])

def unscale_action_vector(scaled_action_vector, minimum, maximum):
    if scaled_action_vector.shape != minimum.shape or scaled_action_vector.shape != maximum.shape:
        raise ValueError("Scaled action vector, minimum, and maximum must have the same shape.")
    return 2 * (scaled_action_vector - minimum) / (maximum - minimum) - 1

class BothovenSolenoidObservationWrapper(EnvironmentWrapper):
    """Adds pixel observations to the observation spec."""

    def __init__(
        self,
        environment: dm_env.Environment,
    ) -> None:
        super().__init__(environment)

        print("Using Solenoid Thresholding Observations!")

        # Enforce that the wrapped environment has a dict observation spec, i.e., that
        # is hasn't been wrapped with `ConcatObservationWrapper`.
        if not isinstance(self._environment.observation_spec(), dict):
            raise ValueError(
                "ObservationActionWrapper requires an environment with a "
                "dictionary observation. Consider using this wrapper before "
                "ConcatObservationWrapper."
            )
        
        self._observation_spec = self._environment.observation_spec()
        self._action_spec = self._environment.action_spec()

    def observation_spec(self):
        return self._observation_spec

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return self._update_solenoid_dims(timestep, None)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return self._update_solenoid_dims(timestep, action)

    def _update_solenoid_dims(self, timestep: dm_env.TimeStep, action: np.ndarray) -> dm_env.TimeStep:
        obs = timestep.observation

        # unscaled_action = None if action is None else unscale_action_vector(action, self._action_spec.minimum, self._action_spec.maximum)

        for key in obs:
            if key.endswith("joints_pos"):
                obs[key][_BOTHOVEN_DISCRETE_IDXS] = 0 if action is None else action[_BOTHOVEN_DISCRETE_IDXS] # 0 is idle joint position

        return timestep._replace(observation=obs)
        
