# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-handed version of `piano_with_shadow_hands.py`."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import spec_utils, types

import robopianist.models.hands.shadow_hand_constants as hand_consts
import robopianist.models.piano.piano_constants as piano_consts
from robopianist.models.arenas import stage
from robopianist.models.hands import HandSide
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.005
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05
_FINGERS_TOO_CLOSE = 0.022 # meters (this is default resting position, need to encourage more spread)
_TARGET_SPREAD = 1.75 * piano_consts.WHITE_KEY_WIDTH

# Reward weighting coefficients.
_ENERGY_PENALTY_COEF = 5e-3
_FINGER_DIST_COEF = 0.5
_SPREAD_COEF = 0.2

# Discrete action indicies.
_DISCRETE_IDXS = np.array([3, 5, 7, 9])
_BOTHOVEN_DISCRETE_IDXS = np.array([2, 4, 6, 8, 10]) # for our hands

class PianoWithOneShadowHand(base.PianoTask):
    def __init__(
        self,
        midi: midi_file.MidiFile,
        hand_side: HandSide,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_fingering_reward: bool = False,
        disable_forearm_reward: bool = False,
        disable_hand_collisions: bool = False,
        disable_colorization: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            disable_fingering_reward: If True, disables the shaping reward for
                fingering. This will also disable the colorization of the fingertips
                and corresponding keys.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
        """
        super().__init__(arena=stage.Stage(), **kwargs)

        if trim_silence:
            midi = midi.trim_silence()
        self._midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._disable_fingering_reward = disable_fingering_reward
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._augmentations = augmentations
        self._use_bothoven = kwargs['use_bothoven_hand']

        # For computing discrete action change frequency penalty
        self._curr_action = None
        self._prev_action = None

        self._hand_side = hand_side
        if self._hand_side == HandSide.LEFT:
            self._hand = self._left_hand
            self._right_hand.detach()
        else:
            self._hand = self._right_hand
            self._left_hand.detach()

        if not disable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory(self._midi)  # Important: call before adding observables.
        self._add_observables()
        self._set_rewards()

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
            energy_reward=self._compute_energy_reward,
            # bothoven_finger_distance_reward=self._bothoven_finger_distance_reward,
            # bothoven_action_change_penalty=self._bothoven_action_change_penalty,
        )
        if not self._disable_fingering_reward:
            self._reward_fn.add("fingering_reward", self._bothoven_compute_fingering_reward)
        else:
            # use OT based fingering
            print('Fingering is unavailable. OT fingering reward is used.')
            self._reward_fn.add("ot_fingering_reward", self._compute_ot_fingering_reward)


    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0

    def _maybe_change_midi(self, random_state) -> None:
        if self._augmentations is not None:
            midi = self._midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._reset_trajectory(midi)

    def _reset_trajectory(self, midi: midi_file.MidiFile) -> None:
        note_traj = midi_file.NoteTrajectory.from_midi(midi, self.control_timestep)
        note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._notes = note_traj.notes
        self._sustains = note_traj.sustains

    # Composer methods.

    def initialize_episode(self, physics, random_state) -> None:
        del physics  # Unused.
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()

    def after_step(self, physics, random_state) -> None:
        del random_state  # Unused.
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        self._goal_current = self._goal_state[0]

        if not self._disable_fingering_reward:
            self._keys_current = self._keys
            if not self._disable_colorization:
                self._colorize_keys(physics)

        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

    def get_reward(self, physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics):
        hand_spec = self._hand.action_spec(physics)
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hand_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([hand_spec, sustain_spec])

    def before_step(self, physics, action, random_state) -> None:
        self._prev_action = self._curr_action
        self._curr_action = action

        sustain = action[-1]
        self.piano.apply_sustain(physics, sustain, random_state)
        self._hand.apply_action(physics, action[:-1], random_state)

    # Helper methods.

    def _compute_sustain_reward(self, physics) -> float:
        """Reward for pressing the sustain pedal at the right time."""
        del physics  # Unused.
        return tolerance(
            self._goal_current[-1] - self.piano.sustain_activation[0],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )

    def _compute_energy_reward(self, physics) -> float:
        """Reward for minimizing energy."""
        # power = self._hand.observables.actuators_power(physics).copy()
        # return -_ENERGY_PENALTY_COEF * np.sum(power)
        return 0

    def _compute_key_press_reward(self, physics) -> float:
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        # It's possible we have no keys to press at this timestep so we need to check
        # that `on` is not empty.
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
            # rew += rews.mean()
        off = np.flatnonzero(1 - self._goal_current[:-1])
        # If there's any false positive, the remaining 0.5 reward is lost.
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))

        # rew -= float(np.sum(self.piano.activation[off]))
        return rew
    
    def _bothoven_action_change_penalty(self, physics) -> float:
        """
        For every discrete action change, enact a penalty. If the action
        is truly good, the reward from a correct key press will out-weigh
        the state change penalty (rewards are computed at 20Hz, so if
        the finger stays down on the correct key, much more reward will be
        obtained than lost).
        """
        del physics # not used
        if self._prev_action is None:
            return 0.0
        # -0.5 penalty for each discrete actions change
        alpha = 0.2
        if self._use_bothoven:
            return -alpha * np.sum(self._prev_action[_BOTHOVEN_DISCRETE_IDXS] != self._curr_action[_BOTHOVEN_DISCRETE_IDXS])
        return -alpha * np.sum(self._prev_action[_DISCRETE_IDXS] != self._curr_action[_DISCRETE_IDXS]) # for reduced shadow hands

    
    def _bothoven_finger_distance_reward(self, physics) -> float:
        """
        Penalize fingers for coming too close to one another.
        """
        def _distance_finger_to_finger(fingertips: Sequence[types.MjcfElement]):
            distances = []
            for i,tip in enumerate(fingertips):
                if i != len(fingertips) - 1:
                    # xy-plane coords of successive tips
                    tip1_pos = physics.bind(tip).xpos.copy()[:2]
                    tip2_pos = physics.bind(fingertips[i+1]).xpos.copy()[:2]
                    distances.append(float(np.linalg.norm(tip2_pos - tip1_pos)))
            return distances
        
        distances = _distance_finger_to_finger(self._hand.fingertip_sites)
        
        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGERS_TOO_CLOSE),
            margin=2*_FINGERS_TOO_CLOSE, # Note that margin is in addition to bounds. So, tolerance == 0.1 when margin + this 
            sigmoid="gaussian",
        )
        # return float(np.mean(rews))
        return _FINGER_DIST_COEF * float(np.mean(1 - rews))

    
    def _bothoven_spread_from_key(self, physics) -> float:
        """
        Reward fingers for moving away from keys that their neighbouring finger is assigned to press.
        """
        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf in hand_keys:
                # Determine adjacent fingers to check
                offsets = []
                if mjcf == 0: # rh thumb
                    offsets = [1]
                elif mjcf == 4: # rh pinky
                    offsets = [-1]
                else:
                    offsets = [1, -1]

                # Get key position once per key
                key_geom = self.piano.keys[key].geom[0]
                key_pos = physics.bind(key_geom).xpos.copy()[1]
                
                key_pos_full = physics.bind(key_geom).xpos.copy()

                # Process adjacent fingers
                for offset in offsets:
                    fingertip = hand.fingertip_sites[mjcf + offset]
                    fingertip_pos = physics.bind(fingertip).xpos.copy()[1]
                    fingertip_pos_full = physics.bind(fingertip).xpos.copy()
                    print(f"Finger: {offset}\tFingertip Pos: {fingertip_pos_full}\tKey Pos: {key_pos_full}")
                    distances.append(float(np.linalg.norm(key_pos - fingertip_pos)))
            return distances
        
        # _keys_current is array of tuples of (key_id, finger)
        # for finger supposed to press certain key.
        distances = _distance_finger_to_key(self._keys_current, self._hand)

        print(f"Distances: {distances}")

        if not distances:
            return 0.0
        
        rews = tolerance(
            np.hstack(distances),
            bounds=(_TARGET_SPREAD - 0.005, _TARGET_SPREAD + 0.005), # 1 if greater than 1.5 * w_key
            margin=0.65 * piano_consts.WHITE_KEY_WIDTH, # 0.1 (i.e. default val_at_margin) if 0.65 key lengths away from target
            sigmoid="gaussian",
        )
        return _SPREAD_COEF * float(np.mean(rews))
    
    def distance_finger_to_finger(self, physics, tip1: types.MjcfElement, tip2: types.MjcfElement):
        # xy-plane coords of tips
        tip1_pos = physics.bind(tip1).xpos.copy()[:2]
        tip2_pos = physics.bind(tip2).xpos.copy()[:2]
        return float(np.linalg.norm(tip2_pos - tip1_pos))
    
    def _bothoven_compute_fingering_reward(self, physics) -> float:
        """
        Reward for minimizing distance between fingers and keys,
        where distance is only in the XY-plane
        """
        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                
                # only compute distances in xy plane (finger should hover key)
                diff = key_geom_pos[1] - fingertip_pos[1]
                distances.append(float(np.linalg.norm(diff)))
            return distances
        
        # _keys_current is array of tuples of (key_id, finger)
        # for finger supposed to press certain key.
        distances = _distance_finger_to_key(self._keys_current, self._hand)

        # Case where there are no keys to press at this timestep.
        # TODO(kevin): Unclear if we should return 0 or 1 here. 0 seems to do better.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))

    def _compute_fingering_reward(self, physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""

        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()

                # coordinate frame for each key is at the center of its box
                # x is +'ve outwards from key, y is +'ve right of key, and
                # z is +'ve above key. Thus, adding 0.5 of box height to z
                # places distance-to-key target on key surface, and adding
                # 0.35 to x makes it lower on the key. (Black and white keys
                # have different geom dimensions here.)
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances
        
        # _keys_current is array of tuples of (key_id, finger)
        # for finger supposed to press certain key.
        distances = _distance_finger_to_key(self._keys_current, self._hand)

        # Case where there are no keys to press at this timestep.
        # TODO(kevin): Unclear if we should return 0 or 1 here. 0 seems to do better.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))

    def _compute_ot_fingering_reward(self, physics) -> float:
        """ OT reward calculation from RP1M https://arxiv.org/abs/2408.11048 """
        # calcuate fingertip positions
        fingertip_pos = [physics.bind(finger).xpos.copy() for finger in self.right_hand.fingertip_sites]
        
        # calcuate the positions of piano keys to press.
        keys_to_press = np.flatnonzero(self._goal_current[:-1]) # keys to press
        # if no key is pressed
        if keys_to_press.shape[0] == 0:
            return 1.

        # calculate key pos
        key_pos = []
        for key in keys_to_press:
            key_geom = self.piano.keys[key].geom[0]
            key_geom_pos = physics.bind(key_geom).xpos.copy()
            key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
            key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
            key_pos.append(key_geom_pos.copy())

        # calcualte the distance between keys and fingers
        dist = np.full((len(fingertip_pos), len(key_pos)), 100.)
        for i, finger in enumerate(fingertip_pos):
            for j, key in enumerate(key_pos):
                dist[i, j] = np.linalg.norm(key - finger)
        
        # calculate the shortest distance
        row_ind, col_ind = linear_sum_assignment(dist)
        dist = dist[row_ind, col_ind]
        rews = tolerance(
            dist,
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))        

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _update_fingering_state(self) -> None:
        if self._t_idx == len(self._notes):
            return

        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]

        # Split fingering into right and left hand.
        self._keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                if self._hand_side == HandSide.RIGHT:
                    self._keys.append((piano_key, finger))
            else:
                if self._hand_side == HandSide.LEFT:
                    self._keys.append((piano_key, finger - 5))

        # For each hand, set the finger to 1 if it is used and 0 otherwise.
        self._fingering_state = np.zeros((5,), dtype=np.float64)
        for _, mjcf_fingering in self._keys:
            self._fingering_state[mjcf_fingering] = 1.0

    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            "joints_pos",
            # "position",
            # "fingertip_positions",
            # "actuators_force",
            # "actuators_velocity",
        ]
        for obs in enabled_observables:
            getattr(self._hand.observables, obs).enabled = True

        # This returns the current state of the piano keys.
        self.piano.observables.state.enabled = False
        self.piano.observables.sustain_state.enabled = False

        # This returns the key activation state (on or off).
        # Disabling for now since pretty much redundant with the state observables.
        self.piano.observables.activation.enabled = False
        self.piano.observables.sustain_activation.enabled = False

        # This returns the goal state for the current timestep and n steps ahead.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        # This adds fingering information for the current timestep.
        def _get_fingering_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = not self._disable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

        # How many time steps are left in the episode, between 0 and 1.
        def _get_steps_left(physics) -> float:
            del physics  # Unused.
            return (len(self._notes) - self._t_idx) / len(self._notes)

        steps_left_observable = observable.Generic(_get_steps_left)
        # Disabled for now, didn't seem to help.
        steps_left_observable.enabled = False
        self._task_observables["steps_left"] = steps_left_observable

    def _colorize_fingertips(self) -> None:
        """Colorize the fingertips of the hands."""
        for i, body in enumerate(self._hand.fingertip_bodies):
                color = hand_consts.FINGERTIP_COLORS[i] + (1.0,)
                for geom in body.find_all("geom"):
                    if not self._use_bothoven:
                        if geom.dclass.dclass == "plastic_visual":
                            geom.rgba = color
                    else:
                        geom.rgba = color # only one geom in finger_arm_large_N body
                # Also color the fingertip sites.
                self._hand.fingertip_sites[i].rgba = color

    def _colorize_keys(self, physics) -> None:
        """Colorize the keys by the corresponding fingertip color."""
        for key, mjcf_fingering in self._keys_current:
            key_geom = self.piano.keys[key].geom[0]
            fingertip_site = self._hand.fingertip_sites[mjcf_fingering]
            if not self.piano.activation[key]:
                physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (1.0,)
