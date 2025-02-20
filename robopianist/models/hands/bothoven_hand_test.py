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

"""Tests for shadow_hand.py."""

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer, mjcf

from robopianist.models.arenas import stage
from robopianist.models.hands import base as base_hand
from robopianist.models.hands import bothoven_hand
from robopianist.models.hands import bothoven_hand_constants as consts
from robopianist.models.hands.base import HandSide
from robopianist.suite.tasks import base as base_task


def _get_env():
    task = base_task.PianoTask(arena=stage.Stage())
    env = composer.Environment(
        task=task, time_limit=1.0, strip_singleton_obs_buffer_dim=True
    )
    return env


class BothovenHandConstantsTest(absltest.TestCase):
    def test_fingertip_bodies_order(self) -> None:
        expected_order = ["finger_arm_large_1", "finger_arm_large_2", "finger_arm_large_3", "finger_arm_large_4", "finger_arm_large_5"]
        self.assertEqual(consts.FINGER_ARM_BODIES, tuple(expected_order))


class BothovenHandTest(parameterized.TestCase):
    @parameterized.product(
        side=[base_hand.HandSide.RIGHT, base_hand.HandSide.LEFT],
        primitive_fingertip_collisions=[False, True],
    )
    def test_compiles_and_steps(
        self,
        side: base_hand.HandSide,
        primitive_fingertip_collisions: bool,
    ) -> None:
        robot = bothoven_hand.BothovenHand(
            side=side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
        )
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        physics.step()

    def test_set_name(self) -> None:
        robot = bothoven_hand.BothovenHand(name="larry")
        self.assertEqual(robot.name, "larry")
        self.assertEqual(robot.mjcf_model.model, "larry")

    def test_default_name(self) -> None:
        robot = bothoven_hand.BothovenHand(side=HandSide.RIGHT)
        self.assertEqual(robot.name, "rh_bothoven_hand")
        robot = bothoven_hand.BothovenHand(side=HandSide.LEFT)
        self.assertEqual(robot.name, "lh_bothoven_hand")

    def test_joints(self) -> None:
        robot = bothoven_hand.BothovenHand()
        for joint in robot.joints:
            self.assertEqual(joint.tag, "joint")
        expected_dofs = consts.NQ
        self.assertLen(robot.joints, expected_dofs)

    def test_actuators(self, ) -> None:
        robot = bothoven_hand.BothovenHand()
        for actuator in robot.actuators:
            self.assertEqual(actuator.tag, "position")
        expected_acts = consts.NU
        self.assertLen(robot.actuators, expected_acts)

    def test_fingertip_sites_order(self) -> None:
        expected_order = ["1", "2", "3", "4", "5"]
        robot = bothoven_hand.BothovenHand()
        for i, site in enumerate(robot.fingertip_sites):
            self.assertEqual(site.tag, "site")
            self.assertEqual(site.name, f"tip_site_{expected_order[i]}")

    @parameterized.named_parameters(
        {"testcase_name": "left_hand", "side": HandSide.LEFT},
        {"testcase_name": "right_hand", "side": HandSide.RIGHT},
    )
    def test_action_spec(self, side: HandSide) -> None:
        robot = bothoven_hand.BothovenHand(side=side)
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        action_spec = robot.action_spec(physics)
        expected_shape = (consts.NU,)
        self.assertEqual(action_spec.shape, expected_shape)


class BothovenHandObservableTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            "root_body",
        ]
    )
    def test_get_element_property(self, name: str) -> None:
        attribute_value = getattr(bothoven_hand.BothovenHand(), name)
        self.assertIsInstance(attribute_value, mjcf.Element)

    @parameterized.parameters(
        [
            "actuators",
            "joints",
            "joint_torque_sensors",
            "actuator_velocity_sensors",
            "actuator_force_sensors",
            "fingertip_sites",
            "fingertip_touch_sensors",
        ]
    )
    def test_get_element_tuple_property(self, name: str) -> None:
        attribute_value = getattr(bothoven_hand.BothovenHand(), name)
        self.assertNotEmpty(attribute_value)
        for element in attribute_value:
            self.assertIsInstance(element, mjcf.Element)

    @parameterized.parameters(
        [
            "joints_pos",
            "joints_pos_cos_sin",
            "joints_vel",
            "joints_torque",
            "actuators_force",
            "actuators_velocity",
            "actuators_power",
            "position",
            "fingertip_force",
        ]
    )
    def test_evaluate_observable(self, name: str) -> None:
        env = _get_env()
        physics = env.physics
        for hand in [env.task.right_hand, env.task.left_hand]:
            observable = getattr(hand.observables, name)
            observation = observable(physics)
            self.assertIsInstance(observation, (float, np.ndarray))


if __name__ == "__main__":
    absltest.main()
