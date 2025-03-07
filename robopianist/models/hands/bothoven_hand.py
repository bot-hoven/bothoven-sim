"""Bothoven hand composer class."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types

from robopianist.models.hands import base
from robopianist.models.hands import bothoven_hand_constants as consts

TIP_SITE_OFFSETS = (0.1045, -0.025, -0.0025)

class BothovenHand(base.Hand):

    def _build(
        self,
        name: Optional[str] = None,
        side: base.HandSide = base.HandSide.RIGHT,
        primitive_fingertip_collisions: bool = False,
    ) -> None:
        """Initializes a ShadowHand.

        Args:
            side: Which side (left or right) to model.
            primitive_fingertip_collisions: Whether to use capsule approximations for
                the fingertip colliders or the true meshes. Using primitive colliders
                speeds up the simulation.
        """
        xml_file = consts.BOTHOVEN_HAND_XML

        if side == base.HandSide.RIGHT:
            self._prefix = "rh_"
        elif side == base.HandSide.LEFT:
            self._prefix = "lh_"
        
        name = name or self._prefix + "bothoven_hand" # e.g. rh_bothoven_hand

        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name # rename model to rh_bothoven_hand or lh_bothoven_hand
        self._n_forearm_dofs = 0

        # Important: call before parsing.
        # self._add_dofs() # adds forearm joints and actuators

        self._parse_mjcf_elements() # gets joints / actuators into self._joints and self._actutators
        self._add_mjcf_elements() # adds fingertip sites and touch/vel/torque/force sensors to MJCF

        self._action_spec = None

    def _build_observables(self) -> "BothovenHandObservables":
        return BothovenHandObservables(self)
    
    def _parse_mjcf_elements(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _add_mjcf_elements(self) -> None:
        # Add sites to the tips of the fingers.
        # I believe the sites created here and stored in self._fingertip_sites are for
        # spatial tracking of the fingertips, whereas fingertip sites created below for
        # the touch sensors are separate because they need to be larger.
        fingertip_sites = []
        for fing_arm_name in consts.FINGER_ARM_BODIES:
            fing_arm_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", fing_arm_name
            )
            tip_site = fing_arm_elem.add(
                "site",
                name="tip_site_" + fing_arm_name[-1],
                pos=TIP_SITE_OFFSETS,
                type="sphere",
                size=(0.004,),
                group=composer.SENSOR_SITES_GROUP,
                rgba=(1, 0.514, 0.886, 0.6) # pink
            )
            fingertip_sites.append(tip_site)
        self._fingertip_sites = tuple(fingertip_sites)

        # Add joint torque sensors.
        non_solenoid_joints = [j for j in self._joints if j.name[:-1] != "Finger_Arm" ]
        joint_torque_sensors = []
        for joint_elem in non_solenoid_joints:
            # add site to body that each joint is attached to 
            # joint_elem.parent gives the body that the joint is UNDER (inside of)
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            # Add torque sensor for each joint to the root (mujoco tag) of the mjcf.
            # Bind sensor to the site created above.
            # Since the torque sensors are tied to sites attached to the bodies, I'm
            # not sure what they actually measure (i guess the movement of the body allows
            # mujoco to infer the torque of its attached joint?)
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

        # Add velocity and force sensors to the actuators.
        non_solenoid_acts = [j for j in self._actuators if j.name[:-2] != "solenoid" ]
        actuator_velocity_sensors = []
        actuator_force_sensors = []
        for actuator_elem in non_solenoid_acts:
            # Add actuatorvel tags to the sensors section of the MJCF
            velocity_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorvel",
                actuator=actuator_elem,
                name=actuator_elem.name + "_velocity",
            )
            actuator_velocity_sensors.append(velocity_sensor_elem)

            # Add actuatorfrc tags to the sensors section of the MJCF
            force_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorfrc",
                actuator=actuator_elem,
                name=actuator_elem.name + "_force",
            )
            actuator_force_sensors.append(force_sensor_elem)
        self._actuator_velocity_sensors = tuple(actuator_velocity_sensors)
        self._actuator_force_sensors = tuple(actuator_force_sensors)

        # Add touch sensors to the fingertips.
        fingertip_touch_sensors = []
        for fing_arm_name in consts.FINGER_ARM_BODIES:
            fing_arm_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", fing_arm_name
            )
            
            # Add touch site to the fingertip body
            touch_site = fing_arm_elem.add(
                "site",
                name="touch_site_" + fing_arm_name[-1],
                pos=TIP_SITE_OFFSETS,
                type="sphere",
                size=(0.01,),
                group=composer.SENSOR_SITES_GROUP,
                rgba=(0, 1, 0, 0.6),
            )
            # Add touch sensor to sensors section of MJCF (bound to site created above)
            touch_sensor = self._mjcf_root.sensor.add(
                "touch",
                site=touch_site,
                name=fing_arm_name[-1] + "_touch",
            )
            fingertip_touch_sensors.append(touch_sensor)
        self._fingertip_touch_sensors = tuple(fingertip_touch_sensors)

    # Accessors.
    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root, "body", "finger_mount_plate")

    @composer.cached_property
    def fingertip_bodies(self) -> Sequence[types.MjcfElement]:
        return tuple(
            mjcf_utils.safe_find(self._mjcf_root, "body", name)
            for name in consts.FINGER_ARM_BODIES
        )

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_sites

    @property
    def actuator_velocity_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_velocity_sensors

    @property
    def actuator_force_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_force_sensors

    @property
    def fingertip_touch_sensors(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_touch_sensors

    # Action specs.

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = spec_utils.create_action_spec(
                physics=physics, actuators=self.actuators, prefix=self.name
            )

        return self._action_spec

    def apply_action(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        physics.bind(self.actuators).ctrl = action

class BothovenHandObservables(base.HandObservables):
    """BothovenHand observables."""

    _entity: BothovenHand

    @composer.observable
    def actuators_force(self):
        """Returns the actuator forces."""
        return observable.MJCFFeature("sensordata", self._entity.actuator_force_sensors)

    @composer.observable
    def actuators_velocity(self):
        """Returns the actuator velocities."""
        return observable.MJCFFeature(
            "sensordata", self._entity.actuator_velocity_sensors
        )

    @composer.observable
    def actuators_power(self):
        """Returns the actuator powers."""

        def _get_actuator_power(physics: mjcf.Physics) -> np.ndarray:
            force = physics.bind(self._entity.actuator_force_sensors).sensordata
            velocity = physics.bind(self._entity.actuator_velocity_sensors).sensordata
            return abs(force) * abs(velocity)

        return observable.Generic(raw_observation_callable=_get_actuator_power)

    @composer.observable
    def fingertip_positions(self):
        """Returns the fingertip positions in world coordinates."""

        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    @composer.observable
    def fingertip_force(self):
        """Returns for each finger, the sum of forces felt at the fingertip."""

        def _get_fingertip_force(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_touch_sensors).sensordata

        return observable.Generic(raw_observation_callable=_get_fingertip_force)
