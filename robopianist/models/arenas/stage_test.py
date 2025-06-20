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

"""Tests for stage.py."""

from absl.testing import absltest
from dm_control import mjcf

from robopianist.models.arenas import stage


class StageTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        arena = stage.Stage()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

        ### me
        with open('my_model.xml', 'w') as f:
            f.write(arena._mjcf_root.to_xml_string())
        ### me
        
        physics.step()


if __name__ == "__main__":
    absltest.main()
