# Copyright 2025 Google LLC
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

import shutil
from pathlib import Path

def mitsuba_remote_to_local(scene_config,override=False) -> str:
  """
  Copies a Mitsuba scene from a source directory to a local temporary directory
  using the pathlib module.

  Args:
    scene_config: An object (likely a dataclass or similar) containing
                  configuration information, including:
                    - tmp_folder: The path to the temporary folder.
                    - scene_folder: The path to the source folder containing the
                                    'mts_scene' subdirectory.
                    - scene_name: The name of the Mitsuba scene XML file.

  Returns:
    The local path to the copied Mitsuba scene XML file.
  """
  # Make sure the tmp folder exists
  Path(scene_config.tmp_folder).mkdir(parents=True, exist_ok=True)

  # From e.g. CNS
  source_dir = Path(scene_config.scene_folder) / 'mts_scene'
  # To tmp
  tmp_dir_target = Path(scene_config.tmp_folder) / 'mts_scene'

  # Copy mitsuba scene to tmp
  if not tmp_dir_target.exists() or override:
    if tmp_dir_target.exists() and override:
        shutil.rmtree(tmp_dir_target)
    try:
        shutil.copytree(source_dir, tmp_dir_target, symlinks=True, dirs_exist_ok=True)
    except Exception as e:
        print(f"Error copying directory: {e}")
  else:
    print(f'Skipping already existing mitsuba scene: {tmp_dir_target}')

  mts_scene_tmp = tmp_dir_target / f'{scene_config.scene_name}.xml'
  return str(mts_scene_tmp)