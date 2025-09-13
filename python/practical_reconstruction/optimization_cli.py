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

import mitsuba as mi  # type: ignore
import numpy as np
from pathlib import Path

from practical_reconstruction import io_utils
from practical_reconstruction import optimization
from practical_reconstruction import scene_configuration
from practical_reconstruction import scene_preparation


def run_config(gin_config_name: str, gin_overrides: list[str], sss_config: bool = False):
  """Runs a synthetic scene optimization with the given config.

  Args:
    gin_config_name: The name of the gin config file to use.
    gin_overrides: A list of gin parameter overrides to use.
  """

  # Deterministic random seed
  np.random.seed(0)

  scene_config = scene_configuration.SceneConfig.get_instance(
      gin_config_name, gin_overrides, sss_config=sss_config
  )

  Path(scene_config.result_folder).mkdir(parents=True,exist_ok=True)

  print('Preparing Mitsuba scene for optimization')
  tmp_mitsuba_xml = io_utils.mitsuba_remote_to_local(scene_config)

  scene = scene_preparation.load_mitsuba_scene(scene_config, tmp_mitsuba_xml)
  params = mi.traverse(scene)

  emitter_keys = scene_preparation.get_emitter_keys(scene_config, params)

  print('Preparing references and sensors for optimization')
  sensors, references = (
      scene_preparation.generate_references_and_retrieve_sensors(
          scene, scene_config, emitter_keys
      )
  )
  all_sensors, all_references = (
      scene_preparation.create_intermediate_resolution(
          scene_config, sensors, references
      )
  )

  print('Preparing optimization variables')
  optimized_keys = scene_preparation.get_scene_keys_for_optimization(
      scene, scene_config
  )
  
  scene_preparation.initialize_optimized_parameter(
      scene_config, params, optimized_keys
  )
  variables = scene_preparation.create_variables(
      params, optimized_keys, scene_config
  )

  if scene_config.sss_optimization:
    integrator = mi.load_dict({
        'type': 'prb_path_volume',
        'max_sss_depth': -1,
        'max_path_depth': scene_config.optimized_path_depth,
    })
  else:
    integrator = mi.load_dict(
        {'type': 'prb_raydiff', 'max_depth': scene_config.optimized_path_depth}
    )

  print('Starting optimization')

  mts_variables, loss_values, opt, frames = optimization.optimize(
      scene_config,
      scene,
      all_sensors,
      all_references,
      emitter_keys,
      integrator,
      params,
      variables,
  )

  print('Saving optimized textures')
  optimization.save_texture_results(params, optimized_keys, scene_config)

  if not scene_config.deng_comparison:
    print('Saving optimized videos')
    optimization.save_optimization_videos(scene_config, frames)

    print('Saving loss data')
    optimization.save_loss_data(scene_config, loss_values, emitter_keys)

  print('--------- Done! ----------')
  del mts_variables, loss_values, opt, frames

  return