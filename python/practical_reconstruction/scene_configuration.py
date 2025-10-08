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

"""Describes different synthetic scene optimization configurations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import dataclasses
import enum
from typing import Optional, TypeVar

import os
import gin
from pathlib import Path 
import mitsuba as mi  # type: ignore

from core import losses

_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_GIN_SCENE_CONFIGS_PATH = os.path.join(_ROOT_PATH,'scenes/scene_configs')
_GIN_SSS_SCENE_CONFIGS_PATH = os.path.join(_ROOT_PATH,'scenes/sss_scene_configs')

_DEFAULT_CLAMP_RANGES = {
    'base_color': (0.001, 0.995),
    'roughness': (0.005, 0.995),
    'alpha': (0.005, 0.995),
    'anisotropic': (0.0, 1.0),
    'metallic': (0.0, 1.0),
    'spec_trans': (0.0, 1.0),
    'eta': (1.0, 2.5),
    'specular': (0.005, 1.0),
    'spec_tint': (0.005, 1.0),
    'sheen': (0.0, 1.0),
    'sheen_tint': (0.005, 1.0),
    'flatness': (0.005, 1.0),
    'clearcoat': (0.0, 1.0),
    'clearcoat_gloss': (0.05, 0.995),
    'normalmap': (0.0, 1.0),
    'single_scattering_albedo': (0.0001, 0.99999),
    'extinction_coefficient': (0.01, 2000),
    'hg_coefficient': (-0.99, 0.99),
}

_DEFAULT_PARAM_LEARNING_RATE = 0.1

_DEFAULT_PARAM_VALUES = {
    'base_color': [0.5, 0.5, 0.5],
    'roughness': 0.5,
    'alpha': 0.25,
    'anisotropic': 0.0,
    'metallic': 0.5,
    'spec_trans': 0.0,
    'eta': 1.5,
    'specular': 0.5,
    'spec_tint': [0.0, 0.0, 0.0],
    'sheen': 0.5,
    'sheen_tint': [0.0, 0.0, 0.0],
    'flatness': 0.0,
    'clearcoat': 0.0,
    'clearcoat_gloss': 0.5,
    'normalmap': [0.5, 0.5, 1.0],
    'opacity': 1.0,
    'single_scattering_albedo': [0.2, 0.2, 0.2],
    'extinction_coefficient': [100, 100, 100],
    'hg_coefficient': 0.0,
}


# Very hacky but this is so much simpler than rewriting the whole thing.
# pytype: disable=signature-mismatch
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-arg-types
class DefaultDictKey(defaultdict):

  def __missing__(self, key):
    self[key] = self.default_factory(key)
    return self[key]


# pytype: enable=wrong-arg-types
# pytype: enable=wrong-arg-count
# pytype: enable=signature-mismatch


@gin.constants_from_enum
class Approach(enum.Enum):
  """Various approaches that can be used for optimization."""

  # Naive texture only optimization (the default in mitsuba)
  NAIVE = 0
  # Texture optimization with mipmapped lookup
  MIPMAP = 1
  # Texture optimization with mipmapped lookup and latent laplacian pyramid
  MIPMAP_PYRAMID = 2
  # Large steps for texture optimization
  LARGE_STEPS = 3
  # Large steps for texture optimization with mipmapped lookup
  LARGE_STEPS_MIPMAP = 4


@gin.constants_from_enum
class Setup(enum.Enum):
  """Setup type determining how lights and cameras are set up."""

  # All the lights are always on and multiple cameras can be used.
  FULL_ON = 0
  # Only one light at a time is on for a given camera at every optimization
  # step.
  OLAT = 1
  # All the lights are always on and multiple cameras can be used.
  ENVMAP_ROTATIONS = 2


@gin.constants_from_enum
class DiffuseSwitch(enum.Enum):
  NONE = 0
  ENTRY_EXIT = 1
  EXIT = 2
  FORCED_EXIT = 3
  ENTRY_FORCED_EXIT = 4


@gin.configurable
@dataclasses.dataclass
class SceneConfig:
  """Configuration settings for a scene optimization.

  Attributes:
    approach: The approach to use for optimization.
    scene_setup: Which setup to use for cameras and lights in the optimization.
    random_lights: Whether to optimize batches of lights (0 : use all lights).
    random_sensors: Whether to optimize one random sensor at a time.
    envmap_rotations: The rotations of the environment map if using the
      ENVMAP_ROTATIONS setup.
    extra_render_resolutions: The extra resolutions at which to render when
      saving the optimization results (not the one used for optimization).
    scene_name: The name of the scene.
    scene_folder: The folder containing the scene data.
    result_folder: The folder to save the optimization results.
    tmp_folder: The folder to save temporary files.
    n_resolutions: The number of resolutions to use.
    n_texture_scales: The number of texture scales to use.
    optimized_path_depth: The depth of the optimized path.
    rerender_spp: The number of samples per pixel for the rerendering pass. No
      rerendering is done if this is set to 0.
    samples_per_pixel_primal: The number of samples per pixel for the primal
      pass.
    samples_per_pixel_gradient: The number of samples per pixel for the gradient
      pass.
    base_learning_rate: The base learning rate for the optimizer.
    beta_1: The beta_1 for the optimizer.
    use_sgd: Whether to use SGD instead of Adam (beta_1 is used as momentum).
    mask_updates: Whether to mask the optimizer updates (only non-zero gradients
      contribute to the Adam state).
    use_mitsuba_reference: Whether to use mitsuba or Blender for generating the
      reference image.
    reference_scaling: The scaling factor for the reference image.
    reference_spp: The number of samples per pixel for the reference image.
    optimized_sensors_indices: The indices of the sensors to optimize.
    optimized_shapes: The shapes to optimize.
    per_material_learning_rates: The learning rate scaling for each material and
      parameter combinations, a dictionary of dictionaries.
    per_material_initial_values: The initial values for each material and
      parameter combinations, a dictionary of dictionaries.
    per_param_clamp_ranges: The clamp ranges for each parameter.
    ensure_frequency_decomposition: Whether to ensure frequency decomposition
      for the mipmapped textures.
    n_iter: The number of iterations to run the optimization.
    output_iterations: The iterations at which to output the frames.
    loss: The loss function to use.
    large_steps_lambda: The lambda for the large steps approach.
    use_conjugate_gradient_large_steps: Whether to use the conjugate gradient
      method for the large steps approach.
    use_gradient_filtering: Whether to use gradient filtering.
    filtering_sigma_d: sigma_d for the edge stopping filter.
    log_domain_filtering: Whether to use log domain filtering for gradient
      filtering.
    a_trous_filtering_steps: The number of a-trous steps to use for gradient
      filtering (F in the paper).
    mip_bias: The mip bias to use for mipmap optimization.
    mip_min_res: The minimum resolution to use for the mipmap creation.
    random_initialisation: Whether to use random initialisation of the
      parameters.
    sss_optimization: Whether we are optimizingthe SSS parameters.
    sss_diffuse_switch: The diffuse switch mode for the SSS optimization. #
      sss_hg_coefficient : The HG coefficients for the SSS optimization.
      TODO(pweier): Add a way to set the coefficients per material.
    sss_extinction_coefficient: The extinction coefficients for the SSS
      optimization. TODO(pweier): Add a way to set the coefficients per
      material.
    sss_albedo_resolution: The resolution of the albedo texture if specified.
    sss_volume_albedo_remapping: Whether to use volume albedo remapping.
    sss_extinction_resolution: The resolution of the extinction coefficient
      texture.
    sss_hg_resolution: The resolution of the HG coefficient texture.
    sss_uniform_extinction: Whether to use a uniform extinction coefficient.
    sss_uniform_hg: Whether to use a uniform HG coefficient.
    per_material_sss_albedo_scaling: The single scattering albedo scaling for
      each material.
    deng_comparison: Whether we compare to Deng et. al 2022.
    deng_dual_sensor_batch_size: The batch size for a pair of front- and
      back-lit sensor.
    deng_displacement_scale: The scale of the displacement map.
    deng_displacement_learning_rate: The learning rate of the displacement map.
    deng_displacement_res: The resolution of the displacement map.
    render_missing_sensors: Whether to render missing sensors.
    laplacian_pyramid_regularisation: Whether to regularise the laplacian
      pyramid.
    regularisation_weight: The weight of the regularisation term.
  """

  approach: Approach = Approach.MIPMAP_PYRAMID
  scene_setup: Setup = Setup.FULL_ON
  random_lights: int = 0
  random_sensors: bool = False
  envmap_rotations: list[float] = dataclasses.field(
      default_factory=lambda: [0.0, 45.0]
  )
  extra_render_resolutions: list[int] = dataclasses.field(
      default_factory=lambda: []
  )
  scene_name: str = ''
  scene_folder: str = ''
  result_folder: str = ''
  tmp_folder: str = ''
  n_resolutions: int = 1
  n_texture_scales: int = 1
  optimized_path_depth: int = 3
  rerender_spp: int = 0
  samples_per_pixel_primal: int = 2
  samples_per_pixel_gradient: int | None = None
  base_learning_rate: float = 0.1
  beta_1: float = 0.9
  use_sgd: bool = False
  mask_updates: bool = False
  use_mitsuba_reference: bool = False
  reference_scaling: float = 1.0
  reference_spp: int = 1024
  optimized_sensors_indices: list[int] = dataclasses.field(
      default_factory=lambda: [0]
  )
  optimized_shapes: list[str] | None = None
  per_material_learning_rates: dict[str, dict[str, float]] = dataclasses.field(
      default_factory=lambda: defaultdict(
          lambda: defaultdict(lambda: _DEFAULT_PARAM_LEARNING_RATE)
      )
  )
  per_material_initial_values: dict[str, dict[str, float]] = dataclasses.field(
      default_factory=lambda: defaultdict(lambda: _DEFAULT_PARAM_VALUES)
  )
  per_param_clamp_ranges: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=lambda: _DEFAULT_CLAMP_RANGES
  )
  ensure_frequency_decomposition: bool = True
  n_iter: int = 128
  output_iterations: list[int] = dataclasses.field(
      default_factory=lambda: [15, 31, 63, 127]
  )
  loss: Callable[..., mi.Float] = losses.l2_norm
  large_steps_lambda: float = 10.0
  use_conjugate_gradient_large_steps: bool = False
  use_gradient_filtering: bool = False
  filtering_sigma_d: float = 0.1
  log_domain_filtering: bool = True
  a_trous_filtering_steps: int = 5
  mip_bias: float = 0.0
  mip_min_res: int = 16
  random_initialisation: bool = False
  sss_optimization: bool = False
  sss_diffuse_switch: DiffuseSwitch = DiffuseSwitch.FORCED_EXIT
  sss_hg_coefficient: list[float] = dataclasses.field(
      default_factory=lambda: [0.0, 0.0, 0.0]
  )
  sss_extinction_coefficient: list[float] = dataclasses.field(
      default_factory=lambda: [1.0, 1.0, 1.0]
  )
  sss_volume_albedo_remapping: bool = True
  sss_albedo_resolution: int | None = None
  sss_extinction_resolution: int = 1
  sss_hg_resolution: int = 1
  sss_uniform_extinction: bool = False
  sss_uniform_hg: bool = False
  per_material_sss_albedo_scaling: dict[str, float] = dataclasses.field(
      default_factory=lambda: defaultdict(lambda: 1.0)
  )
  deng_comparison: bool = False
  deng_dual_sensor_batch_size: int = 1
  deng_displacement_scale: float = 0.005
  deng_displacement_learning_rate: float = 0.0
  deng_displacement_res: int = 512
  render_missing_sensors: bool = False
  laplacian_pyramid_regularisation: bool = False
  regularisation_weight: float = 1.0

  def __post_init__(self):
    if self.n_iter - 1 not in self.output_iterations:
      self.output_iterations.append(self.n_iter - 1)
    # Ensures default learning rate for all materials even if changed in gin
    # If some material was changed, keep its learning rate and set the rest to
    # the default.
    for k in self.per_material_learning_rates.keys():
      self.per_material_learning_rates[k] = defaultdict(
          lambda: _DEFAULT_PARAM_LEARNING_RATE,
          self.per_material_learning_rates[k],
      )
    # For any material that wasn't changed in gin still ensure the default
    # learning rate and preserve changed learning rates from other materials.
    self.per_material_learning_rates = defaultdict(
        lambda: defaultdict(lambda: _DEFAULT_PARAM_LEARNING_RATE),
        self.per_material_learning_rates,
    )

    # Ensures default values for all materials even if changed in gin
    # If some material was changed, keep its default value and set the rest to
    # the default.
    for k in self.per_material_initial_values.keys():
      # pytype: disable=wrong-arg-types
      self.per_material_initial_values[k] = DefaultDictKey(
          lambda key: _DEFAULT_PARAM_VALUES[key],
          self.per_material_initial_values[k],
      )
      # pytype: enable=wrong-arg-types

    # For any material that wasn't changed in gin still ensure the default
    # learning rate and preserve changed learning rates from other materials.
    self.per_material_initial_values = defaultdict(
        lambda: _DEFAULT_PARAM_VALUES,
        self.per_material_initial_values,
    )

    if not self.tmp_folder:
      self.tmp_folder = f'tmp/{self.scene_name}'
    if self.samples_per_pixel_gradient is None:
      self.samples_per_pixel_gradient = self.samples_per_pixel_primal

  @classmethod
  def get_instance(
      cls,
      requested_config: str,
      bindings: Optional[Sequence[str]] = None,
      sss_config: bool = False,
  ) -> SceneConfig:
    """Returns a SceneConfig instance loaded from a Gin config file.

    Args:
      requested_config: The name of the Gin config file (without extension) or
        the path to the Gin config file.
      bindings: An optional sequence of Gin bindings to override the values in
        the config file.
      sss_config: Whether to use the SSS scene configs.

    Returns:
      A SceneConfig instance.
    """
    gin.clear_config()  # Drop potentially existing Gin configs.
    if sss_config:
      config_base_path = _GIN_SSS_SCENE_CONFIGS_PATH
    else:
      config_base_path = _GIN_SCENE_CONFIGS_PATH
    config_file = os.path.join(config_base_path, requested_config + '.gin')
    if not os.path.exists(config_file):
      print(config_file)
      raise ValueError('Unknown configuration specified.')

    if bindings is None:
      bindings = []
    gin.parse_config_files_and_bindings([str(config_file)], bindings)
    return SceneConfig()
