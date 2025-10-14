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

"""Optimization functions for practical reconstruction."""

import drjit as dr  # type: ignore
import matplotlib.pyplot as plt
import mediapy as media
import mitsuba as mi  # type: ignore
import numpy as np
import tqdm
import shutil

from pathlib import Path
from variables import parameters as parameters_lib
from core import image_util
from core import mipmap_util
from core import mitsuba_util
from core import schedule
from core import mitsuba_io

from core.optimizers import filtered_adam
from practical_reconstruction import figutils
from practical_reconstruction import scene_configuration
from practical_reconstruction import scene_preparation


def save_texture_results(params, optimized_keys, scene_config):
  texture_folder = Path(scene_config.result_folder) / 'textures'
  texture_folder.mkdir(parents=True,exist_ok=True)

  for optimized_key in optimized_keys:
    if optimized_key.endswith('.flat_buffer'):
      # Mipmap case
      texture_key = optimized_key.replace('.flat_buffer', '')
      num_mipmaps = len(params[f'{texture_key}.flat_buffer_offsets']) - 1
      for i in range(num_mipmaps):
        image = mipmap_util.mip_tensor_from_flat_buffer(
            i,
            params[f'{texture_key}.flat_buffer'],
            params[f'{texture_key}.flat_buffer_offsets'],
            params[f'{texture_key}.base_mip_shape'],
            params[f'{texture_key}.mip_factor'],
        )
        bitmap_output = mi.Bitmap(image)
        mitsuba_io.write_bitmap(
            bitmap_output,
            texture_folder / f'{texture_key}_mip_{i:02d}.exr',
        )
        mitsuba_io.write_bitmap(
            image_util.tonemap(bitmap_output),
            texture_folder / f'{texture_key}_mip_{i:02d}.png',
        )
    elif optimized_key.endswith('.data'):
      # Bitmap case
      bitmap_output = mi.Bitmap(params[optimized_key])
      mitsuba_io.write_bitmap(
          bitmap_output,
          texture_folder / f'{optimized_key}.exr',
      )
      mitsuba_io.write_bitmap(
          image_util.tonemap(bitmap_output),
          texture_folder / f'{optimized_key}.png',
      )
    elif optimized_key.endswith('.vertex_positions'):
      print(f'Possible displacement map found for {optimized_key}, skipping')
    else:
      # Scalar case
      if not optimized_key.endswith('.value'):
        raise ValueError(f'Cannot save non-scalar key: {optimized_key}')
      scalar_output = params[optimized_key]
      with open(texture_folder / f'{optimized_key}.txt', 'w') as f:
        f.write(str(scalar_output))


def save_displacement_map(optimizer, variables, optimized_keys, scene_config):
  texture_folder = Path(scene_config.result_folder) / 'textures'
  texture_folder.mkdir(parents=True,exist_ok=True)

  for optimized_key in optimized_keys:
    if not optimized_key.endswith('.vertex_positions'):
      continue
    for v in variables:
      if v.key == optimized_key:
        bitmap_output = mi.Bitmap(v.variable.get_value(optimizer))
        mitsuba_io.write_bitmap(
            bitmap_output,
            texture_folder / f'{optimized_key}.exr',
        )
        mitsuba_io.write_bitmap(
            image_util.tonemap(bitmap_output),
            texture_folder / f'{optimized_key}.png',
        )
        return


def save_optimization_videos(scene_config, frames):
  result_folder = Path(scene_config.result_folder)
  result_folder.mkdir(parents=True,exist_ok=True)

  num_sensors = len(scene_config.optimized_sensors_indices)
  target_width = frames[-1].shape[1]
  for sensor_idx in range(num_sensors):
    sensor_name = f'view_{sensor_idx:03d}'
    sensor_frames = frames[sensor_idx::num_sensors]
    media.write_video(
        result_folder / f'video_{sensor_name}.mp4',
        [
            image_util.resize_to_width(figutils.tonemap(f), target_width)
            for f in sensor_frames
        ],
        fps=20,
    )


def save_loss_data(scene_config, loss_values, emitter_keys):
  if scene_config.random_lights > 0:
    return
  result_folder = Path(scene_config.result_folder)
  result_folder.mkdir(parents=True,exist_ok=True)

  iterations = np.arange(1, scene_config.n_iter + 1)
  num_emitters = len(emitter_keys)

  num_sensors = len(scene_config.optimized_sensors_indices)
  fig, axes = plt.subplots(num_sensors, 1, figsize=(1, num_sensors * 0.75))
  if num_sensors == 1:
    axes = [axes]

  fig.tight_layout()
  loss_plot_dir_name = 'loss_plot'
  loss_plot_dir = result_folder / loss_plot_dir_name
  if not loss_plot_dir.exists():
    loss_plot_dir.mkdir(parents=True,exist_ok=True)

  for linear_sensor_idx in range(num_sensors):
    sensor_idx = scene_config.optimized_sensors_indices[linear_sensor_idx]
    for emitter_idx in range(num_emitters):
      sensor_loss_values = loss_values[
          ((linear_sensor_idx * num_emitters) + emitter_idx) :: (
              num_sensors * num_emitters
          )
      ]

      axes[linear_sensor_idx].plot(
          iterations,
          sensor_loss_values,
          label=f'emitter: {emitter_idx}',
      )
      axes[linear_sensor_idx].set_title(
          f'Losses for sensor {sensor_idx}',
          fontsize=figutils.DEFAULT_FONTSIZE_SMALL,
          loc='left',
          y=1.0,
      )
      axes[linear_sensor_idx].set_xlabel('Iterations')
      axes[linear_sensor_idx].set_ylabel('Loss')
      if scene_config.scene_setup == scene_configuration.Setup.OLAT:
        output_name = f'losses_{sensor_idx:03d}_{emitter_idx:03d}'
      else:
        output_name = f'losses_{sensor_idx:03d}'

      np.save(
          f'{scene_config.tmp_folder}/{output_name}.npy',
          np.array(sensor_loss_values),
      )
      shutil.copy(f'{scene_config.tmp_folder}/{output_name}.npy',loss_plot_dir / f'{output_name}.npy')
  fig.legend()

  figutils.savefig(
      fig,
      name=loss_plot_dir_name,
      fig_directory=Path(scene_config.tmp_folder),
      dpi=300,
      pad_inches=0.005,
      bbox_inches='tight',
      compress=False,
      target_width=figutils.COLUMN_WIDTH,
      backend=None,
  )
  shutil.copy(f'{scene_config.tmp_folder}/{loss_plot_dir_name}.pdf',
              loss_plot_dir / f'{loss_plot_dir_name}.pdf')


def _render_extra_resolutions(
    scene_config,
    scene,
    integrator,
    sensor,
    params,
    seed,
    frame_folder_tmp,
    i,
):
  """Renders the scene at extra resolutions.

  Args:
    scene_config: The scene configuration.
    scene: The Mitsuba scene to render.
    integrator: The Mitsuba integrator to use.
    sensor: The Mitsuba sensor to use.
    params: The Mitsuba parameters to use.
    seed: The Mitsuba seed to use.
    frame_folder_tmp: The temporary folder to save the frames to.
    i: The iteration number.
  """
  sensor_params = mi.traverse(sensor)
  sensor_width, sensor_height = sensor_params['film.size']
  for resolution_width in scene_config.extra_render_resolutions:
    ratio = resolution_width // sensor_width
    if ratio != int(resolution_width / sensor_width):
      raise ValueError(
          f'Resolution width {resolution_width} is not a multiple of'
          f' sensor width {sensor_width}!'
      )
    mitsuba_util.set_sensor_resolution(
        sensor, (resolution_width, sensor_height * ratio)
    )
    img = mi.render(
        scene,
        integrator=integrator,
        sensor=sensor,
        spp=scene_config.samples_per_pixel_primal,
        params=params,
        seed=seed,
    )
    output_bitmap = mi.Bitmap(img)

    mitsuba_io.write_bitmap(
        output_bitmap,
        frame_folder_tmp
        / f'{sensor.id()}_iter_{i:03d}_res_{resolution_width}.exr',
    )
    mitsuba_io.write_bitmap(
        image_util.tonemap(output_bitmap),
        frame_folder_tmp
        / f'{sensor.id()}_iter_{i:03d}_res_{resolution_width}.png',
    )
  # Reset the sensor resolution to the original resolution.
  mitsuba_util.set_sensor_resolution(sensor, (sensor_width, sensor_height))


def _render_full_on(
    scene_config,
    scene,
    integrator,
    sensor,
    params,
    seed,
    frame_folder_tmp,
    i,
    emitter_keys,
):
  with dr.suspend_grad():
    scene_preparation.switch_emitter(params, '', emitter_keys, full_on=True)
    img = mi.render(
        scene,
        integrator=integrator,
        sensor=sensor,
        spp=scene_config.samples_per_pixel_primal,
        params=params,
        seed=seed,
    )
    output_bitmap = mi.Bitmap(img)

    mitsuba_io.write_bitmap(
        output_bitmap,
        frame_folder_tmp / f'{sensor.id()}_iter_{i:03d}_full_on.exr',
    )
    mitsuba_io.write_bitmap(
        image_util.tonemap(output_bitmap),
        frame_folder_tmp / f'{sensor.id()}_iter_{i:03d}_full_on.png',
    )
    _render_extra_resolutions(
        scene_config,
        scene,
        integrator,
        sensor,
        params,
        seed,
        frame_folder_tmp,
        i,
    )


def optimize(
    scene_config,
    scene,
    all_sensors,
    all_references,
    emitter_keys,
    integrator,
    params,
    variables,
):
  """Optimizes Mitsuba scene parameters to match reference Blender images.

  Returns:
    A tuple containing the optimized Mitsuba variables, the loss values, the
    optimizer,and the optimized frames.
  """
  if scene_config.deng_comparison:
    return optimize_deng_comparison(
        scene_config,
        scene,
        all_sensors,
        all_references,
        emitter_keys,
        integrator,
        params,
        variables,
    )

  max_sensor_width = all_references[-1][0][0].shape[1]
  resolution_upsampling_schedule = schedule.exponential(
      max_sensor_width, 2, scene_config.n_resolutions, scene_config.n_iter
  )

  result_folder = Path(scene_config.result_folder)
  result_folder.mkdir(parents=True,exist_ok=True)
  frame_folder_tmp = Path(scene_config.tmp_folder) / 'frames'
  frame_folder_tmp.mkdir(parents=True,exist_ok=True)
  frame_folder = result_folder / 'frames'
  frame_folder.mkdir(parents=True,exist_ok=True)

  variables = parameters_lib.MitsubaVariables(variables, params)
  if scene_config.use_gradient_filtering:
    opt = filtered_adam.FilteredAdam(
        lr=scene_config.base_learning_rate,
        sigma_d=scene_config.filtering_sigma_d,
        a_trous_steps=scene_config.a_trous_filtering_steps,
        log_domain_filtering=scene_config.log_domain_filtering,
        beta_1=scene_config.beta_1,
        mask_updates=scene_config.mask_updates,
    )
  elif scene_config.use_sgd:
    opt = mi.ad.SGD(
        lr=scene_config.base_learning_rate, momentum=scene_config.beta_1
    )
  else:
    opt = mi.ad.Adam(
        lr=scene_config.base_learning_rate,
        beta_1=scene_config.beta_1,
        beta_2=(1 - dr.square(1.0 - scene_config.beta_1)),
        mask_updates=scene_config.mask_updates,
    )
  variables.initialize(opt)

  resolution_level = 0
  current_resolution = resolution_upsampling_schedule(0)
  view_sensors = all_sensors[resolution_level]
  view_lights_references = all_references[resolution_level]

  if len(view_lights_references) != len(view_sensors):
    raise ValueError(
        'Number of lights references does not match number of emitter keys!'
    )
  if len(view_lights_references[0]) != len(emitter_keys):
    raise ValueError(
        'Number of lights references does not match number of emitter keys!'
    )

  seed = 0
  pbar = tqdm.trange(scene_config.n_iter)
  pbar.set_description(
      'Rendering at resolution level'
      f' {resolution_level} (width={current_resolution})'
  )

  frames = []
  loss_values = []
  for i in pbar:

    if resolution_upsampling_schedule(i) != current_resolution:
      current_resolution = resolution_upsampling_schedule(i)
      resolution_level += 1
      pbar.set_description(
          'Rendering at resolution level'
          f' {resolution_level} (width={current_resolution})'
      )
      view_sensors = all_sensors[resolution_level]
      view_lights_references = all_references[resolution_level]

    with dr.isolate_grad():
      sampled_view_sensors = view_sensors
      sampled_view_lights_references = view_lights_references
      sensor_index = None
      if scene_config.random_sensors:
        sensor_index = np.random.randint(len(view_sensors), size=1)[0]
        sampled_view_sensors = [view_sensors[sensor_index]]
        sampled_view_lights_references = [view_lights_references[sensor_index]]
      for sensor, lights_ref_img in zip(
          sampled_view_sensors, sampled_view_lights_references
      ):
        sampled_emitter_keys = emitter_keys
        sampled_lights_ref_img = lights_ref_img
        if scene_config.random_lights > 0:
          light_indices = np.random.choice(
              len(lights_ref_img),
              scene_config.random_lights,
              replace=False,
          ).tolist()
          # light_index = np.random.randint(len(lights_ref_img), size=1)[0]
          sampled_emitter_keys = [
              emitter_keys[light_index] for light_index in light_indices
          ]
          sampled_lights_ref_img = [
              lights_ref_img[light_index] for light_index in light_indices
          ]
        for emitter_key, ref_img in zip(
            sampled_emitter_keys, sampled_lights_ref_img
        ):
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            scene_preparation.switch_emitter(params, emitter_key, emitter_keys)
          elif (
              scene_config.scene_setup
              == scene_configuration.Setup.ENVMAP_ROTATIONS
          ):
            envmap_rotation_idx = emitter_keys.index(emitter_key)
            scene_preparation.switch_envmap_rotation(
                scene_config, params, envmap_rotation_idx
            )
          seed += 1
          img = mi.render(
              scene,
              integrator=integrator,
              sensor=sensor,
              spp=scene_config.samples_per_pixel_primal,
              spp_grad=scene_config.samples_per_pixel_gradient,
              params=params,
              seed=seed,
          )
          rendering_loss = scene_config.loss(img, ref_img, weight=None)
          dr.backward(rendering_loss)

          loss_values.append(float(rendering_loss.array[0]))
          with dr.suspend_grad():
            if i in scene_config.output_iterations:
              saved_images = [img.numpy()]
              saved_key_sensor_id = [(emitter_key, sensor.id())]
              if scene_config.random_lights > 0 or scene_config.random_sensors:
                # Render missing lights as well:
                for e_key in emitter_keys:
                  if e_key != emitter_key or (
                      e_key == emitter_key and scene_config.random_sensors
                  ):
                    if (
                        scene_config.scene_setup
                        == scene_configuration.Setup.OLAT
                    ):
                      scene_preparation.switch_emitter(
                          params, e_key, emitter_keys
                      )
                    elif (
                        scene_config.scene_setup
                        == scene_configuration.Setup.ENVMAP_ROTATIONS
                    ):
                      envmap_rotation_idx = emitter_keys.index(e_key)
                      scene_preparation.switch_envmap_rotation(
                          scene_config, params, envmap_rotation_idx
                      )
                    # To reduce the number of rendered images we always render
                    # with the first sensor in the list of optimized sensors if
                    # random_sensors is True.
                    rendered_sensor = (
                        view_sensors[0]
                        if scene_config.random_sensors
                        else sensor
                    )

                    img = mi.render(
                        scene,
                        integrator=integrator,
                        sensor=rendered_sensor,
                        spp=scene_config.samples_per_pixel_primal,
                        params=params,
                        seed=seed,
                    )
                    saved_images.append(img.numpy())
                    saved_key_sensor_id.append((e_key, rendered_sensor.id()))
                if (
                    scene_config.render_missing_sensors
                    and sensor_index is not None
                ):
                  assert (
                      scene_config.scene_setup
                      == scene_configuration.Setup.FULL_ON
                  )
                  assert scene_config.random_sensors
                  # Render missing sensors as well:
                  for missing_sensor_idx in range(len(view_sensors)):
                    if missing_sensor_idx != sensor_index:
                      rendered_sensor = (
                          view_sensors[missing_sensor_idx]
                          if scene_config.random_sensors
                          else sensor
                      )

                      img = mi.render(
                          scene,
                          integrator=integrator,
                          sensor=rendered_sensor,
                          spp=scene_config.samples_per_pixel_primal,
                          params=params,
                          seed=seed,
                      )
                      saved_images.append(img.numpy())
                      saved_key_sensor_id.append(
                          (emitter_key, rendered_sensor.id())
                      )
              for saved_img, (saved_emitter_key, saved_sensor_id) in zip(
                  saved_images, saved_key_sensor_id
              ):
                frames.append(saved_img)
                output_bitmap = mi.Bitmap(saved_img)
                output_name = f'{saved_sensor_id}_iter_{i:03d}'
                if scene_config.scene_setup in [
                    scene_configuration.Setup.OLAT,
                    scene_configuration.Setup.ENVMAP_ROTATIONS,
                ]:
                  saved_emitter_idx = emitter_keys.index(saved_emitter_key)
                  output_name += f'_{saved_emitter_idx:03d}'
                mitsuba_io.write_bitmap(
                    output_bitmap,
                    frame_folder_tmp / f'{output_name}.exr',
                )
                mitsuba_io.write_bitmap(
                    image_util.tonemap(output_bitmap),
                    frame_folder_tmp / f'{output_name}.png',
                )
                # First sensor in the list of optimized sensors if
                # random_sensors.
                rendered_sensor = (
                    view_sensors[0] if scene_config.random_sensors else sensor
                )
                _render_extra_resolutions(
                    scene_config,
                    scene,
                    integrator,
                    rendered_sensor,
                    params,
                    seed,
                    frame_folder_tmp,
                    i,
                )
        # Re-render full on if necessary when OLAT setup is used.
        if (
            scene_config.scene_setup == scene_configuration.Setup.OLAT
            and i in scene_config.output_iterations
        ):
          # First sensor in the list of optimized sensors if random_sensors.
          rendered_sensor = (
              view_sensors[0] if scene_config.random_sensors else sensor
          )
          _render_full_on(
              scene_config,
              scene,
              integrator,
              rendered_sensor,
              params,
              seed,
              frame_folder_tmp,
              i,
              emitter_keys,
          )

    variables.evaluate_regularization_gradients(opt)
    variables.process_gradients(opt)

    # Finally, apply the accumulated gradients to the parameters.
    opt.step()
    variables.update(opt, i)

  if scene_config.rerender_spp > 0:
    print(f'Re-rendering final frame with {scene_config.rerender_spp} spp')
    # Re-render the final frame with the full scene.
    with dr.suspend_grad():
      for sensor in all_sensors[-1]:
        for emitter_key in emitter_keys:
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            scene_preparation.switch_emitter(params, emitter_key, emitter_keys)
          elif (
              scene_config.scene_setup
              == scene_configuration.Setup.ENVMAP_ROTATIONS
          ):
            emitter_idx = emitter_keys.index(emitter_key)
            scene_preparation.switch_envmap_rotation(
                scene_config, params, emitter_idx
            )

          output_name = f'{sensor.id()}_iter_{(scene_config.n_iter-1):03d}'
          if scene_config.scene_setup in [
              scene_configuration.Setup.OLAT,
              scene_configuration.Setup.ENVMAP_ROTATIONS,
          ]:
            emitter_idx = emitter_keys.index(emitter_key)
            output_name += f'_{emitter_idx:03d}'
          output_name += f'_spp_{scene_config.rerender_spp}'
          img = mi.render(
              scene,
              integrator=integrator,
              sensor=sensor,
              spp=scene_config.rerender_spp,
              params=params,
              seed=seed,
          )
          output_bitmap = mi.Bitmap(img)
          mitsuba_io.write_bitmap(
              output_bitmap,
              frame_folder_tmp / f'{output_name}.exr',
          )
          mitsuba_io.write_bitmap(
              image_util.tonemap(output_bitmap),
              frame_folder_tmp / f'{output_name}.png',
          )

  # Copy frames to remote folder and delete the local folder to avoid old
  # results being copied to the remote folder.
  try:
    shutil.rmtree(frame_folder)
  except Exception as e:
    print(f'Error deleting frame folder {str(e)}')
  shutil.copytree(frame_folder_tmp,frame_folder,dirs_exist_ok=True)
  try:
    shutil.rmtree(frame_folder_tmp)
  except Exception as e:
    print(f'Error deleting tmp frame folder {str(e)}')

  return variables, loss_values, opt, frames


def _update_light_position(params, sensor):
    idx = int(sensor.id()[len('elm__') :])
    light_positions = [
["-0.980000 0.000000 -0.198998 -0.694987 -0.000000 -1.000000 -0.000000 6.000000 -0.198998 -0.000000 0.980000 4.400000 0.000000 0.000000 0.000000 1.000000"],
["0.980000 -0.000000 0.198997 4.677945 0.000000 -1.000000 -0.000000 6.000000 0.198997 0.000000 -0.980000 -22.060001 0.000000 0.000000 0.000000 1.000000"],
["-0.966003 -0.059581 0.251571 1.557857 -0.000000 -0.973082 -0.230460 4.847700 0.258531 -0.222625 0.940000 4.200000 0.000000 0.000000 0.000000 1.000000"],
["0.966003 -0.059581 -0.251571 -5.234571 -0.000000 -0.973082 0.230460 11.070121 -0.258531 -0.222625 -0.940000 -21.179998 0.000000 0.000000 0.000000 1.000000"],
["-0.999105 -0.018370 -0.038108 0.109460 0.000000 -0.900806 0.434221 8.171104 -0.042304 0.433832 0.900000 4.000000 0.000000 0.000000 0.000000 1.000000"],
["0.999105 -0.018369 0.038108 1.138376 -0.000000 -0.900806 -0.434221 -3.552860 0.042304 0.433832 -0.900000 -20.299999 0.000000 0.000000 0.000000 1.000000"],
["-0.940579 0.137517 -0.310483 -1.252414 0.000000 -0.914330 -0.404970 3.975151 -0.339574 -0.380906 0.860000 3.800000 0.000000 0.000000 0.000000 1.000000"],
["0.940579 0.137517 0.310483 7.130620 0.000000 -0.914330 0.404970 14.909334 0.339574 -0.380906 -0.860000 -19.420000 0.000000 0.000000 0.000000 1.000000"],
["-0.824106 0.056471 0.563614 3.118070 -0.000000 -0.995018 0.099695 6.498477 0.566436 0.082159 0.820000 3.600000 0.000000 0.000000 0.000000 1.000000"],
["0.824106 0.056471 -0.563614 -12.099509 0.000000 -0.995018 -0.099695 3.806701 -0.566436 0.082159 -0.820000 -18.539999 0.000000 0.000000 0.000000 1.000000"],
["-0.828107 -0.188281 -0.528005 -2.340024 -0.000000 -0.941907 0.335874 7.679367 -0.560570 0.278139 0.780000 3.400000 0.000000 0.000000 0.000000 1.000000"],
["0.828107 -0.188281 0.528005 11.916104 0.000000 -0.941907 -0.335873 -1.389215 0.560570 0.278139 -0.780000 -17.659998 0.000000 0.000000 0.000000 1.000000"],
["-0.973272 -0.149172 0.174612 1.173058 0.000000 -0.760322 -0.649547 2.752267 0.229655 -0.632186 0.740000 3.200000 0.000000 0.000000 0.000000 1.000000"],
["0.973272 -0.149172 -0.174612 -3.541456 0.000000 -0.760322 0.649547 20.290026 -0.229655 -0.632186 -0.740000 -16.780001 0.000000 0.000000 0.000000 1.000000"],
["-0.904947 0.269682 0.329153 1.945767 -0.000000 -0.773526 0.633765 9.168825 0.425524 0.573524 0.700000 3.000000 0.000000 0.000000 0.000000 1.000000"],
["0.904947 0.269682 -0.329153 -6.941376 -0.000000 -0.773526 -0.633765 -7.942829 -0.425524 0.573524 -0.700000 -15.900000 0.000000 0.000000 0.000000 1.000000"],
["-0.683073 0.188221 -0.705680 -3.228399 -0.000000 -0.966222 -0.257713 4.711434 -0.730350 -0.176037 0.660000 2.800000 0.000000 0.000000 0.000000 1.000000"],
["0.683073 0.188221 0.705680 15.824956 0.000000 -0.966222 0.257713 11.669689 0.730350 -0.176037 -0.660000 -15.020000 0.000000 0.000000 0.000000 1.000000"],
["-0.649802 -0.227552 0.725243 3.926216 0.000000 -0.954137 -0.299370 4.503151 0.760104 -0.194531 0.620000 2.600000 0.000000 0.000000 0.000000 1.000000"],
["0.649802 -0.227552 -0.725243 -15.655350 -0.000000 -0.954137 0.299370 12.586136 -0.760104 -0.194531 -0.620000 -14.139999 0.000000 0.000000 0.000000 1.000000"],
["-0.859271 -0.377413 -0.345272 -1.426360 0.000000 -0.674991 0.737826 9.689131 -0.511521 0.633992 0.580000 2.400000 0.000000 0.000000 0.000000 1.000000"],
["0.859271 -0.377413 0.345272 7.895982 0.000000 -0.674991 -0.737826 -10.232173 0.511521 0.633993 -0.580000 -13.259999 0.000000 0.000000 0.000000 1.000000"],
["-0.906250 0.339499 -0.251897 -0.959484 -0.000000 -0.595863 -0.803086 1.984567 -0.422743 -0.727797 0.540000 2.200000 0.000000 0.000000 0.000000 1.000000"],
["0.906250 0.339499 0.251897 5.841729 0.000000 -0.595863 0.803086 23.667904 0.422743 -0.727797 -0.540000 -12.380000 0.000000 0.000000 0.000000 1.000000"],
["-0.555061 0.361198 0.749295 4.046474 0.000000 -0.900801 0.434232 8.171159 0.831810 0.241025 0.500000 2.000000 0.000000 0.000000 0.000000 1.000000"],
["0.555061 0.361198 -0.749295 -16.184486 0.000000 -0.900801 -0.434232 -3.553098 -0.831809 0.241025 -0.500000 -11.500000 0.000000 0.000000 0.000000 1.000000"],
["-0.468595 -0.168426 -0.867209 -4.036044 -0.000000 -0.981657 0.190654 6.953267 -0.883413 0.089340 0.460000 1.800000 0.000000 0.000000 0.000000 1.000000"],
["0.468595 -0.168426 0.867209 19.378593 0.000000 -0.981657 -0.190654 1.805624 0.883413 0.089339 -0.460000 -10.620000 0.000000 0.000000 0.000000 1.000000"],
["-0.626918 -0.578402 0.521944 2.909719 -0.000000 -0.669944 -0.742411 2.287943 0.779085 -0.465431 0.420000 1.600000 0.000000 0.000000 0.000000 1.000000"],
["0.626918 -0.578402 -0.521944 -11.182764 0.000000 -0.669944 0.742411 22.333052 -0.779085 -0.465431 -0.420000 -9.740000 0.000000 0.000000 0.000000 1.000000"],
["-0.954394 0.273866 0.118871 0.894353 0.000000 -0.398159 0.917317 10.586583 0.298551 0.875481 0.380000 1.400000 0.000000 0.000000 0.000000 1.000000"],
["0.954394 0.273866 -0.118871 -2.315154 -0.000000 -0.398158 -0.917317 -14.180965 -0.298551 0.875481 -0.380000 -8.860001 0.000000 0.000000 0.000000 1.000000"],
["-0.427446 0.547898 -0.719095 -3.295477 -0.000000 -0.795423 -0.606054 2.969728 -0.904041 -0.259055 0.340000 1.200000 0.000000 0.000000 0.000000 1.000000"],
["0.427445 0.547898 0.719095 16.120098 0.000000 -0.795423 0.606054 19.333195 0.904041 -0.259055 -0.340000 -7.980000 0.000000 0.000000 0.000000 1.000000"],
["-0.300234 -0.037596 0.953125 5.065623 -0.000000 -0.999223 -0.039415 5.802927 0.953866 -0.011834 0.300000 1.000000 0.000000 0.000000 0.000000 1.000000"],
["0.300233 -0.037596 -0.953125 -20.668743 0.000000 -0.999223 0.039415 6.867123 -0.953866 -0.011834 -0.300000 -7.100000 0.000000 0.000000 0.000000 1.000000"],
["-0.355108 -0.636729 -0.684452 -3.122259 0.000000 -0.732171 0.681121 9.405604 -0.934825 0.241872 0.260000 0.800000 0.000000 0.000000 0.000000 1.000000"],
["0.355108 -0.636729 0.684452 15.357941 0.000000 -0.732171 -0.681121 -8.984660 0.934825 0.241872 -0.260000 -6.220000 0.000000 0.000000 0.000000 1.000000"],
["-0.979663 -0.195527 0.045060 0.525299 0.000000 -0.224567 -0.974459 1.127707 0.200652 -0.954641 0.220000 0.600000 0.000000 0.000000 0.000000 1.000000"],
["0.979663 -0.195527 -0.045060 -0.691314 -0.000000 -0.224567 0.974459 27.438089 -0.200652 -0.954641 -0.220000 -5.340000 0.000000 0.000000 0.000000 1.000000"],
["-0.274623 0.726206 0.630244 3.451221 -0.000000 -0.655445 0.755243 9.776217 0.961552 0.207407 0.180000 0.400000 0.000000 0.000000 0.000000 1.000000"],
["0.274623 0.726206 -0.630244 -13.565372 -0.000000 -0.655445 -0.755243 -10.615351 -0.961552 0.207407 -0.180000 -4.460001 0.000000 0.000000 0.000000 1.000000"],
["-0.141237 0.130710 -0.981309 -4.606544 -0.000000 -0.991245 -0.132034 5.339831 -0.989976 -0.018648 0.140000 0.200000 0.000000 0.000000 0.000000 1.000000"],
["0.141237 0.130710 0.981309 21.888794 -0.000000 -0.991245 0.132034 8.904742 0.989976 -0.018648 -0.140000 -3.580000 0.000000 0.000000 0.000000 1.000000"],
["-0.121530 -0.564057 0.816744 4.383719 -0.000000 -0.822843 -0.568269 3.158655 0.992588 -0.069062 0.100000 0.000000 0.000000 0.000000 0.000000 1.000000"],
["0.121530 -0.564057 -0.816744 -17.668364 0.000000 -0.822843 0.568269 18.501917 -0.992588 -0.069062 -0.100000 -2.700000 0.000000 0.000000 0.000000 1.000000"],
["-0.264139 -0.939272 -0.219086 -0.795430 0.000000 -0.227153 0.973859 10.869295 -0.964485 0.257234 0.060000 -0.200000 0.000000 0.000000 0.000000 1.000000"],
["0.264139 -0.939272 0.219086 5.119891 -0.000000 -0.227153 -0.973859 -15.424898 0.964485 0.257234 -0.060000 -1.820000 0.000000 0.000000 0.000000 1.000000"],
["-0.040202 0.866772 -0.497081 -2.185407 0.000000 -0.497483 -0.867473 1.662633 -0.999192 -0.034874 0.020000 -0.400000 0.000000 0.000000 0.000000 1.000000"],
["0.040202 0.866772 0.497082 11.235791 -0.000000 -0.497484 0.867473 25.084415 0.999192 -0.034874 -0.020000 -0.940000 0.000000 0.000000 0.000000 1.000000"],
["0.020993 0.303807 0.952502 5.062511 0.000000 -0.952712 0.303874 7.519371 0.999780 -0.006379 -0.020000 -0.600000 0.000000 0.000000 0.000000 1.000000"],
["-0.020993 0.303807 -0.952502 -20.655048 -0.000000 -0.952712 -0.303874 -0.685233 -0.999780 -0.006379 0.020000 -0.060000 0.000000 0.000000 0.000000 1.000000"],
["0.066069 -0.417752 -0.906156 -4.230778 0.000000 -0.908140 0.418667 8.093334 -0.997815 -0.027661 -0.060000 -0.800000 0.000000 0.000000 0.000000 1.000000"],
["-0.066069 -0.417752 0.906156 20.235424 0.000000 -0.908140 -0.418667 -3.210671 0.997815 -0.027661 0.060000 0.820000 0.000000 0.000000 0.000000 1.000000"],
["0.251934 -0.888244 0.384126 2.220630 0.000000 -0.396929 -0.917849 1.410754 0.967744 0.231237 -0.100000 -1.000000 0.000000 0.000000 0.000000 1.000000"],
["-0.251934 -0.888243 -0.384126 -8.150770 -0.000000 -0.396929 0.917849 26.192684 -0.967744 0.231238 0.100000 1.700000 0.000000 0.000000 0.000000 1.000000"],
["0.385476 0.859711 0.335119 1.975595 -0.000000 -0.363187 0.931716 10.658581 0.922718 -0.359154 -0.140000 -1.200000 0.000000 0.000000 0.000000 1.000000"],
["-0.385477 0.859711 -0.335119 -7.072618 -0.000000 -0.363187 -0.931716 -14.497758 -0.922718 -0.359155 0.140000 2.580000 0.000000 0.000000 0.000000 1.000000"],
["0.202442 0.448156 -0.870731 -4.053656 -0.000000 -0.889142 -0.457632 3.711840 -0.979294 0.092644 -0.180000 -1.400000 0.000000 0.000000 0.000000 1.000000"],
["-0.202443 0.448156 0.870731 19.456087 0.000000 -0.889142 0.457632 16.067904 0.979294 0.092644 0.180000 3.460000 0.000000 0.000000 0.000000 1.000000"],
["0.227133 -0.242147 0.943279 5.016396 0.000000 -0.968595 -0.248645 4.756773 0.973864 0.056476 -0.220000 -1.600000 0.000000 0.000000 0.000000 1.000000"],
["-0.227133 -0.242147 -0.943279 -20.452141 -0.000000 -0.968595 0.248645 11.470198 -0.973864 0.056476 0.220000 4.340000 0.000000 0.000000 0.000000 1.000000"],
["0.445678 -0.727075 -0.522238 -2.311188 -0.000000 -0.583380 0.812199 10.060997 -0.895193 -0.361980 -0.260000 -1.800000 0.000000 0.000000 0.000000 1.000000"],
["-0.445679 -0.727075 0.522238 11.789227 -0.000000 -0.583380 -0.812199 -11.868387 0.895193 -0.361980 0.260000 5.220000 0.000000 0.000000 0.000000 1.000000"],
["0.880427 0.445805 -0.161575 -0.507873 -0.000000 -0.340744 -0.940156 1.299219 -0.474182 0.827739 -0.300000 -2.000000 0.000000 0.000000 0.000000 1.000000"],
["-0.880427 0.445805 0.161575 3.854641 -0.000000 -0.340744 0.940156 26.683435 0.474182 0.827739 0.300000 6.100000 0.000000 0.000000 0.000000 1.000000"],
["0.415865 0.523671 0.743522 4.017611 -0.000000 -0.817573 0.575825 8.879127 0.909426 -0.239466 -0.340000 -2.200000 0.000000 0.000000 0.000000 1.000000"],
["-0.415865 0.523671 -0.743522 -16.057487 -0.000000 -0.817573 -0.575825 -6.668157 -0.909426 -0.239466 0.340000 6.980000 0.000000 0.000000 0.000000 1.000000"],
["0.381113 -0.070608 -0.921828 -4.309141 -0.000000 -0.997079 0.076371 6.381858 -0.924528 -0.029106 -0.380000 -2.400000 0.000000 0.000000 0.000000 1.000000"],
["-0.381113 -0.070608 0.921828 20.580221 -0.000000 -0.997079 -0.076372 4.319825 0.924528 -0.029106 0.380000 7.860000 0.000000 0.000000 0.000000 1.000000"],
["0.563161 -0.550497 0.616281 3.381404 -0.000000 -0.745790 -0.666182 2.669092 0.826347 0.375168 -0.420000 -2.600000 0.000000 0.000000 0.000000 1.000000"],
["-0.563162 -0.550497 -0.616281 -13.258179 -0.000000 -0.745789 0.666182 20.655994 -0.826347 0.375168 0.420000 8.740000 0.000000 0.000000 0.000000 1.000000"],
["0.999956 -0.008360 -0.004332 0.278342 0.000000 -0.460021 0.887908 10.439542 -0.009416 -0.887869 -0.460000 -2.800000 0.000000 0.000000 0.000000 1.000000"],
["-0.999956 -0.008360 0.004332 0.395293 -0.000000 -0.460020 -0.887908 -13.533984 0.009416 -0.887869 0.460000 9.620000 0.000000 0.000000 0.000000 1.000000"],
["0.651734 0.486488 -0.581870 -2.609348 -0.000000 -0.767185 -0.641426 2.792868 -0.758448 0.418039 -0.500000 -3.000000 0.000000 0.000000 0.000000 1.000000"],
["-0.651734 0.486488 0.581870 13.101131 -0.000000 -0.767185 0.641426 20.111378 0.758448 0.418039 0.500000 10.500000 0.000000 0.000000 0.000000 1.000000"],
["0.541636 0.065293 0.838073 4.490367 0.000000 -0.996979 0.077673 6.388362 0.840613 -0.042070 -0.540000 -3.200000 0.000000 0.000000 0.000000 1.000000"],
["-0.541636 0.065292 -0.838073 -18.137615 0.000000 -0.996979 -0.077672 4.291206 -0.840613 -0.042070 0.540000 11.380000 0.000000 0.000000 0.000000 1.000000"],
["0.666421 -0.367185 -0.648890 -2.944452 -0.000000 -0.870321 0.492485 8.462423 -0.745576 -0.328202 -0.580000 -3.400000 0.000000 0.000000 0.000000 1.000000"],
["-0.666421 -0.367185 0.648890 14.575590 -0.000000 -0.870321 -0.492485 -4.834664 0.745576 -0.328202 0.580000 12.259999 0.000000 0.000000 0.000000 1.000000"],
["0.975295 -0.170524 0.140431 1.002154 -0.000000 -0.635705 -0.771932 2.140339 0.220906 0.752862 -0.620000 -3.600000 0.000000 0.000000 0.000000 1.000000"],
["-0.975295 -0.170524 -0.140431 -2.789478 -0.000000 -0.635705 0.771932 22.982506 -0.220906 0.752862 0.620000 13.139999 0.000000 0.000000 0.000000 1.000000"],
["0.855124 0.329636 0.400128 2.300641 0.000000 -0.771818 0.635844 9.179219 0.518423 -0.543725 -0.660000 -3.800000 0.000000 0.000000 0.000000 1.000000"],
["-0.855124 0.329636 -0.400128 -8.502822 0.000000 -0.771818 -0.635844 -7.988564 -0.518423 -0.543725 0.660000 14.020000 0.000000 0.000000 0.000000 1.000000"],
["0.712814 0.132385 -0.688746 -3.143730 0.000000 -0.982024 -0.188757 5.056216 -0.701354 0.134548 -0.700000 -4.000000 0.000000 0.000000 0.000000 1.000000"],
["-0.712814 0.132385 0.688746 15.452411 0.000000 -0.982024 0.188757 10.152647 0.701353 0.134548 0.700000 14.900000 0.000000 0.000000 0.000000 1.000000"],
["0.777573 -0.193098 0.598409 3.292046 0.000000 -0.951679 -0.307094 4.464532 0.628793 0.238788 -0.740000 -4.200000 0.000000 0.000000 0.000000 1.000000"],
["-0.777573 -0.193098 -0.598409 -12.865001 0.000000 -0.951679 0.307094 12.756061 -0.628793 0.238788 0.740000 15.780001 0.000000 0.000000 0.000000 1.000000"],
["0.963242 -0.157623 -0.217532 -0.787657 0.000000 -0.809765 0.586754 8.933769 -0.268635 -0.565186 -0.780000 -4.400000 0.000000 0.000000 0.000000 1.000000"],
["-0.963242 -0.157623 0.217532 5.085693 0.000000 -0.809765 -0.586754 -6.908585 0.268635 -0.565186 0.780000 16.660000 0.000000 0.000000 0.000000 1.000000"],
["0.967070 0.134922 -0.215805 -0.779027 0.000000 -0.847922 -0.530121 3.349396 -0.254511 0.512664 -0.820000 -4.600000 0.000000 0.000000 0.000000 1.000000"],
["-0.967070 0.134921 0.215806 5.047720 0.000000 -0.847922 0.530121 17.662657 0.254511 0.512664 0.820000 17.539999 0.000000 0.000000 0.000000 1.000000"],
["0.881302 0.103271 0.461130 2.605651 -0.000000 -0.975828 0.218538 7.092691 0.472553 -0.192598 -0.860000 -4.800000 0.000000 0.000000 0.000000 1.000000"],
["-0.881303 0.103271 -0.461130 -9.844864 0.000000 -0.975828 -0.218538 1.192159 -0.472552 -0.192599 0.860000 18.420000 0.000000 0.000000 0.000000 1.000000"],
["0.907515 -0.053941 -0.416542 -1.782710 0.000000 -0.991719 0.128424 6.642122 -0.420020 -0.116547 -0.900000 -5.000000 0.000000 0.000000 0.000000 1.000000"],
["-0.907515 -0.053941 0.416542 9.463923 -0.000000 -0.991719 -0.128425 3.174665 0.420020 -0.116547 0.900000 19.299999 0.000000 0.000000 0.000000 1.000000"],
["0.983575 -0.053131 0.172505 1.162527 0.000000 -0.955698 -0.294350 4.528250 0.180502 0.289515 -0.940000 -5.200000 0.000000 0.000000 0.000000 1.000000"],
["-0.983575 -0.053131 -0.172505 -3.495119 -0.000000 -0.955698 0.294350 12.475699 -0.180502 0.289515 0.940000 20.179998 0.000000 0.000000 0.000000 1.000000"],
["0.999092 0.008287 0.041780 0.508900 -0.000000 -0.980890 0.194562 6.972811 0.042594 -0.194386 -0.980000 -5.400000 0.000000 0.000000 0.000000 1.000000"],
["-0.999092 0.008287 -0.041780 -0.619160 0.000000 -0.980890 -0.194562 1.719633 -0.042594 -0.194385 0.980000 21.060001 0.000000 0.000000 0.000000 1.000000"],
    ]
    raw_string = light_positions[idx-1][0]
    float_values = list(map(float, raw_string.strip().split()))
    matrix_4x4 = np.array(float_values).reshape((4, 4))
    params['directional-light.to_world'] = mi.Transform4f(matrix_4x4)
    # if (idx-1)%2==0:
    #   params['arealight.emitter.radiance.value'] = mi.Color3f(1000.0,1000.0,1000.0)
    # else:
    #   params['arealight.emitter.radiance.value'] = mi.Color3f(3000.0,3000.0,3000.0)
    params.update()

def optimize_deng_comparison(
    scene_config,
    scene,
    all_sensors,
    all_references,
    emitter_keys,
    integrator,
    params,
    variables,
):
  """Optimizes Mitsuba scene parameters to match reference Blender images.

  Returns:
    A tuple containing the optimized Mitsuba variables, the loss values, the
    optimizer,and the optimized frames.
  """
  assert scene_config.deng_comparison
  assert scene_config.scene_name in ['kiwi', 'kiwi_naive']
  assert scene_config.scene_setup == scene_configuration.Setup.FULL_ON
  assert scene_config.random_lights == 0
  assert scene_config.random_sensors
  assert len(emitter_keys) == 1

  max_sensor_width = all_references[-1][0][0].shape[1]
  resolution_upsampling_schedule = schedule.exponential(
      max_sensor_width, 2, scene_config.n_resolutions, scene_config.n_iter
  )

  result_folder = Path(scene_config.result_folder)
  result_folder.mkdir(parents=True,exist_ok=True)
  frame_folder_tmp = Path(scene_config.tmp_folder) / 'frames'
  frame_folder_tmp.mkdir(parents=True,exist_ok=True)
  frame_folder = result_folder / 'frames'
  frame_folder.mkdir(parents=True,exist_ok=True)

  variables = parameters_lib.MitsubaVariables(variables, params)
  if scene_config.use_gradient_filtering:
    opt = filtered_adam.FilteredAdam(
        lr=scene_config.base_learning_rate,
        sigma_d=scene_config.filtering_sigma_d,
        a_trous_steps=scene_config.a_trous_filtering_steps,
        log_domain_filtering=scene_config.log_domain_filtering,
        beta_1=scene_config.beta_1,
    )
  elif scene_config.use_sgd:
    opt = mi.ad.SGD(
        lr=scene_config.base_learning_rate, momentum=scene_config.beta_1
    )
  else:
    opt = mi.ad.Adam(
        lr=scene_config.base_learning_rate,
        beta_1=scene_config.beta_1,
        beta_2=(1 - dr.square(1.0 - scene_config.beta_1)),
    )
  variables.initialize(opt)

  resolution_level = 0
  current_resolution = resolution_upsampling_schedule(0)
  view_sensors = all_sensors[resolution_level]
  view_lights_references = all_references[resolution_level]

  if len(view_lights_references) != len(view_sensors):
    raise ValueError(
        'Number of lights references does not match number of emitter keys!'
    )
  if len(view_lights_references[0]) != len(emitter_keys):
    raise ValueError(
        'Number of lights references does not match number of emitter keys!'
    )

  seed = 0
  pbar = tqdm.trange(scene_config.n_iter)
  pbar.set_description(
      'Rendering at resolution level'
      f' {resolution_level} (width={current_resolution})'
  )

  # pyformat: disable
  backlit_indices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99]
  # pyformat: enable
  frontlit_indices = [
      i for i in range(len(all_sensors[-1])) if i not in backlit_indices
  ]

  frames = []
  loss_values = []
  for i in pbar:
    if resolution_upsampling_schedule(i) != current_resolution:
      current_resolution = resolution_upsampling_schedule(i)
      resolution_level += 1
      pbar.set_description(
          'Rendering at resolution level'
          f' {resolution_level} (width={current_resolution})'
      )
      view_sensors = all_sensors[resolution_level]
      view_lights_references = all_references[resolution_level]

    with dr.isolate_grad():
      front_indices = np.random.choice(
          frontlit_indices,
          scene_config.deng_dual_sensor_batch_size,
          replace=False,
      ).tolist()
      back_indices = np.random.choice(
          backlit_indices,
          scene_config.deng_dual_sensor_batch_size,
          replace=False,
      ).tolist()
      sampled_view_sensors = [
          view_sensors[sensor_index]
          for sensor_index in front_indices + back_indices
      ]
      sampled_view_lights_references = [
          view_lights_references[sensor_index]
          for sensor_index in front_indices + back_indices
      ]
      for sensor, lights_ref_img in zip(
          sampled_view_sensors, sampled_view_lights_references
      ):
        assert len(lights_ref_img) == len(emitter_keys)
        ref_img = lights_ref_img[0]

        _update_light_position(params, sensor)

        seed += 1
        img = mi.render(
            scene,
            integrator=integrator,
            sensor=sensor,
            spp=scene_config.samples_per_pixel_primal,
            spp_grad=scene_config.samples_per_pixel_gradient,
            params=params,
            seed=seed,
        )

        # Clamp negative values to zero.
        ref_img = dr.maximum(ref_img, 0.0)
        # Create a mask for the non-black pixels
        mask = None
        if scene_config.deng_displacement_learning_rate == 0.0:
          tmp = dr.unravel(mi.Color3f, ref_img.array)
          non_zero = (tmp.x > 0) | (tmp.y > 0) | (tmp.x > 0)
          mask = mi.TensorXf(
              dr.ravel(dr.select(non_zero, mi.Color3f(1.0), mi.Color3f(0.0))),
              shape=ref_img.shape,
          )
        rendering_loss = scene_config.loss(img, ref_img, weight=mask)

        dr.backward(rendering_loss)
        if(i%100==0):
          print(rendering_loss)
        loss_values.append(float(rendering_loss.array[0]))
        with dr.suspend_grad():
          if i in scene_config.output_iterations:
            for sensor in all_sensors[-1]:
              emitter_key = emitter_keys[0]

              _update_light_position(params, sensor)

              output_name = f'{sensor.id()}_iter_{i:03d}'
              if scene_config.scene_setup == scene_configuration.Setup.OLAT:
                emitter_idx = emitter_keys.index(emitter_key)
                output_name += f'_{emitter_idx:03d}'
              img = mi.render(
                  scene,
                  integrator=integrator,
                  sensor=sensor,
                  spp=scene_config.samples_per_pixel_primal,
                  params=params,
                  seed=seed,
              )
              output_bitmap = mi.Bitmap(img)
              mitsuba_io.write_bitmap(
                  output_bitmap,
                  frame_folder_tmp / f'{output_name}.exr',
              )
              mitsuba_io.write_bitmap(
                  image_util.tonemap(output_bitmap),
                  frame_folder_tmp / f'{output_name}.png',
              )

    variables.evaluate_regularization_gradients(opt)
    variables.process_gradients(opt)

    # Finally, apply the accumulated gradients to the parameters.
    opt.step()
    variables.update(opt, i)

  if scene_config.rerender_spp > 0:
    print(f'Re-rendering final frame with {scene_config.rerender_spp} spp')
    # Re-render the final frame with the full scene.
    with dr.suspend_grad():
      for sensor in all_sensors[-1]:
        for emitter_key in emitter_keys:
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            scene_preparation.switch_emitter(params, emitter_key, emitter_keys)

          _update_light_position(params, sensor)

          output_name = f'{sensor.id()}_iter_{(scene_config.n_iter-1):03d}'
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            emitter_idx = emitter_keys.index(emitter_key)
            output_name += f'_{emitter_idx:03d}'
          output_name += f'_spp_{scene_config.rerender_spp}'
          img = mi.render(
              scene,
              integrator=integrator,
              sensor=sensor,
              spp=scene_config.rerender_spp,
              params=params,
              seed=seed,
          )
          output_bitmap = mi.Bitmap(img)
          mitsuba_io.write_bitmap(
              output_bitmap,
              frame_folder_tmp / f'{output_name}.exr',
          )
          mitsuba_io.write_bitmap(
              image_util.tonemap(output_bitmap),
              frame_folder_tmp / f'{output_name}.png',
          )

  # Copy frames to remote folder and delete the local folder to avoid old
  # results being copied to the remote folder.
  try:
    shutil.rmtree(frame_folder)
  except Exception as e:
    print(f'Error deleting frame folder {str(e)}')
  shutil.copytree(frame_folder_tmp,frame_folder,dirs_exist_ok=True)
  try:
    shutil.rmtree(frame_folder_tmp)
  except Exception as e:
    print(f'Error deleting tmp frame folder {str(e)}')

  return variables, loss_values, opt, frames
