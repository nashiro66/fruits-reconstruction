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


def _update_light_position(params, camera_idx, light_idx):
  # Update light position for the selected sensor
  light_positions = [
    ["-6.500000 0.000000 0.000000 0.300000 0.000001 9.396926 0.171010 -0.840401 -0.000001 3.420202 -0.469846 18.293852 0.000000 0.000000 0.000000 1.000000"],
    ["-6.500000 0.000002 -0.000000 0.300000 0.000001 9.396926 -0.171010 12.840401 -0.000000 -3.420202 -0.469846 18.293852 0.000000 0.000000 0.000000 1.000000"],
    ["-6.351649 0.331862 -0.104910 4.496384 -0.000001 9.877216 0.078112 2.875512 1.380779 1.526588 -0.482589 18.803574 0.000000 0.000000 0.000000 1.000000"],
    ["-6.351649 -1.094468 -0.091031 3.941252 0.000001 8.570578 -0.257610 16.304405 1.380778 -5.034614 -0.418748 16.249941 0.000000 0.000000 0.000000 1.000000"],
    ["-6.495393 -0.257781 0.013717 -0.248695 0.000000 7.287663 0.342381 -7.695252 -0.244695 6.842771 -0.364125 14.064992 0.000000 0.000000 0.000000 1.000000"],
    ["-6.495393 -0.021125 0.018793 -0.451723 0.000000 9.984242 0.028058 4.877662 -0.244695 0.560771 -0.498858 19.454330 0.000000 0.000000 0.000000 1.000000"],
    ["-6.168086 0.130222 0.157600 -6.003979 -0.000001 9.991476 -0.020640 6.825583 -2.050541 -0.391715 -0.474064 18.462549 0.000000 0.000000 0.000000 1.000000"],
    ["-6.168086 2.125814 0.116543 -4.361718 -0.000001 7.388578 -0.336931 19.477226 -2.050541 -6.394516 -0.350564 13.522580 0.000000 0.000000 0.000000 1.000000"],
    ["-5.463686 2.329839 -0.244523 10.080907 0.000000 9.027852 0.215046 -2.601838 3.521098 3.615210 -0.379426 14.677032 0.000000 0.000000 0.000000 1.000000"],
    ["-5.463686 -1.358760 -0.262195 10.787793 0.000001 9.680313 -0.125415 11.016590 3.521098 -2.108389 -0.406848 15.773907 0.000000 0.000000 0.000000 1.000000"],
    ["-5.470313 -3.396298 0.209990 -8.099610 -0.000001 7.775657 0.314401 -6.576035 -3.510794 5.291910 -0.327194 12.587777 0.000000 0.000000 0.000000 1.000000"],
    ["-5.470313 0.097866 0.270017 -10.500671 -0.000000 9.998359 -0.009060 6.362389 -3.510794 -0.152489 -0.420724 16.328970 0.000000 0.000000 0.000000 1.000000"],
    ["-6.340923 -0.726069 -0.103774 4.450963 -0.000001 9.439081 -0.165104 12.604167 1.429231 -3.221269 -0.460404 17.916149 0.000000 0.000000 0.000000 1.000000"],
    ["-6.340923 -1.890293 -0.056160 2.546408 0.000001 5.108218 -0.429843 23.193733 1.429231 -8.386471 -0.249160 9.466407 0.000000 0.000000 0.000000 1.000000"],
    ["-5.923949 3.505302 -0.107843 4.613736 -0.000001 5.240549 0.425842 -11.033693 2.675224 7.762055 -0.238806 9.052226 0.000000 0.000000 0.000000 1.000000"],
    ["-5.923949 1.298811 -0.195271 8.110845 0.000000 9.489015 0.157786 -0.311451 2.675224 2.876054 -0.432403 16.796135 0.000000 0.000000 0.000000 1.000000"],
    ["-4.525468 -0.664377 0.357371 -13.994832 0.000000 9.957076 0.046277 4.148911 -4.665848 0.644388 -0.346619 13.364748 0.000000 0.000000 0.000000 1.000000"],
    ["-4.525469 4.085328 0.295114 -11.504583 0.000001 8.222492 -0.284564 17.382553 -4.665847 -3.962414 -0.286236 10.949423 0.000000 0.000000 0.000000 1.000000"],
    ["-4.304458 0.367734 -0.374201 15.268055 -0.000000 9.987950 0.024538 5.018463 4.870486 0.324998 -0.330713 12.728526 0.000000 0.000000 0.000000 1.000000"],
    ["-4.304460 -4.528938 -0.298474 12.238949 -0.000001 7.966673 -0.302209 18.088356 4.870486 -4.002604 -0.263786 10.051451 0.000000 0.000000 0.000000 1.000000"],
    ["-5.629382 -4.598324 0.098103 -3.624103 -0.000000 3.924556 0.459885 -12.395420 -3.249624 7.965756 -0.169945 6.297793 0.000000 0.000000 0.000000 1.000000"],
    ["-5.629382 -2.261339 0.222938 -8.617526 -0.000000 8.918556 0.226160 -3.046399 -3.249624 3.917358 -0.386200 14.947990 0.000000 0.000000 0.000000 1.000000"],
    ["-5.920269 2.234655 0.173556 -6.642261 0.000001 8.408242 -0.270654 16.826164 -2.683360 -4.930292 -0.382916 14.816631 0.000000 0.000000 0.000000 1.000000"],
    ["-5.920269 3.943043 0.061132 -2.145266 0.000001 2.961628 -0.477569 25.102751 -2.683360 -8.699492 -0.134874 4.894962 0.000000 0.000000 0.000000 1.000000"],
    ["-3.675665 5.887233 -0.288803 11.852130 -0.000001 7.003352 0.356907 -8.276285 5.360922 4.036527 -0.198015 7.420607 0.000000 0.000000 0.000000 1.000000"],
    ["-3.675665 0.797100 -0.410448 16.717928 -0.000001 9.953188 0.048323 4.067064 5.360922 0.546525 -0.281420 10.756795 0.000000 0.000000 0.000000 1.000000"],
    ["4.302106 2.510774 0.353164 -13.826547 -0.000000 9.422405 -0.167469 12.698742 -4.872565 2.216823 0.311817 -12.972672 0.000000 0.000000 0.000000 1.000000"],
    ["4.302105 6.463549 0.189844 -7.293776 0.000001 5.065047 -0.431119 23.244743 -4.872565 5.706823 0.167618 -7.204730 0.000000 0.000000 0.000000 1.000000"],
    ["3.592300 3.445053 -0.379434 15.477367 0.000000 9.105629 0.206685 -2.267411 5.417138 -2.284540 0.251617 -10.564665 0.000000 0.000000 0.000000 1.000000"],
    ["3.592301 -2.238846 -0.401385 16.355413 -0.000000 9.632412 -0.134319 11.372764 5.417137 1.484661 0.266173 -11.146930 0.000000 0.000000 0.000000 1.000000"],
    ["4.405418 -5.563057 0.240401 -9.316023 0.000000 6.538967 0.378292 -9.131679 -4.779361 -5.127797 0.221591 -9.363658 0.000000 0.000000 0.000000 1.000000"],
    ["4.405419 -1.171019 0.362951 -14.218028 0.000000 9.872366 0.079630 2.814793 -4.779360 -1.079397 0.334553 -13.882127 0.000000 0.000000 0.000000 1.000000"],
    ["6.349392 -1.058252 -0.093012 4.020483 -0.000000 8.691961 -0.247234 15.889346 1.391121 4.830105 0.424528 -17.481133 0.000000 0.000000 0.000000 1.000000"],
    ["6.349392 -2.006408 -0.037240 1.789593 -0.000000 3.480054 -0.468746 24.749849 1.391121 9.157704 0.169971 -7.298844 0.000000 0.000000 0.000000 1.000000"],
    ["5.611467 4.313691 -0.130986 5.539443 -0.000001 5.190787 0.427363 -11.094528 3.280463 -7.378875 0.224061 -9.462442 0.000000 0.000000 0.000000 1.000000"],
    ["5.611467 1.620554 -0.238980 9.859218 -0.000000 9.470449 0.160551 -0.422023 3.280463 -2.772074 0.408793 -16.851728 0.000000 0.000000 0.000000 1.000000"],
    ["4.724389 -1.114659 0.338856 -13.254258 -0.000000 9.867426 0.081147 2.754136 -4.464319 -1.179595 0.358597 -14.843865 0.000000 0.000000 0.000000 1.000000"],
    ["4.724389 3.502377 0.295404 -11.516143 -0.000000 8.602087 -0.254971 16.198841 -4.464320 3.706407 0.312612 -13.004497 0.000000 0.000000 0.000000 1.000000"],
    ["5.144022 0.275446 -0.305347 12.513867 0.000000 9.989844 0.022529 5.098841 3.973542 -0.356585 0.395292 -16.311687 0.000000 0.000000 0.000000 1.000000"],
    ["5.144022 -3.714458 -0.242762 10.010472 0.000000 7.942291 -0.303809 18.152367 3.973542 4.808619 0.314272 -13.070869 0.000000 0.000000 0.000000 1.000000"],
    ["6.284044 -2.083320 0.074064 -2.662562 0.000000 5.794742 0.407495 -10.299814 -1.661562 -7.879133 0.280111 -11.704434 0.000000 0.000000 0.000000 1.000000"],
    ["6.284044 -0.643767 0.123693 -4.647718 0.000000 9.677689 0.125920 0.963192 -1.661563 -2.434731 0.467808 -19.212315 0.000000 0.000000 0.000000 1.000000"],
    ["6.310709 0.435642 0.117789 -4.411574 0.000000 9.833279 -0.090921 9.636825 -1.557224 1.765458 0.477346 -19.593838 0.000000 0.000000 0.000000 1.000000"],
    ["6.310709 1.847992 0.076231 -2.749225 0.000000 6.363876 -0.385685 21.427387 -1.557224 7.489059 0.308928 -12.857102 0.000000 0.000000 0.000000 1.000000"],
    ["5.828282 2.336207 -0.188032 7.821288 -0.000000 8.494363 0.263846 -4.553823 2.877695 -4.731590 0.380827 -15.733091 0.000000 0.000000 0.000000 1.000000"],
    ["5.828282 -0.627658 -0.219125 9.065010 0.000000 9.898993 -0.070886 8.835443 2.877696 1.271213 0.443801 -18.252039 0.000000 0.000000 0.000000 1.000000"],
    ["6.010379 -1.710559 0.170087 -6.503482 0.000000 8.934071 0.224624 -2.984961 -2.474943 -4.154078 0.413055 -17.022202 0.000000 0.000000 0.000000 1.000000"],
    ["6.010379 0.876233 0.185271 -7.110822 -0.000000 9.731606 -0.115064 10.602539 -2.474943 2.127925 0.449928 -18.497122 0.000000 0.000000 0.000000 1.000000"],
    ["6.428964 0.151013 -0.073331 3.233233 -0.000000 9.947407 0.051213 3.951496 0.958341 -1.013058 0.491935 -20.177393 0.000000 0.000000 0.000000 1.000000"],
    ["6.428964 -0.827041 -0.061028 2.741125 -0.000000 8.278531 -0.280472 17.218893 0.958341 5.548147 0.409403 -16.876122 0.000000 0.000000 0.000000 1.000000"],
    ["6.500000 0.000000 0.000000 0.300000 0.000000 9.396926 0.171010 -0.840401 0.000000 -3.420202 0.469846 -19.293852 0.000000 0.000000 0.000000 1.000000"],
    ["6.500000 0.000000 0.000000 0.300000 0.000000 9.396926 -0.171010 12.840401 0.000000 3.420201 0.469846 -19.293852 0.000000 0.000000 0.000000 1.000000"],
]
  raw_string = light_positions[camera_idx * 2 + light_idx][0]
  float_values = list(map(float, raw_string.strip().split()))
  matrix_4x4 = np.array(float_values).reshape((4, 4))
  params['arealight.to_world'] = mi.Transform4f(matrix_4x4)
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

  # Hardcoded for simplicity.
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
  backlit_indices = [1,2,3,4,20,21,23,24,25,26,27,28,30,31,34,38,41,45,47]
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
          len(frontlit_indices),
          scene_config.deng_dual_sensor_batch_size,
          replace=False,
      ).tolist()
      back_indices = np.random.choice(
          len(backlit_indices),
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
      for idx, (sensor, lights_ref_img) in enumerate(zip(
          sampled_view_sensors, sampled_view_lights_references)):
        assert len(lights_ref_img) == len(emitter_keys)
        ref_img = lights_ref_img[0]

        _update_light_position(params, idx, 0)

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

        loss_values.append(float(rendering_loss.array[0]))
        with dr.suspend_grad():
          if i in scene_config.output_iterations:
            for idx, sensor in all_sensors[-1]:
              emitter_key = emitter_keys[0]

              _update_light_position(params, idx, 0)

              output_name = f'{sensor.id()}_iter_{i:03d}'
              if scene_config.scene_setup == scene_configuration.Setup.OLAT:
                emitter_idx = emitter_keys.index(emitter_key)
                output_name += f'_{emitter_idx:03d}'
              img = mi.render(
                  scene,
                  integrator=integrator,
                  sensor=sensor,
                  spp=32,
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
      for idx, sensor in all_sensors[-1]:
        for emitter_key in emitter_keys:
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            scene_preparation.switch_emitter(params, emitter_key, emitter_keys)

          _update_light_position(params, idx, 0)

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
