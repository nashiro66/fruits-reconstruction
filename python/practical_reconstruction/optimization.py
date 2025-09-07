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
  # Update light position for the selected sensor
  light_positions = [
["-6.500000 0.000000 0.000000 0.300000 0.000001 4.980974 0.043578 5.564221 -0.000001 0.435778 -0.498097 4.480974 0.000000 0.000000 0.000000 1.000000"],
["-6.500000 0.000001 -0.000000 0.300000 0.000001 4.980973 -0.043578 6.435779 -0.000000 -0.435780 -0.498097 4.480974 0.000000 0.000000 0.000000 1.000000"],
["-6.351649 -0.111248 -0.105630 1.356295 0.000001 4.972498 -0.052370 6.523700 1.380779 -0.511746 -0.485901 4.359011 0.000000 0.000000 0.000000 1.000000"],
["-6.351649 -0.292981 -0.102093 1.320930 0.000001 4.806015 -0.137921 7.379209 1.380779 -1.347730 -0.469633 4.196327 0.000000 0.000000 0.000000 1.000000"],
["-6.495393 -0.088996 0.016586 0.134141 0.000000 4.405818 0.236406 3.635943 -0.244695 2.362381 -0.440270 3.902695 0.000000 0.000000 0.000000 1.000000"],
["-6.495393 -0.058843 0.017879 0.121207 0.000000 4.749398 0.156308 4.436921 -0.244695 1.561971 -0.474603 4.246031 0.000000 0.000000 0.000000 1.000000"],
["-6.168086 0.470791 0.150544 -1.205442 0.000000 4.772093 -0.149236 7.492355 -2.050541 -1.416151 -0.452841 4.028413 0.000000 0.000000 0.000000 1.000000"],
["-6.168086 0.725055 0.140082 -1.100819 -0.000001 4.440450 -0.229835 8.298348 -2.050540 -2.180987 -0.421370 3.713704 0.000000 0.000000 0.000000 1.000000"],
["-5.463686 0.492355 -0.266341 2.963411 0.000000 4.916698 0.090889 5.091106 3.521098 0.763986 -0.413281 3.632814 0.000000 0.000000 0.000000 1.000000"],
["-5.463686 0.022379 -0.270844 3.008444 0.000000 4.999829 0.004131 5.958689 3.521098 0.034725 -0.420269 3.702692 0.000000 0.000000 0.000000 1.000000"],
["-5.470313 -1.096791 0.246786 -2.167863 -0.000001 4.569083 0.203064 3.969364 -3.510794 1.708955 -0.384528 3.345279 0.000000 0.000000 0.000000 1.000000"],
["-5.470313 -0.651588 0.262083 -2.320827 -0.000000 4.852284 0.120637 4.793627 -3.510794 1.015267 -0.408362 3.583618 0.000000 0.000000 0.000000 1.000000"],
["-6.340923 -0.619251 -0.090842 1.208421 -0.000001 4.131406 -0.281629 8.816291 1.429231 -2.747365 -0.403030 3.530296 0.000000 0.000000 0.000000 1.000000"],
["-6.340923 -0.767587 -0.078709 1.087088 0.000001 3.579597 -0.349091 9.490915 1.429231 -3.405480 -0.349199 2.991992 0.000000 0.000000 0.000000 1.000000"],
["-5.923949 1.413812 -0.149531 1.795307 0.000001 3.633151 0.343514 2.564856 2.675224 3.130710 -0.331117 2.811170 0.000000 0.000000 0.000000 1.000000"],
["-5.923949 1.132676 -0.171810 2.018096 0.000000 4.174462 0.275207 3.247933 2.675225 2.508169 -0.380451 3.304507 0.000000 0.000000 0.000000 1.000000"],
["-4.525468 0.604075 0.353791 -3.237914 0.000001 4.928673 -0.084154 6.841537 -4.665848 -0.585900 -0.343147 2.931470 0.000000 0.000000 0.000000 1.000000"],
["-4.525468 1.209250 0.337927 -3.079269 0.000001 4.707664 -0.168461 7.684607 -4.665848 -1.172866 -0.327760 2.777598 0.000000 0.000000 0.000000 1.000000"],
["-4.304457 -0.790902 -0.366210 3.962096 -0.000001 4.887320 -0.105551 7.055512 4.870488 -0.698985 -0.323650 2.736502 0.000000 0.000000 0.000000 1.000000"],
["-4.304458 -1.414803 -0.346912 3.769122 -0.000001 4.629783 -0.188815 7.888151 4.870486 -1.250379 -0.306595 2.565955 0.000000 0.000000 0.000000 1.000000"],
["-5.629383 -1.966910 0.154266 -1.242665 0.000001 3.085686 0.393428 2.065722 -3.249623 3.407316 -0.267239 2.172386 0.000000 0.000000 0.000000 1.000000"],
["-5.629383 -1.669147 0.186078 -1.560779 0.000001 3.721989 0.333868 2.661317 -3.249623 2.891496 -0.322346 2.723460 0.000000 0.000000 0.000000 1.000000"],
["-5.920268 1.528452 0.138724 -1.087242 -0.000001 3.360365 -0.370242 9.702425 -2.683361 -3.372207 -0.306066 2.560656 0.000000 0.000000 0.000000 1.000000"],
["-5.920269 1.746123 0.110075 -0.800754 -0.000001 2.666393 -0.422970 10.229698 -2.683361 -3.852454 -0.242858 1.928580 0.000000 0.000000 0.000000 1.000000"],
["-3.675665 2.095838 -0.355149 3.851489 0.000000 4.306103 0.254116 3.458842 5.360922 1.436991 -0.243505 1.935045 0.000000 0.000000 0.000000 1.000000"],
["-3.675665 1.447288 -0.386147 4.161473 -0.000000 4.681951 0.175480 4.245195 5.360922 0.992319 -0.264758 2.147582 0.000000 0.000000 0.000000 1.000000"],
["-3.107382 -1.205280 0.422301 -3.923008 0.000001 4.808008 0.137224 4.627755 -5.709131 0.656014 -0.229851 1.798510 0.000000 0.000000 0.000000 1.000000"],
["-3.107382 -0.453653 0.436815 -4.068145 0.000001 4.973251 0.051650 5.483504 -5.709131 0.246916 -0.237751 1.877506 0.000000 0.000000 0.000000 1.000000"],
["-4.135763 -2.613756 -0.283677 3.136768 0.000000 3.677116 -0.338804 9.388041 5.014525 -2.155713 -0.233964 1.839643 0.000000 0.000000 0.000000 1.000000"],
["-4.135762 -3.066647 -0.233980 2.639798 -0.000001 3.032926 -0.397509 9.975092 5.014526 -2.529236 -0.192976 1.429763 0.000000 0.000000 0.000000 1.000000"],
["-6.216394 1.381900 -0.047364 0.773643 -0.000000 1.621159 0.472989 1.270112 1.899062 4.523516 -0.155043 1.050425 0.000000 0.000000 0.000000 1.000000"],
["-6.216394 1.278659 -0.070641 1.006412 -0.000000 2.417867 0.437652 1.623480 1.899062 4.185565 -0.231237 1.812371 0.000000 0.000000 0.000000 1.000000"],
["-2.830673 2.397600 0.380923 -3.509235 0.000001 4.231568 -0.266342 8.663424 -5.851264 -1.159890 -0.184280 1.342797 0.000000 0.000000 0.000000 1.000000"],
["-2.830673 3.022642 0.333502 -3.035025 -0.000000 3.704782 -0.335777 9.357765 -5.851264 -1.462267 -0.161339 1.113389 0.000000 0.000000 0.000000 1.000000"],
["-1.991337 0.228000 -0.475412 5.054116 0.000000 4.994260 0.023952 5.760482 6.187453 0.073378 -0.153004 1.030039 0.000000 0.000000 0.000000 1.000000"],
["-1.991337 -0.601007 -0.472148 5.021482 -0.000001 4.959978 -0.063137 6.631364 6.187453 -0.193424 -0.151954 1.019536 0.000000 0.000000 0.000000 1.000000"],
["-2.352270 -3.455790 0.312785 -2.827851 -0.000000 3.355264 0.370705 2.292952 -6.059441 1.341535 -0.121423 0.714229 0.000000 0.000000 0.000000 1.000000"],
["-2.352270 -2.860144 0.368042 -3.380423 -0.000000 3.948012 0.306809 2.931906 -6.059441 1.110305 -0.142874 0.928737 0.000000 0.000000 0.000000 1.000000"],
["-6.373149 -0.933616 -0.030760 0.607603 0.000001 1.564636 -0.474889 10.748886 1.277880 -4.656209 -0.153410 1.034102 0.000000 0.000000 0.000000 1.000000"],
["-6.373149 -0.972847 -0.014081 0.440809 0.000001 0.716232 -0.494844 10.948436 1.277880 -4.851864 -0.070225 0.202254 0.000000 0.000000 0.000000 1.000000"],
["-1.819810 3.883368 -0.282132 3.121323 -0.000000 2.938852 0.404514 1.954861 6.240055 1.132521 -0.082279 0.322793 0.000000 0.000000 0.000000 1.000000"],
["-1.819809 3.334453 -0.345280 3.752800 -0.000000 3.596635 0.347336 2.526642 6.240056 0.972438 -0.100695 0.506953 0.000000 0.000000 0.000000 1.000000"],
["-0.936766 0.223066 0.494277 -4.642772 0.000000 4.994916 -0.022542 6.225419 -6.432144 -0.032487 -0.071986 0.219856 0.000000 0.000000 0.000000 1.000000"],
["-0.936767 1.077980 0.482894 -4.528945 -0.000000 4.879889 -0.108935 7.089352 -6.432143 -0.156996 -0.070328 0.203279 0.000000 0.000000 0.000000 1.000000"],
["-0.805985 -2.452259 -0.431301 4.613009 0.000000 4.346553 -0.247133 8.471331 6.449836 -0.306439 -0.053896 0.038963 0.000000 0.000000 0.000000 1.000000"],
["-0.805985 -3.163949 -0.382165 4.121655 0.000000 3.851377 -0.318856 9.188557 6.449836 -0.395374 -0.047756 -0.022438 0.000000 0.000000 0.000000 1.000000"],
["-1.749547 -4.766867 0.068249 -0.382485 0.000000 0.708638 0.494953 1.050471 -6.260118 1.332221 -0.019074 -0.309262 0.000000 0.000000 0.000000 1.000000"],
["-1.749547 -4.575935 0.149987 -1.199874 0.000001 1.557348 0.475128 1.248719 -6.260118 1.278860 -0.041918 -0.080822 0.000000 0.000000 0.000000 1.000000"],
["-0.266642 4.100574 0.285363 -2.553634 -0.000001 2.856037 -0.410403 10.104029 -6.494529 -0.168355 -0.011716 -0.382840 0.000000 0.000000 0.000000 1.000000"],
["-0.266642 4.533805 0.209822 -1.798224 -0.000001 2.099991 -0.453763 10.537625 -6.494529 -0.186142 -0.008614 -0.413855 0.000000 0.000000 0.000000 1.000000"],
["0.139237 1.928305 -0.461196 4.911959 0.000000 4.613018 0.192875 4.071251 6.498508 -0.041316 0.009882 -0.598816 0.000000 0.000000 0.000000 1.000000"],
["0.139238 1.098152 -0.487674 5.176740 0.000001 4.877859 0.109840 4.901595 6.498508 -0.023530 0.010449 -0.604489 0.000000 0.000000 0.000000 1.000000"],
["0.438206 -2.475323 0.433118 -4.031183 0.000001 4.341060 0.248097 3.519032 -6.485212 -0.167257 0.029266 -0.792659 0.000000 0.000000 0.000000 1.000000"],
["0.438206 -1.685616 0.469522 -4.395218 0.000001 4.705925 0.168946 4.310541 -6.485212 -0.113896 0.031726 -0.817256 0.000000 0.000000 0.000000 1.000000"],
["1.669134 -4.250199 -0.229941 2.599409 0.000000 2.379190 -0.439766 10.397665 6.282037 1.129275 0.061095 -1.110952 0.000000 0.000000 0.000000 1.000000"],
["1.669135 -4.584918 -0.152644 1.826437 -0.000001 1.579397 -0.474400 10.743998 6.282037 1.218211 0.040557 -0.905573 0.000000 0.000000 0.000000 1.000000"],
["2.549824 4.413087 -0.129521 1.595207 -0.000001 1.408070 0.479764 1.202361 5.978996 -1.882020 0.055236 -1.052359 0.000000 0.000000 0.000000 1.000000"],
["2.549825 4.121131 -0.204185 2.341855 -0.000001 2.219780 0.448024 1.519757 5.978996 -1.757513 0.087078 -1.370777 0.000000 0.000000 0.000000 1.000000"],
["1.342484 1.849582 0.452909 -4.229086 0.000000 4.628890 -0.189034 7.890338 -6.359854 0.390423 0.095603 -1.456033 0.000000 0.000000 0.000000 1.000000"],
["1.342483 2.607949 0.413910 -3.839103 -0.000001 4.230313 -0.266542 8.665418 -6.359854 0.550504 0.087371 -1.373713 0.000000 0.000000 0.000000 1.000000"],
["1.506390 -0.792885 -0.479881 5.098813 0.000000 4.933118 -0.081508 6.815074 6.323036 0.188896 0.114326 -1.643263 0.000000 0.000000 0.000000 1.000000"],
["1.506391 -1.614144 -0.458823 4.888226 -0.000000 4.716637 -0.165932 7.659318 6.323036 0.384551 0.109309 -1.593093 0.000000 0.000000 0.000000 1.000000"],
["2.947504 -3.827540 0.228238 -1.982376 0.000001 2.560797 0.429445 1.705547 -5.793291 -1.947371 0.116122 -1.661225 0.000000 0.000000 0.000000 1.000000"],
["2.947504 -3.373060 0.291235 -2.612346 0.000000 3.267617 0.378453 2.215467 -5.793291 -1.716142 0.148174 -1.981741 0.000000 0.000000 0.000000 1.000000"],
["5.750979 2.107673 0.099370 -0.693701 0.000000 2.132242 -0.452256 10.522560 -3.029231 4.001407 0.188654 -2.386537 0.000000 0.000000 0.000000 1.000000"],
["5.750979 2.248207 0.061261 -0.312611 -0.000000 1.314516 -0.482411 10.824112 -3.029230 4.268209 0.116304 -1.663039 0.000000 0.000000 0.000000 1.000000"],
["2.754608 2.914008 -0.346680 3.766799 0.000001 3.827493 0.321719 2.782811 5.887456 -1.363399 0.162204 -2.122038 0.000000 0.000000 0.000000 1.000000"],
["2.754610 2.267734 -0.392014 4.220143 0.000000 4.328003 0.250367 3.496325 5.887455 -1.061023 0.183415 -2.334148 0.000000 0.000000 0.000000 1.000000"],
["2.527739 -0.749553 0.454504 -4.245043 0.000000 4.933362 0.081359 5.186406 -5.988367 -0.316392 0.191850 -2.418501 0.000000 0.000000 0.000000 1.000000"],
["2.527740 0.051072 0.460615 -4.306152 0.000001 4.999693 -0.005544 6.055436 -5.988366 0.021559 0.194430 -2.444295 0.000000 0.000000 0.000000 1.000000"],
["3.722163 -2.440898 -0.329304 3.593037 0.000000 4.016846 -0.297741 8.977407 5.328743 1.704984 0.230021 -2.800209 0.000000 0.000000 0.000000 1.000000"],
["3.722165 -2.975644 -0.281915 3.119151 -0.000000 3.438800 -0.362969 9.629691 5.328742 2.078509 0.196920 -2.469197 0.000000 0.000000 0.000000 1.000000"],
["6.499726 -0.042236 0.001792 0.282076 0.000000 1.953322 0.460267 1.397334 -0.059647 -4.602472 0.195324 -2.453240 0.000000 0.000000 0.000000 1.000000"],
["6.499726 -0.038482 0.002499 0.275014 0.000000 2.722891 0.419355 1.806449 -0.059647 -4.193374 0.272278 -3.222776 0.000000 0.000000 0.000000 1.000000"],
["4.302104 2.126667 0.308638 -2.786381 0.000000 4.117231 -0.283697 8.836971 -4.872566 1.877685 0.272504 -3.225041 0.000000 0.000000 0.000000 1.000000"],
["4.302106 2.630301 0.267020 -2.370201 0.000001 3.562047 -0.350882 9.508821 -4.872564 2.322357 0.235758 -2.857585 0.000000 0.000000 0.000000 1.000000"],
["3.592301 0.681786 -0.411088 4.410876 -0.000000 4.932622 0.081807 5.181928 5.417137 -0.452117 0.272607 -3.226071 0.000000 0.000000 0.000000 1.000000"],
["3.592301 -0.042418 -0.416681 4.466813 -0.000001 4.999741 -0.005090 6.050897 5.417137 0.028130 0.276317 -3.263165 0.000000 0.000000 0.000000 1.000000"],
["4.405419 -2.064547 0.304200 -2.742004 -0.000000 4.137170 0.280781 3.192185 -4.779360 -1.903015 0.280400 -3.303995 0.000000 0.000000 0.000000 1.000000"],
["4.405418 -1.504943 0.335429 -3.054294 -0.000000 4.561890 0.204674 3.953254 -4.779361 -1.387195 0.309185 -3.591851 0.000000 0.000000 0.000000 1.000000"],
["6.349392 -0.751830 -0.076148 1.061480 0.000000 3.558008 -0.351292 9.512916 1.391121 3.431520 0.347557 -3.975567 0.000000 0.000000 0.000000 1.000000"],
["6.349392 -0.872637 -0.061936 0.919358 -0.000000 2.893942 -0.407739 10.077388 1.391121 3.982914 0.282689 -3.326889 0.000000 0.000000 0.000000 1.000000"],
["5.611467 1.744335 -0.182346 2.123461 0.000000 3.613056 0.345627 2.543726 3.280463 -2.983810 0.311916 -3.619161 0.000000 0.000000 0.000000 1.000000"],
["5.611467 1.401194 -0.209866 2.398659 0.000000 4.158341 0.277636 3.223635 3.280463 -2.396843 0.358991 -4.089907 0.000000 0.000000 0.000000 1.000000"],
["4.724389 0.338686 0.341735 -3.117350 -0.000000 4.975624 -0.049312 6.493123 -4.464319 0.358416 0.361643 -4.116428 0.000000 0.000000 0.000000 1.000000"],
["4.724389 0.926957 0.330662 -3.006620 0.000001 4.814403 -0.134964 7.349639 -4.464319 0.980958 0.349925 -3.999249 0.000000 0.000000 0.000000 1.000000"],
["5.144022 -0.657265 -0.298507 3.285068 -0.000000 4.883033 -0.107517 7.075168 3.973542 0.850875 0.386437 -4.364375 0.000000 0.000000 0.000000 1.000000"],
["5.144022 -1.165632 -0.282558 3.125585 -0.000000 4.622148 -0.190676 7.906763 3.973542 1.508990 0.365791 -4.157913 0.000000 0.000000 0.000000 1.000000"],
["6.284044 -0.814475 0.098501 -0.685005 0.000000 3.853321 0.318621 2.813792 -1.661563 -3.080349 0.372530 -4.225298 0.000000 0.000000 0.000000 1.000000"],
["6.284044 -0.631057 0.111147 -0.811473 0.000000 4.348059 0.246868 3.531320 -1.661563 -2.386661 0.420360 -4.703599 0.000000 0.000000 0.000000 1.000000"],
["6.310709 0.515261 0.108138 -0.781382 -0.000000 4.513789 -0.215075 8.150745 -1.557224 2.088112 0.438234 -4.882340 0.000000 0.000000 0.000000 1.000000"],
["6.310709 0.695212 0.097548 -0.675479 0.000000 4.071741 -0.290188 8.901881 -1.557224 2.817376 0.395317 -4.453166 0.000000 0.000000 0.000000 1.000000"],
["5.828282 0.641638 -0.211858 2.418579 0.000000 4.785345 0.144930 4.550698 2.877695 -1.299529 0.429082 -4.790822 0.000000 0.000000 0.000000 1.000000"],
["5.828282 0.264003 -0.219781 2.497812 -0.000000 4.964313 0.059632 5.403683 2.877695 -0.534693 0.445130 -4.951295 0.000000 0.000000 0.000000 1.000000"],
["6.010379 -0.385919 0.186428 -1.564277 -0.000000 4.896194 0.101355 4.986453 -2.474943 -0.937200 0.452738 -5.027383 0.000000 0.000000 0.000000 1.000000"],
["6.010379 -0.056327 0.190297 -1.602969 0.000000 4.997811 0.014793 5.852066 -2.474943 -0.136790 0.462134 -5.121345 0.000000 0.000000 0.000000 1.000000"],
["6.428964 -0.116861 -0.072786 1.027864 -0.000000 4.936777 -0.079262 6.792614 0.958341 0.783953 0.488282 -5.382825 0.000000 0.000000 0.000000 1.000000"],
["6.428964 -0.241478 -0.069651 0.996513 0.000000 4.724140 -0.163783 7.637834 0.958341 1.619935 0.467251 -5.172512 0.000000 0.000000 0.000000 1.000000"],
["6.500000 0.000000 0.000000 0.300000 0.000000 4.980974 0.043578 5.564221 0.000000 -0.435778 0.498097 -5.480974 0.000000 0.000000 0.000000 1.000000"],
["6.500000 0.000000 0.000000 0.300000 0.000000 4.980974 -0.043578 6.435779 0.000000 0.435779 0.498097 -5.480974 0.000000 0.000000 0.000000 1.000000"],
]
  # print("before(no T):\n", np.array(params['arealight.to_world']))
  idx = int(sensor.id()[len('elm__') :])
  raw_string = light_positions[int((idx-1)/2)-1][0]
  float_values = list(map(float, raw_string.strip().split()))
  matrix_4x4 = np.array(float_values).reshape((4, 4))
  params['arealight.to_world'] = mi.Transform4f(matrix_4x4)
  params.update()
  # print("after(no T):\n", np.array(params['arealight.to_world']))

def tonemap(x, gamma=2.2, exposure=1.0):
    x = np.clip(x * exposure, 0, 1)   # clamp
    return np.power(x, 1/gamma)       # gamma correction

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
  assert scene_config.scene_name in ['kiwi_refine']
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
  backlit_indices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]
  # backlit_indices = [1,3]
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

        rendering_loss = scene_config.loss(img, ref_img, weight=mask)
        # for k in params.keys():
        #   print(k, params[k])
          
        dr.backward(rendering_loss)
        # print(rendering_loss)
        # img_np = np.array(img)
        # ref_np = np.array(ref_img)
        # img_disp = tonemap(img_np)
        # ref_disp = tonemap(ref_np)

        # plt.figure(figsize=(12,5))
        # plt.subplot(1,2,1)
        # plt.title("Rendered img")
        # plt.imshow(img_disp)
        # plt.axis("off")

        # plt.subplot(1,2,2)
        # plt.title("Reference ref_img")
        # plt.imshow(ref_disp)
        # plt.axis("off")
        # plt.show()
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
