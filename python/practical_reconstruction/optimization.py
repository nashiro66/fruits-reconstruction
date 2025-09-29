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
["4.554923 0.152407 -0.205656 10.582809 0.000000 4.986326 0.036953 4.152369 2.062201 -0.336632 0.454247 -23.212334 0.000000 0.000000 0.000000 1.000000"],
["-4.554923 0.846602 0.188041 -9.102048 0.000000 4.559228 -0.205267 16.263325 -2.062202 -1.869947 -0.415339 20.266935 0.000000 0.000000 0.000000 1.000000"],
["3.982244 1.183225 0.278239 -13.611968 0.000000 4.601233 -0.195670 15.783475 -3.023530 1.558407 0.366465 -18.823231 0.000000 0.000000 0.000000 1.000000"],
["-3.982244 0.160232 -0.301928 15.396410 -0.000000 4.992974 0.026498 4.675113 3.023530 0.211040 -0.397665 19.383238 0.000000 0.000000 0.000000 1.000000"],
["4.137138 -1.925474 -0.204368 10.518407 -0.000000 3.639213 -0.342872 23.143600 2.807861 2.837019 0.301119 -15.555934 0.000000 0.000000 0.000000 1.000000"],
["-4.137139 -1.110372 0.257898 -12.594914 0.000000 4.592435 0.197726 -3.886291 -2.807860 1.636037 -0.379991 18.499538 0.000000 0.000000 0.000000 1.000000"],
["4.354821 2.076192 0.131338 -6.266884 -0.000000 2.673016 -0.422552 27.127579 -2.456732 3.680273 0.232810 -12.140509 0.000000 0.000000 0.000000 1.000000"],
["-4.354821 1.501780 -0.194427 10.021350 -0.000001 3.957026 0.305646 -9.282302 2.456732 2.662068 -0.344643 16.732138 0.000000 0.000000 0.000000 1.000000"],
["3.334635 -0.561823 -0.368301 18.715048 -0.000000 4.942822 -0.075400 9.770000 3.725615 0.502863 0.329650 -16.982510 0.000000 0.000000 0.000000 1.000000"],
["-3.334636 0.731723 0.365305 -17.965258 -0.000000 4.902616 -0.098202 10.910082 -3.725615 -0.654933 -0.326969 15.848438 0.000000 0.000000 0.000000 1.000000"],
["3.318174 -0.080402 0.373942 -18.397095 -0.000000 4.998845 0.010748 5.462587 -3.740283 -0.071329 0.331741 -17.087040 0.000000 0.000000 0.000000 1.000000"],
["-3.318176 -1.354511 -0.348640 17.732023 0.000000 4.660616 -0.181071 15.053531 3.740282 -1.201649 -0.309295 14.964740 0.000000 0.000000 0.000000 1.000000"],
["2.751413 2.757752 0.313441 -15.372048 0.000000 3.753881 -0.330278 22.513916 -4.174892 1.817464 0.206570 -10.828478 0.000000 0.000000 0.000000 1.000000"],
["-2.751414 1.519408 -0.388859 19.742943 -0.000000 4.657113 0.181970 -3.098490 4.174892 1.001348 -0.256273 12.313642 0.000000 0.000000 0.000000 1.000000"],
["2.473057 -2.507945 -0.354883 18.044151 0.000000 4.083273 -0.288563 20.428169 4.345572 1.427267 0.201963 -10.598166 0.000000 0.000000 0.000000 1.000000"],
["-2.473057 -1.142926 0.419258 -20.662889 0.000000 4.823967 0.131505 -0.575236 -4.345572 0.650437 -0.238599 11.429943 0.000000 0.000000 0.000000 1.000000"],
["3.503841 -1.832380 0.306031 -15.001532 0.000000 4.289815 0.256856 -6.842788 -3.566945 -1.799963 0.300617 -15.530832 0.000000 0.000000 0.000000 1.000000"],
["-3.503842 -2.768561 -0.224904 11.545181 -0.000000 3.152609 -0.388086 25.404289 3.566944 -2.719583 -0.220925 10.546243 0.000000 0.000000 0.000000 1.000000"],
["2.194997 0.955449 -0.438966 22.248293 -0.000000 4.885610 0.106340 0.683013 4.492436 -0.466831 0.214478 -11.223903 0.000000 0.000000 0.000000 1.000000"],
["-2.194998 2.399179 0.379815 -18.690737 -0.000000 4.227269 -0.267024 19.351219 -4.492436 -1.172236 -0.185577 8.778844 0.000000 0.000000 0.000000 1.000000"],
["1.691467 1.062233 0.458373 -22.618662 0.000000 4.870918 -0.112879 11.643924 -4.705204 0.381861 0.164780 -8.738997 0.000000 0.000000 0.000000 1.000000"],
["-1.691467 -0.569557 -0.467060 23.653021 0.000001 4.963233 -0.060524 9.026206 4.705204 -0.204750 -0.167903 7.895144 0.000000 0.000000 0.000000 1.000000"],
["2.195768 -4.129715 -0.176750 9.137499 0.000000 1.967359 -0.459668 28.983416 4.492060 2.018650 0.086397 -4.819869 0.000000 0.000000 0.000000 1.000000"],
["-2.195770 -3.276142 0.307335 -15.066760 -0.000000 3.420872 0.364659 -12.232962 -4.492059 1.601415 -0.150229 7.011445 0.000000 0.000000 0.000000 1.000000"],
["1.586890 4.247731 0.210679 -10.233948 -0.000000 2.221651 -0.447932 28.396576 -4.741496 1.421636 0.070510 -4.025518 0.000000 0.000000 0.000000 1.000000"],
["-1.586891 3.270996 -0.343254 17.462721 0.000000 3.619686 0.344933 -11.246651 4.741495 1.094742 -0.114881 5.244046 0.000000 0.000000 0.000000 1.000000"],
["0.973068 -1.466863 -0.467990 23.699492 0.000000 4.771122 -0.149546 13.477283 4.904400 0.291036 0.092853 -5.142629 0.000000 0.000000 0.000000 1.000000"],
["-0.973069 0.222220 0.489936 -24.196817 -0.000001 4.994865 -0.022655 7.132752 -4.904400 -0.044091 -0.097207 4.360346 0.000000 0.000000 0.000000 1.000000"],
["1.026869 -1.612397 0.462014 -22.800713 0.000000 4.720772 0.164752 -2.237581 -4.893418 -0.338357 0.096952 -5.347619 0.000000 0.000000 0.000000 1.000000"],
["-1.026870 -3.095339 -0.379004 19.250206 0.000000 3.872591 -0.316276 21.813789 4.893418 -0.649549 -0.079533 3.476647 0.000000 0.000000 0.000000 1.000000"],
["0.767473 2.823286 -0.405463 20.573160 0.000000 4.103258 0.285714 -8.285723 4.940747 -0.438556 0.062983 -3.649141 0.000000 0.000000 0.000000 1.000000"],
["-0.767474 4.039787 0.284449 -13.922436 -0.000000 2.878599 -0.408823 26.441174 -4.940747 -0.627523 -0.044185 1.709249 0.000000 0.000000 0.000000 1.000000"],
["0.271717 2.694309 0.420320 -20.715988 0.000000 4.209418 -0.269830 19.491480 -4.992611 0.146635 0.022875 -1.643767 0.000000 0.000000 0.000000 1.000000"],
["-0.271717 1.094244 -0.487122 24.656107 0.000000 4.878430 0.109586 0.520683 4.992611 0.059553 -0.026511 0.825549 0.000000 0.000000 0.000000 1.000000"],
["0.101845 -3.818668 -0.322605 16.430250 0.000000 3.226719 -0.381946 25.097305 4.998962 0.077799 0.006573 -0.828625 0.000000 0.000000 0.000000 1.000000"],
["-0.101846 -2.485000 0.433756 -21.387785 0.000001 4.338457 0.248552 -6.427580 -4.998962 0.050628 -0.008837 -0.058150 0.000000 0.000000 0.000000 1.000000"],
["-0.478657 0.227147 -0.497185 25.159248 -0.000000 4.994790 0.022820 4.859016 4.977036 0.021846 -0.047816 1.890789 0.000000 0.000000 0.000000 1.000000"],
["0.478656 1.913923 0.459432 -22.671608 -0.000000 4.615520 -0.192275 15.613768 -4.977036 0.184067 0.044185 -2.709249 0.000000 0.000000 0.000000 1.000000"],
["-0.656736 0.032497 0.495658 -24.482880 -0.000001 4.999892 -0.003278 6.163902 -4.956682 -0.004306 -0.065672 2.783611 0.000000 0.000000 0.000000 1.000000"],
["0.656736 -1.664713 -0.466877 23.643858 -0.000000 4.709573 -0.167926 14.396303 4.956682 0.220566 0.061859 -3.592948 0.000000 0.000000 0.000000 1.000000"],
["-1.351134 3.781045 0.297962 -14.598101 0.000000 3.094755 -0.392715 25.635742 -4.813983 -1.061220 -0.083629 3.681424 0.000000 0.000000 0.000000 1.000000"],
["1.351134 2.533931 -0.409312 20.765602 -0.000000 4.251282 0.263184 -7.159217 4.813983 -0.711195 0.114881 -6.244046 0.000000 0.000000 0.000000 1.000000"],
["-1.343589 -2.364195 -0.419587 21.279366 -0.000000 4.356095 -0.245447 18.272366 4.816095 -0.659560 -0.117056 5.352798 0.000000 0.000000 0.000000 1.000000"],
["1.343587 -0.786542 0.475143 -23.457169 -0.000000 4.932870 0.081658 1.917110 -4.816095 -0.219429 0.132555 -7.127746 0.000000 0.000000 0.000000 1.000000"],
["-2.287396 -2.151856 0.389067 -19.153366 0.000001 4.375375 0.241994 -6.099680 -4.446102 1.107071 -0.200164 9.508217 0.000000 0.000000 0.000000 1.000000"],
["2.287395 -3.352772 -0.292006 14.900292 0.000000 3.283841 -0.377046 24.852312 4.446102 1.724908 0.150229 -8.011445 0.000000 0.000000 0.000000 1.000000"],
["-2.391057 1.436341 -0.414967 21.048359 -0.000000 4.724960 0.163547 -2.177340 4.391223 0.782100 -0.225953 10.797644 0.000000 0.000000 0.000000 1.000000"],
["2.391056 2.768991 0.340816 -16.740793 -0.000000 3.880646 -0.315287 21.764343 -4.391224 1.507737 0.185577 -9.778844 0.000000 0.000000 0.000000 1.000000"],
["-2.032528 1.581548 0.428574 -21.128681 -0.000000 4.690794 -0.173103 14.655125 -4.568241 -0.703672 -0.190683 9.034175 0.000000 0.000000 0.000000 1.000000"],
["2.032528 0.020361 -0.456820 23.140980 -0.000000 4.999950 0.002228 5.888574 4.568242 -0.009058 0.203251 -10.662543 0.000000 0.000000 0.000000 1.000000"],
["-2.768702 -3.494389 -0.226352 11.617624 -0.000001 2.718329 -0.419651 26.982542 4.163447 -2.323776 -0.150525 7.026241 0.000000 0.000000 0.000000 1.000000"],
["2.768700 -2.509480 0.332217 -16.310846 0.000000 3.989685 0.301370 -9.068522 -4.163448 -1.668808 0.220925 -11.546243 0.000000 0.000000 0.000000 1.000000"],
["-3.512960 3.174526 0.160670 -7.733514 0.000001 2.257896 -0.446115 28.305777 -3.557964 -3.134372 -0.158638 7.431901 0.000000 0.000000 0.000000 1.000000"],
["3.512961 2.433554 -0.259556 13.277793 -0.000000 3.647534 0.341987 -11.099342 3.557964 -2.402772 0.256273 -13.313642 0.000000 0.000000 0.000000 1.000000"],
["-2.770702 -0.821877 -0.408016 20.700815 -0.000000 4.901548 -0.098733 10.936649 4.162116 -0.547119 -0.271615 13.080730 0.000000 0.000000 0.000000 1.000000"],
["2.770702 0.623186 0.411520 -20.275990 -0.000001 4.943636 -0.074864 9.743209 -4.162116 0.414851 0.273947 -14.197341 0.000000 0.000000 0.000000 1.000000"],
["-3.234591 -0.374375 0.379437 -18.671864 0.000000 4.975839 0.049095 3.545277 -3.812797 0.317601 -0.321896 15.594807 0.000000 0.000000 0.000000 1.000000"],
["3.234591 -1.649549 -0.343750 17.487501 -0.000000 4.507846 -0.216317 16.815870 3.812797 1.399397 0.291621 -15.081041 0.000000 0.000000 0.000000 1.000000"],
["-3.561214 1.992272 0.288940 -14.146979 -0.000000 4.116340 -0.283826 20.191320 -3.509666 -2.021534 -0.293183 14.159169 0.000000 0.000000 0.000000 1.000000"],
["3.561214 0.883892 -0.339654 17.282705 0.000000 4.838838 0.125923 -0.296121 3.509665 -0.896875 0.344643 -17.732138 0.000000 0.000000 0.000000 1.000000"],
["-3.752037 -1.898442 -0.270521 13.826030 -0.000001 4.092747 -0.287218 20.360905 3.304878 -2.155306 -0.307123 14.856137 0.000000 0.000000 0.000000 1.000000"],
["3.752037 -0.858717 0.319137 -15.656837 0.000000 4.828268 0.129917 -0.495827 -3.304878 -0.974903 0.362317 -18.615837 0.000000 0.000000 0.000000 1.000000"],
["-4.328433 -0.159333 -0.249785 12.789277 -0.000000 4.989859 -0.031829 7.591456 2.502932 -0.275540 -0.431965 21.098270 0.000000 0.000000 0.000000 1.000000"],
["4.328433 0.704594 0.240171 -11.708556 0.000000 4.797795 -0.140754 13.037682 -2.502933 1.218486 0.415339 -21.266935 0.000000 0.000000 0.000000 1.000000"],
["-4.354185 0.593537 0.238512 -11.625589 0.000000 4.852023 -0.120743 12.037126 -2.457859 -1.051471 -0.422532 20.626606 0.000000 0.000000 0.000000 1.000000"],
["4.354185 -0.258017 -0.244428 12.521396 0.000000 4.972373 -0.052488 8.624405 2.457860 0.457086 0.433013 -22.150637 0.000000 0.000000 0.000000 1.000000"],
    ]
    raw_string = light_positions[idx-1][0]
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
  backlit_indices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65]
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
