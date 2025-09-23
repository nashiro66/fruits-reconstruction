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
    ["7.500000 0.000000 0.000000 0.300000 0.000000 7.471460 0.043578 2.949549 0.000000 -0.653668 0.498097 -35.366814 0.000000 0.000000 0.000000 1.000000"],
    ["-7.500000 0.000001 -0.000000 0.300000 0.000001 7.471460 -0.043578 9.050451 -0.000000 -0.653669 -0.498097 34.366814 0.000000 0.000000 0.000000 1.000000"],
    ["7.328826 0.166872 0.105630 -7.094067 -0.000000 7.458747 -0.052370 9.665897 -1.593206 0.767620 0.485901 -34.513069 0.000000 0.000000 0.000000 1.000000"],
    ["-7.328826 0.166873 -0.105630 7.694067 0.000001 7.458748 0.052370 2.334103 1.593207 0.767619 -0.485901 33.513069 0.000000 0.000000 0.000000 1.000000"],
    ["7.494684 0.133494 -0.016586 1.461014 0.000000 6.608728 0.236406 -10.548397 0.282341 -3.543571 0.440270 -31.318865 0.000000 0.000000 0.000000 1.000000"],
    ["-7.494684 0.133495 0.016586 -0.861014 0.000001 6.608727 -0.236406 22.548397 -0.282341 -3.543573 -0.440269 30.318865 0.000000 0.000000 0.000000 1.000000"],
    ["7.117022 -0.706185 -0.150544 10.838095 -0.000000 7.158140 -0.149236 16.446486 2.366009 2.124227 0.452841 -32.198891 0.000000 0.000000 0.000000 1.000000"],
    ["-7.117022 -0.706185 0.150544 -10.238094 -0.000000 7.158140 0.149235 -4.446486 -2.366009 2.124224 -0.452841 31.198891 0.000000 0.000000 0.000000 1.000000"],
    ["6.304254 -0.738532 0.266341 -18.343878 0.000000 7.375046 0.090889 -0.362262 -4.062806 -1.145979 0.413281 -29.429699 0.000000 0.000000 0.000000 1.000000"],
    ["-6.304254 -0.738533 -0.266341 18.943876 -0.000000 7.375046 -0.090890 12.362262 4.062806 -1.145981 -0.413281 28.429699 0.000000 0.000000 0.000000 1.000000"],
    ["6.311899 1.645186 -0.246786 17.575043 0.000000 6.853625 0.203063 -8.214446 4.050918 -2.563432 0.384528 -27.416950 0.000000 0.000000 0.000000 1.000000"],
    ["-6.311899 1.645188 0.246786 -16.975044 -0.000000 6.853623 -0.203064 20.214447 -4.050917 -2.563436 -0.384528 26.416950 0.000000 0.000000 0.000000 1.000000"],
    ["7.316449 0.928876 0.090842 -6.058943 -0.000000 6.197108 -0.281629 25.714031 -1.649113 4.121049 0.403030 -28.712067 0.000000 0.000000 0.000000 1.000000"],
    ["-7.316449 0.928876 -0.090842 6.658943 0.000000 6.197108 0.281629 -13.714031 1.649113 4.121048 -0.403030 27.712067 0.000000 0.000000 0.000000 1.000000"],
    ["6.835326 -2.120719 0.149531 -10.167147 0.000000 5.449726 0.343514 -18.046011 -3.086798 -4.696066 0.331117 -23.678185 0.000000 0.000000 0.000000 1.000000"],
    ["-6.835326 -2.120720 -0.149531 10.767147 -0.000001 5.449725 -0.343514 30.046011 3.086798 -4.696066 -0.331117 22.678185 0.000000 0.000000 0.000000 1.000000"],
    ["5.221693 -0.906111 -0.353791 25.065395 -0.000000 7.393009 -0.084154 11.890759 5.383672 0.878849 0.343147 -24.520285 0.000000 0.000000 0.000000 1.000000"],
    ["-5.221694 -0.906112 0.353791 -24.465397 -0.000001 7.393009 0.084154 0.109242 -5.383671 0.878849 -0.343147 23.520285 0.000000 0.000000 0.000000 1.000000"],
    ["4.966681 1.186354 0.366210 -25.334677 0.000000 7.330979 -0.105551 13.388588 -5.619793 1.048480 0.323650 -23.155512 0.000000 0.000000 0.000000 1.000000"],
    ["-4.966681 1.186353 -0.366210 25.934675 0.000001 7.330980 0.105551 -1.388588 5.619793 1.048478 -0.323650 22.155512 0.000000 0.000000 0.000000 1.000000"],
    ["6.495440 2.950367 -0.154267 11.098653 -0.000000 4.628530 0.393428 -21.539946 3.749567 -5.110973 0.267239 -19.206701 0.000000 0.000000 0.000000 1.000000"],
    ["-6.495441 2.950367 0.154266 -10.498652 0.000001 4.628530 -0.393428 33.539948 -3.749566 -5.110973 -0.267239 18.206701 0.000000 0.000000 0.000000 1.000000"],
    ["6.831079 -2.292678 -0.138724 10.010694 0.000000 5.040548 -0.370242 31.916969 3.096185 5.058310 0.306066 -21.924593 0.000000 0.000000 0.000000 1.000000"],
    ["-6.831079 -2.292679 0.138724 -9.410693 -0.000000 5.040547 0.370242 -19.916969 -3.096186 5.058311 -0.306066 20.924593 0.000000 0.000000 0.000000 1.000000"],
    ["4.241152 -3.143757 0.355149 -24.560425 -0.000000 6.459154 0.254116 -11.788103 -6.185679 -2.155488 0.243504 -17.545313 0.000000 0.000000 0.000000 1.000000"],
    ["-4.241152 -3.143759 -0.355149 25.160423 -0.000000 6.459153 -0.254116 23.788103 6.185679 -2.155488 -0.243504 16.545313 0.000000 0.000000 0.000000 1.000000"],
    ["3.585440 1.807922 -0.422301 29.861053 0.000000 7.212013 0.137224 -3.605716 6.587459 -0.984020 0.229851 -16.589569 0.000000 0.000000 0.000000 1.000000"],
    ["-3.585441 1.807922 0.422301 -29.261055 -0.000001 7.212012 -0.137225 15.605716 -6.587459 -0.984022 -0.229851 15.589569 0.000000 0.000000 0.000000 1.000000"],
    ["4.772033 3.920635 0.283677 -19.557381 -0.000000 5.515674 -0.338804 29.716280 -5.785993 3.233568 0.233964 -16.877499 0.000000 0.000000 0.000000 1.000000"],
    ["-4.772034 3.920633 -0.283677 20.157379 0.000000 5.515676 0.338804 -17.716280 5.785991 3.233568 -0.233964 15.877499 0.000000 0.000000 0.000000 1.000000"],
    ["7.172763 -2.072851 0.047364 -3.015504 -0.000000 2.431738 0.472989 -27.109219 -2.191226 -6.785274 0.155042 -11.352975 0.000000 0.000000 0.000000 1.000000"],
    ["-7.172763 -2.072851 -0.047364 3.615504 0.000001 2.431739 -0.472989 39.109219 2.191226 -6.785274 -0.155043 10.352975 0.000000 0.000000 0.000000 1.000000"],
    ["3.266159 -3.596402 -0.380923 26.964638 -0.000000 6.347352 -0.266343 24.643972 6.751460 1.739834 0.184280 -13.399581 0.000000 0.000000 0.000000 1.000000"],
    ["-3.266160 -3.596401 0.380923 -26.364639 -0.000001 6.347353 0.266342 -12.643972 -6.751459 1.739834 -0.184280 12.399581 0.000000 0.000000 0.000000 1.000000"],
    ["2.297698 -0.342003 0.475411 -32.978809 -0.000001 7.491390 0.023952 4.323370 -7.139369 -0.110069 0.153004 -11.210276 0.000000 0.000000 0.000000 1.000000"],
    ["-2.297697 -0.342002 -0.475412 33.578808 -0.000001 7.491390 -0.023952 7.676630 7.139369 -0.110067 -0.153004 10.210276 0.000000 0.000000 0.000000 1.000000"],
    ["2.714157 5.183684 -0.312785 22.194952 0.000001 5.032897 0.370705 -19.949331 6.991663 -2.012302 0.121423 -8.999601 0.000000 0.000000 0.000000 1.000000"],
    ["-2.714157 5.183686 0.312785 -21.594954 -0.000001 5.032896 -0.370705 31.949331 -6.991663 -2.012303 -0.121423 7.999601 0.000000 0.000000 0.000000 1.000000"],
    ["7.122634 2.323870 0.022872 -1.301074 0.000000 1.095397 -0.494638 40.624687 -2.349059 7.046257 0.069352 -5.354651 0.000000 0.000000 0.000000 1.000000"],
    ["-7.122635 2.323869 -0.022872 1.901073 -0.000000 1.095398 0.494638 -28.624687 2.349058 7.046257 -0.069352 4.354651 0.000000 0.000000 0.000000 1.000000"],
    ["2.099781 -5.825052 0.282132 -19.449261 0.000000 4.408278 0.404514 -22.315975 -7.200064 -1.698781 0.082279 -6.259549 0.000000 0.000000 0.000000 1.000000"],
    ["-2.099783 -5.825052 -0.282132 20.049259 0.000000 4.408278 -0.404514 34.315975 7.200063 -1.698783 -0.082279 5.259549 0.000000 0.000000 0.000000 1.000000"],
    ["1.080882 -0.334599 -0.494277 34.899403 -0.000000 7.492374 -0.022542 7.577931 7.421704 0.048731 0.071986 -5.538995 0.000000 0.000000 0.000000 1.000000"],
    ["-1.080884 -0.334599 0.494277 -34.299404 -0.000000 7.492374 0.022542 4.422069 -7.421704 0.048730 -0.071986 4.538995 0.000000 0.000000 0.000000 1.000000"],
    ["0.929983 3.678388 0.431301 -29.891064 -0.000001 6.519829 -0.247133 23.299314 -7.442119 0.459659 0.053896 -4.272741 0.000000 0.000000 0.000000 1.000000"],
    ["-0.929983 3.678386 -0.431301 30.491062 0.000000 6.519831 0.247133 -11.299314 7.442119 0.459659 -0.053896 3.272741 0.000000 0.000000 0.000000 1.000000"],
    ["-0.882609 7.263825 -0.109704 7.979250 0.000000 1.657068 0.487643 -28.135040 7.447886 0.860797 -0.013000 0.410026 0.000000 0.000000 0.000000 1.000000"],
    ["0.882609 7.263825 0.109704 -7.379250 0.000000 1.657068 -0.487643 40.135040 -7.447886 0.860797 0.013000 -1.410026 0.000000 0.000000 0.000000 1.000000"],
    ["0.307662 -6.150860 -0.285363 20.275440 -0.000000 4.284057 -0.410403 34.728199 7.493687 0.252531 0.011716 -1.320118 0.000000 0.000000 0.000000 1.000000"],
    ["-0.307664 -6.150860 0.285363 -19.675442 0.000000 4.284058 0.410403 -22.728199 -7.493687 0.252533 -0.011716 0.320118 0.000000 0.000000 0.000000 1.000000"],
    ["-0.160660 -2.892458 0.461196 -31.983715 0.000001 6.919527 0.192875 -7.501244 -7.498279 0.061975 -0.009882 0.191711 0.000000 0.000000 0.000000 1.000000"],
    ["0.160658 -2.892459 -0.461196 32.583714 -0.000000 6.919526 -0.192875 19.501244 7.498279 0.061974 0.009882 -1.191711 0.000000 0.000000 0.000000 1.000000"],
    ["-0.505623 3.712984 -0.433118 30.618282 -0.000000 6.511590 0.248097 -11.366772 7.482937 0.250887 -0.029266 1.548610 0.000000 0.000000 0.000000 1.000000"],
    ["0.505624 3.712986 0.433118 -30.018284 -0.000001 6.511590 -0.248097 23.366772 -7.482937 0.250886 0.029266 -2.548610 0.000000 0.000000 0.000000 1.000000"],
    ["-1.925925 6.375299 0.229941 -15.795859 0.000001 3.568785 -0.439766 36.783653 -7.248504 -1.693914 -0.061095 3.776662 0.000000 0.000000 0.000000 1.000000"],
    ["1.925925 6.375299 -0.229941 16.395859 -0.000000 3.568784 0.439766 -24.783653 7.248505 -1.693914 0.061095 -4.776662 0.000000 0.000000 0.000000 1.000000"],
    ["-2.942106 -6.619630 0.129521 -8.766449 0.000000 2.112106 0.479764 -27.583469 -6.898841 2.823032 -0.055236 3.366510 0.000000 0.000000 0.000000 1.000000"],
    ["2.942104 -6.619631 -0.129521 9.366449 -0.000001 2.112104 -0.479764 39.583469 6.898842 2.823031 0.055236 -4.366510 0.000000 0.000000 0.000000 1.000000"],
    ["-1.549021 -2.774371 -0.452909 32.003601 0.000000 6.943335 -0.189034 19.232370 7.338292 -0.585635 -0.095603 6.192231 0.000000 0.000000 0.000000 1.000000"],
    ["1.549020 -2.774371 0.452909 -31.403601 0.000001 6.943335 0.189034 -7.232370 -7.338293 -0.585633 0.095603 -7.192231 0.000000 0.000000 0.000000 1.000000"],
    ["-1.738145 1.189327 0.479881 -33.291691 -0.000000 7.399677 -0.081508 11.705523 -7.295811 -0.283344 -0.114326 7.502840 0.000000 0.000000 0.000000 1.000000"],
    ["1.738144 1.189326 -0.479881 33.891689 -0.000000 7.399677 0.081507 0.294478 7.295811 -0.283343 0.114326 -8.502840 0.000000 0.000000 0.000000 1.000000"],
    ["-3.400968 5.741309 -0.228238 16.276623 -0.000001 3.841195 0.429445 -24.061165 6.684566 2.921060 -0.116122 7.628573 0.000000 0.000000 0.000000 1.000000"],
    ["3.400966 5.741311 0.228238 -15.676623 -0.000001 3.841195 -0.429445 36.061165 -6.684566 2.921057 0.116122 -8.628573 0.000000 0.000000 0.000000 1.000000"],
    ["-6.635746 -3.161510 -0.099370 7.255907 -0.000001 3.198366 -0.452256 37.657909 3.495266 -6.002109 -0.188654 12.705755 0.000000 0.000000 0.000000 1.000000"],
    ["6.635745 -3.161510 0.099370 -6.655906 0.000000 3.198365 0.452256 -25.657909 -3.495267 -6.002109 0.188654 -13.705755 0.000000 0.000000 0.000000 1.000000"],
    ["-3.178397 -4.371010 0.346680 -23.967592 0.000001 5.741240 0.321719 -16.520317 -6.793217 2.045101 -0.162204 10.854267 0.000000 0.000000 0.000000 1.000000"],
    ["3.178394 -4.371013 -0.346680 24.567591 -0.000000 5.741238 -0.321719 28.520317 6.793218 2.045099 0.162204 -11.854267 0.000000 0.000000 0.000000 1.000000"],
    ["-2.916623 1.124330 -0.454504 32.115303 0.000001 7.400044 0.081359 0.304843 6.909653 0.474588 -0.191850 12.929502 0.000000 0.000000 0.000000 1.000000"],
    ["2.916621 1.124330 0.454504 -31.515305 -0.000000 7.400044 -0.081359 11.695156 -6.909654 0.474589 0.191850 -13.929502 0.000000 0.000000 0.000000 1.000000"],
    ["-4.294805 3.661346 0.329304 -22.751261 0.000002 6.025269 -0.297741 26.841841 -6.148549 -2.557474 -0.230021 15.601463 0.000000 0.000000 0.000000 1.000000"],
    ["4.294803 3.661345 -0.329304 23.351259 -0.000000 6.025270 0.297741 -14.841841 6.148550 -2.557474 0.230021 -16.601463 0.000000 0.000000 0.000000 1.000000"],
    ["-7.499684 0.063355 -0.001792 0.425472 0.000000 2.929983 0.460267 -26.218662 0.068824 6.903708 -0.195324 13.172678 0.000000 0.000000 0.000000 1.000000"],
    ["7.499684 0.063354 0.001792 0.174528 -0.000000 2.929983 -0.460267 38.218662 -0.068823 6.903708 0.195324 -14.172678 0.000000 0.000000 0.000000 1.000000"],
    ["-4.963970 -3.189996 -0.308638 21.904667 -0.000001 6.175848 -0.283697 25.858795 5.622188 -2.816527 -0.272504 18.575289 0.000000 0.000000 0.000000 1.000000"],
    ["4.963968 -3.189997 0.308638 -21.304668 -0.000000 6.175848 0.283697 -13.858795 -5.622190 -2.816526 0.272504 -19.575289 0.000000 0.000000 0.000000 1.000000"],
    ["-4.144962 -1.022679 0.411088 -28.476128 -0.000000 7.398932 0.081807 0.273499 -6.250543 0.678175 -0.272607 18.582497 0.000000 0.000000 0.000000 1.000000"],
    ["4.144962 -1.022679 -0.411088 29.076126 -0.000000 7.398932 -0.081807 11.726501 6.250544 0.678176 0.272607 -19.582497 0.000000 0.000000 0.000000 1.000000"],
    ["-5.083177 3.096819 -0.304200 21.594027 -0.000001 6.205755 0.280781 -13.654707 5.514645 2.854524 -0.280400 19.127966 0.000000 0.000000 0.000000 1.000000"],
    ["5.083176 3.096823 0.304200 -20.994028 0.000001 6.205754 -0.280782 25.654707 -5.514647 2.854525 0.280399 -20.127966 0.000000 0.000000 0.000000 1.000000"],
    ["-7.326222 1.127744 0.076148 -5.030359 -0.000001 5.337012 -0.351292 30.590412 -1.605140 -5.147280 -0.347557 23.828972 0.000000 0.000000 0.000000 1.000000"],
    ["7.326222 1.127744 -0.076148 5.630359 -0.000000 5.337012 0.351292 -18.590412 1.605139 -5.147281 0.347557 -24.828972 0.000000 0.000000 0.000000 1.000000"],
    ["-6.474771 -2.616503 0.182346 -12.464223 -0.000001 5.419583 0.345627 -18.193922 -3.785149 4.475716 -0.311916 21.334120 0.000000 0.000000 0.000000 1.000000"],
    ["6.474770 -2.616504 -0.182346 13.064223 -0.000000 5.419581 -0.345628 30.193922 3.785150 4.475718 0.311916 -22.334120 0.000000 0.000000 0.000000 1.000000"],
    ["-5.451219 -0.508029 -0.341735 24.221445 -0.000000 7.463435 -0.049312 9.451858 5.151137 -0.537624 -0.361643 24.814997 0.000000 0.000000 0.000000 1.000000"],
    ["5.451219 -0.508028 0.341735 -23.621447 -0.000000 7.463435 0.049312 2.548143 -5.151137 -0.537624 0.361643 -25.814997 0.000000 0.000000 0.000000 1.000000"],
    ["-5.935410 0.985898 0.298507 -20.595474 -0.000000 7.324550 -0.107517 13.526174 -4.584857 -1.276312 -0.386437 26.550617 0.000000 0.000000 0.000000 1.000000"],
    ["5.935410 0.985898 -0.298507 21.195473 -0.000000 7.324550 0.107517 -1.526174 4.584857 -1.276312 0.386437 -27.550617 0.000000 0.000000 0.000000 1.000000"],
    ["-7.250820 1.221711 -0.098500 7.195035 -0.000000 5.779980 0.318621 -16.303457 1.917187 4.620525 -0.372530 25.577082 0.000000 0.000000 0.000000 1.000000"],
    ["7.250820 1.221712 0.098501 -6.595035 0.000000 5.779980 -0.318621 28.303457 -1.917188 4.620525 0.372530 -26.577082 0.000000 0.000000 0.000000 1.000000"],
    ["-7.281588 -0.772892 -0.108138 7.869672 -0.000001 6.770683 -0.215075 21.055218 1.796797 -3.132169 -0.438234 30.176382 0.000000 0.000000 0.000000 1.000000"],
    ["7.281588 -0.772891 0.108138 -7.269671 -0.000000 6.770683 0.215075 -9.055218 -1.796797 -3.132170 0.438234 -31.176382 0.000000 0.000000 0.000000 1.000000"],
    ["-6.724941 -0.962456 0.211858 -14.530052 0.000001 7.178018 0.144930 -4.145110 -3.320417 1.949293 -0.429082 29.535753 0.000000 0.000000 0.000000 1.000000"],
    ["6.724941 -0.962458 -0.211858 15.130053 -0.000000 7.178017 -0.144930 16.145111 3.320418 1.949295 0.429082 -30.535753 0.000000 0.000000 0.000000 1.000000"],
    ["-6.935052 0.578879 -0.186428 13.349942 0.000001 7.344292 0.101355 -1.094834 2.855705 1.405801 -0.452738 31.191675 0.000000 0.000000 0.000000 1.000000"],
    ["6.935052 0.578879 0.186428 -12.749942 -0.000000 7.344292 -0.101355 13.094834 -2.855704 1.405802 0.452738 -32.191673 0.000000 0.000000 0.000000 1.000000"],
    ["-7.418036 0.175292 0.072786 -4.795047 0.000001 7.405165 -0.079261 11.548292 -1.105778 -1.175927 -0.488282 33.679771 0.000000 0.000000 0.000000 1.000000"],
    ["7.418036 0.175291 -0.072786 5.395048 -0.000000 7.405166 0.079261 0.451708 1.105778 -1.175925 0.488283 -34.679771 0.000000 0.000000 0.000000 1.000000"],
    ["-7.500000 0.000001 0.000000 0.300000 0.000001 7.471460 0.043578 2.949549 -0.000001 0.653668 -0.498097 34.366814 0.000000 0.000000 0.000000 1.000000"],
    ["7.500000 0.000000 0.000000 0.300000 0.000000 7.471460 -0.043578 9.050451 0.000000 0.653668 0.498097 -35.366814 0.000000 0.000000 0.000000 1.000000"],
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
