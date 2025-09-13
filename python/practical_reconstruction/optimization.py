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
    ["7.500000 0.000000 0.000000 0.300000 0.000000 7.471460 0.043578 2.949549 0.000000 -0.653668 0.498097 -35.366814 0.000000 0.000000 0.000000 1.000000"],
    ["-7.500000 0.000001 -0.000000 0.300000 0.000001 7.500000 -0.000000 6.000000 0.000000 -0.000001 -0.500000 34.500000 0.000000 0.000000 0.000000 1.000000"],
    ["7.328826 0.166872 0.105630 -7.094067 -0.000000 7.458747 -0.052370 9.665897 -1.593206 0.767620 0.485901 -34.513069 0.000000 0.000000 0.000000 1.000000"],
    ["-7.328826 0.304331 -0.104258 7.598060 0.000001 7.361900 0.095509 -0.685621 1.593207 1.399935 -0.479592 33.071430 0.000000 0.000000 0.000000 1.000000"],
    ["7.494684 0.133494 -0.016586 1.461014 0.000000 6.608728 0.236406 -10.548397 0.282341 -3.543571 0.440270 -31.318865 0.000000 0.000000 0.000000 1.000000"],
    ["-7.494684 0.111304 0.017298 -0.910891 0.000001 6.892640 -0.197107 19.797482 -0.282341 -2.954509 -0.459184 31.642860 0.000000 0.000000 0.000000 1.000000"],
    ["7.117022 -0.706185 -0.150544 10.838095 -0.000000 7.158140 -0.149236 16.446486 2.366009 2.124227 0.452841 -32.198891 0.000000 0.000000 0.000000 1.000000"],
    ["-7.117022 -0.900310 0.145868 -9.910769 -0.000001 6.935801 0.190259 -7.318140 -2.366009 2.708157 -0.438776 30.214287 0.000000 0.000000 0.000000 1.000000"],
    ["6.304254 -0.738532 0.266341 -18.343878 0.000000 7.375046 0.090889 -0.362262 -4.062806 -1.145979 0.413281 -29.429699 0.000000 0.000000 0.000000 1.000000"],
    ["-6.304255 -0.387525 -0.269619 19.173311 -0.000000 7.465805 -0.047692 9.338423 4.062804 -0.601322 -0.418367 28.785717 0.000000 0.000000 0.000000 1.000000"],
    ["6.311899 1.645186 -0.246786 17.575043 0.000000 6.853625 0.203063 -8.214446 4.050918 -2.563432 0.384528 -27.416950 0.000000 0.000000 0.000000 1.000000"],
    ["-6.311899 1.316295 0.255406 -17.578449 0.000001 7.093016 -0.162469 17.372805 -4.050917 -2.050972 -0.397959 27.357143 0.000000 0.000000 0.000000 1.000000"],
    ["7.316449 0.928876 0.090842 -6.058943 -0.000000 6.197108 -0.281629 25.714031 -1.649113 4.121049 0.403030 -28.712067 0.000000 0.000000 0.000000 1.000000"],
    ["-7.316449 1.044102 -0.085099 6.256947 -0.000000 5.805343 0.316565 -16.159544 1.649113 4.632261 -0.377551 25.928574 0.000000 0.000000 0.000000 1.000000"],
    ["6.835326 -2.120719 0.149531 -10.167147 0.000000 5.449726 0.343514 -18.046011 -3.086798 -4.696066 0.331117 -23.678185 0.000000 0.000000 0.000000 1.000000"],
    ["-6.835326 -1.917162 -0.161284 11.589870 0.000001 5.878076 -0.310542 27.737961 3.086798 -4.245317 -0.357143 24.500000 0.000000 0.000000 0.000000 1.000000"],
    ["5.221693 -0.906111 -0.353791 25.065395 -0.000000 7.393009 -0.084154 11.890759 5.383672 0.878849 0.343147 -24.520285 0.000000 0.000000 0.000000 1.000000"],
    ["-5.221695 -1.365186 0.347180 -24.002619 0.000002 7.254859 0.126790 -2.875278 -5.383669 1.324115 -0.336735 23.071430 0.000000 0.000000 0.000000 1.000000"],
    ["4.966681 1.186354 0.366210 -25.334677 0.000000 7.330979 -0.105551 13.388588 -5.619793 1.048480 0.323650 -23.155512 0.000000 0.000000 0.000000 1.000000"],
    ["-4.966682 1.660596 -0.357923 25.354605 -0.000001 7.165092 0.147745 -4.342178 5.619792 1.467608 -0.316327 21.642857 0.000000 0.000000 0.000000 1.000000"],
    ["6.495440 2.950367 -0.154267 11.098653 -0.000000 4.628530 0.393428 -21.539946 3.749567 -5.110973 0.267239 -19.206701 0.000000 0.000000 0.000000 1.000000"],
    ["-6.495441 2.737462 0.170822 -11.657554 0.000001 5.125258 -0.365037 31.552601 -3.749565 -4.742155 -0.295918 20.214287 0.000000 0.000000 0.000000 1.000000"],
    ["6.831079 -2.292678 -0.138724 10.010694 0.000000 5.040548 -0.370242 31.916969 3.096185 5.058310 0.306066 -21.924593 0.000000 0.000000 0.000000 1.000000"],
    ["-6.831079 -2.465313 0.124875 -8.441247 -0.000000 4.537337 0.398121 -21.868475 -3.096186 5.439192 -0.275510 18.785715 0.000000 0.000000 0.000000 1.000000"],
    ["4.241152 -3.143757 0.355149 -24.560425 -0.000000 6.459154 0.254116 -11.788103 -6.185679 -2.155488 0.243504 -17.545313 0.000000 0.000000 0.000000 1.000000"],
    ["-4.241152 -2.667497 -0.372064 26.344473 -0.000001 6.766788 -0.215619 21.093304 6.185679 -1.828943 -0.255102 17.357143 0.000000 0.000000 0.000000 1.000000"],
    ["3.585440 1.807922 -0.422301 29.861053 0.000000 7.212013 0.137224 -3.605716 6.587459 -0.984020 0.229851 -16.589569 0.000000 0.000000 0.000000 1.000000"],
    ["-3.585439 1.248953 0.431199 -29.883900 -0.000000 7.363967 -0.094798 12.635845 -6.587460 -0.679784 -0.234694 15.928572 0.000000 0.000000 0.000000 1.000000"],
    ["4.772033 3.920635 0.283677 -19.557381 -0.000000 5.515674 -0.338804 29.716280 -5.785993 3.233568 0.233964 -16.877499 0.000000 0.000000 0.000000 1.000000"],
    ["-4.772033 4.276577 -0.259817 18.487190 0.000000 5.051755 0.369563 -19.869408 5.785992 3.527133 -0.214286 14.500000 0.000000 0.000000 0.000000 1.000000"],
    ["7.172763 -2.072851 0.047364 -3.015504 -0.000000 2.431738 0.472989 -27.109219 -2.191226 -6.785274 0.155042 -11.352975 0.000000 0.000000 0.000000 1.000000"],
    ["-7.172763 -2.003042 -0.059228 4.445971 -0.000001 3.040841 -0.457060 37.994175 2.191226 -6.556760 -0.193878 13.071429 0.000000 0.000000 0.000000 1.000000"],
    ["3.266159 -3.596402 -0.380923 26.964638 -0.000000 6.347352 -0.266343 24.643972 6.751460 1.739834 0.184280 -13.399581 0.000000 0.000000 0.000000 1.000000"],
    ["-3.266161 -4.080709 0.358577 -24.800425 0.000000 5.975000 0.302209 -15.154665 -6.751459 1.974130 -0.173469 11.642858 0.000000 0.000000 0.000000 1.000000"],
    ["2.297698 -0.342003 0.475411 -32.978809 -0.000001 7.491390 0.023952 4.323370 -7.139369 -0.110069 0.153004 -11.210276 0.000000 0.000000 0.000000 1.000000"],
    ["-2.297698 0.280822 -0.475590 33.591270 0.000000 7.494196 0.019667 4.623301 7.139369 0.090378 -0.153061 10.214287 0.000000 0.000000 0.000000 1.000000"],
    ["2.714157 5.183684 -0.312785 22.194952 0.000001 5.032897 0.370705 -19.949331 6.991663 -2.012302 0.121423 -8.999601 0.000000 0.000000 0.000000 1.000000"],
    ["-2.714157 4.755044 0.341714 -23.619980 -0.000000 5.498381 -0.340051 29.803574 -6.991663 -1.845904 -0.132653 8.785715 0.000000 0.000000 0.000000 1.000000"],
    ["7.122634 2.323870 0.022872 -1.301074 0.000000 1.095397 -0.494638 40.624687 -2.349059 7.046257 0.069352 -5.354651 0.000000 0.000000 0.000000 1.000000"],
    ["-7.353633 1.435309 -0.022506 1.875437 -0.000000 1.717186 0.486718 -28.070271 1.474477 7.158293 -0.112245 7.357143 0.000000 0.000000 0.000000 1.000000"],
    ["2.099781 -5.825052 0.282132 -19.449261 0.000000 4.408278 0.404514 -22.315975 -7.200064 -1.698781 0.082279 -6.259549 0.000000 0.000000 0.000000 1.000000"],
    ["-2.099782 -5.434044 -0.314904 22.343313 0.000000 4.920338 -0.377361 32.415260 7.200064 -1.584751 -0.091837 5.928572 0.000000 0.000000 0.000000 1.000000"],
    ["1.080882 -0.334599 -0.494277 34.899403 -0.000000 7.492374 -0.022542 7.577931 7.421704 0.048731 0.071986 -5.538995 0.000000 0.000000 0.000000 1.000000"],
    ["-1.080883 -0.979511 0.490452 -34.031654 -0.000001 7.434393 0.065990 1.380724 -7.421704 0.142653 -0.071429 4.500000 0.000000 0.000000 0.000000 1.000000"],
    ["0.929983 3.678388 0.431301 -29.891064 -0.000001 6.519829 -0.247133 23.299314 -7.442119 0.459659 0.053896 -4.272741 0.000000 0.000000 0.000000 1.000000"],
    ["-0.929984 4.228244 -0.408287 28.880077 -0.000001 6.171935 0.284075 -13.885275 7.442119 0.528372 -0.051020 3.071429 0.000000 0.000000 0.000000 1.000000"],
    ["-0.882609 7.263825 -0.109704 7.979250 0.000000 1.657068 0.487643 -28.135040 7.447886 0.860797 -0.013000 0.410026 0.000000 0.000000 0.000000 1.000000"],
    ["-2.018707 7.033868 0.109535 -7.367437 -0.000001 1.705981 -0.486893 40.082527 -7.223214 -1.965790 -0.030612 1.642857 0.000000 0.000000 0.000000 1.000000"],
    ["0.307662 -6.150860 -0.285363 20.275440 -0.000000 4.284057 -0.410403 34.728199 7.493687 0.252531 0.011716 -1.320118 0.000000 0.000000 0.000000 1.000000"],
    ["-0.307663 -6.500520 0.248539 -17.097710 0.000000 3.731221 0.433733 -24.361319 -7.493687 0.266888 -0.010204 0.214286 0.000000 0.000000 0.000000 1.000000"],
    ["-0.160660 -2.892458 0.461196 -31.983715 0.000001 6.919527 0.192875 -7.501244 -7.498279 0.061975 -0.009882 0.191711 0.000000 0.000000 0.000000 1.000000"],
    ["0.160657 -2.278515 -0.476247 33.637306 -0.000000 7.145347 -0.151936 16.635509 7.498279 0.048820 0.010204 -1.214286 0.000000 0.000000 0.000000 1.000000"],
    ["-0.505623 3.712984 -0.433118 30.618282 -0.000000 6.511590 0.248097 -11.366772 7.482937 0.250887 -0.029266 1.548610 0.000000 0.000000 0.000000 1.000000"],
    ["0.505624 3.132625 0.453044 -31.413086 0.000000 6.811156 -0.209318 20.652248 -7.482937 0.211672 0.030612 -2.642857 0.000000 0.000000 0.000000 1.000000"],
    ["-1.925925 6.375299 0.229941 -15.795859 0.000001 3.568785 -0.439766 36.783653 -7.248504 -1.693914 -0.061095 3.776662 0.000000 0.000000 0.000000 1.000000"],
    ["1.925925 6.651649 -0.192023 13.741608 -0.000001 2.980282 0.458829 -26.118034 7.248504 -1.767341 0.051020 -4.071429 0.000000 0.000000 0.000000 1.000000"],
    ["-2.942106 -6.619630 0.129521 -8.766449 0.000000 2.112106 0.479764 -27.583469 -6.898841 2.823032 -0.055236 3.366510 0.000000 0.000000 0.000000 1.000000"],
    ["2.942106 -6.425113 -0.167490 12.024331 -0.000000 2.731281 -0.465666 38.596626 6.898841 2.740077 0.071429 -5.500000 0.000000 0.000000 0.000000 1.000000"],
    ["-1.549021 -2.774371 -0.452909 32.003601 0.000000 6.943335 -0.189034 19.232370 7.338292 -0.585635 -0.095603 6.192231 0.000000 0.000000 0.000000 1.000000"],
    ["1.549021 -3.355918 0.435065 -30.154551 0.000001 6.669782 0.228658 -10.006060 -7.338292 -0.708391 0.091837 -6.928572 0.000000 0.000000 0.000000 1.000000"],
    ["-1.738145 1.189327 0.479881 -33.291691 -0.000000 7.399677 -0.081508 11.705523 -7.295811 -0.283344 -0.114326 7.502840 0.000000 0.000000 0.000000 1.000000"],
    ["1.738144 1.812166 -0.471145 33.280132 0.000001 7.264962 0.124192 -2.693458 7.295811 -0.431728 0.112245 -8.357143 0.000000 0.000000 0.000000 1.000000"],
    ["-3.400968 5.741309 -0.228238 16.276623 -0.000001 3.841195 0.429445 -24.061165 6.684566 2.921060 -0.116122 7.628573 0.000000 0.000000 0.000000 1.000000"],
    ["3.400968 5.421079 0.260728 -17.950975 -0.000000 4.388007 -0.405492 34.384460 -6.684566 2.758132 0.132653 -9.785715 0.000000 0.000000 0.000000 1.000000"],
    ["-6.635746 -3.161510 -0.099370 7.255907 -0.000001 3.198366 -0.452256 37.657909 3.495266 -6.002109 -0.188654 12.705755 0.000000 0.000000 0.000000 1.000000"],
    ["6.635746 -3.279389 0.080622 -5.343568 0.000000 2.594944 0.469119 -26.838306 -3.495266 -6.225904 0.153061 -11.214287 0.000000 0.000000 0.000000 1.000000"],
    ["-3.178397 -4.371010 0.346680 -23.967592 0.000001 5.741240 0.321719 -16.520317 -6.793217 2.045101 -0.162204 10.854267 0.000000 0.000000 0.000000 1.000000"],
    ["3.178394 -3.901153 -0.370758 26.253054 0.000001 6.139986 -0.287136 26.099504 6.793218 1.825262 0.173469 -12.642858 0.000000 0.000000 0.000000 1.000000"],
    ["-2.916623 1.124330 -0.454504 32.115303 0.000001 7.400044 0.081359 0.304843 6.909653 0.474588 -0.191850 12.929502 0.000000 0.000000 0.000000 1.000000"],
    ["2.916623 0.525862 0.459308 -31.851536 0.000000 7.478248 -0.038053 8.663689 -6.909653 0.221971 0.193878 -14.071429 0.000000 0.000000 0.000000 1.000000"],
    ["-4.294805 3.661346 0.329304 -22.751261 0.000002 6.025269 -0.297741 26.841841 -6.148549 -2.557474 -0.230021 15.601463 0.000000 0.000000 0.000000 1.000000"],
    ["4.294805 4.077924 -0.306777 21.774374 -0.000000 5.613094 0.331617 -17.213171 6.148549 -2.848458 0.214286 -15.500000 0.000000 0.000000 0.000000 1.000000"],
    ["-7.499684 0.063355 -0.001792 0.425472 0.000000 2.929983 0.460267 -26.218662 0.068824 6.903708 -0.195324 13.172678 0.000000 0.000000 0.000000 1.000000"],
    ["7.499684 0.060770 0.002154 0.149238 -0.000000 3.520557 -0.441491 36.904358 -0.068823 6.622084 0.234694 -16.928572 0.000000 0.000000 0.000000 1.000000"],
    ["-4.963970 -3.189996 -0.308638 21.904667 -0.000001 6.175848 -0.283697 25.858795 5.622188 -2.816527 -0.272504 18.575289 0.000000 0.000000 0.000000 1.000000"],
    ["4.963968 -3.581353 0.288929 -19.924999 0.000001 5.781459 0.318502 -16.295109 -5.622190 -3.162062 0.255102 -18.357143 0.000000 0.000000 0.000000 1.000000"],
    ["-4.144962 -1.022679 0.411088 -28.476128 -0.000000 7.398932 0.081807 0.273499 -6.250543 0.678175 -0.272607 18.582497 0.000000 0.000000 0.000000 1.000000"],
    ["4.144962 -0.481358 -0.415465 29.382576 0.000000 7.477727 -0.038505 8.695367 6.250544 0.319205 0.275510 -19.785715 0.000000 0.000000 0.000000 1.000000"],
    ["-5.083177 3.096819 -0.304200 21.594027 -0.000001 6.205755 0.280781 -13.654707 5.514645 2.854524 -0.280400 19.127966 0.000000 0.000000 0.000000 1.000000"],
    ["5.083175 2.687346 0.321037 -22.172558 0.000000 6.549215 -0.243655 23.055868 -5.514647 2.477086 0.295918 -21.214287 0.000000 0.000000 0.000000 1.000000"],
    ["-7.326222 1.127744 0.076148 -5.030359 -0.000001 5.337012 -0.351292 30.590412 -1.605140 -5.147280 -0.347557 23.828972 0.000000 0.000000 0.000000 1.000000"],
    ["7.326222 1.223004 -0.069306 5.151392 0.000000 4.857447 0.380965 -20.667545 1.605140 -5.582067 0.316326 -22.642857 0.000000 0.000000 0.000000 1.000000"],
    ["-6.474771 -2.616503 0.182346 -12.464223 -0.000001 5.419583 0.345627 -18.193922 -3.785149 4.475716 -0.311916 21.334120 0.000000 0.000000 0.000000 1.000000"],
    ["6.474770 -2.368159 -0.196855 14.079853 -0.000000 5.850810 -0.312822 27.897566 3.785150 4.050906 0.336735 -24.071428 0.000000 0.000000 0.000000 1.000000"],
    ["-5.451219 -0.508029 -0.341735 24.221445 -0.000000 7.463435 -0.049312 9.451858 5.151137 -0.537624 -0.361643 24.814997 0.000000 0.000000 0.000000 1.000000"],
    ["5.451218 -0.952858 0.337483 -23.323792 -0.000001 7.370567 0.092490 -0.474302 -5.151138 -1.008368 0.357143 -25.500000 0.000000 0.000000 0.000000 1.000000"],
    ["-5.935410 0.985898 0.298507 -20.595474 -0.000000 7.324550 -0.107517 13.526174 -4.584857 -1.276312 -0.386437 26.550617 0.000000 0.000000 0.000000 1.000000"],
    ["5.935410 1.372396 -0.291642 20.714970 -0.000000 7.156116 0.149666 -4.476626 4.584857 -1.776660 0.377551 -26.928574 0.000000 0.000000 0.000000 1.000000"],
    ["-7.250820 1.221711 -0.098500 7.195035 -0.000000 5.779980 0.318621 -16.303457 1.917187 4.620525 -0.372530 25.577082 0.000000 0.000000 0.000000 1.000000"],
    ["7.250820 1.088290 0.105224 -7.065701 0.000000 6.174529 -0.283825 25.867716 -1.917188 4.115922 0.397959 -28.357143 0.000000 0.000000 0.000000 1.000000"],
    ["-7.281588 -0.772892 -0.108138 7.869672 -0.000001 6.770683 -0.215075 21.055218 1.796797 -3.132169 -0.438234 30.176382 0.000000 0.000000 0.000000 1.000000"],
    ["7.281588 -0.911322 0.103236 -6.926512 0.000000 6.463745 0.253596 -11.751747 -1.796797 -3.693168 0.418367 -29.785717 0.000000 0.000000 0.000000 1.000000"],
    ["-6.724941 -0.962456 0.211858 -14.530052 0.000001 7.178018 0.144930 -4.145110 -3.320417 1.949293 -0.429082 29.535753 0.000000 0.000000 0.000000 1.000000"],
    ["6.724941 -0.681826 -0.216644 15.465076 -0.000000 7.340175 -0.102672 13.187013 3.320418 1.380923 0.438775 -31.214287 0.000000 0.000000 0.000000 1.000000"],
    ["-6.935052 0.578879 -0.186428 13.349942 0.000001 7.344292 0.101355 -1.094834 2.855705 1.405801 -0.452738 31.191675 0.000000 0.000000 0.000000 1.000000"],
    ["6.935052 0.332952 0.189082 -12.935728 0.000001 7.448849 -0.058296 10.080717 -2.855704 0.808572 0.459184 -32.642860 0.000000 0.000000 0.000000 1.000000"],
    ["-7.418036 0.175292 0.072786 -4.795047 0.000001 7.405165 -0.079261 11.548292 -1.105778 -1.175927 -0.488282 33.679771 0.000000 0.000000 0.000000 1.000000"],
    ["7.418036 0.269780 -0.071491 5.304364 -0.000000 7.273365 0.121987 -2.539059 1.105778 -1.809802 0.479592 -34.071430 0.000000 0.000000 0.000000 1.000000"],
    ["-7.500000 0.000001 0.000000 0.300000 0.000001 7.471460 0.043578 2.949549 -0.000001 0.653668 -0.498097 34.366814 0.000000 0.000000 0.000000 1.000000"],
    ["7.500000 0.000000 0.000000 0.300000 0.000000 7.500000 -0.000000 6.000000 0.000000 0.000001 0.500000 -35.500000 0.000000 0.000000 0.000000 1.000000"],
  ]
  # print("before(no T):\n", np.array(params['arealight.to_world']))
  idx = int(sensor.id()[len('elm__') :])
  raw_string = light_positions[idx-1][0]
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
  backlit_indices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,79,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99]
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
        # for k in params.keys():
        #   print(k, params[k])
          
        dr.backward(rendering_loss)
        print(rendering_loss)
        img_np = np.array(img)
        ref_np = np.array(ref_img)
        img_disp = tonemap(img_np)
        ref_disp = tonemap(ref_np)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Rendered img")
        plt.imshow(img_disp)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Reference ref_img")
        plt.imshow(ref_disp)
        plt.axis("off")
        plt.show()
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