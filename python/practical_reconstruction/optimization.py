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
    ["6.500000 0.000000 0.000000 0.300000 0.000000 7.471460 0.043578 2.949553 0.000000 -0.653668 0.498097 -35.366814 0.000000 0.000000 0.000000 1.000000"],
    ["6.500000 0.000000 0.000000 0.300000 0.000000 6.797308 -0.211309 20.791634 0.000000 3.169637 0.453154 -32.220776 0.000000 0.000000 0.000000 1.000000"],
    ["6.351649 -0.439473 0.102093 -6.846509 -0.000000 7.209023 0.137921 -3.654459 -1.380779 -2.021596 0.469633 -33.374290 0.000000 0.000000 0.000000 1.000000"],
    ["6.351649 0.385103 0.103064 -6.914496 -0.000000 7.277604 -0.120858 14.460037 -1.380779 1.771490 0.474100 -33.687031 0.000000 0.000000 0.000000 1.000000"],
    ["6.495393 -0.088264 -0.017879 1.551554 -0.000000 7.124097 -0.156308 16.941559 0.244696 2.342958 0.474603 -33.722225 0.000000 0.000000 0.000000 1.000000"],
    ["6.495393 -0.210534 -0.012542 1.177927 -0.000000 4.997340 -0.372837 32.098557 0.244696 5.588583 0.332920 -23.804396 0.000000 0.000000 0.000000 1.000000"],
    ["6.168086 1.087582 -0.140082 10.105734 -0.000000 6.660676 0.229835 -10.088434 2.050541 -3.271477 0.421370 -29.995932 0.000000 0.000000 0.000000 1.000000"],
    ["6.168086 -0.108740 -0.157567 11.329707 -0.000000 7.492075 -0.022980 7.608574 2.050541 0.327092 0.473967 -33.677677 0.000000 0.000000 0.000000 1.000000"],
    ["5.463686 0.033567 0.270844 -18.659111 0.000000 7.499744 -0.004131 6.289180 -3.521098 0.052087 0.420269 -29.918848 0.000000 0.000000 0.000000 1.000000"],
    ["5.463686 2.060403 0.233439 -16.040749 0.000000 6.463986 -0.253569 23.749834 -3.521098 3.197128 0.362228 -25.855938 0.000000 0.000000 0.000000 1.000000"],
    ["5.470312 -0.977385 -0.262083 18.645786 0.000001 7.278426 -0.120637 14.444611 3.510796 1.522902 0.408362 -29.085323 0.000000 0.000000 0.000000 1.000000"],
    ["5.470313 -2.812058 -0.194391 13.907358 -0.000000 5.398524 -0.347089 30.296238 3.510794 4.381584 0.302888 -21.702185 0.000000 0.000000 0.000000 1.000000"],
    ["6.340922 -1.151383 0.078709 -5.209615 0.000000 5.369395 0.349092 -18.436405 -1.429231 -5.108221 0.349199 -24.943943 0.000000 0.000000 0.000000 1.000000"],
    ["6.340923 -0.406811 0.106543 -7.158025 0.000000 7.268218 0.123342 -2.633968 -1.429231 -1.804858 0.472689 -33.588253 0.000000 0.000000 0.000000 1.000000"],
    ["5.923949 1.699015 0.171810 -11.726667 0.000000 6.261692 -0.275207 25.264473 -2.675224 3.762256 0.380451 -27.131548 0.000000 0.000000 0.000000 1.000000"],
    ["5.923949 2.759961 0.092158 -6.151034 0.000000 3.358736 -0.447059 37.294132 -2.675224 6.111588 0.204072 -14.785010 0.000000 0.000000 0.000000 1.000000"],
    ["4.525468 1.813873 -0.337927 23.954880 0.000001 7.061496 0.168461 -5.792248 4.665848 -1.759300 0.327760 -23.443186 0.000000 0.000000 0.000000 1.000000"],
    ["4.525468 -0.963590 -0.353116 25.018101 0.000000 7.378890 -0.089492 12.264431 4.665848 0.934599 0.342492 -24.474415 0.000000 0.000000 0.000000 1.000000"],
    ["4.304458 -2.122202 0.346912 -23.983856 -0.000000 6.944674 0.188815 -7.217054 -4.870487 -1.875568 0.306596 -21.961679 0.000000 0.000000 0.000000 1.000000"],
    ["4.304458 0.763958 0.371175 -25.682243 0.000000 7.430377 -0.067970 10.757923 -4.870487 0.675174 0.328038 -23.462687 0.000000 0.000000 0.000000 1.000000"],
    ["5.629382 -2.503723 -0.186078 13.325452 0.000000 5.582982 -0.333868 29.370789 3.249624 4.337244 0.322346 -23.064222 0.000000 0.000000 0.000000 1.000000"],
    ["5.629382 -3.563872 -0.077691 5.738356 0.000001 2.330994 -0.475238 39.266651 3.249624 6.173758 0.134585 -9.920959 0.000000 0.000000 0.000000 1.000000"],
    ["5.920268 2.619186 -0.110075 8.005277 -0.000000 3.999592 0.422970 -23.607883 2.683361 -5.778679 0.242858 -17.500065 0.000000 0.000000 0.000000 1.000000"],
    ["5.920268 1.442716 -0.182634 13.084397 0.000000 6.636021 0.232983 -10.308805 2.683361 -3.183048 0.402944 -28.706070 0.000000 0.000000 0.000000 1.000000"],
    ["3.675664 2.170933 0.386147 -26.730310 0.000000 7.022925 -0.175481 18.283640 -5.360923 1.488479 0.264758 -19.033072 0.000000 0.000000 0.000000 1.000000"],
    ["3.675665 4.776186 0.262049 -18.043432 -0.000000 4.765929 -0.386068 33.024765 -5.360922 3.274746 0.179671 -13.076999 0.000000 0.000000 0.000000 1.000000"],
    ["3.107381 -0.680481 -0.436815 30.877024 0.000000 7.459877 -0.051650 9.615475 5.709131 0.370373 0.237751 -17.142540 0.000000 0.000000 0.000000 1.000000"],
    ["3.107382 -3.865420 -0.355610 25.192696 0.000000 6.073073 -0.293392 26.537466 5.709131 2.103881 0.193552 -14.048662 0.000000 0.000000 0.000000 1.000000"],
    ["4.135761 -4.599971 0.233980 -16.078588 -0.000000 4.549389 0.397509 -21.825649 -5.014527 -3.793853 0.192976 -14.008342 0.000000 0.000000 0.000000 1.000000"],
    ["4.135762 -2.228845 0.355965 -24.617537 0.000001 6.921205 0.192607 -7.482487 -5.014526 -1.838253 0.293584 -21.050892 0.000000 0.000000 0.000000 1.000000"],
    ["6.216394 1.917989 0.070641 -4.644885 -0.000000 3.626800 -0.437652 36.635635 -1.899063 6.278347 0.231237 -16.686594 0.000000 0.000000 0.000000 1.000000"],
    ["-6.216394 -2.190839 -0.002756 0.492908 0.000003 0.141480 -0.499911 40.993771 1.899065 -7.171486 -0.009020 0.131465 0.000000 0.000000 0.000000 1.000000"],
    ["2.830672 4.533963 -0.333503 23.645176 -0.000000 5.557174 0.335777 -17.504354 5.851264 -2.193399 0.161339 -11.793720 0.000000 0.000000 0.000000 1.000000"],
    ["2.830671 1.425259 -0.439954 31.096760 -0.000000 7.330977 0.105552 -1.388638 5.851265 -0.689498 0.212837 -15.398581 0.000000 0.000000 0.000000 1.000000"],
    ["2.352269 -4.290216 -0.368042 26.062958 0.000000 5.922018 -0.306809 27.476658 6.059442 1.665457 0.142874 -10.501157 0.000000 0.000000 0.000000 1.000000"],
    ["2.352269 -6.475752 -0.175727 12.600883 -0.000001 2.827549 -0.463105 38.417370 6.059442 2.513881 0.068217 -5.275190 0.000000 0.000000 0.000000 1.000000"],
    ["6.373148 -1.459272 0.014081 -0.685663 0.000001 1.074344 0.494844 -28.639053 -1.277881 -7.277796 0.070225 -5.415778 0.000000 0.000000 0.000000 1.000000"],
    ["6.373148 -1.158160 0.060837 -3.958575 0.000000 4.641737 0.392736 -21.491495 -1.277881 -5.776067 0.303410 -21.738705 0.000000 0.000000 0.000000 1.000000"],
    ["1.819810 5.001680 0.345280 -23.869604 0.000000 5.394952 -0.347336 30.313511 -6.240056 1.458658 0.100695 -7.548669 0.000000 0.000000 0.000000 1.000000"],
    ["1.819811 6.921181 0.132299 -8.960909 0.000001 2.067149 -0.480633 39.644341 -6.240055 2.018451 0.038583 -3.200793 0.000000 0.000000 0.000000 1.000000"],
    ["0.805984 -4.745923 0.382165 -26.451584 0.000001 5.777067 0.318856 -16.319895 -6.449836 -0.593059 0.047756 -3.842936 0.000000 0.000000 0.000000 1.000000"],
    ["0.805986 -1.243851 0.489162 -33.941372 0.000000 7.394503 0.083568 0.150218 -6.449836 -0.155434 0.061127 -4.778876 0.000000 0.000000 0.000000 1.000000"],
    ["1.749547 -6.863903 -0.149987 10.799120 0.000001 2.336022 -0.475128 39.258968 6.260118 1.918289 0.041918 -3.434242 0.000000 0.000000 0.000000 1.000000"],
    ["-1.749547 7.069221 0.098904 -6.623261 0.000003 1.540405 -0.489340 40.253830 -6.260118 -1.975670 -0.027641 1.434879 0.000000 0.000000 0.000000 1.000000"],
    ["0.266641 6.800706 -0.209822 14.987573 -0.000000 3.149989 0.453762 -25.763372 6.494529 -0.279212 0.008615 -1.103018 0.000000 0.000000 0.000000 1.000000"],
    ["0.266641 4.315919 -0.408402 28.888123 -0.000000 6.131186 0.287970 -14.157923 6.494529 -0.177196 0.016767 -1.673723 0.000000 0.000000 0.000000 1.000000"],
    ["-1.669135 -6.877375 0.152644 -10.385056 -0.000001 2.369097 0.474400 -27.207977 -6.282037 1.827316 -0.040557 2.339015 0.000000 0.000000 0.000000 1.000000"],
    ["-1.669134 -4.811156 0.361439 -25.000736 0.000000 5.609695 0.331872 -17.231064 -6.282037 1.278322 -0.096034 6.222393 0.000000 0.000000 0.000000 1.000000"],
    ["-2.549825 6.181698 0.204185 -13.992978 -0.000001 3.329668 -0.448024 37.361702 -5.978996 -2.636271 -0.087078 5.595436 0.000000 0.000000 0.000000 1.000000"],
    ["2.549825 -6.884898 -0.029227 2.345869 -0.000002 0.476602 -0.498989 40.929260 5.978996 2.936160 0.012464 -1.372489 0.000000 0.000000 0.000000 1.000000"],
    ["-2.947507 -5.059590 -0.291235 20.686420 -0.000000 4.901424 -0.378453 32.491730 5.793289 -2.574216 -0.148174 9.872185 0.000000 0.000000 0.000000 1.000000"],
    ["-2.947505 -6.565994 -0.083564 6.149453 -0.000003 1.406361 -0.491131 40.379166 5.793291 -3.340639 -0.042516 2.476077 0.000000 0.000000 0.000000 1.000000"],
    ["-5.750979 3.372309 -0.061261 4.588279 -0.000001 1.971776 0.482411 -27.768780 3.029230 6.402314 -0.116304 7.641276 0.000000 0.000000 0.000000 1.000000"],
    ["-5.750979 2.461048 -0.165464 11.882480 0.000000 5.325689 0.352055 -18.643820 3.029230 4.672288 -0.314133 21.489283 0.000000 0.000000 0.000000 1.000000"],
    ["-2.754611 3.401601 0.392014 -27.140997 0.000001 6.492005 -0.250367 23.525726 -5.887454 -1.591534 -0.183415 12.339033 0.000000 0.000000 0.000000 1.000000"],
    ["-2.754611 5.885979 0.226108 -15.527535 0.000000 3.744486 -0.433225 36.325729 -5.887454 -2.753921 -0.105791 6.905353 0.000000 0.000000 0.000000 1.000000"],
    ["-3.722164 -4.463465 0.281915 -19.434061 0.000000 5.158201 0.362969 -19.407831 -5.328743 3.117762 -0.196920 13.284378 0.000000 0.000000 0.000000 1.000000"],
    ["-3.722164 -1.751111 0.392928 -27.204948 0.000002 7.189400 0.142400 -3.968035 -5.328743 1.223165 -0.274463 18.712399 0.000000 0.000000 0.000000 1.000000"],
    ["-6.499726 -0.057722 -0.002499 0.474905 0.000001 4.084336 -0.419355 35.354855 0.059646 -6.290061 -0.272278 18.559431 0.000000 0.000000 0.000000 1.000000"],
    ["-6.499726 -0.068728 -0.000240 0.316786 0.000001 0.391987 -0.499317 40.952168 0.059646 -7.489434 -0.026131 1.329153 0.000000 0.000000 0.000000 1.000000"],
    ["-4.302106 3.945449 -0.267020 18.991407 -0.000001 5.343071 0.350882 -18.561745 4.872564 3.483536 -0.235759 16.003099 0.000000 0.000000 0.000000 1.000000"],
    ["-4.302106 1.414211 -0.362761 25.693283 -0.000000 7.258850 0.125770 -2.803936 4.872564 1.248641 -0.320291 21.920351 0.000000 0.000000 0.000000 1.000000"],
    ["-3.592302 -0.063625 0.416681 -28.867689 0.000001 7.499611 0.005090 5.643722 -5.417137 0.042194 -0.276317 18.842157 0.000000 0.000000 0.000000 1.000000"],
    ["-3.592302 3.070007 0.362977 -25.108427 0.000001 6.533027 -0.245579 23.190540 -5.417137 -2.035832 -0.240704 16.349255 0.000000 0.000000 0.000000 1.000000"],
    ["-4.405420 -2.257416 -0.335429 23.780056 -0.000000 6.842833 -0.204675 20.327229 4.779359 -2.080795 -0.309185 21.142960 0.000000 0.000000 0.000000 1.000000"],
    ["-4.405420 -4.470700 -0.215243 15.367023 -0.000001 4.391007 -0.405348 34.374352 4.779359 -4.120910 -0.198402 13.388168 0.000000 0.000000 0.000000 1.000000"],
    ["-6.349392 -1.308956 0.061936 -4.035504 -0.000001 4.340916 0.407739 -22.541719 -1.391122 5.974369 -0.282689 19.288223 0.000000 0.000000 0.000000 1.000000"],
    ["-6.349392 -0.669070 0.097270 -6.508885 0.000001 6.817382 0.208415 -8.589063 -1.391120 3.053792 -0.443961 30.577297 0.000000 0.000000 0.000000 1.000000"],
    ["-5.611467 2.101793 0.209866 -14.390611 0.000001 6.237510 -0.277637 25.434563 -3.280463 -3.595266 -0.358991 24.629345 0.000000 0.000000 0.000000 1.000000"],
    ["-5.611467 3.394199 0.111689 -7.518267 0.000001 3.319571 -0.448357 37.385010 -3.280463 -5.806019 -0.191053 12.873706 0.000000 0.000000 0.000000 1.000000"],
    ["-4.724389 1.390435 -0.330662 23.446341 -0.000000 7.221604 0.134964 -3.447470 4.464320 1.471434 -0.349925 23.994741 0.000000 0.000000 0.000000 1.000000"],
    ["-4.724390 -1.275813 -0.332709 23.589668 -0.000000 7.266322 -0.123838 14.668653 4.464319 -1.350136 -0.352092 24.146418 0.000000 0.000000 0.000000 1.000000"],
    ["-5.144022 -1.748446 0.282558 -19.479095 0.000001 6.933223 0.190676 -7.347342 -3.973541 2.263484 -0.365791 25.105392 0.000000 0.000000 0.000000 1.000000"],
    ["-5.144022 0.604988 0.302984 -20.908909 -0.000001 7.434419 -0.065977 10.618374 -3.973541 -0.783200 -0.392234 26.956383 0.000000 0.000000 0.000000 1.000000"],
    ["-6.284044 -0.946586 -0.111147 8.080308 -0.000001 6.522088 -0.246868 23.280769 1.661562 -3.579993 -0.420360 28.925194 0.000000 0.000000 0.000000 1.000000"],
    ["-6.284044 -1.653371 -0.064704 4.829248 -0.000001 3.796786 -0.431197 36.183784 1.661562 -6.253062 -0.244709 16.629658 0.000000 0.000000 0.000000 1.000000"],
    ["-6.310709 1.042817 -0.097548 7.128356 -0.000000 6.107615 0.290188 -14.313171 1.557224 4.226060 -0.395317 27.172167 0.000000 0.000000 0.000000 1.000000"],
    ["-6.310709 0.171497 -0.119240 8.646772 -0.000001 7.465759 0.047723 2.659365 1.557224 0.695003 -0.483223 33.325607 0.000000 0.000000 0.000000 1.000000"],
    ["-5.828282 0.396004 0.219781 -15.084682 -0.000000 7.446470 -0.059632 10.174221 -2.877695 -0.802040 -0.445130 30.659069 0.000000 0.000000 0.000000 1.000000"],
    ["-5.828282 1.991309 0.177136 -12.099519 0.000001 6.001594 -0.299858 26.990070 -2.877694 -4.033058 -0.358759 24.613125 0.000000 0.000000 0.000000 1.000000"],
    ["-6.010379 -0.084492 -0.190297 13.620784 -0.000001 7.496716 -0.014793 7.035545 2.474942 -0.205187 -0.462134 31.849415 0.000000 0.000000 0.000000 1.000000"],
    ["-6.010379 -1.500397 -0.161986 11.638994 0.000001 6.381398 -0.262702 24.389139 2.474944 -3.643702 -0.393381 27.036648 0.000000 0.000000 0.000000 1.000000"],
    ["-6.428964 -0.362217 0.069651 -4.575593 -0.000001 7.086210 0.163783 -5.464835 -0.958341 2.429902 -0.467251 32.207588 0.000000 0.000000 0.000000 1.000000"],
    ["-6.428964 0.208695 0.072394 -4.767558 -0.000001 7.365213 -0.094366 12.605642 -0.958341 -1.400027 -0.485648 33.495373 0.000000 0.000000 0.000000 1.000000"],
    ["-6.500000 0.000001 0.000000 0.300000 0.000001 7.471460 0.043578 2.949553 -0.000001 0.653667 -0.498097 34.366814 0.000000 0.000000 0.000000 1.000000"],
    ["-6.500000 0.000001 -0.000000 0.300000 0.000001 6.797308 -0.211309 20.791634 -0.000000 -3.169636 -0.453154 31.220776 0.000000 0.000000 0.000000 1.000000"],
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