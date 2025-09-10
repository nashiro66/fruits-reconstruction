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


def _update_light_position(params, sensor, light_positions):
  idx = int(sensor.id()[len('camera_') :])
  # Update light position for the selected sensor
  params['emit-Point.position'] = light_positions[idx]
  params.update()


def _gantry_light_positions():
  light_positions = [
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [27.055610795691802, 48.81986339495665, -31.610352043829998],
      [41.3180656055926, 48.73481598693242, -3.4462441460169386],
      [4.4671667393215575, 47.1369277980911, 42.78219333472567],
      [-42.91042711255302, 47.2987396659678, 3.9648433877314604],
      [-33.3138670015183, 47.353262970810604, -27.76333185326814],
      [-4.0928649644330495, 47.3407616054259, -43.412834290840195],
      [27.63531231151012, 47.2685587001079, -33.816397641445796],
      [43.2847288874415, 47.178949737549196, -4.59548434384601],
      [33.6881687764068, 47.1244264327064, 27.132690897152997],
      [33.6881687764068, 47.1244264327064, 27.132690897152997],
      [43.2847288874415, 47.178949737549196, -4.59548434384601],
      [27.63531231151012, 47.2685587001079, -33.816397641445796],
      [-4.0928649644330495, 47.3407616054259, -43.412834290840195],
      [-33.3138670015183, 47.353262970810604, -27.76333185326814],
      [-42.91042711255302, 47.2987396659678, 3.9648433877314604],
      [4.4671667393215575, 47.1369277980911, 42.78219333472567],
      [43.2160052121517, 47.0472861267224, -4.74011971286851],
      [4.61129874588445, 47.0047545442678, 42.71339045984761],
      [-27.110711342651502, 47.076678220198616, 33.2390870886382],
      [-42.842128441427, 47.1662190544112, 4.1088962216802996],
      [-33.3677017693281, 47.2209252406101, -27.613111405679],
      [-4.237421975160156, 47.20875063686579, -43.34461395103581],
      [27.4845881133758, 47.136826960935004, -33.870310579826004],
      [27.4845881133758, 47.136826960935004, -33.870310579826004],
      [-4.237421975160156, 47.20875063686579, -43.34461395103581],
      [-33.3677017693281, 47.2209252406101, -27.613111405679],
      [-42.842128441427, 47.1662190544112, 4.1088962216802996],
      [-27.110711342651502, 47.076678220198616, 33.2390870886382],
      [4.61129874588445, 47.0047545442678, 42.71339045984761],
      [43.2160052121517, 47.0472861267224, -4.74011971286851],
      [54.346621438817294, -0.786676450502069, 24.2689468400264],
      [-24.57995504968848, -0.724855391664175, 53.815623248847],
      [-55.69939927018449, -0.601267884668769, 20.471749327836],
      [-54.1263850774641, -0.503287498684284, -25.11075868475832],
      [-20.78236285181735, -0.48830981497388004, -56.230285802138],
      [24.80019141104173, -0.565108557522178, -54.657435093579295],
      [55.9196356315378, -0.688696064517584, -21.31356117256785],
      [55.9196356315378, -0.688696064517584, -21.31356117256785],
      [24.80019141104173, -0.565108557522178, -54.657435093579295],
      [-20.78236285181735, -0.48830981497388004, -56.230285802138],
      [-54.1263850774641, -0.503287498684284, -25.11075868475832],
      [-55.69939927018449, -0.601267884668769, 20.471749327836],
      [-24.57995504968848, -0.724855391664175, 53.815623248847],
      [54.346621438817294, -0.786676450502069, 24.2689468400264],
      [54.6571493603019, 4.41870496840774, 24.41354929025032],
      [39.6303138216189, 47.0780994649407, 17.415384223131202],
      [1.6598691824590588, 65.52149640366119, -0.2676721498317754],
      [-37.0116130568947, 48.94500399409722, -18.2771252295649],
  ]
  return [mi.Point3f(light_position) for light_position in light_positions]


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
  light_positions = _gantry_light_positions()

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
      for sensor, lights_ref_img in zip(
          sampled_view_sensors, sampled_view_lights_references
      ):
        assert len(lights_ref_img) == len(emitter_keys)
        ref_img = lights_ref_img[0]

        _update_light_position(params, sensor, light_positions)

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

              _update_light_position(params, sensor, light_positions)

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

          _update_light_position(params, sensor, light_positions)

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
