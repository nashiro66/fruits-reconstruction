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

import shutil
import os
import time
from typing import Any
from xml.etree import ElementTree

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from pathlib import Path
from core import mitsuba_io
from core import image_util
from core import losses
from core import mipmap_util
from core import mitsuba_util
from core import pyramid
from core import schedule
from practical_reconstruction import io_utils
from practical_reconstruction import scene_configuration
from variables import variable_types


def _add_textured_mipmap(
    scene_config, parent_bsdf, name, resolution, default_value, channel_count=3
):
  if len(default_value) != channel_count:
    raise ValueError(
        'Default value must have the same number of channels as the texture!'
    )
  # Create a dummy texture in the mitsuba texture folder to load it at scene
  # creation
  mts_scene_texture_folder = f'{scene_config.tmp_folder}/mts_scene/textures/'
  texture_path = f'{mts_scene_texture_folder}/init_{name}_{resolution}.exr'
  if not os.path.exists(texture_path):
    tensor = mi.TensorXf(
        np.zeros((resolution, resolution, channel_count), dtype=np.float32)
        + [[default_value]]
    )
    mitsuba_io.write_bitmap(mi.Bitmap(tensor), texture_path)

  mipmap_texture = ElementTree.Element(
      'texture', {'type': 'mipmap_flat', 'name': name}
  )
  # Create the nested bitmap texture
  nested_bitmap = ElementTree.SubElement(
      mipmap_texture, 'texture', {'type': 'bitmap', 'name': 'nested_bitmap'}
  )
  ElementTree.SubElement(
      nested_bitmap, 'string', {'name': 'format', 'value': 'variant'}
  )
  ElementTree.SubElement(
      nested_bitmap,
      'string',
      {'name': 'filename', 'value': f'textures/init_{name}_{resolution}.exr'},
  )
  ElementTree.SubElement(
      nested_bitmap, 'boolean', {'name': 'raw', 'value': 'true'}
  )
  if (
      name == 'single_scattering_albedo'
      and scene_config.approach != scene_configuration.Approach.NAIVE
  ):
    volume_albedo_remap = ElementTree.Element(
        'boolean',
        attrib={
            'name': 'volume_albedo_remap',
            'value': (
                'true' if scene_config.sss_volume_albedo_remapping else 'false'
            ),
        },
    )
    mipmap_texture.append(volume_albedo_remap)

  parent_bsdf.append(mipmap_texture)


def transform_xml_for_sss(
    root: ElementTree.Element | Any,
    scene_config: scene_configuration.SceneConfig,
):
  """Transforms the Mitsuba XML to enable SSS optimization.

  This function modifies the Mitsuba XML by changing the type of the principled
  BSDF to uber_principled and adding the single_scattering_albedo,
  extinction_coefficient, and hg_coefficient parameters.

  Args:
    root: The root element of the Mitsuba XML tree.
    scene_config: The scene configuration.
  """
  # Nest all sss_bsdf bsdfs with a diffuse_switch bsdf and add
  # volume_albedo_remap
  for sss_bsdf in root.findall(".//bsdf[@type='sss_bsdf']"):
    # Add volume_albedo_remap to all base_color mipmap_flat textures
    found_mipmapped_sss_albedo = False
    found_mipmapped_base_color = False
    for mipmap_texture in sss_bsdf.findall(
        ".//texture[@type='mipmap_flat'][@name='single_scattering_albedo']"
    ):
      found_mipmapped_sss_albedo = True
      # Create a new volume_albedo_remap element
      volume_albedo_remap = ElementTree.Element(
          'boolean',
          attrib={
              'name': 'volume_albedo_remap',
              'value': (
                  'true'
                  if scene_config.sss_volume_albedo_remapping
                  else 'false'
              ),
          },
      )
      mipmap_texture.append(volume_albedo_remap)

      if mipmap_texture.attrib.get('name') == 'base_color':
        found_mipmapped_base_color = True
        # Remove the base_color texture and replace it with a base color of 1
        sss_bsdf.remove(mipmap_texture)
        ElementTree.SubElement(
            sss_bsdf,
            'rgb',
            {'name': 'base_color', 'value': '1, 1, 1'},
        )

    nested_principled_bsdf = sss_bsdf.find("./bsdf[@type='principled'][@name='nested_bsdf']")
    if nested_principled_bsdf is None:
      raise ValueError("Expecting a nested principled bsdf for the sss_bsdf!")
  
    for mipmap_texture in nested_principled_bsdf.findall(
        ".//texture[@type='mipmap_flat'][@name='base_color']"
    ):
      found_mipmapped_base_color = True
      # Remove the base_color texture and replace it with a base color of 1
      nested_principled_bsdf.remove(mipmap_texture)
      ElementTree.SubElement(
          nested_principled_bsdf,
          'rgb',
          {'name': 'base_color', 'value': '1, 1, 1'},
      )

    if not found_mipmapped_base_color:
      base_color_element = nested_principled_bsdf.find("*[@name='base_color']")
      if base_color_element is not None:
        nested_principled_bsdf.remove(base_color_element)
      ElementTree.SubElement(
          nested_principled_bsdf,
          'rgb',
          {'name': 'base_color', 'value': '1, 1, 1'},
      )

    if (
        not found_mipmapped_sss_albedo
        or scene_config.sss_albedo_resolution is not None
    ):
      # Remove the component and replace it with a mipmap_flat texture
      single_scattering_albedo_element = sss_bsdf.find(
          "*[@name='single_scattering_albedo']"
      )
      if single_scattering_albedo_element is None:
        raise ValueError(
            'single_scattering_albedo is missing from the sss_bsdf bsdf!'
        )
      # A bit tricky here to account for all cases, basically if
      # sss_albedo_resolution is None we add a minmal 2x2 texture, otherwise if
      # it is exactly 1 nothing should be done.
      if (
          scene_config.sss_albedo_resolution is None
          or scene_config.sss_albedo_resolution > 1
      ):
        value = single_scattering_albedo_element.attrib.get('value')
        if value is None:
          if scene_config.sss_albedo_resolution is not None:
            value = '0.75 0.75 0.75'
          else:
            raise ValueError(
                'single_scattering_albedo must have an rgb value if not a'
                ' texture!'
            )

        sss_bsdf.remove(single_scattering_albedo_element)
        sss_albedo_value = [float(x) for x in value.split(' ')]
        if len(sss_albedo_value) != 3:
          raise ValueError(
              'single_scattering_albedo must have an rgb value if not a'
              ' texture!'
          )
        _add_textured_mipmap(
            scene_config,
            sss_bsdf,
            'single_scattering_albedo',
            2
            if not found_mipmapped_base_color
            else scene_config.sss_albedo_resolution,
            default_value=sss_albedo_value,
            channel_count=len(sss_albedo_value),
        )

    extinction_coefficient_element = sss_bsdf.find(
        "*[@name='extinction_coefficient']"
    )
    if len(scene_config.sss_extinction_coefficient) not in [1, 3]:
      raise ValueError(
          'extinction_coefficient must be a list of one scalar or three rgb'
          ' values!'
      )
    # Add scalar or textured extinction coefficient
    if scene_config.sss_extinction_resolution > 1:
      if extinction_coefficient_element is None:
        _add_textured_mipmap(
            scene_config,
            sss_bsdf,
            'extinction_coefficient',
            scene_config.sss_extinction_resolution,
            default_value=scene_config.sss_extinction_coefficient,
            channel_count=len(scene_config.sss_extinction_coefficient),
        )
      else:
        if extinction_coefficient_element.attrib.get('type') != 'mipmap_flat':
          raise ValueError(
              'extinction_coefficient must be a mipmap_flat texture!'
          )
    else:
      if extinction_coefficient_element is None:
        extinction_coefficient = ElementTree.Element(
            'spectrum', {'type': 'srgb', 'name': 'extinction_coefficient'}
        )
        ElementTree.SubElement(
            extinction_coefficient,
            'rgb',
            {
                'name': 'color',
                'value': ' '.join(
                    [str(x) for x in scene_config.sss_extinction_coefficient]
                ),
            },
        )
        ElementTree.SubElement(
            extinction_coefficient,
            'boolean',
            {'name': 'unbounded', 'value': 'true'},
        )
        sss_bsdf.append(extinction_coefficient)
      else:
        if extinction_coefficient_element.attrib.get('type') != 'srgb':
          raise ValueError(
              'extinction_coefficient must be an (unbounded) srgb spectrum!'
          )

    hg_coefficient_element = sss_bsdf.find(
        "*[@name='hg_coefficient']"
    )
    if len(scene_config.sss_hg_coefficient) not in [1, 3]:
      raise ValueError(
          'hg_coefficient must be a list of one scalar or three rgb values!'
      )
    if scene_config.sss_hg_resolution > 1:
      if hg_coefficient_element is None:
        _add_textured_mipmap(
            scene_config,
            sss_bsdf,
            'hg_coefficient',
            scene_config.sss_hg_resolution,
            default_value=scene_config.sss_hg_coefficient,
            channel_count=len(scene_config.sss_hg_coefficient),
        )
      else:
        if hg_coefficient_element.attrib.get('type') != 'mipmap_flat':
          raise ValueError('hg_coefficient must be a mipmap_flat texture!')
    else:
      if hg_coefficient_element is None:
        hg_coefficient = ElementTree.Element(
            'spectrum', {'type': 'srgb', 'name': 'hg_coefficient'}
        )
        ElementTree.SubElement(
            hg_coefficient,
            'rgb',
            {
                'name': 'color',
                'value': ' '.join(
                    [str(x) for x in scene_config.sss_hg_coefficient]
                ),
            },
        )
        ElementTree.SubElement(
            hg_coefficient,
            'boolean',
            {'name': 'unbounded', 'value': 'true'},
        )
        sss_bsdf.append(hg_coefficient)
      else:
        if hg_coefficient_element.attrib.get('type') != 'srgb':
          raise ValueError(
              'hg_coefficient must be an (unbounded) srgb spectrum!'
          )

    if (
        scene_config.sss_diffuse_switch
        != scene_configuration.DiffuseSwitch.NONE
    ):
      new_sss_principled = ElementTree.Element(
          'bsdf',
          {'type': 'sss_bsdf', 'name': 'sss_bsdf'},
      )
      for element in list(sss_bsdf):
        new_sss_principled.append(element)
        sss_bsdf.remove(element)

      sss_bsdf.set('type', 'diffuse_switch')
      diffuse_switch_bsdf = sss_bsdf
      diffuse_mode = ElementTree.Element(
          'string',
          attrib={
              'name': 'diffuse_mode',
              'value': scene_config.sss_diffuse_switch.name.lower(),
          },
      )
      diffuse_switch_bsdf.append(diffuse_mode)
      diffuse_switch_bsdf.append(new_sss_principled)


def load_mitsuba_scene(scene_config, tmp_mitsuba_xml):
  # TODO(pweier) only transform into mipmaps if the optimized materials are mipmaps

  modified_xml = tmp_mitsuba_xml
  tree = ElementTree.parse(modified_xml)
  root = tree.getroot()

  if scene_config.sss_optimization:
    transform_xml_for_sss(root, scene_config)

  if scene_config.approach not in [
      scene_configuration.Approach.MIPMAP,
      scene_configuration.Approach.MIPMAP_PYRAMID,
      scene_configuration.Approach.LARGE_STEPS_MIPMAP,
  ]:
    # Replace all occurrences of mipmap_flat with normal bitmaps
    # Iterate over all <texture> elements with type="mipmap_flat"
    for mipmap_texture in root.findall(".//texture[@type='mipmap_flat']"):
      nested_texture = mipmap_texture.find("texture[@type='bitmap']")
      # Ensure the nested <texture> exists
      if nested_texture is not None:
        # Find the <string> element inside the nested <texture>
        filename_element = nested_texture.find("string[@name='filename']")
        format_element = nested_texture.find("string[@name='format']")
        raw_element = nested_texture.find("boolean[@name='raw']")
        gpu_accel = ElementTree.Element('boolean')
        gpu_accel.set('name', 'accel')
        gpu_accel.set('value', 'false')
        mipmap_texture.remove(nested_texture)
        mipmap_texture.set('type', 'bitmap')
        if filename_element is not None:
          mipmap_texture.append(filename_element)
        if format_element is not None:
          mipmap_texture.append(format_element)
        if raw_element is not None:
          mipmap_texture.append(raw_element)
        mipmap_texture.append(gpu_accel)
  else:
    # Add mip_bias to all mipmap_flat textures
    for mipmap_texture in root.findall(".//texture[@type='mipmap_flat']"):
      bias_exists = any(
          child.tag == 'float' and child.attrib.get('name') == 'mip_bias'
          for child in root
      )
      if not bias_exists:
        mip_bias = ElementTree.Element(
            'float', name='mip_bias', value=str(scene_config.mip_bias)
        )
        mipmap_texture.append(mip_bias)
      else:
        print(
            f'mip_bias already exists for {mipmap_texture.attrib.get("name")}!'
        )
      min_resolution_exists = any(
          child.tag == 'integer'
          and child.attrib.get('name') == 'min_resolution'
          for child in root
      )
      if not min_resolution_exists:
        min_resolution = ElementTree.Element(
            'integer',
            name='min_resolution',
            value=str(scene_config.mip_min_res),
        )
        mipmap_texture.append(min_resolution)
      else:
        print(
            'min_resolution already exists for'
            f' {mipmap_texture.attrib.get("name")}!'
        )

  if scene_config.scene_setup == scene_configuration.Setup.OLAT:
    # Replace all emitters by nesting them in a switch emitter
    # Iterate through all shape emitters and wrap them in a switch emitter
    for shape in root.findall('.//shape'):
      for emitter in shape.findall('emitter'):
        # Create a new switchemitter element
        switchemitter = ElementTree.Element(
            'emitter',
            attrib={'type': 'switchemitter', 'name': emitter.get('name')},
        )

        # Create a nested emitter inside switchemitter
        nested_emitter = ElementTree.Element(
            'emitter',
            attrib={'type': emitter.get('type'), 'name': 'nested_emitter'},
        )

        # Move all children of the original emitter to the nested emitter
        nested_emitter.extend(emitter)

        # Add the nested emitter to switchemitter
        switchemitter.append(nested_emitter)

        # Replace the original emitter with switchemitter
        shape.remove(emitter)
        shape.append(switchemitter)
    # Iterate through all point emitters and wrap them in a switch emitter
    for emitter in root.findall('.//emitter'):
      if emitter.get('type') == 'point':
        # Create a new switchemitter element
        switchemitter = ElementTree.Element(
            'emitter',
            attrib={
                'type': 'switchemitter',
                'id': emitter.get('id'),
                'name': emitter.get('name'),
            },
        )
        # Create a nested emitter inside switchemitter
        nested_emitter = ElementTree.Element(
            'emitter',
            attrib={'type': emitter.get('type'), 'name': 'nested_emitter'},
        )
        # Move all children of the original emitter to the nested emitter
        nested_emitter.extend(emitter)
        # Add the nested emitter to switchemitter
        switchemitter.append(nested_emitter)
        # Replace the original emitter with switchemitter
        root.remove(emitter)
        root.append(switchemitter)

  modified_xml = f'{tmp_mitsuba_xml[:-4]}_modified.xml'
  tree.write(modified_xml, encoding='utf-8', xml_declaration=True)

  scene = mi.load_file(modified_xml)
  params = mi.traverse(scene)
  # Make sure that we call prepare() on any emitters that have
  # such a function exposed.
  for emitter in scene.emitters():
    if hasattr(emitter, 'prepare'):
      emitter.prepare()
      assert emitter.is_prepared

  if scene_config.sss_optimization:
    # Set the albedo scaling for each material
    for mat_key in scene_config.per_material_sss_albedo_scaling.keys():
      prefix = ''
      for param_key in params.keys():
        if f'{mat_key}.' in param_key and 'nested_bsdf' in param_key:
          prefix = 'nested_bsdf.'
          break
      if (
          f'{mat_key}.{prefix}single_scattering_albedo.data.flat_buffer'
          in params
      ):
        param_key = (
            f'{mat_key}.{prefix}single_scattering_albedo.data.flat_buffer'
        )
      elif f'{mat_key}.{prefix}single_scattering_albedo.data' in params:
        param_key = f'{mat_key}.{prefix}single_scattering_albedo.data'
      elif f'{mat_key}.{prefix}single_scattering_albedo.value' in params:
        param_key = f'{mat_key}.{prefix}single_scattering_albedo.value'
      else:
        raise ValueError(
            f'Material {mat_key} does not have a single scattering albedo!'
        )
      params[param_key] = (
          params[param_key]
          * scene_config.per_material_sss_albedo_scaling[mat_key]
      )
    params.update()

  return scene


def get_param_type_from_mipmap_or_bitmap_param(key):
  if key.endswith('.flat_buffer'):
    param_type = key.split('.')[-3]
  else:
    param_type = key.split('.')[-2]
  return param_type


def get_param_type_from_scalar_param(key):
  # Assuming key ends in .value
  return key.split('.')[-2]


def get_shape_to_material_dict(shapes, optimized_shapes=None):
  shape_to_material = {}
  for shape in shapes:
    if optimized_shapes is None or shape.id() in optimized_shapes:
      shape_to_material[shape.id()] = shape.bsdf().id()
  return shape_to_material


def format_float(f):
  """Formats a float such that 0.1 becomes "0_1", 10.0 becomes "10_0", etc."""
  return str(f).replace('.', '_')


def get_emitter_keys(scene_config, params):
  """Returns a list of emitter keys that can be turned on or off."""
  if scene_config.scene_setup == scene_configuration.Setup.OLAT:
    emitter_keys = []
    for k in params.keys():
      if (
          k.startswith('emit-')
          and 'nested_emitter' not in k
          and '.sampling_weight' in k
      ):
        emitter_keys.append('.'.join(k.split('.')[:-1]))
    if not emitter_keys:
      raise ValueError('OLAT setup requires switchable emitters in the scene!')
  elif scene_config.scene_setup == scene_configuration.Setup.ENVMAP_ROTATIONS:
    emitter_keys = []
    has_world_envmap = False
    for k in params.keys():
      if k == 'World.to_world':
        has_world_envmap = True
        break
    if not has_world_envmap:
      raise ValueError('ENVMAP_ROTATIONS setup requires a world envmap!')
    for rotation in scene_config.envmap_rotations:
      stringrotation = format_float(rotation)
      emitter_keys.append(f'World_rot_{stringrotation}')
  else:
    # Dummy key for consistency with OLAT setup reference generation
    emitter_keys = ['FULL_ON']
  return emitter_keys


def switch_envmap_rotation(scene_config, params, emitter_idx):
  if scene_config.scene_setup != scene_configuration.Setup.ENVMAP_ROTATIONS:
    raise ValueError('switch_envmap_rotation called with wrong setup!')
  rotation = scene_config.envmap_rotations[emitter_idx]
  # 90 degrees is the default when exporting from Blender
  params['World.to_world'] = (
      mi.Transform4f().rotate([0, 1, 0], 90).rotate([0, 1, 0], rotation)
  )
  params.update()


def switch_emitter(params, emitter_key, emitter_keys, full_on=False):
  for k in emitter_keys:
    if k == emitter_key or full_on:
      # Turn on the one selected light.
      params[f'{k}.sampling_weight'] = 1.0
      params[f'{k}.nested_emitter.sampling_weight'] = 1.0
    else:
      # Turn off all other emitters
      params[f'{k}.sampling_weight'] = 0.0
      params[f'{k}.nested_emitter.sampling_weight'] = 0.0
  params.update()


def generate_references_and_retrieve_sensors(scene, scene_config, emitter_keys):
  tmp_reference_folder = Path(f'{scene_config.tmp_folder}/references')
  remote_reference_folder = (
      Path(scene_config.scene_folder) / 'references'
  )
  remote_reference_folder.mkdir(parents=True,exist_ok=True)
  tmp_reference_folder.mkdir(parents=True,exist_ok=True)
  # Output paths are a 2d array where rows are sensors and columns are emitters
  output_paths = []

  sensors = scene.sensors()
  params = mi.traverse(scene)

  copy_local_to_remote = False

  if (
      not scene_config.use_mitsuba_reference
      and scene_config.scene_setup != scene_configuration.Setup.OLAT
  ):
    raise ValueError(
        'Only mitsuba references are supported right now'
    )
  else:
    if not scene_config.use_mitsuba_reference:
      raise ValueError('OLAT setup requires mitsuba to generate references!')

    # Keep only the specified sensor indices
    sensors = [(sensors[i]) for i in scene_config.optimized_sensors_indices]

    for tmp_cam_idx, sensor in enumerate(sensors):
      print(f'Rendering camera {sensor.id()}')
      per_sensor_output_paths = []
      if scene_config.scene_setup == scene_configuration.Setup.OLAT:
        emitter_keys_full_on = emitter_keys + ['FULL_ON']
      else:
        emitter_keys_full_on = emitter_keys
      for emitter_idx, emitter_key in enumerate(emitter_keys_full_on):
        cam_idx = scene_config.optimized_sensors_indices[tmp_cam_idx]
        full_on_reference = (
            emitter_key == 'FULL_ON'
            and scene_config.scene_setup == scene_configuration.Setup.OLAT
        )

        if emitter_key != 'FULL_ON':
          output_name = f'ref_view_{cam_idx:03d}_emitter_{emitter_idx:03d}.exr'
          if (
              scene_config.scene_setup
              == scene_configuration.Setup.ENVMAP_ROTATIONS
          ):
            switch_envmap_rotation(scene_config, params, emitter_idx)
          else:
            switch_emitter(params, emitter_key, emitter_keys)
          print(f'with light {emitter_key} and index {emitter_idx:03d}')
        else:
          if (
              scene_config.scene_setup
              == scene_configuration.Setup.ENVMAP_ROTATIONS
          ):
            raise ValueError('ENVMAP_ROTATIONS has incorrect emitter keys!')
          if scene_config.scene_setup == scene_configuration.Setup.OLAT:
            output_name = f'ref_view_{cam_idx:03d}_full_on.exr'
            switch_emitter(params, emitter_key, emitter_keys, full_on=True)
            print('with all light on')
          else:
            output_name = f'ref_view_{cam_idx:03d}.exr'
        output_path = Path(tmp_reference_folder) / output_name
        output_path_remote = Path(remote_reference_folder) / output_name

        # Only render if the file does not exist locally or remotely
        if not output_path_remote.exists() and not output_path.exists():
          with dr.suspend_grad():
            if scene_config.sss_optimization:
              integrator = mi.load_dict({
                  'type': 'prb_path_volume',
                  'max_sss_depth': -1,
                  'max_path_depth': scene_config.optimized_path_depth,
              })
            else:
              integrator = mi.load_dict({
                  'type': 'path',
                  'max_depth': scene_config.optimized_path_depth,
              })

            image = mi.render(
                scene,
                integrator=integrator,
                sensor=sensor,
                params=params,
                spp=scene_config.reference_spp,
            )
            mitsuba_io.write_bitmap(mi.Bitmap(image), output_path)
        else:
          if output_path_remote.exists() and not output_path.exists():
            print(
                f'Reference found at: {output_path_remote}, copying to'
                ' tmp folder'
            )

            if output_path.exists():
                os.remove(output_path)
            try:
                shutil.copy(output_path_remote, output_path)
            except Exception as e:
                print(f"Error copying directory: {e}")
          elif output_path.exists():
            print(f'Reference found locally: {output_path}')
          else:
            raise ValueError(
                f'Reference not found remotely! {output_path_remote}'
            )
        copy_local_to_remote = not output_path_remote.exists()

        # Make a png copy for easy viewing from the remote or local file to local
        if output_path_remote.with_suffix('.png').exists():
          os.remove(output_path_remote.with_suffix('.png'))
        ref_exr = mitsuba_io.read_bitmap(output_path)
        ref_png = image_util.tonemap(ref_exr)
        mitsuba_io.write_bitmap(ref_png, f'{str(output_path)[:-3]}png')
        shutil.copy(f'{str(output_path)[:-3]}png', output_path_remote.with_suffix('.png'))
          
        # Don't use the full on reference for optimization
        if not full_on_reference:
          per_sensor_output_paths.append(output_path)
      output_paths.append(per_sensor_output_paths)

  # In case the reference was generated locally, copy it to the remote folder
  # We do this always, even if the reference was already present remotely,
  # that way we can overwrite it if it was generated locally anew.
  # if copy_local_to_remote:
  #   io_utils.reference_output_local_to_remote(scene_config)

  references = []
  # for emitter_key, per_sensor_output_paths in zip(emitter_keys, output_paths):
  for sensor_idx, per_sensor_output_paths in enumerate(output_paths):
    references.append([
        mi.TensorXf(bitmap)
        for bitmap in mitsuba_io.read_images(
            per_sensor_output_paths,
            f'Loading references for sensor {sensors[sensor_idx].id()}',
        )
    ])

  if len(references) != len(sensors):
    raise ValueError('Mismatched number of references and sensors!')
  if len(references[0]) != len(emitter_keys):
    raise ValueError('Mismatched number of references and emitter keys!')

  return sensors, references


def create_intermediate_resolution(scene_config, sensors, references):
  all_sensors = [sensors]
  all_references = [references]

  for level in range(1, scene_config.n_resolutions):
    downscaled_sensors = []
    downscaled_references = []
    for i, (sensor, light_references) in enumerate(zip(sensors, references)):
      sensor_light_references = []
      for light_reference in light_references:
        width = light_reference.shape[1]
        float_width = float(width / 2**level)
        new_width = int(float_width)
        if float(new_width) != float_width:
          raise ValueError(
              'Resolutions should be divisible by powers of two but got width'
              f' {float_width} which is not an integer.'
          )
        # Create new reference
        ref = image_util.resize_to_width(light_reference, new_width)
        sensor_light_references.append(ref)
      downscaled_references.append(sensor_light_references)

      # Create new sensor
      sensor_id = sensors[i].id()
      sensor_height, sensor_width = sensor_light_references[0].shape[:2]
      sensor = mitsuba_util.copy_sensor(sensor)
      mitsuba_util.set_sensor_resolution(sensor, (sensor_width, sensor_height))
      sensor.set_id(sensor_id)
      downscaled_sensors.append(sensor)

    all_sensors.insert(0, downscaled_sensors)
    all_references.insert(0, downscaled_references)
  return all_sensors, all_references


def get_scene_keys_for_optimization(scene, scene_config):
  params = mi.traverse(scene)
  shape_to_material = get_shape_to_material_dict(
      scene.shapes(), scene_config.optimized_shapes
  )
  # Only keep bsdf params with mipmapped textures and non-zero learning rate
  optimized_keys = []
  for key in params.keys():
    for material_name in shape_to_material.values():
      if (
          key.startswith(f'{material_name}.')
          and (key.endswith('.data.flat_buffer') or key.endswith('.data'))
          and key not in optimized_keys
      ):
        param_type = get_param_type_from_mipmap_or_bitmap_param(key)
        if (
            scene_config.per_material_learning_rates[material_name][param_type]
            != 0.0
        ):
          optimized_keys.append(key)
      # Add scalar parameters for sss params only
      elif (
          key.startswith(f'{material_name}.')
          and key.endswith('.value')
          and (
              'hg_coefficient' in key
              or 'extinction_coefficient' in key
              or 'single_scattering_albedo' in key
          )
      ):
        param_type = get_param_type_from_scalar_param(key)
        if (
            scene_config.per_material_learning_rates[material_name][param_type]
            != 0.0
        ):
          optimized_keys.append(key)

  return optimized_keys


def _get_material_learning_rate(scene_config, key, param_type):
  material_name = key.split('.')[0]
  return scene_config.per_material_learning_rates[material_name][param_type]


def initialize_optimized_parameter(scene_config, params, optimized_keys):
  for key in params.keys():
    if key in optimized_keys:
      if key.endswith('.value'):
        param_type = get_param_type_from_scalar_param(key)
      else:
        param_type = get_param_type_from_mipmap_or_bitmap_param(key)
      material_name = key.split('.')[0]
      default_value = scene_config.per_material_initial_values[material_name][
          param_type
      ]
      # Ensure mitsuba types (only Color3f and Float supported for now)
      if isinstance(default_value, list) and len(default_value) == 3:
        default_value = mi.Color3f(default_value)
      elif isinstance(default_value, float):
        default_value = mi.Float(default_value)
      else:
        raise ValueError(
            f'Unsupported default value {default_value} for type {param_type}!'
        )
      print(
          f"Optimizing {material_name}'s {param_type} from default value :"
          f' {default_value}'
      )
      if isinstance(params[key], mi.TensorXf):
        params[key] = params[key] * 0.0 + mi.TensorXf(
            dr.ravel(default_value), shape=(1, 1, dr.shape(default_value)[0])
        )
      else:
        params[key] = params[key] * 0.0 + default_value

  params.update()


def _get_initial_texture_params(scene_config, params, key, level=0):
  if scene_config.approach in [
      scene_configuration.Approach.MIPMAP,
      scene_configuration.Approach.MIPMAP_PYRAMID,
      scene_configuration.Approach.LARGE_STEPS_MIPMAP,
  ]:
    texture_data_key = key.replace('.flat_buffer', '')
    return mipmap_util.mip_tensor_from_flat_buffer(
        level,
        params[f'{texture_data_key}.flat_buffer'],
        params[f'{texture_data_key}.flat_buffer_offsets'],
        params[f'{texture_data_key}.base_mip_shape'],
        params[f'{texture_data_key}.mip_factor'],
    )
  else:
    return params[key]


def _get_initial_scalar_params(params, key):
  return params[key]


def create_displacement_variable(
    mesh_name: str,
    displacement_scale: float,
    learning_rate: float,
    texture_res: tuple[int, int],
):
  initial_value = np.zeros((texture_res[0], texture_res[1], 1))
  nested_variable = variable_types.TensorVariable(
      key='mesh_displacement_map_texture',
      initial_value=initial_value,
      clamp_range=(-1, 1),
      learning_rate=learning_rate,
      shape=initial_value.shape,
      is_scene_parameter=False,
  )
  optimized_scene_key = f'mesh-{mesh_name}.vertex_positions'
  variable = variable_types.DisplacementMapVariable(
      key=optimized_scene_key,
      nested_variable=nested_variable,
      displacement_scale=displacement_scale,
      is_scene_parameter=True,
  )
  return variable, optimized_scene_key


def pyramid_regularisation(py: pyramid.ImagePyramid):
  divisor = py.factor ** (py.n_levels - 1)
  target_res = (
      py.shape[0] // divisor,
      py.shape[1] // divisor,
  )
  error = 0.0
  result = 0.0
  prev_result = py.pyramid[0]
  for i in range(py.n_levels):
    result = result + py.pyramid[i]
    if i != 0:
      downsampled_result = losses._downsample_image(result)
      # error += dr.mean(dr.square(prev_result - downsampled_result), axis=None)
      error += dr.mean(dr.abs(prev_result - downsampled_result), axis=None)
    prev_result = mi.TensorXf(result)
    if i == py.n_levels - 1:
      break
    target_res = (
        int(target_res[0] * py.factor),
        int(target_res[1] * py.factor),
    )
    result = py._upsample(result, target_res)
  print(error)

  return error


def create_texture_variable(
    scene_config,
    params,
    key: str,
    initial_value: np.ndarray | mi.TensorXf,
    clamp_range: tuple[float, float] | None,
    learning_rate: float = 1.0,
    normal_clamping: bool = False,
    optimizer_key: str | None = None,
    normal_perturbation: bool = False,
    scalar_perturbation: bool = False,
) -> variable_types.Variable:
  """Set up texture optimization variable."""

  if normal_perturbation:
    normals_flat = dr.unravel(mi.Normal3f, initial_value.array) * 2 - 1
    strength = 0.1
    noise_tangent = np.random.normal(0, strength, size=normals_flat.shape)
    noise_bitangent = np.random.normal(0, strength, size=normals_flat.shape)
    # Add a small perturbation around the normal in the tangent plane
    perturbed = (
        normals_flat
        + mi.Normal3f([1, 0, 0]) * noise_tangent
        + mi.Normal3f([0, 1, 0]) * noise_bitangent
    )
    # Normalize and map back to [0, 1]
    perturbed = 0.5 * (dr.normalize(perturbed) + 1.0)
    initial_value = mi.TensorXf(
        dr.ravel(perturbed),
        shape=initial_value.shape,
    )
  elif scalar_perturbation:
    noise = np.random.random(initial_value.shape)
    initial_value += mi.TensorXf(noise) * 0.2
    initial_value = dr.clip(initial_value, 0.005, 0.5)

  if scene_config.approach == scene_configuration.Approach.NAIVE:
    if scene_config.n_texture_scales > 1:
      if initial_value.shape[0] % 2 != 0 or initial_value.shape[1] % 2 != 0:
        raise ValueError('Texture scales only supported for even image sizes!')
      texture_scales = min(
          int(dr.log2i(min(initial_value.shape[0], initial_value.shape[1]))),
          scene_config.n_texture_scales,
      )
    else:
      texture_scales = 1
    upsampling_schedule = schedule.exponential(
        max(initial_value.shape[0], initial_value.shape[1]),
        2,
        texture_scales,
        scene_config.n_iter,
    )
    return variable_types.TensorVariable(
        key=key,
        initial_value=initial_value,
        clamp_range=clamp_range,
        shape=initial_value.shape,
        learning_rate=learning_rate,
        upsampling_schedule=(
            upsampling_schedule if scene_config.n_texture_scales > 1 else None
        ),
        is_scene_parameter=True,
    )
  elif scene_config.approach == scene_configuration.Approach.MIPMAP:
    texture_key = key.replace('.flat_buffer', '')
    max_mipmap_levels = len(params[f'{texture_key}.flat_buffer_offsets']) - 1
    if scene_config.n_texture_scales > 1:
      if initial_value.shape[0] % 2 != 0 or initial_value.shape[1] % 2 != 0:
        raise ValueError('Texture scales only supported for even image sizes!')
      texture_scales = min(
          int(dr.log2i(min(initial_value.shape[0], initial_value.shape[1]))),
          scene_config.n_texture_scales,
      )
    else:
      texture_scales = 1
    if texture_scales > max_mipmap_levels:
      print(int(dr.log2i(min(initial_value.shape[0], initial_value.shape[1]))))
      print(min(initial_value.shape[0], initial_value.shape[1]))
      print(scene_config.n_texture_scales)
      raise ValueError(
          f'Number of texture scales ({texture_scales}) should'
          ' be smaller than the number of mipmap levels'
          f' ({max_mipmap_levels}) for differentiable mipmaps!'
      )
    upsampling_schedule = schedule.exponential(
        initial_value.shape[0],
        2,
        texture_scales,
        scene_config.n_iter,
    )
    return variable_types.DifferentiableMipmapVariable(
        key=texture_key,
        nested_tensor=variable_types.TensorVariable(
            key=texture_key.replace('.data', '.mipmapped_tensor'),
            initial_value=initial_value,
            clamp_range=clamp_range,
            shape=initial_value.shape,
            learning_rate=learning_rate,
            upsampling_schedule=(
                upsampling_schedule
                if scene_config.n_texture_scales > 1
                else None
            ),
            is_scene_parameter=False,
        ),
        is_scene_parameter=True,
    )
  elif scene_config.approach == scene_configuration.Approach.MIPMAP_PYRAMID:
    return variable_types.FlatMipAwareImagePyramidVariable(
        key=key.replace('.flat_buffer', ''),
        optimizer_key=optimizer_key,
        initial_value=initial_value,
        clamp_range=clamp_range,
        shape=None,
        mipmapped=True,
        normal_clamping=normal_clamping,
        ensure_frequency_decomposition=scene_config.ensure_frequency_decomposition,
        learning_rate=learning_rate,
        n_levels=None,
        factor=None,
        is_scene_parameter=True,
        regularization_fn=(
            pyramid_regularisation
            if scene_config.laplacian_pyramid_regularisation
            else None
        ),
        regularization_weight=scene_config.regularisation_weight,
    )
  elif scene_config.approach == scene_configuration.Approach.LARGE_STEPS:
    return variable_types.LaplacianSmoothingVariable(
        key=key.replace('.flat_buffer', ''),
        initial_value=initial_value,
        clamp_range=clamp_range,
        lambda_=scene_config.large_steps_lambda,
        use_conjugate_gradient=scene_config.use_conjugate_gradient_large_steps,
        normal_clamping=normal_clamping,
        learning_rate=learning_rate,
        is_scene_parameter=True,
    )
  elif scene_config.approach == scene_configuration.Approach.LARGE_STEPS_MIPMAP:
    texture_key = key.replace('.flat_buffer', '')
    if scene_config.n_texture_scales != 1:
      raise ValueError('Mipmapped large steps does not support texture scales!')
    return variable_types.DifferentiableMipmapVariable(
        key=texture_key,
        nested_tensor=variable_types.LaplacianSmoothingVariable(
            key=texture_key.replace('.data', '.mipmapped_tensor'),
            initial_value=initial_value,
            clamp_range=clamp_range,
            lambda_=scene_config.large_steps_lambda,
            use_conjugate_gradient=scene_config.use_conjugate_gradient_large_steps,
            normal_clamping=normal_clamping,
            learning_rate=learning_rate,
            is_scene_parameter=False,
        ),
        is_scene_parameter=True,
    )
  else:
    raise ValueError(f'Approach {scene_config.approach} is not supported yet!')


def create_variables(
    params,
    optimized_keys,
    scene_config,
):
  variables = []
  for key in optimized_keys:
    # Scalar parameters
    if key.endswith('.value'):
      param_type = get_param_type_from_scalar_param(key)
      initial_value = _get_initial_scalar_params(params, key)
      if (
          scene_config.sss_uniform_extinction
          and param_type == 'extinction_coefficient'
      ) or (scene_config.sss_uniform_hg and param_type == 'hg_coefficient'):
        initial_value = initial_value.x
        print(f'Uniform {param_type} set at {initial_value}')
      learning_rate = (
          scene_config.base_learning_rate
          * _get_material_learning_rate(scene_config, key, param_type)
      )
      print(
          f'Learning rate for (scalar) {key.split(".")[0]} ({param_type}) is '
          f' {learning_rate}'
      )

      variable = variable_types.ArrayVariable(
          key=key,
          initial_value=initial_value,
          clamp_range=scene_config.per_param_clamp_ranges[param_type],
          learning_rate=learning_rate,
          is_scene_parameter=True,
      )
      variables.append(variable)
    # Texture parameters
    else:
      param_type = get_param_type_from_mipmap_or_bitmap_param(key)
      initial_value = _get_initial_texture_params(scene_config, params, key)

      learning_rate = (
          scene_config.base_learning_rate
          * _get_material_learning_rate(scene_config, key, param_type)
      )
      print(
          f'Learning rate for (texture) {key.split(".")[0]} ({param_type}) is '
          f' {learning_rate}'
      )

      variable = create_texture_variable(
          scene_config,
          params,
          key=key,
          clamp_range=scene_config.per_param_clamp_ranges[param_type],
          initial_value=initial_value,
          learning_rate=learning_rate,
          normal_clamping=param_type == 'normalmap',
          scalar_perturbation=scene_config.random_initialisation,
      )
      variables.append(variable)

  if (
      scene_config.deng_comparison
      and scene_config.deng_displacement_learning_rate > 0.0
  ):
    learning_rate = (
        scene_config.base_learning_rate
        * scene_config.deng_displacement_learning_rate
    )
    print(
        f'Creating displacement variable with learning rate {learning_rate} and'
        f' scale {scene_config.deng_displacement_scale}'
    )
    variable, optimized_scene_key = create_displacement_variable(
        mesh_name=scene_config.scene_name,
        displacement_scale=scene_config.deng_displacement_scale,
        learning_rate=learning_rate,
        texture_res=(
            scene_config.deng_displacement_res,
            scene_config.deng_displacement_res,
        ),
    )
    variables.append(variable)
    optimized_keys.append(optimized_scene_key)

  return variables
