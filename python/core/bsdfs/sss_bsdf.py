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

"""Defines a subsurface compatible BSDF that wraps any existing transmissive BSDF ."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore


def convert_to_texture(value):
  if issubclass(type(value), mi.Texture):
    return value

  if type(value) is float:
    return mi.load_dict({
        'type': 'srgb',
        'unbounded': True,
        'color': [value, value, value],
    })
  else:
    array = value.numpy().tolist()
    if len(array) != 3:
      raise ValueError(f'Expected array of 3 values, got {value}.')

    return mi.load_dict({
        'type': 'srgb',
        'unbounded': True,
        'color': array,
    })


class SSSBSDF(mi.BSDF):

  def __init__(self, props):
    mi.BSDF.__init__(self, props)
    self.nested_bsdf = props['nested_bsdf']

    if not mi.has_flag(self.nested_bsdf.m_flags, mi.BSDFFlags.Transmission):
      raise ValueError('The nested BSDF should have a transmissive component!')

    # Set the BSDF flags
    self.m_flags = self.nested_bsdf.m_flags
    self.m_components = self.nested_bsdf.m_components

    # Among other cases, handles the case where just an array of float is 
    # specified in a loaded dictionary
    self.single_scattering_albedo = convert_to_texture(
        props.get('single_scattering_albedo', 0.0)
    )
    self.extinction_coefficient = convert_to_texture(
        props.get('extinction_coefficient', 1.0)
    )
    self.hg_coefficient = convert_to_texture(props.get('hg_coefficient', 0.0))

  def traverse(self, callback):
    callback.put_object(
        'single_scattering_albedo',
        self.single_scattering_albedo,
        mi.ParamFlags.Differentiable,
    )
    callback.put_object(
        'extinction_coefficient',
        self.extinction_coefficient,
        mi.ParamFlags.Differentiable,
    )
    callback.put_object(
        'hg_coefficient', self.hg_coefficient, mi.ParamFlags.Differentiable
    )
    # This ensures no extra bsdf identifier name is added in the traversal callback
    self.nested_bsdf.traverse(callback)

  def sample(self, ctx, si, sample1, sample2, active):
    return self.nested_bsdf.sample(ctx, si, sample1, sample2, active)

  def eval(self, ctx, si, wo, active):
    return self.nested_bsdf.eval(ctx, si, wo, active)

  def pdf(self, ctx, si, wo, active):
    return self.nested_bsdf.pdf(ctx, si, wo, active)

  def eval_pdf(self, ctx, si, wo, active):
    return self.nested_bsdf.eval_pdf(ctx, si, wo, active)

  def parameters_changed(self, keys):
    self.nested_bsdf.parameters_changed(keys)

  def to_string(self):
    return (
        'SSSBsdf[\n   '
        ' nested_bsdf=%s\nsingle_scattering_albedo:%s,\nextinction_coefficient:%s,\nhg_coefficient:%s\n]'
        % (
            self.nested_bsdf,
            self.single_scattering_albedo,
            self.extinction_coefficient,
            self.hg_coefficient,
        )
    )


def register():
  """Registers the sss bsdf plugin."""
  mi.register_bsdf(
      'sss_bsdf',
      lambda props: SSSBSDF(props),  # pylint: disable=unnecessary-lambda
  )
