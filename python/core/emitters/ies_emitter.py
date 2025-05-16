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

"""Defines an IES emitter that can use a measured IES profile."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np


class IesEmitter(mi.Emitter):
  """This emitter can be used to render with an IES profile.

  Currently, we only support radially symmetric IES profiles for disk lights.

  The emitter further has a `smooth` parameters, which will introduce
  an empirically chosen falloff towards the edge of the emitter to remove
  discontinuities during optimization.

  Lastly, similar to the "switchemitter", this emitter will have a
  zero contribution if its sampling weight is set to 0.
  """

  def __init__(self, props: mi.Properties):
    super().__init__(props)
    self.nested_emitter = props['nested_emitter']

    if not isinstance(self.nested_emitter, mi.Emitter):
      raise TypeError(
          f'Expected mi.Emitter object, but got {type(self.nested_emitter)}.'
      )

    self.smooth = props.get('smooth', True)
    ies_values = props['ies_values']
    ies_values = np.fromstring(ies_values, dtype=np.float32, sep=' ')
    self.ies_texture = mi.Texture1f(ies_values[..., None])
    self.m_flags = mi.EmitterFlags.Surface
    self.is_prepared = False

  def prepare(self):
    """Prepares the emitter for use.

    This has to be invoked before this emitter is used, otherwise it will raise
    an error on use. This function is *not* automatically invoked within Mitsuba
    and has to be called after loading the scene.
    """
    if not self.nested_emitter.get_shape():
      shape = self.get_shape()

      if shape is not None:
        shape_class_name = shape.class_().name()
        if self.smooth and shape_class_name != 'Disk':
          raise ValueError('Smooth IES lights assume a disk light shape.')
        self.nested_emitter.set_shape(shape)

    self.is_prepared = True

  def _check(self):
    """Checks internal state for validity."""
    if not self.is_prepared:
      raise ValueError('Emitter.prepare() has to be called before use!')

  def _eval_attenuation(self, cos_theta, uv):
    ies_value = self.ies_texture.eval(cos_theta)[0]
    if self.smooth:
      x = uv.x
      x = dr.sqr(dr.sqr(dr.sqr(x)))
      falloff = dr.select(
          uv.x <= 1, dr.exp(1 - 1 / dr.maximum(1 - x, 1e-6)), 0.0
      )
    else:
      falloff = 1.0
    return ies_value * falloff

  # See Mitsuba's `emitter.h` for documentation of these interfaces
  def eval(
      self,
      si: mi.SurfaceInteraction3f,
      active: mi.Bool = True,
  ) -> mi.Spectrum:
    self._check()
    if self.sampling_weight() <= 0.0:
      return mi.Spectrum(0.0)
    attenuation = self._eval_attenuation(mi.Frame3f.cos_theta(si.wi), si.uv)
    return self.nested_emitter.eval(si, active) * attenuation

  def eval_direction(
      self,
      it: mi.Interaction3f,
      ds: mi.DirectionSample3f,
      active: mi.Bool = True,
  ) -> mi.Spectrum:
    self._check()
    if self.sampling_weight() <= 0.0:
      return mi.Spectrum(0.0)
    attenuation = self._eval_attenuation(-dr.dot(ds.d, ds.n), ds.uv)
    return self.nested_emitter.eval_direction(it, ds, active) * attenuation

  def sample_direction(
      self,
      it: mi.Interaction3f,
      sample: mi.Point3f,
      active: mi.Bool = True,
  ) -> tuple[mi.DirectionSample3f, mi.Spectrum]:
    self._check()
    direction_sample, weight = self.nested_emitter.sample_direction(
        it, sample, active
    )
    cos_theta = dr.dot(
        dr.normalize(it.p - direction_sample.p), direction_sample.n
    )
    attenuation = self._eval_attenuation(cos_theta, direction_sample.uv)
    direction_sample.emitter = mi.EmitterPtr(self)
    return direction_sample, weight * attenuation

  def pdf_direction(
      self,
      it: mi.Interaction3f,
      ds: mi.DirectionSample3f,
      active: mi.Bool = True,
  ) -> mi.Float:
    self._check()
    return self.nested_emitter.pdf_direction(it, ds, active)

  def bbox(self) -> mi.ScalarBoundingBox3f:
    return self.nested_emitter.bbox()

  def traverse(self, callback: mi.TraversalCallback):
    callback.put_object(
        'nested_emitter', self.nested_emitter, mi.ParamFlags.Differentiable
    )
    super().traverse(callback)

  def to_string(self) -> str:
    return f'IesEmitter[\nnested_emitter: {self.nested_emitter}\n]'


def register():
  """Registers the IES emitter plugin."""
  mi.register_emitter('iesemitter', lambda props: IesEmitter(props))  # pylint: disable=unnecessary-lambda
