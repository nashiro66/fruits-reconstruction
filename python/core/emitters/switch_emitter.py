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

"""Defines a "switch" emitter that can be turned off by setting sampling_weight = 0.0."""

from __future__ import annotations

import mitsuba as mi  # type: ignore


class SwitchEmitter(mi.Emitter):
  """This emitter wraps an inner emitter and supports turning it off.

  This plugin provides a way to selectively turn off an emitter by setting its
  `sampling_weight` (see Mitsuba's emitter.h/cpp) to 0. Normally, setting the
  sampling_weight to zero won't affect the emitter evaluation itself. This
  plugin changes this such that the sampling weight provides a way to completely
  disable a nested emitter. This is useful for OLAT setups, where we need to
  be able switch on and off emitters on the fly. Using this switch emitter, we
  can conveniently (temporarily) disable some emitters without having to
  introduce complex logic to update/restore scene state or having to rebuild the
  scene's BVH.
  """

  def __init__(self, props: mi.Properties):
    mi.Emitter.__init__(self, props)
    self.nested_emitter = props['nested_emitter']
    if not isinstance(self.nested_emitter, mi.Emitter):
      raise TypeError(
          f'Expected mi.Emitter object, but got {type(self.nested_emitter)}.'
      )

    # Just replicate the flags of the nested emitter.
    self.m_flags = self.nested_emitter.flags()

    self.is_prepared = False

  def prepare(self):
    """Prepares the emitter for use.

    This has to be invoked before this emitter is used, otherwise it will raise
    an error on use. This function is *not* automatically invoked within Mitsuba
    and has to be called after loading the scene.
    """
    shape = self.get_shape()
    if not self.nested_emitter.get_shape() and shape is not None:
      self.nested_emitter.set_shape(shape)
    self.is_prepared = True

  def _check(self):
    """Checks internal state for validity."""
    if not self.is_prepared:
      raise ValueError('Emitter.prepare() has to be called before use!')

  # See Mitsuba's `emitter.h` for documentation of these interfaces
  def eval(
      self,
      si: mi.SurfaceInteraction3f,
      active: mi.Bool = True,
  ) -> mi.Spectrum:
    self._check()
    if self.sampling_weight() <= 0.0:
      return mi.Spectrum(0.0)
    return self.nested_emitter.eval(si, active)

  def eval_direction(
      self,
      it: mi.Interaction3f,
      ds: mi.DirectionSample3f,
      active: mi.Bool = True,
  ) -> mi.Spectrum:
    self._check()
    if self.sampling_weight() <= 0.0:
      return mi.Spectrum(0.0)
    return self.nested_emitter.eval_direction(it, ds, active)

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
    direction_sample.emitter = mi.EmitterPtr(self)
    return direction_sample, weight

  def pdf_direction(
      self,
      it: mi.Interaction3f,
      ds: mi.DirectionSample3f,
      active: mi.Bool = True,
  ) -> mi.Float:
    self._check()
    return self.nested_emitter.pdf_direction(it, ds, active)

  def traverse(self, callback: mi.TraversalCallback):
    callback.put_object(
        'nested_emitter', self.nested_emitter, mi.ParamFlags.Differentiable
    )
    super().traverse(callback)

  def to_string(self) -> str:
    return f'SwitchEmitter[\nnested_emitter: {self.nested_emitter}\n]'


def register():
  """Registers the switch emitter plugin."""
  mi.register_emitter('switchemitter', lambda props: SwitchEmitter(props))  # pylint: disable=unnecessary-lambda
