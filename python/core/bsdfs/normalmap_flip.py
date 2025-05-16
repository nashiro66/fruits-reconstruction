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

"""Defines a Mip-mapped Bitmap texture."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

if not mi.variant():
  mi.set_variant("llvm_ad_rgb")


def _fnmadd(a, b, c):
  return -a * b + c


def _fmadd(a, b, c):
  return a * b + c


class NormalMapFlip(mi.BSDF):
  """Normalmap with smart flipping"""

  def __init__(self, props):
    super().__init__(props)

    self.normalmap = props["normalmap"]
    if not issubclass(type(self.normalmap), mi.Texture):
      raise ValueError(f"normalmap property is not a texture!")

    self.safe_flip = props.get("safe_flip", False)
    self.shadow_terminator = props.get("shadow_terminator", True)

    nested_bsdf = None
    for prop in props.unqueried():
      if issubclass(type(props[prop]), mi.BSDF):
        if nested_bsdf is not None:
          raise ValueError("Only a single BSDF child object can be specified.")
        nested_bsdf = props[prop]

    if nested_bsdf is None:
      raise ValueError("Exactly one BSDF child object must be specified.")
    else:
      self.nested_bsdf = nested_bsdf

    self.m_flags = self.nested_bsdf.m_flags
    self.m_components = self.nested_bsdf.m_components

  def traverse(self, callback):
    callback.put_object(
        "nested_bsdf", self.nested_bsdf, mi.ParamFlags.Differentiable
    )
    callback.put_object(
        "normalmap",
        self.normalmap,
        mi.ParamFlags.Differentiable | mi.ParamFlags.Discontinuous,
    )

  # From "A Microfacet-Based Shadowing Function to Solve the Bump Terminator
  # Problem" Estevez et al. 2019, Ray Tracing Gems 2019
  # Return alpha^2 parameter from normal divergence
  def bump_alpha2(self, geo_n: mi.Vector3f, shading_n: mi.Vector3f):
    cos_d = dr.minimum(dr.abs(dr.dot(geo_n, shading_n)), 1.0)
    tan2_d = (1.0 - cos_d * cos_d) / (cos_d * cos_d)
    return dr.clip(0.125 * tan2_d, 0.0, 1.0)

  # Shadowing factor
  def bump_shadowing_function(
      self, geo_n: mi.Vector3f, wo: mi.Vector3f, alpha2: mi.Float
  ):
    cos_i = dr.maximum(dr.abs(dr.dot(geo_n, wo)), 1e-6)
    tan2_i = (1.0 - cos_i * cos_i) / (cos_i * cos_i)
    return 2.0 / (1.0 + dr.sqrt(1.0 + alpha2 * tan2_i))

  def frame(self, si: mi.SurfaceInteraction3f, active: mi.Bool):
    n = dr.normalize(_fmadd(self.normalmap.eval_3(si, active), 2, -1.0))

    if dr.hint(self.safe_flip, mode="scalar"):
      back_facing_and_inward = (dr.dot(si.wi, n) <= 0.0) & (
          mi.Frame3f.cos_theta(si.wi) > 0.0
      )
      n = dr.select(back_facing_and_inward, mi.Normal3f(-n.x, -n.y, n.z), n)

    frame_wrt_si = mi.Frame3f()
    frame_wrt_si.n = n
    frame_wrt_si.s = dr.normalize(
        _fnmadd(frame_wrt_si.n, frame_wrt_si.n.x, mi.Vector3f(1, 0, 0))
    )
    frame_wrt_si.t = dr.cross(frame_wrt_si.n, frame_wrt_si.s)

    frame_wrt_world = mi.Frame3f()
    frame_wrt_world.n = si.to_world(frame_wrt_si.n)
    frame_wrt_world.s = si.to_world(frame_wrt_si.s)
    frame_wrt_world.t = si.to_world(frame_wrt_si.t)

    return frame_wrt_si, frame_wrt_world

  def sample(self, ctx, si, sample1, sample2, active):
    # Sample nested BSDF with perturbed shading frame
    perturbed_frame_wrt_si, perturbed_frame_wrt_world = self.frame(si, active)
    perturbed_si = mi.SurfaceInteraction3f(si)
    perturbed_si.sh_frame = perturbed_frame_wrt_world
    perturbed_si.wi = perturbed_frame_wrt_si.to_local(si.wi)
    bs, weight = self.nested_bsdf.sample(
        ctx, perturbed_si, sample1, sample2, active
    )
    active = active & dr.any(weight != 0.0)

    # Transform sampled 'wo' back to original frame and check orientation
    perturbed_wo = perturbed_frame_wrt_si.to_world(bs.wo)
    active = active & (
        mi.Frame3f.cos_theta(bs.wo) * mi.Frame3f.cos_theta(perturbed_wo) > 0.0
    )
    bs.pdf = dr.select(active, bs.pdf, 0.0)
    bs.wo = perturbed_wo

    if self.shadow_terminator:
      # Compute bump alpha2 and shadowing factor
      alpha2 = self.bump_alpha2(si.n, si.to_world(perturbed_si.sh_frame.n))
      shadowing_factor = self.bump_shadowing_function(
          si.n, si.to_world(bs.wo), alpha2
      )
      weight[alpha2 > 0.0] *= shadowing_factor

    return bs, weight & active

  def eval(self, ctx, si, wo, active):
    # Evaluate nested BSDF with perturbed shading frame
    perturbed_frame_wrt_si, perturbed_frame_wrt_world = self.frame(si, active)
    perturbed_si = mi.SurfaceInteraction3f(si)
    perturbed_si.sh_frame = perturbed_frame_wrt_world
    perturbed_si.wi = perturbed_frame_wrt_si.to_local(si.wi)
    perturbed_wo = perturbed_frame_wrt_si.to_local(wo)

    active = active & (
        mi.Frame3f.cos_theta(wo) * mi.Frame3f.cos_theta(perturbed_wo) > 0.0
    )

    value = (
        self.nested_bsdf.eval(ctx, perturbed_si, perturbed_wo, active) & active
    )

    if self.shadow_terminator:
      # Compute bump alpha2 and shadowing factor
      alpha2 = self.bump_alpha2(si.n, si.to_world(perturbed_si.sh_frame.n))
      shadowing_factor = self.bump_shadowing_function(
          si.n, si.to_world(wo), alpha2
      )
      value[alpha2 > 0.0] *= shadowing_factor

    return value

  def pdf(self, ctx, si, wo, active):
    # Evaluate nested BSDF with perturbed shading frame
    perturbed_frame_wrt_si, perturbed_frame_wrt_world = self.frame(si, active)
    perturbed_si = mi.SurfaceInteraction3f(si)
    perturbed_si.sh_frame = perturbed_frame_wrt_world
    perturbed_si.wi = perturbed_frame_wrt_si.to_local(si.wi)
    perturbed_wo = perturbed_frame_wrt_si.to_local(wo)

    active = active & (
        mi.Frame3f.cos_theta(wo) * mi.Frame3f.cos_theta(perturbed_wo) > 0.0
    )

    return dr.select(
        active,
        self.nested_bsdf.pdf(ctx, perturbed_si, perturbed_wo, active),
        0.0,
    )

  def eval_pdf(self, ctx, si, wo, active):
    perturbed_frame_wrt_si, perturbed_frame_wrt_world = self.frame(si, active)
    perturbed_si = mi.SurfaceInteraction3f(si)
    perturbed_si.sh_frame = perturbed_frame_wrt_world
    perturbed_si.wi = perturbed_frame_wrt_si.to_local(si.wi)
    perturbed_wo = perturbed_frame_wrt_si.to_local(wo)

    active = active & (
        mi.Frame3f.cos_theta(wo) * mi.Frame3f.cos_theta(perturbed_wo) > 0.0
    )

    value, pdf = self.nested_bsdf.eval_pdf(
        ctx, perturbed_si, perturbed_wo, active
    )
    return value & active, dr.select(active, pdf, 0.0)

  def eval_diffuse_reflectance(self, si, active):
    return self.nested_bsdf.eval_diffuse_reflectance(si, active)

  def has_attribute(self, name, active):
    return self.nested_bsdf.has_attribute(name, active)

  def eval_attribute(self, name, si, active):
    return self.nested_bsdf.eval_attribute(name, si, active)

  def to_string(self) -> str:
    return (
        f"NormalMapFlip[\nsafe_flip: {self.safe_flip} \n"
        f"shadow_terminator:  {self.shadow_terminator}\n"
        f"outgoing_perturbation:  {self.outgoing_perturbation}\n"
        f"nested_bsdf:  {self.nested_bsdf}\n"
        f"normalmap:  {self.normalmap}\n]"
    )


def register():
  """Registers the normalmap plugin."""
  mi.register_bsdf(
      "normalmap_flip", lambda props: NormalMapFlip(props)  # pylint: disable=unnecessary-lambda
  )
