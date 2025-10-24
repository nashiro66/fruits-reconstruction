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

"""Utilities for volume conversion."""

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from core.integrators import dwivedi_utils

if not mi.variant():
  mi.set_variant("cuda_ad_rgb")


def index_spectrum(spec, idx):
  m = spec[0]
  if mi.is_rgb:
    m[idx == 1] = spec[1]
    m[idx == 2] = spec[2]
  return m


def eval_hg(g: mi.Spectrum | mi.Float, cos_theta: mi.Float):
  temp = 1.0 + dr.square(g) + 2.0 * g * cos_theta
  return dr.inv_four_pi * (1.0 - dr.square(g)) / (temp * dr.sqrt(temp))


def eval_spectral_hg(g, phase_wo, ray):
  return eval_hg(g, dr.dot(phase_wo, -ray.d))


class SSSInteraction:
  """SSS volume interaction."""

  DRJIT_STRUCT = {
      "t": mi.Float,
      "p": mi.Point3f,
      "wi": mi.Vector3f,
      "time": mi.Float,
      "wavelengths": mi.Color0f,
      "sigma_t": mi.Spectrum,
      "sigma_s": mi.Spectrum,
      "g": mi.Spectrum,
  }

  def __init__(self):
    self.t = dr.inf
    self.sigma_s = mi.Spectrum(0.0)

  def is_valid(self):
    return self.t != dr.inf


class SSSMedium:
  """Context for SSS volume events."""

  DRJIT_STRUCT = {
      "albedo": mi.Spectrum,
      "sigma_t": mi.Spectrum,
      "g": mi.Spectrum,
      "n": mi.Normal3f,
      "diffusion_length": mi.Float,
      "bsdf_has_sss": mi.Bool,
  }

  def __init__(self):
    self.bsdf_has_sss = mi.Bool(False)
    self.albedo = mi.Spectrum(0.0)
    self.n = mi.Normal3f(0.0)
    self.diffusion_length = mi.Float(0.0)

  def populate(self, bsdf, si: mi.SurfaceInteraction3f, active: mi.Bool):
    """Populates the SSSMedium from the BSDF and surface interaction."""
    bsdf_has_sss = (
        bsdf.has_attribute("single_scattering_albedo")
        & bsdf.has_attribute("extinction_coefficient")
        & bsdf.has_attribute("hg_coefficient")
        & active
    )

    self.bsdf_has_sss[active] = bsdf_has_sss
    self.diffusion_length[active] = dwivedi_utils.diffusion_length(
        dr.max(self.albedo)
    )
    self.n[active] = si.n

  # Both considers entries but also "re-entries" where a scattered ray stays
  # inside the volume. Then new parameters are set
  def is_sss_entry(self, ray, si):
    return self.bsdf_has_sss & (dr.dot(ray.d, si.n) < 0)

  def transmittance_eval_pdf(
      self,
      mei: mi.MediumInteraction3f,
      si: mi.SurfaceInteraction3f,
      active_medium: mi.Bool,
      active_boundary_hit: mi.Bool,
  ):
    """Evaluates the transmittance and free-flight PDF.

    Requires a sampled media interaction and the next surface interaction with
    their respective active masks.

    Args:
      mei: The media interaction.
      si: The surface interaction.
      active_medium: Whether we are still in the medium.
      active_boundary_hit: Whether the ray hits the volume boundary.

    Returns:
      The ratio of transmittance and free-flight PDF.
    """
    t = dr.minimum(mei.t, si.t)
    # Transmittance is unchanged but pdf needs to account for Dwivedi stretching
    tr = dr.exp(-t * mei.sigma_t)
    pdf = mi.Spectrum(0.0)
    pdf[active_medium] = tr * mei.sigma_t
    pdf[active_boundary_hit] = tr
    return tr, pdf


def update_weight_matrix(p_over_f, p, f, active):
  n = dr.size_v(mi.Spectrum)
  for i in range(n):
    ratio = p / f[i]
    ratio = dr.select(dr.isfinite(ratio), ratio, 0.0)
    ratio *= p_over_f[i]
    p_over_f[i][active] = dr.select(dr.isnan(ratio), 0.0, ratio)


def mis_weight_from_matrix(p_over_f):
  weight = mi.Spectrum(0.0)
  n = dr.size_v(mi.Spectrum)
  for i in range(n):
    pdf_over_f_sum = dr.sum(p_over_f[i])
    weight[i] = dr.select(pdf_over_f_sum == 0, 0, n / pdf_over_f_sum)
  return weight


def direction_from_cosine(
    n_dir: mi.Vector3f, cos_theta: mi.Float, rand: mi.Float
):
  sin_theta = dr.safe_sqrt(1.0 - dr.square(cos_theta))
  sin_phi, cos_phi = dr.sincos(2.0 * dr.pi * rand)

  dwivedi_frame = mi.Frame3f(n_dir)
  dwivedi_local_dir = mi.Vector3f(
      sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
  )

  return dwivedi_frame.to_world(dwivedi_local_dir)


def balance_mis_weight(pdf_a, pdf_b, weight_a = 1.0, weight_b = 1.0):
  w = weight_a * pdf_a / (weight_a * pdf_a + weight_b * pdf_b)
  return dr.detach(dr.select(dr.isfinite(w), w, 0))

def sample_spectral_hg(
    g: mi.Spectrum, ray: mi.Ray3f, sample2: mi.Vector2f, channel: mi.UInt32
):
  """Samples the Henyey-Greenstein phase function with a given channel.

  Args:
    g: The Henyey-Greenstein coefficient.
    ray: The ray to sample along.
    sample2: The sample point.
    channel: The channel to sample.

  Returns:
    The sampled phase function direction, its spectral weight and pdf.
  """
  g1 = index_spectrum(mi.Spectrum(g), channel)
  sqr_term = (1.0 - dr.square(g1)) / (1.0 - g1 + 2.0 * g1 * sample2.x)
  cos_theta = (1.0 + dr.square(g1) - dr.square(sqr_term)) / (2.0 * g1)

  # Diffuse fallback
  cos_theta[dr.abs(g1) < dr.epsilon(mi.Float)] = 1.0 - 2.0 * sample2.x

  sin_theta = dr.safe_sqrt(1.0 - dr.square(cos_theta))
  sin_phi, cos_phi = dr.sincos(2.0 * dr.pi * sample2.y)

  its_frame = mi.Frame3f(ray.d)
  wo = its_frame.to_world(
      mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
  )
  weight = mi.Spectrum(1.0)
  return wo, weight, eval_spectral_hg(g, wo, ray)


@dr.syntax
def sample_spectral_sss_scattering(
    mode: dr.ADMode,
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.Ray3f,
    si: mi.SurfaceInteraction3f,
    sss_medium: SSSMedium,
    max_sss_depth: int,
    p_over_f,
    active_surface: mi.Bool,
    active_medium: mi.Bool,
    eta: int,
    sss_rr_depth: int,
    channel: mi.UInt32,
    medium_shape_bsdf: mi.BSDFPtr,
    medium_shape_si: mi.SurfaceInteraction3f,
    medium: mi.MediumPtr,
    dwivedi_guiding: mi.Bool = False,
):
  """Samples SSS scattering events.

  Args:
    mode: The AD mode.
    scene: The Mitsuba scene.
    sampler: The Mitsuba sampler.
    ray: The ray to sample along.
    si: The surface interaction.
    sss_medium: The SSS medium.
    max_sss_depth: The maximum SSS depth.
    p_over_f: The ratio of the pdfs.
    active_surface: Whether the surface is active.
    active_medium: Whether the medium is active.
    eta: The Russian roulette stopping probability.
    sss_rr_depth: The depth at which to apply the Russian roulette stopping
      probability.
    channel: The channel to sample.
    medium_shape_bsdf: The BSDF of the medium shape.
    medium_shape_si: The surface interaction of the medium shape.

  Returns:
    The volume throughput, the number of SSS bounces, and the total SSS
    distance. (active_surface and active_medium are also updated)
  """
  primal = mode == dr.ADMode.Primal

  sss_depth = mi.UInt32(0)
  sss_bounces = mi.UInt32(0)
  sss_distance = mi.Float(0.0)
  # Gradient of g
  δg = mi.Spectrum(0.0)

  with dr.resume_grad(when=not primal):
    g = medium_shape_bsdf.eval_attribute(
        "hg_coefficient", medium_shape_si, active_medium
    )
    g_needs_grads = dr.grad_enabled(g)
    g_local = dr.detach(g)
      
  # Whether to follow the Dwivedi guiding or the classic random walk, always
  # classic for the first distance sampling
  # TODO(pweier): Additionaly check for backward or forward guiding (for now, only forward)
  
  phase_ctx = mi.PhaseFunctionContext(sampler)
  phase = dr.zeros(mi.PhaseFunctionPtr)
  
  while dr.hint(
      active_medium,
      max_iterations=max_sss_depth,
      label="SSS Loop (%s)" % mode.name,
  ):
    u = sampler.next_1d(active_medium)
    mei = medium.sample_interaction(ray, u, channel, active_medium)
    #aabb_its, mint, maxt= medium.intersect_aabb(ray)
    # dr.print("Medium sample info:\n{data}", data={
    #     "ray-d": ray.d,
    #     # "ray-o": ray.o,
    #     # "u": u,
    #     "mint": mint,
    #     "maxt": maxt,
    #     "aabb_its": aabb_its,
    #     "t": mei.t,
    #     "p": mei.p,
    #     "sigma_t": mei.combined_extinction,
    #     "sigma_s": mei.sigma_s,
    #     "valid": mei.is_valid()
    # })

    sss_medium.populate(medium_shape_bsdf, medium_shape_si, active_medium)

    ray.maxt[active_medium & mei.is_valid()] = mei.t
    # Compute a surface interaction that tracks derivatives arising
    # from differentiable shape parameters (position, normals, etc.)
    # In primal mode, this is just an ordinary ray tracing operation.
    with dr.resume_grad(when=not primal):
      si[active_medium] = scene.ray_intersect(ray, active_medium)
    mei.t[active_medium & (si.t < mei.t)] = dr.inf

    # Either a volume interaction was found or the next interaction is a
    # surface
    active_boundary_hit = active_medium & ~mei.is_valid()
    active_medium &= mei.is_valid()
    # Reset maxt for the boundary hit for the true next surface interaction
    ray.maxt[active_boundary_hit] = dr.inf

    # Evaluate ratio of transmittance and free-flight PDF
    tr, free_flight_pdf = sss_medium.transmittance_eval_pdf(
        mei,
        si,
        active_medium,
        active_boundary_hit,
    )
    f = mi.Spectrum(1.0)
    f[active_medium] *= mei.sigma_s
    f[active_medium | active_boundary_hit] *= tr
    update_weight_matrix(
        p_over_f,
        free_flight_pdf,
        f,
        active_medium | active_boundary_hit,
    )
    # test = dr.max(mis_weight_from_matrix(p_over_f))  
    # active_medium &= (medium != None) 
    
    sss_bounces[active_medium] += 1
    sss_distance[active_medium] += mei.t
    # Add final distance to boundary hit without adding an extra bounce
    # to not double count the extra sigma_t during the differentiable phase
    sss_distance[active_boundary_hit] += si.t

    # Sample phase function
    phase_wo, phase_weight, phase_pdf = sample_spectral_hg(
          0.6, ray, sampler.next_2d(active_medium), channel
    )
    phase_eval = phase_weight * phase_pdf
    update_weight_matrix(
        p_over_f,
        phase_pdf,
        phase_weight * phase_pdf,
        active_medium,
    )

    # Differentiable evaluation of the phase function
    if dr.hint(not primal, mode="scalar"):
      with dr.resume_grad():
        if dr.hint(g_needs_grads, mode="scalar"):
          # Locally enable the gradient of g_local to compute the gradient of
          # the phase function with respect to g locally
          dr.enable_grad(g_local)
          dr.set_grad(g_local, 1)
          phase_eval_grad = eval_spectral_hg(g_local, phase_wo, ray)
          phase_eval_detach = phase_eval
          inv_phase_eval_detach = dr.select(
              phase_eval_detach != 0, dr.rcp(phase_eval_detach), 0
          )
          # Compute the gradient of the phase function with respect to g
          δg[active_medium] += dr.forward_to(
              inv_phase_eval_detach * phase_eval_grad
          )
          # Clear the gradient of g after the local forward pass
          dr.clear_grad(g_local)

    # Construct next ray
    ray[active_medium] = mi.Ray3f(mei.p, phase_wo)
    sss_depth[active_medium] += 1

    # -------------------- Stopping criterion ---------------------

    volume_throughput_mis = mis_weight_from_matrix(p_over_f)

    # Don't run another iteration if the throughput has reached zero
    volume_throughput_max = dr.max(volume_throughput_mis)
    active_medium &= volume_throughput_max != 0
    active_boundary_hit &= volume_throughput_max != 0

    # Russian roulette stopping probability (must cancel out ior^2
    # to obtain unitless throughput) no min probability for volume scattering
    rr_prob = dr.minimum(volume_throughput_max * eta**2, 1.0)
    # Don't apply Russian roulette on active boundary hits as they are already
    # accounted for during surface interatctions
    rr_prob[active_boundary_hit] = 1.0

    rr_active = sss_depth >= sss_rr_depth
    update_weight_matrix(
        p_over_f, dr.detach(rr_prob), mi.Spectrum(1.0), rr_active
    )
    rr_continue = sampler.next_1d() < rr_prob
    active_medium &= ~rr_active | rr_continue
    # sss_depth < max_sss_depth + 1 ensures that an sss_depth of 1 will
    # effectively trace TWO rays in the volume one for finding the first medium
    # interaction and one to hit the boundary.
    active_medium &= sss_depth < max_sss_depth + 1

    active_surface |= active_boundary_hit

  return sss_bounces, sss_distance, δg