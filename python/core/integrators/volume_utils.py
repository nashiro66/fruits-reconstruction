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


def sample_isotropic(sample2: mi.Float):
  phase_wo = mi.warp.square_to_uniform_sphere(sample2)
  pdf = mi.warp.square_to_uniform_sphere_pdf(phase_wo)
  weight = mi.Spectrum(1.0)
  return phase_wo, weight, pdf


def eval_pdf_isotropic(phase_wo):
  pdf = mi.warp.square_to_uniform_sphere_pdf(phase_wo)
  return pdf, pdf


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


def eval_pdf_hg(g, phase_wo, ray, channel):
  g = index_spectrum(mi.Spectrum(g), channel)
  pdf = eval_hg(g, dr.dot(phase_wo, -ray.d))
  return pdf, pdf


def eval_spectral_hg(g, phase_wo, ray):
  return eval_hg(g, dr.dot(phase_wo, -ray.d))


def pdf_spectral_hg(g, phase_wo, ray):
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
    self.sigma_t = mi.Spectrum(0.0)
    self.sigma_s = mi.Spectrum(0.0)
    self.g = mi.Spectrum(0.0)

  # We don't use these (yet?) but force HG, g = 0.0 is isotropic anyway
  # def sample_phase(self, ray, sample2):
  #   return dr.select(
  #       (self.g[0] != 0.0),
  #       sample_hg(self.g, ray, sample2),
  #       sample_isotropic(sample2),
  #   )

  # def eval_pdf_phase(self, phase_wo, ray):
  #   return dr.select(
  #       self.g[0] != 0.0,
  #       eval_pdf_hg(self.g, phase_wo, ray),
  #       eval_pdf_isotropic(phase_wo),
  #   )

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
    self.sigma_t = mi.Spectrum(0.0)
    self.g = mi.Spectrum(0.0)
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
    self.albedo[active] = bsdf.eval_attribute(
        "single_scattering_albedo", si, bsdf_has_sss
    )
    self.sigma_t[active] = bsdf.eval_attribute(
        "extinction_coefficient", si, bsdf_has_sss
    )
    self.g[active] = bsdf.eval_attribute("hg_coefficient", si, bsdf_has_sss)
    self.diffusion_length[active] = dwivedi_utils.diffusion_length(
        dr.max(self.albedo)
    )
    self.n[active] = si.n

  # Both considers entries but also "re-entries" where a scattered ray stays
  # inside the volume. Then new parameters are set
  def is_sss_entry(self, ray, si):
    return self.bsdf_has_sss & (dr.dot(ray.d, si.n) < 0)

  def sample_interaction(
      self,
      ray: mi.Ray3f,
      sample: mi.Float,
      channel: mi.UInt32,
      active: mi.Bool,
      dwivedi_stretching: mi.Float = 1.0,
  ):
    """Samples a SSS interaction along the ray."""
    sss_its = dr.zeros(SSSInteraction)
    sss_its.wi = -ray.d
    sss_its.time = ray.time
    sss_its.wavelengths = ray.wavelengths

    # Dwivedi guiding implies a simple stretching of the sampled sigma_t
    # Careful to make a copy to get the channel value only!
    sampled_sigma_t = index_spectrum(
        mi.Spectrum(self.sigma_t * dwivedi_stretching), channel
    )
    sampled_t = -dr.log(1.0 - sample) / sampled_sigma_t
    valid_sss_its = active & (sampled_t <= dr.inf)

    sss_its.t = dr.select(valid_sss_its, sampled_t, dr.inf)
    sss_its.p = ray(sampled_t)
    sss_its.sigma_t = self.sigma_t
    sss_its.sigma_s = self.sigma_t * self.albedo
    sss_its.g = self.g

    return sss_its

  def transmittance_eval_pdf(
      self,
      sss_its: SSSInteraction,
      si: mi.SurfaceInteraction3f,
      active_medium: mi.Bool,
      active_boundary_hit: mi.Bool,
      dwivedi_stretching: mi.Float = 1.0,
  ):
    """Evaluates the transmittance and free-flight PDF.

    Requires a sampled SSS interaction and the next surface interaction with
    their respective active masks.

    Args:
      sss_its: The SSS interaction.
      si: The surface interaction.
      active_medium: Whether we are still in the medium.
      active_boundary_hit: Whether the ray hits the volume boundary.

    Returns:
      The ratio of transmittance and free-flight PDF.
    """
    t = dr.minimum(sss_its.t, si.t)
    # Transmittance is unchanged but pdf needs to account for Dwivedi stretching
    tr = dr.exp(-t * sss_its.sigma_t)
    pdf = mi.Spectrum(0.0)
    sigma_t_dwivedi = sss_its.sigma_t * dwivedi_stretching
    tr_dwivedi = dr.exp(-t * sigma_t_dwivedi)
    pdf[active_medium] = tr_dwivedi * sigma_t_dwivedi
    pdf[active_boundary_hit] = tr_dwivedi
    return tr, pdf


  def transmittance_pdf(
      self,
      sss_its: SSSInteraction,
      si: mi.SurfaceInteraction3f,
      active_medium: mi.Bool,
      active_boundary_hit: mi.Bool,
      dwivedi_stretching: mi.Float = 1.0,
  ):
    """Evaluates the transmittance and free-flight PDF.

    Requires a sampled SSS interaction and the next surface interaction with
    their respective active masks.

    Args:
      sss_its: The SSS interaction.
      si: The surface interaction.
      active_medium: Whether we are still in the medium.
      active_boundary_hit: Whether the ray hits the volume boundary.

    Returns:
      The free-flight PDF.
    """
    t = dr.minimum(sss_its.t, si.t)
    pdf = mi.Spectrum(0.0)
    sigma_t_dwivedi = sss_its.sigma_t * dwivedi_stretching
    tr_dwivedi = dr.exp(-t * sigma_t_dwivedi)
    pdf[active_medium] = tr_dwivedi * sigma_t_dwivedi
    pdf[active_boundary_hit] = tr_dwivedi
    return pdf


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


@dr.syntax
def sample_sss_scattering(
    mode: dr.ADMode,
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.Ray3f,
    sss_medium: SSSMedium,
    max_sss_depth: int,
    active_surface: mi.Bool,
    active_medium: mi.Bool,
    eta: int = mi.Float(1.0),
    sss_rr_depth: int = 0xFFFFFFFF,
    channel: mi.UInt32 = 0,
):
  """Samples SSS scattering events.

  Args:
    mode: The AD mode.
    scene: The Mitsuba scene.
    sampler: The Mitsuba sampler.
    ray: The ray to sample along.
    sss_medium: The SSS medium.
    max_sss_depth: The maximum SSS depth.
    active_surface: Whether the surface is active.
    active_medium: Whether the medium is active.
    eta: The Russian roulette stopping probability.
    sss_rr_depth: The depth at which to apply the Russian roulette stopping
      probability.
    channel: The channel to sample.

  Returns:
    The volume throughput, the number of SSS bounces, and the total SSS
    distance. (active_surface and active_medium are also updated)
  """
  sss_depth = mi.UInt32(0)
  sss_bounces = mi.UInt32(0)
  sss_distance = mi.Float(0.0)
  volume_throughput = mi.Spectrum(1.0)

  while dr.hint(
      active_medium,
      max_iterations=max_sss_depth,
      label="SSS Loop (%s)" % mode.name,
  ):
    sss_its = sss_medium.sample_interaction(
        ray, sampler.next_1d(active_medium), channel, active_medium
    )

    ray.maxt[active_medium & sss_its.is_valid()] = sss_its.t
    next_si = scene.ray_intersect(ray, active_medium)
    sss_its.t[active_medium & (next_si.t < sss_its.t)] = dr.inf

    # Either a volume interaction was found or the next interaction is a
    # surface
    active_boundary_hit = active_medium & ~sss_its.is_valid()
    active_medium &= sss_its.is_valid()
    # Reset maxt for the boundary hit for the true next surface interaction
    ray.maxt[active_boundary_hit] = dr.inf

    # Evaluate ratio of transmittance and free-flight PDF
    tr, free_flight_pdf = sss_medium.transmittance_eval_pdf(
        sss_its, next_si, active_medium, active_boundary_hit
    )
    tr_pdf = index_spectrum(free_flight_pdf, channel)

    # single scattering albedo (sigma_s / sigma_t) if non-spectrally varying
    volume_throughput[active_medium] *= sss_its.sigma_s * dr.select(
        tr_pdf > 0, (tr / tr_pdf), 0.0
    )
    # One if non-spectrally varying
    volume_throughput[active_boundary_hit] *= dr.select(
        tr_pdf > 0, (tr / tr_pdf), 0.0
    )

    sss_bounces[active_medium] += 1
    sss_distance[active_medium] += sss_its.t
    # Add final distance to boundary hit without adding an extra bounce
    # to not double count the extra sigma_t during the differentiable phase
    sss_distance[active_boundary_hit] += next_si.t

    # Sample isotropic phase function
    phase_wo = mi.warp.square_to_uniform_sphere(sampler.next_2d(active_medium))

    # Construct next ray
    ray[active_medium] = mi.Ray3f(sss_its.p, phase_wo)
    sss_depth[active_medium] += 1

    # -------------------- Stopping criterion ---------------------

    # Don't run another iteration if the throughput has reached zero
    volume_throughput_max = dr.max(volume_throughput)
    active_medium &= volume_throughput_max != 0
    active_boundary_hit &= volume_throughput_max != 0

    # Russian roulette stopping probability (must cancel out ior^2
    # to obtain unitless throughput) no min probability for volume scattering
    rr_prob = dr.minimum(volume_throughput_max * eta**2, 1.0)

    # Apply only further along the path since, this introduces variance
    rr_active = sss_depth >= sss_rr_depth
    volume_throughput[rr_active] *= dr.rcp(rr_prob)
    rr_continue = sampler.next_1d() < rr_prob
    active_medium &= ~rr_active | rr_continue
    active_medium &= sss_depth < max_sss_depth

    active_surface |= active_boundary_hit

  return volume_throughput, sss_bounces, sss_distance


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
  guiding_fraction = 0.0

  with dr.resume_grad(when=not primal):
    g = medium_shape_bsdf.eval_attribute(
        "hg_coefficient", medium_shape_si, active_medium
    )
    g_needs_grads = dr.grad_enabled(g)
    g_local = dr.detach(g)
    if dr.hint(dwivedi_guiding, mode="scalar"):
      guiding_fraction = 1.0 - dr.maximum(
          0.1, dr.max(dr.abs(g_local)) ** (1 / 3)
      )

  dwivedi_mis_weight = mi.Spectrum(1.0)

  # Whether to follow the Dwivedi guiding or the classic random walk, always
  # classic for the first distance sampling
  dwivedi_guided = mi.Bool(False)
  # TODO(pweier): Additionaly check for backward or forward guiding (for now, only forward)

  while dr.hint(
      active_medium,
      max_iterations=max_sss_depth,
      label="SSS Loop (%s)" % mode.name,
  ):
    # Evaluate Dwivedi guiding stretching of sigma_t
    dwivedi_stretching = dwivedi_utils.dwivedi_distance_stretching(
        dr.dot(ray.d, sss_medium.n),
        sss_medium.diffusion_length,
    )

    sss_its = sss_medium.sample_interaction(
        ray,
        sampler.next_1d(active_medium),
        channel,
        active_medium,
        dr.select(dwivedi_guided, dwivedi_stretching, 1.0),
    )

    ray.maxt[active_medium & sss_its.is_valid()] = sss_its.t
    # Compute a surface interaction that tracks derivatives arising
    # from differentiable shape parameters (position, normals, etc.)
    # In primal mode, this is just an ordinary ray tracing operation.
    with dr.resume_grad(when=not primal):
      si[active_medium] = scene.ray_intersect(ray, active_medium)
    sss_its.t[active_medium & (si.t < sss_its.t)] = dr.inf

    # Either a volume interaction was found or the next interaction is a
    # surface
    active_boundary_hit = active_medium & ~sss_its.is_valid()
    active_medium &= sss_its.is_valid()
    # Reset maxt for the boundary hit for the true next surface interaction
    ray.maxt[active_boundary_hit] = dr.inf

    # Evaluate ratio of transmittance and free-flight PDF
    tr, free_flight_pdf = sss_medium.transmittance_eval_pdf(
        sss_its,
        si,
        active_medium,
        active_boundary_hit,
        dr.select(dwivedi_guided, dwivedi_stretching, 1.0),
    )

    f = mi.Spectrum(1.0)
    f[active_medium] *= sss_its.sigma_s
    f[active_medium | active_boundary_hit] *= tr
    update_weight_matrix(
        p_over_f,
        free_flight_pdf,
        f,
        active_medium | active_boundary_hit,
    )

    sss_bounces[active_medium] += 1
    sss_distance[active_medium] += sss_its.t
    # Add final distance to boundary hit without adding an extra bounce
    # to not double count the extra sigma_t during the differentiable phase
    sss_distance[active_boundary_hit] += si.t

    # Check if performing Dwivedi guiding next
    dwivedi_guided[active_medium] = (
        sampler.next_1d(active_medium) < guiding_fraction
    )

    # Sample phase function
    phase_wo, phase_weight, phase_pdf = sample_spectral_hg(
        sss_its.g, ray, sampler.next_2d(active_medium), channel
    )
    phase_eval = phase_weight * phase_pdf

    # Probability of the dwivedi given the sampled phase function direction
    phase_dwivedi_pdf = dr.inv_two_pi * dwivedi_utils.pdf_dwivedi_directional(
        dr.dot(phase_wo, sss_medium.n), sss_medium.diffusion_length
    )

    # Sample dwivedi phase function cos_theta
    dwivedi_cos_theta = dwivedi_utils.sample_dwivedi_directional(
        sss_medium.diffusion_length, sampler.next_1d(dwivedi_guided)
    )
    dwivedi_wo = direction_from_cosine(
        sss_medium.n, dwivedi_cos_theta, sampler.next_1d(dwivedi_guided)
    )
    phase_wo[dwivedi_guided] = dwivedi_wo

    # phase_pdf contains the probability of the sampled technique,
    # either phase function or dwivedi
    phase_pdf[dwivedi_guided] = (
        dr.inv_two_pi
        * dwivedi_utils.pdf_dwivedi_directional(
            dwivedi_cos_theta, sss_medium.diffusion_length
        )
    )
    # Probability of the phase function for the sampled dwivedi direction
    dwivedi_phase_pdf = eval_spectral_hg(sss_its.g, dwivedi_wo, ray)
    phase_eval[dwivedi_guided] = eval_spectral_hg(sss_its.g, dwivedi_wo, ray)

    # Scale the phase function pdfs to account for the guiding fraction
    phase_pdf *= dr.select(
        dwivedi_guided, guiding_fraction, (1.0 - guiding_fraction)
    )
    dwivedi_phase_pdf *= 1.0 - guiding_fraction
    phase_dwivedi_pdf *= guiding_fraction

    # Compute the associated mis weight
    mis_weight = balance_mis_weight(
        phase_pdf,
        phase_dwivedi_pdf,
    )
    mis_weight[dwivedi_guided] = balance_mis_weight(
        phase_pdf,
        dwivedi_phase_pdf,
    )

    update_weight_matrix(
        p_over_f,
        phase_pdf,
        mis_weight * phase_eval,
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
    ray[active_medium] = mi.Ray3f(sss_its.p, phase_wo)
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


@dr.syntax
def sample_spectral_sss_scattering_naive(
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
    δL: mi.Spectrum,
    L: mi.Spectrum,
):
  """Samples SSS scattering events.

  Args:
    mode: The AD mode.
    scene: The Mitsuba scene.
    sampler: The Mitsuba sampler.
    ray: The ray to sample along.
    sss_medium: The SSS medium.
    max_sss_depth: The maximum SSS depth.
    active_surface: Whether the surface is active.
    active_medium: Whether the medium is active.
    eta: The Russian roulette stopping probability.
    sss_rr_depth: The depth at which to apply the Russian roulette stopping
      probability.

  Returns:
    The volume throughput, the number of SSS bounces, and the total SSS
    distance. (active_surface and active_medium are also updated)
  """
  primal = mode == dr.ADMode.Primal

  sss_depth = mi.UInt32(0)

  while dr.hint(
      active_medium,
      max_iterations=max_sss_depth,
      label="SSS Loop (%s)" % mode.name,
  ):
    with dr.resume_grad(when=not primal):
      sss_medium.populate(medium_shape_bsdf, medium_shape_si, active_medium)
      sss_its = sss_medium.sample_interaction(
          ray, sampler.next_1d(active_medium), channel, active_medium
      )
      sss_its.t = dr.detach(sss_its.t)

      ray.maxt[active_medium & sss_its.is_valid()] = sss_its.t
      # Compute a surface interaction that tracks derivatives arising
      # from differentiable shape parameters (position, normals, etc.)
      # In primal mode, this is just an ordinary ray tracing operation.
      with dr.resume_grad(when=not primal):
        si[active_medium] = scene.ray_intersect(ray, active_medium)
      sss_its.t[active_medium & (si.t < sss_its.t)] = dr.inf

      # Either a volume interaction was found or the next interaction is a
      # surface
      active_boundary_hit = active_medium & ~sss_its.is_valid()
      active_medium &= sss_its.is_valid()
      # Reset maxt for the boundary hit for the true next surface interaction
      ray.maxt[active_boundary_hit] = dr.inf

      # Evaluate ratio of transmittance and free-flight PDF
      tr, free_flight_pdf = sss_medium.transmittance_eval_pdf(
          sss_its, si, active_medium, active_boundary_hit
      )
      f = mi.Spectrum(1.0)
      f[active_medium] *= sss_its.sigma_s
      f[active_medium | active_boundary_hit] *= tr
      update_weight_matrix(
          p_over_f,
          dr.detach(free_flight_pdf),
          f,
          active_medium | active_boundary_hit,
      )
      if dr.hint(not primal and dr.grad_enabled(f), mode="scalar"):
        tr_pdf = index_spectrum(free_flight_pdf, channel)
        weight = mi.Spectrum(1.0)
        weight[active_medium] *= sss_its.sigma_s
        weight[active_medium | active_boundary_hit] *= dr.select(
            tr_pdf > 0.0, tr / dr.detach(tr_pdf), 0.0
        )

        # Differentiable version of the reflected radiance.
        L_volume = dr.select(
            active_medium | active_boundary_hit,
            dr.detach(L / dr.maximum(1e-8, weight)),
            0.0,
        )

        # Propagate derivatives from/to 'Lo' based on 'mode'
        if mode == dr.ADMode.Backward:
          dr.backward_from(δL * L_volume * weight)
        else:
          δL += dr.forward_to(L_volume * weight)

      # Sample isotropic phase function
      with dr.suspend_grad():
        phase_wo, phase_weight, phase_pdf = sample_spectral_hg(
            sss_its.g, ray, sampler.next_2d(active_medium), channel
        )
      update_weight_matrix(
          p_over_f,
          phase_pdf,
          phase_weight * phase_pdf,
          active_medium,
      )

      # Re evaluate the phase function value in an attached manner
      phase_eval = eval_spectral_hg(sss_its.g, phase_wo, ray)
      if dr.hint(not primal and dr.grad_enabled(phase_eval), mode="scalar"):
        # Differentiable version of the reflected radiance.
        L_volume = phase_eval * dr.select(
            active_medium,
            dr.detach(L / dr.maximum(1e-8, phase_eval)),
            0.0,
        )

        # Propagate derivatives from/to 'Lo' based on 'mode'
        if mode == dr.ADMode.Backward:
          dr.backward_from(δL * L_volume)
        else:
          δL += dr.forward_to(L_volume)

    # Construct next ray
    ray[active_medium] = mi.Ray3f(sss_its.p, phase_wo)
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

    # Apply only further along the path since, this introduces variance
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

  return


def albedo_inversion_isotropic(
    surface_colour: np.ndarray,
    scattering_radius: np.ndarray,
):
  """Computes the scattering albedo and extinction from surface albedo.

  Args:
    surface_colour: The surface colour of the object.
    scattering_radius: The scattering radius.
    anisotropy: The anisotropy of the phase function

  Returns:
    The scattering albedo and extinction coefficients.
  """

  a = surface_colour
  d = scattering_radius

  # -5.09406f * A + 2.61188f * A * A - 4.31805f * A * A * A)
  scattering_albedo = 1.0 - np.exp(
      -5.09406 * a + 2.61188 * a * a - 4.31805 * a * a * a
  )
  # scattering_albedo = 1.0 - np.exp(-3.25172*a*a*a + 1.2166*a*a - 4.75914*a)
  s = 1.9 - a + 3.5 * (a - 0.8) * (a - 0.8)

  # this trick is used in the cycles reference implementation
  scattering_albedo = np.clip(scattering_albedo, 0.0, 0.999999)

  # following cycle's implementation
  sigma_t = 1.0 / (d * s)

  # there is additional clamping and rescaling going on for low albedo values
  return scattering_albedo, sigma_t


def albedo_inversion(
    surface_colour: np.ndarray,
    scattering_radius: np.ndarray,
    anisotropy: np.ndarray,
):
  """Computes the scattering albedo and extinction from surface albedo.

  Args:
    surface_colour: The surface colour of the object.
    scattering_radius: The scattering radius.
    anisotropy: The anisotropy of the phase function

  Returns:
    The scattering albedo and extinction coefficients.
  """

  g = anisotropy
  g2 = np.power(anisotropy, 2)
  g3 = np.power(anisotropy, 3)
  g4 = np.power(anisotropy, 4)
  g5 = np.power(anisotropy, 5)
  g6 = np.power(anisotropy, 6)
  g7 = np.power(anisotropy, 7)

  # first, we compute the coefficients
  # can be vastly simplified for the isotropic case
  A = (
      1.8260523782
      + -1.28451056436 * g
      + -1.79904629312 * g2
      + 9.193932892020 * g3
      + -22.8215585862 * g4
      + 32.02348742590 * g5
      + -23.6264803333 * g6
      + 7.210670026580 * g7
  )

  B = 4.98511194385 + 0.127355959438 * np.exp(
      31.1491581433 * g
      + -201.847017512 * g2
      + 841.5760167230 * g3
      + -2018.09288505 * g4
      + 2731.715602860 * g5
      + -1935.41424244 * g6
      + 559.0090544740 * g7
  )

  C = (
      1.09686102424
      + -0.394704063468 * g
      + 1.0525811594100 * g2
      + -8.839637127260 * g3
      + 28.864323066100 * g4
      + -46.88029135810 * g5
      + 38.540283751800 * g6
      + -12.71810425380 * g7
  )

  D = (
      0.496310210422
      + 0.360146581622 * g
      + -2.15139309747 * g2
      + 17.88968992170 * g3
      + -55.2984010333 * g4
      + 82.06598224300 * g5
      + -58.5106008578 * g6
      + 15.84782950210 * g7
  )

  E = 4.23190299701 + 0.00310603949088 * np.exp(
      76.7316253952 * g
      + -594.356773233 * g2
      + 2448.883420300 * g3
      + -5576.68528998 * g4
      + 7116.601719120 * g5
      + -4763.54467887 * g6
      + 1303.531805500 * g7
  )

  F = (
      2.40602999408
      + -2.51814844609 * g
      + 9.184949083560 * g2
      + -79.2191708682 * g3
      + 259.0828682090 * g4
      + -403.613804597 * g5
      + 302.8571243600 * g6
      + -87.4370473567 * g7
  )

  linear_weight = np.power(surface_colour, 0.25)

  scattering_albedo = (1.0 - linear_weight) * A * np.power(
      np.arctan(B * surface_colour), C
  ) + linear_weight * D * np.power(np.arctan(E * surface_colour), F)

  # this trick is used in the cycles reference implementation
  scattering_albedo = np.clip(scattering_albedo, 0.0, 0.999999)

  # following cycle's implementation
  sigma_t = (1.0 / np.maximum(scattering_radius, 1e-16)) / (1.0 - g)

  # there is additional clamping and rescaling going on for low albedo values
  return scattering_albedo, sigma_t
