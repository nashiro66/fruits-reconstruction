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

"""Defines a custom integrator that renders homogeneous volumes with SSS."""

from __future__ import annotations

from typing import Optional

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

from core.integrators import volume_utils

if not mi.variant():
  mi.set_variant("cuda_ad_rgb")

SSSMedium = volume_utils.SSSMedium
SSSInteraction = volume_utils.SSSInteraction
mis_weight_from_matrix = volume_utils.mis_weight_from_matrix
update_weight_matrix = volume_utils.update_weight_matrix


class PrbPathVolumeIntegrator(mi.ad.integrators.common.RBIntegrator):
  """Integrator that renders point of entry homogeneous medium.

  Many assumptions are made in the following:
  - The medium is embedded in the SSS parameters of a custom BSDF
  - The medium is isotropic
  - No NEE on volume interactions
  - No null-scattering events/null BSDF is supported
  - Scalar extinction only, no channel-dependent scattering is supported
  """

  def __init__(self, props: mi.llvm_ad_rgb.Properties):
    super().__init__(props)
    max_path_depth = props.get("max_path_depth", 2)
    self.max_path_depth = max_path_depth if max_path_depth != -1 else 0xFFFFFFFF

    max_sss_depth = props.get("max_sss_depth", 256)
    self.max_sss_depth = max_sss_depth if max_sss_depth != -1 else 0xFFFFFFFF

    rr_depth = props.get("rr_depth", 5)
    self.rr_depth = rr_depth if rr_depth != -1 else 0xFFFFFFFF

    sss_rr_depth = props.get("sss_rr_depth", 5)
    self.sss_rr_depth = sss_rr_depth if sss_rr_depth != -1 else 0xFFFFFFFF

    self.reuse_sss_params = props.get("reuse_sss_params", True)
    self.hide_emitters = props.get("hide_emitters", False)
    self.dwivedi_guiding = props.get("dwivedi_guiding", False)

  @dr.syntax
  def sample(
      self,
      mode: dr.ADMode,
      scene: mi.Scene,
      sampler: mi.Sampler,
      ray: mi.Ray3f,
      δL: Optional[mi.Spectrum],
      state_in: Optional[mi.Spectrum],
      active: mi.Bool,
      **kwargs,  # Absorbs unused arguments
  ) -> tuple[mi.Spectrum, mi.Bool, list[mi.Float], mi.Spectrum]:
    primal = mode == dr.ADMode.Primal

    L = mi.Spectrum(0 if primal else state_in)  # Radiance accumulator
    δL = mi.Spectrum(
        δL if δL is not None else 0
    )  # Differential/adjoint radiance

    # Copy input arguments to avoid mutating the caller's state and ensure
    # that ray differentials are propagated correctly
    tmp_ray = mi.RayDifferential3f(dr.detach(ray))
    if isinstance(ray, mi.RayDifferential3f) and ray.has_differentials:
      tmp_ray.o_x = dr.detach(ray.o_x)
      tmp_ray.o_y = dr.detach(ray.o_y)
      tmp_ray.d_x = dr.detach(ray.d_x)
      tmp_ray.d_y = dr.detach(ray.d_y)
      tmp_ray.has_differentials = True
    ray = tmp_ray

    bsdf_ctx = mi.BSDFContext()
    # Throughput is kept in a matrix to enable (robust) spectral MIS
    p_over_f = dr.full(dr.matrix_t(mi.Spectrum), 1.0)
    eta = mi.Float(1.0)

    depth = mi.UInt32(0)
    sss_medium = dr.zeros(SSSMedium)

    active_medium = mi.Bool(False)
    active_surface = mi.Bool(active)

    prev_si = dr.zeros(mi.SurfaceInteraction3f)
    prev_bsdf = dr.zeros(mi.BSDFPtr)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Bool(True)

    # Sample a color channel for the volume sampling
    channel = mi.UInt32(0)
    if dr.hint(mi.is_rgb, mode="scalar"):
      n_channels = dr.size_v(mi.Spectrum)
      channel = mi.UInt32(
          dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1)
      )

    while dr.hint(
        active,
        max_iterations=self.max_path_depth,
        label="Path Replay Backpropagation (%s)" % mode.name,
    ):
      # Used to determine if we were in a medium at the start of this iteration
      was_in_medium = mi.Bool(active_medium)
      si = dr.zeros(mi.SurfaceInteraction3f)

      # Sample the spectrally varying volume, also updates the passed masks,
      # the throughput matrix p_over_f and the surface interaction si adequately
      (
          sss_bounces,
          sss_distance,
          δg,
      ) = volume_utils.sample_spectral_sss_scattering(
          mode=mode,
          scene=scene,
          sampler=sampler,
          ray=ray,
          si=si,
          sss_medium=sss_medium,
          max_sss_depth=self.max_sss_depth,
          p_over_f=p_over_f,
          active_surface=active_surface,
          active_medium=active_medium,
          eta=eta,
          sss_rr_depth=self.sss_rr_depth,
          channel=channel,
          medium_shape_bsdf=prev_bsdf,
          medium_shape_si=prev_si,
          dwivedi_guiding=self.dwivedi_guiding,
      )

      # -------- Evaluate the volume scattering differentiably ----------
      δ_volume = mi.Spectrum(1.0)
      if dr.hint(not primal, mode="scalar"):
        with dr.resume_grad():
          # Re-evaluate SSS parameters differentiably
          albedo = prev_bsdf.eval_attribute(
              "single_scattering_albedo", prev_si, was_in_medium
          )
          sigma_t = prev_bsdf.eval_attribute(
              "extinction_coefficient", prev_si, was_in_medium
          )
          g = prev_bsdf.eval_attribute("hg_coefficient", prev_si, was_in_medium)
          sigma_s = albedo * sigma_t

          # Naive approach (subject to huge numerical issues)
          # vol_scatter_eval = (sigma_s ** volume_bounces_float) * dr.exp(
          #     -volume_distance * sigma_t
          # )
          # volume_scattering *= dr.replace_grad(
          #     1, dr.rcp(dr.detach(vol_scatter_eval)) * vol_scatter_eval
          # )

          # Applying this https://www.wolframalpha.com/input?i=%5Bd%2Fdx+%5Bf%28x%29%5EN%5D*g%28x%29%5D+%2F+%28f%28x%29%5EN+*+g%28x%29%29
          # and then https://www.wolframalpha.com/input?i=%5Bd%2Fdx+%28e%5E-%28h%28x%29%29%29%5D
          # to simplify the above naive expression and avoid numerical issues
          # Here f(x) = sigma_s
          # g(x) = dr.exp(-h(x))
          # h(x) = volume_distance * sigma_t
          # g(x) / g'(x) = -h'(x)

          sss_bounces_float = mi.Float(sss_bounces)
          fx_grad = sigma_s
          hx_grad = sss_distance * sigma_t

          # δg was computed manually (i.e. it's not a differentiable parameter)
          # To set the gradient of g with δg we therefore just propagate their
          # product since d/dg (g*δg) = δg -> the gradient we need for g
          # dr.detach has no effect but make it clear δg is a constant here
          δ_volume *= dr.replace_grad(1, g * dr.detach(δg))
          δ_volume *= dr.replace_grad(
              1,
              sss_bounces_float * fx_grad * dr.rcp(dr.detach(fx_grad))
              - hx_grad,
          )

      # Compute a surface interaction that tracks derivatives arising
      # from differentiable shape parameters (position, normals, etc.)
      # In primal mode, this is just an ordinary ray tracing operation.
      with dr.resume_grad(when=not primal):
        # Only perform an intersection if we were not in a medium at the start
        # of this iteration. (Always sends a ray towards a light at depth 0)
        needs_intersection = active_surface & ~was_in_medium
        si[needs_intersection] = scene.ray_intersect(
            ray,
            ray_flags=mi.RayFlags.All,
            coherent=(depth == 0),
            active=needs_intersection,
        )

      if isinstance(ray, mi.RayDifferential3f) and ray.has_differentials:
        si.compute_uv_partials(ray)

      bsdf = si.bsdf(ray)

      # ---------------------- Direct emission ----------------------

      prev_bsdf_pdf[was_in_medium] = 0.0

      # Compute MIS weight for emitter sample from previous bounce
      ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

      mis = mi.ad.integrators.common.mis_weight(
          prev_bsdf_pdf,
          scene.pdf_emitter_direction(
              prev_si, ds, (~prev_bsdf_delta & ~was_in_medium)
          ),
      )

      # Hide the environment emitter if necessary
      if dr.hint(self.hide_emitters, mode="scalar"):
        active_surface &= ~((depth == 0) & ~si.is_valid())

      with dr.resume_grad(when=not primal):
        throughput_mis = mis_weight_from_matrix(p_over_f)
        Le = throughput_mis * mis * ds.emitter.eval(si, active_surface)

      # # ---------------- Intersection with emitters ----------------
      # # Only count direct hits for now
      # count_direct = depth == 0
      # emitter = si.emitter(scene)
      # active_e = active_surface & count_direct & (emitter != None)
      # Le = emitter.eval(si, active_e)

      # --------------------- Emitter sampling ---------------------

      active_surface &= (depth + 1 < self.max_path_depth) & si.is_valid()
      active_e = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
      ds, em_weight = scene.sample_emitter_direction(
          si, sampler.next_2d(active_e), test_visibility=primal, active=active_e
      )
      active_e &= ds.pdf != 0.0
      with dr.resume_grad(when=not primal):
        if dr.hint(not primal, mode="scalar"):
          # Explicitly trace shadow ray, since we need derivatives
          # of the emitter's surface.
          # ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
          shadow_ray = dr.detach(si).spawn_ray(ds.d)
          si_emitter = scene.ray_intersect(shadow_ray, active_e)
          shadowed = si_emitter.t < ds.dist * (1 - mi.math.ShadowEpsilon)
          active_e &= ~shadowed

          # Replace direction sample values with differentiable quantities.
          ds.p = si_emitter.p
          ds.uv = si_emitter.uv
          ds.n = si_emitter.sh_frame.n

          # Given the detached emitter sample, *recompute* its
          # contribution with AD to enable light source optimization
          ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
          em_val = scene.eval_emitter_direction(si, ds, active_e)
          em_weight = dr.select(
              active_e,
              dr.replace_grad(em_weight, em_val / ds.pdf),
              0.0,
          )
          dr.disable_grad(ds.d)

        # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        # bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_e)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_e)
        mis_em = dr.select(
            ds.delta,
            1,
            mi.ad.integrators.common.mis_weight(ds.pdf, bsdf_pdf_em),
        )
        # Includes the volume scattering throughput contribution
        Lr_dir = throughput_mis * mis_em * bsdf_value_em * em_weight

      # ------------------ BSDF sampling for next intersection -----------------

      bsdf_sample, bsdf_weight = bsdf.sample(
          bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_surface
      )

      # ---- Update loop variables based on current interaction -----

      L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
      # Update throughput matrix
      update_weight_matrix(
          p_over_f,
          bsdf_sample.pdf,
          bsdf_weight * bsdf_sample.pdf,
          active_surface,
      )
      eta *= bsdf_sample.eta

      tmp_ray = mi.RayDifferential3f(si.spawn_ray(si.to_world(bsdf_sample.wo)))
      # Propagate ray differentials to the spawned ray assuming identical
      # rayfoot size.
      if isinstance(ray, mi.RayDifferential3f) and ray.has_differentials:
        tmp_ray.o_x = ray.o_x
        tmp_ray.o_y = ray.o_y
        tmp_ray.d_x = ray.d_x
        tmp_ray.d_y = ray.d_y
        tmp_ray.has_differentials = True
      ray = tmp_ray

      # -------------------- Stopping criterion ---------------------

      # Don't run another iteration if the throughput has reached zero
      throughput_mis = mis_weight_from_matrix(p_over_f)
      throughput_max = dr.max(throughput_mis)
      active_surface &= throughput_max != 0

      # Russian roulette stopping probability (must cancel out ior^2
      # to obtain unitless throughput, enforces a minimum probability)
      rr_prob = dr.minimum(throughput_max * eta**2, 0.95)

      # Apply only further along the path since, this introduces variance
      rr_active = depth >= self.rr_depth
      update_weight_matrix(
          p_over_f, dr.detach(rr_prob), mi.Spectrum(1.0), rr_active
      )
      rr_continue = sampler.next_1d() < rr_prob
      active_surface &= ~rr_active | rr_continue

      # Evaluate the surface scattering differentiably while accounting for the
      # volume scattering
      if dr.hint(not primal, mode="scalar"):
        with dr.resume_grad():

          # Recompute 'wo' to propagate derivatives to cosine term
          wo = si.to_local(ray.d)

          # Re-evaluate BSDF * cos(theta) differentiably
          bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_surface)

          # Detached version of the above term and inverse
          bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
          inv_bsdf_val_detach = dr.select(
              bsdf_val_detach != 0, dr.rcp(bsdf_val_detach), 0
          )

          # Differentiable version of the indirect reflected radiance.
          Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)

          # Differentiable Monte Carlo estimate of all contributions
          Lo = Le + (Lr_dir + Lr_ind) * δ_volume

          # Propagate derivatives from/to 'Lo' based on 'mode'
          if mode == dr.ADMode.Backward:
            dr.backward_from(δL * Lo)
          else:
            δL += dr.forward_to(Lo)

      depth[si.is_valid()] += 1

      # Fill in SSS medium for next interaction
      sss_medium.populate(bsdf, si, active_surface)
      sss_entry = sss_medium.is_sss_entry(ray, si)
      active_medium = active_surface & sss_entry
      active_surface &= ~active_medium
      # Don't trace another bounce in the medium if we won't hit a light in the
      # next iteration.
      active_medium &= depth + 1 < self.max_path_depth

      # Revert to previous SSS parameters if we continue scattering inside
      # the medium.
      if dr.hint(self.reuse_sss_params, mode="scalar"):
        reuse_sss_params = (
            was_in_medium & sss_entry & (mi.Frame3f.cos_theta(si.wi) < 0.0)
        )
        sss_medium.populate(
            prev_bsdf, prev_si, active_medium & reuse_sss_params
        )

      # Store previous interaction data for SSS medium
      prev_si = dr.detach(si)
      prev_bsdf = dr.detach(bsdf)
      prev_bsdf_pdf = bsdf_sample.pdf
      prev_bsdf_delta = mi.has_flag(
          bsdf_sample.sampled_type, mi.BSDFFlags.Delta
      )

      active &= active_surface | active_medium

    return (
        L if primal else δL,  # Radiance/differential radiance
        (depth != 0),  # Ray validity flag for alpha blending
        [],  # Empty typle of AOVs
        L,  # State the for differential phase
    )


def register():
  """Registers the detached direct integrator plugin."""
  mi.register_integrator(
      "prb_path_volume", lambda props: PrbPathVolumeIntegrator(props)  # pylint: disable=unnecessary-lambda
  )
