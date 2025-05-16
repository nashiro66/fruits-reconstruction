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

"""Utilities for Dwivedi sampling."""

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np


if not mi.variant():
  mi.set_variant("llvm_ad_rgb")


# Equation (10-11) from "Supplemental Material: Improving the DwivediSampling Scheme" Meng et al. 2016
def meng_diffusion_length(alpha, force_low=False, force_high=False):
  alpha = np.clip(alpha, 0, 0.9999)

  kappa_low = 1 - 2 * np.exp(-2 / alpha) * (
      1
      + ((4 - alpha) / alpha) * np.exp(-2 / alpha)
      + ((24 - 12 * alpha + alpha**2) / (alpha**2)) * np.exp(-4 / alpha)
      + ((512 - 384 * alpha + 72 * alpha**2 - 3 * alpha**3) / (alpha**3))
      * np.exp(-6 / alpha)
  )
  kappa_high = np.sqrt(3 * (1 - alpha)) * (
      1
      - (2 / 5) * (1 - alpha)
      - (12 / 175) * (1 - alpha) ** 2
      - (2 / 125) * (1 - alpha) ** 3
      - (166 / 67375) * (1 - alpha) ** 4
  )

  if force_low:
    return 1 / kappa_low
  elif force_high:
    return 1 / kappa_high
  else:
    return np.where(alpha < 0.56, 1 / kappa_low, 1 / kappa_high)


# Equation (67) from "Zero-Variance Theory for Efficient Subsurface Scattering" Eugene D'Eon 2020
def diffusion_length(alpha):
  alpha = dr.clip(alpha, 0, 0.9999)
  return 1.0 / dr.sqrt(
      1.0 - dr.power(alpha, 2.44294 - 0.0215813 * alpha + 0.578637 / alpha)
  )


def pdf_dwivedi_directional(w_z, v_0):
  log_ratio = (v_0 + 1) / (v_0 - 1)
  return 1.0 / (dr.log(log_ratio) * (v_0 - w_z))


def sample_dwivedi_directional(v_0, rand):
  return v_0 - (v_0 + 1) * dr.power((v_0 - 1) / (v_0 + 1), rand)


def pdf_dwivedi_distance(t, w_z, alpha, sigma_t):
  v_0 = diffusion_length(alpha)
  sigma_t_prime = sigma_t * (1 - w_z / v_0)
  return sigma_t_prime * np.exp(-sigma_t_prime * t)


def dwivedi_distance_stretching(w_z, v_0):
  return (1 - w_z / v_0)
