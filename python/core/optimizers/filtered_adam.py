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

"""Filtered Adam optimizer."""

from __future__ import annotations

from collections import defaultdict

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

@dr.syntax
def a_trous_cross_bilateral_post_filtering(
    dtype,
    mean_texture: mi.Texture2f,
    variance_texture: mi.Texture2f,
    theta_texture: mi.Texture2f,
    kernel_2d: mi.Float,
    uv_coords: mi.Point2f,
    offsets: mi.Point2i,
    image_res: tuple[int, int, int],
    sigma_d,
    step_width,
    log_domain_filtering,
) -> tuple[mi.Color3f | mi.Color1f, mi.Color3f | mi.Color1f]:
  """Applies the A-trous cross bilateral post-filtering.

  This function implements the A-trous cross bilateral post-filtering as
  described
  in the paper *Spatiotemporal Bilateral Gradient Filtering for Inverse
  Rendering*
  by Chang et al., SIGGRAPH Asia 2024.

  Args:
    dtype: The data type to use for the computation Color1f(Float) or Color3f.
    mean_texture: The mean texture to filter.
    variance_texture: The variance texture to filter.
    theta_texture: The theta texture used for the edge stopping filtering.
    kernel_2d: The 2D discrete gaussian kernel to use for filtering.
    uv_coords: Pre-compute UV coordinates in [0,1] to evaluate textures.
    offsets: The raveled 2d offsets in the kernel.
    image_res: The image/parameter resolution.
    sigma_d: The sigma_d parameter controlling the strength of the edge stopping
      filter.
    step_width: The step_width computed from the number of filters applied in
      the a-trous algorithm.
    log_domain_filtering: Whether to apply the filtering in log domain or not.

  Returns:
    A tuple containing the filtered mean and variance.
  """
  # Auxiliary input representing the optimized parameter theta
  theta_val = dtype(theta_texture.eval(uv_coords))
  if log_domain_filtering:
    theta_val = dr.log(dr.abs(theta_val) + 1e-8)

  # Initialize per-pixel accumulated filtered mean and variance
  m_weighted_sum = dr.zeros(dtype)
  v_weighted_sum = dr.zeros(dtype)
  step = mi.Vector2f(step_width / image_res[0], step_width / image_res[1])
  m_cum_w = dr.zeros(mi.Float)
  v_cum_w = dr.zeros(mi.Float)
  eps = 1e-30 if dtype == mi.Color1f else 1e-10

  i = mi.UInt32(0)
  while dr.hint(i < dr.width(kernel_2d)):
    uv = uv_coords + dr.gather(mi.Point2i, offsets, index=i) * step
    # Necessary to match Chang et. al as they don't handle borders
    out_of_bounds = mi.Bool(False)
    out_of_bounds[(uv.x > 1.0) | (uv.y > 1.0) | (uv.x < 0.0) | (uv.y < 0.0)] = (
        True
    )

    m_tmp = dtype(mean_texture.eval(uv))
    v_tmp = dtype(variance_texture.eval(uv))
    # Edge stopping filtering using parameter texture
    theta_tmp = dtype(theta_texture.eval(uv))
    if log_domain_filtering:
      theta_tmp = dr.log(dr.abs(theta_tmp) + 1e-8)
    t = theta_val - theta_tmp

    dist2 = dr.norm(t)
    w_d = dr.exp(-dist2 / (sigma_d + eps))

    # Fetch isotropic gaussian kernel weights
    kernel_weight = dr.gather(mi.Float, kernel_2d, index=i)

    # Update filtered mean and variance per-pixel sum
    m_weighted_sum[~out_of_bounds] += m_tmp * w_d * kernel_weight
    m_cum_w[~out_of_bounds] += w_d * kernel_weight

    v_weighted_sum[~out_of_bounds] += v_tmp * w_d * kernel_weight
    v_cum_w[~out_of_bounds] += w_d * kernel_weight
    i += 1

  # Normalize filtered mean and variance
  m_result = dr.select(m_cum_w > 0, m_weighted_sum / m_cum_w, m_weighted_sum)
  v_result = dr.select(v_cum_w > 0, v_weighted_sum / v_cum_w, v_weighted_sum)
  return m_result, v_result


def post_filter_gradients(
    theta, m_t, v_t, sigma_d, F, log_domain_filtering
) -> tuple[mi.TensorXf, mi.TensorXf]:
  """Applies the A-trous cross bilateral post-filtering to the gradients.

  This function implements the A-trous cross bilateral post-filtering as
  described in the paper *Spatiotemporal Bilateral Gradient Filtering for
  Inverse Rendering* by Chang et al., SIGGRAPH Asia 2024.

  Args:
    theta: The optimized parameter theta.
    m_t: The mean texture to filter.
    v_t: The variance texture to filter.
    sigma_d: The sigma_d parameter controlling the strength of the edge stopping
      filter.
    F: The number of filters applied in the a-trous algorithm.
    log_domain_filtering: Whether to apply the filtering in log domain or not.

  Returns:
    A tuple containing the filtered mean and variance.
  """
  param_res = theta.shape
  dtype = mi.Color1f if param_res[2] == 1 else mi.Color3f

  # Create textures for the optimized parameter theta and the gradients.
  # Making sure to not use the hardware accelerationg to avoid costly texture
  # creation synchronization, the texture then essentially behaves like a tensor
  # with handy texture evaluation/interpoliation/clamping methods.
  theta_texture = mi.Texture2f(
      theta, filter_mode=dr.FilterMode.Nearest, use_accel=False
  )
  mean_texture = mi.Texture2f(
      m_t, filter_mode=dr.FilterMode.Nearest, use_accel=False
  )
  variance_texture = mi.Texture2f(
      v_t, filter_mode=dr.FilterMode.Nearest, use_accel=False
  )

  # Pre-compute kernel weights for the 3x3 discrete gaussian kernel
  # 5x5 kernel (1d separable) [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]
  kernel_2d = mi.Float([
      0.00390625,
      0.015625,
      0.0234375,
      0.015625,
      0.00390625,
      0.015625,
      0.0625,
      0.09375,
      0.0625,
      0.015625,
      0.0234375,
      0.09375,
      0.140625,
      0.09375,
      0.0234375,
      0.015625,
      0.0625,
      0.09375,
      0.0625,
      0.015625,
      0.00390625,
      0.015625,
      0.0234375,
      0.015625,
      0.00390625,
  ])
  dr.make_opaque(kernel_2d)

  # Pre-compute offsets for the 5x35discrete gaussian kernel (linearized for
  # efficient gathers later).
  kernel_size = 5
  offsets = mi.Point2i(
      dr.meshgrid(
          dr.arange(
              mi.Int32,
              -(kernel_size // 2),
              kernel_size // 2 + 1,
          ),
          dr.arange(
              mi.Int32,
              -(kernel_size // 2),
              kernel_size // 2 + 1,
          ),
      )
  )
  dr.make_opaque(offsets)

  # Pre-compute UV coordinates for texture evaluation
  u, v = dr.meshgrid(
      dr.linspace(mi.Float, 0.0, 1.0, param_res[0], endpoint=False),
      dr.linspace(mi.Float, 0.0, 1.0, param_res[1], endpoint=False),
  )
  uv_coords = mi.Point2f(u, v)
  # Offset coordinates by half a voxel to hit the center of the new voxels
  for i in range(2):
    uv_coords[i] += 0.5 / param_res[i]

  for i in range(F):
    step_width = mi.Float(1 << i)
    denoised_m, denoised_v = a_trous_cross_bilateral_post_filtering(
        dtype=dtype,
        mean_texture=mean_texture,
        variance_texture=variance_texture,
        theta_texture=theta_texture,
        kernel_2d=kernel_2d,
        uv_coords=uv_coords,
        offsets=offsets,
        image_res=param_res,
        sigma_d=sigma_d,
        step_width=step_width,
        log_domain_filtering=log_domain_filtering,
    )
    denoised_m, denoised_v = dr.ravel(denoised_m), dr.ravel(denoised_v)
    dr.schedule(denoised_m, denoised_v)

    # Prepare the textures for the next a-trous filtering iteration
    mean_texture.set_value(denoised_m)
    variance_texture.set_value(denoised_v)

  return (mean_texture.tensor(), variance_texture.tensor())


class FilteredAdam(mi.ad.optimizers.Optimizer):
  """Implements the Adam optimizer.

  Based on the paper *Adam: A Method for Stochastic Optimization* by Kingman and
  Ba, ICLR 2015 with some modifications to filter gradients following the paper
  *Spatiotemporal Bilateral  Gradient Filtering for Inverse Rendering* by Chang
  et al., SIGGRAPH Asia 2024.

  When optimizing many variables (e.g. a high resolution texture) with
  momentum enabled, it may be beneficial to restrict state and variable
  updates to the entries that received nonzero gradients in the current
  iteration (``mask_updates=True``).
  In the context of differentiable Monte Carlo simulations, many of those
  variables may not be observed at each iteration, e.g. when a surface is
  not visible from the current camera. Gradients for unobserved variables
  will remain at zero by default.
  If we do not take special care, at each new iteration:

  1. Momentum accumulated at previous iterations (potentially very noisy)
     will keep being applied to the variable.
  2. The optimizer's state will be updated to incorporate ``gradient = 0``,
     even though it is not an actual gradient value but rather lack of one.

  Enabling ``mask_updates`` avoids these two issues. This is similar to
  `PyTorch's SparseAdam optimizer
  <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.
  """

  def __init__(
      self,
      lr,
      beta_1=0.9,
      epsilon=1e-8,
      mask_updates=False,
      uniform=False,
      params: dict | None = None,
      sigma_d: float = 0.1,
      a_trous_steps: int = 5,
      log_domain_filtering: bool = False,
  ):
    """Parameter ``lr``:

        learning rate

    Parameter ``beta_1``:
        controls the exponential averaging of first order gradient moments

    Parameter ``mask_updates``:
        if enabled, parameters and state variables will only be updated in a
        given iteration if it received nonzero gradients in that iteration

    Parameter ``uniform``:
        if enabled, the optimizer will use the 'UniformAdam' variant of Adam
        [Nicolet et al. 2021], where the update rule uses the *maximum* of
        the second moment estimates at the current step instead of the
        per-element second moments.

    Parameter ``params`` (:py:class:`dict`):
        Optional dictionary-like object containing parameters to optimize.

    Parameter ``sigma_d``:
        controls the edge stopping filter strength

    Parameter ``a_trous_steps``:
        controls the number of filters applied in the a-trous algortithm (F in
        the paper)

    Parameter ``log_domain_filtering``:
        if enabled, the filtering is done in log domain
    """
    assert 0 <= beta_1 < 1 and lr > 0 and epsilon > 0

    self.beta_1 = beta_1
    # Use same parameterization as in the paper
    self.beta_2 = 1 - dr.square(1.0 - beta_1)
    self.epsilon = epsilon
    self.mask_updates = mask_updates
    self.uniform = uniform
    self.t = defaultdict(lambda: 0)
    self.sigma_d = sigma_d
    self.a_trous_steps = a_trous_steps
    self.log_domain_filtering = log_domain_filtering
    super().__init__(lr, params)

  def step(self):
    """Take a gradient step"""
    for k, p in self.variables.items():
      self.t[k] += 1
      lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (
          1 - self.beta_1 ** self.t[k]
      )
      lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

      lr_t = self.lr_v[k] * lr_scale
      g_p = dr.grad(p)
      shape = dr.shape(g_p)

      if shape == 0:
        continue
      elif shape != dr.shape(self.state[k][0]):
        # Reset state if data size has changed
        self.reset(k)

      m_tp, v_tp = self.state[k]
      m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p
      v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.square(g_p)
      if self.mask_updates:
        nonzero = g_p != 0.0
        m_t = dr.select(nonzero, m_t, m_tp)
        v_t = dr.select(nonzero, v_t, v_tp)

      # ----------------------Apply gradient post-filtering---------------------

      # Only filter tensors
      if len(p.shape) == 3:
        assert not dr.grad_enabled(m_t)
        assert not dr.grad_enabled(v_t)

        m_t, v_t = post_filter_gradients(
            dr.detach(p),
            m_t,
            v_t,
            sigma_d=self.sigma_d,
            F=self.a_trous_steps,
            log_domain_filtering=self.log_domain_filtering,
        )

      # ------------------------------------------------------------------------

      self.state[k] = (m_t, v_t)
      dr.schedule(self.state[k])

      if self.uniform:
        step = lr_t * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
      else:
        step = lr_t * m_t / (dr.sqrt(v_t) + self.epsilon)
      if self.mask_updates:
        step = dr.select(nonzero, step, 0.0)
      u = dr.detach(p) - step
      u = type(p)(u)
      dr.enable_grad(u)
      self.variables[k] = u
      dr.schedule(self.variables[k])

    dr.eval()

  def reset(self, key):
    """Zero-initializes the internal state associated with a parameter"""
    p = self.variables[key]
    shape = dr.shape(p) if dr.is_tensor_v(p) else dr.width(p)
    self.state[key] = (
        dr.zeros(dr.detached_t(p), shape),
        dr.zeros(dr.detached_t(p), shape),
    )
    self.t[key] = 0

  def __repr__(self):
    return (
        'Adam[\n'
        '  variables = %s,\n'
        '  lr = %s,\n'
        '  betas = (%g, %g),\n'
        '  eps = %g\n'
        ']'
        % (
            list(self.keys()),
            dict(self.lr, default=self.lr_default),
            self.beta_1,
            self.beta_2,
            self.epsilon,
        )
    )
