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

"""Exposes type aliases for different Variable classes."""

from variables import array
from variables import constant
from variables import displacement_map
from variables import image_pyramid
from variables import lookat_transform
from variables import normal_map
from variables import tensor
from variables import laplacian_smoothing
from variables import differentiable_mipmap
from variables import flatmip_aware_pyramid
from variables import variable


ArrayVariable = array.ArrayVariable
Variable = variable.Variable
TensorVariable = tensor.TensorVariable
Constant = constant.Constant
DisplacementMapVariable = displacement_map.DisplacementMapVariable
ImagePyramidVariable = image_pyramid.ImagePyramidVariable
LookAtTransformVariable = lookat_transform.LookAtTransformVariable
NormalMapVariable = normal_map.NormalMapVariable
LaplacianSmoothingVariable = laplacian_smoothing.LaplacianSmoothingVariable
DifferentiableMipmapVariable = differentiable_mipmap.DifferentiableMipmapVariable
FlatMipAwareImagePyramidVariable = flatmip_aware_pyramid.FlatMipAwareImagePyramidVariable