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

"""Module that collects and registers all custom textures."""

import mitsuba as mi  # type: ignore


def _textures_variant_callback(old: str, new: str) -> None:
  """Imports and registers all custom textures.

  This follows Mitsuba's structure to allow changing the variant for custom
  plugins, see mitsuba/python/ad/integrators/__init__.py.

  Args:
    old: The old variant.
    new: The new variant.
  """
  del old  # unused

  if new is None or new.startswith('scalar'):
    return

  import importlib
  from core.textures import mipmap_flat

  importlib.reload(mipmap_flat)
  mipmap_flat.register()


def register() -> None:
  """Registers all custom textures."""
  mi.detail.add_variant_callback(_textures_variant_callback)
  _textures_variant_callback('', 'dummy')
