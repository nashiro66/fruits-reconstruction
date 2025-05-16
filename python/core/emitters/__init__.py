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

"""Module that collects and registers all custom emitters."""

import mitsuba as mi  # type: ignore


def _emitters_variant_callback(old: str, new: str) -> None:
  """Imports and registers all custom emitters.

  This follows Mitsuba's structure to allow changing the variant for custom
  plugins, see third_party/py/mitsuba/python/ad/integrators/__init__.py.

  Args:
    old: The old variant.
    new: The new variant.
  """
  del old  # unused

  if new is None or new.startswith('scalar'):
    return

  # pylint: disable=g-import-not-at-top
  import importlib
  from core.emitters import ies_emitter

  importlib.reload(ies_emitter)
  ies_emitter.register()

  from core.emitters import switch_emitter

  importlib.reload(switch_emitter)
  switch_emitter.register()


def register() -> None:
  """Registers all custom emitters."""
  mi.detail.add_variant_callback(_emitters_variant_callback)
  _emitters_variant_callback('', 'dummy')
