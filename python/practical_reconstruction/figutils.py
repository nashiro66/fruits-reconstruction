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

"""Common figure functions and Matplotlib settings."""

import os
import subprocess

import drjit as dr  # type: ignore
import matplotlib
import matplotlib.pyplot as plt
import mitsuba as mi  # type: ignore
import numpy as np
from pathlib import Path
from core import mitsuba_io

if hasattr(matplotlib, 'style'):
  matplotlib.style.use('default')

if not mi.variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

# Just follow https://github.com/NVlabs/flip to setup on local machine
FLIP_EXECUTABLE = (
    "./flip/src/build/Release/flip.exe"
)

PAPER_FIG_OUTPUT_DIR = './practical_reconstruction/figures'

# These values are compute from \the\columnwidth and \the\textwidth in latex
# divided by 72 (as Matplotlib internally assumes a DPI of 72 for conversion.)
COLUMN_WIDTH = 3.37704722222
TEXT_WIDTH = 7.08743055556

DEFAULT_FONTSIZE = 8  # Font size used by captions
DEFAULT_FONTSIZE_SMALL = 6

# Might require installing a few extra packages:
# sudo apt install dvipng texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra cm-super

LARGE_STEPS_NAME_SHORT = 'Nicolet et al.'  # [2021]
LARGE_STEPS_NAME_LONG = 'Nicolet et al. 2021'  # [2021]
GRAD_FILTERING_NAME = 'Chang et al.'  # [2024]
GRAD_FILTERING_NAME_LONG = 'Chang et al. 2024'  # [2024]

_LATEX_PREAMBLE = r"""\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{bm}
\usepackage{bbm}
\usepackage{siunitx}

\newcommand*\diff{\mathop{}\!\mathrm{d}}
\newcommand{\vx}{\mathbf{x}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\bomega}{\bm{\omega}}
"""

_MATPLOTLIB_STYLE = {
    'font.family': 'sans-serif',
    'font.serif': 'Linux Libertine',
    'text.usetex': True,
    'text.color': 'black',
    'font.size': DEFAULT_FONTSIZE,
    'axes.titlesize': DEFAULT_FONTSIZE,
    'axes.labelsize': DEFAULT_FONTSIZE_SMALL,
    'xtick.labelsize': DEFAULT_FONTSIZE_SMALL - 2,
    'ytick.labelsize': DEFAULT_FONTSIZE_SMALL - 2,
    'legend.fontsize': DEFAULT_FONTSIZE_SMALL,
    'figure.titlesize': DEFAULT_FONTSIZE,
    'text.latex.preamble': _LATEX_PREAMBLE,
    'pgf.preamble': _LATEX_PREAMBLE,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.4,
    'xtick.major.size': 0.5,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 0.25,
    'xtick.minor.width': 0.5,
    'ytick.major.size': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 0.25,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 0.75,
    'patch.linewidth': 0.5,
    'grid.linewidth': 0.5,
    # 'axes.titley': -0.3,
    'pgf.texsystem': 'pdflatex',
    'figure.dpi': 250,  # Increases figure size in Colab.
}


def _initialize():
  """Initializes the Matplotlib style."""
  matplotlib.style.use('default')
  matplotlib.rcParams.update(_MATPLOTLIB_STYLE)
  # sns.set()
  # matplotlib.rcParams.update(_MATPLOTLIB_STYLE)


_initialize()


def read_img(
    filename: str,
    exposure: float = 0,
    tonemap: bool = True,
    background_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
  """Reads an image from the given file.

  Args:
    filename: The filename of the image to read.
    exposure: The exposure of the image.
    tonemap: Whether to tonemap the image.
    background_color: The background color of the image. This is useful when
      reading images with an alpha channel

  Returns:
    The loaded image.
  """
  bitmap = mitsuba_io.read_bitmap(filename)
  if tonemap:
    if background_color is not None:
      img = bitmap.convert(
          mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, False
      )
      background_color = np.array(background_color).ravel()[None, None, :]
      img = np.array(img)
      img = img[:, :, :3] + (1.0 - img[..., -1][..., None]) * background_color
    else:
      img = bitmap.convert(
          mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False
      )
      img = np.array(img)
    img = img * 2**exposure
    img = mi.Bitmap(img).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True
    )
    return np.clip(img, 0, 1)
  else:
    return np.array(bitmap)


def tonemap(image: np.ndarray | mi.TensorXf) -> np.ndarray:
  """Tonemaps the given image.

  Args:
    image: The image to tonemap.

  Returns:
    The tonemapped image.
  """
  image = mi.Bitmap(image)
  image = image.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)
  return np.clip(image, 0, 1)


def savefig(
    figure: plt.Figure,
    name: str,
    fig_directory: Path,
    fig_subdirectory: Path | None = None,
    dpi: int = 300,
    pad_inches: float = 0.005,
    bbox_inches: str = 'tight',
    compress: bool = True,
    target_width: float | None = None,
    backend=None,
    file_format='pdf',
):
  """Saves a figure to a file.

  Args:
    name: The name of the figure file.
    fig_subdirectory: The subdirectory to save the figure to. If None, the
      default directory will be used.
    dpi: The DPI of the figure.
    pad_inches: The pad size of the figure.
    bbox_inches: The bounding box specification (according to Matplotlib).
    compress: Whether to compress the figure using ghostscript.
    target_width: If specified, the input figure will be resized to guarantee a
      given output size when using bbox_inches == tight. Note that this means
      the figure size will change when invoking this function, which is usually
      not a problem.
    backend: The savefig backend to use.

  Returns:
    The filename of the (uncompressed) figure.
  """
  if fig_subdirectory:
    output_dir = fig_directory / fig_subdirectory
  else:
    output_dir = fig_directory

  if target_width is not None:
    if bbox_inches != 'tight':
      raise ValueError(
          'bbox_inches must be "tight" when target_width is specified.'
      )
    force_post_crop_size(figure, target_width)

  output_dir.mkdir(parents=True, exist_ok=True)
  filename = output_dir / f'{name}.{file_format}'
  uncompressed_filename = filename
  if compress:
    if file_format != 'pdf':
      raise ValueError('Only PDF files can be compressed.')
    filename = str(filename).replace('.pdf', '_original.pdf')
  figure.savefig(
      filename,
      format=file_format,
      dpi=dpi,
      bbox_inches=bbox_inches,
      pad_inches=pad_inches,
      backend=backend,
  )
  if compress:
    ghostscript_command = (
        f'gs -o {uncompressed_filename} -dQUIET -f -dNOPAUSE -dBATCH'
        ' -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.6'
        f' -dDownsampleColorImages=false -DownsampleGrayImages=false {filename}'
    )
    subprocess.call(ghostscript_command, shell=True)
  return uncompressed_filename


def set_aspect(ax: plt.Axes, aspect: float) -> None:
  """Sets the aspect of the given axis.

  Args:
    ax: The axis to set the aspect of.
    aspect: The aspect to set.
  """
  x_left, x_right = ax.get_xlim()
  y_bottom, y_top = ax.get_ylim()
  ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * aspect)


def disable_ticks(ax: plt.Axes) -> None:
  """Disables the ticks of the given axis.

  Args:
    ax: The axis to disable the ticks of.
  """
  ax.axes.get_xaxis().set_ticklabels([])
  ax.axes.get_yaxis().set_ticklabels([])
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])


def disable_border(ax: plt.Axes) -> None:
  """Disables the border of the given axis.

  Args:
    ax: The axis to disable the border of.
  """
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)


def time_to_string(duration: float) -> str:
  """Converts a time in seconds to a string representation.

  Args:
    duration: A time duration in seconds.

  Returns:
    A string representation of the duration as XXd XXh XXm XXs.
  """
  duration = round(duration)
  m, s = divmod(duration, 60)
  h, m = divmod(m, 60)
  d, h = divmod(h, 24)
  result = f'{d}d ' if d > 0 else ''
  result += f'{h}h ' if h > 0 else ''
  result += f'{m}m ' if m > 0 else ''
  result += f'{s}s'
  return result


# Somehow setting the font size to 6 defaults to a weird non 6 size, using
# 5.9999 instead seems to work.
def math_label(label: str, font_size: float = 6):
  return r'\fontsize{' + str(font_size - 0.0001) + r'}{12}\selectfont' + label


def gridspec_aspect(
    n_rows: int,
    n_cols: int,
    w: float | list[float],
    h: float | list[float],
    wspace: float = 0,
    hspace: float = 0,
):
  if isinstance(w, int) or isinstance(w, float):
    Ws = n_cols * w
  elif isinstance(w, list) or isinstance(w, tuple):
    Ws = sum(w)
  else:
    raise ValueError(f'Unsupported type for w: {type(w)}')

  if isinstance(h, int) or isinstance(h, float):
    Hs = n_rows * h
  elif isinstance(h, list) or isinstance(h, tuple):
    Hs = sum(h)
  else:
    raise ValueError(f'Unsupported type for h: {type(h)}')

  w_spacing = wspace * Ws / n_cols
  h_spacing = hspace * Hs / n_rows

  return (Ws + (n_cols - 1) * w_spacing) / (Hs + (n_rows - 1) * h_spacing)


def force_post_crop_size(figure: plt.Figure, target_width: float):
  """For a given figure, ensures a target size after compute tight bounding box.

  A Matplotlib figure might have some undesired white padding around the main
  content. While this is removed when saving using bbox_inches='tight',
  the figure then has a size different from the originally specified target
  size.

  This function can be used to resize the figure to have the right size
  *after* saving with a tight bounding box.

  Args:
    figure: The input figure which will be resized.
    target_width: The target size of the figure.
  """

  bbox = figure.get_tightbbox(figure.canvas.get_renderer())
  figure.set_size_inches(
      target_width / (bbox.x1 - bbox.x0) * figure.get_size_inches()
  )

  # Ensure correctness post-condition.
  bbox = figure.get_tightbbox(figure.canvas.get_renderer())
  # print(f'Target width: {target_width}, final size: {bbox.x1 - bbox.x0}')


def diagonal_split_image(image_1, image_2, offset=0, angle=45):
  # Ensure both images have the same dimensions
  if image_1.shape != image_2.shape:
    # Resize smaller to larger
    if image_1.shape[1] > image_2.shape[1]:
      image_2 = image_util.resize_to_width(image_2, image_1.shape[1])
    else:
      image_1 = image_util.resize_to_width(image_1, image_2.shape[1])

  # Parameters for the diagonal
  # Create a diagonal mask
  rows, cols, _ = image_1.shape
  center_row, center_col = rows // 2, cols // 2
  # Convert angle to radians
  angle_rad = np.radians(angle)

  # Compute coordinates relative to the rotated diagonal
  xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
  rotated_x = (xs - center_col) * np.cos(angle_rad) + (
      ys - center_row
  ) * np.sin(angle_rad)
  mask = rotated_x > offset
  # Combine the two images using the mask
  combined_image = np.zeros_like(image_1)
  combined_image[mask] = image_2[mask]
  combined_image[~mask] = image_1[~mask]
  # Calculate the diagonal line's start and end points
  # The diagonal line is where rotated_x == offset
  x_line = np.arange(cols)
  y_line = (offset - (x_line - center_col) * np.cos(angle_rad)) / np.sin(
      angle_rad
  ) + center_row

  # Clip the line coordinates to the image boundaries
  valid_mask = (y_line >= 0) & (y_line < rows)
  x_line = x_line[valid_mask]
  y_line = y_line[valid_mask]

  return combined_image, x_line, y_line


def crop_image(
    img: np.ndarray, crop_offset: tuple[int, int], crop_size: tuple[int, int]
):
  return img[
      crop_offset[1] : crop_offset[1] + crop_size[1],
      crop_offset[0] : crop_offset[0] + crop_size[0],
  ]


def teaser_tonemap(image, gamma=1.1, exposure=2):
  return image_util.tonemap((image ** (gamma) * np.sqrt(2) ** exposure))


def ACESFilm(x, apply_linear_to_srgb=True):
  data = dr.unravel(mi.Color3f, x.array)
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  output = dr.clamp(
      (data * (a * data + b)) / (data * (c * data + d) + e), 0.0, 1.0
  )
  if apply_linear_to_srgb:
    for i in range(3):
      output[i] = mi.math.linear_to_srgb(output[i])
  return mi.TensorXf(dr.ravel(output), shape=x.shape)


def flip_error(img, ref):
  # Just follow https://github.com/NVlabs/flip to setup on local machine
  if not os.path.exists(FLIP_EXECUTABLE):
    raise Exception(f"Flip executable not found at {FLIP_EXECUTABLE}")

  ref_path = "./tmp/ref.png"
  test_path = "./tmp/test.png"
  ref_bitmap = mi.Bitmap(ref).convert(
    pixel_format=mi.Bitmap.PixelFormat.RGB,
    component_format=mi.Struct.Type.UInt8,
    srgb_gamma=False,
  )
  test_bitmap = mi.Bitmap(img).convert(
    pixel_format=mi.Bitmap.PixelFormat.RGB,
    component_format=mi.Struct.Type.UInt8,
    srgb_gamma=False,
  )
  mitsuba_io.write_bitmap(ref_bitmap,test_path)
  mitsuba_io.write_bitmap(test_bitmap,ref_path)

  # Define the command and arguments
  command = [
      FLIP_EXECUTABLE,
      "--reference",
      ref_path,
      "--test",
      test_path,
      "--basename",
      "flip",
      "--directory",
      "/tmp/",
      "-ppd",
      "120"
  ]
  result = subprocess.run(command, capture_output=True, text=True)
  return np.array(mitsuba_io.read_bitmap("/tmp/flip.png").convert(
      pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.Float32,
  ))