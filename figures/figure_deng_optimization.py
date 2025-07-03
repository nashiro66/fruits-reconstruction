#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

def set_root_path():
    if os.getcwd().endswith('figures'): 
        os.chdir('../')
set_root_path()
sys.path.append('python/')
sys.path = [p for p in sys.path if "unbiased-inverse-volume-rendering" not in p]
print(os.getcwd())
print(sys.executable)
os.environ.pop("PYTHONPATH")


# In[2]:


from pathlib import Path
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

from practical_reconstruction import optimization_cli
from core import integrators
from core import bsdfs
from core import textures

integrators.register()
bsdfs.register()
textures.register()


# In[ ]:


def format_float(f):
  """Formats a float such that 0.1 becomes "0_1", 10.0 becomes "10_0", etc."""
  return str(f).replace('.', '_')

scene_name = 'kiwi'
technique = 'mipmap_pyramid'
base_learning_rate = 0.0005

skip_existing = False

print(
    f'******** Running {technique} with base learning rate'
    f' {base_learning_rate} ********'
)

override_bindings = []
result_folder = f'results/{scene_name}/{technique}'

result_folder += f'_lr_{format_float(base_learning_rate)}'
override_bindings.append(
    f'SceneConfig.base_learning_rate={base_learning_rate}'
)

# Ensure that the default tmp folder is used
override_bindings.append("SceneConfig.tmp_folder=''")
override_bindings.append(
    f'SceneConfig.use_gradient_filtering=False'
)

override_bindings.append(
    f"SceneConfig.result_folder='{result_folder}'"
)

if technique == 'gradient_filtering':
    gin_config_name = f'{scene_name}/naive'
else:
    gin_config_name = f'{scene_name}/{technique}'

print(f'Next result location: {result_folder}')
if skip_existing and Path(result_folder).exists():
    print('Skipping, already present')
else:
    # Run the config
    optimization_cli.run_config(gin_config_name, override_bindings, sss_config=True)


# # Figure starts here

# In[ ]:


import matplotlib.pyplot as plt
from practical_reconstruction import figutils
from matplotlib import gridspec

def figure_grid_setup(image_shape,image_crop_shape,n_columns = 7,inner_space=0.0,outer_space=0.1,figwidth=figutils.TEXT_WIDTH):
  # Image aspect ratios
  h, w = image_shape
  h_crop, w_crop = image_crop_shape

  top_inner_rows = 1
  top_inner_cols = n_columns
  # Spacing in the inner gridspec
  top_inner_wspace = inner_space
  # same vertical spacing as horizontal spacing
  top_inner_hspace = top_inner_wspace * figutils.gridspec_aspect(
      n_rows=1, n_cols=1, w=[w]*top_inner_cols, h=[h]
  )
  top_height_ratios = [h]
  top_inner_aspect = figutils.gridspec_aspect(
      n_rows=top_inner_rows,
      n_cols=top_inner_cols,
      w=[w] * top_inner_cols,
      h=top_height_ratios,
      wspace=top_inner_wspace,
      hspace=top_inner_hspace,
  )
  # print(top_inner_aspect)

  # Spacing in the main griddpec
  outer_rows = 3
  outer_cols = 1
  outer_wspace = 0.0
  outer_hspace = outer_space
  # If width is 1, we need the sum of the inverses for the height (single column)
  # If height is 1, we need the sum for the width (single row)
  outer_aspect = figutils.gridspec_aspect(
      n_rows=outer_rows,
      n_cols=outer_cols,
      w=1,
      h=[1 / top_inner_aspect] * outer_rows,
      wspace=outer_wspace,
      hspace=outer_hspace,
  )

  fig = plt.figure(
      1, figsize=(figwidth, figwidth / outer_aspect)
  )

  outer_gs = fig.add_gridspec(
      outer_rows,
      outer_cols,
      hspace=outer_hspace,
      wspace=outer_wspace,
      height_ratios=[1 / top_inner_aspect] * outer_rows,
      width_ratios=[1]*outer_cols,
  )

  top_inner_gss = []
  for row in range(outer_rows):
    top_inner_gs = gridspec.GridSpecFromSubplotSpec(
        top_inner_rows,
        top_inner_cols,
        subplot_spec=outer_gs[row],
        wspace=top_inner_wspace,
        hspace=top_inner_wspace,
        width_ratios=[h] * top_inner_cols,
        height_ratios=top_height_ratios,
    )
    top_inner_gss.append(top_inner_gs)

  return (
      fig,
      (top_inner_gss, top_inner_rows, top_inner_cols),
  )


# In[ ]:


import drjit as dr
import numpy as np
from core import mitsuba_io
from core import image_util

def l2_error(ref, img):
  return dr.mean(dr.square(ref - img)).array[0]


def l1_error(ref, img):
  return dr.mean(dr.abs(ref - img)).array[0]

sensor_indices = [20, 1, 25, 12]

lr = '0_0005'
normalmap_lr = '0_01'
scene_folder = 'results/kiwi'

ref_scene_folder = (
    'third_party/kiwi/unmasked_references'
)
ref_masked_scene_folder = (
    'third_party/kiwi/references'
)
deng_scene_folder = (
    'third_party/kiwi/deng_optimized/'
)

crop_sizes = [
    (110, 110),
    (126, 126),
    (94, 94),
    (113, 113),
]

crop_offsets = [
    (159, 92),
    (150, 75),
    (168, 96),
    (157, 86),
]

our_images = []
ref_masked_images = []
ref_images = []
deng_images = []
errors_l2_ours = []
errors_l2_deng = []

boost = np.sqrt(2)

for sensor_idx, crop_size, crop_offset in zip(sensor_indices,crop_sizes,crop_offsets):

  our_filename = f'{scene_folder}/mipmap_pyramid_lr_{lr}/frames/camera_{sensor_idx:03d}_iter_4095_spp_8192.exr'
  our_image = mitsuba_io.read_bitmap(our_filename).convert(
      pixel_format=mi.Bitmap.PixelFormat.RGB,
      component_format=mi.Struct.Type.Float32,
  )
  our_images.append(image_util.tonemap(boost*our_image))
  ref_masked_filename = (
      f'{ref_masked_scene_folder}/ref_view_{sensor_idx:03d}.exr'
  )
  ref_masked_image = mitsuba_io.read_bitmap(ref_masked_filename).convert(
      pixel_format=mi.Bitmap.PixelFormat.RGB,
      component_format=mi.Struct.Type.Float32,
  )
  ref_masked_images.append(image_util.tonemap(ref_masked_image))

  ref_filename = f'{ref_scene_folder}/kiwi_{sensor_idx:05d}.exr'
  ref_image = mitsuba_io.read_bitmap(ref_filename).convert(
      pixel_format=mi.Bitmap.PixelFormat.RGB,
      component_format=mi.Struct.Type.Float32,
  )
  ref_images.append(
      image_util.tonemap(
          boost*image_util.resize_to_width(ref_image, ref_image.size()[0] // 2)
      )
  )
  deng_filename = f'{deng_scene_folder}/optimized_{sensor_idx:03d}_spp_256.exr'
  deng_image = mitsuba_io.read_bitmap(deng_filename).convert(
      pixel_format=mi.Bitmap.PixelFormat.RGB,
      component_format=mi.Struct.Type.Float32,
  )
  deng_images.append(image_util.tonemap(boost*deng_image))

  errors_l2_ours.append(
      l2_error(
          figutils.crop_image(
              mi.TensorXf(ref_masked_image), crop_offset, crop_size
          ),
          figutils.crop_image(
              mi.TensorXf(our_image), crop_offset, crop_size
          ),
      )
  )
  errors_l2_deng.append(
      l2_error(
          figutils.crop_image(
              mi.TensorXf(ref_masked_image), crop_offset, crop_size
          ),
          figutils.crop_image(
              mi.TensorXf(deng_image), crop_offset, crop_size
          ),
      )
  )

our_images = [
    figutils.crop_image(np.array(image), crop_offsets[i], crop_sizes[i])
    for i,image in enumerate(our_images)
]
ref_images = [
    figutils.crop_image(np.array(image), crop_offsets[i], crop_sizes[i])
    for i,image in enumerate(ref_images)
]
deng_images = [
    figutils.crop_image(np.array(image), crop_offsets[i], crop_sizes[i])
    for i,image in enumerate(deng_images)
]


# In[ ]:


FIGURE_DIR = "figures/pdfs"
FIGURE_NAME = "deng_comparison"

def error_format(error, scale):
  return f"{error*scale:.3f}"

(
    fig,
    (top_inner_gss, top_inner_rows, top_inner_cols),
) = figure_grid_setup(
    our_images[0].shape[:2],
    our_images[0].shape[:2],
    n_columns=len(sensor_indices),
    inner_space=0.03,
    outer_space=0.15,
    figwidth=figutils.COLUMN_WIDTH,
)


scale = 1000
scale_txt = figutils.math_label(r"\text{$\times 10^3$}")

line_width = 0.75
crop_color = 'orange'
crop_color = (55/255.0,118/255.0,171/255.0,1.0)

row_titles = [
    "Deng et al. 2022",
    "Ours",
    "Reference",
]

for row, top_inner_gs in enumerate(top_inner_gss):
  if row == 0:
    images = deng_images
    errors = errors_l2_deng
  elif row == 1:
    images = our_images
    errors = errors_l2_ours
  elif row == 2:
    images = ref_images

  for col in range(top_inner_cols):
    ax = fig.add_subplot(top_inner_gs[col])
    ax.spines[:].set_color(crop_color)
    ax.spines[:].set_linewidth(line_width)
    figutils.disable_ticks(ax)
    image = images[col]

    # alpha_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) + 255
    # black_pixels = np.all(image == 0, axis=-1)
    # alpha_channel[black_pixels] = 0
    # image_with_alpha = np.dstack((image, alpha_channel))

    ax.imshow(image, aspect="equal")
    if col == 0:
      ax.set_ylabel(row_titles[row],labelpad=1.5)

    if row == 0 or row == 1:
      label = error_format(errors[col], scale)
    elif row == 2:
      label = r"$L_2$ " + scale_txt + " error" if col == 0 else ""
    ax.set_xlabel(label, labelpad=1.5)

figutils.force_post_crop_size(fig, figutils.COLUMN_WIDTH)


# In[7]:


figutils.savefig(
    fig,
    name=Path(FIGURE_NAME),
    fig_directory=Path(FIGURE_DIR),
    dpi=300,
    pad_inches=0.005,
    bbox_inches="tight",
    compress=False,
    target_width=figutils.COLUMN_WIDTH,
    backend=None,
)


# In[ ]:




