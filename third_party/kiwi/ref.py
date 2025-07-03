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

import numpy as np
with open('third_party/kiwi/mts_scene/volumes/albedo.vol', 'rb') as f:
    f.read(3)  # skip 'VOL'
    version = np.frombuffer(f.read(1), np.uint8)[0]
    dtype = np.frombuffer(f.read(4), np.int32)[0]
    xres = np.frombuffer(f.read(4), np.int32)[0]
    yres = np.frombuffer(f.read(4), np.int32)[0]
    zres = np.frombuffer(f.read(4), np.int32)[0]
    channels = np.frombuffer(f.read(4), np.int32)[0]
    print(f"Resolution: {xres}x{yres}x{zres}, Channels: {channels}")

mi.set_variant("cuda_ad_rgb")

scene_path="third_party/kiwi/mts_scene/kiwi_ref_deng.xml"
output_dir=Path("third_party/kiwi/references")

scene = mi.load_file(scene_path)
shapes = scene.shapes()
#print(scene)
print("=== Shape → BSDF===")
for i, shape in enumerate(shapes):
    bsdf = shape.bsdf()
    shape_name = shape.id()
    bsdf_name = bsdf.id() if bsdf is not None else "(None)"
    print(f"Shape[{i}] '{shape_name}' → BSDF: '{bsdf_name}'")

integrator = scene.integrator()
print(integrator) 

for i, sensor in enumerate(scene.sensors()):
    image = mi.render(scene, sensor=sensor, spp=1024)

    exr_path = output_dir / f"render_{i:02d}.exr"
    png_path = output_dir / f"render_{i:02d}.png"

    #mi.Bitmap(image).write(str(exr_path))

    tonemapped = mi.Bitmap(image).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                                          component_format=mi.Struct.Type.UInt8,
                                          srgb_gamma=True)
    tonemapped.write(str(png_path))
    print(f"Saved: {exr_path} and {png_path}")