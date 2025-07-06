import os
import sys
from pathlib import Path

def set_root_path():
    if os.getcwd().endswith('figures'): 
        os.chdir('../')

def _update_light_position(params, sensor, light_positions):
  idx = int(sensor.id()[len('camera_') :])
  # Update light position for the selected sensor
  params['emit-Point.position'] = light_positions[idx]
  params.update()

def _gantry_light_positions():
  light_positions = [
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [-2.944557817004439, 48.8898011620204, -41.4402460343309],
      [27.055610795691802, 48.81986339495665, -31.610352043829998],
      [41.3180656055926, 48.73481598693242, -3.4462441460169386],
      [4.4671667393215575, 47.1369277980911, 42.78219333472567],
      [-42.91042711255302, 47.2987396659678, 3.9648433877314604],
      [-33.3138670015183, 47.353262970810604, -27.76333185326814],
      [-4.0928649644330495, 47.3407616054259, -43.412834290840195],
      [27.63531231151012, 47.2685587001079, -33.816397641445796],
      [43.2847288874415, 47.178949737549196, -4.59548434384601],
      [33.6881687764068, 47.1244264327064, 27.132690897152997],
      [33.6881687764068, 47.1244264327064, 27.132690897152997],
      [43.2847288874415, 47.178949737549196, -4.59548434384601],
      [27.63531231151012, 47.2685587001079, -33.816397641445796],
      [-4.0928649644330495, 47.3407616054259, -43.412834290840195],
      [-33.3138670015183, 47.353262970810604, -27.76333185326814],
      [-42.91042711255302, 47.2987396659678, 3.9648433877314604],
      [4.4671667393215575, 47.1369277980911, 42.78219333472567],
      [43.2160052121517, 47.0472861267224, -4.74011971286851],
      [4.61129874588445, 47.0047545442678, 42.71339045984761],
      [-27.110711342651502, 47.076678220198616, 33.2390870886382],
      [-42.842128441427, 47.1662190544112, 4.1088962216802996],
      [-33.3677017693281, 47.2209252406101, -27.613111405679],
      [-4.237421975160156, 47.20875063686579, -43.34461395103581],
      [27.4845881133758, 47.136826960935004, -33.870310579826004],
      [27.4845881133758, 47.136826960935004, -33.870310579826004],
      [-4.237421975160156, 47.20875063686579, -43.34461395103581],
      [-33.3677017693281, 47.2209252406101, -27.613111405679],
      [-42.842128441427, 47.1662190544112, 4.1088962216802996],
      [-27.110711342651502, 47.076678220198616, 33.2390870886382],
      [4.61129874588445, 47.0047545442678, 42.71339045984761],
      [43.2160052121517, 47.0472861267224, -4.74011971286851],
      [54.346621438817294, -0.786676450502069, 24.2689468400264],
      [-24.57995504968848, -0.724855391664175, 53.815623248847],
      [-55.69939927018449, -0.601267884668769, 20.471749327836],
      [-54.1263850774641, -0.503287498684284, -25.11075868475832],
      [-20.78236285181735, -0.48830981497388004, -56.230285802138],
      [24.80019141104173, -0.565108557522178, -54.657435093579295],
      [55.9196356315378, -0.688696064517584, -21.31356117256785],
      [55.9196356315378, -0.688696064517584, -21.31356117256785],
      [24.80019141104173, -0.565108557522178, -54.657435093579295],
      [-20.78236285181735, -0.48830981497388004, -56.230285802138],
      [-54.1263850774641, -0.503287498684284, -25.11075868475832],
      [-55.69939927018449, -0.601267884668769, 20.471749327836],
      [-24.57995504968848, -0.724855391664175, 53.815623248847],
      [54.346621438817294, -0.786676450502069, 24.2689468400264],
      [54.6571493603019, 4.41870496840774, 24.41354929025032],
      [39.6303138216189, 47.0780994649407, 17.415384223131202],
      [1.6598691824590588, 65.52149640366119, -0.2676721498317754],
      [-37.0116130568947, 48.94500399409722, -18.2771252295649],
  ]
  return [mi.Point3f(light_position) for light_position in light_positions]

ref_spp=128
def save_images(scene, dir):
    print("start rendering")
    for i, sensor in enumerate(scene.sensors()):
      image = mi.render(scene, sensor=sensor, spp=ref_spp)
      bitmap = mi.Bitmap(image).convert(
          pixel_format=mi.Bitmap.PixelFormat.RGB,
          component_format=mi.Struct.Type.UInt8,
          srgb_gamma=True
      )

      png_path = dir / f"view_{i:03d}.png"
      bitmap.write(str(png_path))
      print(f"saving: {i}")
    print(f"Saved: {dir}")

key1="medium1.sigma_t.data"
key2="medium1.albedo.data"
def optimize(scene, params, opt, ref_images):
   for it in range(iteration_count):
    total_loss = 0.0
    for i, sensor in enumerate(scene.sensors()):
        img = mi.render(scene, params, sensor=sensor, spp=spp, seed=it)
        loss = dr.mean(dr.square(img - ref_images[i]))
        dr.backward(loss)
        opt.step()
        opt[key1] = dr.clip(opt[key1], 1e-6, 100.0)
        opt[key2] = dr.clip(opt[key2], 1e-6, 100.0)
        params.update(opt)
        total_loss += loss
        print(loss)
    print(f"Iteration {it:02d}: error={total_loss}", end='\r')

set_root_path()
sys.path.append('python/')
sys.path = [p for p in sys.path if "unbiased-inverse-volume-rendering" not in p]
print(os.getcwd())
print(sys.executable)
os.environ.pop("PYTHONPATH")

from pathlib import Path
import mitsuba as mi
import drjit as dr
# from practical_reconstruction import scene_configuration
mi.set_variant('cuda_ad_rgb')

# from practical_reconstruction import optimization_cli
# from core import integrators
# from core import bsdfs
# from core import textures
# from practical_reconstruction import scene_preparation

# integrators.register()
# bsdfs.register()
# textures.register()

import numpy as np
mi.set_variant("cuda_ad_rgb")

# ref_scene_path="third_party/kiwi/mts_scene/kiwi_ref.xml"
ref_dir=Path("third_party/kiwi/references")

# ref_scene = mi.load_file(ref_scene_path)

# save_images(init_scene, ref_dir)

init_scene_path="third_party/kiwi/mts_scene/kiwi_init.xml"
init_scene = mi.load_file(init_scene_path)
print(init_scene)

#save_images(init_scene, Path("third_party/kiwi/init"))

ref_images=[]
for i, sensor in enumerate(init_scene.sensors()):
    exr_path = ref_dir / f"render_view_{i:03d}.exr"
    bmp = mi.Bitmap(str(exr_path))
    tensor = mi.TensorXf(bmp)
    ref_image = mi.Bitmap(tensor)
    ref_images.append(tensor)

params = mi.traverse(init_scene)
print(params)

opt = mi.ad.Adam(lr=0.02)
opt[key1] = params[key1]
opt[key2] = params[key2]
params.update(opt)
iteration_count = 1
spp = 128

optimize(init_scene, params, opt, ref_images)
#save_images(init_scene, Path("third_party/kiwi/intermediate"))

# opt[key1] = dr.upsample(opt[key1], shape=(74, 60, 64))
# opt[key2] = dr.upsample(opt[key2], shape=(34, 18, 18))
# params.update(opt)
# print("updated opt")

# # upscale
# #save_images(init_scene, Path("third_party/kiwi/upscale"))
# optimize(init_scene, params, opt, ref_images)

# opt[key1] = dr.upsample(opt[key1], shape=(148, 120, 128))
# opt[key2] = dr.upsample(opt[key2], shape=(68, 36, 36))
# params.update(opt)
# print("updated opt")
# optimize(init_scene, params, opt, ref_images)

# opt[key1] = dr.upsample(opt[key1], shape=(296, 240, 256))
# opt[key2] = dr.upsample(opt[key2], shape=(136, 72, 72))
# params.update(opt)
# print("updated opt")
# optimize(init_scene, params, opt, ref_images)

# opt[key1] = dr.upsample(opt[key1], shape=(592, 480, 512))
# opt[key2] = dr.upsample(opt[key2], shape=(272, 144, 144))
# params.update(opt)
# print("updated opt")
# optimize(init_scene, params, opt, ref_images)

# save_images(init_scene, Path("third_party/kiwi/output"))

sigma_t = opt[key1]
albedo = opt[key2]
grid_sigma_t = mi.VolumeGrid(sigma_t)
grid_sigma_t.write('output/sigma_t.vol')
grid_albedo = mi.VolumeGrid(albedo)
grid_albedo.write('output/albedo.vol')