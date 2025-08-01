import bpy
import math
import mathutils

n = 50
angle_rad = math.radians(45)
phi = (1 + 5 ** 0.5) / 2
camera_distance = 30
light_distance = 20
target = mathutils.Vector((0.3, 0.5, 6.0))
light_size = 10.0

for i in range(n):
    y = 1 - (i / (n - 1)) * 2
    radius = math.sqrt(1 - y ** 2)
    theta = 2 * math.pi * i / phi
    x = math.cos(theta) * radius
    z = math.sin(theta) * radius

    cam_data = bpy.data.cameras.new(f"Camera_{i}")
    cam_obj = bpy.data.objects.new(f"Camera_{i}", cam_data)
    cam_pos = mathutils.Vector((x * camera_distance, y * camera_distance, z * camera_distance)) + target
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = (cam_pos - target).to_track_quat('Z', 'Y').to_euler()
    bpy.context.collection.objects.link(cam_obj)
    
    # light 1
    for j in range(2):
        view_dir = (cam_obj.location - target).normalized()
        base_light_pos = target - view_dir * light_distance
        
        if j==0:
            temp_vec = mathutils.Vector((0, 0, 1))
        else:
            temp_vec = mathutils.Vector((0, 0, -1))
            
        if abs(view_dir.dot(temp_vec)) > 0.999:
            temp_vec = mathutils.Vector((0, 1, 0))
        rotation_axis = view_dir.cross(temp_vec).normalized()
        rotation = mathutils.Matrix.Rotation(angle_rad, 4, rotation_axis)
        offset = base_light_pos - target
        rotated_offset = rotation @ offset
        light_pos = target + rotated_offset

        light_data = bpy.data.lights.new(name=f"AreaLight_{i}", type='AREA')
        light_data.shape = 'SQUARE'
        light_data.size = light_size
        light_data.energy = 1000

        light_obj = bpy.data.objects.new(name=f"AreaLight_{i}", object_data=light_data)
        light_obj.location = light_pos
        light_obj.rotation_euler = (target - light_pos).to_track_quat('-Z', 'Y').to_euler()

        bpy.context.collection.objects.link(light_obj)
