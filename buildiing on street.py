import bpy
import random
import math
import bmesh
class Building:
    dist = 5  # Minimum distance between buildings
    sidewalk = 5  # Width of the sidewalk
    data = {
        'Big': {
            'Height': [50, 10],
            'Width': [30, 5],
            'Depth': [25, 4]
        },
        'Mean': {
            'Height': [30, 8],
            'Width': [20, 4],
            'Depth': [15, 3]
        },
        'Small': {
            'Height': [15, 5],
            'Width': [10, 2],
            'Depth': [8, 2]
        }
    }
    def __init__(self, street_size='Mean'):
        height = max(5, random.gauss(self.data[street_size]['Height'][0], self.data[street_size]['Height'][1]))
        width = max(5, random.gauss(self.data[street_size]['Width'][0], self.data[street_size]['Width'][1]))
        depth = max(5, random.gauss(self.data[street_size]['Depth'][0], self.data[street_size]['Depth'][1]))
        self.size = [height, width, depth]
        self.center_location = (0, 0)  # Will be set later
        self.rotation = 0  # Will be set later
def create_building(building):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    # Set scale
    obj.scale = (building.size[1], building.size[2], building.size[0])
    # Set location and rotation
    obj.location = (building.center_location[0], building.center_location[1], building.size[0] / 2)
    obj.rotation_euler[2] = building.rotation
def create_street(length, min_width=15, max_width=25):
    width = random.uniform(min_width, max_width)
    curve_factor = random.uniform(0, length / 4)  # Random curvature
    angle_offset = random.uniform(0, 2 * math.pi)  # Random starting angle
    # Generate street points
    points = []
    for i in range(int(length * 5)):  # Every 20 cm
        t = i / (length * 5)
        angle = t * 2 * math.pi + angle_offset
        x = t * length + math.sin(angle) * curve_factor
        y = math.cos(angle) * curve_factor
        points.append((x, y))
    return width, points
def create_road_mesh(street_points, street_width):
    mesh = bpy.data.meshes.new("Road")
    road_obj = bpy.data.objects.new("Road", mesh)
    bpy.context.collection.objects.link(road_obj)
    bm = bmesh.new()
    for i in range(len(street_points) - 1):
        p1 = street_points[i]
        p2 = street_points[i + 1]
        # Calculate perpendicular vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0, 1
        # Create vertices
        v1 = bm.verts.new((p1[0] + nx * street_width/2, p1[1] + ny * street_width/2, 0.1))
        v2 = bm.verts.new((p1[0] - nx * street_width/2, p1[1] - ny * street_width/2, 0.1))
        v3 = bm.verts.new((p2[0] - nx * street_width/2, p2[1] - ny * street_width/2, 0.1))
        v4 = bm.verts.new((p2[0] + nx * street_width/2, p2[1] + ny * street_width/2, 0.1))
        # Create face
        bm.faces.new((v1, v2, v3, v4))
    bm.to_mesh(mesh)
    bm.free()
    # Add a material to the road
    mat = bpy.data.materials.new(name="Road_Material")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # Dark gray
    road_obj.data.materials.append(mat)
    return road_obj
def check_collision(new_building, existing_buildings):
    for existing in existing_buildings:
        dx = abs(new_building.center_location[0] - existing.center_location[0])
        dy = abs(new_building.center_location[1] - existing.center_location[1])
        min_distance = max(new_building.size[1], new_building.size[2]) / 2 + \
                       max(existing.size[1], existing.size[2]) / 2 + Building.dist
        if dx < min_distance and dy < min_distance:
            return True
    return False
def place_buildings_on_street(street_points, street_width):
    buildings = []
    street_sizes = ['Small', 'Mean', 'Big']
    max_attempts = 100
    for i, point in enumerate(street_points):
        if i % 10 != 0:  # Try to place a building every 2 meters
            continue
        for side in [-1, 1]:  # Left and right side of the street
            street_size = random.choice(street_sizes)
            new_building = Building(street_size)
            for _ in range(max_attempts):
                # Calculate building position and rotation
                if i < len(street_points) - 1:
                    dx = street_points[i+1][0] - point[0]
                    dy = street_points[i+1][1] - point[1]
                    angle = math.atan2(dy, dx)
                else:
                    angle = 0
                offset = side * (street_width / 2 + Building.sidewalk + new_building.size[2] / 2)
                new_building.center_location = (
                    point[0] - math.sin(angle) * offset,
                    point[1] + math.cos(angle) * offset
                )
                new_building.rotation = angle + (math.pi / 2 if side > 0 else -math.pi / 2)
                if not check_collision(new_building, buildings):
                    buildings.append(new_building)
                    create_building(new_building)
                    break
            if _ == max_attempts - 1:
                print(f"Failed to place building at point {i} on side {side}")
    return buildings
# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
# Set unit scale to meters
bpy.context.scene.unit_settings.scale_length = 1
bpy.context.scene.unit_settings.length_unit = 'METERS'
# Create street
street_length = 300
street_width, street_points = create_street(street_length)
# Create road mesh
road_obj = create_road_mesh(street_points, street_width)
# Place buildings
buildings = place_buildings_on_street(street_points, street_width)
# Add a ground plane
ground_size = max(street_length * 1.2, street_width * 4)
bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, 0))
# Add a sun light
bpy.ops.object.light_add(type='SUN', location=(0, 0, 100))
# Set up the camera
cam_distance = street_length * 0.8
bpy.ops.object.camera_add(location=(cam_distance, -cam_distance/2, cam_distance/2),
                          rotation=(math.radians(60), 0, math.radians(45)))
bpy.context.scene.camera = bpy.context.object
# Set the 3D viewport to look through the camera
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].region_3d.view_perspective = 'CAMERA'
        break
# Select all objects and adjust view
bpy.ops.object.select_all(action='SELECT')
bpy.ops.view3d.view_selected(use_all_regions=False)

