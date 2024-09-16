import bpy
import random
import math

class Building:
    dist = 5  # Minimum distance between buildings
    road_width = 20  # Width of the road
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

def create_building(building):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    
    # Set scale
    obj.scale = (building.size[1], building.size[2], building.size[0])
    
    # Set location
    obj.location = (building.center_location[0], building.center_location[1], building.size[0] / 2)

def create_street(length, width):
    bpy.ops.mesh.primitive_plane_add(size=1)
    street = bpy.context.active_object
    street.scale = (length, width, 1)
    street.location = (0, 0, 0.1)  # Slightly above ground to avoid z-fighting
    
    # Add a material to the street
    mat = bpy.data.materials.new(name="Street_Material")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # Dark gray
    street.data.materials.append(mat)

def check_collision(new_building, existing_buildings):
    for existing in existing_buildings:
        dx = abs(new_building.center_location[0] - existing.center_location[0])
        dy = abs(new_building.center_location[1] - existing.center_location[1])
        min_distance = max(new_building.size[1], new_building.size[2]) / 2 + \
                       max(existing.size[1], existing.size[2]) / 2 + Building.dist
        if dx < min_distance and dy < min_distance:
            return True
    return False

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set unit scale to meters
bpy.context.scene.unit_settings.scale_length = 1
bpy.context.scene.unit_settings.length_unit = 'METERS'

# Create street
street_length = 300
street_width = Building.road_width
create_street(street_length, street_width)

# Create buildings
street_sizes = ['Small', 'Mean', 'Big']
buildings = []
building_count = 40  # Number of buildings to attempt to create
max_attempts = 100  # Maximum number of attempts to place a building

for i in range(building_count):
    street_size = random.choice(street_sizes)
    new_building = Building(street_size)
    
    placed = False
    for _ in range(max_attempts):
        # Determine which side of the street to place the building
        side = 1 if random.random() > 0.5 else -1
        
        # Calculate building position
        x_pos = random.uniform(-street_length / 2, street_length / 2)
        y_pos = side * (street_width / 2 + Building.sidewalk + new_building.size[2] / 2)
        
        new_building.center_location = (x_pos, y_pos)
        
        if not check_collision(new_building, buildings):
            buildings.append(new_building)
            create_building(new_building)
            placed = True
            break
    
    if not placed:
        print(f"Failed to place building {i+1} after {max_attempts} attempts")

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