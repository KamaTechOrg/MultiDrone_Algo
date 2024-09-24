import bpy
import random
import math
import bmesh

class Street:
    def __init__(self, street_length=1000, segment_length=5, street_width=2.0, building_gap=20.0):
        self.street_length = street_length
        self.segment_length = segment_length
        self.street_width = street_width
        self.building_gap = building_gap
        self.points = []  # Stores road segment points
        self.building_positions = []  # Stores building positions

    def create_street(self):
        """Creates road segments with curvature and stores points."""
        curve_factor = random.uniform(0, self.street_length / 4)  # Random curve factor
        angle_offset = random.uniform(0, 2 * math.pi)  # Random starting angle
        start_x = random.uniform(-10, 10)
        start_y = random.uniform(-10, 10)
        self.points = []
        for i in range(int(self.street_length * 5)):  # Every 20 cm
            t = i / (self.street_length * 5)
            angle = t * 2 * math.pi + angle_offset
            x = start_x + t * self.street_length + math.sin(angle) * curve_factor
            y = start_y + math.cos(angle) * curve_factor
            self.points.append((x, y))

    def create_road_mesh(self):
        """Creates a road mesh based on the generated points."""
        mesh = bpy.data.meshes.new("Road")
        road_obj = bpy.data.objects.new("Road", mesh)
        bpy.context.collection.objects.link(road_obj)
        bm = bmesh.new()
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
            else:
                nx, ny = 0, 1
            v1 = bm.verts.new((p1[0] + nx * self.street_width / 2, p1[1] + ny * self.street_width / 2, 0))
            v2 = bm.verts.new((p1[0] - nx * self.street_width / 2, p1[1] - ny * self.street_width / 2, 0))
            v3 = bm.verts.new((p2[0] - nx * self.street_width / 2, p2[1] - ny * self.street_width / 2, 0))
            v4 = bm.verts.new((p2[0] + nx * self.street_width / 2, p2[1] + ny * self.street_width / 2, 0))
            bm.faces.new((v1, v2, v3, v4))
        bm.to_mesh(mesh)
        bm.free()

        mat_road = bpy.data.materials.new(name="Road_Material")
        mat_road.use_nodes = True
        mat_road.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.05, 0.05, 0.05, 1)  # Dark gray
        road_obj.data.materials.append(mat_road)
        return road_obj

    
    def generate_street(self):
        """Generates a random street and adds buildings along it."""
        self.create_street()
        self.create_road_mesh()
        
        
        

class Building:
    dist = 5  # Minimum distance between buildings
    sidewalk = 5  # Width of the sidewalk
    data = {
        'Big': {'Height': [50, 10], 'Width': [30, 5], 'Depth': [25, 4]},
        'Mean': {'Height': [30, 8], 'Width': [20, 4], 'Depth': [15, 3]},
        'Small': {'Height': [15, 5], 'Width': [10, 2], 'Depth': [8, 2]}
    }

    def __init__(self, street_width):
        self.street_size = self.determine_street_size(street_width)
        height = max(5, random.gauss(self.data[self.street_size]['Height'][0], self.data[self.street_size]['Height'][1]))
        width = max(5, random.gauss(self.data[self.street_size]['Width'][0], self.data[self.street_size]['Width'][1]))
        depth = max(5, random.gauss(self.data[self.street_size]['Depth'][0], self.data[self.street_size]['Depth'][1]))
        self.size = [height, width, depth]
        self.center_location = (0, 0)  # Will be set later
        self.rotation = 0  # Will be set later

    def determine_street_size(self, street_width):
        if street_width > 19:
            return 'Big'
        elif 11 <= street_width <= 19:
            return 'Mean'
        else:
            return 'Small'

def create_building(building):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    # Set scale
    obj.scale = (building.size[1], building.size[2], building.size[0])
    # Set location and rotation
    obj.location = (building.center_location[0], building.center_location[1], building.size[0] / 2)
    obj.rotation_euler[2] = building.rotation



def check_collision(new_building, existing_buildings, city_streets):
    # Check collision with existing buildings
    for existing in existing_buildings:
        dx = abs(new_building.center_location[0] - existing.center_location[0])
        dy = abs(new_building.center_location[1] - existing.center_location[1])
        min_distance = max(new_building.size[1], new_building.size[2]) / 2 + \
                       max(existing.size[1], existing.size[2]) / 2 + Building.dist
        if dx < min_distance and dy < min_distance:
            return True
    
    # Check collision with streets
    for street in city_streets:
        for i in range(len(street.points) - 1):
            p1 = street.points[i]
            p2 = street.points[i + 1]
            
            # Calculate the direction vector of the street segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length == 0:
                continue
            
            # Normalize the direction vector
            dx /= length
            dy /= length
            
            # Calculate the perpendicular vector
            px = -dy
            py = dx
            
            # Calculate the four corners of the street segment
            half_width = street.street_width / 2
            street_corners = [
                (p1[0] + px * half_width, p1[1] + py * half_width),
                (p1[0] - px * half_width, p1[1] - py * half_width),
                (p2[0] - px * half_width, p2[1] - py * half_width),
                (p2[0] + px * half_width, p2[1] + py * half_width)
            ]
            
            # Check if the building intersects with this street segment
            if polygon_rectangle_intersection(street_corners, new_building):
                return True
    
    return False

def polygon_rectangle_intersection(polygon, building):
    # Check if any corner of the building is inside the polygon
    building_corners = get_building_corners(building)
    for corner in building_corners:
        if point_in_polygon(corner, polygon):
            return True
    
    # Check if any edge of the building intersects with any edge of the polygon
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        for j in range(len(building_corners)):
            b1 = building_corners[j]
            b2 = building_corners[(j+1) % len(building_corners)]
            if line_intersection(p1, p2, b1, b2):
                return True
    
    return False

def get_building_corners(building):
    half_width = building.size[1] / 2
    half_depth = building.size[2] / 2
    return [
        (building.center_location[0] - half_width, building.center_location[1] - half_depth),
        (building.center_location[0] + half_width, building.center_location[1] - half_depth),
        (building.center_location[0] + half_width, building.center_location[1] + half_depth),
        (building.center_location[0] - half_width, building.center_location[1] + half_depth)
    ]

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return False
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    return 0 <= t <= 1 and 0 <= u <= 1
def place_buildings_on_street(street_points, street_width, city_streets):
    buildings = []
    street_size = street_width
    max_attempts = 100
    for i, point in enumerate(street_points):
        if i % 10 != 0:  # Try to place a building every 2 meters
            continue
        for side in [-1, 1]:  # Left and right side of the street
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
                if not check_collision(new_building, buildings, city_streets):
                    buildings.append(new_building)
                    create_building(new_building)
                    break
            if _ == max_attempts - 1:
                print(f"Failed to place building at point {i} on side {side}")
    return buildings



# Usage in Blender
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.context.scene.unit_settings.scale_length = 1
bpy.context.scene.unit_settings.length_unit = 'METERS'

street = Street(street_length=1000, street_width=30.0)
street.generate_street()

street1 = Street(street_length=1000, street_width=30.0)
street1.generate_street()

city_streets = [street, street1]

place_buildings_on_street(street.points, street.street_width, city_streets)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.view3d.view_selected(use_all_regions=False)
