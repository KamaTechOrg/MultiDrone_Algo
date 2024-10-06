import bpy
import bmesh
import random
import math
import numpy as np
from mathutils import noise

# מחיקת כל האובייקטים הקיימים
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# הגדרת רקע שקוף
bpy.context.scene.render.film_transparent = True
world = bpy.context.scene.world
world.use_nodes = True
world.node_tree.nodes.clear()

def create_water_sources(shape, target_percentage):
    total_cells = shape[0] * shape[1]
    target_water_cells = int(total_cells * target_percentage / 100)
    
    # יצירת מפת רעש לצורה טבעית יותר
    noise_map = np.zeros(shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            noise_map[y, x] = noise.noise((x * 0.01, y * 0.01, 0))  # הקטנת גודל הצעד לרזולוציה גבוהה יותר
    
    # נרמול מפת הרעש
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    
    # יצירת צורת אגם מעוגלת
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # חישוב צורה עגולה
    round_shape = 1 - (dist_from_center / max_dist)
    
    # שילוב הרעש עם הצורה העגולה
    combined_map = round_shape * noise_map
    
    # קביעת סף לפי אחוז המים הרצוי
    threshold = np.sort(combined_map.flatten())[-target_water_cells]
    water_map = combined_map > threshold
    
    return water_map

def create_realistic_puddle(location, size, water_map):
    bm = bmesh.new()
    
    # יצירת שלולית על פי מפת המים
    verts = {}
    for y in range(water_map.shape[0]):
        for x in range(water_map.shape[1]):
            if water_map[y, x]:
                vert = bm.verts.new((x * size, y * size, 0))
                verts[(x, y)] = vert
    
    bm.verts.ensure_lookup_table()
    
    # חיבור הנקודות ליצירת הפאות
    for y in range(water_map.shape[0] - 1):
        for x in range(water_map.shape[1] - 1):
            if (x, y) in verts and (x + 1, y) in verts and (x, y + 1) in verts and (x + 1, y + 1) in verts:
                v1 = verts[(x, y)]
                v2 = verts[(x + 1, y)]
                v3 = verts[(x, y + 1)]
                v4 = verts[(x + 1, y + 1)]
                bm.faces.new((v1, v2, v4, v3))
    
    mesh = bpy.data.meshes.new(f"Puddle_{random.randint(1000, 9999)}")
    bm.to_mesh(mesh)
    bm.free()
    
    puddle = bpy.data.objects.new(f"Puddle_{random.randint(1000, 9999)}", mesh)
    bpy.context.collection.objects.link(puddle)
    
    # יצירת חומר עבור השלולית
    puddle_mat = bpy.data.materials.new(name=f"PuddleMaterial_{puddle.name}")
    puddle_mat.use_nodes = True
    node_tree = puddle_mat.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    
    # הגדרת חומר כחול חצי שקוף
    node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled.inputs['Base Color'].default_value = (0.1, 0.3, 0.8, 1)  # כחול
    node_principled.inputs['Roughness'].default_value = 0.1
    node_principled.inputs['Alpha'].default_value = 0.7  # שקיפות חלקית
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = node_tree.links
    links.new(node_principled.outputs['BSDF'], node_output.inputs['Surface'])
    
    puddle_mat.blend_method = 'BLEND'
    puddle.data.materials.append(puddle_mat)
    
    puddle.location = location
    return puddle

# פונקציה לבדיקת חפיפה בין אגמים
def is_overlapping(location, size, existing_puddles):
    for puddle in existing_puddles:
        distance = math.sqrt((location[0] - puddle[0])**2 + (location[1] - puddle[1])**2)
        if distance < (size + puddle[2]):
            return True
    return False

# רשימת שלוליות קיימות
existing_puddles = []

# יצירת אגמים גדולים
for _ in range(2):
    while True:
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        size = random.uniform(10, 15)  # גודל האגמים בין 10 ל-15
        if not is_overlapping((x, y, 0), size, existing_puddles):
            break
    water_map = create_water_sources((100, 100), 30)  # העלאת הרזולוציה
    create_realistic_puddle((x, y, 0), size / 100, water_map)
    existing_puddles.append((x, y, size))

# יצירת שלוליות קטנות
for _ in range(4):
    while True:
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        size = random.uniform(2, 5)  # גודל השלוליות בין 2 ל-5
        if not is_overlapping((x, y, 0), size, existing_puddles):
            break
    water_map = create_water_sources((100, 100), 30)  # העלאת הרזולוציה
    create_realistic_puddle((x, y, 0), size / 100, water_map)
    existing_puddles.append((x, y, size))

class Street:
    def __init__(self, street_length=1000, segment_length=5, street_width=2.0):
        self.street_length = street_length
        self.segment_length = segment_length
        self.street_width = street_width
        self.points = []  # לשמור את נקודות ההתחלה של המקטעים

    def create_street(self):
        """יוצר מקטעי כביש בצורה הרצויה עם עיקול ושומר את הנקודות."""
        curve_factor = random.uniform(0, self.street_length / 4)  # גורם עיקול רנדומלי
        angle_offset = random.uniform(0, 2 * math.pi)  # זווית התחלתית רנדומלית

        # יצירת נקודת התחלה קרובה לראשית הצירים
        start_x = random.uniform(-10, 10)
        start_y = random.uniform(-10, 10)

        # יצירת נקודות הכביש
        self.points = []
        for i in range(int(self.street_length * 5)):  # כל 20 ס"מ
            t = i / (self.street_length * 5)
            angle = t * 2 * math.pi + angle_offset
            x = start_x + t * self.street_length + math.sin(angle) * curve_factor
            y = start_y + math.cos(angle) * curve_factor
            self.points.append((x, y))

    def create_road_mesh(self):
        """יוצר Mesh של הכביש עם הנקודות שנוצרו."""
        mesh = bpy.data.meshes.new("Road")
        road_obj = bpy.data.objects.new("Road", mesh)
        bpy.context.collection.objects.link(road_obj)
        bm = bmesh.new()
        
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            # חישוב וקטור ניצב
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
            else:
                nx, ny = 0, 1
            
            # יצירת קודקודים
            v1 = bm.verts.new((p1[0] + nx * self.street_width / 2, p1[1] + ny * self.street_width / 2, 0))
            v2 = bm.verts.new((p1[0] - nx * self.street_width / 2, p1[1] - ny * self.street_width / 2, 0))
            v3 = bm.verts.new((p2[0] + nx * self.street_width / 2, p2[1] + ny * self.street_width / 2, 0))
            v4 = bm.verts.new((p2[0] - nx * self.street_width / 2, p2[1] - ny * self.street_width / 2, 0))
            
            bm.faces.new((v1, v2, v4, v3))

        bm.to_mesh(mesh)
        bm.free()

# יצירת הכביש
street = Street(street_length=100, segment_length=5, street_width=2.0)
street.create_street()
street.create_road_mesh()



