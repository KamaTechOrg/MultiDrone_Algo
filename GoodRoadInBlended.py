#import matplotlib.pyplot as plt
import random
import math
import bpy
import bmesh



import numpy as np
class City:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mat = np.zeros((height, width))
        self.objects=[]
        self.obj_colors={
            'empty' : 0,
            'road': 1,
            'car': 2,
            'building': 3,
            'lake': 4,
            'person': 5,
            'tree': 6
        }
        
    def get_mat(self):
        return self.mat
    
    def get_pix(self, x, y):
        for k, v in self.obj_colors.items():
            if v == self.mat[x][y]:
                return k
        return "error"
    
    def set_pix(self, x, y, obj):
        x, y = round(x), round(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            self.mat[x][y] = self.obj_colors[obj]
        x, y = x + 1, y + 1
        if 0 <= x < self.width and 0 <= y < self.height:
            self.mat[x][y] = self.obj_colors[obj]
            
    def get_obj_pix(self, obj):
        for i in range(my_city.height):
            for j in range(my_city.width):
                if my_city.get_pix(i,j) == obj:
                    print(i ,", ",j)





class Street:
    def __init__(self,city, street_length=1000, segment_length=5, street_width=30.0):
        self.street_length = street_length
        self.segment_length = segment_length
        self.street_width = street_width
        self.points = []
        
    def create_streets(self, city, num_streets):
        for _ in range(num_streets):
            self.create_street(city)

    def create_street(self, city):
        curve_factor = 20
        angle_offset = random.uniform(0, 2 * math.pi)
        start_x = random.uniform(0, city.width)
        start_y = random.uniform(0, city.height)

        for i in range(int(self.street_length * 5)):
            t = i / (self.street_length * 5)
            angle = t * 2 * math.pi + angle_offset
            x = start_x + t * self.street_length + math.sin(angle) * curve_factor
            y = start_y + math.cos(angle) * curve_factor
#             if not (0 <= x < city.width and 0 <= y < city.height):
#                 return self.create_street(city)
            self.points.append((x, y))
            city.set_pix(int(x), int(y), 'road')
            self.store_road_pixels(city, x, y)

        city.objects.append(self) 
        
    def store_road_pixels(self,city, x, y):
        half_width = self.street_width / 2
        for offset in range(int(-half_width), int(half_width) + 1):
            pixel_x = int(x + offset)
            pixel_y = int(y)
            city.set_pix(pixel_x, pixel_y,'road')



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
            v3 = bm.verts.new((p2[0] - nx * self.street_width / 2, p2[1] - ny * self.street_width / 2, 0))
            v4 = bm.verts.new((p2[0] + nx * self.street_width / 2, p2[1] + ny * self.street_width / 2, 0))
            # יצירת פאה
            bm.faces.new((v1, v2, v3, v4))
        bm.to_mesh(mesh)
        bm.free()

        # הוספת חומר לכביש
        mat_road = bpy.data.materials.new(name="Road_Material")
        mat_road.use_nodes = True
        mat_road.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.05, 0.05, 0.05, 1)  # שחור כהה
        road_obj.data.materials.append(mat_road)
        return road_obj

    def create_dashed_line(self):
        """יוצר קו מקווקו לאורך הכביש."""
        dashed_length = 10
        gap_length = 10

        # יצירת Mesh לקו המקווקו
        mesh = bpy.data.meshes.new("Dashed_Line")
        line_obj = bpy.data.objects.new("Dashed_Line", mesh)
        bpy.context.collection.objects.link(line_obj)
        bm = bmesh.new()

        # צייר קווים מקווקוים
        for i in range(0, len(self.points) - 1, dashed_length + gap_length):
            if i + dashed_length >= len(self.points):
                break
            for j in range(dashed_length):
                if i + j + 1 >= len(self.points):
                    break
                p1 = self.points[i + j]
                p2 = self.points[i + j + 1]
                # יצירת קודקודים
                v1 = bm.verts.new((p1[0], p1[1], 0.01))
                v2 = bm.verts.new((p2[0], p2[1], 0.01))
                bm.edges.new((v1, v2))

        bm.to_mesh(mesh)
        bm.free()

        # הוספת חומר לקו המקווקו
        mat_line = bpy.data.materials.new(name="Dashed_Line_Material")
        mat_line.use_nodes = True
        mat_line.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 1, 1, 1)  # לבן
        line_obj.data.materials.append(mat_line)

    def generate_street(self):
        """מייצרת כביש רנדומלי ב-Blender."""
        self.create_road_mesh()
        self.create_dashed_line()
        
my_city = City(1000, 1000)
street = Street(my_city)
street.create_streets(my_city,2)
#ציור הכביש - להוציא מהמחלקה של הכביש
street.generate_street()