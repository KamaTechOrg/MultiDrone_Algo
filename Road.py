import bpy
import random
import math
import bmesh

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
        self.create_street()
        self.create_road_mesh()
        self.create_dashed_line()


# דוגמת שימוש ב-Blender
street = Street(street_length=1000, street_width=30.0)

# יצירת הכביש
street.generate_street()



