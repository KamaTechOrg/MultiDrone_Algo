
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
# from Functions.object_tracking import track_frames
from collections import defaultdict
def find_current_index(current_bbox, positions):
    for index, bbox in enumerate(positions):
        if bbox['bbox'] == current_bbox:
            return index
    return -1  # Return -1 if the current bounding box is not found


def calculate_slope(previous_position, current_position):
    # Computes the slope between two positions
    x1, y1 = previous_position
    x2, y2 = current_position
    
    # Handle division by zero: return infinite slope if x coordinates are the same
    if x2 - x1 == 0:
        return float('inf')  # Infinite slope
    return (y2 - y1) / (x2 - x1)  # Calculate slope

def is_collision_from_different_directions(current_bbox1, current_bbox2,obj1, obj2 ):
    # Checks if two objects are moving from different directions based on their bounding boxes
    positions1 = obj1['Bounding Boxes']
    positions2 = obj2['Bounding Boxes']
    
    # Find the current indexes for the bounding boxes
    current_index1 = find_current_index(current_bbox1, positions1)
    current_index2 = find_current_index(current_bbox2, positions2)

    # Ensure there is a previous bounding box for both objects
    if current_index1 <= 0 or current_index2 <= 0:
        return False, None, None  # Not enough positions

    # Get the current and previous positions for obj1
    previous_bbox1 = positions1[current_index1 - 1]['bbox']
    current_pos1 = (current_bbox1[0] + current_bbox1[2] / 2, current_bbox1[1] + current_bbox1[3] / 2)
    previous_pos1 = (previous_bbox1[0] + previous_bbox1[2] / 2,
                     previous_bbox1[1] + previous_bbox1[3] / 2)

    # Get the current and previous positions for obj2
    previous_bbox2 = positions2[current_index2 - 1]['bbox']
    current_pos2 = (current_bbox2[0] + current_bbox2[2] / 2, current_bbox2[1] + current_bbox2[3] / 2)
    previous_pos2 = (previous_bbox2[0] + previous_bbox2[2] / 2,
                     previous_bbox2[1] + previous_bbox2[3] / 2)

    # Calculate slopes for both objects
    slope1 = calculate_slope(previous_pos1, current_pos1)
    slope2 = calculate_slope(previous_pos2, current_pos2)

    # Calculate angles in degrees
    angle1 = math.degrees(math.atan(slope1)) if slope1 != float('inf') else 90
    angle2 = math.degrees(math.atan(slope2)) if slope2 != float('inf') else 90

    # Check if angles differ by at least 45 degrees
    collision_different_directions = abs(angle1 - angle2) >= 45

    return collision_different_directions, angle1, angle2  # Return collision status and angles

def is_bbox_collision(bbox1, bbox2):
    # Function to check if two bounding boxes collide
    x1_min, y1_min, w1, h1 = bbox1
    x1_max = x1_min + w1
    y1_max = y1_min + h1

    x2_min, y2_min, w2, h2 = bbox2
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    # Return True if there is an overlap between bounding boxes
    return not (x1_max < x2_min or x1_min > x2_max or 
                y1_max < y2_min or y1_min > y2_max)

def group_objects_by_frame(tracked_objects):
    # Group objects by frame ID for easier processing
    frame_objects = defaultdict(list)

    for obj in tracked_objects:
        for bbox in obj["Bounding Boxes"]:
            frame_objects[bbox["frame_id"]].append((obj, bbox))

    return frame_objects

def check_slope_change(current_bbox1, current_bbox2,obj1, obj2, angle1, angle2):
    # Check if there is a significant change in slope between the current and next bounding boxes
    positions1 = obj1['Bounding Boxes']
    positions2 = obj2['Bounding Boxes']
    
    # Find the current indexes for the bounding boxes
    current_index1 = find_current_index(current_bbox1, positions1)
    current_index2 = find_current_index(current_bbox2, positions2)

    # Ensure there is a next bounding box for both objects
    if current_index1 < 0 or current_index1 >= len(positions1) - 1 or current_index2 < 0 or current_index2 >= len(positions2) - 1:
        return False  # Not enough positions

    # Get the next positions for obj1
    next_bbox1 = positions1[current_index1 + 1]['bbox']
    current_pos1 = (current_bbox1[0] + current_bbox1[2] / 2, current_bbox1[1] + current_bbox1[3] / 2)
    next_pos1 = (next_bbox1[0] + next_bbox1[2] / 2, next_bbox1[1] + next_bbox1[3] / 2)

    # Get the next positions for obj2
    next_bbox2 = positions2[current_index2 + 1]['bbox']
    current_pos2 = (current_bbox2[0] + current_bbox2[2] / 2, current_bbox2[1] + current_bbox2[3] / 2)
    next_pos2 = (next_bbox2[0] + next_bbox2[2] / 2, next_bbox2[1] + next_bbox2[3] / 2)

    # Calculate slopes for the current and next positions
    slope_current1 = calculate_slope(current_pos1, next_pos1)
    slope_current2 = calculate_slope(current_pos2, next_pos2)

    # Calculate angles in degrees
    new_angle1 = math.degrees(math.atan(slope_current1)) if slope_current1 != float('inf') else 90
    new_angle2 = math.degrees(math.atan(slope_current2)) if slope_current2 != float('inf') else 90

    # Check if the change in angle is significant (more than 5 degrees)
    return abs(new_angle1 - angle1) > 5 or abs(new_angle2 - angle2) > 5  # Adjust threshold as needed

def detect_accidents(tracked_objects):
    # Function to detect accidents based on tracked objects in each frame
    frame_objects = group_objects_by_frame(tracked_objects)

    for frame_id, objects in frame_objects.items():
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, bbox1 = objects[i]
                obj2, bbox2 = objects[j]
                
                # Check if both objects are cars
                if obj1["Label"] == "car" and obj2["Label"] == "car":
                        # Check if bounding boxes collide

                        if is_bbox_collision(bbox1["bbox"], bbox2["bbox"]):
                            # Check if the cars are coming from different directions
                            different_directions, angle1, angle2 = is_collision_from_different_directions(bbox1["bbox"], bbox2["bbox"], obj1, obj2)
                            if different_directions:
                               # Check for significant slope change
                               if check_slope_change(bbox1["bbox"], bbox2["bbox"], obj1, obj2, angle1, angle2):
                                        return True, frame_id  # Return True and the frame ID of the collision

    return False, None  # Return False if no accidents were detected

def check_for_accidents(tracked_objects):
    # Function to check for accidents based on tracked objects
    accident_detected, collision_time = detect_accidents(tracked_objects)
    return accident_detected, collision_time  # Return the accident status and collision time

# if __name__ == "__main__":
#     # Example data for testing accident detection

#     tracked_objects = [
#         {
#             "Object": 1,
#             "Label": "car",
#             "Bounding Boxes": [
#                 {'bbox': (100, 150, 50, 70), 'datetime': '2024-10-06 12:00:00', 'frame_id': 1},
#                 {'bbox': (105, 160, 50, 70), 'datetime': '2024-10-06 12:00:01', 'frame_id': 2},
#                 {'bbox': (110, 170, 50, 70), 'datetime': '2024-10-06 12:00:02', 'frame_id': 3}
#             ]
#         },
#         {
#             "Object": 2,
#             "Label": "pedestrian",
#             "Bounding Boxes": [
#                 {'bbox': (150, 160, 30, 60), 'datetime': '2024-10-06 12:00:00', 'frame_id': 1},
#                 {'bbox': (155, 165, 30, 60), 'datetime': '2024-10-06 12:00:01', 'frame_id': 2},
#                 {'bbox': (160, 170, 30, 60), 'datetime': '2024-10-06 12:00:02', 'frame_id': 3}
#             ]
#         }
#     ]





#     accident, collision_time = check_for_accidents(tracked_objects)
    
#     if accident:
#         print("Accident detected!")  # Notify if an accident was detected
#         print("First collision time:", collision_time)
#     else:
#         print("No accidents detected.")  # Notify if no accidents were detected
