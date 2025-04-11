import json
import numpy as np
from PIL import Image, ImageDraw
import math
import os


def get_camera_matrix(fov, image_width, image_height):
    focal_length = image_width / (2.0 * math.tan(fov * math.pi / 360.0))
    return np.array([
        [focal_length, 0, image_width / 2.0],
        [0, focal_length, image_height / 2.0],
        [0, 0, 1]
    ])


def get_rotation_matrix(pitch, yaw, roll):
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def load_camera_transform(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    loc = data['location']
    rot = data['rotation']
    return loc, rot


def transform_point(point, location, rotation):
    # Subtract camera location to make it relative
    x, y, z = point[0] - location['x'], point[1] - location['y'], point[2] - location['z']
    R = get_rotation_matrix(rotation['pitch'], rotation['yaw'], rotation['roll'])
    return R @ np.array([x, y, z])


def project_points_to_2d(points, location, rotation, camera_matrix):
    pixel_points = []
    for point in points:
        transformed = transform_point(point, location, rotation)
        if transformed[2] <= 0:
            continue
        proj = camera_matrix @ transformed.reshape(3, 1)
        proj = proj / proj[2][0]
        pixel_points.append((int(proj[0][0]), int(proj[1][0])))
    return pixel_points


def get_bbox_corners(center, extent, rotation):
    # Get local bounding box corners
    ex, ey, ez = extent['x'], extent['y'], extent['z']
    corners = [
        [ ex,  ey, -ez],
        [-ex,  ey, -ez],
        [-ex, -ey, -ez],
        [ ex, -ey, -ez],
        [ ex,  ey,  ez],
        [-ex,  ey,  ez],
        [-ex, -ey,  ez],
        [ ex, -ey,  ez],
    ]
    R = get_rotation_matrix(rotation['pitch'], rotation['yaw'], rotation['roll'])
    transformed = []
    for corner in corners:
        rotated = R @ np.array(corner)
        world = rotated + np.array([center['x'], center['y'], center['z']])
        transformed.append(world.tolist())
    return transformed


def annotate_image(frame_number, fov=90, image_width=800, image_height=600):
    img_path = f"output/camera/{frame_number:06d}.png"
    bbox_path = f"output/bboxes/{frame_number:06d}.json"
    cam_tf_path = f"output/camera_transforms/{frame_number:06d}.json"

    if not os.path.exists(img_path) or not os.path.exists(bbox_path) or not os.path.exists(cam_tf_path):
        print(f"Missing data for frame {frame_number}")
        return

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    camera_matrix = get_camera_matrix(fov, image_width, image_height)
    camera_loc, camera_rot = load_camera_transform(cam_tf_path)

    with open(bbox_path, 'r') as f:
        bboxes = json.load(f)

    for bbox in bboxes:
        corners_3d = get_bbox_corners(bbox['location'], bbox['extent'], bbox['rotation'])
        pixel_points = project_points_to_2d(corners_3d, camera_loc, camera_rot, camera_matrix)

        if pixel_points:
            xs, ys = zip(*pixel_points)
            x_min, x_max = max(min(xs), 0), min(max(xs), image_width)
            y_min, y_max = max(min(ys), 0), min(max(ys), image_height)
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

    output_path = f"output/annotated/{frame_number:06d}_annotated.png"
    os.makedirs("output/annotated", exist_ok=True)
    img.save(output_path)
    print(f"Saved: {output_path}")
