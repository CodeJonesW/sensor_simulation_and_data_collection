import carla
import os
import weakref
import csv
from datetime import datetime
import json
import numpy as np
from PIL import Image, ImageDraw


OUTPUT_DIR = "output"
os.makedirs(f"{OUTPUT_DIR}/camera", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/lidar", exist_ok=True)

imu_file = open(f"{OUTPUT_DIR}/imu.csv", "w", newline='')
gps_file = open(f"{OUTPUT_DIR}/gps.csv", "w", newline='')
imu_writer = csv.writer(imu_file)
gps_writer = csv.writer(gps_file)
imu_writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
gps_writer.writerow(["timestamp", "latitude", "longitude", "altitude"])



def get_camera_matrix(fov, image_w, image_h):
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([
        [focal, 0, image_w / 2.0],
        [0, focal, image_h / 2.0],
        [0, 0, 1]
    ])

def get_3d_bbox_corners(bbox):
    ext = bbox.extent
    corners = np.array([
        [ ext.x,  ext.y, -ext.z],
        [-ext.x,  ext.y, -ext.z],
        [-ext.x, -ext.y, -ext.z],
        [ ext.x, -ext.y, -ext.z],
        [ ext.x,  ext.y,  ext.z],
        [-ext.x,  ext.y,  ext.z],
        [-ext.x, -ext.y,  ext.z],
        [ ext.x, -ext.y,  ext.z],
    ])
    return corners.T  # shape: (3, 8)

def world_to_camera(corner, camera_transform, vehicle_transform):
    # Transform bbox corner to world space
    corner_world = vehicle_transform.transform(carla.Location(x=corner[0], y=corner[1], z=corner[2]))
    
    # Camera world transform
    cam_loc = camera_transform.location
    cam_rot = camera_transform.rotation

    # Convert to numpy
    T = np.array([cam_loc.x, cam_loc.y, cam_loc.z])
    R = np.array([
        [np.cos(np.radians(cam_rot.yaw)), -np.sin(np.radians(cam_rot.yaw)), 0],
        [np.sin(np.radians(cam_rot.yaw)),  np.cos(np.radians(cam_rot.yaw)), 0],
        [0, 0, 1]
    ])
    
    relative_pos = np.array([corner_world.x, corner_world.y, corner_world.z]) - T
    camera_coord = R.dot(relative_pos)
    return camera_coord

def project_bbox_to_image(bbox, vehicle_transform, camera_transform, camera_matrix):
    corners = get_3d_bbox_corners(bbox)
    img_points = []

    for i in range(8):
        world_coord = carla.Location(x=corners[0][i], y=corners[1][i], z=corners[2][i])
        rel = vehicle_transform.transform(world_coord)
        camera_coord = camera_transform.inverse().transform(rel)

        if camera_coord.z <= 0:
            continue

        point = np.array([[camera_coord.x], [camera_coord.y], [camera_coord.z]])
        projected = camera_matrix @ point
        projected /= projected[2][0]

        img_points.append((int(projected[0][0]), int(projected[1][0])))

    return img_points


def draw_bbox_on_image(image_path, points):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    if points:
        # Get bounding rectangle
        xs, ys = zip(*points)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    img.show()



def save_camera(image):
    image.save_to_disk(f"{OUTPUT_DIR}/camera/{image.frame:06d}.png")


def save_lidar(point_cloud):
    point_cloud.save_to_disk(f"{OUTPUT_DIR}/lidar/{point_cloud.frame:06d}.ply")


def save_imu(imu_data):
    imu_writer.writerow([
        imu_data.timestamp,
        imu_data.accelerometer.x,
        imu_data.accelerometer.y,
        imu_data.accelerometer.z,
        imu_data.gyroscope.x,
        imu_data.gyroscope.y,
        imu_data.gyroscope.z
    ])


def save_gps(gps_data):
    gps_writer.writerow([
        gps_data.timestamp,
        gps_data.latitude,
        gps_data.longitude,
        gps_data.altitude
    ])


def save_bounding_boxes(world, frame_number):
    vehicles = list(world.get_actors().filter('vehicle.*'))
    walkers = list(world.get_actors().filter('walker.pedestrian.*'))
    actors = vehicles + walkers

    boxes = []
    for actor in actors:
        transform = actor.get_transform()
        bb = actor.bounding_box

        box_data = {
            "id": actor.id,
            "type": actor.type_id,
            "location": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z
            },
            "rotation": {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll
            },
            "extent": {
                "x": bb.extent.x,
                "y": bb.extent.y,
                "z": bb.extent.z
            }
        }
        boxes.append(box_data)

    # Save to file
    output_path = f"output/bboxes/{frame_number:06d}.json"
    with open(output_path, "w") as f:
        json.dump(boxes, f, indent=2)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    vehicle.set_autopilot(True)

    sensor_actors = []

    # --- CAMERA ---
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(save_camera)
    sensor_actors.append(camera)

    # --- LiDAR ---
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(save_lidar)
    sensor_actors.append(lidar)

    # --- IMU ---
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    imu.listen(save_imu)
    sensor_actors.append(imu)

    # --- GPS ---
    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, carla.Transform(), attach_to=vehicle)
    gps.listen(save_gps)
    sensor_actors.append(gps)

    print("Recording sensor data... Press Ctrl+C to stop.")
    try:
        while True:
            tick = world.wait_for_tick()
            frame = tick.frame
            save_bounding_boxes(world, frame)
            camera_tf = camera.get_transform()
            with open(f"output/camera_transforms/{frame:06d}.json", "w") as f:
                json.dump({
                    "location": {
                    "x": camera_tf.location.x,
                    "y": camera_tf.location.y,
                    "z": camera_tf.location.z
                    },
                    "rotation": {
                    "pitch": camera_tf.rotation.pitch,
                    "yaw": camera_tf.rotation.yaw,
                    "roll": camera_tf.rotation.roll
                    }
                }, f, indent=2)



    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for actor in sensor_actors:
            actor.stop()
            actor.destroy()
        vehicle.destroy()
        imu_file.close()
        gps_file.close()
        print("All actors destroyed and files saved.")


if __name__ == '__main__':
    main()
