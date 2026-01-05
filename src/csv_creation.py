import json
import csv
import re

#  Set  file paths here
detection_json_path = "mission_1/data/detections_south_terminal.json"
rotation_txt_path = "mission_1/data/south_terminal/rotations.txt"
output_csv_path = "mission_1/data/results/pose_south_terminal.csv"

def parse_rotations_from_file(rotation_txt_path):
    """Parses the rotation txt file and returns a dict: {id: (yaw, fitness)}"""
    rotation_dict = {}
    current_id = None
    current_yaw = None

    with open(rotation_txt_path, "r") as f:
        for line in f:
            container_match = re.search(r"Container (\d+)", line)
            yaw_match = re.search(r"Yaw=([-+]?[0-9]*\.?[0-9]+)", line)
            fitness_match = re.search(r"ICP fitness=([-+]?[0-9]*\.?[0-9]+)", line)

            if container_match:
                current_id = int(container_match.group(1))

            if yaw_match:
                current_yaw = float(yaw_match.group(1))

            if fitness_match and current_id is not None and current_yaw is not None:
                fitness = float(fitness_match.group(1))
                rotation_dict[current_id] = (current_yaw, fitness)
                current_id = None
                current_yaw = None

    return rotation_dict

def create_pose_csv(detection_json_path, rotation_txt_path, output_csv_path):
    with open(detection_json_path, "r") as f:
        detections = json.load(f)

    rotation_dict = parse_rotations_from_file(rotation_txt_path)
    csv_rows = []

    for detection in detections:
        det_id = detection.get("id")
        geo_bbox = detection.get("geo_bbox")

        if not geo_bbox or det_id is None:
            continue

        east = (geo_bbox[0][0] + geo_bbox[1][0]) / 2
        north = (geo_bbox[0][1] + geo_bbox[1][1]) / 2

        rot_entry = rotation_dict.get(det_id)

        if rot_entry is not None:
            yaw, fitness = rot_entry
            row = {
                "ID": det_id,
                "Easting": round(east, 3),
                "Northing": round(north, 3),
                "Yaw": round(yaw, 2),
                "Fitness": round(fitness, 4)
            }
            csv_rows.append(row)

    with open(output_csv_path, "w", newline="") as csvfile:
        fieldnames = ["ID", "Easting", "Northing", "Yaw", "Fitness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✅ CSV created: {output_csv_path}")

# 🚀 Run the function
create_pose_csv(detection_json_path, rotation_txt_path, output_csv_path)