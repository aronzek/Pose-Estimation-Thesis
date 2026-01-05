# File: generate_coco_annotations.py
# Requires: pip install numpy

import os
import json
import math

IMAGE_DIR = "SyntheticExports/images"
CONTAINER_JSON = "SyntheticExports/unity_containers.json"
ANNOTATION_OUT = "SyntheticExports/coco_annotations.json"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
GROUND_WIDTH = 300.0  # meters (X scale in Unity * 10)
GROUND_HEIGHT = 100.0 # meters (Z scale in Unity * 10)


def world_to_pixel(x, z):
    px = int((x / GROUND_WIDTH) * IMAGE_WIDTH)
    py = int(IMAGE_HEIGHT - (z / GROUND_HEIGHT) * IMAGE_HEIGHT)
    return px, py

def create_coco():
    with open(CONTAINER_JSON) as f:
        containers = json.load(f)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "container"}]
    }

    img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

    annotation_id = 1
    image_id = 1
    for img_name in img_files:
        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT
        })

        for c in containers:
            x, y, z = c["position"]
            w, h = 12.0, 2.4  # container length × width in meters
            rot = math.radians(c["rotationY"])

            # Approx. bbox center
            cx, cz = x, z
            px, py = world_to_pixel(cx, cz)

            # Approx. bbox dimensions in pixels
            pw = int((w / GROUND_WIDTH) * IMAGE_WIDTH)
            ph = int((h / GROUND_HEIGHT) * IMAGE_HEIGHT)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [px - pw // 2, py - ph // 2, pw, ph],
                "area": pw * ph,
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    with open(ANNOTATION_OUT, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"✅ COCO annotations saved to {ANNOTATION_OUT}")

if __name__ == '__main__':
    create_coco()
