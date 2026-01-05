import subprocess
import time
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil 

# Define paths
YOLO_WEIGHTS = "final_code/data/model/igd_od_trudi-be-tu.pt"
ORTHOPHOTO_PATH = "final_code/data/unity_terminal/odm_orthophoto.tif"
DETECTIONS_JSON_PATH = "mission_1/data/detections_unity_test.json"
YOLO_RESULTS_DIR = "yolov5/runs/detect/exp"

def clean_yolo_results():
    if os.path.exists(YOLO_RESULTS_DIR):
        shutil.rmtree(YOLO_RESULTS_DIR)
        print("🧹 Cleaned previous YOLO results.")

# Run YOLO and save detections in JSON
def run_yolo_and_save_json(source, json_output, start=0.01, end=0.01, step=0.01):
    """Runs YOLO multiple times, saves detections in JSON, and visualizes results."""
    best_conf = None
    best_detections = []

    for conf in np.arange(start, end + step, step):
        conf = round(conf, 2)
        print(f"\n🔍 Running YOLO with conf={conf}...\n")

        clean_yolo_results()  # 🔹 Delete YOLO results before each run

        command = [
            "python", "yolov5/detect.py",
            "--weights", YOLO_WEIGHTS,
            "--source", source,
            "--img-size", "1024",
            "--conf", str(conf),
            "--iou-thres", "0.45",
            "--device", "cpu",
            "--save-txt",
            "--project", "yolov5/runs/detect",
            "--name", "exp",
            "--exist-ok"
        ]
        subprocess.run(command)

        print(f"✅ YOLO detection completed for {source} with conf={conf}.")

        # Determine correct label file name
        label_filename = source.split("/")[-1].replace(".tif", ".txt")
        label_path = f"yolov5/runs/detect/exp/labels/{label_filename}"
        
        # 🔹 Check if YOLO actually created the label file before proceeding
        if not os.path.exists(label_path):
            print(f"⚠️ No detections found for conf={conf}, skipping...")
            continue

        # Load and save detections to JSON
        detections = load_detections(label_path, source)
        if len(detections) > len(best_detections):  # Keep best detections
            best_conf = conf
            best_detections = detections

        time.sleep(2)  # Small delay before next run

    # Save the best detections
    with open(json_output, "w") as f:
        json.dump(best_detections, f, indent=4)
    print(f"\n📁 Detections saved to {json_output} with conf={best_conf}")

    # Visualize final detections
    visualize_bboxes(source, json_output)

# Load YOLO detections from the txt file
def load_detections(txt_path, orthophoto_path):
    detections = []
    try:
        with open(txt_path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                values = list(map(float, line.strip().split()))
                if len(values) == 5:  # Class, x_center, y_center, width, height
                    bbox = values[1:]

                    geo_bbox = convert_bbox_to_geo(bbox, orthophoto_path)
                    pixel_bbox = convert_yolo_bbox_to_pixel(bbox, orthophoto_path)

                    detections.append({
                        "id": idx,  # 🆔 Assign unique ID for traceability
                        "class": int(values[0]),
                        "bbox": bbox,
                        "pixel_bbox": pixel_bbox,
                        "geo_bbox": geo_bbox
                    })
    except FileNotFoundError:
        print(f"⚠️ No detections found at {txt_path}")
    return detections

# Convert YOLO bbox to geo-coordinates
def convert_bbox_to_geo(bbox, orthophoto_path):
    with rasterio.open(orthophoto_path) as dataset:
        transform = dataset.transform
        img_width, img_height = dataset.width, dataset.height

        x_center, y_center, w, h = bbox
        x_min = int((x_center - w / 2) * img_width)
        y_min = int((y_center - h / 2) * img_height)
        x_max = int((x_center + w / 2) * img_width)
        y_max = int((y_center + h / 2) * img_height)

        geo_min = transform * (x_min, y_min)
        geo_max = transform * (x_max, y_max)

        return [geo_min, geo_max]

# Convert YOLO bbox to pixel coordinates
def convert_yolo_bbox_to_pixel(yolo_bbox, orthophoto_path, expansion_factor=1.15):
    """Expands and converts YOLO bbox (x, y, w, h) to pixel coordinates."""
    with rasterio.open(orthophoto_path) as dataset:
        img_width, img_height = dataset.width, dataset.height

    x_center, y_center, w, h = yolo_bbox

    # Expand bounding box slightly
    w *= expansion_factor
    h *= expansion_factor

    x_min = max(0, int((x_center - w / 2) * img_width))
    x_max = min(img_width, int((x_center + w / 2) * img_width))
    y_min = max(0, int((y_center - h / 2) * img_height))
    y_max = min(img_height, int((y_center + h / 2) * img_height))

    return [(x_min, y_min), (x_max, y_max)]

# Visualize bounding boxes using the final JSON detections
def visualize_bboxes(orthophoto_path, detections_json_path):
    with open(detections_json_path, "r") as f:
        detections = json.load(f)

    if not detections:
        print("⚠️ No valid detections to visualize.")
        return

    with rasterio.open(orthophoto_path) as dataset:
        orthophoto = dataset.read([1, 2, 3]).transpose(1, 2, 0)  # Convert to RGB format

    img_height, img_width, _ = orthophoto.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(orthophoto)

    for detection in detections:
        bbox_coords = detection["pixel_bbox"]
        rect = plt.Rectangle(
            bbox_coords[0], bbox_coords[1][0] - bbox_coords[0][0], bbox_coords[1][1] - bbox_coords[0][1],
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    plt.title("Final Detected Bounding Boxes")
    plt.show()

# Run the full pipeline
run_yolo_and_save_json(ORTHOPHOTO_PATH, DETECTIONS_JSON_PATH)
