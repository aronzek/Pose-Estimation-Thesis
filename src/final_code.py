# Final Thesis Presentation Script
# Performs YOLO detection, extracts COPC point cloud, runs ICP, and visualizes results.

import os
import json
import shutil
import subprocess
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Disable interactive GUI
import matplotlib.pyplot as plt
import pdal
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# -------------- Configurable Constants --------------
YOLO_WEIGHTS = "final_code/data/model/igd_od_trudi-be-tu.pt"
ORTHOPHOTO_PATH = "final_code/data/unity_terminal/odm_orthophoto.tif"
DETECTIONS_JSON_PATH = "final_code/detections/detections_unity_terminal.json"
YOLO_RESULTS_DIR = "yolov5/runs/detect/exp"

# COPC Processing
COPC_PATH = "final_code/data/unity_terminal/odm_georeferenced_model.copc.laz"
OUTPUT_DIR = "final_code/data/unity_terminal/output_ply_files"
EXTRACTED_PLY_FOLDER = OUTPUT_DIR

# ICP Alignment
TARGET_PLY = "final_code/data/semi_trailer.ply"
OUTPUT_ROTATIONS = "final_code/outputs/unity_rotations.txt"
OUTPUT_LOG = "final_code/outputs/unity_output.txt"

#Visualization
YOLO_DETECTION_IMG = "final_code/outputs/_unity_yolo_detections.png"
EXPANSION_FACTOR = 1.15

# ===================== YOLO DETECTION =====================
def clean_yolo_results():
    """Remove previous YOLO results directory."""
    if os.path.exists(YOLO_RESULTS_DIR):
        shutil.rmtree(YOLO_RESULTS_DIR)

def run_yolo_and_save_json(source, json_output, start=0.01, end=0.01, step=0.01):
    """Run YOLO detection and save results as JSON."""
    best_conf, best_detections = None, []
    
    for conf in np.arange(start, end + step, step):
        conf = round(conf, 2)
        clean_yolo_results()
        
        subprocess.run([
            "python", "yolov5/detect.py",
            "--weights", YOLO_WEIGHTS, "--source", source,
            "--img-size", "1024", "--conf", str(conf),
            "--iou-thres", "0.45", "--device", "cpu",
            "--save-txt", "--project", "yolov5/runs/detect",
            "--name", "exp", "--exist-ok"])
            
        label_path = f"{YOLO_RESULTS_DIR}/labels/{os.path.basename(source).replace('.tif', '.txt')}"
        if not os.path.exists(label_path):
            continue
            
        detections = load_detections(label_path, source)
        if len(detections) > len(best_detections):
            best_conf, best_detections = conf, detections

    with open(json_output, "w") as f:
        json.dump(best_detections, f, indent=4)
    visualize_bboxes(source, json_output, save_path=YOLO_DETECTION_IMG)


def load_bounding_boxes(json_path):
    """Loads bounding boxes from a JSON file and ensures min < max for each axis."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    bboxes = []
    for item in data:
        bbox = item["geo_bbox"]
        min_x, min_y = bbox[0]
        max_x, max_y = bbox[1]

        # Swap if YOLO ordering is wrong
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        bboxes.append([[min_x, min_y], [max_x, max_y]])
    return bboxes

def load_detections(txt_path, ortho_path):
    """Load YOLO detections from text file."""
    detections = []
    with open(txt_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
            values = list(map(float, line.strip().split()))
            if len(values) == 5:
                bbox = values[1:]
                geo_bbox = convert_bbox_to_geo(bbox, ortho_path)
                pixel_bbox = convert_yolo_bbox_to_pixel(bbox, ortho_path)
                detections.append({
                    "id": idx, 
                    "class": int(values[0]), 
                    "bbox": bbox, 
                    "pixel_bbox": pixel_bbox, 
                    "geo_bbox": geo_bbox
                })
    return detections

def convert_bbox_to_geo(bbox, path):
    """Convert YOLO bbox to geographic coordinates."""
    with rasterio.open(path) as ds:
        transform = ds.transform
        w, h = ds.width, ds.height
        x_c, y_c, bw, bh = bbox
        x_min = int((x_c - bw / 2) * w)
        y_min = int((y_c - bh / 2) * h)
        x_max = int((x_c + bw / 2) * w)
        y_max = int((y_c + bh / 2) * h)
        return [transform * (x_min, y_min), transform * (x_max, y_max)]

def convert_yolo_bbox_to_pixel(bbox, path):
    """Convert YOLO bbox to pixel coordinates with expansion."""
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
    x_c, y_c, bw, bh = bbox
    bw *= EXPANSION_FACTOR
    bh *= EXPANSION_FACTOR
    x_min = max(0, int((x_c - bw / 2) * w))
    x_max = min(w, int((x_c + bw / 2) * w))
    y_min = max(0, int((y_c - bh / 2) * h))
    y_max = min(h, int((y_c + bh / 2) * h))
    return [(x_min, y_min), (x_max, y_max)]

def visualize_bboxes(ortho_path, json_path, save_path=None):
    """Visualize detected bounding boxes."""
    with open(json_path) as f:
        detections = json.load(f)
    if not detections:
        return
        
    with rasterio.open(ortho_path) as ds:
        img = ds.read([1, 2, 3]).transpose(1, 2, 0)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for d in detections:
        (x0, y0), (x1, y1) = d["pixel_bbox"]
        ax.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, 
            edgecolor='red', facecolor='none', lw=2))
    plt.title("Detected Bounding Boxes")
    if save_path:
        plt.savefig(save_path)
    plt.close()

# ===================== COPC PROCESSING =====================
def load_las_or_ply(file_path):
    """Enhanced PDAL loader with color range filtering and better error handling"""
    try:
        # Build the PDAL pipeline
        pipeline_json = f"""
        {{
            "pipeline": [
                "{file_path}",
                {{
                    "type": "filters.range",
                    "limits": "Red[0:65535], Green[0:65535], Blue[0:65535]"
                }}
            ]
        }}
        """
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        # Validate output
        if not pipeline.arrays:
            raise RuntimeError(f"No data loaded from {file_path}")
            
        data = pipeline.arrays[0]
        
        # Extract points and colors
        points = np.vstack((data['X'], data['Y'], data['Z'])).T
        colors = np.vstack((data['Red'], data['Green'], data['Blue'])).T / 65535.0
        
        # Debug output
        print(f"Loaded {len(points)} points from {os.path.basename(file_path)}")
        print(f"Color ranges - R: {np.min(colors[:,0]):.2f}-{np.max(colors[:,0]):.2f} "
              f"G: {np.min(colors[:,1]):.2f}-{np.max(colors[:,1]):.2f} "
              f"B: {np.min(colors[:,2]):.2f}-{np.max(colors[:,2]):.2f}")
              
        return points, colors
        
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {str(e)}")
        raise

def save_points_as_ply(points, colors, path):
    """Save point cloud as PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def extract_points_in_bbox(points, bbox, z_range):
    """Extract points within a bounding box and Z range."""
    (x0, y0), (x1, y1) = bbox
    z_min, z_max = z_range
    mask = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1) &
        (points[:, 1] >= y0) & (points[:, 1] <= y1) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask], mask

# ===================== ICP ALIGNMENT =====================
def save_o3d_visualization(geometries, filepath, width=800, height=600):
    """Save a screenshot of the Open3D scene."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filepath)
    vis.destroy_window()


def compute_normals(pcd):
    """Compute normals for point cloud."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def align_centroids(source, target):
    """Align centroids of two point clouds."""
    src_ctr = np.mean(np.asarray(source.points), axis=0)
    tgt_ctr = np.mean(np.asarray(target.points), axis=0)
    translation = tgt_ctr - src_ctr
    T = np.identity(4)
    T[:3, 3] = translation
    source.transform(T)
    return source

def basic_icp(source, target, threshold=1.0, max_iterations=1000):
    """Perform basic ICP alignment."""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    return reg.transformation, reg.fitness

def multi_stage_icp(source, target, thresholds=[1.0, 0.5, 0.1], max_iterations=[1000, 1000, 1000]):
    """Perform multi-stage ICP alignment."""
    transformation = np.identity(4)
    fitness = 0.0
    for threshold, max_iteration in zip(thresholds, max_iterations):
        reg = o3d.pipelines.registration.registration_icp(
            source, target, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        transformation, fitness = reg.transformation, reg.fitness
    return transformation, fitness

def extract_euler_deg(matrix):
    """Extract Euler angles from transformation matrix."""
    return R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)

def save_rotation(euler, file_path, idx, fitness):
    """Save rotation angles to file."""
    roll, pitch, yaw = euler
    with open(OUTPUT_ROTATIONS, "a") as f:
        f.write(f"Container {idx}: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f} | ICP fitness={fitness:.4f}\n")
    with open(OUTPUT_LOG, "a") as log:
        log.write(f"Container {idx} Euler angles: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}\n")

# ===================== MAIN PIPELINE =====================
def main():
    print("🚀 Starting full pipeline...")
    
    # ==================================================================
    # Step 0: Verify ALL Directories and Files Exist
    # ==================================================================
    REQUIRED_PATHS = {
        "YOLO weights": YOLO_WEIGHTS,
        "Orthophoto": ORTHOPHOTO_PATH,
        "COPC file": COPC_PATH,
        "Target PLY": TARGET_PLY,
        "Output directory": OUTPUT_DIR,
        "Detections directory": os.path.dirname(DETECTIONS_JSON_PATH)
    }

    # Check file existence
    for name, path in REQUIRED_PATHS.items():
        if not os.path.exists(path):
            if name.endswith("directory"):
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"✅ Created directory: {path}")
                except Exception as e:
                    print(f"❌ Failed to create {name}: {path}")
                    print(f"Error: {str(e)}")
                    return
            else:
                print(f"❌ Missing {name}: {path}")
                print("Please verify the file exists at this relative path")
                return
        else:
            print(f"✅ Found {name}: {path}")

    # Special check for output directory writability
    try:
        test_file = os.path.join(OUTPUT_DIR, "__test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ Output directory is writable")
    except Exception as e:
        print(f"❌ Cannot write to output directory: {OUTPUT_DIR}")
        print(f"Error: {str(e)}")
        print("Try:")
        print(f"chmod 777 {OUTPUT_DIR}")
        return

    # ==================================================================
    # Step 2: YOLO Detection
    # ==================================================================
    print("\n🔍 Phase 1: Running YOLO detection...")
    run_yolo_and_save_json(ORTHOPHOTO_PATH, DETECTIONS_JSON_PATH)
    
    # Verify detections exist
    if not os.path.exists(DETECTIONS_JSON_PATH):
        print("❌ No detection JSON file created!")
        return
        
    with open(DETECTIONS_JSON_PATH) as f:
        detections = json.load(f)
        print(f"Found {len(detections)} containers in detection results")

    # ==================================================================
    # Step 3: COPC Processing
    # ==================================================================
    print("\n☁️ Phase 2: Processing COPC data...")
    
    # Load COPC with validation
    try:
        copc_pts, copc_colors = load_las_or_ply(COPC_PATH)
        print(f"Loaded {len(copc_pts):,} points from COPC file")
        print(f"X Range: {np.min(copc_pts[:, 0]):.2f} to {np.max(copc_pts[:, 0]):.2f}")
        print(f"Y Range: {np.min(copc_pts[:, 1]):.2f} to {np.max(copc_pts[:, 1]):.2f}")
        print(f"Z Range: {np.min(copc_pts[:, 2]):.2f} to {np.max(copc_pts[:, 2]):.2f}")
    except Exception as e:
        print(f"❌ Failed to load COPC: {str(e)}")
        return

    # Process bounding boxes
    bboxes = load_bounding_boxes(DETECTIONS_JSON_PATH)
    for idx, bbox in enumerate(bboxes):
        print(f"\n📦 Container {idx + 1} Geo BBox:")
        print(f"Min: {bbox[0]}")
        print(f"Max: {bbox[1]}")

    z_med = np.median(copc_pts[:, 2])
    z_range = (z_med - 10, z_med + 10)
    print(f"\nUsing Z range: {z_range[0]:.2f} to {z_range[1]:.2f}")

    # ==================================================================
    # Step 4: Point Cloud Extraction
    # ==================================================================
    print("\n✂️ Phase 3: Extracting point clouds...")
    for i, bbox in enumerate(bboxes):
        print(f"\nProcessing Container {i+1}:")
        
        # Validate bbox coordinates
        if (bbox[0][0] >= bbox[1][0]) or (bbox[0][1] >= bbox[1][1]):
            print(f"⚠️ Invalid bbox (min >= max), skipping")
            continue
            
        extracted, mask = extract_points_in_bbox(copc_pts, bbox, z_range)
        print(f"Extracted {len(extracted)} points")
        
        if len(extracted) == 0:
            print("⚠️ No points extracted, possible causes:")
            print("- Bounding box outside COPC coverage")
            print("- Z-range too restrictive")
            print("- Invalid bounding box coordinates")
            continue
            
        output_path = os.path.join(EXTRACTED_PLY_FOLDER, f"container_{i+1}.ply")
        try:
            save_points_as_ply(extracted, copc_colors[mask], output_path)
            print(f"✅ Saved to {output_path}")
            print(f"File exists: {os.path.exists(output_path)}")
            print(f"File size: {os.path.getsize(output_path):,} bytes")
        except Exception as e:
            print(f"❌ Failed to save PLY: {str(e)}")

    # ==================================================================
    # Step 5: ICP Alignment
    # ==================================================================
    print("\n🔄 Phase 4: ICP Alignment...")

    # Load target model
    try:
        tgt_pts, tgt_cols = load_las_or_ply(TARGET_PLY)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
        tgt_pcd.colors = o3d.utility.Vector3dVector(np.full_like(tgt_pts, [1, 0, 0]))
        tgt_pcd = compute_normals(tgt_pcd)
        print(f"✅ Loaded target with {len(tgt_pts)} points")
    except Exception as e:
        print(f"❌ Failed to load target: {str(e)}")
        return

    # Process each extracted container
    for idx, fname in enumerate(sorted(os.listdir(EXTRACTED_PLY_FOLDER))):
        if not fname.endswith(".ply"):
            continue

        path = os.path.join(EXTRACTED_PLY_FOLDER, fname)
        print(f"\n📦 Aligning {fname}...")

        try:
            pts, cols = load_las_or_ply(path)
            if len(pts) < 100:
                print(f"⚠️ Too few points ({len(pts)}), skipping")
                continue

            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(pts)
            src_pcd.colors = o3d.utility.Vector3dVector(np.full_like(pts, [0, 0, 1]))
            src_pcd = compute_normals(align_centroids(src_pcd, tgt_pcd))

            # Run Basic ICP
            transformation, fitness = basic_icp(src_pcd, tgt_pcd)
            print(f"🔧 Basic ICP Fitness: {fitness:.4f}")

            if fitness > 0.7:
                print(f"✅ Accepted container {idx + 1}")
            elif fitness >= 0.3:
                print(f"🔄 Running multi-stage ICP for container {idx + 1}...")
                refined_transformation, refined_fitness = multi_stage_icp(src_pcd, tgt_pcd)
                print(f"🔧 Refined Fitness: {refined_fitness:.4f}")

                if refined_fitness >= fitness:
                    transformation = refined_transformation
                    fitness = refined_fitness
                    print("✅ Using refined transformation")
                else:
                    print("🔁 Keeping basic ICP result")

            else:
                print(f"⏩ Skipping container {idx + 1} (Fitness too low)")
                continue

            # Save rotation
            euler_deg = extract_euler_deg(transformation)
            print(f"🧭 Rotation (Yaw/Pitch/Roll): {euler_deg}")
            save_rotation(euler_deg, OUTPUT_ROTATIONS, idx + 1, fitness)

            # Save PNG visualization for first 3 alignments
            if idx < 3:
                aligned = src_pcd.transform(transformation.copy())
                vis_path = os.path.join("final_code/outputs", f"icp_alignment_{idx + 1}.png")
                save_o3d_visualization([tgt_pcd, aligned], vis_path)
                print(f"📸 Saved visualization to {vis_path}")

        except Exception as e:
            print(f"❌ Error processing {fname}: {str(e)}")

    print("\n✅ ICP Phase completed!")
    print("\n✅ Pipeline completed!")

if __name__ == "__main__":
    main()