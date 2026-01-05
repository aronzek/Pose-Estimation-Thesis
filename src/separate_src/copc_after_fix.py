import open3d as o3d
import numpy as np
import laspy
import json
import os
import pdal

# Paths
COPC_PATH = "final_code/data/unity_terminal/odm_georeferenced_model.copc.laz"
DETECTIONS_JSON = "mission_1/data/detections_unity_test.json"
OUTPUT_DIR = "mission_1/data/unity_test/output_ply_files"  # Directory to save .ply files

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load COPC using LasPy
def load_copc_with_pdal(copc_path):
    """Loads the COPC point cloud using PDAL and extracts points and colors."""
    pipeline_json = f"""
    {{
        "pipeline": [
            "{copc_path}",
            {{
                "type": "filters.range",
                "limits": "Red[0:65535], Green[0:65535], Blue[0:65535]"
            }}
        ]
    }}
    """
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    arrays = pipeline.arrays
    if not arrays:
        raise RuntimeError("PDAL failed to load any points from COPC.")

    copc_data = arrays[0]

    # Extract points (X, Y, Z)
    points = np.vstack((copc_data['X'], copc_data['Y'], copc_data['Z'])).T

    # Extract colors (Red, Green, Blue) and normalize to [0, 1]
    colors = np.vstack((copc_data['Red'], copc_data['Green'], copc_data['Blue'])).T / 65535.0

    # Print Z-values for debugging
    print(f"Z-values: Min={np.min(points[:, 2])}, Max={np.max(points[:, 2])}")

    return points, colors

# Load Bounding Boxes from JSON
def load_bounding_boxes(json_path):
    """Loads bounding boxes from a JSON file and ensures min_y < max_y."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    bboxes = []
    for item in data:
        bbox = item["geo_bbox"]
        min_x, min_y = bbox[0]
        max_x, max_y = bbox[1]
        if min_y > max_y:
            min_y, max_y = max_y, min_y  # Swap to ensure min_y < max_y
        bboxes.append([[min_x, min_y], [max_x, max_y]])
    return bboxes

def find_bbox_z_range(copc_points):
    overall_median_z = np.median(copc_points[:, 2])  # Global median Z
    print(f"📏 Global Median Z: {overall_median_z:.2f}")
    return overall_median_z - 10, overall_median_z + 10  # Fixed range around median

def extract_points_in_bbox(copc_points, bbox, min_z, max_z):
    """
    Extract points within a bounding box and Z range.
    Returns:
    - points: Extracted points.
    - mask: Boolean mask used to filter the points.
    """
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[1]

    # Filter points within the X, Y, and Z ranges
    mask_x = (copc_points[:, 0] >= min_x) & (copc_points[:, 0] <= max_x)
    mask_y = (copc_points[:, 1] >= min_y) & (copc_points[:, 1] <= max_y)
    mask_z = (copc_points[:, 2] >= min_z) & (copc_points[:, 2] <= max_z)
    mask = mask_x & mask_y & mask_z

    return copc_points[mask], mask

def save_points_as_ply(points, colors, output_path):
    """
    Save extracted points as a .ply file.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

def visualize_copc_with_bboxes(copc_points, copc_colors, bboxes):
    """
    Visualize the COPC point cloud with bounding boxes.
    - COPC retains its original color.
    - Bounding boxes are displayed as blue wireframes.
    """
    # Convert COPC to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(copc_points)
    pcd.colors = o3d.utility.Vector3dVector(copc_colors)  # Preserve original colors

    # Get global Z range once
    min_z, max_z = find_bbox_z_range(copc_points)

    # Generate bounding box visualizations
    bbox_lines = []
    for idx, bbox in enumerate(bboxes):
        # Define 8 corners of the 3D bounding box using the fixed Z range
        corners = [
            [bbox[0][0], bbox[0][1], min_z],  # Bottom-left
            [bbox[1][0], bbox[0][1], min_z],  # Bottom-right
            [bbox[1][0], bbox[1][1], min_z],  # Top-right
            [bbox[0][0], bbox[1][1], min_z],  # Top-left
            [bbox[0][0], bbox[0][1], max_z],  # Upper-bottom-left
            [bbox[1][0], bbox[0][1], max_z],  # Upper-bottom-right
            [bbox[1][0], bbox[1][1], max_z],  # Upper-top-right
            [bbox[0][0], bbox[1][1], max_z],  # Upper-top-left
        ]

        # Define edges connecting corners
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]

        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector(np.array(corners))
        bbox_line_set.lines = o3d.utility.Vector2iVector(lines)
        bbox_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))  # Set bounding box color to blue

        bbox_lines.append(bbox_line_set)

        # Extract points within the bounding box
        extracted_points, mask = extract_points_in_bbox(copc_points, bbox, min_z, max_z)
        if len(extracted_points) > 0:
            print(f"✅ Bounding Box {idx + 1}: Found {len(extracted_points)} points.")
            # Save extracted points as .ply
            output_path = os.path.join(OUTPUT_DIR, f"bbox_{idx + 1}.ply")
            save_points_as_ply(extracted_points, copc_colors[mask], output_path)
            print(f"💾 Saved extracted points to {output_path}")
        else:
            print(f"❌ Bounding Box {idx + 1}: No points found.")

    # Visualize COPC with bounding boxes
    o3d.visualization.draw_geometries([pcd] + bbox_lines, window_name="COPC with Bounding Boxes")

# Main
def main():
    print("📌 Loading COPC...")
    copc_points, copc_colors = load_copc_with_pdal(COPC_PATH)
    print(f"✅ Loaded {len(copc_points)} points from COPC.")

    print("📌 Loading Bounding Boxes...")
    bboxes = load_bounding_boxes(DETECTIONS_JSON)
    print(f"✅ Loaded {len(bboxes)} bounding boxes.")

    visualize_copc_with_bboxes(copc_points, copc_colors, bboxes)

if __name__ == "__main__":
    main()