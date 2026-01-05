import open3d as o3d
import numpy as np
import os
import pdal
from scipy.spatial.transform import Rotation as R

# Paths
PLY_FOLDER = "mission_1/data/south_terminal/output_ply_files"  # Folder with container .ply files
TARGET_PLY = "final_code/data/semi_trailer.ply"  # Fixed target .ply file
OUTPUT_ROTATIONS = "mission_1/data/south_terminal/rotations.txt"  # File to save rotation matrices


def load_ply_with_pdal(file_path):
    """Load a .ply file using PDAL and return points and colors."""
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

    arrays = pipeline.arrays
    if not arrays:
        raise RuntimeError(f"PDAL failed to load points from {file_path}")

    data = arrays[0]
    points = np.vstack((data['X'], data['Y'], data['Z'])).T
    colors = np.vstack((data['Red'], data['Green'], data['Blue'])).T / 65535.0
    return points, colors


def load_ply_as_open3d(points, colors, color_override=None):
    """Convert numpy arrays to an Open3D point cloud with optional color override."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color_override is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), color_override))  # Override colors
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def load_ply_with_open3d(file_path):
    """Load a .ply file from disk using Open3D and return points and colors."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors

def compute_normals(pcd, radius=0.1, max_nn=30):
    """Compute normals for a point cloud."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd


def compute_centroid(pcd):
    """Compute the centroid of a point cloud."""
    return np.mean(np.asarray(pcd.points), axis=0)


def align_centroids(source_pcd, target_pcd):
    """Translate the source point cloud so its centroid matches the target's centroid."""
    source_centroid = compute_centroid(source_pcd)
    target_centroid = compute_centroid(target_pcd)

    translation = target_centroid - source_centroid
    transformation = np.identity(4)
    transformation[:3, 3] = translation  # Set translation part

    source_pcd.transform(transformation)  # Apply translation
    return source_pcd


def basic_icp(source, target, threshold=1.0, max_iterations=1000):
    """Perform a basic ICP alignment."""
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return reg_p2p.transformation, reg_p2p.fitness


def multi_stage_icp(source, target, thresholds=[1.0, 0.5, 0.1], max_iterations=[1000, 1000, 1000]):
    """Perform multi-stage ICP alignment."""
    transformation = np.identity(4)
    fitness = 0.0

    for threshold, max_iteration in zip(thresholds, max_iterations):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )
        transformation = reg_p2p.transformation
        fitness = reg_p2p.fitness
        print(f"🔧 ICP Stage: Threshold={threshold}, Fitness={fitness:.4f}")

    return transformation, fitness


def extract_euler_angles(transformation):
    """Extract Euler angles (in degrees) from a 4x4 transformation matrix."""
    rotation_matrix = transformation[:3, :3]
    r = R.from_matrix(rotation_matrix)
    # You can specify the order: 'xyz' means roll, pitch, yaw
    euler_deg = r.as_euler('xyz', degrees=True)
    return euler_deg  # returns [roll, pitch, yaw]

def save_rotation_degrees(euler_deg, file_path, container_id):
    """Save the Euler angles (in degrees) to a file."""
    roll, pitch, yaw = euler_deg
    with open(file_path, "a") as f:
        f.write(f"Container {container_id} Rotation (degrees):\n")
        f.write(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}\n\n")


def main():
    # Load target point cloud using PDAL
    target_points, target_colors = load_ply_with_pdal(TARGET_PLY)
    target_pcd = load_ply_as_open3d(target_points, target_colors, color_override=[1, 0, 0])  # Red for target
    target_pcd = compute_normals(target_pcd)

    for idx, ply_file in enumerate(os.listdir(PLY_FOLDER)):
        if not ply_file.endswith(".ply"):
            continue

        # Load container point cloud using PDAL
        container_path = os.path.join(PLY_FOLDER, ply_file)
        container_points, container_colors = load_ply_with_pdal(container_path)
        container_pcd = load_ply_as_open3d(container_points, container_colors, color_override=[0, 0, 1])  # Blue for source
        container_pcd = compute_normals(container_pcd)

        if len(container_pcd.points) < len(target_pcd.points):
            print(f"⏩ Skipping container {idx + 1} (fewer points than target).")
            continue

        # Bring centroids closer before ICP
        container_pcd = align_centroids(container_pcd, target_pcd)
        print(f"📍 Aligned centroids for container {idx + 1}")

        #Visualize before alignment
        o3d.visualization.draw_geometries([target_pcd, container_pcd], window_name=f"Before ICP - Container {idx + 1}")

        # Run Basic ICP
        transformation, fitness = basic_icp(container_pcd, target_pcd)
        print(f"🔧 Basic ICP Fitness: {fitness:.4f}")

        if fitness > 0.7:
            print(f"✅ Accepting result for container {idx + 1} (Good fitness).")
        elif fitness >= 0.3:
            print(f"🔄 Running multi-stage ICP for container {idx + 1} (Moderate fitness).")
    
            # Run multi-stage ICP
            refined_transformation, refined_fitness = multi_stage_icp(container_pcd, target_pcd)
            print(f"🔧 Final ICP Fitness after multi-stage: {refined_fitness:.4f}")

            # Keep multi-ICP result **only if it improves or is comparable**
            if refined_fitness >= fitness or refined_fitness > 0.3:
                print(f"✅ Using refined multi-ICP result for container {idx + 1}.")
                transformation, fitness = refined_transformation, refined_fitness
            else:
                print(f"🔄 Multi-stage ICP did not improve fitness, keeping original result.")

        else:
            print(f"⏩ Skipping container {idx + 1} (Fitness too low).")
            continue

        # Extract rotation and save
        euler_deg = extract_euler_angles(transformation)
        print(f"🔄 Rotation Matrix:\n{euler_deg}")
        save_rotation_degrees(euler_deg, OUTPUT_ROTATIONS, idx + 1)

        # Visualize result with distinct colors
        #container_pcd.transform(transformation)
        #o3d.visualization.draw_geometries([target_pcd, container_pcd], window_name=f"Aligned Container {idx + 1}")

if __name__ == "__main__":
    main()