📦 Thesis Pipeline: 3D Container Pose Estimation
This repository contains the full pipeline used in my master's thesis for 3D container detection, georeferenced bounding box extraction, and pose estimation via ICP.

🔧 Environment Setup
Tested with Python 3.10 on Ubuntu. Works with Conda + Mamba.

Create and activate the environment:

conda env create -f final_environment.yml
conda activate thesis-pipeline

Clone YOLOv5:

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..

🗂️ Project Structure
pose_project/
├── final_code/
│ ├── src/
│ │ └── final_code.py # Main pipeline script
│ ├── outputs/
│ │ └── icp_alignment_*.png # Saved alignment visualizations
│ └── detections.json # YOLOv5 output with geo-coordinates
├── data/
│ ├── orthophoto.tif # GeoTIFF input
│ ├── model.ply # Reference 3D model
│ ├── extracted_containers/ # COPC-extracted container PLYs
├── final_environment.yml # Conda environment
└── yolov5/ # YOLOv5 repo

🚀 How to Run
From inside the pose_project/final_code/src/ folder:

python final_code.py

This will:

Run object detection on the georeferenced orthophoto (via YOLOv5)

Convert YOLO detections into real-world coordinates

Extract bounding-box regions from the COPC point cloud

Perform ICP to estimate pose (rotation) of each detected container

Save:

.txt with Euler angles + ICP fitness

.png visualizations for 3 alignments

📄 Outputs
rotation_output.txt: Pose estimation results, including fitness scores

icp_alignment_*.png: Visualization of the first 3 ICP alignments

📌 Notes
All ICP alignment thresholds, visualizations, and output paths are defined in final_code.py

Ensure COPC and orthophoto share the same CRS (EPSG:32633)

Visualizations are headless (no GUI) for safe remote use

👨‍🔬 For Reviewers
This pipeline was used to evaluate pose estimation robustness using YOLOv5 bounding boxes + Open3D-based ICP.
You can modify the final_code.py script to test with new regions, new models, or synthetic data.