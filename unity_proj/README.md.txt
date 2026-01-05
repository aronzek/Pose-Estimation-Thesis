# Unity Drone Simulation for ODM

This Unity project simulates drone image capture for use in photogrammetry pipelines like [ODM (OpenDroneMap)](https://www.opendronemap.org/). It includes container generation, drone capture simulation, and image export for dataset creation.

## 🧭 Scene Setup

- **GeoRoot Object**  
  The `GeoRoot` object acts as the spatial reference origin and should **not** be moved or renamed. It is positioned at the **bottom-left corner of the ground plane** to prevent negative coordinate values, which can cause issues with ODM processing.

## 📦 Container Generation

1. **Delete Existing Containers**  
   Before generating new containers, remove all previous instances under the `Containers` GameObject in the hierarchy.

2. **Generate Containers**  
   - Select the `Ground` GameObject in the hierarchy.
   - Find the `ContainerSpawner` script component.
   - Click the **three-dot menu** (⋮) on the script and choose **"Generate Containers"**.
   - Containers will be placed procedurally in the scene.

3. **Save Container Data**  
   Ensure to use the **"Save Container JSON"** function after generation — this is essential for downstream data analysis.

## 📸 Drone Capture Configuration

- Open the **DroneCaptureManager** GameObject for image capture settings:
  - **Overlap Percentage** between images.
  - **Altitude** (used in the exported JSON metadata).
  - **Ground Size** to determine capture grid density.

- **Camera Settings**:
  - **Field of View** determines the visible capture area.
  - **Resolution** defines the image size.

## ▶️ Capturing Images

- Press the **Play** button in Unity.
- The simulation will automatically generate and save screenshots suitable for ODM processing.

## ⚙️ Unity Version

This project was developed using **Unity 2022.3.42f1**. Use this version to avoid compatibility issues.
