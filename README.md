# PointCloud Pathfinding with Python

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-FC4F37?style=flat&logoColor=white)](http://www.open3d.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

This Python script is a tool that I developed to validate my work on LiDAR-based SLAM to use the 3-dimensional map built by the SLAM algorithm to calculate obstacle-free trajectories using the A* algorithm. 

## Dependencies

The script uses the `Open3D`, `Pillow`/`PIL`, `json`, `pathfinding` libraries. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/be57d8c0-8bbd-46a6-aa34-c95565457526" alt="Screenshot" width="650" />
</div>

## Map Projection

The way the script works is by projecting a slice of the 3D map, perpendicular to the z-axis, onto a 2D image. To achieve this the script requires the z-min and the z-max values to select the points that will be part of the image. The idea behind these coordinates is to allow the user to make a rough estimate of the altitude that the vehicle will have while following the calculated trajectory. 
In the images shown below you can see the results of the image projection. 

<div align="center">
    <img height="380" alt="map" src="https://github.com/user-attachments/assets/72dc84d8-be67-4002-a4e0-41f9f7a58ce3" />
  <img src="https://github.com/user-attachments/assets/f7138118-ffb1-46c9-ac8b-a2c7fc857506" alt="test" width="600" />
</div>
On the second image you can observe light grey areas highlighting the security margins added around the obstacles.

## Pathfinding & Trajectory Generation

After projecting the entire map the A* algorithm is used to estimate an obstacle free trajectory between the desired starting and ending points of the map, provided they are reachable and not occupied by obstacles. At this stage a weighted map is used to avoid trajectories that run too close to the obstacles. 

This operation can be customized with the margin and penalty launch parameters, which allow the user to select the number of pixels around an obstacle which are considered impassable and the weight that will be attributed to the pixels near the obstacle respectively (3 and 5 are the default values). Furthermore, it is possible with the no-diagonal launch parameter to disable or enable diagonal movement on the map (default is allowed).

Below you can observe the map image with the drawn-on trajectory.

<div align="center">

<img width="600"  alt="Screenshot from 2026-11-09 23-11-06" src="https://github.com/user-attachments/assets/85bbfa76-b515-4983-a8e9-06aebc7b5ef5" />

<img width="600"  alt="Screenshot from 2026-11-09 23-11-20" src="https://github.com/user-attachments/assets/e8795716-3492-4d59-87ad-1cbf36039a2d" />

</div>

## Output & Frame Conversion

After this phase the pixels of the trajectory are converted into 3D coordinates and are used to fill a JSON file with a downsampled number of checkpoints (downsampling factor can be specified with the downsample parameter) ready to be used in an offboard control simulation. These coordinates are all at the same altitude specified through the json-z parameter and are converted to the NED frame to be used with the PX4 controller software toolchain, but raw coordinates can be produced setting the raw-frame parameter. 

---

## How to Run the Script

### Installation Requirements
```bash
pip install open3d pillow numpy pathfinding
```
### Execution Example

To project a .pcd file into a 2D map and calculate a trajectory between two world points (X, Y):
```bash

python pathfinding_pcd.py \
    -i input_cloud.pcd \
    --z-min 0.5 \
    --z-max 2.0 \
    --start 0.0 0.0 \
    --goal 15.5 10.2 \
    -r 0.1 \
    --margin 3 \
    --penalty 5 \
    --downsample 30 \
    --json-z -5.0 \
    -o trajectory_out.png \
    -j checkpoints.json
```
# Command-Line Arguments Reference Guide

This guide describes all the available parameters and configuration flags for running the PointCloud pathfinding script.

---

## Input & Output Files

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-i`, `--pcd` | `string` | *None* | **[Required]** Path to the input 3D point cloud file (`.pcd`). |
| `--map-png` | `string` | `map.png` | Filename/path where the generated 2D density heatmap map will be saved. |
| `-o`, `--path-png` | `string` | `out.png` | Filename/path where the final trajectory-drawn map image will be saved. |
| `-j`, `--json-out` | `string` | `path.json` | Filename/path for the output JSON file containing checkpoints and edges. |

---

## Map Slicing & Resolution

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--z-min` | `float` | *None* | **[Required]** Minimum Z-axis threshold (in meters) to slice the point cloud. |
| `--z-max` | `float` | *None* | **[Required]** Maximum Z-axis threshold (in meters) to slice the point cloud. |
| `-r`, `--resolution` | `float` | `0.1` | Map resolution in meters per pixel. Controls the grid granularity. |
| `--buffer` | `int` | `0` | Obstacle density dilation buffer measured in pixels. |

---

## Pathfinding & Costmap Configuration

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--start` | `float` (x2) | *None* | **[Required]** Starting world coordinates `X Y` (or pixel coordinates if `--pixel-coords` is set). |
| `--goal` | `float` (x2) | *None* | **[Required]** Goal/Destination world coordinates `X Y` (or pixel coordinates if `--pixel-coords` is set). |
| `--pixel-coords` | `flag` | `False` | If set, treats `--start` and `--goal` values directly as pixel rows/cols instead of meters. |
| `--margin` | `int` | `3` | Security margin radius around obstacles (in pixels) where safety penalties apply. |
| `--penalty` | `int` | `5` | Cost weight/penalty assigned to grid pixels located near obstacles. |
| `--no-diagonal` | `flag` | `False` | If set, disables 8-neighbor diagonal movement, restricting the A* algorithm to orthogonal movements only. |

---

## Checkpoints, Frames & Transformation

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--downsample` | `int` | `30` | Step factor used to downsample the path points into final waypoints/checkpoints for the JSON. |
| `--json-z` | `float` | `-5.0` | Fixed Z altitude coordinate assigned to all exported JSON checkpoints. |
| `--origin` | `float` (x2) | `0.0 0.0` | Map origin offset `X Y` compared to the simulation world origin. |
| `--yaw-deg` | `float` | `90.0` | Yaw rotation angle in degrees applied during the NED coordinate frame conversion. |
| `--raw-frame` | `flag` | `False` | If set, keeps raw map coordinates instead of converting them into the NED frame. |
