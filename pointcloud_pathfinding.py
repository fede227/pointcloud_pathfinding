import argparse
import json
import math
import os
import numpy as np
import open3d as o3d
from PIL import Image
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def dilate_density(matrix, buffer_pixels):
    if buffer_pixels <= 0:
        return matrix
    dilated = np.copy(matrix)
    for i in range(1, buffer_pixels + 1):
        dilated[:-i, :] = np.maximum(dilated[:-i, :], matrix[i:, :])
        dilated[i:, :] = np.maximum(dilated[i:, :], matrix[:-i, :])
        dilated[:, :-i] = np.maximum(dilated[:, :-i], matrix[:, i:])
        dilated[:, i:] = np.maximum(dilated[:, i:], matrix[:, :-i])
    return dilated

def apply_heatmap(density):
    h, w = density.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = density > 0
    d = density[mask]
    r = np.zeros_like(d)
    g = np.zeros_like(d)
    b = np.zeros_like(d)
    m1 = d < 0.33
    b[m1] = 1.0 - d[m1] / 0.33
    g[m1] = d[m1] / 0.33
    m2 = (d >= 0.33) & (d < 0.66)
    g[m2] = 1.0
    r[m2] = (d[m2] - 0.33) / 0.33
    m3 = d >= 0.66
    r[m3] = 1.0
    g[m3] = 1.0 - (d[m3] - 0.66) / 0.34
    rgb[mask, 0] = (r * 255).astype(np.uint8)
    rgb[mask, 1] = (g * 255).astype(np.uint8)
    rgb[mask, 2] = (b * 255).astype(np.uint8)
    return rgb


def pcd_to_map(pcd_path, map_png_path, z_min, z_max, resolution, buffer_pixels):
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"The file {pcd_path} does not exist.")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("The pointcloud is empty.")
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    sliced = points[z_mask]
    if sliced.size == 0:
        raise ValueError(f"No point in the selected Z interval ({z_min} - {z_max}).")
    x = sliced[:, 0]
    y = sliced[:, 1]
    x_min, y_min = float(np.min(x)), float(np.min(y))
    x_max, y_max = float(np.max(x)), float(np.max(y))
    width = int(np.ceil((x_max - x_min) / resolution)) + 1
    height = int(np.ceil((y_max - y_min) / resolution)) + 1
    density = np.zeros((height, width), dtype=np.float32)
    px = np.clip(np.floor((x - x_min) / resolution).astype(int), 0, width - 1)
    py = np.clip(np.floor((y - y_min) / resolution).astype(int), 0, height - 1)
    np.add.at(density, (py, px), 1)
    max_density = np.max(density)
    if max_density > 0:
        density /= max_density
    density = dilate_density(density, buffer_pixels)
    rgb = np.flipud(apply_heatmap(density))
    Image.fromarray(rgb).save(map_png_path)
    print(f"Map saved at: {map_png_path} ({width}x{height} px)")
    print(f"Incorporated points: {len(sliced)}")
    return rgb, x_min, y_min

def dilate_mask(mask, radius):
    out = mask.copy()
    for i in range(1, radius + 1):
        out[:-i, :] |= mask[i:, :]
        out[i:, :] |= mask[:-i, :]
    tmp = out.copy()
    for i in range(1, radius + 1):
        out[:, :-i] |= tmp[:, i:]
        out[:, i:] |= tmp[:, :-i]
    return out

def compute_weights(img, margin, penalty):
    wall = np.any(img[:, :, :3] > 0, axis=2)
    near = dilate_mask(wall, margin)
    weights = np.ones(wall.shape, dtype=np.int32)
    weights[near] = penalty
    weights[wall] = 0
    return weights

def world_to_pixel(wx, wy, x_min, y_min, resolution, height):
    col = int(round((wx - x_min) / resolution))
    row = height - 1 - int(round((wy - y_min) / resolution))
    return col, row

def pixel_to_world(col, row, x_min, y_min, resolution, height):
    wx = x_min + col * resolution
    wy = y_min + (height - 1 - row) * resolution
    return wx, wy

def map_to_ned(wx, wy, origin_x, origin_y, yaw_deg):
    dx = wx - origin_x
    dy = wy - origin_y
    psi = math.radians(yaw_deg)
    north = dx * math.cos(psi) + dy * math.sin(psi)
    east = dx * math.sin(psi) - dy * math.cos(psi)
    return north, east

def make_label(i):
    s = ""
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s

def make_labels(count):
    labels = ["S"]
    i = 1
    while len(labels) < count:
        label = make_label(i)
        i += 1
        if label != "S":
            labels.append(label)
    return labels

def main(args):
    rgb, x_min, y_min = pcd_to_map(
        args.pcd, args.map_png, args.z_min, args.z_max, args.resolution, args.buffer
    )
    height, width = rgb.shape[:2]
    ox_px, oy_px = world_to_pixel(0.0, 0.0, x_min, y_min, args.resolution, height)
    print(f"x_min={x_min:.2f} m, y_min={y_min:.2f} m")
    print(f"Map origin (0, 0) m = pixel ({ox_px}, {oy_px})")
    if args.pixel_coords:
        start = (int(args.start[0]), int(args.start[1]))
        goal = (int(args.goal[0]), int(args.goal[1]))
    else:
        start = world_to_pixel(args.start[0], args.start[1], x_min, y_min, args.resolution, height)
        goal = world_to_pixel(args.goal[0], args.goal[1], x_min, y_min, args.resolution, height)

    for name, (c, r) in (("Start", start), ("Goal", goal)):
        if not (0 <= c < width and 0 <= r < height):
            print(f"Error: {name} {(c, r)} outside map bounds ({width}x{height}).")
            return
    weights = compute_weights(rgb, args.margin, args.penalty)
    for name, (c, r) in (("Start", start), ("Goal", goal)):
        if weights[r, c] == 0:
            print(f"Error: {name} {(c, r)} is an obstacle.")
            return
    grid = Grid(matrix=weights.tolist())
    start_node = grid.node(*start)
    goal_node = grid.node(*goal)
    movement = DiagonalMovement.always if args.diagonal else DiagonalMovement.never
    finder = AStarFinder(diagonal_movement=movement)
    path, runs = finder.find_path(start_node, goal_node, grid)
    print(f"Iterations: {runs:_} path length: {len(path):_}")
    if not path:
        print("No path found.")
        return
    out = rgb.copy()
    for p in path[1:-1]:
        out[p.y, p.x] = (255, 165, 0)
    Image.fromarray(out).save(args.path_png)
    print(f"Path saved at: {args.path_png}")
    step = max(1, args.downsample)
    sampled = path[::step]
    if sampled[-1] is not path[-1]:
        sampled.append(path[-1])
    checkpoints = {}
    names = []
    labels = make_labels(len(sampled))
    for idx, node in enumerate(sampled):
        label = labels[idx]
        wx, wy = pixel_to_world(node.x, node.y, x_min, y_min, args.resolution, height)
        if not args.raw_frame:
            wx, wy = map_to_ned(wx, wy, args.origin[0], args.origin[1], args.yaw_deg)
        checkpoints[label] = {"x": round(wx, 4), "y": round(wy, 4), "z": args.json_z}
        names.append(label)
    edges = {names[i]: [names[i + 1]] for i in range(len(names) - 1)}
    edges[names[-1]] = []
    with open(args.json_out, "w") as f:
        json.dump({"checkpoints": checkpoints, "edges": edges}, f, indent=4)
    print(f"{len(names)} checkpoints written in: {args.json_out}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From "
    )
    parser.add_argument("-i", "--pcd", required=True, help=".pcd input file")
    parser.add_argument("--map-png", default="map.png", help="PNG map")
    parser.add_argument("-o", "--path-png", default="out.png", help="PNG with the path")
    parser.add_argument("-j", "--json-out", default="path.json", help="JSON with checkpoint")
    parser.add_argument("--z-min", type=float, required=True, help="zmin")
    parser.add_argument("--z-max", type=float, required=True, help="zmax")
    parser.add_argument("-r", "--resolution", type=float, default=0.1, help="meters per pixel")
    parser.add_argument("--buffer", type=int, default=0, help="security margin in pixels (dilation radius)")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument(
        "--pixel-coords",
        action="store_true"
    )
    parser.add_argument("--margin", type=int, default=3, help="security margin in meters")
    parser.add_argument("--penalty", type=int, default=5, help="pixel weight near the obstacles")
    parser.add_argument("--no-diagonal", dest="diagonal", action="store_false")
    parser.add_argument("--downsample", type=int, default=30, help="final checkpoints downsampling factor")
    parser.add_argument("--json-z", type=float, default=-5.0, help="z value of the checkpoints")
    parser.add_argument(
        "--origin",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="map origin offset compared to simulation origin",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=90.0
    )
    parser.add_argument(
        "--raw-frame",
        action="store_true",
        help="Keep raw world coordinates instead of NED frame"
    )
    main(parser.parse_args())
