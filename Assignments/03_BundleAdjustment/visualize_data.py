"""Visualize rendered images with projected 2D points overlaid.
Each point gets a unique color based on its index using OpenCV's applyColorMap."""

import numpy as np
import cv2
import os

OUTPUT_DIR = "data"
VIS_DIR = "data/vis"
os.makedirs(VIS_DIR, exist_ok=True)

points2d = np.load(f"{OUTPUT_DIR}/points2d.npz")

# generate a unique color per point via cv2.applyColorMap
sample_key = list(points2d.keys())[0]
n_points = len(points2d[sample_key])
indices = np.linspace(0, 255, n_points, dtype=np.uint8).reshape(1, -1)
colorbar = cv2.applyColorMap(indices, cv2.COLORMAP_HSV)  # (1, N, 3) BGR
colors = colorbar[0]  # (N, 3) BGR

# visualize selected views
for i in [0, 12, 25, 37, 49]:
    key = f"view_{i:03d}"
    img = cv2.imread(f"{OUTPUT_DIR}/images/{key}.png")

    obs = points2d[key]           # (N, 3): [x, y, visibility]
    pts = obs[:, :2]              # (N, 2)
    vis = obs[:, 2].astype(bool)  # (N,)

    for j in range(n_points):
        if vis[j]:
            x, y = int(pts[j, 0]), int(pts[j, 1])
            color = tuple(int(c) for c in colors[j])  # BGR
            cv2.circle(img, (x, y), 3, color, -1)

    cv2.imwrite(f"{VIS_DIR}/{key}_overlay.png", img)
    print(f"Saved {key}_overlay.png ({vis.sum()}/{n_points} visible)")

print("Done!")
