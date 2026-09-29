import sys
import numpy as np
import cv2
import pandas as pd
import os

def render(mirror_x, mirror_y):
    scale = 244.0
    tx = 12325.0
    ty = 3865.0
    theta = -66.8
    run_name = "run3"
    
    floorplan = cv2.imread("/mnt/x/side_navlori/floorplan.jpg")
    poses_df = pd.read_csv(f"/mnt/x/side_navlori/data/{run_name}/ground_truth/gt_pose.csv")
    pts = poses_df[['x', 'y']].values
    
    if mirror_x: pts[:, 0] = -pts[:, 0]
    if mirror_y: pts[:, 1] = -pts[:, 1]
    
    th = np.radians(theta)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts_tf = (pts @ R.T) * scale + np.array([tx, ty])
    
    # We will just crop the bounding box of the points to make it fast to check
    min_x, min_y = pts_tf.min(axis=0).astype(int) - 500
    max_x, max_y = pts_tf.max(axis=0).astype(int) + 500
    
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(floorplan.shape[1], max_x)
    max_y = min(floorplan.shape[0], max_y)
    
    overlay = floorplan[min_y:max_y, min_x:max_x].copy()
    
    # Adjust pts to cropped region
    pts_tf -= np.array([min_x, min_y])
    
    for x, y in pts_tf:
        ix, iy = int(x), int(y)
        if 0 <= ix < overlay.shape[1] and 0 <= iy < overlay.shape[0]:
            cv2.circle(overlay, (ix, iy), 6, (255, 0, 0), -1)
            
    name = f"test_mx{mirror_x}_my{mirror_y}.png"
    cv2.imwrite(f"/mnt/x/side_navlori/{name}", overlay)
    print(f"Saved {name}")

def main():
    render(False, False)
    render(True, False)
    render(False, True)
    render(True, True)

if __name__ == "__main__":
    main()
