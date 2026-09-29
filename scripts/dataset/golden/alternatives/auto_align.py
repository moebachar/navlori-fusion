import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def rasterize_points(pts, resolution=0.05):
    # pts: N x 2 array in meters
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    
    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_y - min_y) / resolution)) + 1
    
    img = np.zeros((height, width), dtype=np.uint8)
    
    pts_idx = np.round((pts - [min_x, min_y]) / resolution).astype(int)
    for x, y in pts_idx:
        if 0 <= y < height and 0 <= x < width:
            img[y, x] = 255
            
    # dilate to make lines thicker
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    return img, (min_x, min_y)

def main():
    img_path = "/mnt/x/side_navlori/floorplan.jpg"
    print(f"Loading floorplan: {img_path}")
    floorplan = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if floorplan is None:
        print("Could not load floorplan.")
        sys.exit(1)
        
    # Invert floorplan if background is white (we want walls to be white/high value for matching)
    if np.mean(floorplan) > 127:
        floorplan = 255 - floorplan
        
    print("Loading run3 SLAM map...")
    map_data = np.load("/mnt/x/side_navlori/data/run3/ground_truth/map_points.npz")
    pts = map_data["xy"]
    
    print("Rasterizing SLAM map...")
    resolution = 0.05 # 5cm per pixel
    template_img, (min_x, min_y) = rasterize_points(pts, resolution)
    
    best_val = -1
    best_params = None
    best_loc = None
    
    # We will search scales from 20 px/m to 100 px/m
    # In our template, 1 pixel = 0.05m -> 20 px/m.
    # So if actual scale is S px/m, we need to resize template by (S / 20).
    
    print("Searching for best alignment...")
    for scale in np.arange(20, 100, 5):
        resize_factor = scale / (1.0 / resolution)
        resized_template = cv2.resize(template_img, (0,0), fx=resize_factor, fy=resize_factor)
        
        for theta in np.arange(0, 360, 5):
            # Rotate the template
            h, w = resized_template.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), theta, 1.0)
            # Calculate bounding box for rotated image to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - w // 2
            M[1, 2] += (nH / 2) - h // 2
            
            rotated_template = cv2.warpAffine(resized_template, M, (nW, nH))
            
            if rotated_template.shape[0] > floorplan.shape[0] or rotated_template.shape[1] > floorplan.shape[1]:
                continue
                
            res = cv2.matchTemplate(floorplan, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_params = (scale, theta, resize_factor, M)
                best_loc = max_loc

    print(f"Best match score: {best_val}")
    if best_params is None:
        print("No match found.")
        sys.exit(1)
        
    scale, theta, resize_factor, M = best_params
    tx, ty = best_loc
    print(f"Best scale: {scale} px/m, theta: {theta} deg, tx: {tx}, ty: {ty}")
    
    # Let's render the result
    color_floorplan = cv2.imread(img_path)
    
    # Transform points to draw them
    th_rad = -np.radians(theta) # OpenCV rotation is counter-clockwise for negative theta, wait...
    # We will just transform the points directly
    # 1. shift to origin (min_x, min_y)
    # 2. scale by `scale`
    # 3. shift to center of template
    # 4. rotate by theta
    # 5. add tx, ty
    # Actually, easiest is just to overlay the rotated_template.
    
    best_rotated_template = cv2.warpAffine(cv2.resize(template_img, (0,0), fx=resize_factor, fy=resize_factor), M, (M.shape[1]*2, M.shape[0]*2)) # roughly
    
    # We can just redraw the points!
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R = np.array([[c, s], [-s, c]]) # check sign later if needed
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(color_floorplan, cv2.COLOR_BGR2RGB))
    
    # Brute force transform points based on template matching
    # In template matching, the top-left of the rotated template is at `best_loc`.
    # Let's just overlay the template image in red on top of the floorplan.
    h, w = color_floorplan.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Recreate the best rotated template
    resized_template = cv2.resize(template_img, (0,0), fx=resize_factor, fy=resize_factor)
    h_t, w_t = resized_template.shape
    M = cv2.getRotationMatrix2D((w_t//2, h_t//2), theta, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
    nW = int((h_t * sin) + (w_t * cos))
    nH = int((h_t * cos) + (w_t * sin))
    M[0, 2] += (nW / 2) - w_t // 2
    M[1, 2] += (nH / 2) - h_t // 2
    rotated_template = cv2.warpAffine(resized_template, M, (nW, nH))
    
    # Paste into overlay
    try:
        mask = rotated_template > 0
        overlay_region = overlay[ty:ty+nH, tx:tx+nW]
        overlay_region[mask] = [255, 0, 0] # red
        overlay[ty:ty+nH, tx:tx+nW] = overlay_region
    except Exception as e:
        print(f"Error pasting overlay: {e}")
        
    plt.imshow(overlay, alpha=0.5)
    plt.savefig("/mnt/x/side_navlori/alignment_result.png")
    print("Saved /mnt/x/side_navlori/alignment_result.png")

if __name__ == "__main__":
    main()
