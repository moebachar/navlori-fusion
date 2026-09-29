import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pupil_apriltags import Detector

def main():
    bag_path = "/mnt/x/side_navlori/kalibr_test_4"
    topic = "/camera/image_raw"
    
    tagCols = 6
    tagRows = 6
    tagSize = 0.02
    tagSpacing = 0.3 * tagSize
    
    obj_points_map = {}
    for r in range(tagRows):
        for c in range(tagCols):
            tag_id = r * tagCols + c
            x0 = c * (tagSize + tagSpacing)
            y0 = r * (tagSize + tagSpacing)
            bl = [x0, y0 + tagSize, 0]
            br = [x0 + tagSize, y0 + tagSize, 0]
            tr = [x0 + tagSize, y0, 0]
            tl = [x0, y0, 0]
            obj_points_map[tag_id] = np.array([bl, br, tr, tl], dtype=np.float32)

    detector = Detector(families='tag36h11',
                        nthreads=4,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    ts = get_typestore(Stores.ROS2_HUMBLE)
    
    all_obj_points = []
    all_img_points = []
    imsize = None
    
    count = 0
    print("Reading bag...")
    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == topic:
                count += 1
                if count % 10 != 0: continue
                
                msg = ts.deserialize_cdr(rawdata, conn.msgtype)
                if msg.encoding in ["mono8", "8UC1"]:
                    img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                elif msg.encoding in ["rgb8", "bgr8", "8UC3"]:
                    img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if count == 500: # 500th frame (since count increments by 1 before % 10 check)
                    print(f"Saving frame 500 shape {img.shape}")
                    cv2.imwrite("/mnt/x/side_navlori/test_frame_4.png", img)
                    
                if imsize is None:
                    imsize = (msg.width, msg.height)
                
                tags = detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
                
                frame_obj_pts = []
                frame_img_pts = []
                
                for t in tags:
                    tid = t.tag_id
                    if tid in obj_points_map:
                        frame_obj_pts.append(obj_points_map[tid])
                        frame_img_pts.append(t.corners)
                        
                if len(frame_obj_pts) > 8:
                    all_obj_points.append(np.vstack(frame_obj_pts))
                    all_img_points.append(np.vstack(frame_img_pts).astype(np.float32))

    if not all_obj_points:
        print("No tags found!")
        return

    print(f"Found tags in {len(all_img_points)} frames. Running calibration...")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, imsize, None, None
    )
    
    print("\n--- Calibration Results ---")
    print(f"RMS Reprojection Error: {ret:.4f} pixels")
    print(f"Image Size: {imsize[0]}x{imsize[1]}")
    print("\nCamera Matrix (fx, fy, cx, cy):")
    print(mtx)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist)

if __name__ == "__main__":
    main()
