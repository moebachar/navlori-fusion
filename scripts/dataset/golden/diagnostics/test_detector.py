import cv2
from pupil_apriltags import Detector

img = cv2.imread("/mnt/x/side_navlori/test_frame.png", cv2.IMREAD_GRAYSCALE)
detector = Detector(families='tag36h11', debug=1)
tags = detector.detect(img)
print(f"Found {len(tags)} tags.")
for t in tags:
    print(t.tag_id)
