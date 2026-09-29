import cv2
from pupil_apriltags import Detector

img = cv2.imread("/mnt/x/side_navlori/test_frame_4.png", cv2.IMREAD_GRAYSCALE)
img_flip = cv2.flip(img, 1)

detector = Detector(families='tag36h11', debug=0)
tags = detector.detect(img_flip)
print(f"Flipped horizontally (pupil): found {len(tags)}")

img_flip_v = cv2.flip(img, 0)
tags_v = detector.detect(img_flip_v)
print(f"Flipped vertically (pupil): found {len(tags_v)}")

img_flip_both = cv2.flip(img, -1)
tags_b = detector.detect(img_flip_both)
print(f"Flipped both (pupil): found {len(tags_b)}")
