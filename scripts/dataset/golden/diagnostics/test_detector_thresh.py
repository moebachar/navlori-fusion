import cv2
from pupil_apriltags import Detector

img = cv2.imread("/mnt/x/side_navlori/test_frame_4.png", cv2.IMREAD_GRAYSCALE)

detector = Detector(families='tag36h11', debug=0)

for thresh in range(80, 200, 20):
    _, b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    tags = detector.detect(b)
    if len(tags) > 0:
        print(f"Found {len(tags)} at thresh {thresh}")

b_adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
tags = detector.detect(b_adap)
print(f"Adaptive: found {len(tags)}")

img_blur = cv2.GaussianBlur(img, (3,3), 0)
tags = detector.detect(img_blur)
print(f"Blur: found {len(tags)}")

# crop out a single tag
tag_roi = img[200:300, 100:200]
cv2.imwrite("/mnt/x/side_navlori/tag_crop.png", tag_roi)
