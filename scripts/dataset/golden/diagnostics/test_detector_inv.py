import cv2
from pupil_apriltags import Detector

img = cv2.imread("/mnt/x/side_navlori/test_frame_4.png", cv2.IMREAD_GRAYSCALE)
img_inv = 255 - img

detector = Detector(families='tag36h11', debug=0)
tags = detector.detect(img_inv)
print(f"Inverted (pupil): found {len(tags)}")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
parameters.markerBorderBits = 2
detector_cv = cv2.aruco.ArucoDetector(dictionary, parameters)

c, i, r = detector_cv.detectMarkers(img_inv)
print(f"Inverted (opencv): found {len(c)}")

# Try standard Aruco DICT_6X6_1000
dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
detector_aruco = cv2.aruco.ArucoDetector(dict_aruco, parameters)
c_a, i_a, r_a = detector_aruco.detectMarkers(img)
print(f"Aruco 6x6 (opencv): found {len(c_a)}")
