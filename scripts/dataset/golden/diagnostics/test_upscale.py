import cv2
from pupil_apriltags import Detector

img = cv2.imread("/mnt/x/side_navlori/test_frame_4.png", cv2.IMREAD_GRAYSCALE)
img_up = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

detector = Detector(families='tag36h11', debug=0, quad_decimate=1.0)
tags = detector.detect(img)
print(f"Original size (pupil): found {len(tags)}")

tags_up = detector.detect(img_up)
print(f"Upscaled 2x (pupil): found {len(tags_up)}")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
parameters.markerBorderBits = 2
detector_cv = cv2.aruco.ArucoDetector(dictionary, parameters)

c, i, r = detector_cv.detectMarkers(img)
print(f"Original size (opencv): found {len(c)}")

c_up, i_up, r_up = detector_cv.detectMarkers(img_up)
print(f"Upscaled 2x (opencv): found {len(c_up)}")

# try histogram equalization
img_eq = cv2.equalizeHist(img)
img_up_eq = cv2.equalizeHist(img_up)
tags_eq = detector.detect(img_eq)
print(f"Equalized Original size (pupil): found {len(tags_eq)}")
tags_up_eq = detector.detect(img_up_eq)
print(f"Equalized Upscaled 2x (pupil): found {len(tags_up_eq)}")
