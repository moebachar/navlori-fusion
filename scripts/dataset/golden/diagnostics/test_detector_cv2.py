import cv2

img = cv2.imread("/mnt/x/side_navlori/test_frame.png", cv2.IMREAD_GRAYSCALE)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
parameters.markerBorderBits = 2
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

corners, ids, rejected = detector.detectMarkers(img)
print(f"OpenCV with borderBits=2 found {len(corners)} tags.")

parameters.markerBorderBits = 1
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejected = detector.detectMarkers(img)
print(f"OpenCV with borderBits=1 found {len(corners)} tags.")
