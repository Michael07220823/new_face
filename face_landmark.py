import logging
import cv2
import imutils
from new_face import FaceLandmark

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


image = cv2.imread("images/people-3.jpg")
resize_image = imutils.resize(image, width=1280)

shape_5_predictor = FaceLandmark.load_shape_predictor("shape_predictor_5_face_landmarks.dat")
# shape_68_predictor = FaceLandmark.load_shape_predictor("shape_predictor_68_face_landmarks.dat")

face_points = FaceLandmark.dlib_5_points(image=resize_image,
                                          shape_predictor=shape_5_predictor,
                                          vision=True,
                                          save_path="images/dlib_5_points.jpg")

# face_points = FaceLandmark.dlib_68_points(image=resize_image,
#                                           shape_predictor=shape_68_predictor,
#                                           vision=True,
#                                           save_path="images/dlib_68_points.jpg")