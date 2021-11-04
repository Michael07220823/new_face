import logging
import cv2
import imutils
from new_face import FaceDetection

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


image = cv2.imread("images/people.jpg")
resize_image = imutils.resize(image, width=1280)

face_detect = FaceDetection()
mtcnn = face_detect.load_detector(face_detect.MTCNN)

rois, raw_image, face_images = face_detect.mtcnn_detect(mtcnn,
                                                        resize_image,
                                                        conf_threshold=0.5,
                                                        vision=True,
                                                        save_path="images/mtcnn.jpg")