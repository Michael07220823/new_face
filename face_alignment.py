import logging
import cv2
import imutils
from new_face import FaceAlignment

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


image = cv2.imread("images/people-2.jpg")
resize_image = imutils.resize(image, width=1280)

face_alignment = FaceAlignment()
detector, face_aligner = face_alignment.load_detector(face_alignment.DLIB)
# mtcnn_detector = face_alignment.load_detector(face_alignment.MTCNN)

rois, raw_image, align_images = face_alignment.dlib_alignment(detector,
                                                              face_aligner,
                                                              resize_image,
                                                              vision=True,
                                                              save_dir="images/align",
                                                              face_size=256)

rois, raw_image, face_images = face_alignment.mtcnn_alignment(mtcnn_detector,
                                                              resize_image,
                                                              conf_threshold=0.9,
                                                              vision=True,
                                                              save_dir="images/align",
                                                              face_size=256)