import os
import logging
import cv2
from new_face import FaceDetection


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)

image_path = "images/people.jpg"
image = cv2.imread(image_path)
image_height, image_width, image_channel = image.shape
logging.debug("Height:{}, Width: {}".format(image_height, image_width))

face_detect = FaceDetection()
mtcnn = face_detect.load_detector(face_detect.MTCNN)

rois, raw_image, face_images = face_detect.mtcnn_detect(mtcnn,
                                                        image,
                                                        conf_threshold=0.5,
                                                        vision=False,
                                                        save_path="images/mtcnn.jpg")

new_image_name = "{}.txt".format(os.path.basename(os.path.splitext(image_path)[0]))
logging.debug("new_image_name: {}".format(new_image_name))

with open(new_image_name, "w") as f:
    for x, y, w, h in rois:
        center_x = (x + (w / 2)) / image_width
        center_y = (y + (h / 2)) / image_height
        object_width = w / image_width
        object_height = w / image_height
    
        f.write("0 {} {} {} {}\n".format(center_x, center_y, object_width, object_height))
    logging.info("Writed {} face objects to {} successfully !".format(len(rois), new_image_name))