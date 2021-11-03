# new_face
This repository can use many face recognition technology, like face detection, face landmark, face alignment, and face recognition.
---

---
<br>

## Install requirement packages.
> `pip install -r requirements`
---
<br>

## Face Detection
```
import cv2
import logging
from new_face import FaceDetection

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

run = True
vc = cv2.VideoCapture(0)
face_detect = FaceDetection()
haae_detector = face_detect.load_detector(face_detect.HAAR)

while run and vc.isOpened():
    _, frame = vc.read()
    rois, raw_image, face_images = face_detect.haar_detect(haae_detector,
                                                           frame,
                                                           vision=False,
                                                           save_path="haar.jpg")
    for x,y,w,h in rois:
        cv2.rectangle(raw_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow("Face", raw_image)
    key = cv2.waitKey(1)

    if key == ord('q'):
        run = False
        vc.release()
        cv2.destroyAllWindows()
```
---
<br>

## Face Landmark
```
import logging
import cv2
from new_face import FaceLandmark

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


vc = cv2.VideoCapture(0)
# shape_5_predictor = FaceLandmark.load_shape_predictor()
shape_68_predictor = FaceLandmark.load_shape_predictor("shape_predictor_68_face_landmarks.dat")

while vc.isOpened():
    _, image = vc.read()

    # face_points = FaceLandmark.dlib_5_points(image=image,
    #                                          shape_predictor=shape_predictor)

    face_points = FaceLandmark.dlib_68_points(image=image,
                                              shape_predictor=shape_predictor,
                                              get_five_points=False)

    if len(face_points) > 0:
        for point in face_points.keys():
            cv2.circle(image, face_points[point], 1, (0, 255, 0), 2)

    cv2.imshow("Face Landmark", image)
    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        vc.release()
        break
```
---
<br>

## Face Alignment
```
import logging
import cv2
from new_face import FaceAlignment

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


vc = cv2.VideoCapture(0)
face_alignment = FaceAlignment()
# detector, face_aligner = face_alignment.load_detector(face_alignment.DLIB)
mtcnn_detector = face_alignment.load_detector(face_alignment.MTCNN)

if vc.isOpened():
    _, frame = vc.read()

    # rois, raw_image, align_images = face_alignment.dlib_alignment(detector,
    #                                                               face_aligner,
    #                                                               frame,
    #                                                               vision=True,
    #                                                               save_dir="D:\\GitHub\\new_face",
    #                                                               face_size=256)

    rois, raw_image, face_images = face_alignment.mtcnn_alignment(mtcnn_detector,
                                                                  frame,
                                                                  conf_threshold=0.9,
                                                                  vision=True,
                                                                  save_dir="D:\\GitHub\\new_face",
                                                                  face_size=256)
    vc.release()
```
---
<br>

### Reference
* [SHEN, YUEH-CHUN, "LBPCNN Face Recognition Algorithm Implemented on the Raspberry Pi Access Control Monitoring System", 2021](https://hdl.handle.net/11296/hytkck)