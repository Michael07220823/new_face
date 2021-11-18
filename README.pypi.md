# new_face

<p>
    new_face repository includes face detection, face landmark, face alignment, and face recognition technique.
<p><br>

## Necessary softwares
1. [cmake](https://cmake.org/download/)
2. [graphviz](https://graphviz.org/download/)

<br>

## Installation
    pip install -r requirements

or

    pip install new_face

or

    conda env create -f conf/new_face36.yaml -n new_face36
    conda env create -f conf/new_face37.yaml -n new_face37
    conda env create -f conf/new_face38.yaml -n new_face38
    conda env create -f conf/new_face39.yaml -n new_face39
<br>

## Methods List
Face Detection  | Face Landmark  | Face Alignment  | Face Recognition
:--------------:|:--------------:|:---------------:|:----------------:
 haar_detect    | dlib_5_points  | mtcnn_alignment |       LBPH
 dlib_detect    | dlib_68_points | dlib_alignment  |     OpenFace
 ssd_dnn_detect |       ×        |        ×        |      LBPCNN
 mtcnn_detect   |       ×        |        ×        |         ×
<br>

## Face Detection
    import logging
    import cv2
    import imutils
    from new_face import FaceDetection

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
<br>


## Face Landmark
    import logging
    import cv2
    import imutils
    from new_face import FaceLandmark

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
<br>


## Face Alignment
    import logging
    import cv2
    import imutils
    from new_face import FaceAlignment

    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


    image = cv2.imread("images/people-2.jpg")
    resize_image = imutils.resize(image, width=1280)

    face_alignment = FaceAlignment()
    mtcnn_detector = face_alignment.load_detector(face_alignment.MTCNN)

    rois, raw_image, face_images = face_alignment.mtcnn_alignment(mtcnn_detector,
                                                                  resize_image,
                                                                  conf_threshold=0.9,
                                                                  vision=True,
                                                                  save_dir="images/align",
                                                                  face_size=256)
<br>


## Face Recognition
### Dataset Structure
<p>
&emsp;├─dataset<br>
&emsp;│  └─YaleB_align_256<br>
&emsp;│  &emsp;├─yaleB11<br>
&emsp;│  &emsp;├─yaleB12<br>
&emsp;│  &emsp;├─yaleB13<br>
&emsp;│  &emsp;├─yaleB15<br>
&emsp;&emsp;&emsp;&emsp;&emsp;.<br>
&emsp;&emsp;&emsp;&emsp;&emsp;.<br>
&emsp;&emsp;&emsp;&emsp;&emsp;.<br>
</p>

### Train and Predict Model
#### Train **LBPH** model
    python train_lbph.py
<br>

#### Train **OpenFace** model
    python train_openface.py
<br>

#### Train **LBPCNN** model
    python train_lbpcnn.py
<br>

#### Predict by **LBPH** model
    python predict_lbph.py
<br>

#### Predict by **OpenFace** model
    python predict_openface.py
<br>

#### Predict by **LBPCNN** model
    python predict_lbpcnn.py

---

## **Reference**
* [SHEN, YUEH-CHUN, "LBPCNN Face Recognition Algorithm Implemented on the Raspberry Pi Access Control Monitoring System", 2021](https://hdl.handle.net/11296/hytkck)