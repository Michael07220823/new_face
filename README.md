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

<img src="images/mtcnn.jpg" alt="images/mtcnn.jpg" width="480"></img>
<img src="images/mtcnn-2.jpg" alt="images/mtcnn-2.jpg" width="480"></img>

[source 1](https://cdn.downtoearth.org.in/library/large/2019-11-15/0.98645800_1573812770_gettyimages-986704314.jpg)
[source 2](https://www.goodshepherdcentres.ca/wp-content/uploads/2020/06/home-page-banner.jpg)
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

<img src="images/dlib_5_points.jpg" alt="images/dlib_5_points.jpg" width="640"></img>
<img src="images/dlib_68_points.jpg" alt="images/dlib_68_points.jpg" width="640"></img>

[source](https://static01.nyt.com/images/2020/11/19/us/artificial-intelligence-fake-people-faces-promo-1605818328743/artificial-intelligence-fake-people-faces-promo-1605818328743-jumbo-v2.jpg?quality=75&auto=webp)
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

<img src="images/people-2.jpg" alt="images/people-2.jpg" width="640"></img>

<img src="images/align/000001.jpg" alt="images/align/000001.jpg" width="64"></img>
<img src="images/align/000002.jpg" alt="images/align/000002.jpg" width="64"></img>
<img src="images/align/000003.jpg" alt="images/align/000003.jpg" width="64"></img>
<img src="images/align/000004.jpg" alt="images/align/000004.jpg" width="64"></img>
<img src="images/align/000005.jpg" alt="images/align/000005.jpg" width="64"></img>
<img src="images/align/000006.jpg" alt="images/align/000006.jpg" width="64"></img>
<img src="images/align/000007.jpg" alt="images/align/000007.jpg" width="64"></img>
<img src="images/align/000008.jpg" alt="images/align/000008.jpg" width="64"></img>
<img src="images/align/000009.jpg" alt="images/align/000009.jpg" width="64"></img>
<img src="images/align/000010.jpg" alt="images/align/000010.jpg" width="64"></img>
<img src="images/align/000011.jpg" alt="images/align/000011.jpg" width="64"></img>
<img src="images/align/000012.jpg" alt="images/align/000012.jpg" width="64"></img>
<img src="images/align/000013.jpg" alt="images/align/000013.jpg" width="64"></img>
<img src="images/align/000014.jpg" alt="images/align/000014.jpg" width="64"></img>
<img src="images/align/000015.jpg" alt="images/align/000015.jpg" width="64"></img>
<img src="images/align/000016.jpg" alt="images/align/000016.jpg" width="64"></img>
<img src="images/align/000017.jpg" alt="images/align/000017.jpg" width="64"></img>
<img src="images/align/000018.jpg" alt="images/align/000018.jpg" width="64"></img>

[source](https://www.goodshepherdcentres.ca/wp-content/uploads/2020/06/home-page-banner.jpg)
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
<br>

## Real Time Predict
<img src="images/face_recognition.gif" alt="images/align/000018.jpg" width="640"></img>

---

## **Reference**
* [SHEN, YUEH-CHUN, "LBPCNN Face Recognition Algorithm Implemented on the Raspberry Pi Access Control Monitoring System", 2021](https://hdl.handle.net/11296/hytkck)