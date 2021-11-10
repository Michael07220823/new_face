# new_face

<p>
    new_face repository includes face detection, face landmark, face alignment, and face recognition technique.
<p><br>

## Installation
    git clone https://github.com/Michael07220823/new_face.git
    cd new_face/
    pip install -r requirements

or

    pip install new_face
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

<img src="images\people-2.jpg" alt="images\people-2.jpg" width="640"></img>

<img src="images\align\000001.jpg" alt="images\align\000001.jpg" width="128"></img>
<img src="images\align\000002.jpg" alt="images\align\000002.jpg" width="128"></img>
<img src="images\align\000003.jpg" alt="images\align\000003.jpg" width="128"></img>
<img src="images\align\000004.jpg" alt="images\align\000004.jpg" width="128"></img>
<img src="images\align\000005.jpg" alt="images\align\000005.jpg" width="128"></img>
<img src="images\align\000006.jpg" alt="images\align\000006.jpg" width="128"></img>
<img src="images\align\000007.jpg" alt="images\align\000007.jpg" width="128"></img>
<img src="images\align\000008.jpg" alt="images\align\000008.jpg" width="128"></img>
<img src="images\align\000009.jpg" alt="images\align\000009.jpg" width="128"></img>
<img src="images\align\000010.jpg" alt="images\align\000010.jpg" width="128"></img>
<img src="images\align\000011.jpg" alt="images\align\000011.jpg" width="128"></img>
<img src="images\align\000012.jpg" alt="images\align\000012.jpg" width="128"></img>
<img src="images\align\000013.jpg" alt="images\align\000013.jpg" width="128"></img>
<img src="images\align\000014.jpg" alt="images\align\000014.jpg" width="128"></img>
<img src="images\align\000015.jpg" alt="images\align\000015.jpg" width="128"></img>
<img src="images\align\000016.jpg" alt="images\align\000016.jpg" width="128"></img>
<img src="images\align\000017.jpg" alt="images\align\000017.jpg" width="128"></img>
<img src="images\align\000018.jpg" alt="images\align\000018.jpg" width="128"></img>

[source](https://www.goodshepherdcentres.ca/wp-content/uploads/2020/06/home-page-banner.jpg)

<br>


## Reference
* [SHEN, YUEH-CHUN, "LBPCNN Face Recognition Algorithm Implemented on the Raspberry Pi Access Control Monitoring System", 2021](https://hdl.handle.net/11296/hytkck)