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


## Reference
* [SHEN, YUEH-CHUN, "LBPCNN Face Recognition Algorithm Implemented on the Raspberry Pi Access Control Monitoring System", 2021](https://hdl.handle.net/11296/hytkck)