"""
MIT License

Copyright (c) 2021 Overcomer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import math
import logging
import cv2
import dlib
from mtcnn import MTCNN
import imutils
from imutils.face_utils import FaceAligner
from new_timer import AutoTimer
from new_tools import check_image


class FaceAlignment(object):
    """
    FaceAlignment class can use two kind method to alignment face. Two method: MTCNN and Dlib.
    """
    MTCNN = 0
    DLIB = 1
    
    root_dir = os.path.join(os.getenv("HOME"), ".new_face")


    def __compute_center_point(self, left_eye=tuple(), right_eye=tuple()):
        """
        Compute center point of two eyes.

        Args
        ----
        left_eye: x and y axis of left eye.

        right_eye: x and y axis of right eye.

        Return
        -------
        center_point: center point axis of two eyes.
        """
        
        x = int((left_eye[0] + right_eye[0]) / 2)
        y = int((left_eye[1] + right_eye[1]) / 2)
        center_point = (x, y)

        return center_point


    def __compute_degree(self, left_eye=tuple(), right_eye=tuple()):
        """
        __compute_degree method is used to compute slope of two eyes, and transform unit from radian to degree.

        Args:
        -----
        left_eye: Left eye coordinate point x and y.

        right_eye: Right eye coordinate point x and y.


        Return:
        --------
        rotation_degree: Rotational degree.
        """

        if len(left_eye) != 2 or len(right_eye) != 2:
            logging.debug("__compute_degree.left_eye: {}".format(left_eye))
            logging.debug("__compute_degree.right_eye: {}".format(right_eye))
            logging.error("left eye or right eye coordinate less than two !", exc_info=True)
            raise ValueError
        lefteye_x, lefteye_y = left_eye
        righteye_x, righteye_y = right_eye

        # Radian to degree. formula: degree = radian × (180/π) .
        rotation_degree = math.degrees(math.atan2(righteye_y - lefteye_y, righteye_x - lefteye_x))
        logging.debug("FaceAlignment.__compute_degree.rotation_degree: {:.2f}°".format(rotation_degree))

        return rotation_degree


    def __area_expand(self, roi=tuple(), width_scale_factor=0.1):
        """
        __area_expand method is used to expand face area.

        Args:
        -----
        roi: (x, y, w, h)
            x: Face left-top corner x coordinate point.
            y: Face left-top corner y coordinate point.
            w: Width.
            h: Height.


        Return:
        --------
        (nx, ny, nw, nh):
            nx: New face left-top corner x coordinate point.
            ny: New face left-top corner y coordinate point.
            nw: New width.
            nh: New height.
        """

        if len(roi) != 4: 
            logging.debug("FaceAlignment.__area_expand.roi: {}".format(roi))
            logging.error("roi values less than four.", exc_info=True)
            raise ValueError
        
        x, y, w, h = roi
        width_scale_factor = 0.1
        nx = int(x - (width_scale_factor * w))
        ny = y
        nw = int((1 + width_scale_factor * 2) * (w))
        nh = h

        if nx < 0:
            nx = 0 
        if ny < 0:
            ny = 0
        
        logging.debug("FaceAlignment.__area_expand.roi: {}".format(roi))
        logging.debug("FaceAlignment.__area_expand new roi: {}".format((nx, ny, nw, nh)))
        return (nx, ny, nw, nh)


    def __dlib_rect_to_roi(self, rect):
            """
            __dlib_rect_to_roi method is used to get roi values from dlib.rectangle class.

            Args:
            -----
                rect: Face roi


            Return:
            -------
            roi:
            
                (x, y, w, h)
                    x: Face left-top corner x coordinate point.
                    y: Face left-top corner y coordinate point.
                    w: Width.
                    h: Height.                     
            """

            roi = (rect.left(), rect.top(), rect.right(), rect.bottom())
            logging.debug("__dlib_rect_to_roi.roi: {}".format(roi))

            return roi

    
    def load_detector(self, method_ID=int()):
        """
        load_detector method is used to load all method detector for reducing loading time.

        Args:
        -----
        method_ID:

            FaceAlignment.MTCNN: Load mtcnn detector.
            FaceAlignment.DLIB:  Load dlib face detector、shape predictor and face aligner of imutils.
        

        Return:
        -------
        detector: Face detector.

        shape_predictor: Dlib shape_predictor, only return when use FaceAlignment.DLIB.

        face_aligner: imutils face aligner, only return when use FaceAlignment.DLIB.
        """
        
        if type(method_ID) != int: 
            logging.debug("load_detector.method_ID: {}".format(method_ID))
            logging.error("method_ID type isn't int.", exc_info=True)
            raise TypeError
        elif method_ID == FaceAlignment.MTCNN:
            logging.info("Loading mtcnn detector...")
            detector = MTCNN()
            return detector
        elif method_ID == FaceAlignment.DLIB:
            logging.info("Loading dlib detector...")
            shape_landmark_file_path = "face/models/landmark/shape_predictor_5_face_landmarks.dat"

            # logging.info("Please input face size:")
            # face_size = int(input())
            face_size = 256

            # Face detector.
            detector = dlib.get_frontal_face_detector()

            # Face lanmarker.
            if not os.path.exists(shape_landmark_file_path): 
                logging.error("{} path error !".format(shape_landmark_file_path), exc_info=True)
                raise FileNotFoundError
            shape_predictor = dlib.shape_predictor(shape_landmark_file_path)

            # Face alignmenter.
            face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=face_size)

            return detector, face_aligner
        else:
            logging.warning("Not match any method !")


    def mtcnn_alignment(self,
                        detector,
                        image,
                        conf_threshold=0.75,
                        align=True,
                        vision=False,
                        vision_millisecs=0,
                        save_dir=None):
        """
        mtcnn_alignment method is used to alignment face by mtcnn method.
        
        ※Notice: MTCNN need to RGB image, if you use cv2.imread() to read image, you need swap R and B channel.

        Args:
        -----

        detector: Input MTCNN instance.
        
        image: Image can input image array or image path.

        conf_threshold: conf_threshold value is used to judge the face detection true or false.

        align: Align face or not.

        vision: Show face alignment image.

        vision_millisecs: Show image seconds. 
        
        save_dir: Saving path of face alignment images.


        Return:
        --------
        rois:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.

        raw_image: Original image.

        face_images: Image of face alignment.
        """
        
        # Init variable, don't delete.
        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status !=0:
            return rois, raw_image, face_images

        logging.debug("alignment.FaceAlignment.mtcnn.alignment.image shape: {}".format(image.shape))
        raw_image = image.copy()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face and get face five point.
        logging.debug("Detecting face...")
        result = detector.detect_faces(rgb_image)
        logging.debug("mtcnn_alignment.result: {}".format(result))

        # Build directory of saved.
        if save_dir != None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # Face Detection.
        if align == False:
            if len(result) > 0:
                for num, people in enumerate(result, start=1):
                    if people["confidence"] >= conf_threshold:
                        rois.append(people["box"])
                        x, y, w, h = people["box"]

                        face_image = raw_image[y:y+h, x:x+w]
                        face_images.append(face_image)
                        
                        # Show image.
                        if vision:
                            cv2.imshow("Raw Image", imutils.resize(raw_image, width=640))
                            cv2.imshow("Face Image", imutils.resize(face_image, width=250))
                            cv2.waitKey(vision_millisecs)
                            cv2.destroyAllWindows()

                        # Save image.
                        if save_dir != None:
                            image_path = os.path.join(save_dir, "{}.jpg".format(num).zfill(4))
                            cv2.imwrite(image_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            
                            if os.path.exists(image_path):
                                logging.info("Saved '{}' successfully !".format(image_path))
                            else:
                                logging.warning("Saved '{}' failed !".format(image_path))
            return (rois, raw_image, face_images)

        # Face Alignment.
        if len(result) > 0:
            for num, people in enumerate(result, start=1):
                if people["confidence"] >= conf_threshold:
                    # Face five point: left_eye, right_eye, nose, left_mouth, right_mouth.
                    face_point = people["keypoints"]
                    
                    # Get image height and width.
                    (img_h, img_w) = rgb_image.shape[:2]

                    # Face point.
                    lefteye = face_point["left_eye"]
                    righteye = face_point["right_eye"]

                    # Compute center coordinate.
                    rotation_point = self.__compute_center_point(lefteye, righteye)

                    # Compute rotate angle.
                    logging.debug("Aligning face...")
                    rotation_degree = self.__compute_degree(lefteye, righteye)

                    # Rotate image.
                    M = cv2.getRotationMatrix2D(rotation_point, rotation_degree, scale=1.0)
                    rgb_rotated = cv2.warpAffine(rgb_image, M, (img_w, img_h), flags=cv2.INTER_CUBIC)
                    
                    # ROI
                    x, y, w, h = roi = people['box']
                    rois.append(roi)

                    nx, ny, nw, nh = self.__area_expand(roi)
                    align_face_image = cv2.cvtColor(rgb_rotated[ny:ny+nh, nx:nx+nw], cv2.COLOR_RGB2BGR)
                    face_images.append(align_face_image)

                    # Show image.
                    if vision:
                        cv2.imshow("Raw Image", imutils.resize(raw_image, width=640))
                        orig_face_image = image[y:y+h, x:x+w]
                        cv2.imshow("Face Image", imutils.resize(orig_face_image, width=250))
                        cv2.imshow("MTCNN Face Alignment", imutils.resize(align_face_image, width=250))
                        cv2.waitKey(vision_millisecs)
                        cv2.destroyAllWindows()

                    if save_dir != None:
                        # Save image.
                        image_path = os.path.join(save_dir, "{}.jpg".format(num).zfill(4))
                        cv2.imwrite(image_path, align_face_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        
                        if os.path.exists(image_path):
                            logging.info("Saved '{}' successfully !".format(image_path))
                        else:
                            logging.warning("Saved '{}' failed !".format(image_path))
        else:
            logging.debug("MTCNN not detect the face !")

        return (rois, raw_image, face_images)


    def dlib_alignment(self,
                       detector,
                       face_aligner,
                       image,
                       vision=False,
                       vision_millisecs=0,
                       save_path=str()):
        """
        dlib_alignment method is used to alignment face by 5 or 68 point landmark method. It can use shape 5 or 68 point 
        landmark file.

        Args:
        -----
        detector: Input dlib face detector instance.

        face_aligner: Input face_aligner of imutils instance.

        image: Image can input image array or image path.

        vision: Show face alignment image.

        vision_millisecs: Show image seconds.
        
        save_path: Saving path of face alignment image.


        Return:
        --------
        roi:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.

        raw_image: Original image.

        face_image: Image of face alignment.
        """
        
        # Init variable, don't delete.
        roi = None
        raw_image = None
        face_image = None

        # Check image.
        status, image_arry = check_image(image)
        if status !=0:
            return roi, raw_image, face_image

        raw_image = image_arry.copy()

        gray = cv2.cvtColor(image_arry, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        logging.info("Detecting face...")
        rects = detector(gray, 2)
        logging.debug("dlib_alignment.rects: {}".format(rects))

        logging.info("Aligning face image...")
        if len(rects) > 0:
            for rect in rects:
                roi = self.__dlib_rect_to_roi(rect)
                face_image = face_aligner.align(image_arry, gray, rect)
            
            # Show image.
            if vision:
                cv2.imshow("Dlib face alignment", face_image)
                cv2.waitKey(vision_millisecs)
                cv2.destroyAllWindows()
                
            if save_path != str():
                cv2.imwrite(save_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                if os.path.exists(save_path):
                    logging.info("Saved image to '{}' successfully !".format(save_path))
                else:
                    logging.warning("Saved image to '{}' failed !".format(save_path))
        else:
            logging.warning("Dlib not detect the face !")

        return (roi, raw_image, face_image)


if __name__ == "__main__":
    # Set logging config.
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)
    

    vc = cv2.VideoCapture(0)
    face_alignment = FaceAlignment()
    mtcnn_detector = face_alignment.load_detector(face_alignment.MTCNN)

    # image = cv2.imread("face\\data\\americanse.jpg")
    # image = cv2.imread("face\\data\\china.jpg")
    # image = cv2.imread("face\\data\\occlusion.png")
    # image = cv2.imread("D:/Github/face/data/train/kevin/kevin00000001.jpg")

    # with AutoTimer("MTCNN Face Alignment"):
    #     rois, raw_image, align_images = face_alignment.mtcnn_alignment(mtcnn_detector,
    #                                                                    image,
    #                                                                    align=False,
    #                                                                    vision=True,
    #                                                                    save_dir=None)
    # logging.debug(len(rois))
    # logging.debug(len(align_images))
    while  vc.isOpened():
        try:
            _, frame = vc.read()

            with AutoTimer("MTCNN", 4):
                rois, raw_image, face_images = face_alignment.mtcnn_alignment(mtcnn_detector,
                                                                              frame,
                                                                              conf_threshold=0.9,
                                                                              align=True,
                                                                              vision=True)
        except KeyboardInterrupt:
            vc.release()