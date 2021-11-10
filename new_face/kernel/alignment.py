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
from new_tools import check_image
from ..tools.config import root_dir
from ..tools.download import download_models


class FaceAlignment(object):
    """
    FaceAlignment class can use two kind method to alignment face. Two method: MTCNN and Dlib.
    """
    MTCNN = 0
    DLIB = 1


    def __compute_center_point(self, left_eye=tuple(), right_eye=tuple()):
        """
        Compute center point of two eyes.

        Args
        ----
        left_eye: x and y axis of left eye.

        right_eye: x and y axis of right eye.

        Return
        -------
        center_point: Center point axis of two eyes.
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

            model_name = "shape_predictor_68_face_landmarks.dat"
            shape_landmark_file_path = os.path.join(root_dir, model_name)
            if not os.path.exists(shape_landmark_file_path):
                download_models(model_name, root_dir)

            face_size = 256

            detector = dlib.get_frontal_face_detector()

            if not os.path.exists(shape_landmark_file_path): 
                logging.error("{} path error !".format(shape_landmark_file_path), exc_info=True)
                raise FileNotFoundError
            shape_predictor = dlib.shape_predictor(shape_landmark_file_path)

            face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=face_size)

            return detector, face_aligner
        else:
            logging.warning("Not match any method !")


    def mtcnn_alignment(self,
                        detector,
                        image,
                        conf_threshold=0.75,
                        vision=False,
                        vision_millisecs=100,
                        save_dir=None,
                        face_size=256):
        """
        mtcnn_alignment method is used to alignment face by multi-task cascaded convolutional networks method.
        
        ※Notice: MTCNN need to RGB image, if you use cv2.imread() to read image, you need swap R and B channel.

        Args:
        -----
        detector: Input MTCNN instance.
        
        image: Image can input image array or image path.

        conf_threshold: conf_threshold value is used to judge the face detection true or false.

        vision: Show face alignment image.

        vision_millisecs: Show image seconds. 
        
        save_dir: Saving path of face alignment images.

        face_size: Saved face size.


        Return:
        --------
        rois:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.

        raw_image: Raw image.

        face_images: Image of face alignment.
        """
        
        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status !=0:
            return rois, raw_image, face_images

        logging.debug("alignment.FaceAlignment.mtcnn.alignment.image shape: {}".format(image.shape))
        raw_image = image.copy()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logging.debug("Detecting face...")
        result = detector.detect_faces(rgb_image)
        logging.debug("mtcnn_alignment.result: {}".format(result))

        logging.debug("Aligning face...")
        if len(result) > 0:
            for num, people in enumerate(result, start=1):
                if people["confidence"] >= conf_threshold:
                    face_point = people["keypoints"]
                    
                    (img_h, img_w) = rgb_image.shape[:2]

                    lefteye = face_point["left_eye"]
                    righteye = face_point["right_eye"]

                    rotation_point = self.__compute_center_point(lefteye, righteye)

                    rotation_degree = self.__compute_degree(lefteye, righteye)

                    M = cv2.getRotationMatrix2D(rotation_point, rotation_degree, scale=1.0)
                    rgb_rotated = cv2.warpAffine(rgb_image, M, (img_w, img_h), flags=cv2.INTER_CUBIC)
                    
                    x, y, w, h = roi = people['box']
                    rois.append(roi)

                    nx, ny, nw, nh = self.__area_expand(roi)
                    align_face_image = cv2.cvtColor(rgb_rotated[ny:ny+nh, nx:nx+nw], cv2.COLOR_RGB2BGR)

                    align_face_image = cv2.resize(align_face_image, (face_size, face_size))
                    face_images.append(align_face_image)

                    if vision:
                        cv2.imshow("MTCNN Raw Image...", imutils.resize(raw_image, width=640))
                        raw_face_image = image[y:y+h, x:x+w]
                        cv2.imshow("Raw Face Image...", imutils.resize(raw_face_image, width=250))
                        cv2.imshow("MTCNN Align faces...", imutils.resize(align_face_image, width=250))
                        cv2.waitKey(vision_millisecs)

                    if save_dir != None:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                            logging.info("Builed {} directory successfully !") if os.path.exists(save_dir) else logging.warning("Builed {} directory failed !")

                        image_path = os.path.join(save_dir, "{}.jpg".format(str(num).zfill(6)))
                        cv2.imwrite(image_path, align_face_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        
                        if os.path.exists(image_path):
                            logging.info("Saved '{}' successfully !".format(image_path))
                        else:
                            logging.warning("Saved '{}' failed !".format(image_path))
            cv2.destroyAllWindows()
        else:
            logging.debug("MTCNN doesn't detect the face !")

        return (rois, raw_image, face_images)


    def dlib_alignment(self,
                       detector,
                       face_aligner,
                       image,
                       vision=False,
                       vision_millisecs=100,
                       save_dir=None,
                       face_size=256):
        """
        dlib_alignment method is used to alignment face by 5 or 68 point landmark method.
        It can use shape 5 or 68 point landmark file.

        Args:
        -----
        detector: Input dlib face detector instance.

        face_aligner: FaceAligner instance of imutils.

        image: Image array or image path.

        vision: Show face alignment image.

        vision_millisecs: Show image seconds.
        
        save_dir: Saving path of face alignment image.

        face_size: Saved face size.


        Return:
        --------
        rois:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.

        raw_image: Raw image.

        face_image: Image of face alignment.
        """
        
        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status !=0:
            return rois, raw_image, face_images

        raw_image = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        logging.debug("Detecting face...")
        rects = detector(gray, 2)
        logging.debug("alignment.dlib_alignment.rects: {}".format(rects))

        logging.debug("Aligning face image...")
        if len(rects) > 0:
            for rect in rects:
                roi = self.__dlib_rect_to_roi(rect)
                face_image = face_aligner.align(image, gray, rect)

                rois.append(roi)
                face_images.append(face_image)
            
            for num, face in enumerate(face_images, start=1):
                if vision:
                    cv2.imshow("Dlib Raw Image...", imutils.resize(raw_image, width=640))
                    cv2.imshow("Dlib Align faces...", imutils.resize(face, width=250))
                    cv2.waitKey(vision_millisecs)
                
                if save_dir != None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        logging.info("Builed {} dircetory successfully !".format(save_dir)) if os.path.exists(save_dir) else logging.warning("Builed {} directory failed !".format(save_dir))

                    image_path = os.path.join(save_dir, "{}.jpg".format(str(num).zfill(6)))
                    cv2.imwrite(image_path, cv2.resize(face, (face_size, face_size)), [cv2.IMWRITE_JPEG_QUALITY, 100])                    
                    logging.info("Saved image to '{}' successfully !".format(image_path)) if os.path.exists(image_path) else logging.warning("Saved image to '{}' failed !".format(image_path))
            cv2.destroyAllWindows()
        else:
            logging.warning("Dlib doesn't detect the face !")

        return (rois, raw_image, face_images)