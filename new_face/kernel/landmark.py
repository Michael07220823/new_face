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
import logging
import dlib
from new_tools import check_image
from .config import root_dir
from .download import download_models

class FaceLandmark(object):
    """
    FaceLandmark class use two methods to landmark face.
    """

    @classmethod
    def dlib_5_points(cls,
                      image=None):
        """
        dlib_5_points method is use face five points of dlib to mark left eye, right eye, nose of face.

        Args:
        -----
        image: Input image path or image array.


        Returns:
        --------
        five_points:
            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.
        """

        five_point = dict()

        # Read image.
        status, raw_image = check_image(image)
        if status != 0:
            return five_point

        detector = dlib.get_frontal_face_detector()

        # Download model.
        model_name = "shape_predictor_5_face_landmarks.dat"
        predictor_path = os.path.join(root_dir, model_name)
        if not os.path.exists(predictor_path):
            download_models(model_name, save_path=root_dir)

        predictor = dlib.shape_predictor(predictor_path)

        # Detect face and get roi.
        detect_faces = detector(raw_image, 0)
        
        if len(detect_faces) > 0:
            for roi in detect_faces:
                shape_face = predictor(raw_image, roi)

                # Get five points from face.
                lefteye_leftcorner, lefteye_rightcorner, righteye_rightcorner, righteye_leftcorner, nose = shape_face.parts()
                five_point["lefteye_leftcorner"] = (lefteye_leftcorner.x, lefteye_leftcorner.y)
                five_point["lefteye_rightcorner"] = (lefteye_rightcorner.x, lefteye_rightcorner.y)
                five_point["righteye_rightcorner"] = (righteye_rightcorner.x, righteye_rightcorner.y)
                five_point["righteye_leftcorner"] = (righteye_leftcorner.x, righteye_leftcorner.y)
                five_point["nose"] = (nose.x, nose.y)

                return five_point
        else:
            logging.info("Dlib doesn't detect the face !")


    @classmethod
    def dlib_68_points(cls,
                       image=None,
                       get_five_points=False):
        """
        dlib_68_points method is use face sixty-eight points of dlib to mark sixty-eight of face.
        
        Args:
        -----
        image: Input image path or image array.

        get_five_points: Control only get five points of face from sixty-eight points of face.


        Returns:
        --------
        five_points: dict()
            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.

        sixty_points: Sixty_eight points of face.

        0: No detect the face.
        """

        sixty_eight_points = dict()

        # Read image.
        status, raw_image = check_image(image)
        if status != 0:
            return sixty_eight_points
            
        detector = dlib.get_frontal_face_detector()
        
        # Download model.
        model_name = "shape_predictor_68_face_landmarks.dat"
        predictor_path = os.path.join(root_dir, model_name)
        if not os.path.exists(predictor_path):
            download_models(model_name, save_path=root_dir)

        predictor = dlib.shape_predictor(predictor_path)

        # Detect face and get roi.
        detect_faces = detector(raw_image, 0)

        if len(detect_faces) > 0:
            for roi in detect_faces:
                shape_face = predictor(raw_image, roi)

                if get_five_points:
                    five_points = dict()
                    five_points["lefteye_leftcorner"] = (shape_face.part(36).x, shape_face.part(36).y)
                    five_points["lefteye_rightcorner"] = (shape_face.part(39).x, shape_face.part(39).y)
                    five_points["righteye_rightcorner"] = (shape_face.part(45).x, shape_face.part(45).y)
                    five_points["righteye_leftcorner"] = (shape_face.part(42).x, shape_face.part(42).y)
                    five_points["nose"] = (shape_face.part(30).x, shape_face.part(30).y)
                    return five_points
                    
                for num in range(0, 68):
                    sixty_eight_points[num] = (shape_face.part(num).x, shape_face.part(num).y)

                return sixty_eight_points
        else:
            logging.info("Dlib doesn't detect the face !")
            
    
    @classmethod
    def __calc_center_point(cls, x1=int(), y1=int(), x2=int(), y2=int()):
        """
        __calc_center_point method is used to calculate center coordinate of two point.

        Args:
        -----
        x1: x coordinate of x1.

        y1: y coordinate of x1.

        x2: x coordinate of x2.

        y2: y coordinate of x2.

        Returns:
        --------
        (x, y): Center coordinate.
        """

        (x, y) = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return (x, y)

    
    @classmethod
    def fivepoints_to_threepoints(cls, five_point=dict()):
        """
        fivepoints_to_threepoints method used to transfer 5 points to 3 points.

        Args:
        -----
        5 points: 
            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.


        Return:
        -------
        three_points:
            left_eye: left eye center coordinate.
            right_eye: Right eye center coordinate.
            nose: Nose coordinate.
        """
        
        three_points = dict()
        if len(five_point) < 5:
            logging.error("five_point variable element small than 5 !")
            raise ValueError
        lefteye_leftcorner_x1, lefteye_leftcorner_y1 = five_point["lefteye_leftcorner"]
        lefteye_rightcorner_x2, lefteye_rightcorner_y2 = five_point["lefteye_rightcorner"]

        three_points["left_eye"] = cls.__calc_center_point(lefteye_leftcorner_x1, 
                                                          lefteye_leftcorner_y1, 
                                                          lefteye_rightcorner_x2, 
                                                          lefteye_rightcorner_y2)

        righteye_leftcorner_x1, righteye_leftcorner_y1 = five_point["righteye_leftcorner"]
        righteye_rightcorner_x2, righteye_rightcorner_y2 = five_point["righteye_rightcorner"]

        three_points["right_eye"] = cls.__calc_center_point(righteye_leftcorner_x1,
                                                           righteye_leftcorner_y1, 
                                                           righteye_rightcorner_x2,
                                                           righteye_rightcorner_y2)
        three_points["nose"] = five_point["nose"]

        return three_points