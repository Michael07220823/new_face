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
import cv2
import dlib
from new_tools import check_image
from .config import root_dir
from .download import download_models

class FaceLandmark(object):
    """
    FaceLandmark class use two methods to landmark face.
    """

    @classmethod
    def load_shape_predictor(cls,
                             model_name="shape_predictor_5_face_landmarks.dat"):
        """
        load_shape_predictor method is used to load dlib shape predictor.

        Args
        ----
        model_name: shape_predictor_5_face_landmarks.dat or shape_predictor_68_face_landmarks.dat file names.


        Return
        ------
        shape_predictor: shape_predictor instance.
        """

        shape_predictor_path = os.path.join(root_dir, model_name)

        if not os.path.exists(shape_predictor_path):
            download_models(model_name, save_path=root_dir)

        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        
        return shape_predictor


    @classmethod
    def dlib_5_points(cls,
                      image=None,
                      shape_predictor=object(),
                      vision=False,
                      save_path=None):
        """
        dlib_5_points method is use face five points of dlib to mark left eye, right eye, nose of face.

        Args:
        -----
        image: Input image path or image array.

        shape_predictor: shape_predictor instance.

        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image.


        Returns:
        --------
        five_points:
            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.
        """

        five_points = dict()
        five_points_dict = dict()

        status, raw_image = check_image(image)
        if status != 0:
            return five_points

        detector = dlib.get_frontal_face_detector()

        detect_faces = detector(raw_image, 2)
        
        if len(detect_faces) > 0:
            for num, roi in enumerate(detect_faces):
                five_points = dict()

                shape_face = shape_predictor(raw_image, roi)
                lefteye_leftcorner, lefteye_rightcorner, righteye_rightcorner, righteye_leftcorner, nose = shape_face.parts()
                five_points["lefteye_leftcorner"] = (lefteye_leftcorner.x, lefteye_leftcorner.y)
                five_points["lefteye_rightcorner"] = (lefteye_rightcorner.x, lefteye_rightcorner.y)
                five_points["righteye_rightcorner"] = (righteye_rightcorner.x, righteye_rightcorner.y)
                five_points["righteye_leftcorner"] = (righteye_leftcorner.x, righteye_leftcorner.y)
                five_points["nose"] = (nose.x, nose.y)
                five_points_dict[num] = five_points

            if vision:
                for item in five_points_dict:
                    person = five_points_dict[item]
                    for point in person:
                        coordinate = person[point]
                        cv2.circle(raw_image, coordinate, 2, (0, 255, 0), 2)

                cv2.imshow("Face Landmark", raw_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_path != None:
                cv2.imwrite(save_path, raw_image)
                
                if not os.path.exists(save_path):
                    logging.error("{} saved failed !".format(save_path))
                    raise FileNotFoundError
        else:
            logging.info("Dlib doesn't detect the face !")
        
        return five_points_dict


    @classmethod
    def dlib_68_points(cls,
                       image=None,
                       shape_predictor=object(),
                       get_five_points=False,
                       vision=False,
                       save_path=None):
        """
        dlib_68_points method is use face sixty-eight points of dlib to mark sixty-eight of face.
        
        Args:
        -----
        image: Input image path or image array.

        shape_predictor: shape_predictor instance.

        get_five_points: Control only get five points of face from sixty-eight points of face.

        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image.

        
        Returns:
        --------
        five_points_dict:
            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.

        sixty_eight_points_dict: Sixty eight points of face.
        """

        five_points = dict()
        sixty_eight_points = dict()
        five_points_dict = dict()
        sixty_eight_points_dict = dict()


        status, raw_image = check_image(image)
        if status != 0:
            return sixty_eight_points
            
        detector = dlib.get_frontal_face_detector()

        detect_faces = detector(raw_image, 2)

        if len(detect_faces) > 0:
            for num, roi in enumerate(detect_faces):
                shape_face = shape_predictor(raw_image, roi)

                if get_five_points:
                    five_points = dict()
                    five_points["lefteye_leftcorner"] = (shape_face.part(36).x, shape_face.part(36).y)
                    five_points["lefteye_rightcorner"] = (shape_face.part(39).x, shape_face.part(39).y)
                    five_points["righteye_rightcorner"] = (shape_face.part(45).x, shape_face.part(45).y)
                    five_points["righteye_leftcorner"] = (shape_face.part(42).x, shape_face.part(42).y)
                    five_points["nose"] = (shape_face.part(30).x, shape_face.part(30).y)
                    five_points_dict[num] = five_points
                else:
                    sixty_eight_points = dict()
                    for i in range(0, 68):
                        sixty_eight_points[i] = (shape_face.part(i).x, shape_face.part(i).y)

                    sixty_eight_points_dict[num] = sixty_eight_points
            
            if vision:
                if get_five_points:
                    for item in five_points_dict:
                        person = five_points_dict[item]
                        for point in person:
                            coordinate = person[point]
                            cv2.circle(raw_image, coordinate, 2, (0, 255, 0), 2)
                else:       
                    for item in sixty_eight_points_dict:
                        person = sixty_eight_points_dict[item]
                        for point in person:
                            coordinate = person[point]
                            cv2.circle(raw_image, coordinate, 2, (0, 255, 0), 2)

                cv2.imshow("Face Landmark", raw_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if save_path != None:
                cv2.imwrite(save_path, raw_image)
                
                if not os.path.exists(save_path):
                    logging.error("{} saved failed !".format(save_path))
                    raise FileNotFoundError
        else:
            logging.info("Dlib doesn't detect the face !")

        if get_five_points:
            return five_points_dict
        else:
            return sixty_eight_points_dict

    
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