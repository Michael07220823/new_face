# Usage
# python -m face.kernel.landmark

import os
import logging
import dlib
from new_tools import check_image

if __name__ == "__main__":
    # Set logging config.
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)

class FaceLandmark(object):
    """
    FaceLandmark class use three methods to mark face.
    """

    @classmethod
    def dlib_5_points(cls, image=str(), predictor_path=str()):
        """
        dlib_5_points method is use face five points of dlib to mark left eye、right eye、nose of face.

        Args:
        -----
        image: Input image path.

        predictor_path: Input shape five points predictor path. 


        Returns:
        --------
        five_points:

            lefteye_leftcorner: left eye corner coordinate of left eye.
            lefteye_rightcorner: Right eye corner coordinate of left eye.
            righteye_rightcorner: Right eye corner coordinate of right eye.
            righteye_leftcorner: Left eye corner coordinate of right eye.
            nose: Nose coordinate.

        0: No detect the face.
        """

        # Read image.
        image_array = check_image(image)

        detector = dlib.get_frontal_face_detector()

        if not os.path.exists(predictor_path):
            logging.error("{} path error !".format(predictor_path), exc_info=True)
            raise FileNotFoundError
        predictor = dlib.shape_predictor(predictor_path)

        # Detect face and get roi.
        detect_face = detector(image_array, 2)
        
        if len(detect_face) > 0:
            for num, roi in enumerate(detect_face):
                shape_face = predictor(image_array, roi)

                # Get five points from face.
                five_point = dict()
                lefteye_leftcorner, lefteye_rightcorner, righteye_rightcorner, righteye_leftcorner, nose = shape_face.parts()
                five_point["lefteye_leftcorner"] = (lefteye_leftcorner.x, lefteye_leftcorner.y)
                five_point["lefteye_rightcorner"] = (lefteye_rightcorner.x, lefteye_rightcorner.y)
                five_point["righteye_rightcorner"] = (righteye_rightcorner.x, righteye_rightcorner.y)
                five_point["righteye_leftcorner"] = (righteye_leftcorner.x, righteye_leftcorner.y)
                five_point["nose"] = (nose.x, nose.y)

                return five_point
        else:
            logging.info("No detect the face !")


    @classmethod
    def dlib_68_points(cls, image=str(), predictor_path=str(), get_five_points=False):
        """
        dlib_68_points method is use face sixty-eight points of dlib to mark sixty-eight of face.
        
        Args:
        -----
        image: Input image path.

        predictor_path: Input shape sixty-eight points predictor path. 

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

        # Read image.
        image = check_image(image)

        detector = dlib.get_frontal_face_detector()

        if not os.path.exists(predictor_path):
            logging.error("{} path error !".format(predictor_path), exc_info=True)
            raise FileNotFoundError

        predictor = dlib.shape_predictor(predictor_path)

        # Detect face and get roi.
        detect_face = detector(image, 2)

        if len(detect_face) > 0:
            for i, roi in enumerate(detect_face):
                shape_face = predictor(image, roi)
                sixty_eight_points = dict()

                for num in range(0, 68):
                    sixty_eight_points[num] = (shape_face.part(num).x, shape_face.part(num).y)

                # Get five points from face.
                if get_five_points:
                    five_points = dict()
                    five_points["lefteye_leftcorner"] = (shape_face.part(46).x, shape_face.part(46).y)
                    five_points["lefteye_rightcorner"] = (shape_face.part(43).x, shape_face.part(43).y)
                    five_points["righteye_rightcorner"] = (shape_face.part(37).x, shape_face.part(37).y)
                    five_points["righteye_leftcorner"] = (shape_face.part(40).x, shape_face.part(40).y)
                    five_points["nose"] = (shape_face.part(34).x, shape_face.part(34).y)

                    return five_points
                
                return sixty_eight_points
        else:
            logging.info("No detect the face !")
            return 0
            

    
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
    def fivepoint2threepoint(cls, five_point=dict()):
        """
        fivepoint2threepoint method used to transfer 5 points to 3 points.

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
        three_point:

            left_eye: left eye center coordinate.
            right_eye: Right eye center coordinate.
            nose: Nose coordinate.
        """
        
        three_point = dict()
        if len(five_point) < 5:
            logging.error("five_point variable element small than 5 !")
            raise ValueError
        lefteye_leftcorner_x1, lefteye_leftcorner_y1 = five_point["lefteye_leftcorner"]
        lefteye_rightcorner_x2, lefteye_rightcorner_y2 = five_point["lefteye_rightcorner"]

        three_point["left_eye"] = cls.__calc_center_point(lefteye_leftcorner_x1, 
                                                          lefteye_leftcorner_y1, 
                                                          lefteye_rightcorner_x2, 
                                                          lefteye_rightcorner_y2
                                                         )

        righteye_leftcorner_x1, righteye_leftcorner_y1 = five_point["righteye_leftcorner"]
        righteye_rightcorner_x2, righteye_rightcorner_y2 = five_point["righteye_rightcorner"]

        three_point["right_eye"] = cls.__calc_center_point(righteye_leftcorner_x1,
                                                           righteye_leftcorner_y1, 
                                                           righteye_rightcorner_x2,
                                                           righteye_rightcorner_y2
                                                          )
        three_point["nose"] = five_point["nose"]

        return three_point


if __name__ == "__main__":
    image = "face/data/test/people-faces-collection.jpg"
    
    five_point = FaceLandmark.dlib_5_points(image=image,
                                            predictor_path="face/models/landmark/shape_predictor_5_face_landmarks.dat")
    logging.info(five_point)

    five_point = FaceLandmark.dlib_68_points(image=image,
                                             predictor_path="face/models/landmark/shape_predictor_68_face_landmarks.dat",
                                             get_five_points=True)
    logging.info(five_point)
    logging.info(FaceLandmark.fivepoint2threepoint(five_point))
