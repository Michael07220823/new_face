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
import imutils
from mtcnn import MTCNN
from new_tools import check_image
from ..tools.download import download_models
from ..tools.config import root_dir


class FaceDetection(object):
    """
    FaceDetect class can use four kinds method to detect face. Four methods: Haar Cascade、Dlib Hog、SSD DNN and MTCNN.
    """
    HAAR = 0
    DLIB = 1
    SSD_DNN = 2
    MTCNN = 3


    def load_detector(self, method_ID=int()):
        """
        load_detector method is used to load detector of face detection for reducing each loading time.

        Args:
        -----
        method_ID:
            HAAR: Haar Cascade detector.
            DLIB: Dlib HOG detector.
            SSD_DNN: SSD DNN detector.
            MTCNN: MTCNN detector.
        

        Return:
        -------
        detector: Face detector.

        network: SSD DNN network.
        """
        
        models_id_name = {
                            0: "haarcascade_frontalface_default.xml",
                            2: ["deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"]
                        }
        
        if method_ID == self.HAAR:
            model_name = models_id_name[method_ID]
            model_path = os.path.join(root_dir, model_name)
            
            if not os.path.exists(model_path):
                download_models(model_name=model_name, save_path=root_dir)

            detector = cv2.CascadeClassifier(os.path.join(root_dir, model_name))

            return detector

        elif method_ID == self.DLIB:    
            detector = dlib.get_frontal_face_detector()
            return detector

        elif method_ID == self.SSD_DNN:
            network = None

            for model in models_id_name[method_ID]:
                model_path = os.path.join(root_dir, model)
                
                if not os.path.exists(model_path):
                    download_models(model_name=model, save_path=root_dir)
            
            face_config = os.path.join(root_dir, models_id_name[method_ID][0])
            face_model = os.path.join(root_dir, models_id_name[method_ID][1])
            network = cv2.dnn.readNetFromCaffe(face_config, face_model)

            return network

        elif method_ID == self.MTCNN:
            detector = MTCNN()
            
            return detector


    def haar_detect(self,
                    detector,
                    image=None,  
                    scale_factor=1.025, 
                    min_neighbors=18,
                    min_size=15,
                    max_size=200,
                    vision=False,
                    save_path=None):
        """
        haar_detect method detects face by opencv haar cascade method.

        Args:
        -----
        detector: Haar cascade detector instance.

        image: Input image path or image array.

        scale_factor: Parameter specifying how much the image size is reduced at each image scale.

        min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.

        min_size: Minimum possible object size. Objects smaller than that are ignored.

        max_size: Maximum possible object size. Objects larger than that are ignored. If maxSize == minSize model is evaluated on single scale.

        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image.


        Returns:
        --------
        rois: (x, y, w, h)
            x: Face left-top corner x coordinate point.
            y: Face left-top corner y coordinate point.
            w: Width.
            h: Height.

        raw_image: Raw image.

        face_images: Face images.
        """

        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status != 0:
            return rois, raw_image, face_images

        raw_image = image.copy()
        draw_image = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        faces = detector.detectMultiScale(gray,
                                          scaleFactor=scale_factor,
                                          minNeighbors=min_neighbors,
                                          minSize=(min_size, min_size),
                                          maxSize=(max_size, max_size))
    
        logging.debug("detection.haar_detect.faces count: {}".format(len(faces)))

        if len(faces) > 0:
            for roi in faces:    
                (x, y, w, h) = roi
                rois.append(roi)
                face_images.append(image.copy()[y:y+h, x:x+w])

            if vision:
                for num, (x,y,w,h) in enumerate(rois, start=1):
                    cv2.rectangle(draw_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(draw_image, str(num), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            logging.info("Haar Cascade doesn't detected the faces !")

        if vision:
            text = "Haar Detected Faces: {}".format(len(face_images))
            cv2.putText(draw_image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
            cv2.imshow('Haar Cascade Face Detection', imutils.resize(draw_image, width=640))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path != None:
            cv2.imwrite(save_path, draw_image)

            if not os.path.exists(save_path):
                logging.error("{} saved failed !".format(save_path))
                raise FileNotFoundError

        logging.debug("Haar Cascade detected faces: {}".format(len(face_images)))
        return (rois, raw_image, face_images)
    
    
    def dlib_detect(self,
                    detector,
                    image=None,
                    unsample=0,
                    conf_threshold=0.15,
                    vision=False,
                    save_path=None):
        """
        dlib_detect method detects face by HOG and SVM method.

        Args:
        -----
        detector: Dlib face detector instance.

        image: Input image path or image array.
        
        unsample: Number of times to upsample an image before applying face detection. Default 0 is fastest.

        conf_threshold: conf_threshold value is used to judge the face detection true or false.

        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image. 


        Returns:
        --------
        rois: (x, y, w, h)
            x: Face left-top corner x coordinate point.
            y: Face left-top corner y coordinate point.
            w: width.
            h: Height.

        raw_image: Raw image.

        face_images: Face images.
        """

        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status != 0:
            return rois, raw_image, face_images

        raw_image = image.copy()
        draw_image = image.copy()

        faces, scores, idxs = detector.run(image, unsample, conf_threshold)
        logging.debug("detection.dlib_detect.faces count: {}".format(len(faces)))
        logging.debug("detection.dlib_detect.scores count: {}".format(len(scores)))
        logging.debug("detection.dlib_detect.idxs count: {}".format(len(idxs)))
    
        if len(faces) > 0:
            for face in faces:
                roi = self.__rect_to_roi(face)
                rois.append(roi)
                x, y, w, h = roi
                face_images.append(image.copy()[y:h, x:w])

            if vision:
                for num, (x, y, w, h) in enumerate(rois, start=1):
                    cv2.rectangle(draw_image, (x, y), (w, h), (0, 255, 0), 3)
                    cv2.putText(draw_image, str(num), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            logging.info("Dlib HoG doesn't detected the faces !")

        if vision:
            text = "Dlib HoG Detected Faces: {}".format(len(face_images))
            cv2.putText(draw_image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
            cv2.imshow("Dlib HoG Face Detection", imutils.resize(draw_image, width=640))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path != None:
            cv2.imwrite(save_path, draw_image)

            if not os.path.exists(save_path):
                logging.error("{} saved failed !".format(save_path))
                raise FileNotFoundError

        logging.debug("Dlib HOG+SVM detected faces: {}".format(len(face_images)))
        return (rois, raw_image, face_images)


    def ssd_dnn_detect(self,
                       network,
                       image=None,
                       conf_threshold=0.7,
                       vision=False,
                       save_path=None):
        """
        ssd_dnn_detect method detects face by tensorflow or caffe ssd dnn framework.

        Args:
        -----
        network: Caffe or TensorFlow framework instance.

        image: Input image path or image array.

        conf_threshold: conf_threshold value is used to judge the face detection true or false.

        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image. 


        Returns:
        --------
        rois:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.

        raw_image: Raw image.

        face_images: Face images.
        """

        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status != 0:
            return rois, raw_image, face_images
            
        raw_image = image.copy()
        draw_image = image.copy()

        frameHeight = image.shape[0]
        frameWidth = image.shape[1]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)

        network.setInput(blob)
        detections = network.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x = int(detections[0, 0, i, 3] * frameWidth)
                y = int(detections[0, 0, i, 4] * frameHeight)
                w = int(detections[0, 0, i, 5] * frameWidth)
                h = int(detections[0, 0, i, 6] * frameHeight)
                
                rois.append((x, y, w, h))
                face_images.append(image.copy()[y:h, x:w])
        logging.debug("detection.ssd_dnn_detect.face_images count: {}".format(len(face_images)))
                
        if vision:
            if len(face_images) > 0:
                for num, (x, y, w, h) in enumerate(rois, start=1):
                    cv2.rectangle(draw_image, (x, y), (w, h), (0, 255, 0), 3)
                    cv2.putText(draw_image, str(num), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
            else:
                logging.info("SSD DNN doesn't detected the faces !")

            text = "SSD DNN Detected Faces: {}".format(len(face_images))
            cv2.putText(draw_image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
            cv2.imshow("SSD DNN Face Detection", imutils.resize(draw_image, width=640))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path != None:
            cv2.imwrite(save_path, draw_image)
            
            if not os.path.exists(save_path):
                logging.error("{} saved failed !".format(save_path))
                raise FileNotFoundError

        logging.debug("SSD DNN detected faces: {}".format(len(face_images)))
        return (rois, raw_image, face_images)


    def mtcnn_detect(self,
                     detector,
                     image=None,
                     min_face_size=20,
                     conf_threshold=0.75,
                     vision=False,
                     save_path=None):
        """
        mtcnn_detect method detects face by multi-task cascaded convolutional networks method.

        Args:
        -----
        detector: MTCNN object.

        image: Input image path or image array.

        min_face_size: Face detection minimum size. Default 20.

        conf_threshold: conf_threshold value is used to judge the face detection true or false.
        
        vision: Show image.

        save_path: Save images of detected faces. If vision is False, will doesn't save image.
        

        Returns:
        --------
        rois: (x, y, w, h)
            x: Face left-top corner x coordinate point.
            y: Face left-top corner y coordinate point.
            w: Width.
            h: Height.

        raw_image: Raw image.

        face_images: Face images.
        """
        
        rois = list()
        raw_image = None
        face_images = list()

        status, image = check_image(image)
        if status != 0:
            return rois, raw_image, face_images
            
        raw_image = image.copy()
        draw_image = image.copy()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detector.min_face_size = min_face_size
        result = detector.detect_faces(rgb_image)
        logging.debug("detection.mtcnn_detect.result count: {}".format(len(result)))

        if len(result) > 0:
            for persorn in result:
                if persorn["confidence"] >= conf_threshold:
                    x, y, w, h = persorn["box"]
                    rois.append((x, y, w, h))
                    face_images.append(raw_image.copy()[y:y+h, x:x+w])

            if vision:
                for num, (x,y,w,h) in enumerate(rois, start=1):
                    cv2.rectangle(draw_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(draw_image, str(num), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            logging.info("MTCNN doesn't detected the faces ! ")

        if vision:
            text = "MTCNN Detected Faces: {}".format(len(face_images))
            cv2.putText(draw_image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
            cv2.imshow("MTCNN Face Detection", imutils.resize(draw_image, width=640))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path != None:
            cv2.imwrite(save_path, draw_image)
            
            if not os.path.exists(save_path):
                logging.error("{} saved failed !".format(save_path))
                raise FileNotFoundError

        logging.debug("MTCNN detected faces: {}".format(len(face_images)))
        return (rois, raw_image, face_images)


    def __rect_to_roi(self, rect):
        """
        __rect_to_roi method is used to get roi values from dlib.rectangle class.

        Args:
        -----
        rect: Face roi.
        
        
        Return:
        -------
        rois:
            (x, y, w, h)
                x: Face left-top corner x coordinate point.
                y: Face left-top corner y coordinate point.
                w: Width.
                h: Height.
        """

        roi = (rect.left(), rect.top(), rect.right(), rect.bottom())
        logging.debug("detection.__rect_to_roi.roi: {}".format(roi))

        return roi