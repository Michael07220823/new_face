import os
import pickle
import logging
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

from new_tools import check_image


class LBPH(object):
    """
    LBPH class use to show LBPH parameters、trained model、updated model、load model and predict model.

    Attributes:
    -----------
    neighbors: set LBP neighbors point count.

    radius: set LBP circle radius value.

    grid_x: set x axis grid count.

    grid_y: set y axis grid count.

    recognizer: LBPHFaceRecognizer object. 
    """

    def __init__(self,
                 neighbors=8,
                 radius=1,
                 grid_x=8,
                 grid_y=8):
        """
        Init label encoder and LBPH recognizer.

        Args
        ----
        neighbors: LBP sample points.

        radius: LBP radius.

        grid_x: Grid x axis amount.

        grid_y: Grid y axis amount.
        """
        self.label_encoder = None
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=radius, 
                                                             neighbors=neighbors, 
                                                             grid_x=grid_x, 
                                                             grid_y=grid_y)
    @property
    def neighbors(self):
        return self.recognizer.getNeighbors()
    
    @property
    def radius(self):
        return self.recognizer.getRadius()
    
    @property
    def grid_x(self):
        return self.recognizer.getGridX()
    
    @property
    def grid_y(self):
        return self.recognizer.getGridY()
    
    def show_lbph_params(self):
        """
        Show LBPH all parameters.
        """

        logging.info("LBPH Neighbors: {}".format(self.recognizer.getNeighbors()))
        logging.info("LBPH Radius:    {}".format(self.recognizer.getRadius()))
        logging.info("LBPH Grid X:    {}".format(self.recognizer.getGridX()))
        logging.info("LBPH Grid Y:    {}".format(self.recognizer.getGridY()))


    def train_model(self,
                    images=np.array([]),
                    labels=np.array([]),
                    label_encoder_path=str(),
                    model_path=str()):
        """
        train LBPH model and save LBPH model and label encoder object.

        Args:
        -----
        images: Images array.
        
        labels: Labels array.
        
        label_encoder_path: Label encoder path.
        
        model_path: LBPH model path.
        """
        
        logging.info("Encoding labels...")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        with open(label_encoder_path, "wb") as lab_file:
            pickle.dump(label_encoder, lab_file)

        if os.path.exists(label_encoder_path):
            logging.info("Saved label encoder object successfully !")
        else:
            logging.warning("Saved label encoder object failed !")
            
        logging.info("Training model...")
        self.recognizer.train(images, encoded_labels)
        
        self.recognizer.write(model_path)
        if os.path.exists(model_path):
            logging.info("Saved model successfully !")
            logging.info("Saved model to {}.".format(os.path.dirname(model_path)))
        else:
            logging.warning("Saved model failed !")


    def update_model(self, 
                     model_path=str(),
                     images=np.array([]),
                     labels=np.array([])):
        """
        update_model can update the model without retraining.

        Args:
        -----
        model_path: Path of model saves.

        images: Images array.
        
        labels: Labels array.
        """

        logging.info("Updating model...")
        self.recognizer.update(images, labels)

        logging.info("Saving model...")
        self.recognizer.write(model_path)

        logging.info("Update model successfuly !")


    def load_model(self,
                   label_encoder_path=str(),
                   model_path=str()):
        """
        Load LBPH model and label encoder object.
        
        Args:
        -----
        label_encoder_path: Label encoder path.
        
        model_path: Model path.
        """
        if not os.path.exists(label_encoder_path):
            logging.critical("{} path error !".format(label_encoder_path), exc_info=True)
            raise FileNotFoundError
        if not os.path.exists(model_path):
            logging.critical("{} path error !".format(model_path), exc_info=True)
            raise FileNotFoundError

        logging.info("Loading label ecnoder...")
        with open(label_encoder_path, "rb") as lab:
            self.label_encoder = pickle.load(lab)

        logging.info("Loading model...")
        self.recognizer.read(model_path)


    def predict(self,
                gray_image=np.array([])):
        """
        Predict image.
        
        Args
        ----
        gray_image: Gray scale image.
        
        Return
        ------
        predict_id: LBPH model predict ID.
        
        predict_distance: LBPH model predict distance. The smaller the better.
        """

        state, gray_image = check_image(gray_image)
        if state == 0:
            predict_id, predict_distance = self.recognizer.predict(gray_image)
            logging.debug("Prediction result: {}:{:.6f}".format(predict_id, predict_distance))

        return predict_id, predict_distance