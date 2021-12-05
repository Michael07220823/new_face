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
import pickle
import logging
import cv2
import numpy as np
from imutils import resize
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from new_tools import check_image
from new_face.tools.config import root_dir
from new_face.tools.download import download_models


class OpenFace(object):
    def __init__(self):
        self.label_encoder = None
        self.embedder_model = None
        self.classifier = None


    def extract_embeddings(self,
                           images_path=list(),
                           vision=False):
        """
        Use OpenFace nn4.small2.v1.t7 network model to extract face feature and output feature vectors.
        
        Args
        ----
        embedding_network_path: OpenFace nn4.small2.v1.t7 network model path.
        
        images_path: Images path.
        
        vision: Show image.
        
        Return
        ------
        embedding_vectors: Face feature vectors.
        """
        
        logging.info("Extracting images...")
        
        total = 0
        embedding_vectors = list()

        for num, image_path in enumerate(images_path, start=1):
            logging.info("Processing {}/{} images...".format(num, len(images_path)))
            
            state, bgr_image = check_image(image_path)
            if state == 0:
                if vision:
                    cv2.imshow("Extracting image...", resize(bgr_image, width=250))
                    cv2.waitKey(10)

                face_blob = cv2.dnn.blobFromImage(bgr_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

                self.embedder_model.setInput(face_blob)
                vec = self.embedder_model.forward()

                embedding_vectors.append(vec.flatten())
                total += 1
                
        if vision:
            cv2.destroyAllWindows()
            
        logging.info("Extracted {} count face feature vectors.".format(total))

        return embedding_vectors


    def train_model(self,
                    C=1,
                    kernel="linear",
                    gamma="scale",
                    embedding_vectors=list(),
                    labels=list(),
                    classifier_path=str(),
                    label_encoder_path=str()):
        """
        Train svm classifier by face feature vectors.
        
        Args
        ----
        embedding_vectors: Extract face feature vector by OpenFace nn4.small2.v1.t7 network model.
        
        labels: People name.
        
        classifier_path: Classifier saved path.
        
        label_encoder_path: Label encoder saved path.
        """
        
        logging.info("Encoding labels...")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        logging.info("Training SVM classifier...")
        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        self.classifier.fit(embedding_vectors, encoded_labels)

        logging.info("Saving Label encoder...")
        with open(label_encoder_path, "wb") as f: 
            f.write(pickle.dumps(label_encoder))
        if os.path.exists(label_encoder_path): 
            logging.info("Saved label encoder successfully !")
        
        logging.info("Saving classifier...")
        with open(classifier_path, "wb") as f:
            f.write(pickle.dumps(self.classifier))
        if os.path.exists(classifier_path):
            logging.info("Saved classifier successfully !")
            logging.info("Saved classifier to {}.".format(os.path.dirname(classifier_path)))
            
        logging.info("Trained OpenFace classifier finish !")


    def load_model(self,
                   label_encoder_path=str(),
                   classifier_path=str()):
        """
        Load OpenFace label encoder、classifier and embedding network model.
        
        Args
        ----
        label_encoder_path: Label encoder saved path.
        
        classifier_path: Classifier saved path.

        
        Return
        ------
        label_encoder_path: Label encoder saved path.
        
        classifier_path: Classifier saved path.
        """

        logging.info("Loading label encoder...")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as lab:
                self.label_encoder = pickle.load(lab)
        
        logging.info("Loading classifier...")
        if os.path.exists(classifier_path):
            with open(classifier_path, "rb") as classifier:
                self.classifier = pickle.load(classifier)
        
        logging.info("Loading OpenFace nn4.small2.v1.t7 model...")
        model_name = "nn4.small2.v1.t7"
        model_path = os.path.join(root_dir, model_name)
        if not os.path.exists(model_path):
            download_models(model_name, root_dir)
        self.embedder_model = cv2.dnn.readNetFromTorch(model_path)

        if label_encoder_path == str() or classifier_path == str():
            logging.critical("Please input label encoder and classifier path !")
            raise ValueError

        logging.info("Loaded OpenFace model finish !")


    def predict(self,
                bgr_image=np.ndarray):
        """
        Use OpenFace nn4.small2.v1.t7 network model predict image.
        
        Args
        ----
        classifier: Classifier saved path.
        
        embedder_network: Embedder network model saved path.
        
        bgr_image: BGR format image.

        Return
        ------
        predict_id: 

        probability:
        """
        
        predict_id = None
        probability = None
        
        state, bgr_image = check_image(bgr_image)
        if state == 0:
            face_blob = cv2.dnn.blobFromImage(bgr_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder_model.setInput(face_blob)
            vec = self.embedder_model.forward()

            predicts = self.classifier.predict_proba(vec)[0]
            predict_id = np.argmax(predicts)
            probability = predicts[predict_id]
            logging.debug("Prediction result: {}:{:.6f}".format(predict_id, probability))

        return predict_id, probability