import os
import logging
import cv2
import numpy as np
from new_face import OpenFace, FaceDetection


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)


openface = OpenFace()
vc = cv2.VideoCapture(0)
face_detection = FaceDetection()
network = face_detection.load_detector(face_detection.SSD_DNN)

openface.load_model("models/openface/20211117_214104_YaleB_align_256/20211117_214104_YaleB_align_256_100_rbf_svm_label.pickle",
                    "models/openface/20211117_214104_YaleB_align_256/20211117_214104_YaleB_align_256_100_rbf_svm.pickle")

while vc.isOpened():
    _, frame = vc.read()
    rois, raw, faces = face_detection.ssd_dnn_detect(network,
                                                     frame)
                                                     
    if len(faces) > 0:
        # OpenFace input shape is 256x256.
        face = cv2.resize(faces[0], (256, 256))
        
        pred_ID, pred_proba = openface.predict(face)
        pred_name = openface.label_encoder.classes_[pred_ID]
        logging.info("Prediction: {}:{:.4f}%".format(pred_name, pred_proba * 100))

        x, y, w, h = rois[0]
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "{}:{:.4f}%".format(pred_name, pred_proba * 100), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("OpenFace prediction", frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        vc.release()
        cv2.destroyAllWindows()
        break