import os
import logging
import cv2
import numpy as np
from skimage.feature import local_binary_pattern as LBP
from new_face import LBPCNN, FaceDetection


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


lbpcnn = LBPCNN()
vc = cv2.VideoCapture(0)
face_detection = FaceDetection()
network = face_detection.load_detector(face_detection.SSD_DNN)

LBP_sample_point = 8
LBP_radius = 2
LBP_method = "uniform"

lbpcnn.load_model("models/lbpcnn/20211116_185342/20211116_185342_lbpcnn_label_encoder.pickle",
                  "models/lbpcnn/20211116_185342/20211116_185342_best_checkpoint/lbpcnn_0005.ckpt")


while vc.isOpened():
    _, frame = vc.read()
    rois, raw, faces = face_detection.ssd_dnn_detect(network,
                                                     frame)

    if len(faces) > 0:
        # LBPCNN input shape is 256x256.
        face = cv2.resize(faces[0], (256, 256))

        pred_ID, pred_proba = lbpcnn.predict(face)
        pred_name = lbpcnn.label_encoder.classes_[pred_ID]
        logging.info("Prediction: {}:{:.2f}%".format(pred_name, pred_proba * 100))

        x, y, w, h = rois[0]
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "{}:{:.2f}%".format(pred_name, pred_proba * 100), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    cv2.imshow("LBPCNN prediction", frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        vc.release()
        cv2.destroyAllWindows()
        break
