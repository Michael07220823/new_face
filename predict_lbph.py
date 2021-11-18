import logging
import cv2
from new_face import LBPH, FaceDetection


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


lbph = LBPH()
vc = cv2.VideoCapture(0)
face_detection = FaceDetection()
network = face_detection.load_detector(face_detection.SSD_DNN)

lbph.load_model("models/lbph/20211117_091601_YaleB_align_256_8_2_8_8/20211117_091601_YaleB_align_256_8_2_8_8-lbph_labels.pickle",
                "models/lbph/20211117_091601_YaleB_align_256_8_2_8_8/20211117_091601_YaleB_align_256_8_2_8_8-lbph_model.yaml")


while vc.isOpened():
    _, frame = vc.read()
    rois, raw, faces = face_detection.ssd_dnn_detect(network,
                                                     frame)
    if len(faces) > 0:
        # LBPH input shape is 256x256.
        face = cv2.resize(faces[0], (256, 256))
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        pred_ID, pred_distance = lbph.predict(gray_face)
        pred_name = lbph.label_encoder.classes_[pred_ID]
        logging.info("Prediction: {}:{:.4f}".format(pred_name, pred_distance))

        x, y, w, h = rois[0]
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "{}:{:.4f}".format(pred_name, pred_distance), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    cv2.imshow("LBPH prediction", frame)
    key = cv2.waitKey(10)
    
    if key == ord('q'):
        vc.release()
        cv2.destroyAllWindows()
        break