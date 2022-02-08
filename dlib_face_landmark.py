import cv2
import dlib
import imutils
from imutils.face_utils.facealigner import FaceAligner

vc = cv2.VideoCapture(0)
_, frame = vc.read()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/overcomer/.new_face/shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(predictor, desiredFaceWidth=256)


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 2)
print(rects)

for rect in rects:
    print(rect)
    face_aligned = face_aligner.align(frame, gray, rect)
    cv2.imshow("Original", frame)
    cv2.imshow("Face", face_aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vc.release()