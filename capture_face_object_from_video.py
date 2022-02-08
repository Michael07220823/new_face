# Usage
# python capture_face_object_from_video.py --source "data/480P/2022-zui-xin-dian-ying-duan-jin-dian-ying-gao-qing-1080p-full-movies.mp4" --output outputs/MTCNN/480P

import os
import logging
import argparse
import cv2
from new_face import FaceDetection
from new_timer import AutoTimer

FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, type=str, help="Video or directory.")
ap.add_argument("-o", "--output", required=False, type=str, default="outputs", help="Video or directory output directory.")
args = vars(ap.parse_args())


def main():
    video = None
    dir_path = None
    output_path = None
    frame_counter = 0
    pass_counter = 0

    try:
        # Source
        source = args["source"]
        if os.path.exists(source) and not os.path.isdir(source):
            video = cv2.VideoCapture(source)
            frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            frames_position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            dir_path = source

        # Output
        output_path = args["output"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logging.info("Builed {} successfully !".format(output_path))

        # Capture frame and write yolo format text.
        while video != None and video.isOpened() and frames_position < frames_count:
            state, frame = video.read()

            if state:
                image_height, image_width, image_channel = frame.shape
                logging.debug("Height:{}, Width: {}".format(image_height, image_width))

                face_detect = FaceDetection()
                detector = face_detect.load_detector(face_detect.MTCNN)
                # detector = face_detect.load_detector(face_detect.SSD_DNN)

                rois, raw_image, face_images = face_detect.mtcnn_detect(detector,
                                                                        frame,
                                                                        conf_threshold=0.9)
                # rois, raw_image, face_images = face_detect.ssd_dnn_detect(detector,
                #                                                           frame,
                #                                                           conf_threshold=0.9)
                if len(face_images) > 0:
                    # Image path.
                    image_name = "{}".format(frame_counter).zfill(8) + ".png"
                    image_path = os.path.join(output_path, image_name)
                    logging.debug("image_name: {}".format(image_name))
                    logging.debug("image_path: {}".format(image_path))

                    # Text path.
                    object_text_name = "{}".format(frame_counter).zfill(8) + ".txt"
                    object_text_path = os.path.join(output_path, object_text_name)
                    logging.debug("object_text_name: {}".format(object_text_name))
                    logging.debug("object_text_path: {}".format(object_text_path))
                    frame_counter += 1

                    # Save image.
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                    logging.info("Saved {} successfully !".format(image_path)) if os.path.exists(image_path) else logging.error("Saved {} failed !".format(image_path))

                    # Save text.
                    with open(object_text_path, "w") as f:
                        for x, y, w, h in rois:
                            center_x = (x + (w / 2)) / image_width
                            center_y = (y + (h / 2)) / image_height
                            object_width = w / image_width
                            object_height = w / image_height
                        
                            f.write("0 {} {} {} {}\n".format(center_x, center_y, object_width, object_height))
                        logging.info("Writed {} face objects to {} successfully !".format(len(rois), object_text_path))
                else:
                    logging.info("Doesn't detect the faces!")
            else: 
                pass_counter += 1
                logging.warning("Pass frame {} count !".format(pass_counter))
            
            # Get video frame index.
            frames_position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    except KeyboardInterrupt:
        video.release()
        logging.info("Captured 'ctrl+c' to interrupt program !")

if __name__ == "__main__":
    with AutoTimer("Captured face objects", 4):
        main()