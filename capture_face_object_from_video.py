# Usage
# python capture_face_object_from_video.py --source 0 -pi 0
# python capture_face_object_from_video.py --source data/YaleB_align_256/yaleB11 -pi 1


import os
import logging
import argparse
import cv2
from new_face import FaceDetection
from new_timer import AutoTimer
from new_tools import check_image

FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, type=str, help="Video or directory.")
ap.add_argument("-pi", "--person_index", required=True, type=str, default='0', help="Person index, ex: adam = 0, david = 1,...etc.")
ap.add_argument("-o", "--output", required=False, type=str, default="outputs", help="Video or directory output directory.")
args = vars(ap.parse_args())


def main():
    video = None
    video_source = 0
    person_index = args["person_index"]
    image_paths = None
    output_path = None
    frame_counter = 0
    pass_counter = 0

    face_detect = FaceDetection()
    detector = face_detect.load_detector(face_detect.MTCNN)

    try:
        # Source
        source = args["source"]
        if os.path.exists(source) and os.path.isfile(source):
            video_source = 0
            video = cv2.VideoCapture(source)
            frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            frames_position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        elif source.isdigit():
            video_source = 1
            video = cv2.VideoCapture(int(source))
        else:
            video_source = 2
            image_dir = source

        # Output
        output_path = args["output"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logging.info("Builed {} successfully !".format(output_path))

        # Capture frame and write yolo format text.
        if video_source == 0 or video_source == 1:
            while video != None and video.isOpened():
                state, frame = video.read()
                state, frame = check_image(frame)

                if state == 0:
                    image_height, image_width, image_channel = frame.shape
                    logging.debug("Height:{}, Width: {}".format(image_height, image_width))

                    # detector = face_detect.load_detector(face_detect.SSD_DNN)

                    rois, raw_image, face_images = face_detect.mtcnn_detect(detector,
                                                                            frame,
                                                                            conf_threshold=0.9)
                    # rois, raw_image, face_images = face_detect.ssd_dnn_detect(detector,
                    #                                                           frame,
                    #                                                           conf_threshold=0.9)
                    if len(face_images) > 0:
                        # Image path.
                        image_name = "{}".format(frame_counter).zfill(10) + ".jpg"
                        image_path = os.path.join(output_path, image_name)
                        logging.debug("image_name: {}".format(image_name))
                        logging.debug("image_path: {}".format(image_path))

                        # Text path.
                        object_text_name = "{}".format(frame_counter).zfill(10) + ".txt"
                        object_text_path = os.path.join(output_path, object_text_name)
                        logging.debug("object_text_name: {}".format(object_text_name))
                        logging.debug("object_text_path: {}".format(object_text_path))
                        frame_counter += 1

                        # Save image.
                        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        logging.info("Saved {} successfully !".format(image_path)) if os.path.exists(image_path) else logging.error("Saved {} failed !".format(image_path))

                        # Save text.
                        with open(object_text_path, "w", encoding="utf8") as f:
                            # for x1, y1, x2, y2 in rois:
                            #     logging.debug("{} {} {} {}".format(x1, y1, x2, y2))
                                # object_width = abs(x2 - x1)
                                # object_height = abs(y2 - y1)
                                # center_x = (x1 + (object_width / 2)) / image_width
                                # center_y = (y1 + (object_height / 2)) / image_height
                                # object_width = object_width  / image_width
                                # object_height = object_height / image_height

                            for x, y, w, h in rois:
                                center_x = (x + (w / 2)) / image_width
                                center_y = (y + (h / 2)) / image_height
                                object_width = w / image_width
                                object_height = h / image_height
                            
                                f.write("{} {} {} {} {}\n".format(person_index, center_x, center_y, object_width, object_height))
                            logging.info("Writed {} face objects to {} successfully !".format(len(rois), object_text_path))
                    else:
                        logging.info("Doesn't detect the faces!")
                else: 
                    pass_counter += 1
                    logging.warning("Pass frame {} count !".format(pass_counter))
                
                # Get video frame index.
                frames_position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            image_paths = os.listdir(image_dir)
            for img_path in image_paths:
                full_image_path = os.path.join(image_dir, img_path)
                rois, raw, face_images = face_detect.mtcnn_detect(detector,
                                                                  full_image_path,
                                                                  conf_threshold=0.9)
                image_height, image_width, image_channel = raw.shape
                
                if len(face_images) > 0:
                        # Image path.
                        image_name = "{}".format(frame_counter).zfill(10) + ".jpg"
                        image_path = os.path.join(output_path, image_name)
                        logging.debug("image_name: {}".format(image_name))
                        logging.debug("image_path: {}".format(image_path))

                        # Text path.
                        object_text_name = "{}".format(frame_counter).zfill(10) + ".txt"
                        object_text_path = os.path.join(output_path, object_text_name)
                        logging.debug("object_text_name: {}".format(object_text_name))
                        logging.debug("object_text_path: {}".format(object_text_path))
                        frame_counter += 1

                        # Save image.
                        cv2.imwrite(image_path, raw, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        logging.info("Saved {} successfully !".format(image_path)) if os.path.exists(image_path) else logging.error("Saved {} failed !".format(image_path))

                        # Save text.
                        with open(object_text_path, "w", encoding="utf8") as f:
                            # for x1, y1, x2, y2 in rois:
                            #     logging.debug("{} {} {} {}".format(x1, y1, x2, y2))
                                # object_width = abs(x2 - x1)
                                # object_height = abs(y2 - y1)
                                # center_x = (x1 + (object_width / 2)) / image_width
                                # center_y = (y1 + (object_height / 2)) / image_height
                                # object_width = object_width  / image_width
                                # object_height = object_height / image_height

                            for x, y, w, h in rois:
                                center_x = (x + (w / 2)) / image_width
                                center_y = (y + (h / 2)) / image_height
                                object_width = w / image_width
                                object_height = h / image_height
                            
                                f.write("{} {} {} {} {}\n".format(person_index, center_x, center_y, object_width, object_height))
                            logging.info("Writed {} face objects to {} successfully !".format(len(rois), object_text_path))
    except KeyboardInterrupt:
        if video_source == 0 or video_source == 1: video.release()
        logging.info("Captured 'ctrl+c' to interrupt program !")

if __name__ == "__main__":
    with AutoTimer("Captured face objects", 4):
        main()