# Usage
# python capture_face_ssd_dnn.py --video_source 0 --flip false --size 256 --name michael --output output --vision true
# python capture_face_ssd_dnn.py --video_source images/doctor.mp4 --flip false --size 256 --name unknown --output output --vision true
# python capture_face_ssd_dnn.py --video_source images/michael.jpg --flip false --size 256 --name michael --output output --vision true
# python capture_face_ssd_dnn.py --video_source images --flip false --size 256 --name unknown --output output --vision true

import os
import argparse
import logging
import cv2
from new_timer import AutoTimer
from new_face import FaceDetection
from new_tools import check_image, IMAGE_FORMAT, VIDEO_FORMAT


run = True

with AutoTimer("Capture face", 0):
    # Set logging config.
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


    # Arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-vs", "--video_source", required=True, default=0, help="Input video source. value: 0,1,...,8 or d:\\xxxx.mp4")
    ap.add_argument("-f", "--flip", required=False, type=str, default="false", help="Flip video frame. value: 0, 1, -1, false")
    ap.add_argument("-s", "--size", required=False, type=int, default=256, help="Resize the face image to the specified size. value: 96, 128, 224")
    ap.add_argument("-n", "--name", required=False, type=str, help="People name. value: Michael, Eric, Addison.")
    ap.add_argument("-o", "--output", required=True, type=str, default="output", help="Output images path. value: d:\\xxxx\\")
    ap.add_argument("-v", "--vision", required=False, type=str, default="false",  help="Show face image of alignment. value: true or false.")
    args = vars(ap.parse_args())

    # Load face detector and alignmenter.
    logging.info("Loading face aligner...")
    face_detection = FaceDetection()
    ssd_network = face_detection.load_detector(face_detection.SSD_DNN)

    # Check video source type.
    """
    vc status code:
        0: Webcam.
        1: Video file.
        2: Image file.
        3: Directory.
    """
    logging.info("Checking video source...")
    video_source = args["video_source"]
    logging.debug("capture_face.video_source: {}".format(video_source))

    # Webcam
    if video_source.isnumeric():
        vc_code = 0
        vc = cv2.VideoCapture(int(video_source))
    # Video or image.
    elif os.path.isfile(video_source):
        deputy_name = os.path.splitext(video_source)[-1]
        logging.debug("capture_face.deputy_name: {}".format(deputy_name))
        
        if deputy_name.upper() in VIDEO_FORMAT or deputy_name.lower() in VIDEO_FORMAT:
            vc_code = 1
            vc = cv2.VideoCapture(video_source)
        elif deputy_name in IMAGE_FORMAT:
            vc_code = 2
    # Directory.
    elif os.path.isdir(video_source):
        vc_code = 3
        file_list = os.listdir(video_source)
        logging.debug("capture_face.file_list: {}".format(file_list))
    # Not any type.
    else:
        logging.warning("Video source can't recognition !")
        vc_code = 4
    logging.debug("capture_face.vc_code: {}".format(vc_code))

    # Build output path.
    root_dir = os.path.join(args["output"], args["name"])
    logging.info("Output directory: {}".format(root_dir))
    if not os.path.exists(root_dir): 
        os.makedirs(root_dir)

    # Webcam.
    if vc_code == 0:
        face_counter = 0

        try:
            vc_state = vc.isOpened()
            logging.debug("capture_face.vc_state: {}".format(vc_state))

            while vc_state and run:
                state, frame = vc.read()
                logging.debug("capture_face.state: {}".format(state))

                # Flip frame.
                if state and args["flip"] != "false":
                    frame = cv2.flip(frame, int(args["flip"]))

                # Detection face.
                rois, raw_image, face_images = face_detection.ssd_dnn_detect(ssd_network,
                                                                             frame,
                                                                             conf_threshold=0.9)
    
                # Save and show face images.
                if len(face_images) > 0:
                    for face in face_images:
                        face_counter += 1
                        image_path = os.path.join(root_dir, "{}_{}.jpg".format(args["name"], str(face_counter).zfill(8)))
                        resize_face = cv2.resize(face, (args["size"], args["size"]), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(image_path, resize_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                        if not os.path.exists(image_path):
                            logging.warning("{} saved failed !".format(image_path))
                            raise ValueError

                        if args["vision"] == "true":
                            cv2.imshow("Captured face...", resize_face)
                            key = cv2.waitKey(10)

                            if key == ord('q'):
                                vc.release()
                                cv2.destroyAllWindows()
                                run = False

                        logging.info("Captured {} face images...".format(face_counter))
            cv2.destroyAllWindows()
        except Exception as err:
            logging.exception("{}".format(err), exc_info=True)
            vc.release()
            cv2.destroyAllWindows()

    # Video.
    elif vc_code == 1:
        face_counter = 0

        # Take video basic information.
        video_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_rate = vc.get(cv2.CAP_PROP_FPS)
        video_total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)

        logging.info("Video Source: {}".format(video_source))
        logging.info("Video Width : %d pixs" % video_width)
        logging.info("Video Height: %d pixs" % video_height)
        logging.info("Video Rate  : %.2f fps" % video_rate)
        logging.info("Total Frame : %d frame" % video_total_frames)

        try:
            vc_state = vc.isOpened()
            logging.debug("capture_face.vc_state: {}".format(vc_state))

            while vc_state and vc.get(cv2.CAP_PROP_POS_FRAMES) < video_total_frames and run:
                state, frame = vc.read()

                # Flip frame.
                if state and args["flip"] != "false":
                    frame = cv2.flip(frame, int(args["flip"]))

                # Detection face.
                rois, raw_image, face_images = face_detection.ssd_dnn_detect(ssd_network,
                                                                             frame,
                                                                             conf_threshold=0.9)
                # Save and show face images.
                if len(face_images) > 0:
                    for face in face_images:
                        face_counter += 1
                        image_path = os.path.join(root_dir, "{}_{}.jpg".format(args["name"], str(face_counter).zfill(8)))
                        resize_face = cv2.resize(face, (args["size"], args["size"]), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(image_path, resize_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                        if not os.path.exists(image_path):
                            logging.warning("{} saved failed !".format(image_path))
                            raise ValueError

                        if args["vision"] == "true":
                            cv2.imshow("Capture face...", resize_face)
                            key = cv2.waitKey(10)

                            if key == ord('q'):
                                vc.release()
                                cv2.destroyAllWindows()
                                run = False

                        logging.info("Captured {} face images...".format(face_counter))
            cv2.destroyAllWindows()
        except Exception as err:
            logging.exception("{}".format(err), exc_info=True)
            vc.release()
            cv2.destroyAllWindows()

    # Image.
    elif vc_code == 2:
        try:
            state, image = check_image(video_source)

            if args["flip"] != "false":    
                if state != 0:
                    logging.critical("{} path error !".format(video_source))
                    raise ValueError
                image = cv2.flip(image, int(args["flip"]))

            # Detection face.
            rois, raw_image, face_images = face_detection.ssd_dnn_detect(ssd_network,
                                                                         image,
                                                                         conf_threshold=0.9)
            # Save and show face images.
            if len(face_images) > 0:
                for face in face_images:
                    image_path = os.path.join(root_dir, "{}.jpg".format(args["name"]))
                    resize_face = cv2.resize(face, (args["size"], args["size"]), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(image_path, resize_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    if not os.path.exists(image_path):
                        logging.warning("{} saved failed !".format(image_path))
                        raise ValueError

                    if args["vision"] == "true":
                        cv2.imshow("Capture face...", resize_face)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    logging.info("Finished captured face image !")
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    # Directory.
    elif vc_code == 3:
        vc = None
        face_counter = 0

        try:
            for item in file_list:
                full_path = os.path.join(video_source, item)
                logging.debug("capture_face.full_path: {}".format(full_path))
                
                # Image
                if os.path.isfile(full_path) and os.path.splitext(full_path)[-1].lower() in IMAGE_FORMAT:
                    state, image = check_image(full_path)

                    if args["flip"] != "false":
                        image = cv2.flip(image, int(args["flip"]))
                    
                    # Detection face.
                    rois, raw_image, face_images = face_detection.ssd_dnn_detect(ssd_network,
                                                                                 image,
                                                                                 conf_threshold=0.9)
                    # Save and show face images.
                    if len(face_images) > 0:
                        for face in face_images:
                            face_counter += 1
                            image_path = os.path.join(root_dir, "{}_{}.jpg".format(args["name"], str(face_counter).zfill(8)))
                            resize_face = cv2.resize(face, (args["size"], args["size"]), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(image_path, resize_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                            if not os.path.exists(image_path):
                                logging.warning("{} saved failed !".format(image_path))
                                raise ValueError

                            if args["vision"] == "true":
                                cv2.imshow("Capture face...", resize_face)
                                key = cv2.waitKey(10)

                                if key == ord('q'):
                                    vc.release()
                                    cv2.destroyAllWindows()
                                    run = False

                            logging.info("Captured {} face images...".format(face_counter))
                # Video.
                elif os.path.isfile(full_path) and os.path.splitext(full_path)[-1].lower() in VIDEO_FORMAT:
                    face_counter = 0

                    # Take directory name.
                    dir_name = os.path.splitext(item)[0]
                    dir_path = os.path.join(root_dir, dir_name)
                    logging.debug("capture_face.dir_name: {}".format(dir_name))
                    logging.debug("capture_face.dir_path: {}".format(dir_path))

                    # Build directory of saved extraction face video.
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                        logging.info("{} builed successfully !".format(dir_path))

                    # Init video.
                    vc = cv2.VideoCapture(full_path)

                    # Take video basic information.
                    video_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
                    video_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    video_rate = vc.get(cv2.CAP_PROP_FPS)
                    video_total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)

                    logging.info("Video Source: {}".format(video_source))
                    logging.info("Video Width : %d pixs" % video_width)
                    logging.info("Video Height: %d pixs" % video_height)
                    logging.info("Video Rate  : %.2f fps" % video_rate)
                    logging.info("Total Frame : %d frame" % video_total_frames)

                    vc_state = vc.isOpened()
                    logging.debug("capture_face.vc_state: {}".format(vc_state))

                    while vc_state and vc.get(cv2.CAP_PROP_POS_FRAMES) < video_total_frames and run:
                        state, frame = vc.read()
                        
                        if args["flip"] != "false":
                            frame = cv2.flip(frame, int(args["flip"]))
                        
                        # Detection face.
                        rois, raw_image, face_images = face_detection.ssd_dnn_detect(ssd_network,
                                                                                     frame,
                                                                                     conf_threshold=0.9)

                        # Save and show face images.
                        if len(face_images) > 0:
                            for face in face_images:
                                face_counter += 1
                                image_path = os.path.join(dir_path, "{}_{}.jpg".format(dir_name, str(face_counter).zfill(8)))
                                resize_face = cv2.resize(face, (args["size"], args["size"]), interpolation=cv2.INTER_AREA)
                                cv2.imwrite(image_path, resize_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                if not os.path.exists(image_path):
                                    logging.warning("{} saved failed !".format(image_path))
                                    raise ValueError

                                if args["vision"] == "true":
                                    cv2.imshow("Capture face...", resize_face)
                                    key = cv2.waitKey(10)

                                    if key == ord('q'):
                                        vc.release()
                                        cv2.destroyAllWindows()
                                        run = False
                                    
                                logging.info("Captured {} face images...".format(face_counter))
            logging.info("Finished captured face image !")
            cv2.destroyAllWindows()
        except Exception as err:
            logging.exception("{}".format(err), exc_info=True)

            if vc != None:
                vc.release()

            cv2.destroyAllWindows()
    else:
        logging.critical("Error !")