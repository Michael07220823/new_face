import os
import logging
import cv2
from new_tools.media import check_image, FPS
from new_timer import get_now_time, AutoTimer

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

with AutoTimer("Record video", decimal=2):
    # User input his name.
    name = input("{} [INFO] Please input your name: ".format(get_now_time()))
    video_dir = input("{} [INFO] Please input save directory: ".format(get_now_time()))
    if not os.path.exists(video_dir):
        raise FileNotFoundError("{} directory doesn't exist !".format(video_dir))

    # Calculate fps.
    fps = FPS()

    # Webcam source and some information.
    run = True
    vc = cv2.VideoCapture(0)
    video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = 60
    video_name = os.path.join(video_dir, "{}_{}.mp4".format(get_now_time("%Y%m%d%H%M%S"), name))

    logging.info("Video info")
    logging.info("Video Resolution: {} × {}.".format(video_width, video_height))
    logging.info("Video FPS: {} fps.".format(video_fps))
    logging.info("Video file path: {}.".format(video_name))

    # Video format.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, video_fps, (video_width, video_height))

    fps.start()
    while run and vc.isOpened():
        _, frame = vc.read()
        state, frame = check_image(frame)

        if state == 0:
            # Save frame.
            video.write(frame)

            # Show frame.
            cv2.putText(frame, "Video Resolution: {} x {}".format(video_width, video_height), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, "Video FPS: {} fps.".format(video_fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, "Video name: {}".format(os.path.basename(video_name)), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, "Video Real-Time FPS: {:.2f} fps.".format(fps.fps()), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                fps.stop()
                logging.debug("fps elapsed time: {}".format(fps.elapsed_time))
                logging.info("Video Real-Time FPS: {:.2f} fps.".format(fps.fps()))
                
                run = False
                cv2.destroyAllWindows()
                video.release()
                vc.release()

        # Update frame
        fps.update()
        logging.debug("fps frame: {}".format(fps.frame))

        # Get elapsed time.
        fps.stop()
    
    logging.info("{} saved successfully !".format(video_name)) if os.path.exists(video_name) else logging.error("{} saved failed !".format(video_name))