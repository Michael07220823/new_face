# coding=utf-8
import os
import logging
import cv2
import numpy as np
from imutils import resize
from sklearn.model_selection import train_test_split
from new_face import LBPH
from new_timer import AutoTimer, get_now_time

# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


build_time = get_now_time("%Y%m%d_%H%M%S")

logging.info("Initializing variables...")
LBP_sample_point = 8
LBP_radius = 2
LBP_grid_x = 8
LBP_grid_y = 8

temp_labels = list()
temp_images_path = list()

train_labels = list()
train_images_path = list()
train_images_array = list()

test_labels = list()
test_images_path = list()
test_images_array = list()

dataset = "YaleB_align_256"
dataset_path = "dataset/YaleB_align_256"

model_dir = "models/lbph/{}_{}_{}_{}_{}_{}".format(build_time, dataset, LBP_sample_point, LBP_radius, LBP_grid_x, LBP_grid_y)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
lbph = LBPH(neighbors=LBP_sample_point,
            radius=LBP_radius,
            grid_x=LBP_grid_x,
            grid_y=LBP_grid_y)

model_path = os.path.join(model_dir, "{}_{}_{}_{}_{}_{}-lbph_model.yaml".format(build_time, dataset,
                         LBP_sample_point, LBP_radius, LBP_grid_x, LBP_grid_y))
label_encoder_path = os.path.join(model_dir, "{}_{}_{}_{}_{}_{}-lbph_labels.pickle".format(build_time, dataset,
                                  LBP_sample_point, LBP_radius, LBP_grid_x, LBP_grid_y))


with AutoTimer("Training LBPH model", 0):
    for root, dirs, files in os.walk(dataset_path):
        if len(files) <= 0:
            continue
        for file in files:
            # Windows
            name = root.split("\\")[-1]
            # Linux
            # name = root.split("/")[-1]
            logging.debug(name)

            file_path = os.path.join(root, file)
            logging.debug(file_path)

            temp_labels.append(name)
            temp_images_path.append(file_path)

        logging.debug(len(temp_labels))
        logging.debug(len(temp_images_path))
        x_train, x_test, y_train, y_test = train_test_split(temp_images_path,
                                                            temp_labels,
                                                            test_size=0.2,
                                                            train_size=0.8,
                                                            random_state=42)
        train_images_path.extend(x_train)
        train_labels.extend(y_train)
        test_images_path.extend(x_test)
        test_labels.extend(y_test)
        temp_labels.clear()
        temp_images_path.clear()

    logging.info("training labels count: {}".format(len(train_labels)))
    logging.info("training images count: {}".format(len(train_images_path)))
    logging.info("testing labels count: {}".format(len(test_labels)))
    logging.info("testing images count: {}".format(len(test_images_path)))

    # Read sample path to gray image.
    logging.info("Reading training images to gray images...")
    for num, path in enumerate(train_images_path, start=1):
        logging.info("Processing {}/{} images...".format(num, len(train_images_path)))
        gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        train_images_array.append(gray_image)

    logging.info("Processed {} gray-scale images.".format(len(train_images_array)))
    
    # Transforms data type from list to ndarray instance.
    logging.info("Transforming data type from list to numpy array...")
    train_images_array = np.array(train_images_array)
    logging.debug("train_lbph.train_images.shape: {}".format(train_images_array.shape))

    # Transform list to ndarray instance.
    train_labels = np.array(train_labels)

    # Train and save LBPH model.
    lbph.train_model(train_images_array,
                     train_labels,
                     label_encoder_path,
                     model_path)