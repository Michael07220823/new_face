# coding=utf-8
import os
import pickle
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern as LBP
from new_face import LBPCNN
from new_tools import check_image
from new_timer import AutoTimer, get_now_time


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

build_time = get_now_time("%Y%m%d_%H%M%S")

temp_labels = list()
temp_images_path = list()

train_labels = list()
train_images_path = list()
train_images_array = list()

test_labels = list()
test_images_path = list()
test_images_array = list()

LBP_sample_point = 8
LBP_radius = 2
LBP_method = "uniform"

classes = 28
epochs = 10
batch_size = 32
lbpcnn = LBPCNN()
label_encoder = LabelEncoder()

model_dir = "models/lbpcnn"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

dataset_path = "dataset/YaleB_align_256"
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


with AutoTimer("Training LBPCNN model", decimal=0):
    # LBP frature extraction.
    logging.info("Extracting LBP feature...")

    # Training data.
    logging.info("Extracting training data...")
    for num, path in enumerate(train_images_path, start=1):
        logging.info("Processing {}/{} training images...".format(num, len(train_images_path)))
        # Gray image.
        gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        state, gray_image = check_image(gray_image)
        
        if state == 0:
            # LBP image.
            lbp_image = LBP(gray_image, P=LBP_sample_point, R=LBP_radius, method=LBP_method)
            
            # Reshape LBP image to shape (x, 256, 256, 1)
            lbp_image = np.expand_dims(lbp_image, axis=-1)
            train_images_array.append(lbp_image)

    # Testing data.
    logging.info("Extracting testing data...")
    for num, path in enumerate(test_images_path, start=1):
        logging.info("Processing {}/{} testing images...".format(num, len(test_images_path)))
        
        # Gray image.
        gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        state, gray_image = check_image(gray_image)
        
        if state == 0:
            # LBP image.
            lbp_image = LBP(gray_image, P=LBP_sample_point, R=LBP_radius, method=LBP_method)
            
            # Reshape LBP image to shape (x, 256, 256, 1)
            lbp_image = np.expand_dims(lbp_image, axis=-1)
            test_images_array.append(lbp_image)
    
    # Feature Normalizaion.
    logging.info("Normalizing feature...")
    train_images_array = np.asarray(train_images_array, dtype=np.float32) / 255.0
    test_images_array = np.asarray(test_images_array, dtype=np.float32) / 255.0

    # Label encoder to int array.
    logging.info("Encoding labels...")
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    
    # Save LabelEncoder instance.
    logging.info("Saving label encoder...")
    label_encoder_dir = os.path.join(model_dir, build_time)
    if not os.path.exists(label_encoder_dir):
        os.makedirs(label_encoder_dir)
        
    label_encoder_path = os.path.join(model_dir, build_time, "{}_lbpcnn_label_encoder.pickle".format(build_time))
    with open(label_encoder_path, "wb") as lab:
        pickle.dump(label_encoder, lab)

    logging.info("train_images_array.shape: {}".format(train_images_array.shape))
    logging.info("train_labels.shape: {}".format(train_labels.shape))
    logging.info("train_labels[:10]: {}".format(train_labels[:10]))
    print()

    logging.info("test_images_array.shape: {}".format(test_images_array.shape))
    logging.info("test_labels.shape: {}".format(test_labels.shape))
    logging.info("test_labels[:10]: {}".format(test_labels[:10]))
    print()
    
    # Build LBPCNN model
    lbpcnn.build_LBPCNN_model(classes=classes, input_shape=(256, 256, 1), learning_rate=2.5e-4) 
    
    # train SVM classifier.
    train_history = lbpcnn.train_model(train_images_array,
                                       train_labels,
                                       validation_data=(test_images_array, test_labels),
                                       model=lbpcnn.model,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       time_record=build_time,
                                       save_path=model_dir)

    logging.debug(train_history.history["accuracy"])
    logging.debug(train_history.history["val_accuracy"])
    logging.debug(train_history.history["loss"])
    logging.debug(train_history.history["val_loss"])
    

    # Draw training and validation accuracy and loss chart.
    # Accuracy chart.
    plt.title("LBPCNN Training Accuracy History")
    plt.plot(train_history.history["accuracy"])
    plt.plot(train_history.history["val_accuracy"])
    plt.ylabel("Accuracy Rate")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="center right")
    plt.grid(color="gray",
             alpha=0.2,
             linewidth=1,
             linestyle="--")
    plt.savefig(os.path.join(model_dir, build_time, "{}_accuracy.jpg".format(build_time)), bbox_inches ='tight')
    plt.show()
    
    # Loss chart.
    plt.title("LBPCNN Training Loss History")
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="center right")
    plt.grid(color="gray",
             alpha=0.2,
             linewidth=1,
             linestyle="--")
    plt.savefig(os.path.join(model_dir, build_time, "{}_loss.jpg".format(build_time)), bbox_inches ='tight')
    plt.show()