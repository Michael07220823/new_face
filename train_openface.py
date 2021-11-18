# coding=utf-8
import os
import logging
import numpy
from sklearn.model_selection import train_test_split
from new_face import OpenFace, root_dir
from new_timer import AutoTimer, get_now_time


# Set logging config.
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


insert_to_mongodb_time = get_now_time("%Y%m%d_%H%M%S")

logging.info("Initializing variables...")
build_time = get_now_time("%Y%m%d_%H%M%S")

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

model_dir = "models/openface"
build_dir = os.path.join(model_dir, "{}_{}".format(build_time, dataset))
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

C = 100
gamma="auto"
svm_kernel = "rbf"    
openface = OpenFace()

embedding_network_path = os.path.join(root_dir, "nn4.small2.v1.t7")
label_encoder_path = os.path.join(build_dir, "{}_{}_{}_{}_svm_label.pickle".format(build_time, dataset, C, svm_kernel))
classifier_path = os.path.join(build_dir, "{}_{}_{}_{}_svm.pickle".format(build_time, dataset, C, svm_kernel))

# Load openface embedder.
openface.load_model(embedder_network_path=embedding_network_path)

with AutoTimer("Training OpenFace model", 0):
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

    # Transform list to ndarray instance.
    train_labels = numpy.array(train_labels)

    # Feature extract from openface model.
    face_feature_vectors = openface.extract_embeddings(train_images_path,
                                                       vision=True)

    # train SVM classifier.
    openface.train_model(C=C,
                         kernel=svm_kernel,
                         gamma=gamma,
                         embedding_vectors=face_feature_vectors,
                         labels=train_labels,
                         classifier_path=classifier_path,
                         label_encoder_path=label_encoder_path)