import os
import gdown
import tarfile
from tarfile import ReadError
import logging


FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


dataset_dir = "dataset"
dataset_name = "YaleB_align_256.tar.gz"

try:
    url = "https://drive.google.com/u/0/uc?export=download&id=1M9PuZ6EEHeyxolrl8Hg5ZlQtq3W-dgid"
    dataset_path = os.path.join(dataset_dir, dataset_name)
      
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    logging.info("Downloading {} dataset...".format(dataset_path))
    gdown.download(url, dataset_path, quiet=False)

    if os.path.exists(dataset_path):
        logging.info("{} downlads successfuly !".format(dataset_path))

        dataset = tarfile.open(dataset_path, "r")

        dataset.extractall(dataset_dir)
        dataset.close()
    
        if os.path.exists(dataset_path.split('.')[0]):
            os.remove(dataset_path)
            logging.info("Extracted files successfuly !")
        else:
            logging.error("Extracted files failed !")
except ReadError as ReadErr:
    logging.error(ReadErr, exc_info=True)