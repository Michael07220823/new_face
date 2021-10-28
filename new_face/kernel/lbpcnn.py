"""
MIT License

Copyright (c) 2021 Overcomer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pickle
import logging
import numpy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from .openface import OpenFace
from new_tools import check_image


class LBPCNN(OpenFace):
    def __init__(self):
        """
        Init label_encoder and model.
        """

        self.label_encoder = None
        self.model = None


    def build_LBPCNN_model(self, name="LBPCNN", classes=18):
        """
        LBPCNN architecture.

        Args
        ----
        name: Specify CNN model name.

        classes: Specify classes amount.
        """
        model = Sequential(name=name)

        # CNN 1.
        model.add(layers.Conv2D(filters=40, kernel_size=(3, 3), padding="same", input_shape=(256, 256, 1), activation="relu", name="cnn1_1"))
        model.add(layers.Conv2D(filters=40, kernel_size=(3, 3), padding="same", activation="relu", name="cnn1_2"))
        model.add(layers.BatchNormalization(name="bn1"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="mp1"))

        # CNN 2.
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="cnn2_1"))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="cnn2_2"))
        model.add(layers.BatchNormalization(name="bn2"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="mp2"))

        # CNN 3.
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu", name="cnn3_1"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu", name="cnn3_2"))
        model.add(layers.BatchNormalization(name="bn3"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="mp3"))

        # CNN 4.
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="cnn4_1"))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="cnn4_2"))
        model.add(layers.BatchNormalization(name="bn4"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="mp4"))

        # CNN 5.
        model.add(layers.Conv2D(filters=40, kernel_size=(3, 3), padding="same", activation="relu", name="cnn5_1"))
        model.add(layers.Conv2D(filters=40, kernel_size=(3, 3), padding="same", activation="relu", name="cnn5_2"))
        model.add(layers.BatchNormalization(name="bn5"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="mp5"))

        # # Flatten.
        model.add(layers.Flatten(name="flatten"))

        # # Fully Conected 1.
        model.add(layers.Dense(256, activation="relu", name="fc1"))
        model.add(layers.Dropout(0.25, name="dp1"))
        model.add(layers.BatchNormalization(name="bn6"))
        
        # Fully Conected 2.
        model.add(layers.Dense(128, kernel_regularizer=regularizers.L1L2(0.0001), bias_regularizer=regularizers.L1L2(0.0001), activation="relu", name="fc2"))
        model.add(layers.Dropout(0.25, name="dp2"))
        model.add(layers.BatchNormalization(name="bn7"))

        # Softmax.
        model.add(layers.Dense(28, activation="softmax", name="softmax"))

        # Model compile.
        model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=Adam(learning_rate=2.5e-4),
                    metrics=["accuracy"])

        # Show model layer.
        model.summary()

        self.model = model
    
    
    def load_model(self,
                   label_encoder_path=str(),
                   model_path=str()):
        """
        Load LBPCNN model from SaveModel class. You need to specify model diretory path.

        Args
        ----
        label_encoder_path: Specify label encoder path.

        model_path: Specify LBPCNN model path.
        """
        
        logging.info("Loading LBPCNN model...")
        if not os.path.exists(label_encoder_path):
            logging.critical("'{}' error ! Loaded LBPCNN label encoder failed.".format(label_encoder_path))
            raise FileNotFoundError

        if not os.path.exists(model_path):
            logging.critical("'{}' error ! Loaded LBPCNN model failed.".format(model_path))
            raise FileNotFoundError
            
        with open(label_encoder_path, "rb") as lab:
            self.label_encoder = pickle.load(lab)

        self.model = load_model(model_path)
        
        logging.info("Loaded LBPCNN model successfully !")

    
    def train_model(self,
                    images,
                    labels,
                    validation_data=tuple(),
                    model=Sequential(),
                    epochs=200,
                    batch_size=128,
                    verbose=1,
                    time_record=str(),
                    save_path=str()):
        """
        Train LBPCNN model.
        
        Aegs
        ----
        images: Training images array.
        
        labels: Training labels.
        
        validation_data: Input validation images array and validation labels.
        
        model: Sequential model instance.
        
        epochs: Training model count.
        
        batch_size: Images amount of once training model. Power of 2 like 2,4,8,...,128.
        
        verbose: Training model real time detail information.
        
        time_record: Build training model directory prefixes.
        
        save_path: Save model root directory.
        
        Return
        ------
        train_history: LBPCNN model training history.
        """
        
        logging.info("Training LBPCNN model...")
        
        # Make model saved path.
        root_path = os.path.join(save_path, "{}".format(time_record))
        logging.info("Building LBPCNN model saved path to {}...".format(root_path))
        if not os.path.join(root_path):
            os.makedirs(root_path)
        
        # Save LBPCNN model architecture image.
        plot_image = os.path.join(root_path, "{}_LBPCNN_model_architecture.png".format(time_record))
        logging.info("Saving LBPCNN model architecture image to {}...".format(plot_image))
        plot_model(model,
                   to_file=plot_image,
                   show_shapes=True,
                   show_layer_names=True)
        
        # Tensorboard callback.
        log_dir = os.path.join(root_path, "logs\\fit\\{}".format(time_record))
        logging.info("Building tensorboard log directory to {}...".format(log_dir))
        os.makedirs(log_dir)
        
        tensorboard_callback = TensorBoard(log_dir=log_dir,
                                           histogram_freq=1,
                                           write_graph=True,
                                           write_images=True,
                                           embeddings_freq=False)

        # ModelCheckpoint callback.
        checkpoint_dir = os.path.join(root_path, "{}_best_checkpoint".format(time_record))
        logging.info("Building checkpoint directory to {}...".format(checkpoint_dir))
        os.makedirs(checkpoint_dir)
        
        checkpoint_filepath = os.path.join(checkpoint_dir, "lbpcnn_{epoch:04d}.ckpt")
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                              monitor='val_loss',
                                              mode='min',
                                              verbose=0,
                                              save_best_only=True)
        # Train LBPCNN model.
        train_history = model.fit(ｘ=images,
                                  ｙ=labels,
                                  validation_data=validation_data,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  verbose=verbose,
                                  callbacks=[tensorboard_callback, checkpoint_callback])
        logging.info("Saved LBPCNN model to {}.".format(root_path))
        logging.info("Finished LBPCNN model training !")

        return train_history
        
    
    def predict(self,
                image):
        """
        Predict image.

        Args
        ----
        image: LBP image format.

        Return
        ------
        pred_id: Prediction index.

        pred_proba: Prediction probability.
        """
        
        pred_id = None
        pred_proba = None
        
        state, image = check_image(image)
        if state == 0:
            predition = self.model.predict(image)[0]
            logging.debug("lbpcnn.LBPCNN.predict.prediction: {}".format(predition))
            logging.debug("lbpcnn.LBPCNN.predict.prediction shape: {}".format(predition.shape))
            pred_id = numpy.argmax(predition)
            pred_proba = predition[pred_id]
            
            logging.debug("Prediction result: {}:{:.6f}".format(pred_id, pred_proba))
            
        return pred_id, pred_proba