# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:20:05 2021

@author: Erin S
"""

"""CNN Classfier Class"""

# Imports
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

from utils.config import Config



class classifierModel():
    """Base Model class for CNN classifier models"""
    def __init__(self, cfg, baseModel, model_name):
        
        self.baseModel = baseModel
        self.model_name = model_name
        self.config = Config.from_json(cfg)
        self.image_path = self.config.data.path
        self.image_size = self.config.data.image_size
        self.batch_size = self.config.train.batch_size
        self.num_epochs = self.config.train.epochs
        self.dense = self.config.model.dense_layer
        self.dropout = self.config.model.dropout
        self.num_classes = self.config.model.num_classes
        self.validation_split = self.config.train.val_split
        self.init_lr = self.config.model.init_lr
        self.preprocess = None
        self.callbacks = None
        self.model = None
    
     
    def create_generators(self):
        """ Create Keras train and validate data generators """
        
        train_datagen = ImageDataGenerator(
                        horizontal_flip = True,
                        vertical_flip = True,                   
                        validation_split = self.validation_split,
                        preprocessing_function = self.preprocess)  

        train_generator = train_datagen.flow_from_directory(
                         self.image_path,
                         target_size = (self.image_size,self.image_size),
                         batch_size = self.batch_size,
                         class_mode = 'categorical',
                         subset = 'training')

        validation_generator = train_datagen.flow_from_directory(
                               self.image_path, 
                               target_size = (self.image_size,self.image_size),
                               batch_size = self.batch_size,
                               class_mode ='categorical',
                               subset ='validation')    # set as validation data)
     
        return train_datagen, train_generator, validation_generator
        
   
    def build(self):
        """ Build head model for training classifier. Two fully connected layers and two dropout layers are included """
        
        # Ensure layer of base model are not trainable for feature extraction
        for layer in self.baseModel.layers:
            layer.trainable = False
        
        headModel = self.baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.dense, activation="relu")(headModel)
        headModel = Dropout(self.dropout)(headModel)
        headModel = Dense(self.dense, activation="relu")(headModel)
        headModel = Dropout(self.dropout)(headModel)
        headModel = Dense(self.num_classes, activation="softmax")(headModel)
        self.model = Model(inputs = self.baseModel.input, outputs = headModel)
        
        return

    def set_callbacks(self):
        """Create desired training callbacks """
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
        return 
    
    
    def normalize_VGG(self):
        self.preprocess = vgg_preprocess
        return
    
    def normalize_ResNet(self):
        self.preprocess = resnet_preprocess
        return
    
    def normalize_inception(self):
        self.preprocess = inception_preprocess
        return
    
    def normalize(self):
        """Set data preprocessing function specific to model name """
        
        if 'vgg' in self.model_name:
            self.normalize_VGG()
            
        elif 'resnet' in self.model_name:
            self.normalize_ResNet()
            
        elif 'inception' in self.model_name:
            self.normalize_inception()
         
        else:
            raise ValueError(self.model_name)
        
        return    
            
    def train(self):
        """ Compile and train CNN classifier model """
        
        #Set unique preprocess function to be used in generators
        self.normalize()
       
        # Create generators
        train_datagen, train_generator, validation_generator = self.create_generators()
        
        # Build Model
        self.build()
              
        # Create callbacks
        self.set_callbacks()
       
        # Compile and train model
        print("[INFO] compiling model...")
        opt = Adam(lr = self.init_lr, decay = self.init_lr / self.num_epochs)
        self.model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
        
        model_history = self.model.fit(train_generator,
                                       steps_per_epoch = train_generator.samples // self.batch_size,
                                       validation_data = validation_generator, 
                                       validation_steps = validation_generator.samples // self.batch_size,
                                       epochs = self.num_epochs,
                                       callbacks = self.callbacks)
                                  
        return model_history


    def predict(self):
        """ TODO """
        return
    
  
    
    