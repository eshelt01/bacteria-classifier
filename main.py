# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 00:06:24 2021

@author: Erin S
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
from config.config import CFG
from config.config import CFG_RESNET
from config.config import CFG_INCEPTION
from models.base_model import classifierModel

#------ VGG 16 Model -------#   
baseModel_vgg16 = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), pooling = 'avg')
    
classifier_vgg16 = classifierModel(CFG, baseModel_vgg16, 'vgg16')
history_vgg16 = classifier_vgg16.train()    

classifier_vgg16.save_classifier('./saved_models/VGG16_model/')
classifier_vgg16.plot_acc_loss()
classifier_vgg16.evaluate_classifier()


#------ VGG 19 Model -------# 
baseModel_vgg19 = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), pooling = 'avg')
    
classifier_vgg19 = classifierModel(CFG, baseModel_vgg19, 'vgg19')
history_vgg19 = classifier_vgg19.train()    

classifier_vgg19.save_classifier('./saved_models/VGG19_model/')
classifier_vgg19.plot_acc_loss()
classifier_vgg19.evaluate_classifier()


#------ ResNet50 Model -------# 
baseModel_res50 = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), pooling = 'avg')
    
classifier_res50 = classifierModel(CFG_RESNET, baseModel_res50, 'resnet50')
history_res50 = classifier_res50.train()    

classifier_res50.save_classifier('./saved_models/ResNet50_model/')
classifier_res50.plot_acc_loss()
classifier_res50.evaluate_classifier()


#------ InceptionV3 Model -------# 
baseModel_incv3 = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)), pooling = 'avg')
    
classifier_incv3 = classifierModel(CFG_INCEPTION, baseModel_incv3, 'inceptionv3')
history_incv3 = classifier_incv3.train()    

classifier_incv3.save_classifier('./saved_models/InceptionV3_model/')
classifier_incv3.plot_acc_loss()
classifier_incv3.evaluate_classifier()