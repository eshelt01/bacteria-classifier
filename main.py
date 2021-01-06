# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 00:06:24 2021

@author: Ed
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from config.config import CFG
from models.base_model import classifierModel

   
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), pooling = 'avg')
    
my_VGG16 = classifierModel(CFG,baseModel, 'vgg16')

Vgg16_history = my_VGG16.train()    
my_VGG16.save_classifer('./saved_models/VGG16_model/')