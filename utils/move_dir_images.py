# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:54:06 2020

@author: Erin S
"""

import os
import random

# Split data into train/test before data augmentation and validation split in Keras
# Folder 'Bacteria' split into Train/Test subdirectories
# Folder for each bacteria species within Train and Test

n = 0.1 # Fraction of images to move

for label in os.listdir('./Bacteria/Train'):
    pics = os.listdir('Bacteria/Train/' +str(label))
    num_pics = len(pics)
    num_test_pics = round(n * num_pics)
    
    test_pics = random.sample(pics, num_test_pics)  # images in each folder

    for p in test_pics:
        file_path = './Bacteria/Train/' +str(label) +'/' + str(p)
        val_path = './Bacteria/Test/' +str(label) + '/'+ str(p)

        os.rename(f'{file_path}', f'{val_path}')
