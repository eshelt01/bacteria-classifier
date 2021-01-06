# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:42:03 2021

@author: Erin S
"""
# Script to convert tif images to jpegs
import os
from PIL import Image

yourpath = 'C:/Users/Ed/Documents/bacteria-classifier/Bacteria'

for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        
        print(os.path.join(root, name))
        
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpeg"):
                print ("A jpeg file already exists for %s" % name)
            
            # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpeg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print ("Generating jpeg for %s" % name)
                    im.convert(mode="RGB")
                    im.save(outfile)
                except:
                    print('Exception')
                    
# Remove all tif files


for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            os.remove(os.path.join(root, name))
                                