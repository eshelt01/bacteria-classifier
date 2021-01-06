# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:10:42 2021

@author: Erin S
"""

""" Model configuration in JSON format"""
CFG = {
    "data": {
        "path": "./Bacteria/Train",
        "image_size": 224
    },
    "train": {
        "batch_size": 32,
        "epochs": 30,
        "val_split": 0.25,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_tensor": [224, 224, 3],
        "dense_layer": 256,
        "dropout": 0.2,
        "num_classes": 33,
        "init_lr": 1e-3
        }
}