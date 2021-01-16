# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:10:42 2021

@author: Erin S
"""

""" Default model configuration in JSON format"""
CFG = {
    "data": {
        "train_path": "./Bacteria/Train",
        "test_path": "./Bacteria/Test",
        "image_size": 224
    },
    "train": {
        "batch_size": 32,
        "epochs": 30,
        "val_split": 0.2,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_tensor": [224, 224, 3],
        "dense_layer": 512,
        "dropout": 0.5,
        "num_classes": 33,
        "init_lr": 1e-3
        }
}

""" Resnet model configuration in JSON format"""
CFG_RESNET = {
    "data": {
        "train_path": "./Bacteria/Train",
        "test_path": "./Bacteria/Test",
        "image_size": 224
    },
    "train": {
        "batch_size": 32,
        "epochs": 50,
        "val_split": 0.2,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_tensor": [224, 224, 3],
        "dense_layer": 512,
        "dropout": 0.2,
        "num_classes": 33,
        "init_lr": 1e-3
        }
}

""" Inception model configuration in JSON format"""
CFG_INCEPTION = {
    "data": {
        "train_path": "./Bacteria/Train",
        "test_path": "./Bacteria/Test",
        "image_size": 299
    },
    "train": {
        "batch_size": 32,
        "epochs": 30,
        "val_split": 0.2,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input_tensor": [299, 299, 3],
        "dense_layer": 256,
        "dropout": 0.2,
        "num_classes": 33,
        "init_lr": 1e-3
        }
}