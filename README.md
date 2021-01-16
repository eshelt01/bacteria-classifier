# bacteria-classifier
Classifier for images of different species of bacteria using convolutional neural networks.
A transfer learning approach in Tensorflow is used, with VGG, Inception and ResNet models. 

The bacteria dataset can be found as in the public DIBaS bacterial dataset, which contains microscopy images of 33 different species of bacteria.

Final results:

 Model | Test Accuracy
------------ | -------------
VGG16 | 95.3%
VGG19 | 93.8%
ResNet50 | 97.0%
InceptionV3 | 98.5%
 
The best model was InceptionV3 with 98.5% test accuracy.

 ### References for this work: 
  - "An Automated Deep Learning Approach for Bacterial Image Classification", M.Talo (2019)
   - "Novel neural network application for bacterial colony classification", Lei Huang & Tong Wu (2018)
