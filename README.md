# Facial Recognition on Google Photos

### Project Overview:

This project aims to create an image classification system for personal Google Photos, using 
deep learning models and computer vision techniques. Unlike typical facial recognition algorithms, 
the project will use a custom dataset and apply transfer learning to extract high-level features 
from pre-trained models such as Inception and ResNet. Due to the limited amount of data available, 
the project will also implement data preprocessing techniques including data augmentation, cropping, 
and rotation, as well as potentially using deep learning models like Generative Adversarial 
Networks (GAN) or Autoencoders to generate more data. The end result will be a custom model 
that is fine-tuned to accurately classify images in the user's Google Photos collection.

### Requirements:

We can install all the required libraries using following command.

```python
pip3 install -r requirements.txt
```

### Model

The model is composed of several blocks, including Data Preparation, Object Detection, Transfer Learning 
for Feature Extraction, and Fine-tuning. To prepare the data, I utilized the Google Photos website to 
select specific individuals' photos and then utilized Google API client services to download the data to 
my local device. Object detection was performed using the YOLO (You Only Look Once) technique. Feature 
extraction was achieved using various models such as MobileNet, ResNet, Inception, and Inception-ResNet. 
The visual representation of model can be shown as below

![Model](https://github.com/umesh-gattem/CV_Project/blob/master/CV%20Project%20Model.png)






