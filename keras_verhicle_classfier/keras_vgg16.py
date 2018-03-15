#/usr/bin/python
#--*-- Coding:utf-8 --*--

from keras.models import  Model, load_model
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import cv2

def get_labelist(label_file):
    label_list = []
    fr = open(label_file,"rb")
    for line in fr:
        label = line.strip().split(" ")[1]
        label_list.append(label)
    return label_list

def get_labelidx(img_path,model):
    img = cv2.imread(img_path)
    label_list = np.arange(16)
    lb = LabelBinarizer().fit(label_list)
    predictions = model.predict(np.array([img]))
    predict_idx = lb.inverse_transform(predictions)
    return predict_idx

def predict_one(img_path,label_file):
    model = load_model("vgg16.h5")
    label_index = get_labelidx(img_path,model)
    label_list = get_labelist(label_file)
    return label_list[label_index[0]]

if __name__ == "__main__":
    label_file = "label.txt"
    img_path = "1003_1.jpg"
    label = predict_one(img_path,label_file)
    print label
