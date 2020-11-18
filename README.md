# Emotion_detection
For CSV file reffer to link:- https://drive.google.com/file/d/1IFCFZpNQZ71euN-Phk19UvLalT5U7uja/view?usp=sharing
## Import Libraries
- import sys, os
- import pandas as pd
- import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

import os 
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image 
## Read CSV File
## Train model
### First Model
Accuracy- 57 %
epochs-30
folder:-model_1acc_57%_ep_30
### Second Model
Accuracy-57%
epochs-60
folder:-model_1acc_57%_ep_60
