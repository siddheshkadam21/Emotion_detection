# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:16:58 2020

@author: SIDDHESH
"""

import os 
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image 

#lode model
model = model_from_json(open("fer.json").read())
#load weights
model.load_weights("fer.h5")

face_harr_cascade=cv2.CascadeClassifier("frontface_haarcascade.xml")
cap=cv2.VideoCapture(0)

while True:
    ret,test_image=cap.read()#capture frame and returns boolen values and capture image 
    if not ret:
        continue
    gray_img=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    faces_detection=face_harr_cascade.detectMultiScale(gray_img,1.32,5)
    for (x,y,w,h) in faces_detection:
        cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),thickness=5)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e face area from image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels=image.img_to_array(roi_gray)
        img_pixels=np.expand_dims(img_pixels,axis=0)
        img_pixels/=255
        prediction=model.predict(img_pixels)
        #find max indexed array
        max_index=np.argmax(prediction[0])
        emotions=("angry","disgust","fear","happy","sad","suprise","neutral")
        predicted_emotion=emotions[max_index]
        cv2.putText(test_image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    resized_img=cv2.resize(test_image,(1000,700))
    cv2.imshow("Facial emotion analysis",resized_img)
    
    if cv2.waitKey(10)==ord("q"):#wait until "q" key is pressed
        break
cap.release()
cv2.destroyAllWindows
    
        
        