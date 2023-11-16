"""
using tensorflow for this shit
"""

"""
dependencies
"""

import cv2                              
import numpy as np                  
import os                               
from matplotlib import pyplot as plt    
import time
import mediapipe as mp

import mpholistic

# access webcame via open cv
# extract frames
cap = cv2.VideoCapture(0)                       # webcam access

# set mediapipe model
with mpholistic.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #0.5, 0.5 may change
    while cap.isOpened():                           # loop
        # read camera feed
        ret, frame = cap.read()    

        # make detection
        image, results = mpholistic.media_pipedetection(frame, holistic)
        print(results)

        # show frame                 
        cv2.imshow('OpenCV Feed', frame)

        # quiting the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# breaking the window thing
cap.release             
cv2.destroyAllWindows()