"""
using tensorflow for this shit
install briefcase
python -m pip isntall briefcase

actually, maybe kivy would work, just gotta figure out how
"""

"""
dependencies
"""

import cv2                                                                          
from matplotlib import pyplot as plt    

import functions

# access webcame via open cv
# extract frames
cap = cv2.VideoCapture(0)                       # webcam access

# set mediapipe model
with functions.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #0.5, 0.5 may change
    while cap.isOpened():                           # loop
        # read camera feed
        ret, frame = cap.read()    

        # make detection
        image, results = functions.media_pipedetection(frame, holistic)
        # results basically already tracks the body, face hands and all, 
            # stores the x, y and z components of the landmarks
        # face and hand will return no values if it detects nothing
        # pose landmarks will return a value regardless of detection, it will just be low
        print(results)

        # draw landmarks
        functions.draw_landmarks(image, results)

        # show frame                 
        cv2.imshow('OpenCV Feed', image)

        # quiting the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# breaking the window thing
cap.release             
cv2.destroyAllWindows()