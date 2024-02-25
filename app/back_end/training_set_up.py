import cv2                                                                          
from matplotlib import pyplot as plt    
import numpy as np

import functions

# number of videos per action
no_sequences = 7
# number of frames per video 
frame_num = 119
# a list of words
actions = np.array([
    "Ang_pangalan_ko",
    "Ano",
    "Ano_ang_pangalan_mo",
    "Bukas",
    "Hello",
    "Hindi",
    "I_am_fine",
    "Ingat",
    "Kahapon",
    "Kumusta_ka_na",
    "Magandang_gabi",
    "Magandang_umaga",
    "Magkano_po_ito",
    "Mamaya",
    "Ngayon",
    "Ngayong_araw",
    "Oo",
    "Pasensya_na",
    "Please",
    "Sino"
])

#set mediapipe model
with functions.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #0.5, 0.5 may change
    
    #loop through actions
    for action in actions: 

        #loop through videos
        for sequence in range(no_sequences):

            #loop through frames in videos
            for frame in range(frame_num):

                # make detection
                image, results = functions.media_pipedetection(frame, holistic)