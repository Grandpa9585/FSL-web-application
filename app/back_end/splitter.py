""" 

    ,"Ano", MOV
    "Ano_ang_pangalan_mo", mp4
    "Bukas", MOV
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
    
"""

import cv2
import os
import numpy as np
import functions

actions = [
    "Ang_Pangalan_Ko",
    "Ano"
]

dirname = os.path.dirname(__file__)

with functions.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #0.5, 0.5 may change
    for action in actions:
        for i in range(7):
            frameNr = 0
            flepath = dirname+"\\MP_Data\\"+action
            capture = cv2.VideoCapture(flepath+"\\"+action+"_"+str(i + 1)+".mp4")
            while(True):
                success, frame = capture.read()

                if success:
                    image, results = functions.media_pipedetection(frame, holistic)

                    functions.draw_landmarks(image, results)

                    cv2.imwrite(flepath+"\\"+str(i + 1)+"\\"+str(frameNr)+".jpg", image)
                else:
                    break

                frameNr+=1
