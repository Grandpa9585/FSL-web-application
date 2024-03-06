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
    
    "Ang_Pangalan_Ko",
    "Ano_Ang_Pangalan_Mo",
    "Hello",
    "Hindi",
    "Kamusta_Ka",
    "Magandang_Gabi",
    "Magandang_Umaga",
    "Okay_Lang_Ako",
    "Oo",
    "Pasensya_Na"
"""

import cv2
import os
import numpy as np
import functions

actions = np.array([
    "Ang_Pangalan_Ko"#,
    #"Ano"
])

dirname = os.path.dirname(__file__)

with functions.mp_holistic.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic: #0.5, 0.5 may change
    for action in actions:
        for i in range(77):
            frameNr = 0
            flepath = dirname+"\\MP_Data\\"+action
            capture = cv2.VideoCapture(flepath+"\\"+action+"_"+str(i + 1)+".mp4")
            while(True):
                success, frame = capture.read()

                if success:
                    image, results = functions.media_pipedetection(frame, holistic)

                    keypoints = functions.extract_keypoints(results)
                    npy_path = flepath+"\\"+str(i + 1)+"\\"+str(frameNr)
                    np.save(npy_path, keypoints)

                    print(npy_path)
                    # functions.draw_landmarks(image, results)
                    # cv2.imwrite(flepath+"\\"+str(i + 1)+"\\"+str(frameNr)+".jpg", image)
                else:
                    break

                frameNr+=1
