import cv2                                                                          
from matplotlib import pyplot as plt    
import numpy as np

import functions

import os
import splitter

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

label_map = {label:num for num, label, in enumerate(splitter.actions)}

sequences, labels =[], []
for action in splitter.actions:
    for sequence in range(77):
        window = []
        for frame_num in range(109):
            res = np.load(os.path.join(splitter.flepath, action, str(sequence + 1), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(109,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(splitter.actions.shape[0], activation='softmax'))

model.compile(optimi)
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