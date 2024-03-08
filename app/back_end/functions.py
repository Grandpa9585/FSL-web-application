"""
to do: actually understand what these mfs mean
"""
import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.hands     # hand model, detect
mp_drawing = mp.solutions.drawing_utils # drawing utilities, draw

# function that detects landmarks of the user
def media_pipedetection(image, model):
    # bgr, what cv sees. convert it to rgb
    # symmetrical
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion BGR to RGB
    image.flags.writeable = False                    # image not writable
    results = model.process(image)                  # prediction
    image.flags.writeable = True                     # image writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB to BGR
    return image, results

# function that renders restults to the screen
def draw_landmarks(image, results):
    # its a draw-er
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS) # literally why is it under hand connections lmfao
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121/(num+1), 22*num, 76), thickness=2, circle_radius=2)
                                      )

def extract_keypoints(results):
    # extracts the points of the landmarks
    # if the points exists, put the points in an array [x, y, z, a(if applicable)] then
    # "flattens" the multidimensioal array
    # else return an array thats full of zeroes
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    arr = np.array([np.zeros(21*3), np.zeros(21*3)])
    
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            if num == 2:
                break
            arr[num] = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    # returns a giant array that has all of the points
    # this is the output values for a frame for use in any algorithm
    # return np.concatenate([pose, face, left_hand, right_hand])
    return np.concatenate([arr[0], arr[1]])
    # !if there is an error here, try getting the length of the damn things somehow
