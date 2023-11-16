"""
to do: actually understand what these mfs mean
"""
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic     # holistic model, detect
mp_drawing = mp.solutions.drawing_utils # drawing utilities, draw

def media_pipedetection(image, model):
    # bgr, what cv sees. convert it to rgb
    # symmetrical
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion BGR to RGB
    image.flags.writeable = False                    # image not writable
    results = model.process(image)                  # prediction
    image.flags.writeable = True                     # image writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB to BGR
    return image, results
    pass
