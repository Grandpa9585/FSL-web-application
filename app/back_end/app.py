from flask import Flask, render_template, Response

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import cv2
import numpy as np

import functions

app = Flask(__name__)

camera = cv2.VideoCapture(0)

import splitter

# detection vars
sequence = []
sentence = []
threshold = 0.4 

def gen_frame():
    with functions.mp_holistic.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                width = 1920
                height = 1080
                # cropped = img[start_row:end_row, start_col:end_col]

                image, results = functions.media_pipedetection(frame, holistic)
                model = Sequential()
                model.load_weights('ExpandedWordList4.h5')

                functions.draw_landmarks(image, results)

                # prediction 1920 x 1080
                keypoints = functions.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-109:]

                if len(sequence) == 109:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(splitter.actions[np.argmax(res)])

                # display logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if splitter.actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(splitter.actions[np.argmax(res)])
                        else:
                            sentence.append(splitter.actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image) # setup for data transmission
                image = buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)