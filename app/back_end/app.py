from flask import Flask, render_template, Response
import cv2

import functions

app = Flask(__name__)

camera = cv2.VideoCapture(0)

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
                image, results = functions.media_pipedetection(frame, holistic)

                functions.draw_landmarks(image, results)

                # prediction
                keypoints = functions.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[:109]

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