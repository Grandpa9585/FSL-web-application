import cv2
import mediapipe as mp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import numpy as np

mp_hands = mp.solutions.hands

class HandTrackingApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Create Camera widget
        self.camera = Camera(resolution=(640, 480), play=True)
        self.layout.add_widget(self.camera)

        # Create an Image widget to display the camera feed with hand landmarks
        self.hand_image = Image()
        self.layout.add_widget(self.hand_image)

        # Initialize Mediapipe Hands
        self.hands = mp_hands.Hands()

        # Schedule the update method to run at a regular interval
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 frames per second

        return self.layout

    def update(self, dt):
        # Check if the camera texture is available
        if self.camera.texture:
            # Get the texture data
            texture_data = np.frombuffer(self.camera.texture.pixels, dtype=np.uint8)

            # Reshape the texture data to match the camera resolution
            frame = texture_data.reshape((self.camera.texture.height, self.camera.texture.width, 4))

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            # Process the frame with Mediapipe Hands
            results = self.process_hand_tracking(frame_rgb)

            # Draw hand landmarks on the frame
            self.draw_landmarks(frame_rgb, results)

            # Update the Image widget with the processed frame
            self.update_hand_image(frame_rgb)

    def process_hand_tracking(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using Mediapipe Hands
        results = self.hands.process(frame_rgb)

        return results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    def update_hand_image(self, processed_frame):
        # Convert the processed frame to a Kivy texture
        texture = Texture.create(size=(processed_frame.shape[1], processed_frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(processed_frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        # Update the Image widget texture
        self.hand_image.texture = texture

if __name__ == '__main__':
    HandTrackingApp().run()
