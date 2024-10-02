import cv2
import numpy as np
import os
from djitellopy import Tello

# Constants
CAMERA_WIDTH, CAMERA_HEIGHT = 960, 720
SET_POINT_X, SET_POINT_Y, SET_POINT_Z = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2, 300
TOLERANCE_X, TOLERANCE_Y = 50, 50
TOLERANCE_Z_MIN, TOLERANCE_Z_MAX = 250, 400
SLOWDOWN_THRESHOLD_X, SLOWDOWN_THRESHOLD_Y, SLOWDOWN_THRESHOLD_Z = 80, 80, 80
DRONE_SPEED_X, DRONE_SPEED_Y, DRONE_SPEED_Z = 12, 12, 12
MIN_FACE_SIZE = (int(0.1 * CAMERA_WIDTH), int(0.1 * CAMERA_HEIGHT))

class DroneController:
    def __init__(self):
        self.drone = Tello()
        # self.drone.connect()
        # self.drone.streamoff()
        # self.drone.streamon()
        
        # Update the paths to the Haar cascade XML files
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
        self.face_cascade = cv2.CascadeClassifier(os.path.join(model_dir, 'haarcascade_frontalface_alt2.xml'))
        self.profile_face = cv2.CascadeClassifier(os.path.join(model_dir, 'haarcascade_profileface.xml'))
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = ['None', 'Abi']
        self.recognizer.read('face_recog_model.yml')

    def get_frame(self):
        frame = self.drone.get_frame_read().frame
        return frame

    def detect_faces(self, gray):
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.13, minNeighbors=5, minSize=MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = self.profile_face.detectMultiScale(gray, 1.3, 5)
        return faces

    def recognize_face(self, gray, x, y, w, h):
        id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 100 and id == 1:
            return self.names[id], confidence
        return "unknown", confidence
        
    def calculate_distances(self, x, y, w):
        targ_cord_x, targ_cord_y, targ_cord_z = x + w / 2, y + w / 2, w * 2
        return (
            targ_cord_x - SET_POINT_X,
            targ_cord_y - SET_POINT_Y,
            targ_cord_z - SET_POINT_Z
        )

    def draw_overlay(self, frame, x, y, w, h, dx, dy, dz):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 12, (255, 0, 0), 2)

        movement_text = f"{'X' if abs(dx) <= TOLERANCE_X else 'Left' if dx < 0 else 'Right'} "
        movement_text += f"{'Y' if abs(dy) <= TOLERANCE_Y else 'Up' if dy < 0 else 'Down'} "
        movement_text += f"{'Z' if TOLERANCE_Z_MIN <= w * 2 <= TOLERANCE_Z_MAX else 'Backward' if w * 2 < TOLERANCE_Z_MIN else 'Forward'}"
        cv2.putText(frame, movement_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        vTrue = np.array((SET_POINT_X, SET_POINT_Y, SET_POINT_Z))
        vTarget = np.array((x + w / 2, y + h / 2, w * 2))
        vDistance = vTrue - vTarget

        cv2.putText(frame, f"Distance: {vDistance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Target: {vTarget}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current: {vTrue}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def draw_grid(self, frame):
        h, w = frame.shape[:2]
        for x in np.linspace(start=w/3, stop=2*w/3, num=2):
            cv2.line(frame, (int(x), 0), (int(x), h), (0, 255, 0), 3)
        for y in np.linspace(start=h/3, stop=2*h/3, num=2):
            cv2.line(frame, (0, int(y)), (w, int(y)), (0, 255, 0), 3)

    def control_drone(self, dx, dy, dz):
        right_left_velocity = 0
        up_down_velocity = 0
        forward_backward_velocity = 0

        if dx < -TOLERANCE_X:
            right_left_velocity = -DRONE_SPEED_X
        elif dx > TOLERANCE_X:
            right_left_velocity = DRONE_SPEED_X

        if dy < -TOLERANCE_Y:
            up_down_velocity = DRONE_SPEED_Y
        elif dy > TOLERANCE_Y:
            up_down_velocity = -DRONE_SPEED_Y

        face_size = dz + SET_POINT_Z  # This is equivalent to w * 2
        if face_size < TOLERANCE_Z_MIN:
            forward_backward_velocity = DRONE_SPEED_Z
        elif face_size > TOLERANCE_Z_MAX:
            forward_backward_velocity = -DRONE_SPEED_Z

        # Slowdown Threshold
        if abs(dx) <= SLOWDOWN_THRESHOLD_X:
            right_left_velocity = int(right_left_velocity / 2)
        if abs(dy) <= SLOWDOWN_THRESHOLD_Y:
            up_down_velocity = int(up_down_velocity / 2)
        if TOLERANCE_Z_MIN <= face_size <= TOLERANCE_Z_MAX:
            forward_backward_velocity = int(forward_backward_velocity / 2)

        self.drone.send_rc_control(right_left_velocity, forward_backward_velocity, up_down_velocity, 0)

    def run(self):
        while True:
            frame = self.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)

            for (x, y, w, h) in faces:
                id, confidence = self.recognize_face(gray, x, y, w, h)
                dx, dy, dz = self.calculate_distances(x, y, w)
                self.draw_overlay(frame, x, y, w, h, dx, dy, dz)
                self.control_drone(dx, dy, dz)

                # Draw name and confidence
                cv2.putText(frame, f"{id}: {confidence:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.draw_grid(frame)
            cv2.rectangle(frame, 
                          (SET_POINT_X - TOLERANCE_X, SET_POINT_Y + TOLERANCE_Y), 
                          (SET_POINT_X + TOLERANCE_X, SET_POINT_Y - TOLERANCE_Y), 
                          (255, 0, 0), 3)
            cv2.circle(frame, (SET_POINT_X, SET_POINT_Y), 1, (255, 255, 0), 2)

            # Get Battery
            battery = self.drone.get_battery()
            cv2.putText(frame, f"Battery: {battery}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Drone Camera', frame)

            if cv2.waitKey(10) & 0xFF == 27 or battery < 10:  # ESC key or low battery
                break

        self.drone.land()
        self.drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = DroneController()
    controller.run()