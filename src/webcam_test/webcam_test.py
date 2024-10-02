import cv2
import numpy as np
import os

# Constants
# Width of the camera frame in pixels
CAMERA_WIDTH = 960

# Height of the camera frame in pixels
CAMERA_HEIGHT = 720

# X-coordinate of the target point (center of the frame)
SET_POINT_X = CAMERA_WIDTH // 2

# Y-coordinate of the target point (center of the frame)
SET_POINT_Y = CAMERA_HEIGHT // 2

# Z-coordinate of the target point (depth)
SET_POINT_Z = 300

# Tolerance for X-axis movement (in pixels)
TOLERANCE_X = 50

# Tolerance for Y-axis movement (in pixels)
TOLERANCE_Y = 50

# Tolerance for Z-axis movement (in pixels)
TOLERANCE_Z_MIN = 60
TOLERANCE_Z_MAX = 250

# Minimum size of a detectable face (width, height) in pixels
MIN_FACE_SIZE = (int(0.1 * CAMERA_WIDTH), int(0.1 * CAMERA_HEIGHT))

class WebcamController:
    def __init__(self):
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # this is the magic!
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Update the paths to the Haar cascade XML files
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
        self.face_cascade = cv2.CascadeClassifier(os.path.join(model_dir, 'haarcascade_frontalface_alt2.xml'))
        self.profile_face = cv2.CascadeClassifier(os.path.join(model_dir, 'haarcascade_profileface.xml'))
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = ['None', 'Abi']
        self.recognizer.read('face_recog_model.yml')

    def get_frame(self):
        ret, frame = self.cap.read()
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 10, (0, 255, 0), 2)

        movement_text = f"{'X' if abs(dx) <= TOLERANCE_X else 'Left' if dx < 0 else 'Right'} "
        movement_text += f"{'Y' if abs(dy) <= TOLERANCE_Y else 'Up' if dy < 0 else 'Down'} "
        movement_text += f"{'Forward' if dz <= TOLERANCE_Z_MIN else 'Backward' if dz >= TOLERANCE_Z_MAX else 'Z'}"
        print(dz)
        cv2.putText(frame, movement_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Display vector coordinates
        vTrue = np.array((SET_POINT_X, SET_POINT_Y, SET_POINT_Z))
        vTarget = np.array((x + w / 2, y + h / 2, w * 2))
        vDistance = vTrue - vTarget

        cv2.putText(frame, f"Distance: {vDistance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Target: {vTarget}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Center: {vTrue}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def draw_grid(self, frame):
        h, w = frame.shape[:2]
        for x in np.linspace(start=w/3, stop=2*w/3, num=2):
            cv2.line(frame, (int(x), 0), (int(x), h), (0, 255, 0), 3)
        for y in np.linspace(start=h/3, stop=2*h/3, num=2):
            cv2.line(frame, (0, int(y)), (w, int(y)), (0, 255, 0), 3)

    def run(self):
        while True:
            frame = self.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)

            for (x, y, w, h) in faces:
                id, confidence = self.recognize_face(gray, x, y, w, h)
                dx, dy, dz = self.calculate_distances(x, y, w)
                self.draw_overlay(frame, x, y, w, h, dx, dy, dz)

                # Draw name and confidence
                cv2.putText(frame, f"{id}: {confidence:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.draw_grid(frame)
            cv2.rectangle(frame, 
                          (SET_POINT_X - TOLERANCE_X, SET_POINT_Y + TOLERANCE_Y), 
                          (SET_POINT_X + TOLERANCE_X, SET_POINT_Y - TOLERANCE_Y), 
                          (255, 0, 0), 3)
            cv2.circle(frame, (SET_POINT_X, SET_POINT_Y), 1, (255, 255, 0), 2)

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(10) & 0xFF == 27:  # ESC key
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = WebcamController()
    controller.run()