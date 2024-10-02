import cv2
import os

# Set up the camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up the face detector
face_detector = cv2.CascadeClassifier('src\Model\haarcascade_frontalface_alt2.xml')

# Get the user ID
face_id = input('\n enter user id end press <return> ==>  ')

# Initialize the counter
count = 0

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# adjust dataset count
MAX_DATASET = 30
while count < MAX_DATASET:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop over each face and save it to the dataset folder
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f"src/Dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

    # Check for the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
