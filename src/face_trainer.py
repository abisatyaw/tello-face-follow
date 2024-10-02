import cv2
import numpy as np
from PIL import Image
import os

# The function `getImagesAndLabels` takes the path to the dataset as an
# argument and returns two lists: a list of images and a list of labels
# corresponding to each image in the list.
def getImagesAndLabels(path):
    # Get a list of all the image paths in the dataset
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    # Initialize two empty lists to store the images and labels.
    faceSamples = []
    ids = []

    # Iterate over each image in the dataset.
    for imagePath in imagePaths:
        # Read the image into memory using OpenCV's `imread` function.
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        # Split the filename of the image to get the ID of the person.
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Use OpenCV's `CascadeClassifier` to detect faces in the image.
        # The classifier is loaded from the "haarcascade_frontalface_alt2.xml"
        # file which is a pre-trained model for detecting faces.
        cascade_path = "src\Model\haarcascade_frontalface_alt2.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(img)

        # Iterate over each face detected in the image.
        for (x,y,w,h) in faces:
            # Append the detected face to the `faceSamples` list.
            faceSamples.append(img[y:y+h,x:x+w])
            # Append the ID of the person to the `ids` list.
            ids.append(id)

    # Return the two lists.
    return faceSamples, ids


if __name__ == "__main__":
    # Create a recognizer object using OpenCV's `LBPHFaceRecognizer_create`
    # function.
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Print a message to the user indicating that the program is starting to
    # train the model.
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    # The path to the dataset of images is stored in the `path` variable.
    path = 'src\Dataset'

    # Call the `getImagesAndLabels` function to get the images and labels from
    # the dataset.
    faces, ids = getImagesAndLabels(path)

    # Train the model using the images and labels.
    recognizer.train(faces, np.array(ids))

    # Save the model to a file named "face_recog_model.yml".
    # Print a message to the user indicating how many faces were trained and
    # exit the program.
    recognizer.write('face_recog_model.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

