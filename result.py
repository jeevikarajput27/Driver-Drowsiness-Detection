import cv2
import os
import numpy as np
from keras.models import load_model
import sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pre-trained CNN model for eye state classification
model = load_model('models/CNN_DDD.h5')

# Initialize variables to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Open the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect left eye within the face ROI
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in left_eye:
            left_eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            left_eye_roi = cv2.resize(left_eye_roi, (24, 24))
            left_eye_roi = left_eye_roi / 255.0
            left_eye_roi = np.expand_dims(left_eye_roi, axis=0)
            left_eye_roi = np.expand_dims(left_eye_roi, axis=-1)
            left_pred = model.predict_classes(left_eye_roi)
            predicted_labels.append(left_pred[0])
            true_labels.append(1)  # Assuming 1 represents open eyes

            # Draw rectangle around left eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect right eye within the face ROI
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in right_eye:
            right_eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            right_eye_roi = cv2.resize(right_eye_roi, (24, 24))
            right_eye_roi = right_eye_roi / 255.0
            right_eye_roi = np.expand_dims(right_eye_roi, axis=0)
            right_eye_roi = np.expand_dims(right_eye_roi, axis=-1)
            right_pred = model.predict_classes(right_eye_roi)
            predicted_labels.append(right_pred[0])
            true_labels.append(1)  # Assuming 1 represents open eyes

            # Draw rectangle around right eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam feed and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert true and predicted labels to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print the calculated metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
