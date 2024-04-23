import cv2
import os
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the model and cascade classifiers
model = load_model('models/CNN_DDD.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haar cascade files\haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

def detect_drowsiness(frame, gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        leyes = leye_cascade.detectMultiScale(roi_gray)
        reyes = reye_cascade.detectMultiScale(roi_gray)
        for (lex, ley, lew, leh) in leyes:
            l_eye = roi_gray[ley:ley + leh, lex:lex + lew]
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255.0
            l_eye = l_eye.reshape(-1, 24, 24, 1)
            lpred = model.predict(l_eye)
        for (rex, rey, rew, reh) in reyes:
            r_eye = roi_gray[rey:rey + reh, rex:rex + rew]
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255.0
            r_eye = r_eye.reshape(-1, 24, 24, 1)
            rpred = model.predict(r_eye)
        if np.argmax(lpred) == 1 or np.argmax(rpred) == 1:
            return 1  # Eye is open
        else:
            return 0  # Eye is closed
    return -1  # No face detected

# Evaluate the model on a test dataset
test_dir = 'test_data'
y_true = []  # True labels (0 for closed, 1 for open)
y_pred = []  # Predicted labels (0 for closed, 1 for open)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    true_label = int(img_name.split('_')[0])  # Extract true label from image name
    y_true.append(true_label)
    predicted_label = detect_drowsiness(img, gray_img)
    y_pred.append(predicted_label)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(conf_matrix))
plt.xticks(tick_marks, ['Closed', 'Open'], rotation=45)
plt.yticks(tick_marks, ['Closed', 'Open'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
