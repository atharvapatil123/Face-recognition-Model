import numpy as np
import cv2
import os

##### KNN CODE #######
# After storing different face coordinates, when a face comes in front of the camera, then according to KNN algorithmn, the differences are calculated. Top k faces are selected

def distance(v1, v2):
    # Eucledian sqrt((x1-x2)**2 + (y1-y1)**2)
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        # Compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy]) 

    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x:x[0])[:k]

    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)

    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./../haarcascade_frontalface_default.xml')

dataset_path = './face_dataset/'

face_data = []
labels = []
class_id = 0    # labels for every class
names = {}      # mapping between id and name

# Dataset Preparation

for fx in os.listdir(dataset_path):
    # For interaction with os, looping in our stored dataset-values  in the file
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        # Taking entire string except ".npy" 
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        # Every class ID is associated with every data_item length : array of ones of shape of data_item
        # n * array of ones => array of n
        # EG 2 * array of ones => array of 2's
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    # Convert frames to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect multi faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 5

        face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        # Resize face to 100x100

        out = knn(trainset, face_section.flatten())

        # Draw rectangle in the original image
        cv2.putText(frame, names[int(out)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
    
    cv2.imshow('Faces', frame)

    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

    








         