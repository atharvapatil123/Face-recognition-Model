import numpy as np
import cv2
import os

##### KNN CODE #######
# After storing different face coordinates, when a face comes in front of the camera, then according to KNN algorithmn, the differences are calculated. Top k faces are selected

def distance(v1, v2):
    # Eucledian sqrt((x1-x2)**2 + (y1-y1)**2)
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    print(train,"\n")
    dist = []
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        # print(train[i, :-1])
        iy = train[i, -1]
        # print(train[i, -1])

        # Compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy]) 
        # print(dist)

    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # print(dk)

    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    # print(labels)

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # print(output[1])

    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    # print(index)
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
    # print(fx)
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        # names.sort()
        # Taking entire string except ".npy" 
        data_item = np.load(dataset_path + fx)
        # print(data_item)
        face_data.append(data_item)
        # print(face_data)

        # print(data_item.shape)
        # print(np.ones(data_item.shape[0],))
        target = class_id * np.ones((data_item.shape[0],))
        # Every class ID is associated with every data_item length : array of ones of shape of data_item
        # n * array of ones => array of n
        # EG 2 * array of ones => array of 2's
        # print(target)
        class_id += 1
        labels.append(target)
        # print(labels)

print(names)
# print(face_data)
face_dataset = np.concatenate(face_data, axis=0)
# print(face_dataset)
# x =  np.concatenate(labels, axis=0)
# print(x)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
# -1 mean unknown number of row or column, here 1 column and row as per numpy
# print(face_labels.shape[0])
# print(face_labels)
# print(face_labels.shape)
# print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
# print(trainset)
# print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    # Convert frames to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect multi faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(faces)
    for face in faces:
        x, y, w, h = face
        # print(face)

        # Get the face ROI
        offset = 5

        face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset]
        # print(face_section)
        # print("\n NEXT \n")
        face_section = cv2.resize(face_section, (100, 100))
        # print(face_section)

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

    








         