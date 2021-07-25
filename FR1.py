import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./../haarcascade_frontalface_default.xml')

skip = 0
face_data = []
# List in which we will be storing each face date value
dataset_path = './face_dataset/'

file_name = input("Enter your name: ")

while(True):
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if ret == True:
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        # 1.3 means scale down image to 30%
        if(len(faces)==0):continue

        l = 1

        faces = sorted(faces, key = lambda x: x[2]*x[3], reverse = True)
        # Sorting wrt size/area i.e. width*height 

        skip += 1 # Increment skip for every face

        for face in faces[:1]:
            x, y, w, h = face

            offset = 5 #Padding

            face_offset = frame[y-offset: y+h+offset, x-offset: x+w+offset]
            face_selection = cv2.resize(face_offset, (100, 100))
            # Resize face to 100x100

            if skip%10 == 0:
                # After every 10th frame, store the face
                face_data.append(face_selection)
                print(len(face_data))

            cv2.imshow(str(l), face_selection)
            l+=1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


        cv2.imshow('faces', frame)

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            break
    else:
        break

# KNN Algorithmn works only on numpy arrays
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)
 
np.save(dataset_path + file_name, face_data)
print('Dataset saved at : {}'.format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows()