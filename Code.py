# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:08:19 2017

@author: lenovo
"""


import cv2
import os
import sys
import numpy as np

# ### Training Data

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Narendra Modi", "Arvind Kejriwal"]


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector
    face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60));
    
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    print("Found {0} faces!".format(len(faces)))
    
    #extract the face area
    for i in range(0, len(faces)+1):
        (x, y, w, h) = faces[i]
    
    #return only the face part of the image
        return gray[y:y+w, x:x+h], faces[i]
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", img)
    cv2.waitKey(100)

data_folder_path = 'F:\Precog\Training data'
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("F:\Precog\Training data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

#load test images
test_img1 = cv2.imread("F:/Precog/Test data/img1.jpg")
cv2.resize(test_img1, (400, 500))
test_img2 = cv2.imread("F:/Precog/Test data/img2.jpg")
cv2.resize(test_img2, (400, 500))
test_img3 = cv2.imread("F:/Precog/Test data/img3.jpg")
cv2.resize(test_img3, (400, 500))
test_img4 = cv2.imread("F:/Precog/Test data/img4.jpg")
cv2.resize(test_img4, (400, 500))
test_img5 = cv2.imread("F:/Precog/Test data/img5.jpg")
cv2.resize(test_img5, (400, 500))
test_img6 = cv2.imread("F:/Precog/Test data/img6.jpg")
cv2.resize(test_img6, (400, 500))
test_img7 = cv2.imread("F:/Precog/Test data/img7.jpg")
cv2.resize(test_img7, (400, 500))
test_img8 = cv2.imread("F:/Precog/Test data/img8.jpg")
cv2.resize(test_img8, (400, 500))

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
predicted_img6 = predict(test_img6)
predicted_img7 = predict(test_img7)
predicted_img8 = predict(test_img8)

random = ["image"]

#display both images
cv2.imshow(random[0], cv2.resize(predicted_img1, (400,500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img5, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img6, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img7, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(random[0], cv2.resize(predicted_img8, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()


