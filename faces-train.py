
import os
from PIL import Image
import pickle # allows
import numpy as np
import cv2

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
#wherever this file is saved, look at where this path is. Give directory
#ie. /Dev
image_dir = os.path.join(BASE_DIR,"images") #image file
face_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
y_labels = []
x_train = []
current_id = 0 #for every label id created, we add one
label_ids = {} #dictionary
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"): 
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(" ", "-").lower() #get label for face
			print(label,path)# print out file
			if not label in label_ids: #giving ids
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
	
			#y_labels.append(label) # wanna make labels some number
			#x_train.append(path) # wanna verify this image, turn into a numpy array, convert into a gray image
			pil_image = Image.open(path).convert("L") #give image of path, convert to gray scale.	
			size = (550,550)
			final_image = pil_image.resize(size,Image.ANTIALIAS)
		
			image_array = np.array(final_image, "uint8")
			print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)#do face detect inside image
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w] #region of interest
				x_train.append(roi)
				y_labels.append(id_) #with ten faces, we have a label for it. Note we should not have multiple faces

print(y_labels)
print(x_train)			

with open("labels.pickle",'wb') as f: #from pickle library	
	pickle.dump(label_ids,f) 
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")

