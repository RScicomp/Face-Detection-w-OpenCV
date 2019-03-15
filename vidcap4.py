import numpy as np
import pickle
import cv2

face_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml") #from face-train.py

labels = {}

with open("labels.pickle",'rb') as f: #from pickle library
        og_labels = pickle.load(f) #use of pickle for label dictionary
	labels = {v:k for k,v in og_labels.items()}#invert
	
eye_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('OpenCV/haarcascades/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while(True):

        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

        for (x,y,w,h) in faces:
#               print(x,y,w,h)
 
                roi_gray = gray[y:y+h,x:x+w]
                roi_color = frame[y:y+h, x:x+w]
		                
		id_, conf = recognizer.predict(roi_gray) #predict region of interest
		if conf>=45 and conf <=85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font, 1,color,stroke,cv2.LINE_AA)
			
		img_item = "7.png"
                cv2.imwrite(img_item,roi_color)
		
                color = (255,255,255)
                stroke = 2
                end_cord_x= x+w
                end_cord_y= y + h
                cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color)

                eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors =5)
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(frame,(ex,ey), (ex+ew,ey+eh), (255,0,0),2)
#               detect_smile = smile_cascade.detectMultiScale(gray)
#               for (sx,sy,sw,sh) in detect_smile:
#                       cv2.rectangle(frame,(sx,sy),(sx+sw, sy+sh), (0,255,0),2)
		
		

        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()
