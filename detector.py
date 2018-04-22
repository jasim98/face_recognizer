import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(Id==1):
           name="Person1"
        elif(Id==2):
            name="Person2"
        else:
            name="UNKNOWN"
        cv2.putText(im,str(name), (x,y+h),cv2.FONT_HERSHEY_PLAIN, 1.5, (225,0,0),2)
        cv2.imshow('Recognizer',im)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break;
cam.release()
cv2.destroyAllWindows()
