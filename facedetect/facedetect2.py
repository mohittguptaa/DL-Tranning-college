import cv2 # type: ignore
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(faceCascade)
cap=cv2.VideoCapture(0)
name="Naman";eid=101;dept="Accountant";
while(True):
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        text=f'{name} {id} {dept}'
        (textW,texth),base=cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX,0.5,1)

        cv2.rectangle(img,(x,y-20),(x+textW,y),(255,0,0),cv2.FILLED)
        cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    cv2.imshow('Data',img)
    k=cv2.waitKey(30)
    if(k==27):   # esc
        break
cap.release()
cv2.destroyWindow()