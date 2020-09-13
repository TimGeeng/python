import cv2
img = cv2.imread('C:\\Users\\timg\\Documents\\python\\AI\\OpenCV\\face_decect\\face\\2020.jpg')
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
flie = 'C:\\OpenCV-data\\haarcascades\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(flie)
faces = face_cascade.detectMultiScale(grey,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x + w,y + h),(255,0,0),3)
cv2.imshow('Results',img)
cv2.waitKey(0)
cv2.destroyAllWindows()