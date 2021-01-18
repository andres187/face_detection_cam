#pylint:disable=no-member
import cv2 as cv
    
haar_cascade = cv.CascadeClassifier('front_face.xml')

capture = cv.VideoCapture(0);

while True: 
    isTrue, frame = capture.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    print(f'Numbers of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect: 
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    cv.imshow('Detected video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break


capture.release()

cv.waitKey(0)