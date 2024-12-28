import cv2
import numpy as np
from time import sleep

min=80 
max=80 

offset=6   

pos=550 

delay= 60 

detect = []
car= 0

	
def page_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('test.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos), (1200, pos), (255,127,0), 3) 
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contour = (w >= min) and (h >= max)
        if not valid_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = page_center(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(pos+offset) and y>(pos-offset):
                car+=1
                cv2.line(frame1, (25, pos), (1200, pos), (0,127,255), 3)  
                detect.remove((x,y))
                print("car is detected : "+str(car))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(car), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilated)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
