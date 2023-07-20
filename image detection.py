import cv2
import numpy as np
##import matplotlib


img=cv2.imread('D:\course\python\Project in ML\shape.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i=0
for contour in contours:
    if i==0:
        i=1
        continue
    approx=cv2.approxPolyDP(contour,cv2.arcLength(contour,True),True)

    cv2.drawContours(img,[contour],0,(0,0,255),5)
    M=cv2.moments(contour)
    if M['m00']!=0.0:
        x=int(M['m10']/M['m00'])
        y=int(M['m01']/M['m00'])

    if len(approx)==3:
        cv2.putText(img,'triangle',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 2)
  
    elif len(approx) == 4:
        cv2.putText(img, 'Quadrilateral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 2)
  

cv2.imshow('shapes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
##cv2.imshow('img',gray)
