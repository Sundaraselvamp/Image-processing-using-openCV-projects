import cv2
import numpy as np

image=cv2.imread('D:\course\python\Project in ML\coin\dataset\coin.jpg')


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

rectangle=(x,y,width,height)

mask=np.zeros(image.shape,dtype='uint8')
bgdmodel=np.zeros((1,65),dtype='float64')
fgdmodel=np.zeros((1,65),dtype='float64')
cv2.grabCut(image,mask,rectangle,bgdmodel,fgdmodel,iterCount=5,model=cv2.GC_INIT_WITH_RECT)
mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply morphological operations (e.g., erosion or dilation) to refine the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

# Multiply the original image with the mask to extract the objects
result = cv2.bitwise_and(image, image, mask=mask_binary)

# Save the separated objects
cv2.imwrite('object1.jpg', result)

# If there are multiple objects, you can repeat the process to extract and save each one



