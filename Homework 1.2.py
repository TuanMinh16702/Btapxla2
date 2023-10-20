import cv2
import matplotlib.pyplot as plt
import numpy as np


mammogram = np.fromfile('binfile/Mammogram.bin', dtype= np.uint8).reshape(256,256)

binary_mammogram = cv2.threshold(mammogram,95,255,cv2.THRESH_BINARY)[1]

contours = cv2.findContours(binary_mammogram, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

contours_img = np.zeros_like(binary_mammogram)
cv2.drawContours(contours_img,contours,-1,255,1)



plt.subplot(2, 2, 1)
plt.imshow(mammogram, cmap='gray')
plt.title('Original Image')


plt.subplot(2, 2, 2)
plt.imshow(contours_img, cmap='gray')
plt.title('After')

plt.show()