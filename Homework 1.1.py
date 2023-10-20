import cv2
import matplotlib.pyplot as plt
import numpy as np


mammogram = np.fromfile('binfile/Mammogram.bin', dtype= np.uint8).reshape(256,256)

threshold_value = 95
_, binary_mammogram = cv2.threshold(mammogram, threshold_value, 255, cv2.THRESH_BINARY)


plt.subplot(2,2,1)
plt.imshow(mammogram, cmap = 'gray')
plt.title('Hình gốc')


plt.subplot(2,2,2)
plt.imshow(binary_mammogram, cmap = 'gray')
plt.title('Hình sau khi chỉnh')

plt.show()