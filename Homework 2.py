import cv2
import matplotlib.pyplot as plt
import numpy as np

lady = np.fromfile('binfile/lady.bin', dtype= np.uint8).reshape(256,256)
min_ori = np.min(lady)
max_ori = np.max(lady)

stretched_img = np.uint8(((lady - min_ori) / (max_ori - min_ori))* 255)

cv2.imshow('haha',lady)
cv2.imshow('hihi',stretched_img)

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(lady, cv2.COLOR_BGR2RGB))
plt.title('Hình gốc')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(stretched_img, cv2.COLOR_BGR2RGB))
plt.title('Hình sau khi chỉnh')

hist_ori = cv2.calcHist([lady],[0],None,[256],[0,256])
plt.subplot(2,2,3)
plt.bar(np.arange(256), hist_ori.ravel())
plt.title('Biều đồ ảnh gốc')

hist_stretched = cv2.calcHist([stretched_img],[0],None,[256],[0,256])
plt.subplot(2,2,4)
plt.bar(np.arange(256), hist_stretched.ravel())
plt.title('Biều đồ sau khi chỉnh')
plt.show()