import cv2
import matplotlib.pyplot as plt
import numpy as np

johnny = np.fromfile('binfile/johnny.bin', dtype= np.uint8).reshape(256,256)

eqJohnny = cv2.equalizeHist(johnny)
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(johnny,cv2.COLOR_BGR2RGB))
plt.title('Hình gốc')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(eqJohnny,cv2.COLOR_BGR2RGB))
plt.title('Hình sau khi chỉnh')

hist_ori = cv2.calcHist([johnny],[0],None,[256],[0,256])
plt.subplot(2,2,3)
plt.bar(np.arange(256), hist_ori.ravel())
plt.title('Biều đồ ảnh gốc')

hist_eq = cv2.calcHist([eqJohnny],[0],None,[256],[0,256])
plt.subplot(2,2,4)
plt.bar(np.arange(256), hist_eq.ravel())
plt.title('Biều đồ ảnh sau khi chỉnh')


plt.show()