import cv2
import numpy as np
import matplotlib.pyplot as plt

Bin_file_path = "actontBinbin.sec"

actonBin = np.fromfile('binfile/actontBin.bin', dtype= np.uint8).reshape(256,256)





result = cv2.matchTemplate(actonBin, template, cv2.TM_CCOEFF_NORMED)


threshold = 0.2  # Giá trị ngưỡng tùy chỉnh

J2 = np.where(result >= threshold, 255, 0).astype(np.uint8)

#Show ảnh gốc
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Ảnh gốc')
plt.imshow(actonBin, cmap='gray')
plt.axis('off')

#Show ảnh J2
plt.subplot(1, 2, 2)
plt.title('Kết quả')
plt.imshow(J2, cmap='gray')
plt.axis('off')

plt.show()