import cv2
import matplotlib.pyplot as plt
import numpy as np


mammogram = np.fromfile('binfile/Mammogram.bin', dtype= np.uint8).reshape(256,256)

binary_mammogram = cv2.threshold(mammogram,95,255,cv2.THRESH_BINARY)[1]

contours = cv2.findContours(binary_mammogram, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

code = ""
for a in contours:
    for b in a:
        x,y = b[0]
        code += f"({x} + {y}) ->"
print(code)