"""
Simple Dummy Images Generator 
With Different angle for neuron selectivity checking 
05/12/2023
"""
import numpy as np
import cv2
import os

width = 28
height = 28
folder_name = "./dummy_inputs/"

# Create folder if not already existing
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

img_list = []

img = np.zeros((width, height))
# img[width//2] = 1

cv2.line(img, (0,0), (28,28), (255,0,0), thickness=1)

for i in range(width):
    for j in range(height):
        img = np.zeros((width, height))
        point1 = (i,j)
        point2 = (width - i, height - j)
        cv2.line(img, point1, point2, (255,0,0), thickness=1)
        img_list.append(img)

for i in range(len(img_list)):
    cv2.imwrite(folder_name+str(i)+'.jpg', img_list[i])

if __name__ == "__main__":
    pass
