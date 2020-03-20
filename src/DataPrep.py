#TODO: bring in images, convert to RGB matrix, output to a training example file. Export label to a training label file.
import cv2, sys, glob
import numpy as np


np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
f = open("TrainLabels.txt","a+")
pictures = glob.glob('day_night_images/training/night/*.jpg')
for picture in pictures:
    print(picture)
    #img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE)
    #img2 = img.flatten()
    #img2 = np.array2string(img2[1:101], separator=',')
    #f.write(img2[1:-1] + '\n')
    f.write("night\n")
f.close()
