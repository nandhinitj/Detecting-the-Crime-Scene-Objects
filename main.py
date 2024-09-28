import os
import numpy as np
import cv2 as cv
from sklearn.preprocessing import label_binarize



an = 0
if an == 1:
    dir1 = './Dataset/Dataset1/'
    dir2 = os.listdir(dir1)
    for i in range(len(dir2)):
        file = dir1 + dir2[i]
        reads = cv.VideoCapture(file)
        cnt = int(reads.get(cv.CAP_PROP_FRAME_COUNT))
        for j in range(cnt):
            print(i,j,cnt)
            reads.set(cv.CAP_PROP_POS_FRAMES,j)
            res, frame = reads.read()
            frame = cv.resize(frame,[750,750])
            cv.imwrite('./images/image-' + str(i+1)+'-'+ str(j + 1) + '.png',frame)

an = 0
if an == 1:
    im1 = []
    im2 = []
    im3 = []
    dir = './Original/'
    dirr = './bound/'
    dirrr = './crop/'
    dir1 = os.listdir(dir)
    dir2 = os.listdir(dirr)
    dir3 = os.listdir(dirrr)
    for i in range(len(dir1)):
        file = dir + dir1[i]
        file1 = dirr + dir2[i]
        file2 = dirrr + dir3[i]
        read = cv.imread(file)
        read1 = cv.imread(file1)
        read2 = cv.imread(file2)
        read = cv.resize(read,[512,512])
        read1 = cv.resize(read1, [512, 512])
        read2 = cv.resize(read2, [512, 512])
        im1.append(read)
        im2.append(read1)
        im3.append(read2)
    np.save('Original.npy',im1)
    np.save('Bound.npy',im2)
    np.save('Crop.npy',im3)


