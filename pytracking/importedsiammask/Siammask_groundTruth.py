import cv2
import numpy as np 
from matplotlib import pyplot as plt
import os
c=0
#mak folder
dirPath="/home/robotics-meta/Desktop/Jason/SiamMask/Final data/18.10.2022/DICOM/plaque0003/"
filelist=os.listdir(dirPath)
#this is for fully segmented files
#filelist=sorted(filelist,key=lambda x: int((x.split('.')[0]).split('_')[1]))
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".png")):
        filelist.remove(fichier)
filelist=sorted(filelist,key=lambda x: int(((x.split('.')[0]).split('_')[0]).split('M')[1]))    
#filelist=sorted(filelist,key=lambda x: int((x.split('.')[0])))
print(filelist)
#where to put ground truth

f = open("/home/robotics-meta/Desktop/Jason/SiamMask/data/Vessel_VOT/Vessel_0003/groundtruth.txt", "a")


# To read image 
for file in filelist:
    img = cv2.imread(dirPath+file, cv2.IMREAD_COLOR)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i,j,0] == 1:
                img[i,j,0]=255
                img[i,j,1]=255
                img[i,j,2]=255

    #plt.imshow(img, interpolation='nearest')
    #plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray, 127, 255, 0) 
    # finding the contours 
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    # take the first contour 
    cnt = contours[0] 
    # computing the bounding rectangle of the contour 
    
    #x, y, w, h = cv2.boundingRect(cnt) 
    #new tech 
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)
   
    
    # draw contour 
    #img = cv2.drawContours(img, [cnt], 0, (0, 255, 255), 2) 
    #f.write(str(x)+","+str(y)+","+str(w)+","+str(h)+"\n")
    #float formated 
    f.write(str(box[0][0])+","+str(box[0][1])+","+str(box[1][0])+","+str(box[1][1])+","+str(box[2][0])+","+str(box[2][1])+","+str(box[3][0])+","+str(box[3][1])+"\n")
    # draw the bounding rectangle 
    #img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i,j,0] == 255:
                img[i,j,0]=0
                img[i,j,1]=0
                img[i,j,2]=0
    # display the image with bounding rectangle drawn on it 
    #cv2.imwrite("./Vessel_0003_formated/"+str(c)+"box.png", img)
    c=c+1
    #cv2.imshow("Bounding Rectangle", img) 
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
f.close()

