import face_recognition
import cv2
import numpy as np
import os
import glob
import random
dirname = os.path.dirname(__file__)
number_files = 0
list_of_files = []



def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist
i=0
j=0
for i in range (500):
    if i<1:
        dir = '000/'
    elif i<10:
        dir = '00'+str(i)+'/'
    elif i<100:
        dir = '0'+str(i)+'/'
    else:
        dir=str(i)+'/';
    path = os.path.join(dirname, dir)

    list_of_files.extend(glob.glob(path+'*.bmp'))

samePersonAverageDistance =[]
sumd=0
sumc=0
for i in range (500):
        count=0
        psumd=0
        for ii in range (i*5,(i+1)*5-1):
            try:
                arr1 = face_recognition.load_image_file(list_of_files[ii])
                test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
            except Exception:
                continue
            else:
                for iii in range (ii+1,(i+1)*5):
                   try:
                     arr1 = face_recognition.load_image_file(list_of_files[iii])
                     test_face_encoding2  = face_recognition.face_encodings(arr1)[0]
                   except Exception:
                      continue
                   else:
                      dis =embedding_distance(test_face_encoding1, test_face_encoding2)
                      print('第%d人的编号为%d的照片和编号为%d的照片的欧式距离 = %f '%(i,ii,iii,dis))
                      psumd+=dis
                      count+=1
                      sumd+=dis
                      sumc+=1
        print('第%d人的平均欧式距离为:%f' %(i,psumd/float(count)))
        print("==============")
        samePersonAverageDistance.append(psumd/float(count))
print('500人,每人五张照片之间欧式距离平均值为%f:' %(sumd/sumc))
print("同一人的五张照片之间的欧式距离的平均值的数据如下:")
print(samePersonAverageDistance)








