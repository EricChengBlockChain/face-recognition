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
samePersonTopDistanceList =[]
samePersonAverageDistance =[]
maxDistance=0.41
sumd=0
sumc=0
noface=[]

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
                      if maxDistance<=dis:
                        maxDistance=dis
                        samePersonTopDistanceList.append(dis)
                      psumd+=dis
                      count+=1
                      sumd+=dis
                      sumc+=1
        if count==0:
           print('第%d人无法识别出人脸: '%i)
           noface.append(i)
           continue
        print('第%d人的平均欧式距离为:%f' %(i,psumd/float(count)))
        print("========================================")
        print("")
        samePersonAverageDistance.append(psumd/float(count))
print('500人,每人五张照片之间欧式距离平均值为%f:' %(sumd/sumc))
print("")
print("同一人的五张照片之间的欧式距离的平均值的数据如下:")
print(samePersonAverageDistance)
print("")
if len(samePersonTopDistanceList)!=0:
  print('其中欧式距离大于0.41的总计%d组情况,欧式距离为:'%len(samePersonTopDistanceList))
  print(samePersonTopDistanceList)
print("")
if len(noface)!=0:
  print('其中第',noface,'人的照片无法识别出人脸')









