import face_recognition
import cv2
import numpy as np
import os
import glob
import random
dirname = os.path.dirname(__file__)
#print (dirname)
number_files = 0
list_of_files = []



def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist
# 不同人的最小欧式距离dm
# 不同人的平均欧式距离da
# 同一人的最大欧式距离sm
# 同一人的平均欧式距离sa
total_number_of_face_pair1 = 0.00001
total_number_of_face_pair2 = 0.00001
total_number_of_face_pair3 = 0
total_number_of_face_pair4 = 0
dl=0#不同人之间的距离
davrl=0#不同人之间的平均距离
dminl=1#不同人之间的最小距离
dsum=0
sl=0#同一人之间的距离
savrl=0#同一人之间的平均距离
smaxl=0#同一人之间的最大距离
ssum=0
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
    #print ("path: ",path)
#    for j in range (5):
    list_of_files.extend(glob.glob(path+'*.bmp'))
#    print(list_of_files[698])
#    number_files = len(list_of_files)
#    print("number_files: ",number_files)
#for j in range (number_files):
#        print("list_of_files ",j," :",list_of_files[j])
for ii in range (5):
    try:
        print("**********0")
        i=random.randint(0,25)
        print("**********",i)
        arr1 = face_recognition.load_image_file(list_of_files[i])
        test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
        print("**********1")
    except Exception:
        continue
    else:
        for jj in range (5):
            try:
                j=random.randint(0,2500)
                arr3 = face_recognition.load_image_file(list_of_files[j])
                test_face_encoding2  = face_recognition.face_encodings(arr3)[0]
                print("**********2")
            except Exception:
                print(Exception)
                continue
            else:
                    dis = embedding_distance(test_face_encoding1, test_face_encoding2)
                    if int(i/5)==int(j/5):
                        ssum+=dis
                        total_number_of_face_pair1+=1
                        if smaxl<dis:
                            smaxl=dis
                    else:
                        dsum+=dis
                        total_number_of_face_pair2+=1
                        if dminl>dis:
                            dminl=dis
                    print(i,"和",j,"的欧式距离为 = ",dis)
davrl = dsum/total_number_of_face_pair2
savrl = ssum/total_number_of_face_pair1
print("===============================================")
print("不同人之间的平均欧式距离: ",davrl)
print("同一人之间的平均距离: ",savrl)
print("同一人之间的最大距离: ",smaxl)
print("不同人之间的最小距离: ",dminl)
if smaxl<dminl :
    print("阈值设置在 (",smaxl,",",dminl,")即可")
    print("此时在CASIA-FaceV5数据集中,精确度为100%")
else:
    x = (smaxl+dminl)/2
    print("建议阈值设置为: ",x)
    for i in range (50):
        try:
            arr1 = face_recognition.load_image_file(list_of_files[i])
            test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
        except Exception:
            continue
        else:
            for j in range (1000):
                try:
                    arr3 = face_recognition.load_image_file(list_of_files[j])
                    test_face_encoding2  = face_recognition.face_encodings(arr3)[0]
                except Exception:
                    continue 
                else:
                        dis = embedding_distance(test_face_encoding1, test_face_encoding2)
                        if int(i/5)==int(j/5):
                            if dis>x:
                                total_number_of_face_pair3+=1
                        else:
                            if dis<x:
                                total_number_of_face_pair4+=1
    mr = total_number_of_face_pair3/(total_number_of_face_pair1+total_number_of_face_pair2)
    rt = total_number_of_face_pair4/(total_number_of_face_pair1+total_number_of_face_pair2)
    print("此时在CASIA-FaceV5数据集中,精确度为",(1-mr-rt)*100,"%")
    print("误识率为: ",mr*100,"%")
    print("拒真率为: ",rt*100,"%")
print("===============================================")

#6222021001138479230




























