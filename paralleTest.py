import multiprocessing as mp
import numpy as np
import face_recognition
import os
import glob
import random
import psutil

print("一共有%d核."%(mp.cpu_count()))
data = [[i for i in range(0,22)],[i for i in range(22,44)],[i for i in range(44,67)],[i for i in range(67,91)],[i for i in range(91,117)],[i for i in range(117,145)],[i for i in range(145,175)],[i for i in range(175,208)],[i for i in range(208,246)],[i for i in range(246,290)],[i for i in range(290,347)],[i for i in range(347,499)]]

results=[]

dirname = os.path.dirname(__file__)
i=0
j=0
list_of_files = []
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



def embedding_distance(feature_1, feature_2):
       dist = np.linalg.norm(feature_1 - feature_2)
       return dist

def proceId(id):
     if 0<=id<22:
      return 1,float(id-0)/22
     elif 22<=id<44:
      return 2,float(id-22)/22
     elif 44<=id<67:
      return 3,float(id-44)/23
     elif 67<=id<91:
      return 4,float(id-67)/24
     elif 91<=id<117:
      return 5,float(id-91)/26
     elif 117<=id<145:
      return 6,float(id-117)/28
     elif 145<=id<175:
      return 7,float(id-145)/30
     elif 175<=id<208:
      return 8,float(id-175)/33
     elif 208<=id<246:
      return 9,float(id-208)/38
     elif 246<=id<290:
      return 10,float(id-246)/44
     elif 290<=id<347:
      return 11,float(id-290)/57
     else :
      return 12,float(id-347)/153

def distanceOfdifPerson(index):
      dises=[]
      for k in index:
          pidd,percentage = proceId(k)
          print("\n\n第",pidd,"进程,已经完成了",percentage,"\n\n")
          if k<499:
           i=k
           try:
               arr1 = face_recognition.load_image_file(list_of_files[random.randint(i*5,(i+1)*5-1)])
               test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
           except Exception:
               continue
           else:
               for j in range (i+1,500):
                   try:
                       arr2 = face_recognition.load_image_file(list_of_files[random.randint(j*5,(j+1)*5-1)])
                       test_face_encoding2  = face_recognition.face_encodings(arr2)[0]
                   except Exception:
                       continue
                   else:
                       dis =embedding_distance(test_face_encoding1, test_face_encoding2)
                       # if j%50==0:
                       pidd,percentage = proceId(i)
                       print("进程",pidd,"\t处理:第",i,"\t人和第%d\t人的欧式距离为: %f"%(j,dis))
                       dises.append(dis)
      print("\n第%d进程结束!!!!!!!!\n"%(pidd))
      return dises

if __name__ == '__main__':
  pool = mp.Pool(mp.cpu_count())
  results = pool.map(distanceOfdifPerson, [index for index in data])
  print(len(results))
  print("tops = results.sort()")
  tops = results.sort()
  print("print(results[:50])")
  print(results[:50])
  print("avrdis = float(sum(results))/float(124750)")
  avrdis = float(sum(results))/float(124750)
  print("print(不同人之间的平均欧式距离为: ,avrdis)")
  print("不同人之间的平均欧式距离为: ",avrdis)
  pool.close()
  pool.join()






