import multiprocessing as mp
import numpy as np
import face_recognition
import os
import glob
import random
import psutil

indexes = np.arange(0,499,1,dtype=np.int16)
#print(indexes)
data = indexes.tolist()
#print("indexes lengths: %d"%(len(data)))
results=[]


def embedding_distance(feature_1, feature_2):
       dist = np.linalg.norm(feature_1 - feature_2)
       return dist
       dirname = os.path.dirname(__file__)
       number_files = 0
       list_of_files = []

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

def distanceOfdifPerson(index):
           print("index%d===="%(index))
           dises=[]
           i=index
           try:
               arr1 = face_recognition.load_image_file(list_of_files[random.randint(i*5,(i+1)*5-1)])
               test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
               print("i=%d"%(i))
           except Exception:
               return (index,dises)
           else:
               for j in range (i+1,500):
                   try:
                       arr2 = face_recognition.load_image_file(list_of_files[random.randint(j*5,(j+1)*5-1)])
                       test_face_encoding2  = face_recognition.face_encodings(arr2)[0]
                   except Exception:
                       continue
                   else:
                       dis =embedding_distance(test_face_encoding1, test_face_encoding2)
                       print("%d and %d's distance is %f"%(i,j,dis))
                       dises.append(dis)
                   return (index ,dises)

def collect_result(dises):
       global results
       results+=dises

if __name__ == '__main__':
  pool = mp.Pool(mp.cpu_count())
  print("一共有%d核."%(mp.cpu_count()))
  for index in enumerate(data):
           pool.apply_async(distanceOfdifPerson, args=index, callback=collect_result)
  pool.close()
  pool.join()
print(results[:10])






