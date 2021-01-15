import face_recognition
import cv2
import numpy as np
import os
import glob
import time
dirname = os.path.dirname(__file__)
number_files = 0
list_of_files = []

def getFileSize(filePath, size=0):
    for root, dirs, files in os.walk(filePath):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
            #print(f)
    return size


counts=0
fsize=0

for i in range (98):
    size = 0
    sizes = 0
    if i<1:
        dir = '000/'
    elif i<10:
        dir = '00'+str(i)+'/'
    else :
        dir = '0'+str(i)+'/'
    path = os.path.join(dirname, dir)
    sizes = getFileSize(path,size)
    #print(path," 目录中的文件大小为: ",sizes)
    fsize += sizes
    list_of_files.extend(glob.glob(path+'*.bmp'))

ii = len(list_of_files)
time_start = time.time()
cpu_time_start = time.process_time()
for c in range (ii):
    try:
        arr1 = face_recognition.load_image_file(list_of_files[c])
        test_face_encoding1  = face_recognition.face_encodings(arr1)[0]
    except Exception:
        continue
    else:
        continue
time_end = time.time()
cpu_time_end = time.process_time()
times = time_end - time_start
times_cpu = cpu_time_end - cpu_time_start
print("总共包含照片数量为: ",ii,"张")
print("照片总大小为: ",fsize/(1024*1024),"MB")
print("cpu执行加载以及提取照片特征执行总时间为: ",times_cpu," s")
s = fsize/(1024*1024*1024*times_cpu)
print("总照片大小(GB)/cpu加载以及提取所有照片特征执行总时间(s): ",s,"GB/S")

