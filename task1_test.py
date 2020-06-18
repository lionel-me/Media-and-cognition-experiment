# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:20:45 2020

@author: lionel
"""
from sklearn.externals import joblib
import numpy
import cv2
import torch
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import json

classi=joblib.load('svm.model')
#加载测试好的模型。
transform = T.Compose([
            #T.Grayscale(),       
            T.Resize([64,64]),
            #T.RandomHorizontalFlip(),
            #T.ToTensor(),
    ])
dataset = ImageFolder('Test', transform=transform)
#这里的Test文件与本程序文件在同一根目录，助教运行时可能需要修改读取路径。
winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 6
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
winStride=(8,8)
padding = (0,0)
x=numpy.zeros((3800,10584))
for i in range(3800):
    #a=dataset[i][0].view(3072)
    #a=a.numpy()
    t=numpy.array(dataset[i][0])
    a=hog.compute(t,winStride,padding)
    a=a.flatten()
    x[i]=a
#对测试集进行hog预处理。
    
pre=classi.predict(x)
#利用模型得到测试集训练结果。

imgs=[]
name=[]
for i in range(3800):
    temp=dataset.imgs[i][0][-8:]
    name.append(temp)
#利用.imgs方法得到图片名信息。
 
pre=pre.tolist()
change={0.0:'i2',1.0:'i4',2.0:'i5',3.0:'io',4.0:'ip',5.0:'p11',
          6.0:'p23',7.0:'p26',8.0:'p5',9.0:'pl30',10.0:'pl40',11.0:'pl5',
          12.0:'pl50',13.0:'pl60',14.0:'pl80',15.0:'pn',16.0:'pne',17.0:'po',18.0:'w57'}
res=[change[i] if i in change else i for i in pre]
#将标签由数字转回交通标志类别名称。

result=dict(zip(name,res))
#将图片名与预测结果存入字典中。

json_str = json.dumps(result)
with open('pred.json','w') as json_file:
    json_file.write(json_str)
#将字典转为json格式并保存。
    