import json
from PIL import Image
import os
import numpy as np
import torch
import torchvision
import transforms 
import json
from torchvision import transforms as T
import cv2
from sklearn.externals import joblib

classi=joblib.load('svm.model')#载入svm分类器，模型路径可自定义

model = torch.load('model.pth')#载入fasterrcnn模型参数，参数路径可自定义
model.eval()#训练模式

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

imgs = list(sorted(os.listdir(os.path.join('/home/ass02/Datasets/image_exp/Detection/', "test"))))#载入测试集图片名信息，路径可自定义

winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 6
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
winStride=(8,8)
padding = (0,0)#hog参数

get_tag = {0.0:'i2',1.0:'i4',2.0:'i5',3.0:'io',4.0:'ip',5.0:'p11',
          6.0:'p23',7.0:'p26',8.0:'p5',9.0:'pl30',10.0:'pl40',11.0:'pl5',
          12.0:'pl50',13.0:'pl60',14.0:'pl80',15.0:'pn',16.0:'pne',17.0:'po',18.0:'w57'}#数字与交通标志名的对应

dict_m = {}
target={}
threshold=0.5#门限，可自行设置

for i in range(len(imgs)):
    # 检测交通标志
    current_path = os.path.join("/home/ass02/Datasets/image_exp/Detection/test/",imgs[i])
    current_image = Image.open(current_path).convert("RGB")
    current_image_tensor,_ = transforms.ToTensor()(current_image,target)
    with torch.no_grad():
        current_pred = model([current_image_tensor.to(device)])
    
    current_boxes = []
    current_scores = []
    for j in range(len(current_pred[0]['scores'])):
        if(current_pred[0]['scores'].cpu()[j].numpy()>threshold):
            current_boxes.append(current_pred[0]['boxes'].cpu()[j].numpy().tolist())
            current_scores.append(current_pred[0]['scores'].cpu()[j].numpy().tolist())
    
    # 交通标志分类
    current_dict = {'objects':[]}
    current_count = len(current_scores)
    for j in range(current_count):
        current_region = current_image.crop(current_boxes[j])
        current_region_std = np.array(T.Resize([64,64])(current_region))
        current_region_hog = hog.compute(current_region_std,winStride,padding).flatten()[np.newaxis,:]
        pre=classi.predict(current_region_hog)
        current_class = pre[0]

        obj_bbox = {'xmin':current_boxes[j][0], 'xmax':current_boxes[j][2],
                    'ymin':current_boxes[j][1], 'ymax':current_boxes[j][3]}
        obj_dict = {'bbox':obj_bbox, 'category':get_tag[current_class],'score':current_scores[j]}
        current_dict['objects'].append(obj_dict)
    dict_m[imgs[i][:-4]] = current_dict
dict_all = {'imgs':dict_m}#写入字典

json_str = json.dumps(dict_all)
with open('task4_fasterrcnn_nothreshold_new.json','w') as json_file:
    json_file.write(json_str)#将字典保存为json文件，文件名可自定义

